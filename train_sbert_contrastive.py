import argparse
import itertools
import json
import os
import random
from collections import defaultdict
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from dataset import TripletDataset
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_scheduler


# -----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
def load_groups_from_csv(csv_path, data_col='word'):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[data_col])
    df[data_col] = df[data_col].astype(str)
    return df


# -----------------------------------------------------------------------------
def split_by_group(df: pd.DataFrame, train_min_size: int = 50, val_test_min: int = 10, val_ratio=0.5, random_state=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # count each group of words
    group_counts = df['id'].value_counts()

    # select groups for training
    train_group_ids = group_counts[group_counts >= train_min_size].index.tolist()

    # remaining groups become valdation and test groups
    remaining_group_ids = group_counts[(group_counts < train_min_size) & (group_counts >= val_test_min)].index.tolist()

    # split the test and validation groups with no overlap
    val_group_ids, test_group_ids = train_test_split(remaining_group_ids, test_size=1 - val_ratio, random_state=random_state)

    # assign rows to splits based on group ids
    train_df = df[df['id'].isin(train_group_ids)].copy()
    val_df = df[df['id'].isin(val_group_ids)].copy()
    test_df = df[df['id'].isin(test_group_ids)].copy()

    return train_df, val_df, test_df


# -----------------------------------------------------------------------------
def build_triplet_list(groups: pd.DataFrame):

    group_to_words = defaultdict(list)
    for _, row in groups.iterrows():
        group_to_words[row['id']].append(row['word'])

    all_group_ids = list(group_to_words.keys())

    triplet_set = set()

    progress = tqdm(all_group_ids, initial=0, desc="Building", leave=False)
    for anchor_group in progress:
        positives = group_to_words[anchor_group]
        if len(positives) < 2:
            continue

        # all unique anchor-positive pairs
        ap_pairs = list(itertools.combinations(positives, 2))
        random.shuffle(ap_pairs)
        # TODO: should we restrict the max number of triplets in a group? (EG ap_pairs[:max_trips]:)
        for anchor, postive in ap_pairs:
            # pick a negative from a different group
            negative_group_choices = [g for g in all_group_ids if g != anchor_group and len(group_to_words[g]) > 0]
            if not negative_group_choices:
                continue
            neg_group = random.choice(negative_group_choices)
            negative = random.choice(group_to_words[neg_group])

            triplet = (anchor, postive, negative)
            print(f"triplet: {triplet}")
            # skip dups
            if triplet in triplet_set:
                continue

            triplet_set.add(triplet)

    return triplet_set


# -----------------------------------------------------------------------------
def visualize_embeddings(model, groups, path, file_prefix, max_groups=10, title="t-SNE of fine-tuned word embeddings"):
    os.makedirs(path, exist_ok=True)
    model.eval()
    words, labels, embeddings = [], [], []
    for i, group in enumerate(groups[:max_groups]):
        for word in group:
            try:
                emb = model.encode(word)
                embeddings.append(emb)
                labels.append(f"group_{i}")
                words.append(word)
            except Exception as e:
                print(e)
                continue
    if not embeddings:
        print("No embeddings to visualize.")
        return
    tsne = TSNE(n_components=2, random_state=42)
    vectors = np.array(embeddings)
    reduced = tsne.fit_transform(vectors)
    df = pd.DataFrame({'x': reduced[:, 0].tolist(), 'y': reduced[:, 1].tolist(), 'label': labels, 'word': words})
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='label', style='label', s=60)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(path, file_prefix + '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    model.train()


# -----------------------------------------------------------------------------
def plot_tsne(model, group_dict, path, max_groups=10, epoch=None):
    os.makedirs(path, exist_ok=True)
    title = "t-SNE"
    file_stem = title
    if epoch:
        if int(epoch) < 0:
            title = title + " - pre-trained only"
            file_stem = file_stem + "-PRE"
        else:
            title = title + f" - Epoch {epoch}"
            file_stem = file_stem + f"-E{epoch}"
    else:
        title = title + " - Final fine-tuned word embeddings"
        file_stem = file_stem + "-FINAL"
    visualize_embeddings(model, group_dict, path, file_stem, max_groups=max_groups, title=title)


# -----------------------------------------------------------------------------
# Save an interim checkpoint such that we can restart from this point
def save_checkpoint(model, optimizer, scheduler, epoch, step, path):
    os.makedirs(path, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
    }, os.path.join(path, f'checkpoint-epoch{epoch:03d}-step{step:09d}.pt'))


# -----------------------------------------------------------------------------
# evaluate the model while training
def evaluate_triplet_model(model, dataloader, margin):
    model.eval()
    device = model.device
    total_gap = 0.0
    correct = 0
    total = 0
    margin_violations = 0

    with torch.no_grad():
        evaluate = tqdm(dataloader, initial=0, desc="Evaluating", leave=True)
        for batch in evaluate:
            anchors = [ex.texts[0] for ex in batch]
            positives = [ex.texts[1] for ex in batch]
            negatives = [ex.texts[2] for ex in batch]
            emb_anchors = model.encode(anchors, convert_to_tensor=True, device=device)
            emb_positives = model.encode(positives, convert_to_tensor=True, device=device)
            emb_negatives = model.encode(negatives, convert_to_tensor=True, device=device)

            positive_sims = util.cos_sim(emb_anchors, emb_positives).diagonal()
            negative_sims = util.cos_sim(emb_anchors, emb_negatives).diagonal()

            # Triplet accuracy
            correct += (positive_sims > negative_sims).sum().item()

            # Cosine gap
            total_gap += torch.sum(positive_sims - negative_sims).item()

            # Margin violations
            margin_violations += (positive_sims < (negative_sims + margin)).sum().item()

            total += len(batch)

    accuracy = correct / total
    avg_gap = total_gap / total
    violations_rate = margin_violations / total
    model.train()
    return accuracy, avg_gap, violations_rate


# -----------------------------------------------------------------------------
def save_triplets_to_jsonl(examples, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for ex in examples:
            assert len(ex.texts) == 3
            json.dump({
                "anchor": ex.texts[0],
                "positive": ex.texts[1],
                "negative": ex.texts[2],
            }, f)
            f.write('\n')


# -----------------------------------------------------------------------------
# .fit() didn't allow us to do gradient clipping, save t-SNE images, etc
# this should be a manual loop replacement
def train(model, training_dataloader, validation_dataloader, groups, output_path, epochs=10, margin=0.3, lr=1e-5, tsne_every=1, clip_norm=5.0, log_every=1000, save_every=2500, resume=True):
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # decay every epoch
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    steps_per_epoch = len(training_dataloader)
    num_training_steps = epochs * steps_per_epoch
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    start_epoch = 0
    start_step = 0
    checkpoint_dir = os.path.join(output_path, 'checkpoints')

    # Load from a check point, if resuming
    if resume:
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
        if checkpoints:
            latest = checkpoints[-1]
            ckpt = torch.load(os.path.join(checkpoint_dir, latest))
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch']
            start_step = ckpt['step']
            print(f'📌 Resuming from {latest}: epoch={start_epoch}, step={start_step}')

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        epoch_grad_norms = []
        # progress = tqdm(dataloader, initial=start_step if epoch == start_epoch else 0, desc=f"Epoch {epoch}", leave=True)
        progress = tqdm(training_dataloader, initial=0, desc=f"Epoch {epoch}", leave=True)
        for step, batch in enumerate(progress):
            if epoch == start_epoch and step < start_step:
                # skip completed steps
                continue

            # Extract texts from InputExample
            anchor_texts = [ex.texts[0] for ex in batch]
            positive_texts = [ex.texts[1] for ex in batch]
            negative_texts = [ex.texts[2] for ex in batch]

            # Tokenize each set of texts
            anchor_input = model.tokenize(anchor_texts)
            positive_input = model.tokenize(positive_texts)
            negative_input = model.tokenize(negative_texts)

            # Get the device of the model and move things to it
            device = model.device
            anchor_input = {k: v.to(device) for k, v in model.tokenize(anchor_texts).items()}
            positive_input = {k: v.to(device) for k, v in model.tokenize(positive_texts).items()}
            negative_input = {k: v.to(device) for k, v in model.tokenize(negative_texts).items()}

            # Get embeddings
            anchor_emb = model.forward(anchor_input)['sentence_embedding']
            positive_emb = model.forward(positive_input)['sentence_embedding']
            negative_emb = model.forward(negative_input)['sentence_embedding']

            # Compute cosine similarity
            cos_sim_ap = F.cosine_similarity(anchor_emb, positive_emb)
            cos_sim_an = F.cosine_similarity(anchor_emb, negative_emb)

            # Compute triplet loss
            triplet_loss = F.relu(cos_sim_an - cos_sim_ap + margin).mean()

            optimizer.zero_grad()
            triplet_loss.backward()

            # compute total gradiant norm (L2 norm)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5
            epoch_grad_norms.append(grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            scheduler.step()

            # add loss as a string so that it's formated to 4 decimal places
            progress.set_postfix(loss=f'{triplet_loss.item():.4f}')

            # save checkpoints every so often so that we can restart
            if ((epoch * steps_per_epoch) + step) % save_every == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, step, os.path.join(output_path, "checkpoints"))
                tqdm.write(f"    📝 Save checkpoint Epoch: {epoch:>3} Step: {step:>8}")

            if step % log_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                tqdm.write(f"    Step {step:>8}/{steps_per_epoch} | Loss: {triplet_loss.item():.4f} | Grad norm: {grad_norm:.2f} | LR: {current_lr:.2e}")

            total_loss += triplet_loss.item()

        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
        current_lr = optimizer.param_groups[0]['lr']
        tqdm.write(f"\nEpoch {epoch}/{epochs} | Loss: {total_loss:.4f} | Avg Grad Norm: {avg_grad_norm:.2f} | LR: {current_lr:.2e}")
        accuracy, avg_gap, violations_rate = evaluate_triplet_model(model, validation_dataloader, margin)
        tqdm.write(f"    📊 Validation accuracy: {accuracy:.4f} | Avg cosine similarity gap: {avg_gap:.4f} | Margin violation rate: {violations_rate:.4f}\n")

        if (epoch + 1) % tsne_every == 0:
            plot_tsne(model, groups, path=os.path.join(output_path, "analysis"), epoch=epoch)

        tqdm.write(f"    📝 Saving interim model to Epoch-{epoch}-fine_tuned_model...")
        model.save(os.path.join(output_path, f"Epoch-{epoch}-fine_tuned_model"))

    return model


# -----------------------------------------------------------------------------
# just return the list as-is
def input_example_collate_func(batch):
    return batch


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train_sbert_constrastive", description="Trains a contrastive model of words."
    )
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file.')
    parser.add_argument('--out_path', type=str, required=True, help='Path to output files.')
    parser.add_argument('--model_name', type=str, required=False, default="distiluse-base-multilingual-cased-v1", help='Sentence transformer model. Default: \'distiluse-base-multilingual-cased-v1\'')
    parser.add_argument('--batch_size', type=int, required=False, help='Training batch size. Default: auto tune')
    parser.add_argument('--epochs', type=int, required=False, default=5, help='Number of epochs to train. Default: 5')
    parser.add_argument('--group_col', type=str, required=False, default='id', help='Column header in CSV file that denotes the group. Default: \'id\'')
    parser.add_argument('--data_col', type=str, required=False, default='word', help='Column header in CSV file that denotes the data. Default: \'word\'')
    parser.add_argument('--min_large', type=int, required=False, default=20, help='Defines the minium size of a \'large\' group. Default: 20')
    parser.add_argument('--min_small', type=int, required=False, default=2, help='Defines the minium size of a \'small\' group. Default: 2')
    parser.add_argument('--margin', type=float, required=False, default=0.3, help='Defines the margin between groups. Default: 0.3')
    args = parser.parse_args()

    csv_path = args.csv_path
    os.makedirs(args.out_path, exist_ok=True)
    analysis_path = os.path.join(args.out_path, "analysis")
    os.makedirs(analysis_path, exist_ok=True)
    checkpoint_path = os.path.join(args.out_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    print(f"📌 output path = {args.out_path}")
    print(f"📌 analysis path = {analysis_path}")
    print(f"📌 checkpoint path = {checkpoint_path}")

    # check for CUDA
    device = "cpu"
    pin_memory = False
    batch_size = args.batch_size
    if not batch_size:
        batch_size = 32
    if torch.cuda.is_available():
        batch_size = 64 if not args.batch_size else args.batch_size
        device = "cuda"
        pin_memory = True
    print(f"📌 Using device: {device}")
    print(f"📌 Batch size: {batch_size}")
    print(f"📌 Pin memory: {pin_memory}")

    # initialize the random number generators to something repeatable
    set_seed()

    print(f"⏳ Loading sentence-transformer model ({args.model_name})...")
    word_model = SentenceTransformer(args.model_name)
    word_model.to(device)

    print(f"⏳ Loading and splitting groups from {args.csv_path}...")
    groups = load_groups_from_csv(csv_path, data_col=args.data_col)
    training_df, validation_df, test_df = split_by_group(groups, train_min_size=args.min_large, val_test_min=args.min_small, val_ratio=0.5, random_state=42)

    print("🚧 Creating triplet training dataset")
    training_set = build_triplet_list(training_df)
    training_dataset = TripletDataset(training_set)
    triplet_data_file = os.path.join(args.out_path, "training_triplets.jsonl")
    print(f"📝 Saving training triplets to {triplet_data_file}...")
    save_triplets_to_jsonl(training_dataset, triplet_data_file)

    print("🚧 Creating triplet validation dataset")
    validation_set = build_triplet_list(validation_df)
    validation_dataset = TripletDataset(validation_set)
    triplet_data_file = os.path.join(args.out_path, "validation_triplets.jsonl")
    print(f"📝 Saving validation triplets to {triplet_data_file}...")
    save_triplets_to_jsonl(validation_dataset, triplet_data_file)

    print("🚧 Creating triplet test dataset")
    test_set = build_triplet_list(test_df)
    test_dataset = TripletDataset(test_set)
    triplet_data_file = os.path.join(args.out_path, "test_triplets.jsonl")
    print(f"📝 Saving test triplets to {triplet_data_file}...")
    save_triplets_to_jsonl(test_dataset, triplet_data_file)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, collate_fn=input_example_collate_func)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, collate_fn=input_example_collate_func)

    print("🎨 Visualizing pre-trained only embeddings...")
    training_group = training_df.groupby('id')['word'].apply(list)
    plot_tsne(word_model, training_group, path=analysis_path, epoch=-1)

    print("⚙️ Training...")
    word_model = train(word_model, training_dataloader, validation_dataloader, training_group, args.out_path, margin=args.margin)
    print(f"📝 Saving final model to {args.out_path}/FINAL-fine_tuned_model...")
    word_model.save(os.path.join(args.out_path, "FINAL-fine_tuned_model"))

    print("🎨 Visualizing final embeddings...")
    plot_tsne(word_model, training_group, path=analysis_path)
