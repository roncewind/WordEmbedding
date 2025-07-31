import argparse
import json
import os
from collections import Counter
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap
from matplotlib.lines import Line2D
from sentence_transformers import InputExample, SentenceTransformer, util
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# -----------------------------------------------------------------------------
class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        return self.triplets[index]


# -----------------------------------------------------------------------------
def load_triplets_from_jsonl(filename):
    triplets = []
    with open(filename, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            print(f"⏳ Reading line {index}", end='\r')
            data = json.loads(line)
            triplets.append(InputExample(texts=[data["anchor"], data["positive"], data["negative"]]))
    print("\n... ✅ done reading.")
    return triplets


# -----------------------------------------------------------------------------
def plot_f1_vs_threshold(positive_sims: np.ndarray, negative_sims: np.ndarray, path):

    thresholds = np.arange(0.0, 1.01, 0.01)
    true_labels = np.concatenate([np.ones(len(positive_sims)), np.zeros(len(negative_sims))])
    all_sims = np.concatenate([positive_sims, negative_sims])

    f1_scores = []
    precisions = []
    recalls = []

    best_f1 = 0.0
    best_threshold = 0.0

    for t in thresholds:
        preds = (all_sims > t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', zero_division=0)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    # plot F1 vs threshold
    plt.figure(figsize=(10, 7))
    plt.plot(thresholds, precisions, label='Precision', color='blue')
    plt.plot(thresholds, recalls, label='Recall', color='green')
    plt.plot(thresholds, f1_scores, label='F1 Score', color='orange')
    plt.axvline(x=float(best_threshold), color='red', linestyle='--', label=f'Best threshold = {best_threshold:.2f}')
    plt.title("Precision, Recall, and F1 Score vs Cosine Similarity Threshold")
    plt.xlabel("Cosine Similarity Threshold")
    plt.ylabel("Scores")
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncols=3)
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    plt.savefig(os.path.join(path, 'f1_vs_cosine_similarity.png'), bbox_inches='tight', dpi=150)
    plt.close()
    return best_threshold, best_f1


# -----------------------------------------------------------------------------
def plot_similarity_histogram(positive_sims: np.ndarray, negative_sims: np.ndarray, threshold, path):

    sns.histplot(positive_sims, color='green', label='Positive pairs', stat='density', bins=50)
    sns.histplot(negative_sims, color='red', label='Negative pairs', stat='density', bins=50)
    # plt.axvline(x=threshold, color='gray', linestyle='--', label=f'Best Threshold {threshold:.2f}')
    plt.axvline(x=threshold, color='gray', linestyle='--')
    _, y_max = plt.ylim()
    plt.text(threshold - 0.01, y_max * 0.5, f'Best Threshold {threshold:.2f}', rotation=90, va='center', ha='right')
    plt.title("Cosine Similarity Distributions")
    plt.xlabel("Cosine Similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'cosine_similarity_distributions.png'), bbox_inches='tight', dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# evaluate the model while training
def evaluate_triplet_model(model, dataloader, margin, path):
    model.eval()
    device = model.device
    total_gap = 0.0
    correct = 0
    total = 0
    margin_violations = 0
    categories = Counter()

    all_positive_sims = []
    all_negative_sims = []

    with torch.no_grad():
        progress = tqdm(dataloader, initial=0, desc="Progress", leave=True)
        for batch in progress:
            anchors = [ex.texts[0] for ex in batch]
            positives = [ex.texts[1] for ex in batch]
            negatives = [ex.texts[2] for ex in batch]
            emb_anchors = model.encode(anchors, convert_to_tensor=True, device=device)
            emb_positives = model.encode(positives, convert_to_tensor=True, device=device)
            emb_negatives = model.encode(negatives, convert_to_tensor=True, device=device)

            positive_sims = util.cos_sim(emb_anchors, emb_positives).diagonal()
            negative_sims = util.cos_sim(emb_anchors, emb_negatives).diagonal()

            all_positive_sims.append(positive_sims.cpu())
            all_negative_sims.append(negative_sims.cpu())

            # Triplet accuracy
            correct += (positive_sims > negative_sims).sum().item()

            # Cosine gap
            total_gap += torch.sum(positive_sims - negative_sims).item()

            # Margin violations
            margin_violations += (positive_sims < (negative_sims + margin)).sum().item()

            total += len(batch)

            # calculate confusion
            for sp, sn in zip(positive_sims, negative_sims):
                sp = sp.item()
                sn = sn.item()
                if sp > sn:
                    if sp >= sn + margin:
                        categories["Correct & Satisfied"] += 1
                    else:
                        categories["Correct but Violated"] += 1
                else:
                    categories["Incorrect"] += 1

    accuracy = correct / total
    avg_gap = total_gap / total
    violations_rate = margin_violations / total
    model.train()

    all_pos = torch.cat(all_positive_sims).numpy()
    all_neg = torch.cat(all_negative_sims).numpy()
    best_threshold, best_f1 = plot_f1_vs_threshold(all_pos, all_neg, path)
    print(f"📊 Best F1 Score: {best_f1:.4f} at threshold {best_threshold:.2f}")
    plot_similarity_histogram(all_pos, all_neg, best_threshold, path)

    return accuracy, avg_gap, violations_rate, categories, best_threshold


# -----------------------------------------------------------------------------
# plot t-SNE for each triplet
def plot_triplet_tsne(model, triplets, path, sample_size=1000):
    device = model.device
    model.eval()

    if sample_size < len(triplets):
        triplets = triplets[:sample_size]

    anchors = [ex.texts[0] for ex in triplets]
    positives = [ex.texts[1] for ex in triplets]
    negatives = [ex.texts[2] for ex in triplets]

    all_texts = anchors + positives + negatives
    roles = (["anchor"] * len(anchors) + ["positive"] * len(positives) + ["negative"] * len(negatives))

    embeddings = model.encode(all_texts, convert_to_tensor=False, device=device)

    tsne = TSNE(n_components=2, init="random", random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)

    # Plot
    color_map = {"anchor": "blue", "positive": "green", "negative": "red"}
    colors = [color_map[r] for r in roles]

    plt.figure(figsize=(10, 8))

    # 1. plot points
    plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.6, s=10)

    # 2. draw connecting lines
    N = len(triplets)
    for i in range(N):
        a, p, n = reduced[i], reduced[i + N], reduced[i + 2 * N]
        plt.plot([a[0], p[0]], [a[1], p[1]], 'g-', alpha=0.3)
        plt.plot([a[0], n[0]], [a[1], n[1]], 'r-', alpha=0.3)

    # 3. add title and legend
    plt.title("t-SNE of Triplet Embeddings")
    plt.legend(handles=[
        Line2D([0], [0], marker='o', color='w', label='Anchor', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Positive', markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Negative', markerfacecolor='red', markersize=8),
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(path, 't-SNE_triplet_embeddings.png'), bbox_inches='tight', dpi=150)
    plt.close()
    model.train()


# -----------------------------------------------------------------------------
# plot UMAP for each triplet
def plot_triplet_umap(model, triplets, path, sample_size=1000):
    device = model.device
    model.eval()

    if sample_size < len(triplets):
        triplets = triplets[:sample_size]

    anchors = [ex.texts[0] for ex in triplets]
    positives = [ex.texts[1] for ex in triplets]
    negatives = [ex.texts[2] for ex in triplets]

    all_texts = anchors + positives + negatives
    roles = (["anchor"] * len(anchors) + ["positive"] * len(positives) + ["negative"] * len(negatives))

    embeddings: np.ndarray = model.encode(all_texts, convert_to_tensor=False, device=device)

    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = cast(np.ndarray, reducer.fit_transform(embeddings))

    # Plot
    color_map = {"anchor": "blue", "positive": "green", "negative": "red"}
    colors = [color_map[r] for r in roles]

    plt.figure(figsize=(10, 8))

    # 1. plot points
    plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.6, s=10)

    # 2. draw connecting lines
    N = len(triplets)
    for i in range(N):
        a, p, n = reduced[i], reduced[i + N], reduced[i + 2 * N]
        plt.plot([a[0], p[0]], [a[1], p[1]], 'g-', alpha=0.3)
        plt.plot([a[0], n[0]], [a[1], n[1]], 'r-', alpha=0.3)

    # 3. add title and legend
    plt.title("UMAP of Triplet Embeddings")
    plt.legend(handles=[
        Line2D([0], [0], marker='o', color='w', label='Anchor', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Positive', markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Negative', markerfacecolor='red', markersize=8),
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'UMAP_triplet_embeddings.png'), bbox_inches='tight', dpi=150)
    plt.close()
    model.train()


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="test_triplets", description="Tests a fine tuned model with test set saved in JSONL file."
    )
    parser.add_argument('--jsonl_path', type=str, required=True, help='Path to JSONL test examples file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to fine tuned sentence transormer model.')
    parser.add_argument('--out_path', type=str, required=True, help='Path to save files for further analysis, will create an "analysis directory".')
    parser.add_argument('--device', type=str, required=False, help='Device to run on (cpu/cuda). Default: auto')
    parser.add_argument('--batch_size', type=int, required=False, help='Training batch size. Default: auto tune')
    parser.add_argument('--margin', type=float, required=False, default=0.3, help='Defines the margin between groups. Default: 0.3')
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    analysis_path = os.path.join(args.out_path, "analysis")
    os.makedirs(analysis_path, exist_ok=True)
    print(f"📌 output path = {args.out_path}")
    print(f"📌 analysis path = {analysis_path}")

    jsonl_path = args.jsonl_path

    batch_size = args.batch_size
    if not batch_size:
        batch_size = 32

    device = args.device
    if not device:
        # check for CUDA
        device = "cpu"
        if torch.cuda.is_available():
            batch_size = 64 if not args.batch_size else args.batch_size
            device = "cuda"
    print(f"📌 Using device: {device}")
    print(f"📌 Batch size: {batch_size}")

    print("⏳ Loading model...")
    model = SentenceTransformer(args.model_path)

    print(f"📖 Reading samples from: {jsonl_path}")
    triplets = load_triplets_from_jsonl(jsonl_path)
    dataset = TripletDataset(triplets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    print("⏳ Evaluating tests...")
    accuracy, avg_gap, violations_rate, categories, best_threshold = evaluate_triplet_model(model, dataloader, args.margin, analysis_path)
    print(f"📊 Test accuracy: {accuracy:.4f} | Avg cosine similarity gap: {avg_gap:.4f} | Margin violation rate: {violations_rate:.4f} | Best threshold: {best_threshold:.2f}")
    for key in categories:
        print(f"\t{key:24}: {categories[key]} ({categories[key] / sum(categories.values()):.2%})")
    plot_triplet_tsne(model, triplets, analysis_path)
    plot_triplet_umap(model, triplets, analysis_path)

# python test_triplets.py --model_path output/20250714/FINAL-fine_tuned_model --jsonl_path output/20250714/test_triplets.jsonl --device cuda --batch_size 128 --out_path output/20250714
