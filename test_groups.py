import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer, losses
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------------------------------------------
def average_pairwise_similarity(embeddings):
    sim_matrix = cosine_similarity(embeddings)
    tril = np.tril_indices(len(sim_matrix), k=-1)
    return sim_matrix[tril].mean()


# -----------------------------------------------------------------------------
def compute_group_similarity(model, group_words):
    # clean-up our words, make sure they're strings and not blank
    words = [w for w in group_words if isinstance(w, str) and w.strip()]
    # need at least 2 words to do a similarity
    if len(words) < 2:
        return None, None
    embeddings = model.encode(words)
    ave_sim = average_pairwise_similarity(embeddings)
    return ave_sim, embeddings


# -----------------------------------------------------------------------------
def visualize_tsne(group_embeddings_dict, base_path, title="t-SNE of Sample Groups"):
    all_vectors, all_labels = [], []
    for group_id, (words, embeddings) in group_embeddings_dict.items():
        all_vectors.extend(embeddings)
        all_labels.extend([f"group_{group_id}"] * len(embeddings))

    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(all_vectors)
    df = pd.DataFrame({'x': reduced[:, 0], 'y': reduced[:, 1], 'group': all_labels})
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='group', style='group', s=60)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(base_path + '-TSNE.png')


# -----------------------------------------------------------------------------
def load_groups_from_csv(csv_path, group_col='id', data_col='word', min_size=2):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[data_col])
    df[data_col] = df[data_col].astype(str)
    grouped = df.groupby(group_col)[data_col].apply(list)
    groups = {gid: words for gid, words in grouped.items() if len(words) >= min_size}
    return groups


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="test_groups", description="Tests a fine tuned model with new groups."
    )
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV test examples file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to fine tuned sentence transormer model.')
    parser.add_argument('--visualize_groups', type=int, required=False, default=3, help='How many groups to visualize with t-SNE. Default: 3')
    parser.add_argument('--group_col', type=str, required=False, default='id', help='Column header in CSV file that denotes the group. Default: \'id\'')
    parser.add_argument('--data_col', type=str, required=False, default='word', help='Column header in CSV file that denotes the data. Default: \'word\'')
    parser.add_argument('--min_size', type=int, required=False, default=2, help='Defines the minium size of a \'small\' group. Default: 2')
    args = parser.parse_args()

    csv_path = args.csv_path
    base_path, _ = os.path.splitext(csv_path)

    print("⏳ Loading model...")
    model = SentenceTransformer(args.model_path)

    print("📖 Reading groups from: {csv_path}")
    groups = load_groups_from_csv(csv_path, group_col=args.group_col, data_col=args.data_col, min_size=args.min_size)

    results = []
    tsne_samples = {}

    print(f"🤔 Evaluating {len(groups)} groups...")
    for i, (gid, words) in enumerate(groups.items()):
        avg_sim, emb = compute_group_similarity(model, words)
        if avg_sim is not None:
            results.append({'group_id': gid, 'avg_similarity': avg_sim, 'size': len(words)})
            if len(tsne_samples) < args.visualize_groups:
                tsne_samples[gid] = (words, emb)

    results_df = pd.DataFrame(results)
    results_df.sort_values("avg_similarity", ascending=False, inplace=True)

    print("\n📊 Group Similarity Summary:")
    print(results_df.round(3).head(10))

    output_csv = base_path + "-test-groups.csv"
    print(f"📝 Saving results to {output_csv}...")
    results_df.to_csv(output_csv, index=False)

    if tsne_samples:
        print("🎨 Visualizing sample groups with t-SNE...")
        visualize_tsne(tsne_samples, base_path)
