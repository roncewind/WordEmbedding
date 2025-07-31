import argparse

from sentence_transformers import SentenceTransformer, util

# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="compare_words", description="Compares two words."
    )
    parser.add_argument('--model_path', type=str, required=True, help='Path to fine tuned sentence transormer model.')
    parser.add_argument('word1', help='First word to compare.')
    parser.add_argument('word2', help='Second word to compare.')
    args = parser.parse_args()

    print("⏳ Loading model...")
    model = SentenceTransformer(args.model_path)

    word1 = args.word1
    word2 = args.word2

    # Encode words to embeddings
    embedding1 = model.encode(word1, convert_to_tensor=True)
    embedding2 = model.encode(word2, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_similarity = util.cos_sim(embedding1, embedding2).item()

    print(f"Cosine similarity between '{word1}' and '{word2}': {cosine_similarity:.4f}")
