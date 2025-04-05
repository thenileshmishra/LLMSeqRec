import os
import numpy as np
import pandas as pd
import torch
torch.manual_seed(42)


os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"  # Optional

def main():
    metadata_path = "data/processed/item_metadata.csv"
    output_path = "data/processed/llm_embeddings.npy"

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    meta_df = pd.read_csv(metadata_path)
    if 'movieId' not in meta_df.columns or 'metadata' not in meta_df.columns:
        raise ValueError("Expected columns [movieId, metadata] in item_metadata.csv")

    item_ids = meta_df['movieId'].tolist()
    texts = meta_df['metadata'].astype(str).tolist()

    from sentence_transformers import SentenceTransformer
    model_name = "sentence-transformers/all-roberta-large-v1"
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings)
    np.save('data/processed/movie_ids.npy', np.array(item_ids))


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings saved to {output_path}")
    print("Done.")

if __name__ == "__main__":
    main()
