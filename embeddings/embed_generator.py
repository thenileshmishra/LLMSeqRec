import os
import numpy as np
import pandas as pd

def main():
    """
    Reads item_metadata.csv, uses a sentence-transformer model to generate embeddings,
    and saves them as a .npy file.
    """
    # 1. Install Dependencies (commented out, as you'd typically run pip install externally)
    #    pip install sentence-transformers
    
    # 2. Import the SentenceTransformer class
    from sentence_transformers import SentenceTransformer
    
    # 3. Define paths
    metadata_path = "../data/processed/item_metadata.csv"
    output_path = "../data/processed/llm_embeddings.npy"
    
    # 4. Load item metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    meta_df = pd.read_csv(metadata_path)
    if 'movieId' not in meta_df.columns or 'metadata' not in meta_df.columns:
        raise ValueError("Expected columns [movieId, metadata] in item_metadata.csv")
    
    # 5. Prepare a list of texts
    #    We'll keep track of item IDs in the same order so the array index matches the item ID order.
    item_ids = meta_df['movieId'].tolist()
    texts = meta_df['metadata'].astype(str).tolist()  # Ensure all are strings
    
    # 6. Load the pre-trained model (768-dimensional embeddings)
    model_name = "sentence-transformers/all-roberta-large-v1"
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    

    #7 Encode all metadata in a single batch (might need batch_size parameter if memory is limited)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    
    # 8. Convert embeddings to a NumPy array
    embeddings = np.array(embeddings)  # shape: (num_items, 768)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # 9. Save embeddings
    os.makedirs('embeddings', exist_ok=True)
    np.save(output_path, embeddings)
    
    print(f"Embeddings saved to {output_path}")
    print("Done.")

if __name__ == "__main__":
    main()
