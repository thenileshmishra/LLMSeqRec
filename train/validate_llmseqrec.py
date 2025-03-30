import numpy as np
import pandas as pd
import tensorflow as tf
from LLMSeqRec.models.llmseqrec import LLMSeqRecModel

def load_validation_data(csv_path):
    data = pd.read_csv(csv_path, header=None)
    print("VAL CSV shape:", data.shape)  # Debug

    user_ids = data.iloc[:, 0].values
    sequences = data.iloc[:, 1:-1].values
    labels = data.iloc[:, -1].values

    print("Loaded sequences shape:", sequences.shape)  # Debug
    print("Loaded labels shape:", labels.shape)
    return sequences, labels

def evaluate(model, sequences, labels, top_k=10):
    hit_count = 0
    ndcg_sum = 0
    total = len(sequences)

    batch_size = 32
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_seq = sequences[start:end]
        batch_labels = labels[start:end]

        logits = model(batch_seq, training=False)  # (batch_size, num_items)
        topk_preds = tf.math.top_k(logits, k=top_k).indices.numpy()  # (batch_size, top_k)

        for pred, true_label in zip(topk_preds, batch_labels):
            if true_label in pred:
                hit_count += 1
                rank = np.where(pred == true_label)[0][0] + 1  # rank starts from 1
                ndcg_sum += 1 / np.log2(rank + 1)

    hit_rate = hit_count / total
    ndcg = ndcg_sum / total

    return hit_rate, ndcg

def run_validation():
    val_csv = "LLMSeqRec/data/processed/val_sequences.csv"
    emb_path = "LLMSeqRec/data/processed/llm_embeddings.npy"

    sequences, labels = load_validation_data(val_csv)
    num_items, embed_dim = np.load(emb_path).shape

    # Rebuild model
    model = LLMSeqRecModel(
        num_items=num_items,
        max_seq_len=sequences.shape[1],
        embed_dim=embed_dim,
        num_blocks=2,
        num_heads=2,
        ffw_dim=1024,
        dropout_rate=0.1
    )
    model.load_llm_embeddings(np.load(emb_path))
    _ = model(sequences[:1], training=False)  # warm-up

    hit, ndcg = evaluate(model, sequences, labels, top_k=10)
    print(f"\n[Validation Results]")
    print(f"Hit@10:  {hit:.4f}")
    print(f"NDCG@10: {ndcg:.4f}")

if __name__ == "__main__":
    run_validation()