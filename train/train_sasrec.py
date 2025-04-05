import os
import numpy as np
import tensorflow as tf
import pandas as pd

print("Using GPU:", tf.config.list_physical_devices('GPU'))

from LLMSeqRec.models.sasrec import SASRecModel

# --- Data Loading Functions ---
def load_train_sequences(csv_path):
    """
    Load training data: 
      - First column: user IDs (unused for model but kept for reference)
      - Columns 2 to -1: sequence of item IDs
      - Last column: next-item label
    """
    data = pd.read_csv(csv_path, header=None)
    user_ids = data.iloc[:, 0].values
    item_seqs = data.iloc[:, 1:-1].values  # (num_samples, seq_len)
    labels = data.iloc[:, -1].values         # (num_samples,)
    return user_ids, item_seqs, labels

class SASRecDataset(tf.data.Dataset):
    """
    Custom dataset for SASRec, yielding (input_seq, label, negative_samples).
    """
    def __new__(cls, item_seqs, labels, num_items, k_neg=50, batch_size=32, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((item_seqs, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(item_seqs), reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=False)
        
        def add_negatives(seq_batch, label_batch):
            label_batch = tf.cast(label_batch, tf.int32)
            neg_samples = tf.random.uniform(
                shape=(tf.shape(label_batch)[0], k_neg),
                minval=0,
                maxval=num_items,
                dtype=tf.int32
            )
            # Avoid accidental positives.
            label_exp = tf.expand_dims(label_batch, axis=1)
            label_exp = tf.tile(label_exp, [1, k_neg])
            mask = tf.equal(neg_samples, label_exp)
            resampled = tf.random.uniform(
                shape=(tf.shape(label_batch)[0], k_neg),
                minval=0,
                maxval=num_items,
                dtype=tf.int32
            )
            neg_samples = tf.where(mask, resampled, neg_samples)
            return seq_batch, label_batch, neg_samples
        
        ds = ds.map(add_negatives, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

@tf.function
def training_step(model, optimizer, item_seq, pos_label, neg_samples):
    """
    One training step with negative sampling.
    """
    with tf.GradientTape() as tape:
        logits = model(item_seq, training=True)
        batch_indices = tf.range(tf.shape(item_seq)[0], dtype=tf.int32)
        pos_indices = tf.stack([batch_indices, tf.cast(pos_label, tf.int32)], axis=1)
        pos_logits = tf.gather_nd(logits, pos_indices)  # shape (batch_size,)
        
        neg_logits = tf.map_fn(
            lambda i: tf.gather(logits[i], neg_samples[i]),
            tf.range(tf.shape(item_seq)[0]),
            fn_output_signature=tf.float32
        )
        
        pos_logits_expanded = tf.expand_dims(pos_logits, axis=1)
        all_logits = tf.concat([pos_logits_expanded, neg_logits], axis=1)
        log_probs = all_logits - tf.reduce_logsumexp(all_logits, axis=1, keepdims=True)
        pos_log_prob = log_probs[:, 0]
        loss = -tf.reduce_mean(pos_log_prob)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def load_validation_data(csv_path):
    data = pd.read_csv(csv_path, header=None)
    sequences = data.iloc[:, 1:-1].values
    labels = data.iloc[:, -1].values
    return sequences, labels

def evaluate(model, sequences, labels, top_k=10):
    hit_count, ndcg_sum, total = 0, 0, len(sequences)
    batch_size = 32
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_seq = sequences[start:end]
        batch_labels = labels[start:end]
        
        logits = model(batch_seq, training=False)
        topk_preds = tf.math.top_k(logits, k=top_k).indices.numpy()
        
        for pred, true_label in zip(topk_preds, batch_labels):
            if true_label in pred:
                hit_count += 1
                rank = np.where(pred == true_label)[0][0] + 1
                ndcg_sum += 1 / np.log2(rank + 1)
    hit_rate = hit_count / total
    ndcg = ndcg_sum / total
    return hit_rate, ndcg

def main():
    # File paths (update these as needed)
    train_csv = "LLMSeqRec/data/processed/train_sequences.csv"
    val_csv = "LLMSeqRec/data/processed/val_sequences.csv"
    
    # Hyperparameters
    BATCH_SIZE = 32
    K_NEG = 200  # Increased negative sampling
    EPOCHS = 20  # Increased epochs
    max_seq_len = 200  # Increased sequence length
    embed_dim = 256  # Increased embedding dimension
    
    # Learning rate scheduler
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Load training and validation data
    user_ids, item_seqs, labels = load_train_sequences(train_csv)
    val_seqs, val_labels = load_validation_data(val_csv)
    num_samples, seq_len = item_seqs.shape
    
    # Determine number of items (from dataset or a separate file)
    num_items = int(max(np.max(item_seqs), np.max(labels))) + 1
    
    # Build the SASRec baseline model.
    model = SASRecModel(
        num_items=num_items,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        num_blocks=2,
        num_heads=2,
        ffw_dim=256,
        dropout_rate=0.1
    )
    _ = model(item_seqs[:1], training=False)  # warm-up
    
    # Create TF dataset
    train_ds = SASRecDataset(item_seqs, labels, num_items, k_neg=K_NEG,
                             batch_size=BATCH_SIZE, shuffle=True)
    
    # Create logs folder if not exists
    os.makedirs("LLMSeqRec/logs", exist_ok=True)

    # Define log file paths for SASRec
    train_log_file = "LLMSeqRec/logs/sasrec_train_log.csv"
    metrics_log_file = "LLMSeqRec/logs/sasrec_metrics.csv"

    # Write headers if the files do not exist
    if not os.path.exists(train_log_file):
        with open(train_log_file, "w") as f:
            f.write("epoch,loss\n")
    if not os.path.exists(metrics_log_file):
        with open(metrics_log_file, "w") as f:
            f.write("epoch,hit_at_10,ndcg_at_10\n")

    # Training loop with early stopping
    best_ndcg = 0
    patience = 3
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        epoch_loss, steps = 0.0, 0
        for seq_batch, pos_batch, neg_batch in train_ds:
            loss_val = training_step(model, optimizer, seq_batch, pos_batch, neg_batch)
            epoch_loss += loss_val.numpy()
            steps += 1
            if steps % 100 == 0:
                print(f" Step {steps}, avg_loss={epoch_loss / steps:.4f}")
        avg_loss = epoch_loss / max(steps, 1)
        
        # Validation: Compute metrics
        hit, ndcg = evaluate(model, val_seqs, val_labels, top_k=10)
        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f} | Hit@10: {hit:.4f} | NDCG@10: {ndcg:.4f}")
        
        # Log the results
        with open(train_log_file, "a") as f:
            f.write(f"{epoch},{avg_loss:.4f}\n")
        with open(metrics_log_file, "a") as f:
            f.write(f"{epoch},{hit:.4f},{ndcg:.4f}\n")
        
        # Early stopping
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete!")


if __name__ == "__main__":
    main()