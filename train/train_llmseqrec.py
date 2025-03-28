import os
import numpy as np
import tensorflow as tf
import pandas as pd
import sys
print("Nilesh Misrha", sys.path)

# Use absolute import for the model.
from LLMSeqRec.models.llmseqrec import LLMSeqRecModel


########################################################################
# 1. Load and Parse Preprocessed Training Data
########################################################################

def load_train_sequences(csv_path):
    """
    Reads train_sequences.csv (no headers) and returns:
      - item_seqs: np.array of shape (num_samples, seq_len) containing item IDs
      - labels: np.array of shape (num_samples,) containing the next-item label
        (in this simplified example, we treat the last item in the row as the label)
    """
    data = pd.read_csv(csv_path, header=None)
    user_ids = data.iloc[:, 0].values
    item_seqs = data.iloc[:, 1:-1].values  # shape: (num_samples, seq_len-1)
    labels = data.iloc[:, -1].values       # shape: (num_samples,)
    return user_ids, item_seqs, labels


########################################################################
# 2. Build a TF Dataset with Negative Sampling
########################################################################

class SASRecDataset(tf.data.Dataset):
    """
    A custom Dataset to yield (input_seq, label, negative_samples).
    Implements negative sampling of size k for each example.
    """
    def __new__(cls, item_seqs, labels, num_items, k_neg=200, batch_size=32, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((item_seqs, labels))
        
        if shuffle:
            ds = ds.shuffle(buffer_size=len(item_seqs), reshuffle_each_iteration=True)
        
        ds = ds.batch(batch_size, drop_remainder=False)
        
        def add_negatives(seq, lbl):
            # seq: shape (batch_size, seq_len)
            # lbl: shape (batch_size,)
            batch_size_ = tf.shape(seq)[0]
            neg_candidates = tf.random.uniform(
                shape=[batch_size_, k_neg],
                minval=0,
                maxval=num_items,
                dtype=tf.int32
            )
            
            def filter_negatives(neg_row, label_item):
                mask = tf.equal(neg_row, tf.cast(label_item, tf.int32))
                re_draw = tf.random.uniform(
                    shape=[k_neg],
                    minval=0,
                    maxval=num_items,
                    dtype=tf.int32
                )
                return tf.where(mask, re_draw, neg_row)
            
            # Apply filtering row-wise.
            neg_candidates = tf.map_fn(
                lambda x: filter_negatives(x[0], x[1]),
                (neg_candidates, lbl),
                dtype=tf.int32
            )
            
            return (seq, lbl, neg_candidates)
        
        ds = ds.map(add_negatives, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


########################################################################
# 3. Define the Training Loop with Sampled Softmax
########################################################################

@tf.function
def training_step(model, optimizer, item_seq, pos_label, neg_samples):
    """
    Performs one training step with negative sampling.
    
    - item_seq: (batch_size, seq_len) input item IDs
    - pos_label: (batch_size,) ground-truth next item (to be cast to int32)
    - neg_samples: (batch_size, k_neg) negative candidate IDs
    """
    with tf.GradientTape() as tape:
        # Forward pass: logits shape (batch_size, num_items)
        logits = model(item_seq, training=True)
        
        # Prepare indices for gathering positive logits.
        batch_indices = tf.range(tf.shape(item_seq)[0], dtype=tf.int32)
        # Ensure pos_label is int32.
        pos_label = tf.cast(pos_label, tf.int32)
        pos_indices = tf.stack([batch_indices, pos_label], axis=1)
        pos_logits = tf.gather_nd(logits, pos_indices)  # shape (batch_size,)
        
        # Gather negative logits using tf.map_fn to remain in scope.
        # For each example, gather logits at indices given by neg_samples.
        neg_logits = tf.map_fn(
            lambda i: tf.gather(logits[i], neg_samples[i]),
            tf.range(tf.shape(item_seq)[0]),
            fn_output_signature=tf.float32
        )  # shape (batch_size, k_neg)
        
        # Concatenate positive logits and negative logits.
        pos_logits_expanded = tf.expand_dims(pos_logits, axis=1)  # (batch_size, 1)
        all_logits = tf.concat([pos_logits_expanded, neg_logits], axis=1)  # (batch_size, 1+k_neg)
        
        # Compute log-softmax over the concatenated logits.
        log_probs = all_logits - tf.reduce_logsumexp(all_logits, axis=1, keepdims=True)
        pos_log_prob = log_probs[:, 0]  # negative log likelihood for positive items.
        loss = -tf.reduce_mean(pos_log_prob)
        
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


########################################################################
# 4. Main Training Function
########################################################################

def main():
    # File paths
    train_csv = "LLMSeqRec/data/processed/train_sequences.csv"
    llm_emb_path = "LLMSeqRec/data/processed/llm_embeddings.npy"
    
    # Hyperparameters
    BATCH_SIZE = 64
    K_NEG = 200
    EPOCHS = 3
    
    # A) Load training data.
    user_ids, item_seqs, labels = load_train_sequences(train_csv)
    num_samples, seq_len = item_seqs.shape
    print(f"Loaded training data: {num_samples} sequences, each length {seq_len}.")
    
    # Infer num_items and embedding dimension from the pretrained embeddings.
    llm_embeddings = np.load(llm_emb_path)
    num_items, embed_dim = llm_embeddings.shape
    print(f"Loaded LLM embeddings: num_items={num_items}, embed_dim={embed_dim}")
    
    # B) Build Model & Load LLM Embeddings.
    model = LLMSeqRecModel(
        num_items=num_items,
        max_seq_len=seq_len,
        embed_dim=embed_dim,
        num_blocks=2,
        num_heads=2,
        ffw_dim=1024,
        dropout_rate=0.1
    )
    model.load_llm_embeddings(llm_embeddings)
    _ = model(item_seqs[:1], training=False)
    print("[INFO] Model built successfully.")
    
    # C) Prepare TensorFlow Dataset.
    train_ds = SASRecDataset(item_seqs, labels, num_items, k_neg=K_NEG,
                             batch_size=BATCH_SIZE, shuffle=True)
    
    # D) Training Loop.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        epoch_loss = 0.0
        steps = 0
        
        for (seq_batch, pos_batch, neg_batch) in train_ds:
            loss_val = training_step(model, optimizer, seq_batch, pos_batch, neg_batch)
            epoch_loss += loss_val.numpy()
            steps += 1
            if steps % 100 == 0:
                print(f" Step {steps}, avg_loss={epoch_loss / steps:.4f}")
        
        epoch_loss /= max(steps, 1)
        print(f"[Epoch {epoch}]  Average Loss: {epoch_loss:.4f}")
    
    print("Training complete!")


if __name__ == "__main__":
    main()
