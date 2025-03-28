import tensorflow as tf

class TransformerBlock(tf.keras.layers.Layer):
    """
    A custom Transformer Encoder block with:
      - Multi-Head Self-Attention (causal masking)
      - Dropout
      - Residual & LayerNorm
      - Feed-Forward (Dense -> Dropout -> Residual + LayerNorm)
    """
    def __init__(self, embed_dim, num_heads, ffw_dim, rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-Head Attention
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads,  # key_dim * num_heads = embed_dim
            dropout=rate
        )

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Feed Forward Network
        self.ffw = tf.keras.Sequential([
            tf.keras.layers.Dense(ffw_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])

        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training, causal_mask=None):
        # x shape: (batch_size, seq_len, embed_dim)
        # Apply Self-Attention
        attn_output = self.attention(query=x,
                                     value=x,
                                     key=x,
                                     attention_mask=causal_mask,
                                     training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)  # Residual

        # Feed Forward
        ffw_output = self.ffw(out1)
        ffw_output = self.dropout2(ffw_output, training=training)
        out2 = self.norm2(out1 + ffw_output)  # Residual
        return out2

def create_causal_attention_mask(batch_size, seq_len):
    """
    Creates a lower-triangular (causal) mask of shape:
      (batch_size, num_heads=1?, seq_len, seq_len)
    so each position i only attends to positions [0..i].
    For MultiHeadAttention in TF2, attention_mask can be 1.0/0.0 or True/False.
    We'll use 1.0 for "allow" and 0.0 for "mask out."
    """
    # Lower-triangular: i >= j
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # shape (seq_len, seq_len)
    # Expand for batch_size and heads dimension if needed
    mask = tf.reshape(mask, (1, 1, seq_len, seq_len))  # (1, 1, seq_len, seq_len)
    mask = tf.tile(mask, [batch_size, 1, 1, 1])        # (batch_size, 1, seq_len, seq_len)
    return mask


class LLMSeqRecModel(tf.keras.Model):
    """
    SASRec-like model with:
      - Dual embeddings: (1) LLM-based, (2) ID-based, summed
      - Trainable positional embeddings
      - Multiple Transformer blocks
      - Output: logit scores over all items from final hidden state
    """
    def __init__(self,
                 num_items,
                 max_seq_len=150,
                 embed_dim=768,   # same as LLM
                 num_blocks=2,
                 num_heads=2,
                 ffw_dim=1024,   # feed-forward hidden dim
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # 1) LLM-based embedding (pretrained weights will be loaded)
        #    Keep it trainable for fine-tuning
        self.llm_item_embedding = tf.keras.layers.Embedding(
            input_dim=num_items,
            output_dim=embed_dim,
            embeddings_initializer='zeros',  # We'll set_weights() later
            name="llm_item_embedding",
            trainable=True
        )

        # 2) ID-based embedding (trainable from random init)
        self.id_item_embedding = tf.keras.layers.Embedding(
            input_dim=num_items,
            output_dim=embed_dim,
            embeddings_initializer='glorot_uniform',
            name="id_item_embedding",
            trainable=True
        )

        # Positional embedding
        self.pos_embedding = tf.keras.layers.Embedding(
            input_dim=max_seq_len,
            output_dim=embed_dim,
            name="pos_embedding",
            trainable=True
        )

        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(embed_dim=embed_dim,
                             num_heads=num_heads,
                             ffw_dim=ffw_dim,
                             rate=dropout_rate)
            for _ in range(num_blocks)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def load_llm_embeddings(self, pretrained_weights):
        """
        Load the pretrained LLM embeddings (shape = (num_items, embed_dim))
        into self.llm_item_embedding.
        """
        if pretrained_weights.shape != (self.num_items, self.embed_dim):
            raise ValueError(f"Expected shape ({self.num_items}, {self.embed_dim}), "
                             f"got {pretrained_weights.shape}")
        self.llm_item_embedding.build((None,))
        self.llm_item_embedding.set_weights([pretrained_weights])
        print("[INFO] LLM embeddings loaded and ready for fine-tuning.")

    @property
    def combined_item_embeddings(self):
        """
        Return the sum of LLM and ID embeddings for all items:
          shape = (num_items, embed_dim)
        Used for 'tied' output layer scoring.
        """
        return (self.llm_item_embedding.weights[0] +
                self.id_item_embedding.weights[0])  # shape: (num_items, embed_dim)

    def call(self, item_seq, training=False):
        """
        Forward pass:
         item_seq: (batch_size, seq_len) of item IDs
         returns: logits for the *final position* in the sequence, shape (batch_size, num_items)
        """
        batch_size = tf.shape(item_seq)[0]
        seq_len = tf.shape(item_seq)[1]

        # 1) Build item + position embeddings
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = tf.expand_dims(positions, 0)  # shape (1, seq_len)
        positions = tf.tile(positions, [batch_size, 1])  # shape (batch_size, seq_len)

        llm_emb = self.llm_item_embedding(item_seq)  # (batch_size, seq_len, embed_dim)
        id_emb = self.id_item_embedding(item_seq)    # (batch_size, seq_len, embed_dim)
        pos_emb = self.pos_embedding(positions)      # (batch_size, seq_len, embed_dim)

        x = llm_emb + id_emb + pos_emb  # sum-based fusion
        x = self.dropout(x, training=training)

        # 2) Causal mask
        causal_mask = create_causal_attention_mask(batch_size, seq_len)
        # Some Keras versions accept `attention_mask` as a boolean
        # but we can pass a float mask (1.0 for keep, 0.0 for ignore).
        # We'll just pass it directly.

        # 3) Pass through Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x, training=training, causal_mask=causal_mask)

        # x shape after blocks: (batch_size, seq_len, embed_dim)

        # 4) Final hidden state = x[:, -1, :]
        #    i.e. the embedding for the last item in the sequence
        final_state = x[:, -1, :]  # (batch_size, embed_dim)

        # 5) Dot-product with combined item embeddings => logits
        #    shape: (batch_size, num_items)
        logits = tf.matmul(final_state, self.combined_item_embeddings, transpose_b=True)

        return logits

    def get_config(self):
        # For Keras serialization (optional)
        config = super().get_config()
        config.update({
            "num_items": self.num_items,
            "max_seq_len": self.max_seq_len,
            "embed_dim": self.embed_dim,
            "transformer_blocks": len(self.transformer_blocks),
        })
        return config
