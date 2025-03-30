import tensorflow as tf

class TransformerBlock(tf.keras.layers.Layer):
    """
    A custom Transformer Encoder block with:
      - Multi-Head Self-Attention (with causal masking)
      - Dropout
      - Residual & LayerNormalization
      - Feed-Forward network
    """
    def __init__(self, embed_dim, num_heads, ffw_dim, rate=0.1):
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads,  # ensure key_dim * num_heads = embed_dim
            dropout=rate
        )
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.ffw = tf.keras.Sequential([
            tf.keras.layers.Dense(ffw_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x, training, causal_mask=None):
        attn_output = self.attention(query=x,
                                     value=x,
                                     key=x,
                                     attention_mask=causal_mask,
                                     training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)
        
        ffw_output = self.ffw(out1)
        ffw_output = self.dropout2(ffw_output, training=training)
        out2 = self.norm2(out1 + ffw_output)
        return out2

def create_causal_attention_mask(batch_size, seq_len):
    """
    Creates a lower-triangular causal mask of shape (batch_size, 1, seq_len, seq_len)
    """
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    mask = tf.reshape(mask, (1, 1, seq_len, seq_len))
    mask = tf.tile(mask, [batch_size, 1, 1, 1])
    return mask

class SASRecModel(tf.keras.Model):
    """
    Baseline SASRec model using only ID-based embeddings.
    
    Architecture:
      - ID-based embedding + positional embedding (summed)
      - Multiple Transformer blocks with causal masking
      - Final dot-product with learned embeddings (tied weights)
    """
    def __init__(self,
                 num_items,
                 max_seq_len=150,
                 embed_dim=128,
                 num_blocks=2,
                 num_heads=2,
                 ffw_dim=256,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # ID-based embedding: only learnable embeddings from scratch.
        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=num_items,
            output_dim=embed_dim,
            embeddings_initializer='glorot_uniform',
            name="item_embedding",
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
    
    @property
    def tied_item_embeddings(self):
        """
        Use the same item embeddings for output scoring.
        """
        return self.item_embedding.weights[0]
    
    def call(self, item_seq, training=False):
        """
        Forward pass:
          - item_seq: (batch_size, seq_len) of item IDs.
          - returns: logits for the final position (batch_size, num_items)
        """
        batch_size = tf.shape(item_seq)[0]
        seq_len = tf.shape(item_seq)[1]
        
        # Generate positional indices and embed them.
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = tf.expand_dims(positions, 0)
        positions = tf.tile(positions, [batch_size, 1])
        
        # Embed items and positions.
        id_emb = self.item_embedding(item_seq)  # (batch_size, seq_len, embed_dim)
        pos_emb = self.pos_embedding(positions)   # (batch_size, seq_len, embed_dim)
        
        # Sum the embeddings.
        x = id_emb + pos_emb
        x = self.dropout(x, training=training)
        
        # Create causal mask.
        causal_mask = create_causal_attention_mask(batch_size, seq_len)
        
        # Pass through Transformer blocks.
        for block in self.transformer_blocks:
            x = block(x, training=training, causal_mask=causal_mask)
        
        # Use final hidden state (last token in sequence).
        final_state = x[:, -1, :]  # shape: (batch_size, embed_dim)
        
        # Compute logits via dot product with tied item embeddings.
        logits = tf.matmul(final_state, self.tied_item_embeddings, transpose_b=True)
        return logits
