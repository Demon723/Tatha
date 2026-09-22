"""
tatha_attention.py
==================
Attention mechanism for Tatha predictive coding layers.

Provides:
  - Self-attention over temporal context
  - Multi-head attention for belief layer
  - Cross-attention between layers
  - Transformer-style positional encoding
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """Attention mechanism configuration."""
    num_heads: int = 4
    attention_dim: int = 8
    dropout: float = 0.1
    max_seq_len: int = 100


class MultiHeadAttention:
    """Multi-head self-attention for belief sequences."""
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.d_model = config.attention_dim * config.num_heads
        self.head_dim = config.attention_dim

        # Query, Key, Value projections
        self.W_q = np.random.randn(self.d_model, self.d_model) * 0.02
        self.W_k = np.random.randn(self.d_model, self.d_model) * 0.02
        self.W_v = np.random.randn(self.d_model, self.d_model) * 0.02
        self.W_o = np.random.randn(self.d_model, self.d_model) * 0.02
        self.b_q = np.zeros(self.d_model)
        self.b_k = np.zeros(self.d_model)
        self.b_v = np.zeros(self.d_model)
        self.b_o = np.zeros(self.d_model)

    def _scale_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention."""
        dk = K.shape[-1]
        scores = Q @ K.T / np.sqrt(dk)
        if mask is not None:
            scores = scores + mask * -1e9
        attn_weights = self._softmax(scores)
        return attn_weights @ V, attn_weights

    def _softmax(self, x):
        """Numerically stable softmax."""
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x, mask=None):
        """
        Forward pass through multi-head attention.
        x: (seq_len, d_model)
        Returns: (seq_len, d_model)
        """
        seq_len = x.shape[0]

        # Project
        Q = x @ self.W_q + self.b_q
        K = x @ self.W_k + self.b_k
        V = x @ self.W_v + self.b_v

        # Reshape into heads: (num_heads, seq_len, head_dim)
        H = self.config.num_heads
        Q = Q.reshape(H, seq_len, self.head_dim).transpose(1, 0, 2)
        K = K.reshape(H, seq_len, self.head_dim).transpose(1, 0, 2)
        V = V.reshape(H, seq_len, self.head_dim).transpose(1, 0, 2)

        # Attention per head
        outputs = []
        attn_weights_list = []
        for h in range(H):
            out, attn = self._scale_dot_product_attention(
                Q[h], K[h], V[h], mask
            )
            outputs.append(out)
            attn_weights_list.append(attn)

        # Concatenate heads
        concat = np.concatenate(outputs, axis=-1)  # (seq_len, d_model)

        # Output projection
        output = concat @ self.W_o + self.b_o
        return output, attn_weights_list


class TemporalAttentionWrapper:
    """Wraps attention around the belief layer for temporal context."""
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.attention = MultiHeadAttention(config)
        self.layer_norm_1 = np.ones(config.attention_dim * config.num_heads)
        self.layer_norm_2 = np.ones(config.attention_dim * config.num_heads)
        self.ff_weights = np.random.randn(
            config.attention_dim * config.num_heads,
            config.attention_dim * config.num_heads
        ) * 0.02
        self.ff_bias = np.zeros(config.attention_dim * config.num_heads)
        self._context_buffer: list[np.ndarray] = []

    def update_context(self, belief: np.ndarray):
        """Add current belief to temporal context."""
        self._context_buffer.append(belief.copy())
        if len(self._context_buffer) > self.config.max_seq_len:
            self._context_buffer.pop(0)

    def forward(self, belief: np.ndarray) -> np.ndarray:
        """
        Apply attention over temporal context.
        belief: (attention_dim * num_heads,)
        Returns: updated belief with attention
        """
        self.update_context(belief)

        if len(self._context_buffer) < 2:
            return belief

        # Stack context
        seq = np.array(self._context_buffer)  # (seq_len, d_model)

        # Apply self-attention
        attended, _ = self.attention.forward(seq)

        # Use the latest position's output
        output = attended[-1]

        # Residual connection + layer norm
        output = output + belief
        output = output * self.layer_norm_1 / np.linalg.norm(output + 1e-8)

        # Feed-forward
        ff_out = output @ self.ff_weights + self.ff_bias
        output = output + ff_out
        output = output * self.layer_norm_2 / np.linalg.norm(output + 1e-8)

        return output


class PositionalEncoding:
    """Sinusoidal positional encoding for belief sequences."""
    def __init__(self, d_model: int, max_len: int = 100):
        self.d_model = d_model
        self.max_len = max_len
        self.encoding = self._build_encoding()

    def _build_encoding(self) -> np.ndarray:
        """Build sinusoidal positional encoding matrix."""
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def add(self, x: np.ndarray) -> np.ndarray:
        """Add positional encoding to input."""
        seq_len = x.shape[0]
        return x + self.encoding[:seq_len]
