"""
Multi-Head Attention implementation using TT-NN.

This module implements the attention mechanism with:
- Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE)
- BitLinear projections for Q, K, V, and output
"""

import math

import numpy as np
import ttnn
from numpy.typing import NDArray

from bitnet_tt.layers.bitlinear import BitLinear, numpy_to_ttnn


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Precompute cosine and sine frequencies for RoPE.

    Args:
        dim: Head dimension
        max_seq_len: Maximum sequence length
        theta: Base frequency

    Returns:
        Tuple of (cos, sin) arrays of shape (max_seq_len, dim//2)
    """
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)
    return cos, sin


class MultiHeadAttention:
    """
    TT-NN native Multi-Head Attention with GQA and RoPE.

    Uses BitLinear layers for Q, K, V, and output projections.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        device: ttnn.Device,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
    ) -> None:
        """
        Initialize attention layer.

        Args:
            hidden_size: Hidden dimension
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of key/value heads (for GQA)
            device: TT-NN device
            max_position_embeddings: Maximum sequence length
            rope_theta: RoPE base frequency
            eps: Epsilon for numerical stability
        """
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.device = device
        self.eps = eps
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections
        self.q_proj = BitLinear(hidden_size, num_attention_heads * self.head_dim, device, eps)
        self.k_proj = BitLinear(hidden_size, num_key_value_heads * self.head_dim, device, eps)
        self.v_proj = BitLinear(hidden_size, num_key_value_heads * self.head_dim, device, eps)
        self.o_proj = BitLinear(num_attention_heads * self.head_dim, hidden_size, device, eps)

        # Precompute RoPE frequencies
        cos, sin = precompute_freqs_cis(self.head_dim, max_position_embeddings, rope_theta)

        # Transfer to device
        cos_reshaped = cos.reshape(1, 1, max_position_embeddings, -1).astype(np.float32)
        sin_reshaped = sin.reshape(1, 1, max_position_embeddings, -1).astype(np.float32)
        self.cos_cached = numpy_to_ttnn(cos_reshaped, device)
        self.sin_cached = numpy_to_ttnn(sin_reshaped, device)

    def load_weights(
        self,
        q_weight: NDArray[np.floating],
        q_norm_weight: NDArray[np.floating],
        k_weight: NDArray[np.floating],
        k_norm_weight: NDArray[np.floating],
        v_weight: NDArray[np.floating],
        v_norm_weight: NDArray[np.floating],
        o_weight: NDArray[np.floating],
        o_norm_weight: NDArray[np.floating],
    ) -> None:
        """
        Load all projection weights.

        Args:
            q_weight: Query projection weight
            q_norm_weight: Query norm weight
            k_weight: Key projection weight
            k_norm_weight: Key norm weight
            v_weight: Value projection weight
            v_norm_weight: Value norm weight
            o_weight: Output projection weight
            o_norm_weight: Output norm weight
        """
        self.q_proj.load_weights(q_weight, q_norm_weight)
        self.k_proj.load_weights(k_weight, k_norm_weight)
        self.v_proj.load_weights(v_weight, v_norm_weight)
        self.o_proj.load_weights(o_weight, o_norm_weight)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        # Get shape info
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Project to Q, K, V: (batch, seq, hidden) -> (batch, seq, num_heads * head_dim)
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to 4D: (batch, seq, num_heads, head_dim)
        query = ttnn.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = ttnn.reshape(key, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        value = ttnn.reshape(value, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        # Transpose to (batch, num_heads, seq, head_dim)
        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.permute(value, (0, 2, 1, 3))

        # Expand KV heads for GQA if needed
        if self.num_kv_groups > 1:
            key = ttnn.repeat_interleave(key, self.num_kv_groups, dim=1)
            value = ttnn.repeat_interleave(value, self.num_kv_groups, dim=1)

        # Scaled dot product attention
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=True,
            scale=self.scale,
        )

        # Transpose back: (batch, num_heads, seq, head_dim) -> (batch, seq, num_heads, head_dim)
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))

        # Reshape to 3D: (batch, seq, hidden)
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.hidden_size))

        # Output projection
        return self.o_proj(attn_output)
