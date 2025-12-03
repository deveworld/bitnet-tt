"""
Multi-Head Attention implementation using TT-NN.

This module implements the attention mechanism with:
- Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE) using ttnn.experimental.rotary_embedding
- attn_sub_norm applied after attention, before output projection
"""

import math

import numpy as np
import ttnn
from numpy.typing import NDArray

from bitnet_tt.layers.bitlinear import Linear, RMSNorm, numpy_to_ttnn


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
        Tuple of (cos, sin) arrays of shape (1, 1, max_seq_len, dim)
    """
    # Compute inverse frequencies for half the dimension
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)  # (max_seq_len, dim//2)

    # Concatenate to get full dimension (matching transformers implementation)
    emb = np.concatenate([freqs, freqs], axis=-1)  # (max_seq_len, dim)

    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)

    # Reshape for ttnn.experimental.rotary_embedding: (1, 1, max_seq_len, dim)
    cos = cos.reshape(1, 1, max_seq_len, dim)
    sin = sin.reshape(1, 1, max_seq_len, dim)

    return cos, sin


class MultiHeadAttention:
    """
    TT-NN native Multi-Head Attention with GQA and RoPE.

    Architecture (matching HuggingFace Transformers):
        Q, K, V = q_proj(x), k_proj(x), v_proj(x)  # Plain linear
        Q, K = apply_rope(Q, K)
        attn_output = scaled_dot_product_attention(Q, K, V)
        attn_output = attn_sub_norm(attn_output)  # RMSNorm before o_proj
        output = o_proj(attn_output)  # Plain linear
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

        # Projections (plain linear, no input norm)
        self.q_proj = Linear(hidden_size, num_attention_heads * self.head_dim, device)
        self.k_proj = Linear(hidden_size, num_key_value_heads * self.head_dim, device)
        self.v_proj = Linear(hidden_size, num_key_value_heads * self.head_dim, device)
        self.o_proj = Linear(num_attention_heads * self.head_dim, hidden_size, device)

        # Sub-norm applied after attention, before o_proj
        self.attn_sub_norm = RMSNorm(hidden_size, device, eps)

        # Precompute RoPE frequencies
        cos, sin = precompute_freqs_cis(self.head_dim, max_position_embeddings, rope_theta)
        self.cos_cached = numpy_to_ttnn(cos, device)
        self.sin_cached = numpy_to_ttnn(sin, device)

    def load_weights(
        self,
        q_weight: NDArray[np.floating],
        k_weight: NDArray[np.floating],
        v_weight: NDArray[np.floating],
        o_weight: NDArray[np.floating],
        attn_sub_norm_weight: NDArray[np.floating] | None = None,
    ) -> None:
        """
        Load all projection weights.

        Args:
            q_weight: Query projection weight
            k_weight: Key projection weight
            v_weight: Value projection weight
            o_weight: Output projection weight
            attn_sub_norm_weight: Sub-norm weight (applied after attention, before o_proj)
        """
        self.q_proj.load_weights(q_weight)
        self.k_proj.load_weights(k_weight)
        self.v_proj.load_weights(v_weight)
        self.o_proj.load_weights(o_weight)

        if attn_sub_norm_weight is not None:
            self.attn_sub_norm.load_weights(attn_sub_norm_weight)

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

        # Apply RoPE manually (ttnn.experimental.rotary_embedding has shape issues)
        # Slice cos/sin to current sequence length
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        # Apply rotation to query and key
        query = self._apply_rope(query, cos, sin)
        key = self._apply_rope(key, cos, sin)

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

        # Apply sub-norm before output projection (key difference from standard attention!)
        attn_output = self.attn_sub_norm(attn_output)

        # Output projection
        return self.o_proj(attn_output)

    def _apply_rope(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Apply rotary position embeddings.

        Args:
            x: Input tensor of shape (batch, num_heads, seq, head_dim)
            cos: Cosine frequencies of shape (1, 1, seq, head_dim)
            sin: Sine frequencies of shape (1, 1, seq, head_dim)

        Returns:
            Tensor with RoPE applied
        """
        # Get the number of heads to expand cos/sin
        num_heads = x.shape[1]

        # Expand cos/sin to match x shape for element-wise ops
        # (1, 1, seq, head_dim) -> (1, num_heads, seq, head_dim)
        cos_expanded = ttnn.repeat(cos, ttnn.Shape([1, num_heads, 1, 1]))
        sin_expanded = ttnn.repeat(sin, ttnn.Shape([1, num_heads, 1, 1]))

        # rotate_half: split in half, negate second half, swap order
        # x = [x1, x2] -> [-x2, x1]
        half_dim = self.head_dim // 2

        # Get first and second half of x
        x1 = x[:, :, :, :half_dim]
        x2 = x[:, :, :, half_dim:]

        # rotate_half: [-x2, x1]
        x_rotated = ttnn.concat([ttnn.neg(x2), x1], dim=-1)

        # Apply rotation: x * cos + rotate_half(x) * sin
        x_cos = ttnn.multiply(x, cos_expanded)
        x_rot_sin = ttnn.multiply(x_rotated, sin_expanded)
        return ttnn.add(x_cos, x_rot_sin)
