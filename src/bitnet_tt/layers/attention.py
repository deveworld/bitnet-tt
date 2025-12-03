"""
Multi-Head Attention implementation using TT-NN.

This module implements the attention mechanism with:
- Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE)
- KV-Cache for efficient autoregressive generation
- attn_sub_norm applied after attention, before output projection
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import ttnn
from numpy.typing import NDArray

from bitnet_tt.layers.bitlinear import Linear, RMSNorm, numpy_to_ttnn


@dataclass
class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.

    Stores past key and value states to avoid recomputation.
    """
    key_cache: ttnn.Tensor | None = None
    value_cache: ttnn.Tensor | None = None
    seq_len_cached: int = 0

    def update(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        device: ttnn.Device,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Update cache with new key/value states.

        Args:
            key_states: New key states (batch, num_kv_heads, seq_len, head_dim)
            value_states: New value states (batch, num_kv_heads, seq_len, head_dim)
            device: TT-NN device

        Returns:
            Updated (key_states, value_states) including cached values
        """
        if self.key_cache is None:
            # First token - just store
            self.key_cache = key_states
            self.value_cache = value_states
            self.seq_len_cached = key_states.shape[2]
            return key_states, value_states

        # Concatenate with cached values along sequence dimension
        self.key_cache = ttnn.concat([self.key_cache, key_states], dim=2)
        self.value_cache = ttnn.concat([self.value_cache, value_states], dim=2)
        self.seq_len_cached = self.key_cache.shape[2]

        return self.key_cache, self.value_cache

    def reset(self) -> None:
        """Reset the cache."""
        self.key_cache = None
        self.value_cache = None
        self.seq_len_cached = 0


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

    # Reshape for broadcasting: (1, 1, max_seq_len, dim)
    cos = cos.reshape(1, 1, max_seq_len, dim)
    sin = sin.reshape(1, 1, max_seq_len, dim)

    return cos, sin


class MultiHeadAttention:
    """
    TT-NN native Multi-Head Attention with GQA, RoPE, and KV-Cache.

    Architecture (matching HuggingFace Transformers):
        Q, K, V = q_proj(x), k_proj(x), v_proj(x)  # Plain linear
        Q, K = apply_rope(Q, K)
        K, V = update_kv_cache(K, V)  # Cache for generation
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
        layer_idx: int = 0,
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
            layer_idx: Layer index for cache management
        """
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.device = device
        self.eps = eps
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.layer_idx = layer_idx

        # Projections (plain linear, no input norm)
        self.q_proj = Linear(hidden_size, num_attention_heads * self.head_dim, device)
        self.k_proj = Linear(hidden_size, num_key_value_heads * self.head_dim, device)
        self.v_proj = Linear(hidden_size, num_key_value_heads * self.head_dim, device)
        self.o_proj = Linear(num_attention_heads * self.head_dim, hidden_size, device)

        # Sub-norm applied after attention, before o_proj
        self.attn_sub_norm = RMSNorm(hidden_size, device, eps)

        # Precompute RoPE frequencies (stored as numpy for flexible slicing)
        self.cos_np, self.sin_np = precompute_freqs_cis(
            self.head_dim, max_position_embeddings, rope_theta
        )
        self.max_position_embeddings = max_position_embeddings

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
        position_ids: NDArray[np.integer] | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[ttnn.Tensor, Optional[KVCache]]:
        """
        Forward pass with optional KV-Cache support.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Position indices for RoPE (batch, seq_len)
            past_key_value: Optional KV-Cache from previous forward passes
            use_cache: Whether to return updated KV-Cache

        Returns:
            Tuple of (output tensor, updated KV-Cache if use_cache else None)
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Convert to ROW_MAJOR for reshape/permute operations
        query = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
        key = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)

        # Reshape to 4D: (batch, seq, num_heads, head_dim)
        query = ttnn.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = ttnn.reshape(key, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        value = ttnn.reshape(value, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        # Transpose to (batch, num_heads, seq, head_dim)
        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.permute(value, (0, 2, 1, 3))

        # Convert back to TILE_LAYOUT for compute
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        # Determine position IDs for RoPE
        if position_ids is None:
            if past_key_value is not None and past_key_value.seq_len_cached > 0:
                # For cached generation, positions start after cached sequence
                start_pos = past_key_value.seq_len_cached
                position_ids = np.arange(start_pos, start_pos + seq_len, dtype=np.int64)
            else:
                position_ids = np.arange(seq_len, dtype=np.int64)

        # Get RoPE embeddings for current positions
        cos = self.cos_np[:, :, position_ids, :]
        sin = self.sin_np[:, :, position_ids, :]

        # Convert to TT-NN tensors
        cos_ttnn = numpy_to_ttnn(cos.astype(np.float32), self.device)
        sin_ttnn = numpy_to_ttnn(sin.astype(np.float32), self.device)

        # Apply RoPE to query and key
        query = self._apply_rope(query, cos_ttnn, sin_ttnn)
        key = self._apply_rope(key, cos_ttnn, sin_ttnn)

        # Update KV-Cache
        updated_cache = None
        if use_cache:
            if past_key_value is None:
                past_key_value = KVCache()
            key, value = past_key_value.update(key, value, self.device)
            updated_cache = past_key_value

        # Get full sequence length (including cached)
        kv_seq_len = key.shape[2]

        # Expand KV heads for GQA if needed
        if self.num_kv_groups > 1:
            key = ttnn.repeat_interleave(key, self.num_kv_groups, dim=1)
            value = ttnn.repeat_interleave(value, self.num_kv_groups, dim=1)

        # Scaled dot product attention
        # Note: is_causal should only be True for first forward (no cache)
        is_causal = past_key_value is None or past_key_value.seq_len_cached == seq_len

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=is_causal,
            scale=self.scale,
        )

        # Convert to ROW_MAJOR for reshape/permute
        attn_output = ttnn.to_layout(attn_output, ttnn.ROW_MAJOR_LAYOUT)

        # Transpose back: (batch, num_heads, seq, head_dim) -> (batch, seq, num_heads, head_dim)
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))

        # Reshape to 3D: (batch, seq, hidden)
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.hidden_size))

        # Convert back to TILE_LAYOUT for compute
        attn_output = ttnn.to_layout(attn_output, ttnn.TILE_LAYOUT)

        # Apply sub-norm before output projection
        attn_output = self.attn_sub_norm(attn_output)

        # Output projection
        output = self.o_proj(attn_output)

        return output, updated_cache

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
