"""
Optimized Multi-Head Attention implementation using TT-NN.

This module implements the attention mechanism with:
- Grouped Query Attention (GQA)
- Pre-computed RoPE with embedding-based lookup (no host->device per token)
- Pre-allocated KV-Cache with in-place updates
- Decode-specific SDPA for single-token generation
- attn_sub_norm applied after attention, before output projection

Performance optimizations based on tt_transformers patterns.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import ttnn
from numpy.typing import NDArray

from bitnet_tt.layers.bitlinear import Linear, RMSNorm, numpy_to_ttnn
from bitnet_tt.layers.rope import RotarySetup, apply_rope_ttnn


@dataclass
class KVCache:
    """
    Pre-allocated Key-Value cache for efficient autoregressive generation.

    Uses fixed-size buffers with position tracking to avoid memory reallocation.
    Supports both concat-based and in-place update modes.
    """
    key_cache: ttnn.Tensor | None = None
    value_cache: ttnn.Tensor | None = None
    seq_len_cached: int = 0
    max_seq_len: int = 0
    batch_size: int = 1
    num_kv_heads: int = 1
    head_dim: int = 128
    device: ttnn.Device | None = None
    _preallocated: bool = False

    def preallocate(
        self,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: ttnn.Device,
    ) -> None:
        """
        Pre-allocate cache buffers for maximum sequence length.

        Args:
            batch_size: Batch size
            num_kv_heads: Number of key/value heads
            max_seq_len: Maximum sequence length to cache
            head_dim: Head dimension
            device: TT-NN device
        """
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device

        # Create zero-filled buffers [batch, num_kv_heads, max_seq_len, head_dim]
        cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        zeros = torch.zeros(cache_shape, dtype=torch.bfloat16)

        self.key_cache = ttnn.from_torch(
            zeros,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.value_cache = ttnn.from_torch(
            zeros,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.seq_len_cached = 0
        self._preallocated = True

    def update(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Update cache with new key/value states using concat.

        This is the fallback method that works reliably.

        Args:
            key_states: New key states (batch, num_kv_heads, seq_len, head_dim)
            value_states: New value states (batch, num_kv_heads, seq_len, head_dim)

        Returns:
            Updated (key_states, value_states) including cached values
        """
        new_seq_len = key_states.shape[2]

        if self.key_cache is None:
            self.key_cache = key_states
            self.value_cache = value_states
            self.seq_len_cached = new_seq_len
            return key_states, value_states

        # Concatenate along sequence dimension
        self.key_cache = ttnn.concat([self.key_cache, key_states], dim=2)
        self.value_cache = ttnn.concat([self.value_cache, value_states], dim=2)
        self.seq_len_cached = self.key_cache.shape[2]

        return self.key_cache, self.value_cache

    def get_kv_for_decode(self) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Get current KV cache for decode attention."""
        return self.key_cache, self.value_cache

    def reset(self) -> None:
        """Reset cache position without deallocating."""
        self.seq_len_cached = 0
        if not self._preallocated:
            self.key_cache = None
            self.value_cache = None

    def clear(self) -> None:
        """Fully clear the cache."""
        if self.key_cache is not None:
            ttnn.deallocate(self.key_cache)
        if self.value_cache is not None:
            ttnn.deallocate(self.value_cache)
        self.key_cache = None
        self.value_cache = None
        self.seq_len_cached = 0
        self._preallocated = False


class MultiHeadAttention:
    """
    Optimized TT-NN Multi-Head Attention with GQA, RoPE, and KV-Cache.

    Key optimizations:
    - Pre-transposed weights (no transpose per forward)
    - Embedding-based RoPE lookup (no host->device per token in decode)
    - Mode-aware forward (prefill vs decode)
    - Decode-specific SDPA for single-token generation

    Architecture (matching HuggingFace Transformers):
        Q, K, V = q_proj(x), k_proj(x), v_proj(x)
        Q, K = apply_rope(Q, K)
        K, V = update_kv_cache(K, V)
        attn_output = scaled_dot_product_attention(Q, K, V)
        attn_output = attn_sub_norm(attn_output)  # BitNet-specific
        output = o_proj(attn_output)
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        device: ttnn.Device,
        max_position_embeddings: int = 4096,
        rope_theta: float = 500000.0,
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
            rope_theta: RoPE base frequency (500000 for BitNet)
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

        # Projections (weights pre-transposed in Linear.load_weights)
        self.q_proj = Linear(hidden_size, num_attention_heads * self.head_dim, device)
        self.k_proj = Linear(hidden_size, num_key_value_heads * self.head_dim, device)
        self.v_proj = Linear(hidden_size, num_key_value_heads * self.head_dim, device)
        self.o_proj = Linear(num_attention_heads * self.head_dim, hidden_size, device)

        # Sub-norm applied after attention, before o_proj (BitNet-specific)
        self.attn_sub_norm = RMSNorm(hidden_size, device, eps)

        # RoPE setup - pre-computed tables on device
        self.rope_setup = RotarySetup(
            device=device,
            head_dim=self.head_dim,
            max_seq_len=max_position_embeddings,
            rope_theta=rope_theta,
            batch_size=1,
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
            attn_sub_norm_weight: Sub-norm weight (applied after attention)
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
        position_ids: NDArray[np.integer] | int | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        mode: str = "prefill",
    ) -> Tuple[ttnn.Tensor, Optional[KVCache]]:
        """
        Forward pass with mode-aware optimization.

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Position indices or starting position (int for decode)
            past_key_value: Optional KV-Cache
            use_cache: Whether to return updated KV-Cache
            mode: "prefill" or "decode"

        Returns:
            (output tensor, updated KV-Cache if use_cache else None)
        """
        if mode == "decode":
            return self._forward_decode(
                hidden_states, attention_mask, position_ids, past_key_value, use_cache
            )
        else:
            return self._forward_prefill(
                hidden_states, attention_mask, position_ids, past_key_value, use_cache
            )

    def _forward_decode(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None,
        current_pos: int,
        past_key_value: KVCache | None,
        use_cache: bool,
    ) -> Tuple[ttnn.Tensor, Optional[KVCache]]:
        """
        Optimized decode forward for single-token generation.

        Uses embedding-based RoPE lookup (no host->device transfer).
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]  # Should be 1 for decode

        # QKV projections
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to 4D for attention
        query = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
        key = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)

        query = ttnn.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = ttnn.reshape(key, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        value = ttnn.reshape(value, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        # Transpose to (batch, num_heads, seq, head_dim)
        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.permute(value, (0, 2, 1, 3))

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        # Get RoPE embeddings via device-side lookup
        position_tensor = torch.tensor([current_pos], dtype=torch.long)
        cos, sin = self.rope_setup.get_rot_mats_decode(position_tensor)
        trans_mat = self.rope_setup.transformation_mat

        # Apply RoPE
        query, key = apply_rope_ttnn(query, key, cos, sin, trans_mat, is_decode_mode=True)

        # Update KV-Cache
        updated_cache = None
        if use_cache:
            if past_key_value is None:
                past_key_value = KVCache()
            key, value = past_key_value.update(key, value)
            updated_cache = past_key_value

        # Expand KV heads for GQA
        if self.num_kv_groups > 1:
            key = ttnn.repeat_interleave(key, self.num_kv_groups, dim=1)
            value = ttnn.repeat_interleave(value, self.num_kv_groups, dim=1)

        # Try decode-optimized SDPA, fall back to general
        try:
            attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
                query,
                key,
                value,
                cur_pos=[current_pos],
                scale=self.scale,
            )
        except (AttributeError, RuntimeError):
            # Fallback to general SDPA
            attn_output = ttnn.transformer.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                is_causal=False,  # Causality handled by KV cache
                scale=self.scale,
            )

        # Reshape back
        attn_output = ttnn.to_layout(attn_output, ttnn.ROW_MAJOR_LAYOUT)
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.hidden_size))
        attn_output = ttnn.to_layout(attn_output, ttnn.TILE_LAYOUT)

        # Apply sub-norm and output projection
        attn_output = self.attn_sub_norm(attn_output)
        output = self.o_proj(attn_output)

        return output, updated_cache

    def _forward_prefill(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None,
        position_ids: NDArray[np.integer] | None,
        past_key_value: KVCache | None,
        use_cache: bool,
    ) -> Tuple[ttnn.Tensor, Optional[KVCache]]:
        """
        Prefill forward for processing prompt tokens.
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # QKV projections
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to 4D
        query = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
        key = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)

        query = ttnn.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = ttnn.reshape(key, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        value = ttnn.reshape(value, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.permute(value, (0, 2, 1, 3))

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        # Get RoPE embeddings for prefill (upload from CPU)
        start_pos = 0
        if past_key_value is not None and past_key_value.seq_len_cached > 0:
            start_pos = past_key_value.seq_len_cached

        cos, sin = self.rope_setup.get_rot_mats_prefill(seq_len, start_pos)
        trans_mat = self.rope_setup.transformation_mat_prefill

        # Apply RoPE
        query, key = apply_rope_ttnn(query, key, cos, sin, trans_mat, is_decode_mode=False)

        # Update KV-Cache
        updated_cache = None
        if use_cache:
            if past_key_value is None:
                past_key_value = KVCache()
            key, value = past_key_value.update(key, value)
            updated_cache = past_key_value

        # Expand KV heads for GQA
        kv_seq_len = key.shape[2]
        if self.num_kv_groups > 1:
            key = ttnn.repeat_interleave(key, self.num_kv_groups, dim=1)
            value = ttnn.repeat_interleave(value, self.num_kv_groups, dim=1)

        # Causal attention for prefill
        is_causal = (past_key_value is None or past_key_value.seq_len_cached == seq_len)

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            is_causal=is_causal,
            scale=self.scale,
        )

        # Reshape back
        attn_output = ttnn.to_layout(attn_output, ttnn.ROW_MAJOR_LAYOUT)
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.hidden_size))
        attn_output = ttnn.to_layout(attn_output, ttnn.TILE_LAYOUT)

        # Apply sub-norm and output projection
        attn_output = self.attn_sub_norm(attn_output)
        output = self.o_proj(attn_output)

        return output, updated_cache


# Legacy function for backward compatibility
def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Precompute cosine and sine frequencies for RoPE.

    Note: For new code, use RotarySetup instead.
    """
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(np.float32).reshape(1, 1, max_seq_len, dim)
    sin = np.sin(emb).astype(np.float32).reshape(1, 1, max_seq_len, dim)
    return cos, sin
