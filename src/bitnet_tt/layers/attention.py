"""
Optimized Multi-Head Attention implementation using TT-NN.

This module implements the attention mechanism with:
- Grouped Query Attention (GQA)
- Native TT-NN RoPE with pre-uploaded cos/sin cache
- KV-Cache with in-place update (decode) / concat (prefill)
- Decode-specific SDPA for single-token generation
- attn_sub_norm applied after attention, before output projection

Key optimizations based on MEMO.md:
- ttnn.experimental.rotary_embedding with token_index for decode
- ttnn.kv_cache.update_cache_for_token_() for in-place KV update
- ttnn.transformer.scaled_dot_product_attention_decode for decode
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import ttnn
from numpy.typing import NDArray

from bitnet_tt.layers.bitlinear import Linear, RMSNorm, numpy_to_ttnn


@dataclass
class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.

    Uses pre-expanded GQA cache for decode optimization.
    Cache format: [batch, num_heads, seq_len, head_dim] (already expanded!)

    Key insight: Store cache already expanded for GQA to avoid
    expanding all positions every decode step.
    """
    key_cache: ttnn.Tensor | None = None
    value_cache: ttnn.Tensor | None = None
    seq_len_cached: int = 0
    max_seq_len: int = 4096
    batch_size: int = 1
    num_kv_heads: int = 1
    num_heads: int = 1  # Full head count (for pre-expanded cache)
    head_dim: int = 128
    device: ttnn.Device | None = None
    _preallocated: bool = False
    _gqa_expanded: bool = False  # Whether cache stores expanded heads

    def preallocate(
        self,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: ttnn.Device,
    ) -> None:
        """Pre-allocate cache buffers for maximum sequence length."""
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device

        # Shape: [batch, num_kv_heads, max_seq_len, head_dim]
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

    def update_prefill(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        num_kv_groups: int = 1,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Update cache for prefill (store EXPANDED KV, return for attention).

        Key optimization: Store KV already expanded for GQA to avoid
        expanding all cached positions every decode step.

        Args:
            key_states: [batch, kv_heads, seq, head_dim]
            value_states: [batch, kv_heads, seq, head_dim]
            num_kv_groups: Number of query heads per KV head (for GQA expansion)

        Returns:
            Expanded key and value for attention
        """
        self.seq_len_cached = key_states.shape[2]
        self.num_kv_heads = key_states.shape[1]
        self.num_heads = self.num_kv_heads * num_kv_groups
        self._gqa_expanded = (num_kv_groups > 1)

        # Expand KV for GQA now (during prefill, not every decode step)
        if num_kv_groups > 1:
            key_expanded = ttnn.repeat_interleave(key_states, num_kv_groups, dim=1)
            value_expanded = ttnn.repeat_interleave(value_states, num_kv_groups, dim=1)
        else:
            key_expanded = key_states
            value_expanded = value_states

        # Store EXPANDED KV for decode (this is the key optimization!)
        self._prefill_key = key_expanded
        self._prefill_value = value_expanded

        return key_expanded, value_expanded

    def update_decode_expanded(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        num_kv_groups: int = 1,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Update pre-expanded cache for decode.

        Key optimization: Expand only the NEW token's KV (1 position),
        then concat to cache. Avoids expanding all cached positions.

        Args:
            key_states: [batch, kv_heads, 1, head_dim] - single new token
            value_states: [batch, kv_heads, 1, head_dim] - single new token
            num_kv_groups: Number of query heads per KV head

        Returns:
            Updated (expanded) key and value caches
        """
        # Expand only the NEW token's KV (cheap - just 1 position!)
        if num_kv_groups > 1:
            key_expanded = ttnn.repeat_interleave(key_states, num_kv_groups, dim=1)
            value_expanded = ttnn.repeat_interleave(value_states, num_kv_groups, dim=1)
        else:
            key_expanded = key_states
            value_expanded = value_states

        # Concat expanded KV to expanded cache
        if self.key_cache is None:
            self.key_cache = key_expanded
            self.value_cache = value_expanded
        else:
            new_key = ttnn.concat([self.key_cache, key_expanded], dim=2)
            new_value = ttnn.concat([self.value_cache, value_expanded], dim=2)
            ttnn.deallocate(self.key_cache)
            ttnn.deallocate(self.value_cache)
            self.key_cache = new_key
            self.value_cache = new_value

        self.seq_len_cached = self.key_cache.shape[2]
        return self.key_cache, self.value_cache

    def update_decode_sharded(
        self,
        key_states_1bkd: ttnn.Tensor,
        value_states_1bkd: ttnn.Tensor,
        current_pos: int,
    ) -> None:
        """
        Update pre-allocated cache using paged_update_cache with sharded input.

        Args:
            key_states_1bkd: K in [1, batch, kv_heads, head_dim] format
            value_states_1bkd: V in [1, batch, kv_heads, head_dim] format
            current_pos: Current sequence position
        """
        if not self._preallocated:
            raise RuntimeError("Cache not preallocated")

        # Create position tensor
        cur_pos_tensor = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        # Create sharded memory config for 1BKD input
        # Shape: [1, batch, kv_heads, head_dim] but need 2D shard spec
        # Use HEIGHT_SHARDED with appropriate core grid
        shard_config = ttnn.create_sharded_memory_config(
            shape=(32, self.head_dim),  # TILE_SIZE x head_dim
            core_grid=ttnn.CoreGrid(y=1, x=1),  # Single core for small tensor
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Convert to sharded format
        key_sharded = ttnn.interleaved_to_sharded(key_states_1bkd, shard_config)
        value_sharded = ttnn.interleaved_to_sharded(value_states_1bkd, shard_config)

        # Update cache in-place
        ttnn.experimental.paged_update_cache(
            self.key_cache, key_sharded,
            update_idxs_tensor=cur_pos_tensor
        )
        ttnn.experimental.paged_update_cache(
            self.value_cache, value_sharded,
            update_idxs_tensor=cur_pos_tensor
        )

        ttnn.deallocate(cur_pos_tensor)
        ttnn.deallocate(key_sharded)
        ttnn.deallocate(value_sharded)
        self.seq_len_cached = current_pos + 1

    def update_decode_concat(
        self,
        key_states_1bkd: ttnn.Tensor,
        value_states_1bkd: ttnn.Tensor,
        current_pos: int,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Update cache using concat (fallback when sharding fails).
        """
        # Transpose 1BKD to BKSD for concat
        key_bksd = ttnn.to_layout(key_states_1bkd, ttnn.ROW_MAJOR_LAYOUT)
        key_bksd = ttnn.permute(key_bksd, (1, 2, 0, 3))
        key_bksd = ttnn.to_layout(key_bksd, ttnn.TILE_LAYOUT)

        value_bksd = ttnn.to_layout(value_states_1bkd, ttnn.ROW_MAJOR_LAYOUT)
        value_bksd = ttnn.permute(value_bksd, (1, 2, 0, 3))
        value_bksd = ttnn.to_layout(value_bksd, ttnn.TILE_LAYOUT)

        # Concat-based cache update
        if self.key_cache is None:
            self.key_cache = key_bksd
            self.value_cache = value_bksd
        else:
            new_key = ttnn.concat([self.key_cache, key_bksd], dim=2)
            new_value = ttnn.concat([self.value_cache, value_bksd], dim=2)
            ttnn.deallocate(self.key_cache)
            ttnn.deallocate(self.value_cache)
            self.key_cache = new_key
            self.value_cache = new_value

        self.seq_len_cached = current_pos + 1
        return self.key_cache, self.value_cache

    def reset(self) -> None:
        """Reset cache position (keep pre-allocated buffers)."""
        self.seq_len_cached = 0
        if hasattr(self, '_prefill_key'):
            self._prefill_key = None
            self._prefill_value = None

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


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Precompute cosine and sine frequencies for RoPE."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)
    emb = np.concatenate([freqs, freqs], axis=-1)

    cos = np.cos(emb).astype(np.float32).reshape(1, 1, max_seq_len, dim)
    sin = np.sin(emb).astype(np.float32).reshape(1, 1, max_seq_len, dim)

    return cos, sin


class MultiHeadAttention:
    """
    Optimized TT-NN Multi-Head Attention with GQA, RoPE, and KV-Cache.

    Key optimizations:
    - Pre-transposed weights (no transpose per forward)
    - Pre-uploaded RoPE cos/sin cache
    - Native TT-NN ops for decode mode
    - Mode-aware forward (prefill vs decode)
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
        """Initialize attention layer."""
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.device = device
        self.eps = eps
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.layer_idx = layer_idx
        self.max_position_embeddings = max_position_embeddings

        # Projections (weights pre-transposed in Linear.load_weights)
        self.q_proj = Linear(hidden_size, num_attention_heads * self.head_dim, device)
        self.k_proj = Linear(hidden_size, num_key_value_heads * self.head_dim, device)
        self.v_proj = Linear(hidden_size, num_key_value_heads * self.head_dim, device)
        self.o_proj = Linear(num_attention_heads * self.head_dim, hidden_size, device)

        # Sub-norm applied after attention, before o_proj (BitNet-specific)
        self.attn_sub_norm = RMSNorm(hidden_size, device, eps)

        # Precompute and upload RoPE frequencies to device
        cos_np, sin_np = precompute_freqs_cis(
            self.head_dim, max_position_embeddings, rope_theta
        )

        # Upload as 2D [max_seq_len, head_dim] for embedding lookup
        cos_2d = cos_np.squeeze((0, 1))  # [max_seq_len, head_dim]
        sin_2d = sin_np.squeeze((0, 1))  # [max_seq_len, head_dim]

        self.cos_cache_2d = ttnn.from_torch(
            torch.from_numpy(cos_2d),
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,  # ROW_MAJOR for embedding weight
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.sin_cache_2d = ttnn.from_torch(
            torch.from_numpy(sin_2d),
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,  # ROW_MAJOR for embedding weight
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def load_weights(
        self,
        q_weight: NDArray[np.floating],
        k_weight: NDArray[np.floating],
        v_weight: NDArray[np.floating],
        o_weight: NDArray[np.floating],
        attn_sub_norm_weight: NDArray[np.floating] | None = None,
    ) -> None:
        """Load all projection weights."""
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
            position_ids: Position indices (int for decode mode)
            past_key_value: Optional KV-Cache
            use_cache: Whether to return updated KV-Cache
            mode: "prefill" or "decode"
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # QKV projections
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Determine current position
        if mode == "decode" and isinstance(position_ids, int):
            current_pos = position_ids
        elif past_key_value is not None and past_key_value.seq_len_cached > 0:
            current_pos = past_key_value.seq_len_cached
        else:
            current_pos = 0

        # Use BKSD format for both prefill and decode
        # Note: 1BKD format + paged ops requires sharding, SDPA decode requires seq%32==0
        # So we use simple concat-based cache with general SDPA for all cases
        return self._forward_simple(
            query, key, value, attention_mask, past_key_value,
            current_pos, seq_len, use_cache, batch_size, mode
        )

    def _forward_simple(
        self,
        query: ttnn.Tensor,
        key: ttnn.Tensor,
        value: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None,
        past_key_value: KVCache | None,
        current_pos: int,
        seq_len: int,
        use_cache: bool,
        batch_size: int,
        mode: str,
    ) -> Tuple[ttnn.Tensor, Optional[KVCache]]:
        """
        Optimized forward using pre-expanded GQA cache.

        Key optimization: Cache stores already-expanded KV heads.
        - Prefill: Expand full sequence, store in cache
        - Decode: Expand only 1 new token, concat to expanded cache

        This avoids expanding all cached positions every decode step.
        """
        # Reshape to BKSD: [batch, heads, seq, head_dim]
        # Optimization: reshape in ROW_MAJOR, then transpose in TILE_LAYOUT
        query = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
        key = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)

        query = ttnn.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = ttnn.reshape(key, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        value = ttnn.reshape(value, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        # Convert to TILE before transpose (transpose may be faster in TILE_LAYOUT)
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        # Use transpose(1, 2) instead of permute - may be more efficient
        query = ttnn.transpose(query, 1, 2)
        key = ttnn.transpose(key, 1, 2)
        value = ttnn.transpose(value, 1, 2)

        # Apply RoPE
        query, key = self._apply_rope_manual(query, key, current_pos, seq_len)

        # Update KV-Cache with pre-expanded GQA optimization
        updated_cache = None
        if use_cache:
            if past_key_value is None:
                past_key_value = KVCache()

            if mode == "prefill":
                # Prefill: expand and store in cache
                key_expanded, value_expanded = past_key_value.update_prefill(
                    key, value, self.num_kv_groups
                )
            else:
                # Decode: expand only new token, concat to expanded cache
                key_expanded, value_expanded = past_key_value.update_decode_expanded(
                    key, value, self.num_kv_groups
                )
            updated_cache = past_key_value
        else:
            # No cache: expand inline
            if self.num_kv_groups > 1:
                key_expanded = ttnn.repeat_interleave(key, self.num_kv_groups, dim=1)
                value_expanded = ttnn.repeat_interleave(value, self.num_kv_groups, dim=1)
            else:
                key_expanded = key
                value_expanded = value

        # SDPA (cache is already expanded, no GQA expansion needed!)
        is_causal = (mode == "prefill")
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query, key_expanded, value_expanded,
            attn_mask=attention_mask,
            is_causal=is_causal,
            scale=self.scale,
        )

        # Reshape output: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        # Optimization: transpose in TILE, reshape in ROW_MAJOR
        attn_output = ttnn.transpose(attn_output, 1, 2)  # [b, s, h, d]
        attn_output = ttnn.to_layout(attn_output, ttnn.ROW_MAJOR_LAYOUT)
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.hidden_size))
        attn_output = ttnn.to_layout(attn_output, ttnn.TILE_LAYOUT)

        # Apply sub-norm and output projection
        attn_output = self.attn_sub_norm(attn_output)
        output = self.o_proj(attn_output)

        return output, updated_cache

    def _forward_prefill(
        self,
        query: ttnn.Tensor,
        key: ttnn.Tensor,
        value: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None,
        past_key_value: KVCache | None,
        current_pos: int,
        seq_len: int,
        use_cache: bool,
        batch_size: int,
    ) -> Tuple[ttnn.Tensor, Optional[KVCache]]:
        """Prefill forward with BKSD tensor format."""
        # Reshape to 4D BKSD: [batch, heads, seq, head_dim]
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

        # Apply RoPE for prefill
        query, key = self._apply_rope_prefill(query, key, current_pos, seq_len)

        # Update KV-Cache for prefill
        updated_cache = None
        if use_cache:
            if past_key_value is None:
                past_key_value = KVCache()
            key_full, value_full = past_key_value.update_prefill(key, value)
            updated_cache = past_key_value
        else:
            key_full, value_full = key, value

        # Expand KV heads for GQA
        if self.num_kv_groups > 1:
            key_expanded = ttnn.repeat_interleave(key_full, self.num_kv_groups, dim=1)
            value_expanded = ttnn.repeat_interleave(value_full, self.num_kv_groups, dim=1)
        else:
            key_expanded = key_full
            value_expanded = value_full

        # SDPA for prefill (causal attention)
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query, key_expanded, value_expanded,
            attn_mask=attention_mask,
            is_causal=True,
            scale=self.scale,
        )

        # Reshape output: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        attn_output = ttnn.to_layout(attn_output, ttnn.ROW_MAJOR_LAYOUT)
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.hidden_size))
        attn_output = ttnn.to_layout(attn_output, ttnn.TILE_LAYOUT)

        # Apply sub-norm and output projection
        attn_output = self.attn_sub_norm(attn_output)
        output = self.o_proj(attn_output)

        return output, updated_cache

    def _apply_rope_decode_1bkd(
        self,
        query: ttnn.Tensor,
        key: ttnn.Tensor,
        token_idx: int,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Apply RoPE for decode mode with 1BKD format [1, batch, heads, head_dim]."""
        # Use embedding lookup with single position
        pos_tensor = ttnn.from_torch(
            torch.tensor([[token_idx]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        cos_ttnn = ttnn.embedding(pos_tensor, self.cos_cache_2d, layout=ttnn.TILE_LAYOUT)
        sin_ttnn = ttnn.embedding(pos_tensor, self.sin_cache_2d, layout=ttnn.TILE_LAYOUT)
        cos_ttnn = ttnn.reshape(cos_ttnn, (1, 1, 1, self.head_dim))
        sin_ttnn = ttnn.reshape(sin_ttnn, (1, 1, 1, self.head_dim))
        ttnn.deallocate(pos_tensor)

        query_rotated = self._apply_rope_to_tensor_1bkd(query, cos_ttnn, sin_ttnn, self.num_heads)
        key_rotated = self._apply_rope_to_tensor_1bkd(key, cos_ttnn, sin_ttnn, self.num_kv_heads)

        return query_rotated, key_rotated

    def _apply_rope_to_tensor_1bkd(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        num_heads: int,
    ) -> ttnn.Tensor:
        """Apply RoPE to 1BKD tensor [1, batch, heads, head_dim].

        Uses TTNN broadcasting in multiply (no explicit repeat needed).
        cos/sin are [1, 1, 1, head_dim], x is [1, batch, heads, head_dim].
        """
        half_dim = self.head_dim // 2
        x1 = x[:, :, :, :half_dim]
        x2 = x[:, :, :, half_dim:]
        x_rotated = ttnn.concat([ttnn.neg(x2), x1], dim=-1)

        # Use broadcasting in multiply instead of explicit repeat
        x_cos = ttnn.multiply(x, cos)
        x_rot_sin = ttnn.multiply(x_rotated, sin)
        return ttnn.add(x_cos, x_rot_sin)

    def _apply_rope_prefill(
        self,
        query: ttnn.Tensor,
        key: ttnn.Tensor,
        start_pos: int,
        seq_len: int,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Apply RoPE for prefill mode."""
        # Native RoPE op has shape requirements that don't match our tensors
        # Use manual implementation for now
        return self._apply_rope_manual(query, key, start_pos, seq_len)

    def _apply_rope_manual(
        self,
        query: ttnn.Tensor,
        key: ttnn.Tensor,
        start_pos: int,
        seq_len: int,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Manual RoPE implementation using device-cached cos/sin tables."""
        if seq_len == 1:
            # Decode mode: use embedding lookup with single position
            pos_tensor = ttnn.from_torch(
                torch.tensor([[start_pos]], dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
            cos_ttnn = ttnn.embedding(pos_tensor, self.cos_cache_2d, layout=ttnn.TILE_LAYOUT)
            sin_ttnn = ttnn.embedding(pos_tensor, self.sin_cache_2d, layout=ttnn.TILE_LAYOUT)
            cos_ttnn = ttnn.reshape(cos_ttnn, (1, 1, 1, self.head_dim))
            sin_ttnn = ttnn.reshape(sin_ttnn, (1, 1, 1, self.head_dim))
            ttnn.deallocate(pos_tensor)
        else:
            # Prefill mode: use embedding lookup for range
            pos_indices = torch.arange(start_pos, start_pos + seq_len, dtype=torch.int32)
            pos_tensor = ttnn.from_torch(
                pos_indices.unsqueeze(0),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
            cos_ttnn = ttnn.embedding(pos_tensor, self.cos_cache_2d, layout=ttnn.TILE_LAYOUT)
            sin_ttnn = ttnn.embedding(pos_tensor, self.sin_cache_2d, layout=ttnn.TILE_LAYOUT)
            cos_ttnn = ttnn.reshape(cos_ttnn, (1, 1, seq_len, self.head_dim))
            sin_ttnn = ttnn.reshape(sin_ttnn, (1, 1, seq_len, self.head_dim))
            ttnn.deallocate(pos_tensor)

        query_rotated = self._apply_rope_to_tensor(query, cos_ttnn, sin_ttnn)
        key_rotated = self._apply_rope_to_tensor(key, cos_ttnn, sin_ttnn)

        return query_rotated, key_rotated

    def _apply_rope_to_tensor(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Apply RoPE to a single tensor.

        Uses TTNN broadcasting in multiply (no explicit repeat needed).
        cos/sin are [1, 1, seq, head_dim], x is [batch, heads, seq, head_dim].
        """
        half_dim = self.head_dim // 2
        x1 = x[:, :, :, :half_dim]
        x2 = x[:, :, :, half_dim:]
        x_rotated = ttnn.concat([ttnn.neg(x2), x1], dim=-1)

        # Use broadcasting in multiply instead of explicit repeat
        # This eliminates 4 repeat ops per layer (120 total for 30 layers)
        x_cos = ttnn.multiply(x, cos)
        x_rot_sin = ttnn.multiply(x_rotated, sin)
        return ttnn.add(x_cos, x_rot_sin)

    def _forward_decode_optimized(
        self,
        xqkv_fused: ttnn.Tensor,
        past_key_value: KVCache,
        current_pos: int,
        rot_mats: list,
        transformation_mat: ttnn.Tensor,
        batch_size: int,
    ) -> Tuple[ttnn.Tensor, KVCache]:
        """
        Optimized decode forward using tt_transformers patterns.

        Key optimizations (from tt_transformers):
        1. nlp_create_qkv_heads_decode - single fused op for QKV head creation
        2. rotary_embedding_llama - fused RoPE with transformation matrix
        3. paged_update_cache - in-place cache update
        4. scaled_dot_product_attention_decode - decode-optimized SDPA
        5. nlp_concat_heads_decode - single fused op for head concatenation

        Args:
            xqkv_fused: Fused QKV output [batch, 1, qkv_dim] from concatenated projection
            past_key_value: Pre-allocated KV cache
            current_pos: Current sequence position
            rot_mats: [cos, sin] rotation matrices from RotarySetup
            transformation_mat: Transformation matrix for rotary_embedding_llama
            batch_size: Batch size

        Returns:
            (output, updated_cache)
        """
        # 1. Create QKV heads using optimized fused op
        # Input: [batch, 1, (n_heads + 2*n_kv_heads) * head_dim]
        # Output: q [1, batch, n_heads, head_dim], k [1, batch, n_kv_heads, head_dim], v [1, batch, n_kv_heads, head_dim]
        q_heads_1bqd, k_heads_1bkd, v_heads_1bkd = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv_fused)

        # 2. Apply RoPE using fused rotary_embedding_llama
        q_heads_1bqd = ttnn.experimental.rotary_embedding_llama(
            q_heads_1bqd,
            rot_mats[0],  # cos
            rot_mats[1],  # sin
            transformation_mat,
            is_decode_mode=True,
        )
        k_heads_1bkd = ttnn.experimental.rotary_embedding_llama(
            k_heads_1bkd,
            rot_mats[0],
            rot_mats[1],
            transformation_mat,
            is_decode_mode=True,
        )

        # 3. Update KV cache in-place using paged_update_cache
        # Create position tensor for cache update
        cur_pos_tensor = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        ttnn.experimental.paged_update_cache(
            past_key_value.key_cache,
            k_heads_1bkd,
            update_idxs_tensor=cur_pos_tensor,
        )
        ttnn.experimental.paged_update_cache(
            past_key_value.value_cache,
            v_heads_1bkd,
            update_idxs_tensor=cur_pos_tensor,
        )
        ttnn.deallocate(k_heads_1bkd)
        ttnn.deallocate(v_heads_1bkd)

        past_key_value.seq_len_cached = current_pos + 1

        # 4. Scaled dot-product attention for decode
        attn_output_1g4d = ttnn.transformer.scaled_dot_product_attention_decode(
            q_heads_1bqd,
            past_key_value.key_cache,
            past_key_value.value_cache,
            cur_pos_tensor=cur_pos_tensor,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_heads_1bqd)
        ttnn.deallocate(cur_pos_tensor)

        # 5. Concat heads using optimized fused op
        attn_output = ttnn.experimental.nlp_concat_heads_decode(
            attn_output_1g4d,
            num_heads=self.num_heads,
        )
        ttnn.deallocate(attn_output_1g4d)

        # Apply sub-norm and output projection
        attn_output = self.attn_sub_norm(attn_output)
        output = self.o_proj(attn_output)

        return output, past_key_value
