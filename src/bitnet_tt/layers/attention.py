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

from bitnet_tt.layers.bitlinear import Linear, RMSNorm


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
        num_heads: int | None = None,
    ) -> None:
        """
        Pre-allocate cache buffers for maximum sequence length.

        Args:
            batch_size: Batch size
            num_kv_heads: Number of KV heads (before GQA expansion)
            max_seq_len: Maximum sequence length
            head_dim: Head dimension
            device: TT-NN device
            num_heads: Full head count for GQA expansion. If provided,
                       cache is pre-expanded to [batch, num_heads, max_seq, head_dim]
                       to avoid per-token GQA expansion during decode.
        """
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device

        # Determine cache head count - use GQA-expanded form if num_heads provided
        if num_heads is not None:
            # Pre-expanded cache for GQA (avoids expansion during decode)
            cache_heads = num_heads
            self.num_heads = num_heads
            self._gqa_expanded = True
        else:
            # Non-expanded cache (requires GQA expansion during decode)
            cache_heads = num_kv_heads
            self.num_heads = num_kv_heads
            self._gqa_expanded = False

        # Shape: [batch, cache_heads, max_seq_len, head_dim]
        cache_shape = (batch_size, cache_heads, max_seq_len, head_dim)
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
        Update cache for prefill and return GQA-expanded KV for attention.

        For preallocated cache: stores NON-EXPANDED KV, expands when returning.
        For concat-based cache: stores EXPANDED KV directly.

        Args:
            key_states: [batch, kv_heads, seq, head_dim]
            value_states: [batch, kv_heads, seq, head_dim]
            num_kv_groups: Number of query heads per KV head (for GQA expansion)

        Returns:
            GQA-expanded key and value for attention
        """
        seq_len = key_states.shape[2]
        self.num_kv_heads = key_states.shape[1]
        self.num_heads = self.num_kv_heads * num_kv_groups
        self._gqa_expanded = num_kv_groups > 1

        # If preallocated, store NON-EXPANDED KV, expand only when returning
        if self._preallocated and self.key_cache is not None:
            # Use fill_cache with NON-EXPANDED key/value (5 heads)
            for batch_idx in range(key_states.shape[0]):
                k_batch = key_states[batch_idx : batch_idx + 1, :, :, :]
                v_batch = value_states[batch_idx : batch_idx + 1, :, :, :]
                self.key_cache = ttnn.fill_cache(self.key_cache, k_batch, batch_idx)
                self.value_cache = ttnn.fill_cache(self.value_cache, v_batch, batch_idx)

            self.seq_len_cached = seq_len

            # Return GQA-expanded view for attention
            key_for_attn = self.key_cache[:, :, :seq_len, :]
            value_for_attn = self.value_cache[:, :, :seq_len, :]
            if num_kv_groups > 1:
                key_for_attn = ttnn.repeat_interleave(key_for_attn, num_kv_groups, dim=1)
                value_for_attn = ttnn.repeat_interleave(value_for_attn, num_kv_groups, dim=1)
            return key_for_attn, value_for_attn
        else:
            # Not preallocated: expand and store for concat-based decode
            if num_kv_groups > 1:
                key_expanded = ttnn.repeat_interleave(key_states, num_kv_groups, dim=1)
                value_expanded = ttnn.repeat_interleave(value_states, num_kv_groups, dim=1)
            else:
                key_expanded = key_states
                value_expanded = value_states

            self._prefill_key = key_expanded
            self._prefill_value = value_expanded
            self.key_cache = key_expanded
            self.value_cache = value_expanded
            self.seq_len_cached = seq_len
            return key_expanded, value_expanded

    def update_decode_expanded(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        current_pos: int | None = None,
        num_kv_groups: int = 1,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Update pre-expanded cache for decode using concat.

        Key optimization: Expand only the NEW token's KV (1 position).

        Note: In-place update via paged_update_cache requires HEIGHT_SHARDED input
        which is not yet properly configured. Using concat-based approach for now.
        This means Trace capture is not compatible with the current implementation.

        Args:
            key_states: [batch, kv_heads, 1, head_dim] - single new token
            value_states: [batch, kv_heads, 1, head_dim] - single new token
            current_pos: Current sequence position (unused, kept for API compatibility)
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

        # Concat-based update
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

    def update_decode_inplace(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        current_pos: int,
        num_kv_groups: int = 1,
        current_pos_tensor: ttnn.Tensor | None = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Update pre-allocated cache using paged_update_cache for in-place updates.

        This method enables Metal Trace by using tensor position (not frozen int).
        Cache stores NON-EXPANDED KV heads. GQA expansion happens when returning.

        Args:
            key_states: [batch, kv_heads, 1, head_dim] - single new token (NON-EXPANDED)
            value_states: [batch, kv_heads, 1, head_dim] - single new token (NON-EXPANDED)
            current_pos: Current sequence position (cache_idx) - for seq_len_cached tracking
            num_kv_groups: Number of query heads per KV head (for GQA expansion when returning)
            current_pos_tensor: Device tensor with position (for Trace compatibility)

        Returns:
            GQA-expanded key and value for attention (sliced from cache)
        """
        if not self._preallocated:
            raise RuntimeError("Cache must be preallocated for in-place update")

        # Convert BKSD [batch, kv_heads, 1, head_dim] to 1BKD [1, batch, kv_heads, head_dim]
        # for paged_update_cache - format matches TT transformer pattern
        key_1bkd = ttnn.to_layout(key_states, ttnn.ROW_MAJOR_LAYOUT)
        key_1bkd = ttnn.permute(key_1bkd, (2, 0, 1, 3))  # [1, batch, kv_heads, head_dim]
        key_1bkd = ttnn.to_layout(key_1bkd, ttnn.TILE_LAYOUT)

        value_1bkd = ttnn.to_layout(value_states, ttnn.ROW_MAJOR_LAYOUT)
        value_1bkd = ttnn.permute(value_1bkd, (2, 0, 1, 3))  # [1, batch, kv_heads, head_dim]
        value_1bkd = ttnn.to_layout(value_1bkd, ttnn.TILE_LAYOUT)

        # Create position tensor if not provided
        pos_tensor_created_locally = current_pos_tensor is None
        if current_pos_tensor is None:
            current_pos_tensor = ttnn.from_torch(
                torch.tensor([current_pos], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )

        # Convert to sharded format for paged_update_cache
        # TT pattern: L1_HEIGHT_SHARDED with [TILE_SIZE, head_dim] shard shape
        shard_config = ttnn.create_sharded_memory_config(
            shape=(32, self.head_dim),  # TILE_SIZE x head_dim
            core_grid=ttnn.CoreGrid(y=1, x=1),  # Single core for batch=1
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        key_sharded = ttnn.interleaved_to_sharded(key_1bkd, shard_config)
        value_sharded = ttnn.interleaved_to_sharded(value_1bkd, shard_config)

        # In-place update using paged_update_cache (tensor position for Trace)
        ttnn.experimental.paged_update_cache(
            self.key_cache, key_sharded, update_idxs_tensor=current_pos_tensor
        )
        ttnn.experimental.paged_update_cache(
            self.value_cache, value_sharded, update_idxs_tensor=current_pos_tensor
        )

        # Cleanup temporary tensors
        ttnn.deallocate(key_1bkd)
        ttnn.deallocate(value_1bkd)
        ttnn.deallocate(key_sharded)
        ttnn.deallocate(value_sharded)
        if pos_tensor_created_locally:
            ttnn.deallocate(current_pos_tensor)

        self.seq_len_cached = current_pos + 1

        # Slice cache up to current position (required for standard SDPA)
        # NOTE: This int-based slicing is NOT Trace-safe. For Trace, need 1BKD format + sdpa_decode
        key_for_attn = self.key_cache[:, :, : self.seq_len_cached, :]
        value_for_attn = self.value_cache[:, :, : self.seq_len_cached, :]

        # Expand for GQA when returning (same as concat-based path)
        if num_kv_groups > 1:
            key_for_attn = ttnn.repeat_interleave(key_for_attn, num_kv_groups, dim=1)
            value_for_attn = ttnn.repeat_interleave(value_for_attn, num_kv_groups, dim=1)

        return key_for_attn, value_for_attn

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
            self.key_cache, key_sharded, update_idxs_tensor=cur_pos_tensor
        )
        ttnn.experimental.paged_update_cache(
            self.value_cache, value_sharded, update_idxs_tensor=cur_pos_tensor
        )

        ttnn.deallocate(cur_pos_tensor)
        ttnn.deallocate(key_sharded)
        ttnn.deallocate(value_sharded)
        self.seq_len_cached = current_pos + 1

    def update_decode_preallocated(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        current_pos: int,
        num_kv_groups: int = 1,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Update pre-allocated GQA-expanded cache using slice write.

        This enables trace compatibility by avoiding concat (which changes memory).

        Args:
            key_states: [batch, kv_heads, 1, head_dim] - single new token
            value_states: [batch, kv_heads, 1, head_dim] - single new token
            current_pos: Current sequence position
            num_kv_groups: Number of query heads per KV head

        Returns:
            Full key and value caches up to current_pos+1
        """
        if not self._preallocated or not self._gqa_expanded:
            raise RuntimeError("Cache must be preallocated with GQA expansion")

        # Expand new token's KV for GQA
        if num_kv_groups > 1:
            key_expanded = ttnn.repeat_interleave(key_states, num_kv_groups, dim=1)
            value_expanded = ttnn.repeat_interleave(value_states, num_kv_groups, dim=1)
        else:
            key_expanded = key_states
            value_expanded = value_states

        # In-place update using ttnn slice write
        # Write new KV at position current_pos in the pre-allocated cache
        try:
            # Try slice assignment (may not be supported in all TT-NN versions)
            # Shape: cache [batch, num_heads, max_seq, head_dim]
            # Shape: expanded [batch, num_heads, 1, head_dim]
            ttnn.experimental.tensor.write_tensor_to_existing(
                key_expanded,
                self.key_cache,
                slice_start=(0, 0, current_pos, 0),
            )
            ttnn.experimental.tensor.write_tensor_to_existing(
                value_expanded,
                self.value_cache,
                slice_start=(0, 0, current_pos, 0),
            )
        except (AttributeError, RuntimeError):
            # Fallback: use paged_update_cache if available
            try:
                cur_pos_tensor = ttnn.from_torch(
                    torch.tensor([current_pos], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                )
                # Reshape to 1BKD format for paged_update_cache
                # [batch, heads, 1, head_dim] -> [1, batch, heads, head_dim]
                key_1bkd = ttnn.permute(key_expanded, (2, 0, 1, 3))
                value_1bkd = ttnn.permute(value_expanded, (2, 0, 1, 3))

                ttnn.experimental.paged_update_cache(
                    self.key_cache, key_1bkd, update_idxs_tensor=cur_pos_tensor
                )
                ttnn.experimental.paged_update_cache(
                    self.value_cache, value_1bkd, update_idxs_tensor=cur_pos_tensor
                )
                ttnn.deallocate(cur_pos_tensor)
            except Exception:
                # Final fallback: concat (breaks trace but works)
                new_key = ttnn.concat([self.key_cache[:, :, :current_pos, :], key_expanded], dim=2)
                new_value = ttnn.concat(
                    [self.value_cache[:, :, :current_pos, :], value_expanded], dim=2
                )
                ttnn.deallocate(self.key_cache)
                ttnn.deallocate(self.value_cache)
                self.key_cache = new_key
                self.value_cache = new_value

        self.seq_len_cached = current_pos + 1

        # Return cache up to current position for attention
        key_for_attn = self.key_cache[:, :, : self.seq_len_cached, :]
        value_for_attn = self.value_cache[:, :, : self.seq_len_cached, :]
        return key_for_attn, value_for_attn

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
        if hasattr(self, "_prefill_key"):
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

    def copy_from_prefill(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        seq_len: int,
    ) -> None:
        """
        Copy prefill results into pre-allocated cache.

        This enables the optimized decode path after standard prefill.

        Args:
            key_states: Key tensor [batch, n_kv_heads, seq_len, head_dim]
            value_states: Value tensor [batch, n_kv_heads, seq_len, head_dim]
            seq_len: Sequence length of prefill
        """
        if not self._preallocated:
            raise RuntimeError("Cache not preallocated")

        prefill_seq_len = key_states.shape[2]
        if prefill_seq_len > self.max_seq_len:
            raise RuntimeError(
                f"Prefill seq_len {prefill_seq_len} > max_seq_len {self.max_seq_len}"
            )

        # Use ttnn.fill_cache to copy prefill KV into preallocated cache
        # fill_cache writes to cache[batch_idx, :, :seq_len, :]
        # Since we have batch=1, use batch_idx=0
        for batch_idx in range(key_states.shape[0]):
            # Slice single batch item: [1, heads, seq, head_dim]
            k_batch = key_states[batch_idx : batch_idx + 1, :, :, :]
            v_batch = value_states[batch_idx : batch_idx + 1, :, :, :]

            # fill_cache expects input shape matching cache dimensions
            # Cache: [batch, heads, max_seq, head_dim]
            # Input: [1, heads, prefill_seq, head_dim]
            self.key_cache = ttnn.fill_cache(self.key_cache, k_batch, batch_idx)
            self.value_cache = ttnn.fill_cache(self.value_cache, v_batch, batch_idx)

        self.seq_len_cached = prefill_seq_len
        # Keep _preallocated=True since we're using the preallocated buffer


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

        # Fused QKV projection for optimized decode (created when weights are loaded)
        self.qkv_fused_weight: ttnn.Tensor | None = None
        self._qkv_dim = (num_attention_heads + 2 * num_key_value_heads) * self.head_dim

        # Sub-norm applied after attention, before o_proj (BitNet-specific)
        self.attn_sub_norm = RMSNorm(hidden_size, device, eps)

        # Precompute and upload RoPE frequencies to device
        cos_np, sin_np = precompute_freqs_cis(self.head_dim, max_position_embeddings, rope_theta)

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

        # TILE layout cos/sin for rotary_embedding_llama
        # Format: [1, 1, max_seq_len, head_dim]
        self.cos_cache_tile = ttnn.from_torch(
            torch.from_numpy(cos_np.astype(np.float32)),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.sin_cache_tile = ttnn.from_torch(
            torch.from_numpy(sin_np.astype(np.float32)),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Transformation matrix for rotary_embedding_llama (decode mode)
        # TT pattern uses 32x32 matrix for single tile
        trans_mat = self._get_rot_transformation_mat()
        self.transformation_mat_decode = ttnn.from_torch(
            trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # DRAM to avoid L1 clash
        )

        # Store core grid for dynamic sharding
        self.core_grid = device.compute_with_storage_grid_size()

    def _get_rot_transformation_mat(self) -> torch.Tensor:
        """
        Create transformation matrix for rotary_embedding_llama.

        TT pattern: 32x32 matrix that performs the rotation operation.
        Based on models/tt_transformers/tt/common.py:get_rot_transformation_mat
        """
        dhead = 32  # TILE_SIZE
        rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
        rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
        rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
        return rot_emb_matrix

    def get_rot_mats(
        self, position_ids: torch.Tensor, batch_size: int = 1
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Get rotation matrices (cos, sin) for given positions.

        TT pattern: Use embedding lookup, then transpose and shard.
        Based on models/tt_transformers/tt/rope.py:RotarySetup.get_rot_mats

        Args:
            position_ids: [batch] tensor of position indices
            batch_size: Batch size for sharding config

        Returns:
            (cos, sin) tuple of sharded tensors
        """
        # Pad to nearest 32 for TILE alignment
        padded_batch = ((batch_size + 31) // 32) * 32
        if len(position_ids) < padded_batch:
            position_ids = torch.nn.functional.pad(
                position_ids, (0, padded_batch - len(position_ids)), value=0
            )

        # Create position tensor [1, padded_batch]
        rot_idxs = ttnn.from_torch(
            position_ids.reshape(1, -1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Embedding lookup: [1, batch, head_dim]
        cos = ttnn.embedding(rot_idxs, self.cos_cache_2d, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_cache_2d, layout=ttnn.TILE_LAYOUT)

        # Expand to 4D: [1, 1, batch, head_dim]
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)

        # Transpose: [1, batch, 1[32], head_dim]
        cos = ttnn.transpose(cos, 1, 2)
        sin = ttnn.transpose(sin, 1, 2)

        # Slice to actual batch if padded
        if batch_size < padded_batch:
            cos = cos[:, :batch_size, :, :]
            sin = sin[:, :batch_size, :, :]

        # Keep interleaved - rotary_embedding_llama will handle memory layout
        ttnn.deallocate(rot_idxs)
        return cos, sin

    def load_weights(
        self,
        q_weight: NDArray[np.floating],
        k_weight: NDArray[np.floating],
        v_weight: NDArray[np.floating],
        o_weight: NDArray[np.floating],
        attn_sub_norm_weight: NDArray[np.floating] | None = None,
    ) -> None:
        """Load all projection weights and create fused QKV weight."""
        self.q_proj.load_weights(q_weight)
        self.k_proj.load_weights(k_weight)
        self.v_proj.load_weights(v_weight)
        self.o_proj.load_weights(o_weight)

        if attn_sub_norm_weight is not None:
            self.attn_sub_norm.load_weights(attn_sub_norm_weight)

        # Create fused QKV weight for optimized decode
        # Concatenate Q, K, V weights: [hidden_size, qkv_dim]
        # Original weights are [out_features, in_features], transposed for matmul
        # Q: [num_heads * head_dim, hidden_size] -> transposed: [hidden_size, num_heads * head_dim]
        # K: [num_kv_heads * head_dim, hidden_size] -> transposed: [hidden_size, num_kv_heads * head_dim]
        # V: [num_kv_heads * head_dim, hidden_size] -> transposed: [hidden_size, num_kv_heads * head_dim]
        q_t = q_weight.T.astype(np.float32)  # [hidden_size, num_heads * head_dim]
        k_t = k_weight.T.astype(np.float32)  # [hidden_size, num_kv_heads * head_dim]
        v_t = v_weight.T.astype(np.float32)  # [hidden_size, num_kv_heads * head_dim]
        qkv_fused = np.concatenate([q_t, k_t, v_t], axis=1)  # [hidden_size, qkv_dim]

        self.qkv_fused_weight = ttnn.from_torch(
            torch.from_numpy(qkv_fused),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
        position_ids: NDArray[np.integer] | int | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        mode: str = "prefill",
        rot_mats: list | None = None,
        transformation_mat: ttnn.Tensor | None = None,
        current_pos_tensor: ttnn.Tensor | None = None,  # int32 for KV cache
        pos_tensor: ttnn.Tensor | None = None,  # uint32 for RoPE
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
            rot_mats: [cos, sin] rotation matrices from RotarySetup (for optimized decode)
            transformation_mat: Transformation matrix for rotary_embedding_llama (for optimized decode)
            current_pos_tensor: Optional tensor containing current position (for trace optimization)
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Determine current position
        if mode == "decode" and isinstance(position_ids, int):
            current_pos = position_ids
        elif past_key_value is not None and past_key_value.seq_len_cached > 0:
            current_pos = past_key_value.seq_len_cached
        else:
            current_pos = 0

        # Use optimized prefill path when pre-allocated cache is provided
        if (
            mode == "prefill"
            and past_key_value is not None
            and hasattr(past_key_value, "_preallocated")
            and past_key_value._preallocated
        ):
            return self._forward_prefill_with_preallocated_cache(
                hidden_states, past_key_value, batch_size, seq_len
            )

        # Use optimized decode path when rot_mats are provided and cache is pre-allocated
        if (
            mode == "decode"
            and rot_mats is not None
            and transformation_mat is not None
            and self.qkv_fused_weight is not None
            and past_key_value is not None
            and past_key_value._preallocated
        ):
            # Fused QKV projection
            xqkv_fused = ttnn.matmul(hidden_states, self.qkv_fused_weight)
            return self._forward_decode_optimized(
                xqkv_fused,
                past_key_value,
                current_pos,
                rot_mats,
                transformation_mat,
                batch_size,
                current_pos_tensor=current_pos_tensor,
            )

        # Standard path: separate QKV projections
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # NOTE: 1BKD path disabled - rotary_embedding_llama requires HEIGHT_SHARDED inputs
        # which cannot be properly created for batch=1. TT's 1BKD optimizations are designed
        # for batch≥32 scenarios. Using stable BKSD path for batch=1 decode.
        #
        # if (
        #     mode == "decode"
        #     and past_key_value is not None
        #     and past_key_value._preallocated
        #     and self.qkv_fused_weight is not None
        # ):
        #     if rot_mats is None:
        #         rot_mats = self.get_rot_mats(...)
        #     xqkv_fused = ttnn.matmul(hidden_states, self.qkv_fused_weight)
        #     return self._forward_decode_1bkd(...)

        # Use BKSD format for prefill and fallback decode
        return self._forward_simple(
            query,
            key,
            value,
            attention_mask,
            past_key_value,
            current_pos,
            seq_len,
            use_cache,
            batch_size,
            mode,
            current_pos_tensor,
            pos_tensor,
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
        current_pos_tensor: ttnn.Tensor | None = None,  # int32 for KV cache
        pos_tensor: ttnn.Tensor | None = None,  # uint32 for RoPE
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

        # Apply RoPE (uses pos_tensor which is uint32)
        query, key = self._apply_rope_manual(query, key, current_pos, seq_len, pos_tensor)

        # Update KV-Cache with pre-expanded GQA optimization
        updated_cache = None
        if use_cache:
            if past_key_value is None:
                past_key_value = KVCache()

            if mode == "prefill":
                # Prefill: expand and store in cache (BKSD)
                key_expanded, value_expanded = past_key_value.update_prefill(
                    key, value, self.num_kv_groups
                )
            else:
                # Decode: use in-place update if cache is preallocated
                if past_key_value._preallocated:
                    key_expanded, value_expanded = past_key_value.update_decode_inplace(
                        key, value, current_pos, self.num_kv_groups, current_pos_tensor
                    )
                else:
                    # Non-preallocated decode: use concat-based update
                    key_expanded, value_expanded = past_key_value.update_decode_expanded(
                        key, value, current_pos, self.num_kv_groups
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
            updated_cache = None

        # SDPA with BKSD format
        is_causal = mode == "prefill"
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key_expanded,
            value_expanded,
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

    def _forward_decode_1bkd(
        self,
        xqkv_fused: ttnn.Tensor,
        past_key_value: KVCache,
        current_pos: int,
        rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor],
        batch_size: int,
        current_pos_tensor: ttnn.Tensor | None,
    ) -> Tuple[ttnn.Tensor, KVCache]:
        """
        Decode forward with 1BKD tensor format for Metal Trace compatibility.

        Uses TT-Metal pattern:
        - nlp_create_qkv_heads_decode for 1BKD heads
        - rotary_embedding_llama for RoPE (maintains sharded state)
        - paged_update_cache for cache update
        - sdpa_decode for attention
        - nlp_concat_heads_decode for output

        Args:
            xqkv_fused: Fused QKV output [batch, 1, qkv_dim] from matmul
            past_key_value: Pre-allocated KV cache
            current_pos: Current sequence position (int)
            rot_mats: (cos, sin) from get_rot_mats
            batch_size: Batch size
            current_pos_tensor: Position as tensor (int32) for cache update
        """
        # 1. Reshape fused QKV to [1, 1, batch, qkv_dim] for nlp_create_qkv_heads_decode
        # TT pattern uses reshape with tile shape for proper padding
        xqkv_fused = ttnn.to_layout(xqkv_fused, ttnn.ROW_MAJOR_LAYOUT)
        fqkv_shape = xqkv_fused.shape

        if len(fqkv_shape) == 3:
            # 3D: [batch, seq, qkv_dim] -> [1, 1, batch, qkv_dim]
            qkv_dim = fqkv_shape[-1]
            # Use reshape with tile shape (second arg) for proper TILE padding
            xqkv_fused = ttnn.reshape(xqkv_fused, (1, 1, batch_size, qkv_dim), (1, 1, 32, qkv_dim))
        elif len(fqkv_shape) == 4:
            # 4D: already correct format, pass through
            pass
        else:
            raise ValueError(f"Unexpected xqkv_fused shape: {fqkv_shape}")

        xqkv_fused = ttnn.to_layout(xqkv_fused, ttnn.TILE_LAYOUT)

        # 2. Create QKV heads - outputs 1BKD sharded tensors
        # TT pattern: use L1_HEIGHT_SHARDED_MEMORY_CONFIG for Wormhole (model_config.py Line 930)
        q_heads_1bkd, k_heads_1bkd, v_heads_1bkd = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv_fused)

        # 3. Apply RoPE using rotary_embedding_llama (maintains sharded state)
        cos, sin = rot_mats
        q_heads_1bkd = ttnn.experimental.rotary_embedding_llama(
            q_heads_1bkd, cos, sin, self.transformation_mat_decode, is_decode_mode=True
        )
        k_heads_1bkd = ttnn.experimental.rotary_embedding_llama(
            k_heads_1bkd, cos, sin, self.transformation_mat_decode, is_decode_mode=True
        )

        # 4. Create position tensor if not provided
        pos_tensor_local = current_pos_tensor
        pos_created_locally = current_pos_tensor is None
        if pos_created_locally:
            pos_tensor_local = ttnn.from_torch(
                torch.tensor([current_pos], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )

        # 5. Update cache with paged_update_cache
        # TT pattern: nlp outputs are sharded, call paged_update_cache directly
        ttnn.experimental.paged_update_cache(
            past_key_value.key_cache, k_heads_1bkd, update_idxs_tensor=pos_tensor_local
        )
        ttnn.experimental.paged_update_cache(
            past_key_value.value_cache, v_heads_1bkd, update_idxs_tensor=pos_tensor_local
        )
        past_key_value.seq_len_cached = current_pos + 1

        # Cleanup K, V after cache update
        ttnn.deallocate(k_heads_1bkd)
        ttnn.deallocate(v_heads_1bkd)

        # 6. SDPA decode
        # Q must be in DRAM when not sharded
        q_heads_dram = ttnn.to_memory_config(q_heads_1bkd, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_heads_1bkd)

        attn_output_1bkd = ttnn.transformer.scaled_dot_product_attention_decode(
            q_heads_dram,
            past_key_value.key_cache,
            past_key_value.value_cache,
            cur_pos_tensor=pos_tensor_local,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_heads_dram)
        if pos_created_locally:
            ttnn.deallocate(pos_tensor_local)

        # 7. Convert to sharded for nlp_concat_heads_decode
        # TT pattern: L1_HEIGHT_SHARDED_MEMORY_CONFIG
        attn_output_sharded = ttnn.to_memory_config(
            attn_output_1bkd, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        ttnn.deallocate(attn_output_1bkd)

        # 8. Concat heads
        attn_output_concat = ttnn.experimental.nlp_concat_heads_decode(
            attn_output_sharded,
            num_heads=self.num_heads,
        )
        ttnn.deallocate(attn_output_sharded)

        # 9. Convert from sharded to interleaved for RMS norm
        attn_output = ttnn.sharded_to_interleaved(attn_output_concat, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_output_concat)

        # 10. Apply sub-norm and output projection
        # TT pattern: nlp_concat_heads_decode output goes directly to matmul(wo)
        attn_output = self.attn_sub_norm(attn_output)
        output = self.o_proj(attn_output)

        return output, past_key_value

    def _apply_rope_1bkd(
        self,
        x: ttnn.Tensor,
        current_pos: int,
        pos_tensor: ttnn.Tensor | None,
    ) -> ttnn.Tensor:
        """Apply RoPE to 1BKD tensor [1, batch, heads, head_dim]."""
        # For 1BKD, we need to apply RoPE per-head
        # Since we don't have rot_mats, use manual computation
        # nlp_create_qkv_heads_decode outputs sharded tensors, convert to interleaved first
        x_interleaved = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        # Convert to BKSD temporarily for RoPE, then back to 1BKD
        x_bksd = ttnn.to_layout(x_interleaved, ttnn.ROW_MAJOR_LAYOUT)
        x_bksd = ttnn.permute(x_bksd, (1, 2, 0, 3))  # [batch, heads, 1, dim]
        x_bksd = ttnn.to_layout(x_bksd, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x_interleaved)

        # Apply RoPE using existing infrastructure
        rotated = self._apply_rope_single(x_bksd, current_pos, 1, pos_tensor)

        # Convert back to 1BKD
        rotated = ttnn.to_layout(rotated, ttnn.ROW_MAJOR_LAYOUT)
        rotated = ttnn.permute(rotated, (2, 0, 1, 3))  # [1, batch, heads, dim]
        rotated = ttnn.to_layout(rotated, ttnn.TILE_LAYOUT)

        return rotated

    def _apply_rope_single(
        self,
        x: ttnn.Tensor,
        current_pos: int,
        seq_len: int,
        _pos_tensor: ttnn.Tensor | None,  # Reserved for future Trace-safe RoPE
    ) -> ttnn.Tensor:
        """Apply RoPE to single BKSD tensor using embedding lookup."""
        # Create position tensor for embedding lookup
        pos_tensor = ttnn.from_torch(
            torch.tensor([[current_pos]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        # Get cos/sin via embedding lookup
        cos_ttnn = ttnn.embedding(pos_tensor, self.cos_cache_2d, layout=ttnn.TILE_LAYOUT)
        sin_ttnn = ttnn.embedding(pos_tensor, self.sin_cache_2d, layout=ttnn.TILE_LAYOUT)
        cos_ttnn = ttnn.reshape(cos_ttnn, (1, 1, 1, self.head_dim))
        sin_ttnn = ttnn.reshape(sin_ttnn, (1, 1, 1, self.head_dim))
        ttnn.deallocate(pos_tensor)

        # Apply rotary: x * cos + rotate_half(x) * sin
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        x_rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)

        x_cos = ttnn.mul(x, cos_ttnn)
        x_rot_sin = ttnn.mul(x_rot, sin_ttnn)

        return ttnn.add(x_cos, x_rot_sin)

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
            query,
            key_expanded,
            value_expanded,
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
        pos_tensor: ttnn.Tensor | None = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Manual RoPE implementation using device-cached cos/sin tables."""
        if seq_len == 1:
            # Decode mode: use embedding lookup with single position
            pos_tensor_created_locally = pos_tensor is None
            if pos_tensor is None:
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
            # Only deallocate if we created it locally; keep external tensor alive for Trace
            if pos_tensor_created_locally:
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
        current_pos_tensor: ttnn.Tensor | None = None,
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
            current_pos_tensor: Optional tensor containing current position (for trace)

        Returns:
            (output, updated_cache)
        """
        # 1. Create QKV heads using optimized fused op
        # Input: [batch, 1, (n_heads + 2*n_kv_heads) * head_dim]
        # Output: q [1, batch, n_heads, head_dim], k [1, batch, n_kv_heads, head_dim], v [1, batch, n_kv_heads, head_dim]

        # Reshape to 4D for nlp_create_qkv_heads_decode: [1, batch, 1, qkv_dim]
        # xqkv_fused is [batch, 1, qkv_dim]
        xqkv_fused_4d = ttnn.reshape(xqkv_fused, (1, batch_size, 1, self._qkv_dim))
        ttnn.deallocate(xqkv_fused)

        q_heads_1bqd, k_heads_1bkd, v_heads_1bkd = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused_4d,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv_fused_4d)

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
        # Create position tensor for cache update if not provided
        if current_pos_tensor is not None:
            cur_pos_tensor = current_pos_tensor
        else:
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

        # Only deallocate if we created it locally
        if current_pos_tensor is None:
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

    def _forward_prefill_with_preallocated_cache(
        self,
        hidden_states: ttnn.Tensor,
        past_key_value: KVCache,
        batch_size: int,
        seq_len: int,
    ) -> Tuple[ttnn.Tensor, KVCache]:
        """
        Prefill forward that stores non-expanded KV in pre-allocated cache.

        This enables seamless transition to optimized decode path.
        Cache stores [batch, n_kv_heads, max_seq_len, head_dim] (NOT GQA expanded).

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            past_key_value: Pre-allocated KV cache
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            (output, updated_cache)
        """
        # QKV projections
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to BKSD: [batch, heads, seq, head_dim]
        query = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
        key = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)

        query = ttnn.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = ttnn.reshape(key, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        value = ttnn.reshape(value, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        query = ttnn.transpose(query, 1, 2)  # [batch, num_heads, seq, head_dim]
        key = ttnn.transpose(key, 1, 2)  # [batch, num_kv_heads, seq, head_dim]
        value = ttnn.transpose(value, 1, 2)  # [batch, num_kv_heads, seq, head_dim]

        # Apply RoPE
        query, key = self._apply_rope_manual(query, key, 0, seq_len)

        # Store non-expanded KV in pre-allocated cache
        # Copy KV states to the start of the pre-allocated cache
        # For prefill, we store positions 0..seq_len-1

        # Update cache position tracking
        past_key_value.seq_len_cached = seq_len

        # For prefill, we need to slice the pre-allocated cache and copy
        # TT-NN slice and assign: cache[:, :, :seq_len, :] = key/value
        # Since TT-NN doesn't have direct slice assignment, we replace the cache
        # with a new tensor that has the prefill data

        # Deallocate old cache and assign new data
        if past_key_value.key_cache is not None:
            ttnn.deallocate(past_key_value.key_cache)
        if past_key_value.value_cache is not None:
            ttnn.deallocate(past_key_value.value_cache)

        # Pad key/value to max_seq_len if needed for SDPA compatibility
        if seq_len < past_key_value.max_seq_len:
            # Pad with zeros
            pad_len = past_key_value.max_seq_len - seq_len
            pad_shape = (batch_size, self.num_kv_heads, pad_len, self.head_dim)
            zeros = torch.zeros(pad_shape, dtype=torch.bfloat16)
            zeros_ttnn = ttnn.from_torch(
                zeros,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            key_padded = ttnn.concat([key, zeros_ttnn], dim=2)
            value_padded = ttnn.concat([value, zeros_ttnn], dim=2)
            ttnn.deallocate(key)
            ttnn.deallocate(value)
            ttnn.deallocate(zeros_ttnn)
            past_key_value.key_cache = key_padded
            past_key_value.value_cache = value_padded
        else:
            past_key_value.key_cache = key
            past_key_value.value_cache = value

        # For prefill SDPA, we need to expand KV for GQA
        if self.num_kv_groups > 1:
            key_expanded = ttnn.repeat_interleave(
                past_key_value.key_cache[:, :, :seq_len, :], self.num_kv_groups, dim=1
            )
            value_expanded = ttnn.repeat_interleave(
                past_key_value.value_cache[:, :, :seq_len, :], self.num_kv_groups, dim=1
            )
        else:
            key_expanded = past_key_value.key_cache[:, :, :seq_len, :]
            value_expanded = past_key_value.value_cache[:, :, :seq_len, :]

        # SDPA (causal attention for prefill)
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key_expanded,
            value_expanded,
            attn_mask=None,
            is_causal=True,
            scale=self.scale,
        )

        # Cleanup expanded tensors
        if self.num_kv_groups > 1:
            ttnn.deallocate(key_expanded)
            ttnn.deallocate(value_expanded)

        # Reshape output: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.to_layout(attn_output, ttnn.ROW_MAJOR_LAYOUT)
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.hidden_size))
        attn_output = ttnn.to_layout(attn_output, ttnn.TILE_LAYOUT)

        # Apply sub-norm and output projection
        attn_output = self.attn_sub_norm(attn_output)
        output = self.o_proj(attn_output)

        # Mark cache as properly initialized for optimized decode
        past_key_value._preallocated = True

        return output, past_key_value
