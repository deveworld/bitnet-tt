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

from bitnet_tt.config import get_compute_kernel_config
from bitnet_tt.layers.bitlinear import Linear, RMSNorm, quantize_and_transpose_weight
from bitnet_tt.utils.rope import precompute_freqs_cis


def fill_cache_range(
    cache_tensor: ttnn.Tensor,
    input_tensor: ttnn.Tensor,
    batch_start: int = 0,
) -> ttnn.Tensor:
    """
    Fill a contiguous batch range in KV cache.

    Newer TT-NN runtimes can ingest the full batch range in one call. Older
    runtimes only support batch=1 and need the conservative per-batch loop.
    """
    input_batch = int(input_tensor.shape[0])
    if input_batch <= 1:
        return ttnn.fill_cache(cache_tensor, input_tensor, batch_start)

    try:
        return ttnn.fill_cache(cache_tensor, input_tensor, batch_start)
    except RuntimeError as exc:
        if "Input tensor batch size must be 1" not in str(exc):
            raise

    for batch_idx in range(input_batch):
        batch_tensor = input_tensor[batch_idx : batch_idx + 1, :, :, :]
        cache_tensor = ttnn.fill_cache(cache_tensor, batch_tensor, batch_start + batch_idx)
    return cache_tensor


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
    _gqa_expanded: bool = False
    _shard_config: object = None

    def preallocate(
        self,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: ttnn.Device,
        num_heads: int | None = None,
        use_paged: bool = False,
        cache_dtype: str = "bf16",
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
            use_paged: If True, allocate DRAM cache for paged_update_cache updates
        """
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device
        self._use_paged = use_paged

        # Determine cache head count
        if use_paged:
            # Batch-32 decode already provides enough height for paged cache updates,
            # so keep the cache compact in KV-head space and avoid per-token head slicing.
            cache_heads = num_kv_heads
            self.num_heads = num_heads if num_heads is not None else num_kv_heads
            self._gqa_expanded = False
        elif num_heads is not None:
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
        _cache_dtype_map = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
        ttnn_cache_dtype = _cache_dtype_map.get(cache_dtype, ttnn.bfloat16)
        cache_shape = (batch_size, cache_heads, max_seq_len, head_dim)
        zeros = torch.zeros(cache_shape, dtype=torch.bfloat16)

        self.key_cache = ttnn.from_torch(
            zeros,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn_cache_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.value_cache = ttnn.from_torch(
            zeros,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn_cache_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.seq_len_cached = 0
        self._preallocated = True

        if use_paged:
            self._shard_config = ttnn.create_sharded_memory_config(
                shape=(32, head_dim),
                core_grid=ttnn.CoreGrid(y=4, x=8),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

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
            cache_heads = self.key_cache.shape[1]
            kv_heads = key_states.shape[1]

            if self._use_paged and cache_heads > kv_heads:
                pad_amount = cache_heads - kv_heads
                key_padded = ttnn.pad(key_states, [(0, 0), (0, pad_amount), (0, 0), (0, 0)], 0.0)
                value_padded = ttnn.pad(
                    value_states, [(0, 0), (0, pad_amount), (0, 0), (0, 0)], 0.0
                )

                self.key_cache = fill_cache_range(self.key_cache, key_padded, 0)
                self.value_cache = fill_cache_range(self.value_cache, value_padded, 0)

                ttnn.deallocate(key_padded)
                ttnn.deallocate(value_padded)
            else:
                self.key_cache = fill_cache_range(self.key_cache, key_states, 0)
                self.value_cache = fill_cache_range(self.value_cache, value_states, 0)

            self.seq_len_cached = seq_len

            key_for_attn = self.key_cache[:, : self.num_kv_heads, :seq_len, :]
            value_for_attn = self.value_cache[:, : self.num_kv_heads, :seq_len, :]
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

    def update_decode_simple(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        current_pos: int,
        num_kv_groups: int = 1,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Update pre-allocated cache using ttnn.kv_cache.update_cache_for_token_.

        This is the PREFERRED decode method for Phase 1 optimization:
        - Uses kv_cache API that accepts [batch, heads, 1, head_dim] directly
        - Avoids memory allocation (no concat)
        - No permute/padding overhead

        Args:
            key_states: [batch, kv_heads, 1, head_dim] - single new token
            value_states: [batch, kv_heads, 1, head_dim] - single new token
            current_pos: Current sequence position (update index)
            num_kv_groups: Number of query heads per KV head (for GQA expansion)

        Returns:
            GQA-expanded key and value for attention (sliced from cache)
        """
        if not self._preallocated:
            raise RuntimeError("Cache must be preallocated for in-place update")

        # Use kv_cache.update_cache_for_token_ which accepts [batch, heads, 1, head_dim] directly
        self.key_cache = ttnn.kv_cache.update_cache_for_token_(
            self.key_cache, key_states, update_index=current_pos, batch_offset=0
        )
        self.value_cache = ttnn.kv_cache.update_cache_for_token_(
            self.value_cache, value_states, update_index=current_pos, batch_offset=0
        )

        self.seq_len_cached = current_pos + 1

        # Slice cache up to current position for attention
        key_for_attn = self.key_cache[:, :, : self.seq_len_cached, :]
        value_for_attn = self.value_cache[:, :, : self.seq_len_cached, :]

        # Expand for GQA when returning
        if num_kv_groups > 1:
            key_for_attn = ttnn.repeat_interleave(key_for_attn, num_kv_groups, dim=1)
            value_for_attn = ttnn.repeat_interleave(value_for_attn, num_kv_groups, dim=1)

        return key_for_attn, value_for_attn

    def update_decode_trace(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        current_pos: int,
        current_pos_tensor: ttnn.Tensor,
        num_kv_groups: int = 1,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Trace-compatible cache update using paged_update_cache with position tensor.

        Uses paged_update_cache with update_idxs_tensor for trace compatibility:
        - Position tensor allows trace to update different positions each execution
        - Requires 1BKD format and HEIGHT_SHARDED memory config
        - Cache must have 32 padded heads

        NOTE: This still allocates during trace due to format conversion operations.
        True trace-compatible decode requires HEIGHT_SHARDED throughout (batch>=32 or heads%32==0).
        """
        if not self._preallocated:
            raise RuntimeError("Cache must be preallocated for trace-compatible update")

        kv_heads = key_states.shape[1]
        padded_heads = 32
        pad_amount = padded_heads - kv_heads

        key_rm = ttnn.to_layout(key_states, ttnn.ROW_MAJOR_LAYOUT)
        key_1bkd = ttnn.permute(key_rm, (2, 0, 1, 3))
        key_1bkd = ttnn.to_layout(key_1bkd, ttnn.TILE_LAYOUT)

        value_rm = ttnn.to_layout(value_states, ttnn.ROW_MAJOR_LAYOUT)
        value_1bkd = ttnn.permute(value_rm, (2, 0, 1, 3))
        value_1bkd = ttnn.to_layout(value_1bkd, ttnn.TILE_LAYOUT)

        key_padded = ttnn.pad(key_1bkd, [(0, 0), (0, 0), (0, pad_amount), (0, 0)], 0.0)
        value_padded = ttnn.pad(value_1bkd, [(0, 0), (0, 0), (0, pad_amount), (0, 0)], 0.0)

        key_sharded = ttnn.to_memory_config(key_padded, self._shard_config)
        value_sharded = ttnn.to_memory_config(value_padded, self._shard_config)

        ttnn.experimental.paged_update_cache(
            self.key_cache, key_sharded, update_idxs_tensor=current_pos_tensor
        )
        ttnn.experimental.paged_update_cache(
            self.value_cache, value_sharded, update_idxs_tensor=current_pos_tensor
        )

        self.seq_len_cached = current_pos + 1

        key_for_attn = self.key_cache[:, : self.num_kv_heads, :, :]
        value_for_attn = self.value_cache[:, : self.num_kv_heads, :, :]

        return key_for_attn, value_for_attn

    def update_decode_expanded(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        current_pos: int | None = None,
        num_kv_groups: int = 1,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Update pre-expanded cache for decode using concat.

        DEPRECATED: Use update_decode_simple instead for better performance.
        This method uses concat which allocates new memory every decode step,
        making Trace capture impossible.

        Key optimization: Expand only the NEW token's KV (1 position).

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



def _permute_halfsplit_to_adjacent(weight_t: NDArray, num_heads: int, head_dim: int) -> NDArray:
    """Permute transposed Q/K weight columns: HF half-split → TT adjacent-pair RoPE order.

    After this permutation, activations from matmul(x, W_permuted) will have
    adjacent-pair layout matching rotary_embedding_llama's convention.
    """
    half = head_dim // 2
    perm = np.empty(head_dim, dtype=np.intp)
    for k in range(half):
        perm[2 * k] = k
        perm[2 * k + 1] = k + half

    out = weight_t.copy()
    for h in range(num_heads):
        col_start = h * head_dim
        out[:, col_start:col_start + head_dim] = weight_t[:, col_start + perm]
    return out


def build_fused_qkv_weight(
    q_weight: NDArray[np.floating],
    k_weight: NDArray[np.floating],
    v_weight: NDArray[np.floating],
    num_heads: int = 0,
    num_kv_heads: int = 0,
    head_dim: int = 0,
    permute_for_rope: bool = False,
) -> tuple[NDArray[np.float32], float, float, float]:
    """
    Build a single pre-transposed QKV matrix from separately quantized weights.

    Returns pure ternary {-1,0,+1} fused weight and per-projection scales.
    If permute_for_rope=True, permutes Q/K head dimensions from HF half-split
    to TT adjacent-pair order for rotary_embedding_llama compatibility.
    """
    q_t, q_scale = quantize_and_transpose_weight(q_weight)
    k_t, k_scale = quantize_and_transpose_weight(k_weight)
    v_t, v_scale = quantize_and_transpose_weight(v_weight)

    if permute_for_rope and head_dim > 0:
        q_t = _permute_halfsplit_to_adjacent(q_t, num_heads, head_dim)
        k_t = _permute_halfsplit_to_adjacent(k_t, num_kv_heads, head_dim)

    fused = np.concatenate([q_t, k_t, v_t], axis=1)
    return fused, q_scale, k_scale, v_scale


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
        use_fused_rope: bool = True,
        layer_idx: int = 0,
        weight_dtype: str = "bfp4",
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
        # Effective SDPA scale after load_weights folds q_scale*k_scale in.
        # Default = self.scale (no fold) until load_weights runs.
        self._sdpa_scale = self.scale
        self.use_fused_rope = use_fused_rope
        self.layer_idx = layer_idx
        self.max_position_embeddings = max_position_embeddings
        self._weight_dtype = weight_dtype

        # Projections (weights pre-transposed in Linear.load_weights)
        self.q_proj = Linear(hidden_size, num_attention_heads * self.head_dim, device, weight_dtype=weight_dtype)
        self.k_proj = Linear(hidden_size, num_key_value_heads * self.head_dim, device, weight_dtype=weight_dtype)
        self.v_proj = Linear(hidden_size, num_key_value_heads * self.head_dim, device, weight_dtype=weight_dtype)
        self.o_proj = Linear(num_attention_heads * self.head_dim, hidden_size, device, weight_dtype=weight_dtype)

        # Fused QKV projection for optimized decode (created when weights are loaded)
        self.qkv_fused_weight: ttnn.Tensor | None = None
        self._qkv_fused_linear: Linear | None = None
        self._qkv_use_packed_ternary: bool = False
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
        # HEIGHT_SHARDED transformation matrix for rotary_embedding_llama decode mode
        _trans_mat_dram = ttnn.from_torch(
            trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _trans_mat_sharded_config = ttnn.create_sharded_memory_config(
            shape=(32, 32),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.transformation_mat_decode = ttnn.to_memory_config(
            _trans_mat_dram, _trans_mat_sharded_config
        )

        # HEIGHT_SHARDED config for cos/sin in rotary_embedding_llama decode mode
        self._rope_cos_sin_config = ttnn.create_sharded_memory_config(
            shape=(32, self.head_dim),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # HEIGHT_SHARDED config for SDPA decode output, shaped so
        # nlp_concat_heads_decode can consume it directly (it requires
        # a sharded input with padded_heads=32). 32 batch-rows × 32 cores.
        padded_heads = ((self.num_heads + 31) // 32) * 32
        self._sdpa_output_sharded_config = ttnn.create_sharded_memory_config(
            shape=(padded_heads, self.head_dim),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # 128x128 rotation matrix for prefill adjacent-pair RoPE
        # rot[2k, 2k+1] = -1, rot[2k+1, 2k] = 1 for each pair
        rot_mat_128 = torch.zeros(1, 1, self.head_dim, self.head_dim)
        for k in range(self.head_dim // 2):
            rot_mat_128[0, 0, 2 * k, 2 * k + 1] = -1
            rot_mat_128[0, 0, 2 * k + 1, 2 * k] = 1
        self._rope_rotation_mat = ttnn.from_torch(
            rot_mat_128,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Store core grid for dynamic sharding
        self.core_grid = device.compute_with_storage_grid_size()
        self._sdpa_decode_compute_kernel_config = None
        self._sdpa_decode_program_cache: dict[int, ttnn.SDPAProgramConfig] = {}
        try:
            self._sdpa_decode_compute_kernel_config = get_compute_kernel_config("hifi2")
        except Exception:
            self._sdpa_decode_compute_kernel_config = None

    def _get_decode_k_chunk_size(self, active_seq_len: int, padded_seq_len: int) -> int:
        """Match TT decode chunk sizing closely enough to avoid slow fallbacks."""
        if active_seq_len <= 128:
            chunk_size = 32
        elif active_seq_len <= 1024:
            chunk_size = 128
        else:
            chunk_size = 512

        max_power_divisor = 1
        while padded_seq_len % (max_power_divisor * 2) == 0:
            max_power_divisor *= 2

        return min(chunk_size, max_power_divisor)

    def _get_sdpa_decode_program_config(self, active_seq_len: int, padded_seq_len: int):
        """Build and cache SDPA decode program configs keyed by visible cache length."""
        k_chunk_size = self._get_decode_k_chunk_size(active_seq_len, padded_seq_len)
        cached = self._sdpa_decode_program_cache.get(k_chunk_size)
        if cached is not None:
            return cached

        # On Blackhole the default 11x10 = 110 core grid combined with
        # num_kv_heads=5 and padded_batch=32 trips
        # sdpa_decode_program_factory's `idx < num_output_cores` check:
        # num_cores_per_batch rounds down to 2 and num_active_cores becomes
        # 80, producing 40 output candidates for only B=32 output cores.
        # Using an 8x4 = 32 core grid makes num_cores_per_batch=1 and
        # num_active_cores=32, matching num_output_cores exactly.
        sdpa_grid = ttnn.CoreCoord(8, 4)

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_grid,
            q_chunk_size=32,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )
        self._sdpa_decode_program_cache[k_chunk_size] = program_config
        return program_config

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

        qkv_fused, q_s, k_s, v_s = build_fused_qkv_weight(
            q_weight, k_weight, v_weight,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            permute_for_rope=self.use_fused_rope,
        )
        self._qkv_scales = (q_s, k_s, v_s)

        # Fused QKV weight: route through a Linear so that packed_ternary
        # lands in the BFP2_b mantissa-only format consumed by
        # ttnn.experimental.ternary_matmul. Scales are applied per-head
        # after nlp_create_qkv_heads_decode, so the Linear is built with
        # scale=1.0.
        self._qkv_fused_linear = Linear(
            in_features=self.hidden_size,
            out_features=self._qkv_dim,
            device=self.device,
            weight_dtype=self._weight_dtype,
        )
        self._qkv_fused_linear.load_pretransposed_weight(qkv_fused, scale=1.0)
        self.qkv_fused_weight = self._qkv_fused_linear.weight
        self._qkv_use_packed_ternary = self._qkv_fused_linear._use_packed_ternary

        # QKV scale fold. The three per-projection quantization scales
        # (q_scale, k_scale, v_scale) do not need to be applied as separate
        # elementwise multiplies on the head tensors:
        #
        #   softmax((Q*q_s) @ (K*k_s).T * self.scale) @ (V*v_s)
        #     = softmax(Q @ K.T * (self.scale*q_s*k_s)) @ V * v_s
        #
        # q_s*k_s collapses into SDPA's scale argument. v_s survives as a
        # positive scalar factor on the attention output, which is then
        # immediately fed into attn_sub_norm (an RMSNorm) that absorbs any
        # positive scalar factor exactly — so v_s can simply be dropped.
        #
        # This removes 3 ttnn.multiply dispatches per layer (~90 ops/step)
        # from the decode path and 3 from the prefill path.
        self._q_scale = float(q_s)
        self._k_scale = float(k_s)
        self._v_scale = float(v_s)
        self._sdpa_scale = self.scale * self._q_scale * self._k_scale
        # Suppress the trailing ttnn.multiply inside each per-projection
        # Linear call (prefill path), since scales are now folded / absorbed.
        self.q_proj.weight_scale = 1.0
        self.k_proj.weight_scale = 1.0
        self.v_proj.weight_scale = 1.0

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
        cos_sin_tensors: Tuple[ttnn.Tensor, ttnn.Tensor]
        | None = None,  # Pre-computed cos/sin for trace
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

        # Standard prefill/fallback paths keep separate Q/K/V projections. The fused
        # QKV weight is still used by the batch32 decode core, which consumes the fused
        # output directly without slicing. In the generic path, TTNN matmul currently
        # produces a compact trailing dimension that breaks the subsequent split logic.
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

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
            cos_sin_tensors,
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
        cos_sin_tensors: Tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> Tuple[ttnn.Tensor, Optional[KVCache]]:
        """
        Optimized forward using pre-expanded GQA cache.

        Key optimization: Cache stores already-expanded KV heads.
        - Prefill: Expand full sequence, store in cache
        - Decode: Expand only 1 new token, concat to expanded cache

        This avoids expanding all cached positions every decode step.
        """
        # Reshape to BKSD: [batch, heads, seq, head_dim]
        # reshape and transpose work directly in TILE_LAYOUT
        query = ttnn.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = ttnn.reshape(key, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        value = ttnn.reshape(value, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        query = ttnn.transpose(query, 1, 2)
        key = ttnn.transpose(key, 1, 2)
        value = ttnn.transpose(value, 1, 2)

        # Apply RoPE (uses pos_tensor which is uint32, cos_sin_tensors for trace mode)
        query, key = self._apply_rope_manual(
            query, key, current_pos, seq_len, pos_tensor, cos_sin_tensors
        )

        # Update KV-Cache with pre-expanded GQA optimization
        updated_cache = None
        if use_cache:
            if past_key_value is None:
                past_key_value = KVCache()

            if mode == "prefill":
                key_expanded, value_expanded = past_key_value.update_prefill(
                    key, value, self.num_kv_groups
                )
            else:
                if (
                    past_key_value._preallocated
                    and hasattr(past_key_value, "_use_paged")
                    and past_key_value._use_paged
                    and current_pos_tensor is not None
                ):
                    key_expanded, value_expanded = past_key_value.update_decode_trace(
                        key, value, current_pos, current_pos_tensor, self.num_kv_groups
                    )
                elif past_key_value._preallocated:
                    key_expanded, value_expanded = past_key_value.update_decode_simple(
                        key, value, current_pos, self.num_kv_groups
                    )
                else:
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

        use_decode_sdpa = (
            mode == "decode"
            and past_key_value is not None
            and hasattr(past_key_value, "_preallocated")
            and past_key_value._preallocated
            and current_pos_tensor is not None
        )

        if use_decode_sdpa:
            # permute works directly in TILE_LAYOUT - no layout conversion needed
            q_1bkd = ttnn.permute(query, (2, 0, 1, 3))

            attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
                q_1bkd,
                key_expanded,
                value_expanded,
                cur_pos_tensor=current_pos_tensor,
                scale=self._sdpa_scale,
            )

            # permute works directly in TILE_LAYOUT - no layout conversion needed
            attn_output = ttnn.permute(attn_output, (1, 2, 0, 3))
        else:
            is_causal = mode == "prefill"
            attn_output = ttnn.transformer.scaled_dot_product_attention(
                query,
                key_expanded,
                value_expanded,
                attn_mask=attention_mask,
                is_causal=is_causal,
                scale=self._sdpa_scale,
            )

        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.hidden_size))

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
        # 1. Reshape fused QKV to [1, 1, batch, qkv_dim] for nlp_create_qkv_heads_decode.
        # Fast path: batch32 decode hands us a 4D TILE tensor directly from
        # ternary_matmul — shape is already [1,1,32,qkv_dim] and layout TILE,
        # so no conversion is needed (saves 3 ops / layer = 90 ops / step).
        fqkv_shape = xqkv_fused.shape
        if len(fqkv_shape) == 4 and xqkv_fused.layout == ttnn.TILE_LAYOUT:
            pass
        else:
            # Slow fallback: 3D prefill/batchN inputs need reshape, which
            # requires the intermediate to be ROW_MAJOR because the target
            # shape may not be tile-aligned on every axis.
            xqkv_fused = ttnn.to_layout(xqkv_fused, ttnn.ROW_MAJOR_LAYOUT)
            if len(fqkv_shape) == 3:
                qkv_dim = fqkv_shape[-1]
                xqkv_fused = ttnn.reshape(
                    xqkv_fused, (1, 1, batch_size, qkv_dim), (1, 1, 32, qkv_dim)
                )
            elif len(fqkv_shape) != 4:
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

        # Per-projection quantization scales are folded at init time:
        # q_scale*k_scale into self._sdpa_scale, v_scale absorbed by the
        # downstream attn_sub_norm RMSNorm. See MultiHeadAttention.load_weights
        # for the derivation.

        # 3. Apply RoPE
        cos_dram, sin_dram = rot_mats
        if self.use_fused_rope:
            # Fused path: rotary_embedding_llama with HEIGHT_SHARDED inputs.
            # cos/sin are pre-sharded once per step by
            # generator_batch32._decode_step_batch32 (caller owns the lifetime
            # and deallocates after the layer loop), so we skip the per-layer
            # to_memory_config here. (rotary_embedding_llama_fused_qk would
            # save another dispatch but requires Q and K to be on
            # non-overlapping core ranges, which we don't split into.)
            q_heads_1bkd = ttnn.experimental.rotary_embedding_llama(
                q_heads_1bkd, cos_dram, sin_dram, self.transformation_mat_decode,
                is_decode_mode=True,
            )
            k_heads_1bkd = ttnn.experimental.rotary_embedding_llama(
                k_heads_1bkd, cos_dram, sin_dram, self.transformation_mat_decode,
                is_decode_mode=True,
            )
        else:
            # Manual path: half-split RoPE with HF cos/sin (16 t/s, higher accuracy)
            cos, sin = cos_dram, sin_dram
            q_dram = ttnn.to_memory_config(q_heads_1bkd, ttnn.DRAM_MEMORY_CONFIG)
            k_dram = ttnn.to_memory_config(k_heads_1bkd, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(q_heads_1bkd)
            ttnn.deallocate(k_heads_1bkd)
            q_bhsd = ttnn.permute(q_dram, (1, 2, 0, 3))
            k_bhsd = ttnn.permute(k_dram, (1, 2, 0, 3))
            ttnn.deallocate(q_dram)
            ttnn.deallocate(k_dram)
            q_bhsd = self._apply_rope_to_tensor(q_bhsd, cos, sin)
            k_bhsd = self._apply_rope_to_tensor(k_bhsd, cos, sin)
            q_heads_1bkd = ttnn.permute(q_bhsd, (2, 0, 1, 3))
            k_heads_1bkd = ttnn.permute(k_bhsd, (2, 0, 1, 3))
            ttnn.deallocate(q_bhsd)
            ttnn.deallocate(k_bhsd)
            v_dram = ttnn.to_memory_config(v_heads_1bkd, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(v_heads_1bkd)
            v_heads_1bkd = v_dram

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

        # 5. Update cache with paged_update_cache (requires HEIGHT_SHARDED input).
        # paged_fused_update_cache requires K and V to live on non-overlapping
        # core ranges (production llama3 splits them to different cores),
        # which we don't do — so we keep two separate calls here.
        #
        # Optimistic: skip both typecast and to_memory_config entirely. The
        # KV cache is created with cache_dtype=bf16 by default (matches the
        # K/V tensors right out of RoPE), and paged_update_cache appears to
        # accept the L1_HEIGHT_SHARDED_MEMORY_CONFIG output of the upstream
        # head-split + RoPE path directly. If a future caller sets a
        # cache_dtype that differs from the K/V dtype, re-introduce a
        # conditional typecast here.
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

        # 6. SDPA decode — Q is already in DRAM after permute
        q_heads_dram = q_heads_1bkd

        # Limit the visible cache length to a 32-token tile multiple. When the
        # backing cache already matches that compact logical view, reuse it
        # directly to avoid per-token K/V slice overhead in every layer.
        #
        # IMPORTANT: cur_pos_tensor is a correctness mask inside SDPA, not a
        # compute short-circuit. Passing the full preallocated cache makes
        # SDPA iterate over every k_chunk in the full cache, even if most of
        # them are masked out. Measured regression was 37.5 -> 31 t/s. So we
        # keep the slice — the dispatch cost is much smaller than the wasted
        # SDPA k-chunk work would be.
        active_seq_len = current_pos + 1
        padded_seq_len = ((active_seq_len + 31) // 32) * 32
        padded_seq_len = min(padded_seq_len, past_key_value.key_cache.shape[2])
        sdpa_program_config = self._get_sdpa_decode_program_config(active_seq_len, padded_seq_len)
        can_reuse_full_cache = (
            past_key_value.key_cache.shape[1] == self.num_kv_heads
            and past_key_value.key_cache.shape[2] == padded_seq_len
        )
        if can_reuse_full_cache:
            key_cache_for_sdpa = past_key_value.key_cache
            value_cache_for_sdpa = past_key_value.value_cache
        else:
            key_cache_for_sdpa = ttnn.slice(
                past_key_value.key_cache,
                (0, 0, 0, 0),
                (
                    past_key_value.key_cache.shape[0],
                    self.num_kv_heads,
                    padded_seq_len,
                    past_key_value.key_cache.shape[3],
                ),
            )
            value_cache_for_sdpa = ttnn.slice(
                past_key_value.value_cache,
                (0, 0, 0, 0),
                (
                    past_key_value.value_cache.shape[0],
                    self.num_kv_heads,
                    padded_seq_len,
                    past_key_value.value_cache.shape[3],
                ),
            )
        attn_output_1bkd = ttnn.transformer.scaled_dot_product_attention_decode(
            q_heads_dram,
            key_cache_for_sdpa,
            value_cache_for_sdpa,
            cur_pos_tensor=pos_tensor_local,
            scale=self._sdpa_scale,
            program_config=sdpa_program_config,
            compute_kernel_config=self._sdpa_decode_compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_heads_dram)
        if not can_reuse_full_cache:
            ttnn.deallocate(key_cache_for_sdpa)
            ttnn.deallocate(value_cache_for_sdpa)
        if pos_created_locally:
            ttnn.deallocate(pos_tensor_local)

        # 7. Concat heads via nlp_concat_heads_decode. SDPA's GQA path can
        # only emit interleaved output, so we reshard into the HEIGHT_SHARDED
        # layout the concat_heads kernel expects, then desharded back for
        # the downstream sub-norm.
        attn_sdpa_sharded = ttnn.to_memory_config(
            attn_output_1bkd, self._sdpa_output_sharded_config
        )
        ttnn.deallocate(attn_output_1bkd)
        attn_concat_sharded = ttnn.experimental.nlp_concat_heads_decode(
            attn_sdpa_sharded, num_heads=self.num_heads
        )
        ttnn.deallocate(attn_sdpa_sharded)
        attn_output = ttnn.sharded_to_interleaved(
            attn_concat_sharded, ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(attn_concat_sharded)

        # 8. Apply sub-norm and output projection.
        # L1 output keeps the norm result close to the downstream ternary_matmul
        # reader, avoiding a DRAM round-trip for the activation tensor.
        attn_output = self.attn_sub_norm(attn_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        output = self.o_proj(attn_output)

        return output, past_key_value

    def _apply_rope_manual(
        self,
        query: ttnn.Tensor,
        key: ttnn.Tensor,
        start_pos: int,
        seq_len: int,
        pos_tensor: ttnn.Tensor | None = None,
        cos_sin_tensors: Tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Manual RoPE implementation using device-cached cos/sin tables.

        Args:
            query: Query tensor [batch, heads, seq, head_dim]
            key: Key tensor [batch, heads, seq, head_dim]
            start_pos: Starting position for RoPE
            seq_len: Sequence length
            pos_tensor: Optional position tensor for embedding lookup (deprecated for trace)
            cos_sin_tensors: Optional pre-computed (cos, sin) tensors for trace mode.
                             When provided, skips embedding lookup entirely.
                             Expected shape: [1, 1, 1, head_dim] for decode.
        """
        if cos_sin_tensors is not None:
            # Trace mode: use pre-computed cos/sin directly (no embedding lookup)
            cos_ttnn, sin_ttnn = cos_sin_tensors
        elif seq_len == 1:
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
        """Apply RoPE to a single tensor. Convention depends on use_fused_rope."""
        if self.use_fused_rope:
            # Adjacent-pair: x @ rot_matrix_128
            x_rotated = ttnn.matmul(x, self._rope_rotation_mat)
        else:
            # HF half-split: [-x2, x1]
            half_dim = self.head_dim // 2
            x1 = x[:, :, :, :half_dim]
            x2 = x[:, :, :, half_dim:]
            x_rotated = ttnn.concat([ttnn.neg(x2), x1], dim=-1)

        # Use broadcasting in multiply instead of explicit repeat
        # This eliminates 4 repeat ops per layer (120 total for 30 layers)
        x_cos = ttnn.multiply(x, cos)
        x_rot_sin = ttnn.multiply(x_rotated, sin)
        return ttnn.add(x_cos, x_rot_sin)

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

        past_key_value.seq_len_cached = seq_len

        if past_key_value._use_paged and past_key_value.key_cache is not None:
            cache_heads = past_key_value.key_cache.shape[1]
            kv_heads = key.shape[1]

            if cache_heads > kv_heads:
                pad_amount = cache_heads - kv_heads
                key_padded = ttnn.pad(key, [(0, 0), (0, pad_amount), (0, 0), (0, 0)], 0.0)
                value_padded = ttnn.pad(value, [(0, 0), (0, pad_amount), (0, 0), (0, 0)], 0.0)

                past_key_value.key_cache = fill_cache_range(past_key_value.key_cache, key_padded, 0)
                past_key_value.value_cache = fill_cache_range(past_key_value.value_cache, value_padded, 0)

                ttnn.deallocate(key_padded)
                ttnn.deallocate(value_padded)
            else:
                past_key_value.key_cache = fill_cache_range(past_key_value.key_cache, key, 0)
                past_key_value.value_cache = fill_cache_range(past_key_value.value_cache, value, 0)
            ttnn.deallocate(key)
            ttnn.deallocate(value)
        else:
            if past_key_value.key_cache is not None:
                ttnn.deallocate(past_key_value.key_cache)
            if past_key_value.value_cache is not None:
                ttnn.deallocate(past_key_value.value_cache)

            if seq_len < past_key_value.max_seq_len:
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

        kv_heads_slice = self.num_kv_heads
        if self.num_kv_groups > 1:
            key_expanded = ttnn.repeat_interleave(
                past_key_value.key_cache[:, :kv_heads_slice, :seq_len, :],
                self.num_kv_groups,
                dim=1,
            )
            value_expanded = ttnn.repeat_interleave(
                past_key_value.value_cache[:, :kv_heads_slice, :seq_len, :],
                self.num_kv_groups,
                dim=1,
            )
        else:
            key_expanded = past_key_value.key_cache[:, :kv_heads_slice, :seq_len, :]
            value_expanded = past_key_value.value_cache[:, :kv_heads_slice, :seq_len, :]

        # SDPA (causal attention for prefill)
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key_expanded,
            value_expanded,
            attn_mask=None,
            is_causal=True,
            scale=self._sdpa_scale,
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
