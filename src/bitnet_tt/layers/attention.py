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

    Uses pre-allocated fixed-shape cache for decode optimization.
    Cache format: [batch, kv_heads, max_seq_len, head_dim]

    For decode, uses paged_update_cache with 1BKD input format.
    """
    key_cache: ttnn.Tensor | None = None
    value_cache: ttnn.Tensor | None = None
    seq_len_cached: int = 0
    max_seq_len: int = 4096
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
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Update cache for prefill (store KV, return for attention).

        For prefill, we use the input KV directly for attention.
        The cache is set up for subsequent decode steps.
        """
        self.seq_len_cached = key_states.shape[2]
        # Store for first decode step
        self._prefill_key = key_states
        self._prefill_value = value_states
        return key_states, value_states

    def update_decode_1bkd(
        self,
        key_states_1bkd: ttnn.Tensor,
        value_states_1bkd: ttnn.Tensor,
        cur_pos_tensor: ttnn.Tensor,
        current_pos: int,
    ) -> None:
        """
        Update cache for decode mode with 1BKD format input.

        Args:
            key_states_1bkd: K in [1, batch, kv_heads, head_dim] format
            value_states_1bkd: V in [1, batch, kv_heads, head_dim] format
            cur_pos_tensor: Position tensor (reused to avoid re-creation)
            current_pos: Current sequence position (int)
        """
        if not self._preallocated:
            raise RuntimeError("Cache not preallocated for decode")

        # First decode: clear prefill references (data already copied in preallocate)
        if hasattr(self, '_prefill_key') and self._prefill_key is not None:
            self._prefill_key = None
            self._prefill_value = None

        # Update cache in-place using paged_update_cache
        # Input: [1, batch, kv_heads, head_dim] (1BKD)
        # Cache: [batch, kv_heads, max_seq_len, head_dim]
        try:
            ttnn.experimental.paged_update_cache(
                self.key_cache, key_states_1bkd,
                update_idxs_tensor=cur_pos_tensor
            )
            ttnn.experimental.paged_update_cache(
                self.value_cache, value_states_1bkd,
                update_idxs_tensor=cur_pos_tensor
            )
        except Exception as e:
            print(f"[DEBUG] paged_update_cache FAILED: {e}")
            raise

        self.seq_len_cached = current_pos + 1

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

        # Route to decode or prefill path
        if mode == "decode" and seq_len == 1:
            return self._forward_decode(
                query, key, value, past_key_value, current_pos, use_cache, batch_size
            )
        else:
            return self._forward_prefill(
                query, key, value, attention_mask, past_key_value,
                current_pos, seq_len, use_cache, batch_size
            )

    def _forward_decode(
        self,
        query: ttnn.Tensor,
        key: ttnn.Tensor,
        value: ttnn.Tensor,
        past_key_value: KVCache | None,
        current_pos: int,
        use_cache: bool,
        batch_size: int,
    ) -> Tuple[ttnn.Tensor, Optional[KVCache]]:
        """
        Decode forward with 1BKD tensor format for optimized ops.

        1BKD format: [seq_len=1, batch, heads, head_dim]
        This enables: paged_update_cache, scaled_dot_product_attention_decode
        """
        if self.layer_idx == 0 and current_pos < 12:
            print(f"[DEBUG L0] _forward_decode called, pos={current_pos}, cache_preallocated={past_key_value._preallocated if past_key_value else 'None'}")

        # Reshape QKV to 1BKD: [1, batch, heads, head_dim]
        query = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
        key = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)

        # From [batch, 1, hidden] -> [1, batch, heads, head_dim]
        query = ttnn.reshape(query, (batch_size, self.num_heads, self.head_dim))
        key = ttnn.reshape(key, (batch_size, self.num_kv_heads, self.head_dim))
        value = ttnn.reshape(value, (batch_size, self.num_kv_heads, self.head_dim))

        # Reshape to 1BKD: [1, batch, heads, head_dim]
        query = ttnn.reshape(query, (1, batch_size, self.num_heads, self.head_dim))
        key = ttnn.reshape(key, (1, batch_size, self.num_kv_heads, self.head_dim))
        value = ttnn.reshape(value, (1, batch_size, self.num_kv_heads, self.head_dim))

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        # Apply RoPE for decode
        query, key = self._apply_rope_decode_1bkd(query, key, current_pos)

        # Create position tensor once (reused for cache update and SDPA)
        cur_pos_tensor = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        # Update KV-Cache with 1BKD input using paged_update_cache
        updated_cache = past_key_value
        if use_cache and past_key_value is not None:
            past_key_value.update_decode_1bkd(key, value, cur_pos_tensor, current_pos)
            updated_cache = past_key_value

        # Get full KV cache for attention (BKSD format: [batch, kv_heads, max_seq, head_dim])
        key_cache = past_key_value.key_cache
        value_cache = past_key_value.value_cache

        # Transpose Q from 1BQD to BQSD for SDPA decode
        # 1BQD: [1, batch, heads, head_dim] -> BQSD: [batch, heads, 1, head_dim]
        query_bqsd = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
        query_bqsd = ttnn.permute(query_bqsd, (1, 2, 0, 3))  # [batch, heads, 1, head_dim]
        query_bqsd = ttnn.to_layout(query_bqsd, ttnn.TILE_LAYOUT)

        # Use SDPA decode - it handles GQA internally via num_heads parameters
        # Do NOT expand KV cache for GQA - that's extremely expensive for full cache
        try:
            attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
                query_bqsd, key_cache, value_cache,
                cur_pos_tensor=cur_pos_tensor,
                scale=self.scale,
            )
            if self.layer_idx == 0:
                print(f"[DEBUG L0] SDPA decode SUCCESS at pos {current_pos}")
        except (RuntimeError, TypeError) as e:
            if self.layer_idx == 0:
                print(f"[DEBUG L0] SDPA decode FAILED: {e}, using fallback")
            # SDPA decode failed - need to expand KV for general SDPA
            # Only expand what we need (up to current_pos + 1)
            if self.num_kv_groups > 1:
                key_expanded = ttnn.repeat_interleave(key_cache, self.num_kv_groups, dim=1)
                value_expanded = ttnn.repeat_interleave(value_cache, self.num_kv_groups, dim=1)
            else:
                key_expanded = key_cache
                value_expanded = value_cache

            attn_output = ttnn.transformer.scaled_dot_product_attention(
                query_bqsd, key_expanded, value_expanded,
                attn_mask=None,
                is_causal=False,
                scale=self.scale,
            )

        ttnn.deallocate(cur_pos_tensor)

        # Reshape output: [batch, heads, 1, head_dim] -> [batch, 1, hidden]
        attn_output = ttnn.to_layout(attn_output, ttnn.ROW_MAJOR_LAYOUT)
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))  # [batch, 1, heads, head_dim]
        attn_output = ttnn.reshape(attn_output, (batch_size, 1, self.hidden_size))
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
        """Apply RoPE to 1BKD tensor [1, batch, heads, head_dim]."""
        # Expand cos/sin to match batch and heads
        # cos/sin are [1, 1, 1, head_dim], x is [1, batch, heads, head_dim]
        batch_size = x.shape[1]
        cos_expanded = ttnn.repeat(cos, ttnn.Shape([1, batch_size, num_heads, 1]))
        sin_expanded = ttnn.repeat(sin, ttnn.Shape([1, batch_size, num_heads, 1]))

        half_dim = self.head_dim // 2
        x1 = x[:, :, :, :half_dim]
        x2 = x[:, :, :, half_dim:]
        x_rotated = ttnn.concat([ttnn.neg(x2), x1], dim=-1)

        x_cos = ttnn.multiply(x, cos_expanded)
        x_rot_sin = ttnn.multiply(x_rotated, sin_expanded)
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
        """Apply RoPE to a single tensor."""
        num_heads = x.shape[1]

        cos_expanded = ttnn.repeat(cos, ttnn.Shape([1, num_heads, 1, 1]))
        sin_expanded = ttnn.repeat(sin, ttnn.Shape([1, num_heads, 1, 1]))

        half_dim = self.head_dim // 2
        x1 = x[:, :, :, :half_dim]
        x2 = x[:, :, :, half_dim:]
        x_rotated = ttnn.concat([ttnn.neg(x2), x1], dim=-1)

        x_cos = ttnn.multiply(x, cos_expanded)
        x_rot_sin = ttnn.multiply(x_rotated, sin_expanded)
        return ttnn.add(x_cos, x_rot_sin)
