"""
Batch-32 Padded Generator for BitNet with Metal Trace support.

This generator implements the tt-transformers pattern for high-performance decode:
1. Pad batch dimension to 32 (even for single-user inference)
2. Use HEIGHT_SHARDED memory throughout decode
3. Enable Metal Trace for 2-3x speedup

Key insight from tt-transformers (model_config.py:1506-1509):
    if batch < 32:
        zeros = torch.zeros(1, seq_len, 32, self.dim)
        zeros[:, :, :batch, :] = x
        x = zeros

This enables HEIGHT_SHARDED operations that work with trace without allocating buffers.

Performance target: 30+ t/s (currently 186 t/s for attention-only benchmark)
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Optional, Tuple

import numpy as np
import torch
import ttnn
from numpy.typing import NDArray

from bitnet_tt.config import get_compute_kernel_config
from bitnet_tt.layers.attention import KVCache

if TYPE_CHECKING:
    from bitnet_tt.model.bitnet import BitNetModel


PADDED_BATCH = 32
TILE_SIZE = 32


def round_up_to_tile(value: int, tile: int = TILE_SIZE) -> int:
    """Round an integer up to the next tile multiple."""
    return ((value + tile - 1) // tile) * tile


@dataclass
class GenerationStats:
    """Statistics for text generation."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    prompt_time: float = 0.0
    generation_time: float = 0.0
    token_times: list[float] = field(default_factory=list)
    trace_captured: bool = False

    @property
    def total_time(self) -> float:
        return self.prompt_time + self.generation_time

    @property
    def tokens_per_second(self) -> float:
        if self.generation_time > 0:
            return self.generated_tokens / self.generation_time
        return 0.0

    @property
    def time_to_first_token(self) -> float:
        return self.prompt_time

    @property
    def avg_token_time_ms(self) -> float:
        if self.token_times:
            return (sum(self.token_times) / len(self.token_times)) * 1000
        return 0.0

    def __str__(self) -> str:
        trace_str = " (Traced)" if self.trace_captured else ""
        return (
            f"\n[Stats{trace_str}] "
            f"Prompt: {self.prompt_tokens} tokens ({self.prompt_time * 1000:.1f}ms) | "
            f"Generated: {self.generated_tokens} tokens ({self.generation_time:.2f}s) | "
            f"Speed: {self.tokens_per_second:.2f} t/s | "
            f"Avg: {self.avg_token_time_ms:.2f}ms/token"
        )


def pad_batch_to_32(x: torch.Tensor, batch_dim: int = 0) -> torch.Tensor:
    """Pad tensor's batch dimension to 32."""
    shape = list(x.shape)
    batch = shape[batch_dim]
    if batch >= PADDED_BATCH:
        return x

    shape[batch_dim] = PADDED_BATCH
    padded = torch.zeros(shape, dtype=x.dtype)

    slices = [slice(None)] * len(shape)
    slices[batch_dim] = slice(0, batch)
    padded[tuple(slices)] = x
    return padded


def create_batch32_core_grid() -> ttnn.CoreGrid:
    """Create 32-core grid for HEIGHT_SHARDED operations."""
    return ttnn.CoreGrid(y=4, x=8)


def create_height_sharded_config(shard_height: int, head_dim: int) -> ttnn.MemoryConfig:
    """Create HEIGHT_SHARDED memory config for QKV heads."""
    return ttnn.create_sharded_memory_config(
        shape=(shard_height, head_dim),
        core_grid=create_batch32_core_grid(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def create_sdpa_output_sharded_config(num_heads: int, head_dim: int) -> ttnn.MemoryConfig:
    """Create HEIGHT_SHARDED config for SDPA output before nlp_concat_heads_decode."""
    padded_heads = ((num_heads + 31) // 32) * 32
    return ttnn.create_sharded_memory_config(
        shape=(padded_heads, head_dim),
        core_grid=create_batch32_core_grid(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class Batch32RotarySetup:
    """
    RoPE setup optimized for batch-32 HEIGHT_SHARDED decode.

    Key difference from standard RotarySetup:
    - cos/sin are materialized directly as HEIGHT_SHARDED decode inputs
    - Transformation matrix is also sharded
    - All outputs compatible with rotary_embedding_llama is_decode_mode=True
    """

    def __init__(
        self,
        device: ttnn.Device,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float = 500000.0,
    ):
        self.device = device
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta

        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        cos_half = cos[:, : head_dim // 2]
        sin_half = sin[:, : head_dim // 2]
        cos_interleaved = torch.stack([cos_half, cos_half], dim=-1).flatten(-2)
        sin_interleaved = torch.stack([sin_half, sin_half], dim=-1).flatten(-2)
        self._cos_interleaved_host = cos_interleaved.to(torch.bfloat16)
        self._sin_interleaved_host = sin_interleaved.to(torch.bfloat16)

        self.cos_matrix = ttnn.from_torch(
            cos_interleaved,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.sin_matrix = ttnn.from_torch(
            sin_interleaved,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        trans_mat = torch.zeros(1, 1, TILE_SIZE, TILE_SIZE)
        for i in range(TILE_SIZE // 2):
            trans_mat[0, 0, i, i + TILE_SIZE // 2] = -1.0
            trans_mat[0, 0, i + TILE_SIZE // 2, i] = 1.0

        self.trans_mat_config = create_height_sharded_config(TILE_SIZE, TILE_SIZE)

        self.transformation_mat = ttnn.from_torch(
            trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.cos_sin_config = create_height_sharded_config(TILE_SIZE, head_dim)
        self._host_cos_sin_cache: dict[int, Tuple[ttnn.Tensor, ttnn.Tensor]] = {}

    def _get_cos_sin_torch(self, position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build interleaved cos/sin tensors for one decode position."""
        if position < 0 or position >= self.max_seq_len:
            raise ValueError(
                f"Position {position} is out of range for max_seq_len={self.max_seq_len}"
            )

        cos_row = self._cos_interleaved_host[position : position + 1]
        sin_row = self._sin_interleaved_host[position : position + 1]

        cos = cos_row.expand(PADDED_BATCH, -1).unsqueeze(0).unsqueeze(2).contiguous()
        sin = sin_row.expand(PADDED_BATCH, -1).unsqueeze(0).unsqueeze(2).contiguous()
        return cos, sin

    def create_cos_sin_device_tensors(self, position: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Create HEIGHT_SHARDED device tensors for one decode position."""
        cos_torch, sin_torch = self._get_cos_sin_torch(position)
        cos_device = ttnn.from_torch(
            cos_torch,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=self.cos_sin_config,
        )
        sin_device = ttnn.from_torch(
            sin_torch,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=self.cos_sin_config,
        )
        return cos_device, sin_device

    def get_cos_sin_host_tensor(self, position: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Create host tensors that can be copied into persistent trace inputs."""
        cached = self._host_cos_sin_cache.get(position)
        if cached is not None:
            return cached

        cos_torch, sin_torch = self._get_cos_sin_torch(position)
        cos_host = ttnn.from_torch(
            cos_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
        )
        sin_host = ttnn.from_torch(
            sin_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
        )
        self._host_cos_sin_cache[position] = (cos_host, sin_host)
        return cos_host, sin_host

    def get_sharded_trans_mat(self) -> ttnn.Tensor:
        """Get HEIGHT_SHARDED transformation matrix."""
        return ttnn.to_memory_config(self.transformation_mat, self.trans_mat_config)

    def release_host_cache(self) -> None:
        """Release cached host-side cos/sin tensors."""
        for cos_host, sin_host in self._host_cos_sin_cache.values():
            ttnn.deallocate(cos_host)
            ttnn.deallocate(sin_host)
        self._host_cos_sin_cache.clear()


class Batch32KVCache:
    """
    KV Cache optimized for batch-32 HEIGHT_SHARDED decode with Metal Trace.

    Key differences from standard KVCache:
    - Cache shape: [batch=32, num_kv_heads, max_seq_len, head_dim]
    - Uses paged_update_cache with HEIGHT_SHARDED input
    - Compatible with Metal Trace (no allocations during update)
    """

    def __init__(
        self,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: ttnn.Device,
    ):
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.seq_len_cached = 0

        # Allocate cache: [batch=32, kv_heads, max_seq, head_dim]
        cache_shape = (PADDED_BATCH, num_kv_heads, max_seq_len, head_dim)
        zeros = torch.zeros(cache_shape, dtype=torch.bfloat16)

        self.key_cache = ttnn.from_torch(
            zeros,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.value_cache = ttnn.from_torch(
            zeros.clone(),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Sharded config for K/V input (from nlp_create_qkv_heads_decode)
        self.kv_shard_config = create_height_sharded_config(TILE_SIZE, head_dim)

    def update(
        self,
        key_sharded: ttnn.Tensor,
        value_sharded: ttnn.Tensor,
        current_pos_tensor: ttnn.Tensor,
    ) -> None:
        """
        Update cache in-place using paged_update_cache.

        Args:
            key_sharded: HEIGHT_SHARDED K from nlp_create_qkv_heads_decode
            value_sharded: HEIGHT_SHARDED V from nlp_create_qkv_heads_decode
            current_pos_tensor: INT32 tensor with current position
        """
        ttnn.experimental.paged_update_cache(
            self.key_cache, key_sharded, update_idxs_tensor=current_pos_tensor
        )
        ttnn.experimental.paged_update_cache(
            self.value_cache, value_sharded, update_idxs_tensor=current_pos_tensor
        )
        # Note: seq_len_cached is updated externally by generator

    def get_for_attention(self, seq_len: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Get sliced cache for attention computation.

        SDPA decode requires k_chunk_size to be multiple of 32,
        so we always return at least 32 sequence positions.
        """
        min_seq = max(seq_len, 32)
        return (
            self.key_cache[:, :, :min_seq, :],
            self.value_cache[:, :, :min_seq, :],
        )

    def reset(self) -> None:
        """Reset cache position."""
        self.seq_len_cached = 0


class Batch32Generator:
    """
    High-performance text generator using batch-32 padding + Metal Trace.

    Achieves 30+ t/s by:
    1. Padding batch to 32 for HEIGHT_SHARDED compatibility
    2. Using nlp_create_qkv_heads_decode for sharded QKV
    3. Using rotary_embedding_llama with sharded inputs
    4. Capturing Metal Trace for decode loop

    Usage:
        generator = Batch32Generator(model)
        output = generator.generate("Hello, world!", max_new_tokens=50)
    """

    def __init__(
        self,
        model: "BitNetModel",
        tokenizer: Any = None,
        enable_trace: bool = True,
        decode_matmul_fidelity: str = "hifi2",
    ):
        self.model = model
        self.device = model.device
        self.config = model.config
        self.tokenizer = tokenizer
        self.enable_trace = enable_trace
        self.decode_matmul_fidelity = decode_matmul_fidelity

        # Trace state
        self._trace_id: Optional[int] = None
        self._trace_inputs: Optional[dict] = None
        self._trace_output: Optional[ttnn.Tensor] = None
        self._decode_inputs: Optional[dict] = None
        self._embed_host_cache: dict[int, ttnn.Tensor] = {}
        self._pos_host_cache: dict[int, ttnn.Tensor] = {}
        self._decode_matmul_kernel_config = None
        try:
            self._decode_matmul_kernel_config = get_compute_kernel_config(decode_matmul_fidelity)
        except Exception:
            self._decode_matmul_kernel_config = None

        # Batch-32 RoPE setup
        self.rotary_setup = Batch32RotarySetup(
            device=self.device,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_position_embeddings,
            rope_theta=self.config.rope_theta,
        )
        self._transformation_mat = self.rotary_setup.get_sharded_trans_mat()
        self._sdpa_output_sharded_config = create_sdpa_output_sharded_config(
            self.config.num_attention_heads,
            self.config.head_dim,
        )
        for layer in self.model.layers:
            layer.self_attn.transformation_mat_decode = self._transformation_mat

        # Store embedding weight on host for trace execution
        self._embedding_weight_host = ttnn.to_torch(model.embed_tokens.weight)

        # KV caches (allocated per generation)
        self._kv_caches: Optional[list[KVCache]] = None

        if self.tokenizer is None:
            self._load_default_tokenizer()

    def _load_default_tokenizer(self) -> None:
        """Load the default tokenizer for BitNet."""
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except ImportError:
            print("Warning: transformers not installed. Cannot load tokenizer.")
            self.tokenizer = None

    def _allocate_kv_caches(self, max_seq_len: int) -> list[KVCache]:
        """Allocate paged KV caches for all layers using virtual batch=32."""
        max_seq_len = round_up_to_tile(max_seq_len)
        caches = []
        for _ in range(self.config.num_layers):
            cache = KVCache(
                max_seq_len=max_seq_len,
                batch_size=PADDED_BATCH,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                device=self.device,
            )
            cache.preallocate(
                batch_size=PADDED_BATCH,
                num_kv_heads=self.config.num_key_value_heads,
                max_seq_len=max_seq_len,
                head_dim=self.config.head_dim,
                device=self.device,
                use_paged=True,
            )
            caches.append(cache)
        return caches

    def _decode_step_batch32(
        self,
        token_embeds: ttnn.Tensor,
        current_pos: int,
        current_pos_tensor: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Single decode step using batch-32 HEIGHT_SHARDED pipeline.

        This is the core optimized decode that can be traced.

        Args:
            token_embeds: [1, 32, hidden_size] token embeddings (seq=32 virtual batch)
            current_pos: Current decode position
            current_pos_tensor: INT32 tensor for KV cache update
            cos: HEIGHT_SHARDED cos tensor [1, 32, 1, head_dim]
            sin: HEIGHT_SHARDED sin tensor [1, 32, 1, head_dim]

        Returns:
            logits: [1, 1, vocab_size]
        """
        hidden = token_embeds

        for layer_idx in range(self.config.num_layers):
            layer = self.model.layers[layer_idx]
            kv_cache = self._kv_caches[layer_idx]
            prev_hidden = hidden

            # Input LayerNorm
            normed = layer.input_layernorm(hidden)

            # Reshape for fused QKV: [1, 32, hidden] -> [1, 1, 32, hidden]
            # Use [1, 32, hidden] instead of [32, 1, hidden] to avoid layout conversion hang
            normed_rm = ttnn.to_layout(normed, ttnn.ROW_MAJOR_LAYOUT)
            normed_1bkd = ttnn.reshape(normed_rm, (1, 1, PADDED_BATCH, self.config.hidden_size))
            normed_1bkd = ttnn.to_layout(normed_1bkd, ttnn.TILE_LAYOUT)

            # Fused QKV matmul
            attention = layer.self_attn
            if self._decode_matmul_kernel_config is not None:
                qkv_fused = ttnn.matmul(
                    normed_1bkd,
                    attention.qkv_fused_weight,
                    compute_kernel_config=self._decode_matmul_kernel_config,
                )
            else:
                qkv_fused = ttnn.matmul(normed_1bkd, attention.qkv_fused_weight)
            attn_output_proj, _ = attention._forward_decode_1bkd(
                qkv_fused,
                kv_cache,
                current_pos,
                (cos, sin),
                PADDED_BATCH,
                current_pos_tensor=current_pos_tensor,
            )
            ttnn.deallocate(normed)

            attn_output_rm = ttnn.to_layout(attn_output_proj, ttnn.ROW_MAJOR_LAYOUT)
            attn_output_3d = ttnn.reshape(
                attn_output_rm,
                (1, PADDED_BATCH, self.config.hidden_size),
            )
            attn_output_3d = ttnn.to_layout(attn_output_3d, ttnn.TILE_LAYOUT)
            ttnn.deallocate(attn_output_proj)
            ttnn.deallocate(attn_output_rm)

            # Residual
            hidden_attn = ttnn.add(hidden, attn_output_3d)
            ttnn.deallocate(attn_output_3d)

            # FFN
            residual = hidden_attn
            hidden_normed = layer.post_attention_layernorm(hidden_attn)
            hidden_mlp = layer.mlp(hidden_normed, mode="decode")
            hidden = ttnn.add(residual, hidden_mlp)
            ttnn.deallocate(hidden_normed)
            ttnn.deallocate(hidden_mlp)
            ttnn.deallocate(residual)
            if layer_idx > 0:
                ttnn.deallocate(prev_hidden)

            # Cleanup intermediate tensors
            ttnn.deallocate(normed_rm)
            ttnn.deallocate(normed_1bkd)

        # Final norm + LM head
        hidden = self.model.norm(hidden)
        # Only batch row 0 contributes to user-visible output. Running the LM
        # head on all 32 padded rows adds unnecessary decode work.
        hidden_single = ttnn.slice(
            hidden,
            [0, 0, 0],
            [1, 1, self.config.hidden_size],
        )
        if self._decode_matmul_kernel_config is not None:
            logits = ttnn.matmul(
                hidden_single,
                self.model.lm_head_weight,
                compute_kernel_config=self._decode_matmul_kernel_config,
            )
        else:
            logits = ttnn.matmul(hidden_single, self.model.lm_head_weight)
        ttnn.deallocate(hidden_single)

        return logits

    def _set_kv_cache_length(self, seq_len: int) -> None:
        """Keep batch-32 KV cache metadata aligned with the most recent decode position."""
        if self._kv_caches is None:
            return
        for cache in self._kv_caches:
            cache.seq_len_cached = seq_len

    def _allocate_decode_inputs(self) -> dict:
        """Pre-allocate persistent decode input tensors on device."""
        embed_shape = (1, PADDED_BATCH, self.config.hidden_size)
        pos_shape = (PADDED_BATCH,)

        embeds = ttnn.from_torch(
            torch.zeros(embed_shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        pos_tensor = ttnn.from_torch(
            torch.zeros(pos_shape, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cos, sin = self.rotary_setup.create_cos_sin_device_tensors(position=0)

        return {
            "embeds": embeds,
            "pos_tensor": pos_tensor,
            "cos": cos,
            "sin": sin,
        }

    def _get_embed_host_tensor(self, token_id: int) -> ttnn.Tensor:
        """Get cached host tensor for one decoded token embedding."""
        cached = self._embed_host_cache.get(token_id)
        if cached is not None:
            return cached

        embed_padded = torch.zeros((1, PADDED_BATCH, self.config.hidden_size), dtype=torch.bfloat16)
        embed_padded[0, 0, :] = self._embedding_weight_host[token_id]
        embed_host = ttnn.from_torch(
            embed_padded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
        )
        self._embed_host_cache[token_id] = embed_host
        return embed_host

    def _get_pos_host_tensor(self, current_pos: int) -> ttnn.Tensor:
        """Get cached host tensor for one decode position."""
        cached = self._pos_host_cache.get(current_pos)
        if cached is not None:
            return cached

        pos_host = ttnn.from_torch(
            torch.full((PADDED_BATCH,), current_pos, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None,
        )
        self._pos_host_cache[current_pos] = pos_host
        return pos_host

    def _clear_host_decode_cache(self) -> None:
        """Release cached host tensors used to feed decode inputs."""
        for tensor in self._embed_host_cache.values():
            ttnn.deallocate(tensor)
        self._embed_host_cache.clear()

        for tensor in self._pos_host_cache.values():
            ttnn.deallocate(tensor)
        self._pos_host_cache.clear()

        self.rotary_setup.release_host_cache()

    def _copy_decode_inputs(self, inputs: dict, token_id: int, current_pos: int) -> None:
        """Update persistent decode inputs with the next token and position."""
        embed_host = self._get_embed_host_tensor(token_id)
        ttnn.copy_host_to_device_tensor(embed_host, inputs["embeds"])

        pos_host = self._get_pos_host_tensor(current_pos)
        ttnn.copy_host_to_device_tensor(pos_host, inputs["pos_tensor"])

        cos_host, sin_host = self.rotary_setup.get_cos_sin_host_tensor(current_pos)
        ttnn.copy_host_to_device_tensor(cos_host, inputs["cos"])
        ttnn.copy_host_to_device_tensor(sin_host, inputs["sin"])

    def _ensure_decode_inputs(self) -> dict:
        """Lazily allocate reusable non-trace decode inputs."""
        if self._decode_inputs is None:
            self._decode_inputs = self._allocate_decode_inputs()
        return self._decode_inputs

    def _capture_trace(self, token_id: int, current_pos: int) -> None:
        """Capture Metal Trace for decode loop."""
        if self._trace_id is not None:
            return

        self._trace_inputs = self._allocate_decode_inputs()
        self._copy_decode_inputs(self._trace_inputs, token_id, current_pos)
        saved_seq_len = [cache.seq_len_cached for cache in self._kv_caches or []]

        warmup_logits = self._decode_step_batch32(
            self._trace_inputs["embeds"],
            current_pos,
            self._trace_inputs["pos_tensor"],
            self._trace_inputs["cos"],
            self._trace_inputs["sin"],
        )
        self._set_kv_cache_length(current_pos + 1)
        ttnn.deallocate(warmup_logits)
        for cache, saved_len in zip(self._kv_caches or [], saved_seq_len):
            cache.seq_len_cached = saved_len
        ttnn.synchronize_device(self.device)

        self._trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)

        self._trace_output = self._decode_step_batch32(
            self._trace_inputs["embeds"],
            current_pos,
            self._trace_inputs["pos_tensor"],
            self._trace_inputs["cos"],
            self._trace_inputs["sin"],
        )
        self._set_kv_cache_length(current_pos + 1)

        ttnn.end_trace_capture(self.device, self._trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

    def _execute_trace(
        self,
        token_id: int,
        current_pos: int,
    ) -> ttnn.Tensor:
        """Execute captured trace with new inputs."""
        self._copy_decode_inputs(self._trace_inputs, token_id, current_pos)

        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)
        self._set_kv_cache_length(current_pos + 1)

        return self._trace_output

    def _execute_decode_untraced(
        self,
        token_id: int,
        current_pos: int,
    ) -> ttnn.Tensor:
        """Run one decode step using reusable device inputs without trace capture."""
        decode_inputs = self._ensure_decode_inputs()
        self._copy_decode_inputs(decode_inputs, token_id, current_pos)
        logits = self._decode_step_batch32(
            decode_inputs["embeds"],
            current_pos,
            decode_inputs["pos_tensor"],
            decode_inputs["cos"],
            decode_inputs["sin"],
        )
        self._set_kv_cache_length(current_pos + 1)
        return logits

    def _release_trace(self) -> None:
        """Release trace resources."""
        if self._trace_id is not None:
            ttnn.release_trace(self.device, self._trace_id)
            self._trace_id = None

        if self._trace_inputs is not None:
            for tensor in self._trace_inputs.values():
                ttnn.deallocate(tensor)
            self._trace_inputs = None

        self._trace_output = None

    def _release_decode_inputs(self) -> None:
        """Release reusable non-trace decode input tensors."""
        if self._decode_inputs is not None:
            for tensor in self._decode_inputs.values():
                ttnn.deallocate(tensor)
            self._decode_inputs = None

    def _prefill_batch32(
        self,
        tokens: NDArray[np.int64],
    ) -> Tuple[ttnn.Tensor, int]:
        """
        Prefill phase - run with original batch size, then transfer to batch32 cache.

        Args:
            tokens: [1, seq_len] token IDs

        Returns:
            (logits, seq_len)
        """
        batch_size, seq_len = tokens.shape

        tokens_tt = ttnn.from_torch(
            torch.from_numpy(tokens.astype(np.int64)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        logits, model_kv_caches = self.model(
            input_ids=tokens_tt,
            past_key_values=None,
            use_cache=True,
            mode="prefill",
        )

        self._transfer_prefill_to_batch32_cache(model_kv_caches, seq_len)

        ttnn.deallocate(tokens_tt)

        return logits, seq_len

    def _transfer_prefill_to_batch32_cache(
        self,
        model_kv_caches: list,
        seq_len: int,
    ) -> None:
        num_q_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        num_kv_groups = num_q_heads // num_kv_heads

        for layer_idx, model_cache in enumerate(model_kv_caches):
            if model_cache is None:
                continue

            batch32_cache = self._kv_caches[layer_idx]

            if hasattr(model_cache, "key_cache") and model_cache.key_cache is not None:
                k_prefill = model_cache.key_cache
                v_prefill = model_cache.value_cache

                k_torch = ttnn.to_torch(k_prefill)
                v_torch = ttnn.to_torch(v_prefill)
                if k_torch.shape[1] == num_q_heads:
                    k_torch = k_torch[:, ::num_kv_groups, :, :].contiguous()
                elif k_torch.shape[1] != num_kv_heads:
                    raise ValueError(
                        f"Unexpected prefill K head count: {k_torch.shape[1]} "
                        f"(expected {num_q_heads} or {num_kv_heads})"
                    )

                if v_torch.shape[1] == num_q_heads:
                    v_torch = v_torch[:, ::num_kv_groups, :, :].contiguous()
                elif v_torch.shape[1] != num_kv_heads:
                    raise ValueError(
                        f"Unexpected prefill V head count: {v_torch.shape[1]} "
                        f"(expected {num_q_heads} or {num_kv_heads})"
                    )

                k_padded = torch.zeros(
                    (
                        PADDED_BATCH,
                        batch32_cache.key_cache.shape[1],
                        batch32_cache.max_seq_len,
                        self.config.head_dim,
                    ),
                    dtype=torch.bfloat16,
                )
                v_padded = torch.zeros_like(k_padded)

                actual_batch = min(k_torch.shape[0], PADDED_BATCH)
                actual_heads = min(k_torch.shape[1], self.config.num_key_value_heads)
                actual_seq = min(k_torch.shape[2], seq_len)

                k_padded[:actual_batch, :actual_heads, :actual_seq, :] = k_torch[
                    :actual_batch, :actual_heads, :actual_seq, :
                ]
                v_padded[:actual_batch, :actual_heads, :actual_seq, :] = v_torch[
                    :actual_batch, :actual_heads, :actual_seq, :
                ]

                ttnn.deallocate(batch32_cache.key_cache)
                ttnn.deallocate(batch32_cache.value_cache)

                batch32_cache.key_cache = ttnn.from_torch(
                    k_padded,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                batch32_cache.value_cache = ttnn.from_torch(
                    v_padded,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                batch32_cache.seq_len_cached = seq_len

    def _sample_token(
        self,
        logits: ttnn.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ) -> int:
        """Sample next token from the last logical position of batch element 0."""
        # Avoid allocating new device buffers while a trace is active.
        logits_torch = ttnn.to_torch(logits).float()
        last_logits = logits_torch[0, -1, :]
        logits_np = last_logits.numpy()

        if temperature != 1.0:
            logits_np = logits_np / temperature

        if top_k is not None and top_k > 0:
            top_k_indices = np.argpartition(logits_np, -top_k)[-top_k:]
            top_k_logits = logits_np[top_k_indices]
            probs = np.exp(top_k_logits - np.max(top_k_logits))
            probs = probs / probs.sum()
            sampled_idx = np.random.choice(len(top_k_indices), p=probs)
            return int(top_k_indices[sampled_idx])
        else:
            return int(np.argmax(logits_np))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ) -> str:
        """
        Generate text from prompt using batch-32 + trace.

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering

        Returns:
            Generated text including prompt
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]

        # Allocate caches
        max_seq_len = round_up_to_tile(input_ids.shape[1] + max_new_tokens + 32)
        self._kv_caches = self._allocate_kv_caches(max_seq_len)

        # Prefill
        logits, seq_len = self._prefill_batch32(input_ids)

        # Sample first token
        next_token = self._sample_token(logits, temperature, top_k)
        generated_ids = list(input_ids[0]) + [next_token]

        if next_token == self.tokenizer.eos_token_id:
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Decode loop
        current_pos = seq_len
        trace_captured = False

        try:
            while len(generated_ids) - input_ids.shape[1] < max_new_tokens:
                logits_owned = True

                if self.enable_trace and trace_captured:
                    logits = self._execute_trace(next_token, current_pos)
                    logits_owned = False
                else:
                    logits = self._execute_decode_untraced(next_token, current_pos)

                next_token = self._sample_token(logits, temperature, top_k)
                generated_ids.append(next_token)
                current_pos += 1

                if (
                    self.enable_trace
                    and not trace_captured
                    and next_token != self.tokenizer.eos_token_id
                    and len(generated_ids) - input_ids.shape[1] < max_new_tokens
                ):
                    self._capture_trace(next_token, current_pos)
                    trace_captured = True

                    next_token = self._sample_token(self._trace_output, temperature, top_k)
                    generated_ids.append(next_token)
                    current_pos += 1

                if logits_owned:
                    ttnn.deallocate(logits)

                if next_token == self.tokenizer.eos_token_id:
                    break

        finally:
            if self.enable_trace:
                self._release_trace()
            self._release_decode_inputs()
            self._clear_host_decode_cache()

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ) -> Generator[Tuple[str, GenerationStats], None, None]:
        """
        Generate text with streaming output.

        Yields tokens as they are generated along with statistics.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        stats = GenerationStats()

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        stats.prompt_tokens = input_ids.shape[1]

        # Allocate caches
        max_seq_len = round_up_to_tile(input_ids.shape[1] + max_new_tokens + 32)
        self._kv_caches = self._allocate_kv_caches(max_seq_len)

        # Prefill
        prefill_start = time.perf_counter()
        logits, seq_len = self._prefill_batch32(input_ids)
        stats.prompt_time = time.perf_counter() - prefill_start

        # Sample first token
        next_token = self._sample_token(logits, temperature, top_k)
        generated_ids = list(input_ids[0]) + [next_token]
        stats.generated_tokens = 1

        # Yield first token
        current_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        prev_len = len(self.tokenizer.decode(list(input_ids[0]), skip_special_tokens=True))
        yield current_text[prev_len:], stats

        if next_token == self.tokenizer.eos_token_id:
            return

        # Decode loop
        current_pos = seq_len
        prev_text_len = len(current_text)
        trace_captured = False

        try:
            while stats.generated_tokens < max_new_tokens:
                token_start = time.perf_counter()

                logits_owned = True

                if self.enable_trace and trace_captured:
                    logits = self._execute_trace(next_token, current_pos)
                    logits_owned = False
                else:
                    logits = self._execute_decode_untraced(next_token, current_pos)

                token_time = time.perf_counter() - token_start
                stats.token_times.append(token_time)
                stats.generation_time += token_time

                next_token = self._sample_token(logits, temperature, top_k)
                generated_ids.append(next_token)
                stats.generated_tokens += 1
                current_pos += 1

                if (
                    self.enable_trace
                    and not trace_captured
                    and next_token != self.tokenizer.eos_token_id
                    and stats.generated_tokens < max_new_tokens
                ):
                    capture_start = time.perf_counter()
                    self._capture_trace(next_token, current_pos)
                    trace_captured = True
                    stats.trace_captured = True
                    capture_time = time.perf_counter() - capture_start
                    stats.token_times.append(capture_time)
                    stats.generation_time += capture_time

                    next_token = self._sample_token(self._trace_output, temperature, top_k)
                    generated_ids.append(next_token)
                    stats.generated_tokens += 1
                    current_pos += 1

                current_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                new_text = current_text[prev_text_len:]
                prev_text_len = len(current_text)
                yield new_text, stats

                if logits_owned:
                    ttnn.deallocate(logits)

                if next_token == self.tokenizer.eos_token_id:
                    break

        finally:
            if self.enable_trace:
                self._release_trace()
            self._release_decode_inputs()
            self._clear_host_decode_cache()

    def reset(self) -> None:
        """Reset generator state."""
        if self._trace_id is not None:
            try:
                self._release_trace()
            except Exception:
                self._trace_id = None
                self._trace_inputs = None
                self._trace_output = None

        try:
            self._release_decode_inputs()
        except Exception:
            self._decode_inputs = None

        try:
            self._clear_host_decode_cache()
        except Exception:
            self._embed_host_cache.clear()
            self._pos_host_cache.clear()
            self.rotary_setup._host_cos_sin_cache.clear()

        if self._kv_caches is not None:
            for cache in self._kv_caches:
                cache.reset()
