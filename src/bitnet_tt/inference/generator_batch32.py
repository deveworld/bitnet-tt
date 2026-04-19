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

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Optional, Tuple

import numpy as np
import torch
import ttnn

def _flatten_first(x):
    while isinstance(x, list):
        x = x[0] if x else 0
    return x
from numpy.typing import NDArray

from bitnet_tt.config import get_compute_kernel_config
from bitnet_tt.layers.attention import KVCache, fill_cache_range

if TYPE_CHECKING:
    from bitnet_tt.model.bitnet import BitNetModel


PADDED_BATCH = 32
TILE_SIZE = 32
TRACE_CACHE_BUCKET = 64
MIN_TRACE_CACHE_SEQ_LEN = 128
TRACE_CACHE_SLACK_TOKENS = 1
MAX_DECODE_EMBED_CACHE = 128
MAX_DECODE_POS_CACHE = 256


def round_up_to_tile(value: int, tile: int = TILE_SIZE) -> int:
    """Round an integer up to the next tile multiple."""
    return ((value + tile - 1) // tile) * tile


def choose_trace_cache_seq_len(
    requested_seq_len: int,
    bucket: int = TRACE_CACHE_BUCKET,
    min_seq_len: int = MIN_TRACE_CACHE_SEQ_LEN,
) -> int:
    """
    Bucket cache capacity so nearby requests reuse the same trace programs.

    Small prompt sizes otherwise bounce between capacities like 96 and 128,
    which forces another trace warmup/compile pass even though the active
    decode length is nearly identical.
    """
    rounded = round_up_to_tile(requested_seq_len)
    rounded = max(rounded, min_seq_len)
    return round_up_to_tile(rounded, bucket)


def choose_single_user_cache_seq_len(requested_seq_len: int) -> int:
    """Use exact tile-aligned capacity for single-user batch32 decode throughput."""
    return choose_trace_cache_seq_len(
        requested_seq_len,
        bucket=TILE_SIZE,
        min_seq_len=0,
    )


def _on_device_argmax_single(logits: ttnn.Tensor) -> int:
    """Return the argmax token id while transferring only the final index."""
    try:
        token_indices = ttnn.argmax(logits, dim=-1)
        token_id = int(_flatten_first(token_indices.cpu().to_list()))
        ttnn.deallocate(token_indices)
        return token_id
    except Exception:
        # Older layouts may still require the conservative row-major -> tile path.
        logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        last_logits = logits_rm[:, -1:, :]
        last_logits_tile = ttnn.to_layout(last_logits, ttnn.TILE_LAYOUT)
        token_indices = ttnn.argmax(last_logits_tile, dim=-1)
        token_id = int(_flatten_first(token_indices.cpu().to_list()))
        ttnn.deallocate(logits_rm)
        ttnn.deallocate(last_logits)
        ttnn.deallocate(last_logits_tile)
        ttnn.deallocate(token_indices)
        return token_id


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


@dataclass
class WarmupStats:
    """Statistics for startup warmup / precompile passes."""

    prompt_tokens: int = 0
    cache_seq_len: int = 0
    prompt_time: float = 0.0
    sample_time: float = 0.0
    decode_time: float = 0.0
    trace_captured: bool = False

    @property
    def total_time(self) -> float:
        return self.prompt_time + self.sample_time + self.decode_time


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
        use_fused_rope: bool = True,
    ):
        self.device = device
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.use_fused_rope = use_fused_rope

        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        if use_fused_rope:
            # Interleaved: [f0, f0, f1, f1, ...] for adjacent-pair rotation
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
        else:
            # HF duplicated-halves: [f0, f1, ..., f63, f0, f1, ..., f63]
            emb = torch.cat([freqs, freqs], dim=-1)

        cos_full = emb.cos()  # [max_seq, head_dim] interleaved (TT format)
        sin_full = emb.sin()
        self._cos_host = cos_full.to(torch.bfloat16)
        self._sin_host = sin_full.to(torch.bfloat16)

        self.cos_matrix = ttnn.from_torch(
            cos_full,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.sin_matrix = ttnn.from_torch(
            sin_full,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Adjacent-pair swap: matches tt-transformers get_rot_transformation_mat
        # and the interleaved cos/sin layout expected by rotary_embedding_llama.
        trans_mat = torch.zeros(1, 1, TILE_SIZE, TILE_SIZE)
        trans_mat[..., torch.arange(0, TILE_SIZE, 2), torch.arange(1, TILE_SIZE, 2)] = 1
        trans_mat[..., torch.arange(1, TILE_SIZE, 2), torch.arange(0, TILE_SIZE, 2)] = -1

        self.trans_mat_config = create_height_sharded_config(TILE_SIZE, TILE_SIZE)

        self.transformation_mat = ttnn.from_torch(
            trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.cos_sin_config = create_height_sharded_config(TILE_SIZE, head_dim)
        self._host_cos_sin_cache: OrderedDict[int, Tuple[ttnn.Tensor, ttnn.Tensor]] = OrderedDict()

    def _get_cos_sin_torch(self, position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build interleaved cos/sin for one decode position.

        Returns [1, 1, PADDED_BATCH, head_dim] tensors for HEIGHT_SHARDED decode.
        The position's cos/sin are broadcast (repeated) across the batch dimension.
        """
        if position < 0 or position >= self.max_seq_len:
            raise ValueError(
                f"Position {position} is out of range for max_seq_len={self.max_seq_len}"
            )

        cos_row = self._cos_host[position]  # [head_dim]
        sin_row = self._sin_host[position]
        if self.use_fused_rope:
            # [1, PADDED_BATCH, 1, head_dim] for rotary_embedding_llama decode mode
            cos = cos_row.view(1, 1, 1, -1).expand(1, PADDED_BATCH, 1, -1).contiguous()
            sin = sin_row.view(1, 1, 1, -1).expand(1, PADDED_BATCH, 1, -1).contiguous()
        else:
            # [1, 1, 1, head_dim] for manual RoPE broadcast
            cos = cos_row.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            sin = sin_row.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return cos, sin

    def create_cos_sin_device_tensors(self, position: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Create DRAM device tensors for one decode position.

        Shape: [1, PADDED_BATCH, 1, head_dim]. Sharding to HEIGHT_SHARDED
        happens inside the decode forward (so it's part of the trace).
        """
        cos_torch, sin_torch = self._get_cos_sin_torch(position)
        cos_device = ttnn.from_torch(
            cos_torch,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin_device = ttnn.from_torch(
            sin_torch,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return cos_device, sin_device

    def get_cos_sin_host_tensor(self, position: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Create host tensors that can be copied into persistent trace inputs."""
        cached = self._host_cos_sin_cache.get(position)
        if cached is not None:
            self._host_cos_sin_cache.move_to_end(position)
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
        while len(self._host_cos_sin_cache) >= MAX_DECODE_POS_CACHE:
            _, (old_cos, old_sin) = self._host_cos_sin_cache.popitem(last=False)
            ttnn.deallocate(old_cos)
            ttnn.deallocate(old_sin)

        self._host_cos_sin_cache[position] = (cos_host, sin_host)
        return cos_host, sin_host

    def lookup_decode_cos_sin(
        self, pos_tensor_u32: ttnn.Tensor
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Look up cos/sin for a batch of decode positions directly from
        the pre-uploaded cos_matrix / sin_matrix via ttnn.embedding. This
        is the device-side alternative to get_cos_sin_host_tensor +
        per-step copy_host_to_device_tensor — the lookup can live inside
        a captured trace, eliminating two H2D dispatches per decode step.

        Args:
            pos_tensor_u32: [batch] uint32 tensor of positions (all equal
                in batch-32 decode). Already on device.

        Returns:
            (cos, sin) each shaped [1, batch, head_dim] TILE_LAYOUT,
            ready to be reshaped to [1, batch, 1, head_dim] and
            resharded into cos_sin_config by the caller.
        """
        cos = ttnn.embedding(pos_tensor_u32, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(pos_tensor_u32, self.sin_matrix, layout=ttnn.TILE_LAYOUT)
        return cos, sin

    def get_sharded_trans_mat(self) -> ttnn.Tensor:
        """Get HEIGHT_SHARDED transformation matrix for rotary_embedding_llama decode mode."""
        return ttnn.to_memory_config(self.transformation_mat, self.trans_mat_config)

    def get_trans_mat_for_decode(self) -> ttnn.Tensor:
        """Get transformation matrix in correct config for decode rotary_embedding_llama."""
        if not hasattr(self, '_cached_sharded_trans_mat'):
            self._cached_sharded_trans_mat = self.get_sharded_trans_mat()
        return self._cached_sharded_trans_mat

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
        # BITNET_DECODE_MATMUL_FIDELITY env overrides the constructor arg
        # (e.g. "hifi4") so experiments can sweep without touching the
        # bench scripts.
        self.decode_matmul_fidelity = os.environ.get(
            "BITNET_DECODE_MATMUL_FIDELITY", decode_matmul_fidelity
        )

        # Trace state
        self._trace_id: Optional[int] = None
        self._trace_inputs: Optional[dict] = None
        self._trace_output: Optional[ttnn.Tensor] = None
        self._trace_argmax_output: Optional[ttnn.Tensor] = None
        self._trace_capture_pos: Optional[int] = None
        self._warmed_trace_keys: set[tuple[int, int]] = set()
        self._decode_inputs: Optional[dict] = None
        self._argmax_output_tensors: list[tuple[tuple[int, ...], ttnn.Tensor]] = []
        self._logits_host_tensor: Optional[ttnn.Tensor] = None
        self._logits_host_tensors: list[tuple[Any, ttnn.Tensor]] = []
        self._cached_prefill_key: Optional[tuple[int, ...]] = None
        self._cached_prefill_seq_len: int = 0
        self._cached_prefill_logits: Optional[ttnn.Tensor] = None
        self._cached_prefill_cache_seq_len: int = 0
        self._supports_host_tensor_to_torch: Optional[bool] = None
        self._embed_host_cache: OrderedDict[int, ttnn.Tensor] = OrderedDict()
        self._pos_host_cache: OrderedDict[int, ttnn.Tensor] = OrderedDict()
        self._decode_matmul_kernel_config = None
        try:
            self._decode_matmul_kernel_config = get_compute_kernel_config(decode_matmul_fidelity)
        except Exception:
            self._decode_matmul_kernel_config = None

        # Batch-32 RoPE setup
        use_fused_rope = self.model.layers[0].self_attn.use_fused_rope if self.model.layers else True
        self.rotary_setup = Batch32RotarySetup(
            device=self.device,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_position_embeddings,
            rope_theta=self.config.rope_theta,
            use_fused_rope=use_fused_rope,
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

        # Enable the sharded multicore rms_norm path on every non-fused
        # RMSNorm in the model. ttnn.rms_norm dispatches a single-core
        # kernel on interleaved input (~56 us / call on [1,1,32,2560]);
        # with a width-sharded input it drops to ~33 us / call. Per decode
        # step we have 30x post_attention_layernorm + 30x attn_sub_norm
        # + 30x ffn_sub_norm + 1x final_norm = 91 calls, so the per-call
        # saving compounds.
        for layer in self.model.layers:
            layer.post_attention_layernorm.enable_sharded_fast_path()
            layer.self_attn.attn_sub_norm.enable_sharded_fast_path()
            layer.mlp.ffn_sub_norm.enable_sharded_fast_path()
        self.model.norm.enable_sharded_fast_path()

        # KV caches (allocated per generation)
        self._kv_caches: Optional[list[KVCache]] = None

        # Device-side cos/sin lookup is on by default; set
        # BITNET_DECODE_COS_SIN_LOOKUP=0 to revert to the old per-step
        # H2D copy path.
        self._use_cos_sin_lookup = os.environ.get(
            "BITNET_DECODE_COS_SIN_LOOKUP", "1"
        ) not in ("0", "false", "False")

        # Device-side embed lookup (BITNET_EMBED_DEVICE_LOOKUP, default
        # off). Measured +0.31 t/s but shifted greedy match/32 from 6
        # to 1 because the device embed keeps fp32 precision one op
        # longer than the host bf16 round-trip. Full writeup in
        # docs/session_3_ceiling_analysis.md.
        self._use_embed_lookup = os.environ.get(
            "BITNET_EMBED_DEVICE_LOOKUP", "0"
        ) not in ("0", "false", "False")
        self._token_id_host_cache: OrderedDict[int, ttnn.Tensor] = OrderedDict()

        if self.tokenizer is None:
            self._load_default_tokenizer()

    def _load_default_tokenizer(self) -> None:
        """Load the default tokenizer for BitNet."""
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except ImportError:
            print("Warning: transformers not installed. Cannot load tokenizer.")
            self.tokenizer = None

    def _default_warmup_token_id(self) -> int:
        """Choose a deterministic token id for synthetic warmup prompts."""
        for attr in ("eos_token_id", "pad_token_id", "bos_token_id"):
            token_id = getattr(self.tokenizer, attr, None)
            if token_id is not None:
                return int(token_id)
        return 0

    def _build_warmup_input_ids(
        self,
        prompt: Optional[str],
        prompt_tokens: int,
        token_id: Optional[int],
    ) -> NDArray[np.int64]:
        """Build prompt token ids for startup warmup."""
        if prompt is not None:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not loaded")
            input_ids = self.tokenizer(prompt, return_tensors="np")["input_ids"]
            if input_ids.size == 0 or input_ids.shape[1] == 0:
                raise ValueError("Warmup prompt must contain at least one token")
            return input_ids

        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive when prompt is omitted")
        warm_token_id = self._default_warmup_token_id() if token_id is None else int(token_id)
        return np.full((1, prompt_tokens), warm_token_id, dtype=np.int64)

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
            token_embeds: [1, 1, 32, hidden_size] token embeddings in 1BKD format
            current_pos: Current decode position
            current_pos_tensor: INT32 tensor for KV cache update
            cos: HEIGHT_SHARDED cos tensor [1, 32, 1, head_dim]
            sin: HEIGHT_SHARDED sin tensor [1, 32, 1, head_dim]

        Returns:
            logits: [1, 1, vocab_size]
        """
        hidden = token_embeds

        # Pre-shard cos/sin ONCE per step (fused RoPE path). Every layer
        # shares the same _rope_cos_sin_config, so the per-layer
        # ttnn.to_memory_config inside _forward_decode_1bkd was running
        # the same conversion 30 times in a row. Hoisting saves 60 dispatched
        # ops per decode step. Caller owns the lifetime and deallocates
        # after the layer loop.
        first_attn = self.model.layers[0].self_attn
        head_dim = self.config.head_dim
        # When the device-side LUT is enabled, derive cos/sin from
        # pos_tensor via ttnn.embedding inside the captured trace. This
        # kills the two per-step H2D copies that would otherwise refresh
        # the cos/sin slots before every execute_trace.
        if self._use_cos_sin_lookup:
            pos_u32 = ttnn.typecast(current_pos_tensor, ttnn.uint32)
            cos_raw, sin_raw = self.rotary_setup.lookup_decode_cos_sin(pos_u32)
            ttnn.deallocate(pos_u32)
            cos = ttnn.reshape(cos_raw, (1, PADDED_BATCH, 1, head_dim))
            sin = ttnn.reshape(sin_raw, (1, PADDED_BATCH, 1, head_dim))
        if first_attn.use_fused_rope:
            cos_shared = ttnn.to_memory_config(cos, first_attn._rope_cos_sin_config)
            sin_shared = ttnn.to_memory_config(sin, first_attn._rope_cos_sin_config)
        else:
            cos_shared = cos
            sin_shared = sin

        # L1 memory config for norm→matmul intermediates. Keeping the norm
        # output in L1 avoids a DRAM round-trip: the downstream ternary_matmul
        # reads activation from L1 instead of DRAM (lower latency, no BW
        # contention with weight reads). Decode tensors are small enough
        # (~160 KB interleaved across 110 cores) to fit comfortably.
        _norm_mem = ttnn.L1_MEMORY_CONFIG

        for layer_idx in range(self.config.num_layers):
            layer = self.model.layers[layer_idx]
            kv_cache = self._kv_caches[layer_idx]
            prev_hidden = hidden

            # Fused RMSNorm + QKV matmul
            attention = layer.self_attn
            if attention._qkv_use_packed_ternary:
                qkv_fused = ttnn.experimental.ternary_matmul(
                    hidden,
                    attention.qkv_fused_weight,
                    use_packed_ternary=True,
                    norm_weight=layer.input_layernorm.weight,
                    norm_epsilon=layer.input_layernorm.eps,
                )
            else:
                normed = layer.input_layernorm(hidden, memory_config=_norm_mem)
                if self._decode_matmul_kernel_config is not None:
                    qkv_fused = ttnn.matmul(
                        normed,
                        attention.qkv_fused_weight,
                        compute_kernel_config=self._decode_matmul_kernel_config,
                    )
                else:
                    qkv_fused = ttnn.matmul(normed, attention.qkv_fused_weight)
                ttnn.deallocate(normed)
            attn_output_proj, _ = attention._forward_decode_1bkd(
                qkv_fused,
                kv_cache,
                current_pos,
                (cos_shared, sin_shared),
                PADDED_BATCH,
                current_pos_tensor=current_pos_tensor,
            )

            # Residual
            hidden_attn = ttnn.add(hidden, attn_output_proj)
            ttnn.deallocate(attn_output_proj)

            # FFN
            residual = hidden_attn
            hidden_normed = layer.post_attention_layernorm(hidden_attn, memory_config=_norm_mem)
            hidden_mlp = layer.mlp(hidden_normed, mode="decode")
            hidden = ttnn.add(residual, hidden_mlp)
            ttnn.deallocate(hidden_normed)
            ttnn.deallocate(hidden_mlp)
            ttnn.deallocate(residual)
            if layer_idx > 0:
                ttnn.deallocate(prev_hidden)

        # Release the pre-sharded cos/sin now that all layers are done.
        if first_attn.use_fused_rope:
            ttnn.deallocate(cos_shared)
            ttnn.deallocate(sin_shared)

        # Final norm + LM head
        hidden = self.model.norm(hidden)
        # Only batch row 0 contributes to user-visible output. Running the LM
        # head on all 32 padded rows adds unnecessary decode work.
        hidden_single = ttnn.slice(
            hidden,
            [0, 0, 0, 0],
            [1, 1, 1, self.config.hidden_size],
        )
        # LM head: split along vocab-dim into N chunks to avoid the
        # single-kernel per_core_N blowup that makes the monolithic matmul
        # ~3x slower (microbench: 2.2 ms full vs 0.72 ms with split=4).
        # Keep chunk outputs in L1 — the downstream concat+to_layout+argmax
        # chain runs faster when its inputs aren't on DRAM (-40 us/step).
        lm_head_chunks = getattr(self.model, "lm_head_weight_chunks", None)
        if lm_head_chunks is not None:
            chunk_outs = []
            for w_chunk in lm_head_chunks:
                if self._decode_matmul_kernel_config is not None:
                    out = ttnn.matmul(
                        hidden_single, w_chunk,
                        compute_kernel_config=self._decode_matmul_kernel_config,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                else:
                    out = ttnn.matmul(hidden_single, w_chunk,
                                      memory_config=ttnn.L1_MEMORY_CONFIG)
                chunk_outs.append(out)
            logits = ttnn.concat(chunk_outs, dim=-1)
            for out in chunk_outs:
                ttnn.deallocate(out)
        else:
            if self._decode_matmul_kernel_config is not None:
                logits = ttnn.matmul(
                    hidden_single,
                    self.model.lm_head_weight,
                    compute_kernel_config=self._decode_matmul_kernel_config,
                )
            else:
                logits = ttnn.matmul(hidden_single, self.model.lm_head_weight)
        ttnn.deallocate(hidden_single)
        logits = ttnn.reshape(logits, (1, 1, self.config.vocab_size))

        return logits

    def _set_kv_cache_length(self, seq_len: int) -> None:
        """Keep batch-32 KV cache metadata aligned with the most recent decode position."""
        if self._kv_caches is None:
            return
        for cache in self._kv_caches:
            cache.seq_len_cached = seq_len

    def _clear_prefill_cache(self) -> None:
        """Release cached prompt prefill state."""
        if self._cached_prefill_logits is not None:
            ttnn.deallocate(self._cached_prefill_logits)
        self._cached_prefill_key = None
        self._cached_prefill_seq_len = 0
        self._cached_prefill_logits = None
        self._cached_prefill_cache_seq_len = 0

    def _make_prefill_cache_key(self, input_ids: NDArray[np.int64]) -> tuple[int, ...]:
        """Key prompt-prefill reuse by prompt tokens only."""
        return tuple(map(int, input_ids.reshape(-1)))

    def _try_reuse_prefill(
        self,
        input_ids: NDArray[np.int64],
        max_seq_len: int,
    ) -> Optional[Tuple[ttnn.Tensor, int]]:
        """Reuse cached prompt logits and KV prefix when the prompt matches exactly."""
        if (
            self._cached_prefill_logits is None
            or self._cached_prefill_key is None
            or self._kv_caches is None
            or len(self._kv_caches) == 0
        ):
            return None

        current_cache_capacity = min(cache.max_seq_len for cache in self._kv_caches)
        if current_cache_capacity < max_seq_len:
            return None
        if self._cached_prefill_cache_seq_len != current_cache_capacity:
            return None

        cache_key = self._make_prefill_cache_key(input_ids)
        if cache_key != self._cached_prefill_key:
            return None

        self._set_kv_cache_length(self._cached_prefill_seq_len)
        return self._cached_prefill_logits, self._cached_prefill_seq_len

    def _try_extend_prefill_from_cached_prefix(
        self,
        input_ids: NDArray[np.int64],
        max_seq_len: int,
    ) -> Optional[Tuple[ttnn.Tensor, int]]:
        """Reuse the cached prompt prefix, then decode only the uncached suffix."""
        if (
            self._cached_prefill_logits is None
            or self._cached_prefill_key is None
            or self._kv_caches is None
            or len(self._kv_caches) == 0
        ):
            return None

        current_cache_capacity = min(cache.max_seq_len for cache in self._kv_caches)
        if current_cache_capacity < max_seq_len:
            return None
        if self._cached_prefill_cache_seq_len != current_cache_capacity:
            return None

        prompt_tokens = tuple(map(int, input_ids.reshape(-1)))
        cached_tokens = self._cached_prefill_key
        prefix_len = 0
        max_prefix = min(len(prompt_tokens), len(cached_tokens))
        while prefix_len < max_prefix and prompt_tokens[prefix_len] == cached_tokens[prefix_len]:
            prefix_len += 1

        if prefix_len == 0 or prefix_len == len(prompt_tokens):
            return None

        if self._trace_id is not None:
            self._release_trace()

        self._set_kv_cache_length(prefix_len)
        logits: Optional[ttnn.Tensor] = None
        current_pos = prefix_len
        for token_id in prompt_tokens[prefix_len:]:
            if logits is not None:
                ttnn.deallocate(logits)
            logits = self._execute_decode_untraced(token_id, current_pos)
            current_pos += 1

        if logits is None:
            return None

        self._cache_prefill(input_ids, logits, len(prompt_tokens))
        return logits, len(prompt_tokens)

    def _cache_prefill(
        self,
        input_ids: NDArray[np.int64],
        logits: ttnn.Tensor,
        seq_len: int,
    ) -> None:
        """Keep the last prompt prefill around so repeat generations can skip prefill."""
        cache_key = self._make_prefill_cache_key(input_ids)
        if self._cached_prefill_logits is not None and self._cached_prefill_logits is not logits:
            ttnn.deallocate(self._cached_prefill_logits)
        self._cached_prefill_key = cache_key
        self._cached_prefill_seq_len = seq_len
        self._cached_prefill_logits = logits
        self._cached_prefill_cache_seq_len = (
            min(cache.max_seq_len for cache in self._kv_caches) if self._kv_caches else 0
        )

    def _ensure_kv_caches(self, max_seq_len: int) -> None:
        """Reuse existing batch-32 caches when they are already large enough."""
        if self._kv_caches is None:
            self._kv_caches = self._allocate_kv_caches(max_seq_len)
            return

        if any(cache.max_seq_len < max_seq_len for cache in self._kv_caches):
            self._release_trace()
            self._clear_prefill_cache()
            for cache in self._kv_caches:
                ttnn.deallocate(cache.key_cache)
                ttnn.deallocate(cache.value_cache)
            self._kv_caches = self._allocate_kv_caches(max_seq_len)
            return

        self._set_kv_cache_length(0)

    def _allocate_decode_inputs(self) -> dict:
        """Pre-allocate persistent decode input tensors on device."""
        embed_shape = (1, 1, PADDED_BATCH, self.config.hidden_size)
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

        token_id_tensor = ttnn.from_torch(
            torch.zeros(pos_shape, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cos, sin = self.rotary_setup.create_cos_sin_device_tensors(position=0)

        return {
            "embeds": embeds,
            "pos_tensor": pos_tensor,
            "token_id": token_id_tensor,
            "cos": cos,
            "sin": sin,
        }

    def _get_embed_host_tensor(self, token_id: int) -> ttnn.Tensor:
        """Get cached host tensor for one decoded token embedding."""
        cached = self._embed_host_cache.get(token_id)
        if cached is not None:
            self._embed_host_cache.move_to_end(token_id)
            return cached

        embed_padded = torch.zeros(
            (1, 1, PADDED_BATCH, self.config.hidden_size), dtype=torch.bfloat16
        )
        embed_padded[0, 0, 0, :] = self._embedding_weight_host[token_id]
        embed_host = ttnn.from_torch(
            embed_padded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
        )
        while len(self._embed_host_cache) >= MAX_DECODE_EMBED_CACHE:
            _, old_tensor = self._embed_host_cache.popitem(last=False)
            ttnn.deallocate(old_tensor)

        self._embed_host_cache[token_id] = embed_host
        return embed_host

    def _get_token_id_host_tensor(self, token_id: int) -> ttnn.Tensor:
        """Host [PADDED_BATCH] uint32 tensor with token_id broadcast across
        all rows. Used by the device-side embedding lookup path so the
        per-step H2D shrinks from the full 160 KB embed slab to 128 B."""
        cached = self._token_id_host_cache.get(token_id)
        if cached is not None:
            self._token_id_host_cache.move_to_end(token_id)
            return cached

        token_id_host = ttnn.from_torch(
            torch.full((PADDED_BATCH,), token_id, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None,
        )
        while len(self._token_id_host_cache) >= MAX_DECODE_POS_CACHE:
            _, old_tensor = self._token_id_host_cache.popitem(last=False)
            ttnn.deallocate(old_tensor)

        self._token_id_host_cache[token_id] = token_id_host
        return token_id_host

    def _get_pos_host_tensor(self, current_pos: int) -> ttnn.Tensor:
        """Get cached host tensor for one decode position."""
        cached = self._pos_host_cache.get(current_pos)
        if cached is not None:
            self._pos_host_cache.move_to_end(current_pos)
            return cached

        pos_host = ttnn.from_torch(
            torch.full((PADDED_BATCH,), current_pos, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None,
        )
        while len(self._pos_host_cache) >= MAX_DECODE_POS_CACHE:
            _, old_tensor = self._pos_host_cache.popitem(last=False)
            ttnn.deallocate(old_tensor)

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

        for tensor in self._token_id_host_cache.values():
            ttnn.deallocate(tensor)
        self._token_id_host_cache.clear()

        self.rotary_setup.release_host_cache()

    def _copy_decode_inputs(self, inputs: dict, token_id: int, current_pos: int) -> None:
        """Update persistent decode inputs with the next token and position."""
        if self._use_embed_lookup:
            token_id_host = self._get_token_id_host_tensor(token_id)
            ttnn.copy_host_to_device_tensor(token_id_host, inputs["token_id"])
        else:
            embed_host = self._get_embed_host_tensor(token_id)
            ttnn.copy_host_to_device_tensor(embed_host, inputs["embeds"])

        pos_host = self._get_pos_host_tensor(current_pos)
        ttnn.copy_host_to_device_tensor(pos_host, inputs["pos_tensor"])

        # When cos/sin come from the device-side LUT, skip the two
        # H2D dispatches that would otherwise refresh inputs["cos"] /
        # inputs["sin"]. The trace does the embedding lookup from
        # pos_tensor directly, so the device slots for cos/sin are
        # never read after trace capture.
        if not self._use_cos_sin_lookup:
            cos_host, sin_host = self.rotary_setup.get_cos_sin_host_tensor(current_pos)
            ttnn.copy_host_to_device_tensor(cos_host, inputs["cos"])
            ttnn.copy_host_to_device_tensor(sin_host, inputs["sin"])

    def _resolve_decode_embed(self, inputs: dict) -> ttnn.Tensor:
        """Return the device tensor the decode step should use as
        `token_embeds`. When the embed lookup flag is on we derive the
        [1, 1, 32, hidden] tile-layout tensor from the token_id slot
        via ttnn.embedding against the model's pre-loaded embedding
        weight; otherwise we return the persistent embed slot directly.
        This lets the device-side lookup be captured inside the trace
        with no change to _decode_step_batch32's signature."""
        if not self._use_embed_lookup:
            return inputs["embeds"]
        raw = ttnn.embedding(inputs["token_id"], self.model.embed_tokens.weight)
        tile = ttnn.to_layout(raw, ttnn.TILE_LAYOUT)
        ttnn.deallocate(raw)
        shaped = ttnn.reshape(
            tile, (1, 1, PADDED_BATCH, self.config.hidden_size)
        )
        return shaped

    def _ensure_decode_inputs(self) -> dict:
        """Lazily allocate reusable non-trace decode inputs."""
        if self._decode_inputs is None:
            self._decode_inputs = self._allocate_decode_inputs()
        return self._decode_inputs

    def _ensure_trace_inputs(self) -> dict:
        """Lazily allocate reusable trace input tensors."""
        if self._trace_inputs is None:
            self._trace_inputs = self._allocate_decode_inputs()
        return self._trace_inputs

    def _capture_trace(self, token_id: int, current_pos: int) -> bool:
        """Capture Metal Trace for decode loop. Returns True if a new trace was captured."""
        if self._trace_id is not None and self._trace_capture_pos == current_pos:
            return False
        if self._trace_id is not None:
            self._release_trace()

        self._trace_inputs = self._ensure_trace_inputs()
        self._copy_decode_inputs(self._trace_inputs, token_id, current_pos)
        cache_key = (
            current_pos,
            self._kv_caches[0].max_seq_len if self._kv_caches else 0,
        )
        if cache_key not in self._warmed_trace_keys:
            saved_seq_len = [cache.seq_len_cached for cache in self._kv_caches or []]

            warmup_embed = self._resolve_decode_embed(self._trace_inputs)
            warmup_logits = self._decode_step_batch32(
                warmup_embed,
                current_pos,
                self._trace_inputs["pos_tensor"],
                self._trace_inputs["cos"],
                self._trace_inputs["sin"],
            )
            if self._use_embed_lookup:
                # lookup path allocates a fresh embed per call; release it
                # here (the persistent inputs["embeds"] slot is untouched).
                ttnn.deallocate(warmup_embed)
            # Prime argmax(use_multicore=True) program cache / semaphore state
            # outside trace capture. The op's first call performs host-side
            # writes that are rejected inside begin_trace_capture.
            warmup_logits_rm = ttnn.to_layout(warmup_logits, ttnn.ROW_MAJOR_LAYOUT)
            warmup_argmax = ttnn.argmax(warmup_logits_rm, dim=-1, use_multicore=True)
            ttnn.deallocate(warmup_logits_rm)
            ttnn.deallocate(warmup_argmax)
            self._set_kv_cache_length(current_pos + 1)
            ttnn.deallocate(warmup_logits)
            for cache, saved_len in zip(self._kv_caches or [], saved_seq_len):
                cache.seq_len_cached = saved_len
            ttnn.synchronize_device(self.device)

        self._trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)

        trace_embed = self._resolve_decode_embed(self._trace_inputs)
        self._trace_output = self._decode_step_batch32(
            trace_embed,
            current_pos,
            self._trace_inputs["pos_tensor"],
            self._trace_inputs["cos"],
            self._trace_inputs["sin"],
        )
        if self._use_embed_lookup:
            # lookup path allocates a fresh embed per call; release it
            # before end_trace_capture so the trace replays the free.
            ttnn.deallocate(trace_embed)
        trace_logits_rm = ttnn.to_layout(self._trace_output, ttnn.ROW_MAJOR_LAYOUT)
        # use_multicore=True parallelizes the vocab-dim reduction across the
        # core grid instead of running a single-core kernel. Measured delta:
        # ~1.9 ms -> ~0.08 ms per step (26x) for vocab=128256.
        self._trace_argmax_output = ttnn.argmax(trace_logits_rm, dim=-1, use_multicore=True)
        self._set_kv_cache_length(current_pos + 1)

        ttnn.end_trace_capture(self.device, self._trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)
        self._trace_capture_pos = current_pos
        self._warmed_trace_keys.add(cache_key)
        return True

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
        embed = self._resolve_decode_embed(decode_inputs)
        logits = self._decode_step_batch32(
            embed,
            current_pos,
            decode_inputs["pos_tensor"],
            decode_inputs["cos"],
            decode_inputs["sin"],
        )
        if self._use_embed_lookup:
            ttnn.deallocate(embed)
        self._set_kv_cache_length(current_pos + 1)
        return logits

    def _release_trace(self) -> None:
        """Release trace resources."""
        if self._trace_id is not None:
            ttnn.release_trace(self.device, self._trace_id)
            self._trace_id = None

        self._trace_output = None
        self._trace_argmax_output = None
        self._trace_capture_pos = None

    def _release_trace_inputs(self) -> None:
        """Release reusable trace input tensors."""
        if self._trace_inputs is not None:
            for tensor in self._trace_inputs.values():
                ttnn.deallocate(tensor)
            self._trace_inputs = None

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
        Prefill phase — writes KV directly into batch32 paged cache.

        By passing the preallocated batch32 caches to the model, the prefill
        attention's update_prefill writes directly via fill_cache_range,
        avoiding the lossy transfer step (stride-slice + pad + re-upload).

        Args:
            tokens: [1, seq_len] token IDs

        Returns:
            (logits, seq_len)
        """
        batch_size, seq_len = tokens.shape

        tokens_tt = ttnn.from_torch(
            torch.from_numpy(tokens.astype(np.int32)),
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
                created_compact = False
                if k_prefill.shape[1] == num_q_heads:
                    # Prefill stores GQA-expanded KV heads. The repeated heads are
                    # contiguous, so keep only the first head from each group on-device.
                    k_compact = ttnn.slice(
                        k_prefill,
                        (0, 0, 0, 0),
                        k_prefill.shape,
                        (1, num_kv_groups, 1, 1),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    v_compact = ttnn.slice(
                        v_prefill,
                        (0, 0, 0, 0),
                        v_prefill.shape,
                        (1, num_kv_groups, 1, 1),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    created_compact = True
                elif k_prefill.shape[1] == num_kv_heads:
                    k_compact = k_prefill
                    v_compact = v_prefill
                else:
                    raise ValueError(
                        f"Unexpected prefill K/V head count: {k_prefill.shape[1]} "
                        f"(expected {num_q_heads} or {num_kv_heads})"
                    )

                if batch32_cache.key_cache.shape[1] > k_compact.shape[1]:
                    pad_amount = batch32_cache.key_cache.shape[1] - k_compact.shape[1]
                    k_upload = ttnn.pad(k_compact, [(0, 0), (0, pad_amount), (0, 0), (0, 0)], 0.0)
                    v_upload = ttnn.pad(v_compact, [(0, 0), (0, pad_amount), (0, 0), (0, 0)], 0.0)
                    padded_upload = True
                else:
                    k_upload = k_compact
                    v_upload = v_compact
                    padded_upload = False

                actual_batch = min(k_upload.shape[0], PADDED_BATCH)
                k_upload_actual = k_upload[:actual_batch, :, :, :]
                v_upload_actual = v_upload[:actual_batch, :, :, :]
                # Match dtype to cache (e.g. bf16 prefill → bfp8 cache)
                if k_upload_actual.dtype != batch32_cache.key_cache.dtype:
                    k_upload_actual = ttnn.typecast(k_upload_actual, batch32_cache.key_cache.dtype)
                    v_upload_actual = ttnn.typecast(v_upload_actual, batch32_cache.value_cache.dtype)
                batch32_cache.key_cache = fill_cache_range(batch32_cache.key_cache, k_upload_actual, 0)
                batch32_cache.value_cache = fill_cache_range(batch32_cache.value_cache, v_upload_actual, 0)

                if padded_upload:
                    ttnn.deallocate(k_upload)
                    ttnn.deallocate(v_upload)
                if created_compact:
                    ttnn.deallocate(k_compact)
                    ttnn.deallocate(v_compact)
                batch32_cache.seq_len_cached = seq_len

    def _sample_token(
        self,
        logits: ttnn.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        device_argmax: Optional[ttnn.Tensor] = None,
    ) -> int:
        """Sample next token from the last logical position of batch element 0."""
        # Fastest path: pre-computed argmax from inside the trace. The argmax
        # runs as part of the traced kernel so no extra dispatch or CQ sync.
        # Reading the tiny int32 result costs ~0.06 ms vs ~4 ms for post-trace ops.
        if device_argmax is not None and (temperature <= 0.0 or top_k is None or top_k <= 1):
            return (int(_flatten_first(device_argmax.cpu().to_list())) if device_argmax.dtype == ttnn.uint32 else int(ttnn.to_torch(device_argmax).flatten()[0].item()))

        use_direct_logits = int(logits.shape[1]) == 1
        last_row = logits
        if not use_direct_logits:
            last_row = ttnn.slice(
                logits,
                (0, max(int(logits.shape[1]) - 1, 0), 0),
                (1, int(logits.shape[1]), int(logits.shape[2])),
            )

        # Fast path: greedy argmax on device (non-trace ops).
        # ROW_MAJOR argmax is ~2× faster than TILE on large vocab tensors.
        if temperature <= 0.0 or top_k is None or top_k <= 1:
            try:
                row_rm = ttnn.to_layout(last_row, ttnn.ROW_MAJOR_LAYOUT)
                token_indices = ttnn.argmax(row_rm, dim=-1)
                token_id = int(_flatten_first(token_indices.cpu().to_list()))
                ttnn.deallocate(token_indices)
                ttnn.deallocate(row_rm)
                if not use_direct_logits:
                    ttnn.deallocate(last_row)
                return token_id
            except Exception:
                pass  # fall through to host path (last_row still live)

        # Slow path: copy full logits to host (needed for temperature sampling,
        # or as fallback if device argmax fails on this shape/layout).
        try:
            host_tensor = None
            for cached_spec, cached_tensor in self._logits_host_tensors:
                try:
                    if cached_spec == last_row.spec:
                        host_tensor = cached_tensor
                        break
                except Exception:
                    continue
            if host_tensor is None:
                host_tensor = ttnn.allocate_tensor_on_host(last_row.spec, self.device)
                self._logits_host_tensors.append((last_row.spec, host_tensor))
            self._logits_host_tensor = host_tensor
            if self._supports_host_tensor_to_torch is not False:
                try:
                    last_logits = ttnn.to_torch(
                        last_row,
                        host_tensor=host_tensor,
                    )[0, 0, :].float()
                    self._supports_host_tensor_to_torch = True
                except TypeError:
                    self._supports_host_tensor_to_torch = False
                    ttnn.copy_device_to_host_tensor(last_row, host_tensor)
                    ttnn.synchronize_device(self.device)
                    last_logits = ttnn.to_torch(host_tensor)[0, 0, :].float()
            else:
                ttnn.copy_device_to_host_tensor(last_row, host_tensor)
                ttnn.synchronize_device(self.device)
                last_logits = ttnn.to_torch(host_tensor)[0, 0, :].float()
        finally:
            if not use_direct_logits:
                ttnn.deallocate(last_row)

        if temperature <= 0.0 or top_k is None or top_k <= 1:
            return int(torch.argmax(last_logits).item())

        if temperature != 1.0:
            last_logits = last_logits / temperature

        if top_k is not None and top_k > 0:
            top_k_values, top_k_indices = torch.topk(last_logits, k=top_k)
            probs = torch.softmax(top_k_values, dim=-1)
            sampled_idx = torch.multinomial(probs, 1)
            return int(top_k_indices[sampled_idx].item())

        return int(torch.argmax(last_logits).item())

    def warmup(
        self,
        prompt: Optional[str] = None,
        *,
        prompt_tokens: int = 6,
        max_new_tokens: int = 32,
        token_id: Optional[int] = None,
    ) -> WarmupStats:
        """
        Pre-compile prompt prefill and first decode/trace for serving startup.

        This pays the heavy first-request compilation cost ahead of time using
        either a real prompt or a synthetic fixed-shape prompt.
        """
        input_ids = self._build_warmup_input_ids(prompt, prompt_tokens, token_id)
        stats = WarmupStats(prompt_tokens=int(input_ids.shape[1]))

        max_seq_len = choose_single_user_cache_seq_len(
            input_ids.shape[1] + max_new_tokens + TRACE_CACHE_SLACK_TOKENS
        )
        stats.cache_seq_len = max_seq_len
        self._ensure_kv_caches(max_seq_len)

        prefill_start = time.perf_counter()
        cached_prefill = self._try_reuse_prefill(input_ids, max_seq_len)
        if cached_prefill is None:
            cached_prefill = self._try_extend_prefill_from_cached_prefix(input_ids, max_seq_len)
            if cached_prefill is None:
                logits, seq_len = self._prefill_batch32(input_ids)
                self._cache_prefill(input_ids, logits, seq_len)
            else:
                logits, seq_len = cached_prefill
        else:
            logits, seq_len = cached_prefill
        stats.prompt_time = time.perf_counter() - prefill_start

        if max_new_tokens <= 0:
            return stats

        sample_start = time.perf_counter()
        next_token = self._sample_token(logits, temperature=0.0, top_k=1)
        stats.sample_time = time.perf_counter() - sample_start

        decode_start = time.perf_counter()
        if self.enable_trace:
            captured_new_trace = self._capture_trace(next_token, seq_len)
            stats.trace_captured = True
            if not captured_new_trace:
                self._execute_trace(next_token, seq_len)
        else:
            warm_logits = self._execute_decode_untraced(next_token, seq_len)
            ttnn.deallocate(warm_logits)
        stats.decode_time = time.perf_counter() - decode_start

        return stats

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
        max_seq_len = choose_single_user_cache_seq_len(
            input_ids.shape[1] + max_new_tokens + TRACE_CACHE_SLACK_TOKENS
        )
        self._ensure_kv_caches(max_seq_len)

        # Prefill
        cached_prefill = self._try_reuse_prefill(input_ids, max_seq_len)
        logits_owned = cached_prefill is None
        if cached_prefill is None:
            cached_prefill = self._try_extend_prefill_from_cached_prefix(input_ids, max_seq_len)
            if cached_prefill is None:
                logits, seq_len = self._prefill_batch32(input_ids)
                self._cache_prefill(input_ids, logits, seq_len)
                logits_owned = False
            else:
                logits, seq_len = cached_prefill
                logits_owned = False
        else:
            logits, seq_len = cached_prefill

        # Sample first token
        next_token = self._sample_token(logits, temperature, top_k)
        generated_ids = list(input_ids[0]) + [next_token]
        if logits_owned:
            ttnn.deallocate(logits)

        if next_token == self.tokenizer.eos_token_id:
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Decode loop
        current_pos = seq_len
        trace_captured = False

        if self.enable_trace and len(generated_ids) - input_ids.shape[1] < max_new_tokens:
            captured_new_trace = self._capture_trace(next_token, current_pos)
            trace_captured = True
            trace_logits = (
                self._trace_output
                if captured_new_trace
                else self._execute_trace(next_token, current_pos)
            )
            next_token = self._sample_token(
                trace_logits, temperature, top_k,
                device_argmax=self._trace_argmax_output,
            )
            generated_ids.append(next_token)
            current_pos += 1

        while len(generated_ids) - input_ids.shape[1] < max_new_tokens:
            logits_owned = True

            if self.enable_trace and trace_captured:
                logits = self._execute_trace(next_token, current_pos)
                logits_owned = False
            else:
                logits = self._execute_decode_untraced(next_token, current_pos)

            next_token = self._sample_token(
                logits, temperature, top_k,
                device_argmax=self._trace_argmax_output if not logits_owned else None,
            )
            generated_ids.append(next_token)
            current_pos += 1

            if logits_owned:
                ttnn.deallocate(logits)

            if next_token == self.tokenizer.eos_token_id:
                break

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
        max_seq_len = choose_single_user_cache_seq_len(
            input_ids.shape[1] + max_new_tokens + TRACE_CACHE_SLACK_TOKENS
        )
        self._ensure_kv_caches(max_seq_len)

        # Prefill
        prefill_start = time.perf_counter()
        cached_prefill = self._try_reuse_prefill(input_ids, max_seq_len)
        logits_owned = cached_prefill is None
        if cached_prefill is None:
            cached_prefill = self._try_extend_prefill_from_cached_prefix(input_ids, max_seq_len)
            if cached_prefill is None:
                logits, seq_len = self._prefill_batch32(input_ids)
                self._cache_prefill(input_ids, logits, seq_len)
                logits_owned = False
            else:
                logits, seq_len = cached_prefill
                logits_owned = False
        else:
            logits, seq_len = cached_prefill
        stats.prompt_time = time.perf_counter() - prefill_start

        # Sample first token
        next_token = self._sample_token(logits, temperature, top_k)
        generated_ids = list(input_ids[0]) + [next_token]
        stats.generated_tokens = 1
        if logits_owned:
            ttnn.deallocate(logits)

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

        if self.enable_trace and stats.generated_tokens < max_new_tokens:
            capture_start = time.perf_counter()
            captured_new_trace = self._capture_trace(next_token, current_pos)
            trace_captured = True
            stats.trace_captured = True
            if captured_new_trace:
                trace_logits = self._trace_output
            else:
                trace_logits = self._execute_trace(next_token, current_pos)
            trace_time = time.perf_counter() - capture_start
            stats.token_times.append(trace_time)
            stats.generation_time += trace_time

            next_token = self._sample_token(
                trace_logits, temperature, top_k,
                device_argmax=self._trace_argmax_output,
            )
            generated_ids.append(next_token)
            stats.generated_tokens += 1
            current_pos += 1

            current_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            new_text = current_text[prev_text_len:]
            prev_text_len = len(current_text)
            yield new_text, stats

            if next_token == self.tokenizer.eos_token_id:
                return

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

            next_token = self._sample_token(
                logits, temperature, top_k,
                device_argmax=self._trace_argmax_output if not logits_owned else None,
            )
            generated_ids.append(next_token)
            stats.generated_tokens += 1
            current_pos += 1

            # Decode only the new token to avoid O(n²) full-sequence re-decode.
            new_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
            yield new_text, stats

            if logits_owned:
                ttnn.deallocate(logits)

            if next_token == self.tokenizer.eos_token_id:
                break

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
            self._release_trace_inputs()
        except Exception:
            self._trace_inputs = None

        try:
            self._clear_host_decode_cache()
        except Exception:
            self._embed_host_cache.clear()
            self._pos_host_cache.clear()
            self.rotary_setup._host_cos_sin_cache.clear()

        try:
            self._clear_prefill_cache()
        except Exception:
            self._cached_prefill_key = None
            self._cached_prefill_seq_len = 0
            self._cached_prefill_logits = None
            self._cached_prefill_cache_seq_len = 0

        if self._logits_host_tensors:
            for _spec, tensor in self._logits_host_tensors:
                try:
                    ttnn.deallocate(tensor)
                except Exception:
                    pass
        if self._argmax_output_tensors:
            for _shape, tensor in self._argmax_output_tensors:
                try:
                    ttnn.deallocate(tensor)
                except Exception:
                    pass
        self._argmax_output_tensors = []
        self._logits_host_tensors = []
        self._logits_host_tensor = None
        self._supports_host_tensor_to_torch = None

        if self._kv_caches is not None:
            for cache in self._kv_caches:
                cache.reset()
