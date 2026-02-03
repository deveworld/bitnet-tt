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

from bitnet_tt.layers.attention import KVCache


if TYPE_CHECKING:
    from bitnet_tt.model.bitnet import BitNetModel


PADDED_BATCH = 32
TILE_SIZE = 32


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
    - cos/sin are sharded to HEIGHT_SHARDED after embedding lookup
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
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        # Interleaved format for rotary_embedding_llama
        cos_half = cos[:, : head_dim // 2]
        sin_half = sin[:, : head_dim // 2]
        cos_interleaved = torch.stack([cos_half, cos_half], dim=-1).flatten(-2)
        sin_interleaved = torch.stack([sin_half, sin_half], dim=-1).flatten(-2)

        # Upload as embedding matrices [max_seq_len, head_dim]
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

    def get_sharded_cos_sin(self, position: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Get HEIGHT_SHARDED cos/sin for decode at given position.

        Pattern from tt-transformers/rope.py:
        1. Embedding lookup for position
        2. Reshape to [1, batch, 1, head_dim]
        3. Shard to HEIGHT_SHARDED
        """
        # Create position indices for all 32 batch elements (same position)
        rot_idxs = ttnn.from_torch(
            torch.full((1, PADDED_BATCH), position, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Embedding lookup: [1, batch, head_dim]
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)

        # Reshape: [1, batch, head_dim] -> [1, batch, 1, head_dim] -> [1, 1, batch, head_dim]
        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, batch, head_dim]
        sin = ttnn.unsqueeze_to_4D(sin)

        # Transpose: [1, 1, batch, head_dim] -> [1, batch, 1, head_dim]
        cos = ttnn.transpose(cos, 1, 2)
        sin = ttnn.transpose(sin, 1, 2)

        # Shard to HEIGHT_SHARDED
        cos_sharded = ttnn.to_memory_config(cos, self.cos_sin_config)
        sin_sharded = ttnn.to_memory_config(sin, self.cos_sin_config)

        ttnn.deallocate(rot_idxs)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

        return cos_sharded, sin_sharded

    def get_sharded_trans_mat(self) -> ttnn.Tensor:
        """Get HEIGHT_SHARDED transformation matrix."""
        return ttnn.to_memory_config(self.transformation_mat, self.trans_mat_config)


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
    ):
        self.model = model
        self.device = model.device
        self.config = model.config
        self.tokenizer = tokenizer
        self.enable_trace = enable_trace

        # Trace state
        self._trace_id: Optional[int] = None
        self._trace_inputs: Optional[dict] = None
        self._trace_output: Optional[ttnn.Tensor] = None

        # Batch-32 RoPE setup
        self.rotary_setup = Batch32RotarySetup(
            device=self.device,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_position_embeddings,
            rope_theta=self.config.rope_theta,
        )

        # Store embedding weight on host for trace execution
        self._embedding_weight_host = ttnn.to_torch(model.embed_tokens.weight)

        # KV caches (allocated per generation)
        self._kv_caches: Optional[list[Batch32KVCache]] = None

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

    def _allocate_kv_caches(self, max_seq_len: int) -> list[Batch32KVCache]:
        """Allocate batch-32 KV caches for all layers."""
        caches = []
        for _ in range(self.config.num_layers):
            cache = Batch32KVCache(
                max_seq_len=max_seq_len,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                device=self.device,
            )
            caches.append(cache)
        return caches

    def _decode_step_batch32(
        self,
        token_embeds: ttnn.Tensor,
        current_pos: int,
        current_pos_tensor: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Single decode step using batch-32 HEIGHT_SHARDED pipeline.

        This is the core optimized decode that can be traced.

        Args:
            token_embeds: [1, 1, 32, hidden_size] token embeddings
            current_pos: Current position (int)
            current_pos_tensor: INT32 tensor for KV cache update

        Returns:
            logits: [32, 1, vocab_size]
        """
        hidden = token_embeds

        # Get sharded RoPE tensors
        cos_s, sin_s = self.rotary_setup.get_sharded_cos_sin(current_pos)
        trans_mat = self.rotary_setup.get_sharded_trans_mat()

        for layer_idx in range(self.config.num_layers):
            layer = self.model.layers[layer_idx]
            kv_cache = self._kv_caches[layer_idx]

            # Input LayerNorm
            normed = layer.input_layernorm(hidden)

            # Reshape for fused QKV: [1, 1, 32, hidden]
            normed_rm = ttnn.to_layout(normed, ttnn.ROW_MAJOR_LAYOUT)
            normed_1bkd = ttnn.reshape(normed_rm, (1, 1, PADDED_BATCH, self.config.hidden_size))
            normed_1bkd = ttnn.to_layout(normed_1bkd, ttnn.TILE_LAYOUT)

            # Fused QKV matmul
            attention = layer.self_attn
            qkv_fused = ttnn.matmul(normed_1bkd, attention.qkv_fused_weight)

            # Split into Q/K/V heads (outputs HEIGHT_SHARDED)
            q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv_fused,
                num_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_key_value_heads,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )

            # RoPE (requires HEIGHT_SHARDED inputs)
            q_rot = ttnn.experimental.rotary_embedding_llama(
                q_heads, cos_s, sin_s, trans_mat, is_decode_mode=True
            )
            k_rot = ttnn.experimental.rotary_embedding_llama(
                k_heads, cos_s, sin_s, trans_mat, is_decode_mode=True
            )

            # Update KV cache
            kv_cache.update(k_rot, v_heads, current_pos_tensor)
            kv_cache.seq_len_cached = current_pos + 1

            # Get full cache for attention
            k_cache, v_cache = kv_cache.get_for_attention(current_pos + 1)

            # SDPA decode outputs to DRAM [1, batch, heads, head_dim]
            attn_output_dram = ttnn.transformer.scaled_dot_product_attention_decode(
                q_rot,
                k_cache,
                v_cache,
                cur_pos_tensor=current_pos_tensor,
                scale=1.0 / (self.config.head_dim**0.5),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Convert to HEIGHT_SHARDED for nlp_concat_heads_decode
            sdpa_sharded_config = create_sdpa_output_sharded_config(
                self.config.num_attention_heads, self.config.head_dim
            )
            attn_output = ttnn.to_memory_config(attn_output_dram, sdpa_sharded_config)
            ttnn.deallocate(attn_output_dram)

            # Concat heads (output is WIDTH_SHARDED)
            attn_output = ttnn.experimental.nlp_concat_heads_decode(
                attn_output, num_heads=self.config.num_attention_heads
            )

            # Convert to interleaved for LayerNorm compatibility
            attn_output = ttnn.to_memory_config(attn_output, ttnn.DRAM_MEMORY_CONFIG)

            # Attn sub-norm + output projection
            attn_output = layer.self_attn.attn_sub_norm(attn_output)
            attn_output = layer.self_attn.o_proj(attn_output)

            # Residual
            hidden = ttnn.add(hidden, attn_output)

            # FFN
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden, mode="decode")
            hidden = ttnn.add(residual, hidden)

            # Cleanup intermediate tensors
            ttnn.deallocate(normed_rm)
            ttnn.deallocate(normed_1bkd)
            ttnn.deallocate(qkv_fused)

        # Cleanup RoPE tensors
        ttnn.deallocate(cos_s)
        ttnn.deallocate(sin_s)
        ttnn.deallocate(trans_mat)

        # Final norm + LM head
        hidden = self.model.norm(hidden)
        logits = ttnn.matmul(hidden, self.model.lm_head_weight)

        return logits

    def _setup_trace_inputs(self) -> dict:
        """Pre-allocate trace input tensors on device."""
        embed_shape = (PADDED_BATCH, 1, self.config.hidden_size)
        pos_shape = (32,)

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

        return {"embeds": embeds, "pos_tensor": pos_tensor}

    def _capture_trace(self, current_pos: int) -> None:
        """Capture Metal Trace for decode loop."""
        if self._trace_id is not None:
            return

        self._trace_inputs = self._setup_trace_inputs()

        self._trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)

        self._trace_output = self._decode_step_batch32(
            self._trace_inputs["embeds"],
            current_pos,
            self._trace_inputs["pos_tensor"],
        )

        ttnn.end_trace_capture(self.device, self._trace_id, cq_id=0)

    def _execute_trace(
        self,
        embed_vec: torch.Tensor,
        current_pos: int,
    ) -> ttnn.Tensor:
        """Execute captured trace with new inputs."""
        embed_padded = torch.zeros((PADDED_BATCH, 1, self.config.hidden_size), dtype=torch.bfloat16)
        embed_padded[0, :, :] = embed_vec

        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(embed_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            self._trace_inputs["embeds"],
        )

        pos_data = torch.tensor([current_pos] * 32, dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(pos_data, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
            self._trace_inputs["pos_tensor"],
        )

        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)

        return self._trace_output

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
        for layer_idx, model_cache in enumerate(model_kv_caches):
            if model_cache is None:
                continue

            batch32_cache = self._kv_caches[layer_idx]

            if hasattr(model_cache, "key_cache") and model_cache.key_cache is not None:
                k_prefill = model_cache.key_cache
                v_prefill = model_cache.value_cache

                k_torch = ttnn.to_torch(k_prefill)
                v_torch = ttnn.to_torch(v_prefill)

                k_padded = torch.zeros(
                    (
                        PADDED_BATCH,
                        self.config.num_key_value_heads,
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
        """Sample next token from logits (batch[0] only)."""
        # Get logits for batch[0], last position
        logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        last_logits = logits_rm[0, -1:, :]  # [1, vocab]
        logits_np = ttnn.to_torch(last_logits).float().numpy().flatten()

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
        max_seq_len = input_ids.shape[1] + max_new_tokens + 32
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
            for i in range(max_new_tokens - 1):
                embed_vec = self._embedding_weight_host[next_token].unsqueeze(0).unsqueeze(0)

                if self.enable_trace and trace_captured:
                    logits = self._execute_trace(embed_vec, current_pos)
                else:
                    embed_padded = torch.zeros(
                        (PADDED_BATCH, 1, self.config.hidden_size), dtype=torch.bfloat16
                    )
                    embed_padded[0, :, :] = embed_vec

                    embeds = ttnn.from_torch(
                        embed_padded,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                    )

                    pos_tensor = ttnn.from_torch(
                        torch.tensor([current_pos] * 32, dtype=torch.int32),
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=self.device,
                    )

                    logits = self._decode_step_batch32(embeds, current_pos, pos_tensor)

                    ttnn.deallocate(embeds)
                    ttnn.deallocate(pos_tensor)

                    if self.enable_trace and i == 0:
                        self._capture_trace(current_pos + 1)
                        trace_captured = True

                next_token = self._sample_token(logits, temperature, top_k)
                generated_ids.append(next_token)
                current_pos += 1

                if not (self.enable_trace and trace_captured):
                    ttnn.deallocate(logits)

                if next_token == self.tokenizer.eos_token_id:
                    break

        finally:
            if self.enable_trace:
                self._release_trace()

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
        max_seq_len = input_ids.shape[1] + max_new_tokens + 32
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
            for i in range(max_new_tokens - 1):
                token_start = time.perf_counter()

                embed_vec = self._embedding_weight_host[next_token].unsqueeze(0).unsqueeze(0)

                if self.enable_trace and trace_captured:
                    logits = self._execute_trace(embed_vec, current_pos)
                else:
                    embed_padded = torch.zeros(
                        (PADDED_BATCH, 1, self.config.hidden_size), dtype=torch.bfloat16
                    )
                    embed_padded[0, :, :] = embed_vec

                    embeds = ttnn.from_torch(
                        embed_padded,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                    )

                    pos_tensor = ttnn.from_torch(
                        torch.tensor([current_pos] * 32, dtype=torch.int32),
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=self.device,
                    )

                    logits = self._decode_step_batch32(embeds, current_pos, pos_tensor)

                    ttnn.deallocate(embeds)
                    ttnn.deallocate(pos_tensor)

                    if self.enable_trace and i == 0:
                        self._capture_trace(current_pos + 1)
                        trace_captured = True

                token_time = time.perf_counter() - token_start
                stats.token_times.append(token_time)
                stats.generation_time += token_time

                next_token = self._sample_token(logits, temperature, top_k)
                generated_ids.append(next_token)
                stats.generated_tokens += 1
                current_pos += 1

                current_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                new_text = current_text[prev_text_len:]
                prev_text_len = len(current_text)
                yield new_text, stats

                if not (self.enable_trace and trace_captured):
                    ttnn.deallocate(logits)

                if next_token == self.tokenizer.eos_token_id:
                    break

        finally:
            if self.enable_trace:
                self._release_trace()

    def reset(self) -> None:
        """Reset generator state."""
        if self._trace_id is not None:
            try:
                ttnn.release_trace(self.device, self._trace_id)
            except Exception:
                pass
        self._trace_id = None
        self._trace_inputs = None
        self._trace_output = None

        if self._kv_caches is not None:
            for cache in self._kv_caches:
                cache.reset()
