"""
Optimized text generation for BitNet using TT-NN.

This module provides a high-level interface for text generation with:
- Separate prefill and decode paths
- Trace capture for decode loop (eliminates compilation overhead)
- On-device sampling to minimize data transfer
- RotarySetup for optimized RoPE (tt_transformers pattern)

Performance optimizations based on tt_transformers patterns:
- Pre-computed cos/sin matrices with embedding lookup
- Pre-allocated KV cache
- Trace capture for decode loop
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Optional

import numpy as np
import torch
import ttnn
from numpy.typing import NDArray

from bitnet_tt.layers.attention import KVCache
from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn, ttnn_to_numpy
from bitnet_tt.layers.rope_optimized import RotarySetup


def _on_device_argmax(logits: ttnn.Tensor) -> NDArray[np.int64]:
    """
    Perform argmax on device and return only the token index.

    Avoids transferring the full logits tensor (128KB for vocab_size=128256).
    """
    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    last_logits = logits_rm[:, -1:, :]
    last_logits_tile = ttnn.to_layout(last_logits, ttnn.TILE_LAYOUT)
    token_indices = ttnn.argmax(last_logits_tile, dim=-1)
    indices_np = ttnn.to_torch(token_indices).numpy().astype(np.int64)
    return indices_np.flatten()


def _on_device_topk(logits: ttnn.Tensor, k: int) -> tuple[NDArray[np.floating], NDArray[np.int64]]:
    """
    Perform top-k selection on device.

    Avoids transferring the full logits tensor.
    """
    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    last_logits = logits_rm[:, -1:, :]
    last_logits_tile = ttnn.to_layout(last_logits, ttnn.TILE_LAYOUT)
    top_k_values, top_k_indices = ttnn.topk(last_logits_tile, k, dim=-1)
    values_np = ttnn.to_torch(top_k_values).float().numpy().squeeze(1)
    indices_np = ttnn.to_torch(top_k_indices).numpy().astype(np.int64).squeeze(1)
    return values_np, indices_np


def _softmax(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute softmax along the last axis."""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


if TYPE_CHECKING:
    from bitnet_tt.model.bitnet import BitNetModel


@dataclass
class GenerationStats:
    """Statistics for text generation."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    prompt_time: float = 0.0
    generation_time: float = 0.0
    token_times: list[float] = field(default_factory=list)

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
    def avg_token_time(self) -> float:
        if self.token_times:
            return sum(self.token_times) / len(self.token_times)
        return 0.0

    def __str__(self) -> str:
        return (
            f"\n[Stats] "
            f"Prompt: {self.prompt_tokens} tokens ({self.prompt_time * 1000:.1f}ms) | "
            f"Generated: {self.generated_tokens} tokens ({self.generation_time:.2f}s) | "
            f"Speed: {self.tokens_per_second:.2f} tokens/s | "
            f"TTFT: {self.time_to_first_token * 1000:.1f}ms"
        )


class TextGenerator:
    """
    Optimized TT-NN text generator for BitNet.

    Key optimizations:
    - Separate prefill and decode paths
    - Trace capture for decode loop (2-3x speedup)
    - On-device sampling to minimize data transfer

    Usage:
        generator = TextGenerator(model, tokenizer)

        # Simple generation
        output = generator.generate("Hello, world!")

        # Streaming generation
        for token, stats in generator.generate_streaming("Hello"):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        model: "BitNetModel",
        tokenizer: Any = None,
        enable_trace: bool = False,
        batch_size: int = 1,
    ) -> None:
        self.model = model
        self.device = model.device
        self.config = model.config
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.enable_trace = enable_trace

        self._trace_id: Optional[int] = None
        self._trace_inputs: Optional[list] = None
        self._trace_output: Optional[ttnn.Tensor] = None

        self.rotary_setup = RotarySetup(
            device=self.device,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_position_embeddings,
            rope_theta=self.config.rope_theta,
            batch_size=batch_size,
        )

        self._preallocated_kv_caches: Optional[list[KVCache]] = None

        self._embedding_weight_host: Optional[torch.Tensor] = None
        if enable_trace:
            self._embedding_weight_host = ttnn.to_torch(model.embed_tokens.weight)

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

    def _preallocate_kv_caches(
        self, use_paged: bool = False, max_seq_len: int | None = None
    ) -> list[KVCache]:
        """
        Pre-allocate KV caches for all layers.

        Args:
            use_paged: If True, use 32-padded heads for paged_update_cache (Trace compatible)
            max_seq_len: Optional max sequence length override
        """
        caches = []
        max_seq_len = max_seq_len or self.config.max_position_embeddings
        for layer_idx in range(self.config.num_layers):
            cache = KVCache(
                max_seq_len=max_seq_len,
                batch_size=self.batch_size,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                device=self.device,
            )
            cache.preallocate(
                batch_size=self.batch_size,
                num_kv_heads=self.config.num_key_value_heads,
                max_seq_len=max_seq_len,
                head_dim=self.config.head_dim,
                device=self.device,
                use_paged=use_paged,
            )
            caches.append(cache)
        return caches

    def get_rot_mats_decode(self, current_pos: int) -> list:
        """
        Get rotation matrices for decode step using RotarySetup.

        Uses embedding lookup instead of per-token tensor creation.
        """
        position_ids = torch.tensor([current_pos], dtype=torch.long)
        return self.rotary_setup.get_rot_mats(position_ids)

    def get_transformation_mat(self) -> ttnn.Tensor:
        """Get transformation matrix for rotary_embedding_llama."""
        return self.rotary_setup.get_transformation_mats()["decode"]

    def prefill_forward(
        self,
        tokens: NDArray[np.int64],
        kv_cache: Optional[list[KVCache]] = None,
        use_preallocated: bool = False,
        max_seq_len: int | None = None,
    ) -> tuple[ttnn.Tensor, list[KVCache]]:
        """
        Process prompt tokens (prefill phase).

        Args:
            tokens: Input token IDs (batch, seq_len)
            kv_cache: Optional existing KV cache
            use_preallocated: Use pre-allocated KV caches for optimized decode
            max_seq_len: Optional max sequence length override

        Returns:
            (logits, updated_kv_cache)
        """
        if use_preallocated and kv_cache is None:
            if self._preallocated_kv_caches is None:
                self._preallocated_kv_caches = self._preallocate_kv_caches(
                    use_paged=True,  # Always use paged cache for in-place updates
                    max_seq_len=max_seq_len,
                )
            for cache in self._preallocated_kv_caches:
                cache.seq_len_cached = 0
            kv_cache = self._preallocated_kv_caches

        input_tensor = numpy_int_to_ttnn(tokens, self.device)
        logits, kv_cache = self.model(
            input_tensor,
            past_key_values=kv_cache,
            use_cache=True,
            mode="prefill",
        )
        ttnn.deallocate(input_tensor)

        # For non-preallocated caches, transfer prefill results to cache
        # (Preallocated caches are handled directly in update_prefill via fill_cache)
        for cache in kv_cache:
            if (
                not cache._preallocated
                and hasattr(cache, "_prefill_key")
                and cache._prefill_key is not None
            ):
                # _prefill_key/value are already GQA-expanded for concat-based decode
                cache.key_cache = cache._prefill_key
                cache.value_cache = cache._prefill_value
                cache._prefill_key = None
                cache._prefill_value = None
                cache._gqa_expanded = True

        return logits, kv_cache

    def decode_forward(
        self,
        token: NDArray[np.int64],
        current_pos: int,
        kv_cache: list[KVCache],
        use_optimized: bool | None = None,  # Auto-detect based on cache type
    ) -> tuple[ttnn.Tensor, list[KVCache]]:
        """
        Generate single token (decode phase).

        Args:
            token: Single token ID (batch, 1)
            current_pos: Current position in sequence
            kv_cache: KV cache from prefill
            use_optimized: Use optimized decode path with rot_mats
                           (None = auto-detect based on cache._preallocated)

        Returns:
            (logits, updated_kv_cache)
        """
        input_tensor = numpy_int_to_ttnn(token, self.device)

        if use_optimized is None:
            use_optimized = (
                self.enable_trace
                and kv_cache is not None
                and len(kv_cache) > 0
                and kv_cache[0] is not None
                and hasattr(kv_cache[0], "_preallocated")
                and kv_cache[0]._preallocated
                and hasattr(kv_cache[0], "_use_paged")
                and kv_cache[0]._use_paged
            )

        rot_mats = None
        transformation_mat = None
        current_pos_tensor = None
        cos_sin_tensors = None

        if use_optimized:
            rot_mats = self.get_rot_mats_decode(current_pos)
            transformation_mat = self.get_transformation_mat()
            current_pos_tensor = ttnn.from_torch(
                torch.tensor([current_pos], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            cos_sin_tensors = self.rotary_setup.create_cos_sin_device_tensors(current_pos)

        logits, kv_cache = self.model(
            input_tensor,
            past_key_values=kv_cache,
            use_cache=True,
            mode="decode",
            current_pos=current_pos,
            rot_mats=rot_mats,
            transformation_mat=transformation_mat,
            current_pos_tensor=current_pos_tensor,
            cos_sin_tensors=cos_sin_tensors,
        )

        if current_pos_tensor is not None:
            ttnn.deallocate(current_pos_tensor)
        if cos_sin_tensors is not None:
            ttnn.deallocate(cos_sin_tensors[0])
            ttnn.deallocate(cos_sin_tensors[1])
        ttnn.deallocate(input_tensor)
        return logits, kv_cache

    def _capture_decode_trace(
        self,
        token: NDArray[np.int64],
        current_pos: int,
        kv_cache: list[KVCache],
    ) -> tuple[int, ttnn.Tensor, dict[str, ttnn.Tensor]]:
        """
        Capture a trace for the decode forward pass.

        This compiles the decode graph once and reuses it for all subsequent
        decode steps, eliminating compilation overhead (2-3x speedup).

        Key fixes for trace mode:
        1. Embedding OUTSIDE trace: ttnn.embedding does write operations incompatible with trace
        2. Pre-compute cos/sin OUTSIDE trace as device tensors
        3. Pass inputs_embeds to model instead of input_ids during trace

        Returns:
            (trace_id, output_tensor, dict of trace input tensors)
        """
        input_tensor = numpy_int_to_ttnn(token, self.device)

        inputs_embeds = self.model.embed_tokens(input_tensor)

        pos_tensor_int32 = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cos_device, sin_device = self.rotary_setup.create_cos_sin_device_tensors(current_pos)
        cos_sin_tensors = (cos_device, sin_device)

        rot_mats = self.get_rot_mats_decode(current_pos)
        transformation_mat = self.get_transformation_mat()

        saved_seq_len = [cache.seq_len_cached for cache in kv_cache]
        _, _ = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=kv_cache,
            use_cache=True,
            mode="decode",
            current_pos=current_pos,
            current_pos_tensor=pos_tensor_int32,
            rot_mats=rot_mats,
            transformation_mat=transformation_mat,
            cos_sin_tensors=cos_sin_tensors,
        )
        for cache, saved_len in zip(kv_cache, saved_seq_len):
            cache.seq_len_cached = saved_len
        ttnn.synchronize_device(self.device)

        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        logits, _ = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=kv_cache,
            use_cache=True,
            mode="decode",
            current_pos=current_pos,
            current_pos_tensor=pos_tensor_int32,
            rot_mats=rot_mats,
            transformation_mat=transformation_mat,
            cos_sin_tensors=cos_sin_tensors,
        )
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

        trace_inputs = {
            "input": input_tensor,
            "embeds": inputs_embeds,
            "pos_int32": pos_tensor_int32,
            "cos": cos_device,
            "sin": sin_device,
        }
        return trace_id, logits, trace_inputs

    def _execute_decode_trace(
        self,
        token: NDArray[np.int64],
        current_pos: int,
    ) -> ttnn.Tensor:
        """
        Execute the captured decode trace with new input.

        Updates trace inputs:
        1. Recompute embedding on HOST and copy to device tensor
        2. Position tensor (int32) for KV cache
        3. Pre-computed cos/sin tensors for RoPE
        """
        if token.dtype != np.int32:
            token = token.astype(np.int32)

        token_idx = token.flatten()[0]
        embed_vec = (
            self._embedding_weight_host[token_idx].unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
        )
        new_embeds_host = ttnn.from_torch(
            embed_vec,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
        )
        ttnn.copy_host_to_device_tensor(new_embeds_host, self._trace_inputs["embeds"])

        new_pos_int32 = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None,
        )
        ttnn.copy_host_to_device_tensor(new_pos_int32, self._trace_inputs["pos_int32"])

        cos_host, sin_host = self.rotary_setup.get_cos_sin_host_tensor(current_pos)
        ttnn.copy_host_to_device_tensor(cos_host, self._trace_inputs["cos"])
        ttnn.copy_host_to_device_tensor(sin_host, self._trace_inputs["sin"])

        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)

        return self._trace_output

    def reset_trace(self) -> None:
        """Reset the decode trace (call when starting a new generation)."""
        if self._trace_id is not None:
            try:
                ttnn.release_trace(self.device, self._trace_id)
            except Exception:
                pass
        self._trace_id = None
        self._trace_inputs = None
        self._trace_output = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = 50,
        do_sample: bool = True,
        use_optimized: bool = False,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            do_sample: Whether to sample
            use_optimized: Use optimized path with pre-allocated caches (default: False)
                           Note: Currently disabled as paged_update_cache requires HEIGHT_SHARDED

        Returns:
            Generated text including the prompt
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")

        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids: NDArray[np.int64] = inputs["input_ids"]

        generated_ids = self._generate_tokens(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            use_optimized=use_optimized,
        )

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def _generate_tokens(
        self,
        input_ids: NDArray[np.int64],
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        do_sample: bool,
        use_optimized: bool = False,
    ) -> NDArray[np.int64]:
        """
        Generate tokens with optimized prefill/decode split.

        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            do_sample: Whether to sample
            use_optimized: Use optimized path (default: False, currently disabled)
        """
        batch_size, seq_len = input_ids.shape
        generated = input_ids.copy()

        # Reset trace for new generation
        self.reset_trace()

        use_prealloc = use_optimized or self.enable_trace
        padded_seq_len = ((seq_len + max_new_tokens + 31) // 32) * 32
        max_seq_len = min(self.config.max_position_embeddings, padded_seq_len)
        logits, kv_cache = self.prefill_forward(
            input_ids, use_preallocated=use_prealloc, max_seq_len=max_seq_len
        )

        next_token = self._sample_next_token(logits, temperature, top_k, do_sample)
        ttnn.deallocate(logits)

        generated = np.concatenate([generated, next_token.reshape(batch_size, 1)], axis=1)

        eos_id = self.tokenizer.eos_token_id if self.tokenizer is not None else None
        finished = next_token == eos_id if eos_id is not None else np.zeros(batch_size, dtype=bool)
        if eos_id is not None and finished.all():
            return generated

        # Phase 2: Decode - generate remaining tokens
        # current_pos tracks where to write the next KV (prefill filled 0 to seq_len-1)
        current_pos = seq_len

        for step in range(max_new_tokens - 1):
            # Decode forward with trace capture if enabled
            if self.enable_trace:
                if self._trace_id is None:
                    # First decode step: run WITHOUT trace to write real KV data
                    # This prevents warmup from corrupting the cache position
                    logits, kv_cache = self.decode_forward(
                        generated[:, -1:],
                        current_pos,
                        kv_cache,
                        use_optimized=True,  # Use optimized path with paged cache
                    )

                    next_token = self._sample_next_token(logits, temperature, top_k, do_sample)
                    ttnn.deallocate(logits)

                    if eos_id is not None:
                        next_token = np.where(finished, eos_id, next_token)
                        finished = np.logical_or(finished, next_token == eos_id)

                    generated = np.concatenate(
                        [generated, next_token.reshape(batch_size, 1)], axis=1
                    )
                    current_pos += 1

                    if eos_id is not None and finished.all():
                        break

                    # Now capture trace at the NEXT position (current_pos is already incremented)
                    # Warmup writes to current_pos, trace capture also writes to current_pos
                    # Since we're at a fresh position, there's no corruption
                    self._trace_id, self._trace_output, self._trace_inputs = (
                        self._capture_decode_trace(
                            generated[:, -1:],
                            current_pos,
                            kv_cache,
                        )
                    )
                    logits = self._trace_output
                else:
                    # Subsequent steps: execute captured trace
                    logits = self._execute_decode_trace(generated[:, -1:], current_pos)
            else:
                logits, kv_cache = self.decode_forward(
                    generated[:, -1:],
                    current_pos,
                    kv_cache,
                )

            next_token = self._sample_next_token(logits, temperature, top_k, do_sample)
            if not self.enable_trace:
                ttnn.deallocate(logits)

            if eos_id is not None:
                next_token = np.where(finished, eos_id, next_token)
                finished = np.logical_or(finished, next_token == eos_id)

            generated = np.concatenate([generated, next_token.reshape(batch_size, 1)], axis=1)
            current_pos += 1

            if eos_id is not None and finished.all():
                break

        return generated

    def _sample_next_token(
        self,
        logits: ttnn.Tensor,
        temperature: float,
        top_k: int | None,
        do_sample: bool,
    ) -> NDArray[np.int64]:
        """Sample the next token from logits."""
        if not do_sample:
            return _on_device_argmax(logits)
        elif top_k is not None and top_k > 0:
            top_k_values, top_k_indices = _on_device_topk(logits, top_k)
            return self._sample_from_topk(top_k_values, top_k_indices, temperature)
        else:
            logits_np = ttnn_to_numpy(logits)
            return self._sample_token(logits_np[:, -1, :], temperature, top_k, True)

    def _sample_from_topk(
        self,
        top_k_values: NDArray[np.floating],
        top_k_indices: NDArray[np.int64],
        temperature: float,
    ) -> NDArray[np.int64]:
        """Sample from pre-computed top-k values."""
        batch_size = top_k_values.shape[0]
        k = top_k_values.shape[1]

        if temperature != 1.0:
            top_k_values = top_k_values / temperature

        probs = _softmax(top_k_values)
        sampled_local_indices = np.array(
            [np.random.choice(k, p=probs[i]) for i in range(batch_size)]
        )
        return np.take_along_axis(
            top_k_indices, sampled_local_indices.reshape(-1, 1), axis=-1
        ).flatten()

    def _sample_token(
        self,
        logits: NDArray[np.floating],
        temperature: float,
        top_k: int | None,
        do_sample: bool,
    ) -> NDArray[np.int64]:
        """Sample from full distribution."""
        batch_size = logits.shape[0]

        if not do_sample:
            return np.argmax(logits, axis=-1)

        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None and top_k > 0:
            top_k_indices = np.argpartition(logits, -top_k, axis=-1)[:, -top_k:]
            top_k_logits = np.take_along_axis(logits, top_k_indices, axis=-1)
            probs = _softmax(top_k_logits)
            sampled_indices = np.array(
                [np.random.choice(top_k, p=probs[i]) for i in range(batch_size)]
            )
            return np.take_along_axis(
                top_k_indices, sampled_indices.reshape(-1, 1), axis=-1
            ).flatten()
        else:
            probs = _softmax(logits)
            return np.array(
                [np.random.choice(logits.shape[-1], p=probs[i]) for i in range(batch_size)]
            )

    def _is_eos(self, token_id: int) -> bool:
        """Check if token is end-of-sequence."""
        if self.tokenizer is None:
            return False
        return token_id == self.tokenizer.eos_token_id

    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = 50,
        do_sample: bool = True,
        use_cache: bool = True,  # Always True in optimized version
        use_optimized: bool = True,  # Use in-place KV cache update for better performance
    ) -> Generator[tuple[str, GenerationStats], None, None]:
        """
        Generate text with streaming output.

        Yields tokens as they are generated along with statistics.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            do_sample: Whether to sample
            use_cache: Always True (kept for API compatibility)
            use_optimized: Use optimized path (default: False, currently disabled)
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")

        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids: NDArray[np.int64] = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape

        stats = GenerationStats(prompt_tokens=seq_len)
        generated = input_ids.copy()
        prev_text_len = len(self.tokenizer.decode(generated[0], skip_special_tokens=True))

        # Reset trace
        self.reset_trace()

        prefill_start = time.perf_counter()
        use_prealloc = use_optimized or self.enable_trace
        padded_seq_len = ((seq_len + max_new_tokens + 31) // 32) * 32
        max_seq_len = min(self.config.max_position_embeddings, padded_seq_len)
        logits, kv_cache = self.prefill_forward(
            input_ids, use_preallocated=use_prealloc, max_seq_len=max_seq_len
        )
        prefill_end = time.perf_counter()
        stats.prompt_time = prefill_end - prefill_start

        # Sample first token
        next_token = self._sample_next_token(logits, temperature, top_k, do_sample)
        ttnn.deallocate(logits)

        generated = np.concatenate([generated, next_token.reshape(batch_size, 1)], axis=1)
        stats.generated_tokens += 1

        # Yield first token
        current_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        new_text = current_text[prev_text_len:]
        prev_text_len = len(current_text)
        yield new_text, stats

        if self._is_eos(next_token[0]):
            return

        # Phase 2: Decode loop
        current_pos = seq_len

        for step in range(max_new_tokens - 1):
            token_start = time.perf_counter()

            if self.enable_trace:
                if self._trace_id is None:
                    # First decode step: run WITHOUT trace to write real KV data
                    # This prevents warmup from corrupting the cache position
                    logits, kv_cache = self.decode_forward(
                        generated[:, -1:],
                        current_pos,
                        kv_cache,
                        use_optimized=True,
                    )

                    token_end = time.perf_counter()
                    token_time = token_end - token_start
                    stats.token_times.append(token_time)
                    stats.generation_time += token_time

                    next_token = self._sample_next_token(logits, temperature, top_k, do_sample)
                    ttnn.deallocate(logits)
                    generated = np.concatenate(
                        [generated, next_token.reshape(batch_size, 1)], axis=1
                    )
                    stats.generated_tokens += 1
                    current_pos += 1

                    current_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                    new_text = current_text[prev_text_len:]
                    prev_text_len = len(current_text)
                    yield new_text, stats

                    if self._is_eos(next_token[0]):
                        break

                    # Capture trace at the NEXT position (current_pos already incremented)
                    token_start = time.perf_counter()
                    self._trace_id, self._trace_output, self._trace_inputs = (
                        self._capture_decode_trace(
                            generated[:, -1:],
                            current_pos,
                            kv_cache,
                        )
                    )
                    logits = self._trace_output
                else:
                    logits = self._execute_decode_trace(generated[:, -1:], current_pos)
            else:
                logits, kv_cache = self.decode_forward(
                    generated[:, -1:],
                    current_pos,
                    kv_cache,
                )

            token_end = time.perf_counter()
            token_time = token_end - token_start
            stats.token_times.append(token_time)
            stats.generation_time += token_time

            next_token = self._sample_next_token(logits, temperature, top_k, do_sample)
            if not self.enable_trace:
                ttnn.deallocate(logits)

            generated = np.concatenate([generated, next_token.reshape(batch_size, 1)], axis=1)
            stats.generated_tokens += 1
            current_pos += 1

            current_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            new_text = current_text[prev_text_len:]
            prev_text_len = len(current_text)
            yield new_text, stats

            if self._is_eos(next_token[0]):
                break

    def chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response in chat format."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

    def chat_streaming(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        use_cache: bool = True,  # Always True in optimized version
        use_optimized: bool = True,  # Use in-place KV cache for better performance
    ) -> Generator[tuple[str, GenerationStats], None, None]:
        """Generate a chat response with streaming output."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        yield from self.generate_streaming(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            use_cache=use_cache,
            use_optimized=use_optimized,
        )
