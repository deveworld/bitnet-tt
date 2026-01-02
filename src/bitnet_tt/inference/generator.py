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
        enable_trace: bool = False,  # Disabled: pos_tensor buffer issue with external passing
        batch_size: int = 1,
    ) -> None:
        """
        Initialize the text generator.

        Args:
            model: BitNet model instance
            tokenizer: HuggingFace tokenizer (optional)
            enable_trace: Whether to use trace capture for decode (default: True)
            batch_size: Maximum batch size for generation
        """
        self.model = model
        self.device = model.device
        self.config = model.config
        self.tokenizer = tokenizer
        self.enable_trace = enable_trace
        self.batch_size = batch_size

        # Trace state
        self._trace_id: Optional[int] = None
        self._trace_inputs: Optional[list] = None
        self._trace_output: Optional[ttnn.Tensor] = None

        # Initialize RotarySetup for optimized RoPE (tt_transformers pattern)
        # This pre-computes and uploads cos/sin matrices ONCE
        self.rotary_setup = RotarySetup(
            device=self.device,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_position_embeddings,
            rope_theta=self.config.rope_theta,
            batch_size=batch_size,
        )

        # Pre-allocate KV caches for all layers
        self._preallocated_kv_caches: Optional[list[KVCache]] = None

        # Load default tokenizer if not provided
        if self.tokenizer is None:
            self._load_default_tokenizer()

    def _load_default_tokenizer(self) -> None:
        """Load the default tokenizer for BitNet."""
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T")
        except ImportError:
            print("Warning: transformers not installed. Cannot load tokenizer.")
            self.tokenizer = None

    def _preallocate_kv_caches(self) -> list[KVCache]:
        """
        Pre-allocate KV caches for all layers.

        This is required for the optimized decode path using update_cache.
        Cache format: [batch, num_kv_heads, max_seq_len, head_dim] (NON-EXPANDED)

        Key insight: Cache stores raw KV heads (5). GQA expansion happens
        when returning from cache for attention. This matches update_cache API.
        """
        caches = []
        for layer_idx in range(self.config.num_layers):
            cache = KVCache(
                max_seq_len=self.config.max_position_embeddings,
                batch_size=self.batch_size,
                num_kv_heads=self.config.num_key_value_heads,
                # num_heads NOT passed - cache stores non-expanded kv_heads
                head_dim=self.config.head_dim,
                device=self.device,
            )
            cache.preallocate(
                batch_size=self.batch_size,
                num_kv_heads=self.config.num_key_value_heads,
                max_seq_len=self.config.max_position_embeddings,
                head_dim=self.config.head_dim,
                device=self.device,
                # num_heads=None -> cache with num_kv_heads shape for update_cache
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
    ) -> tuple[ttnn.Tensor, list[KVCache]]:
        """
        Process prompt tokens (prefill phase).

        Args:
            tokens: Input token IDs (batch, seq_len)
            kv_cache: Optional existing KV cache
            use_preallocated: Use pre-allocated KV caches for optimized decode

        Returns:
            (logits, updated_kv_cache)
        """
        # Optionally use pre-allocated caches for optimized decode path
        if use_preallocated and kv_cache is None:
            if self._preallocated_kv_caches is None:
                self._preallocated_kv_caches = self._preallocate_kv_caches()
            # Reset caches for new generation
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

        # Auto-detect whether to use optimized path based on cache state
        # NOTE: Optimized path with rotary_embedding_llama requires HEIGHT_SHARDED inputs
        # which is not yet properly configured. Using basic decode path for now.
        if use_optimized is None:
            # Disable optimized path until HEIGHT_SHARDED memory config is fixed
            use_optimized = False
            # Original logic (requires HEIGHT_SHARDED fix):
            # use_optimized = (
            #     kv_cache is not None
            #     and len(kv_cache) > 0
            #     and kv_cache[0] is not None
            #     and hasattr(kv_cache[0], '_preallocated')
            #     and kv_cache[0]._preallocated
            # )

        # Get rot_mats for optimized decode path
        rot_mats = None
        transformation_mat = None
        if use_optimized:
            rot_mats = self.get_rot_mats_decode(current_pos)
            transformation_mat = self.get_transformation_mat()

        logits, kv_cache = self.model(
            input_tensor,
            past_key_values=kv_cache,
            use_cache=True,
            mode="decode",
            current_pos=current_pos,
            rot_mats=rot_mats,
            transformation_mat=transformation_mat,
        )
        ttnn.deallocate(input_tensor)
        return logits, kv_cache

    def _capture_decode_trace(
        self,
        token: NDArray[np.int64],
        current_pos: int,
        kv_cache: list[KVCache],
    ) -> tuple[int, ttnn.Tensor, list[ttnn.Tensor]]:
        """
        Capture a trace for the decode forward pass.

        This compiles the decode graph once and reuses it for all subsequent
        decode steps, eliminating compilation overhead (2-3x speedup).
        Also manages persistent pos_tensor to avoid device writes during trace.

        Returns:
            (trace_id, output_tensor, [input_tensor, pos_tensor])
        """
        # Compile run (warm up)
        input_tensor = numpy_int_to_ttnn(token, self.device)
        # Create persistent pos_tensor for trace
        pos_tensor = ttnn.from_torch(
            torch.tensor([[current_pos]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        ttnn.synchronize_device(self.device)  # Ensure buffer is allocated

        _, _ = self.model(
            input_tensor,
            past_key_values=kv_cache,
            use_cache=True,
            mode="decode",
            current_pos=current_pos,
            pos_tensor=pos_tensor,  # Pass persistent tensor
        )
        ttnn.synchronize_device(self.device)

        # Capture trace
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        logits, _ = self.model(
            input_tensor,
            past_key_values=kv_cache,
            use_cache=True,
            mode="decode",
            current_pos=current_pos,
            pos_tensor=pos_tensor,  # Pass persistent tensor
        )
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

        return trace_id, logits, [input_tensor, pos_tensor]

    def _execute_decode_trace(
        self,
        token: NDArray[np.int64],
        current_pos: int,
    ) -> ttnn.Tensor:
        """
        Execute the captured decode trace with new input.

        Args:
            token: New token to process
            current_pos: Current position in sequence (for KV cache update)

        Returns:
            Output logits tensor
        """
        # 1. Update Input Token
        new_input = numpy_int_to_ttnn(token, self.device)
        ttnn.copy_host_to_device_tensor(new_input, self._trace_inputs[0])

        # 2. Update Position Tensor
        new_pos = ttnn.from_torch(
            torch.tensor([[current_pos]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(new_pos, self._trace_inputs[1])

        # Execute trace
        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)

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

        # Phase 1: Prefill - process all prompt tokens
        # Use pre-allocated cache with in-place update for Trace compatibility
        logits, kv_cache = self.prefill_forward(input_ids, use_preallocated=use_optimized)

        # Sample first token
        next_token = self._sample_next_token(logits, temperature, top_k, do_sample)
        ttnn.deallocate(logits)

        generated = np.concatenate([generated, next_token.reshape(batch_size, 1)], axis=1)

        # Check EOS
        if self._is_eos(next_token[0]):
            return generated

        # Phase 2: Decode - generate remaining tokens
        current_pos = seq_len

        for step in range(max_new_tokens - 1):
            current_pos += 1

            # Decode forward with trace capture if enabled
            if self.enable_trace:
                if self._trace_id is None:
                    # First decode step: capture trace
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

            # Sample next token
            next_token = self._sample_next_token(logits, temperature, top_k, do_sample)
            ttnn.deallocate(logits)

            generated = np.concatenate([generated, next_token.reshape(batch_size, 1)], axis=1)

            if self._is_eos(next_token[0]):
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

        # Phase 1: Prefill - use pre-allocated cache with in-place update
        prefill_start = time.perf_counter()
        logits, kv_cache = self.prefill_forward(input_ids, use_preallocated=use_optimized)
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
            current_pos += 1

            # Decode forward with trace capture if enabled
            if self.enable_trace:
                if self._trace_id is None:
                    # First decode step: capture trace
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

            token_end = time.perf_counter()
            token_time = token_end - token_start
            stats.token_times.append(token_time)
            stats.generation_time += token_time

            next_token = self._sample_next_token(logits, temperature, top_k, do_sample)
            ttnn.deallocate(logits)

            generated = np.concatenate([generated, next_token.reshape(batch_size, 1)], axis=1)
            stats.generated_tokens += 1

            # Yield new text
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
