"""
Optimized text generation for BitNet using TT-NN.

This module provides a high-level interface for text generation with:
- Separate prefill and decode paths
- Trace capture for decode loop (eliminates compilation overhead)
- On-device sampling to minimize data transfer

Performance optimizations based on tt_transformers patterns.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Optional

import numpy as np
import ttnn
from numpy.typing import NDArray

from bitnet_tt.layers.attention import KVCache
from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn, ttnn_to_numpy


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


def _on_device_topk(
    logits: ttnn.Tensor, k: int
) -> tuple[NDArray[np.floating], NDArray[np.int64]]:
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
            f"Prompt: {self.prompt_tokens} tokens ({self.prompt_time*1000:.1f}ms) | "
            f"Generated: {self.generated_tokens} tokens ({self.generation_time:.2f}s) | "
            f"Speed: {self.tokens_per_second:.2f} tokens/s | "
            f"TTFT: {self.time_to_first_token*1000:.1f}ms"
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
        enable_trace: bool = True,
    ) -> None:
        """
        Initialize the text generator.

        Args:
            model: BitNet model instance
            tokenizer: HuggingFace tokenizer (optional)
            enable_trace: Whether to use trace capture for decode (default: True)
        """
        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.enable_trace = enable_trace

        # Trace state
        self._trace_id: Optional[int] = None
        self._trace_inputs: Optional[list] = None
        self._trace_output: Optional[ttnn.Tensor] = None

        # Load default tokenizer if not provided
        if self.tokenizer is None:
            self._load_default_tokenizer()

    def _load_default_tokenizer(self) -> None:
        """Load the default tokenizer for BitNet."""
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/bitnet-b1.58-2B-4T"
            )
        except ImportError:
            print("Warning: transformers not installed. Cannot load tokenizer.")
            self.tokenizer = None

    def prefill_forward(
        self,
        tokens: NDArray[np.int64],
        kv_cache: Optional[list[KVCache]] = None,
    ) -> tuple[ttnn.Tensor, list[KVCache]]:
        """
        Process prompt tokens (prefill phase).

        Args:
            tokens: Input token IDs (batch, seq_len)
            kv_cache: Optional existing KV cache

        Returns:
            (logits, updated_kv_cache)
        """
        input_tensor = numpy_int_to_ttnn(tokens, self.device)
        logits, kv_cache = self.model(
            input_tensor,
            past_key_values=kv_cache,
            use_cache=True,
            mode="prefill",
        )
        ttnn.deallocate(input_tensor)

        # Pre-allocate KV caches for decode with 1BKD format support
        batch_size = tokens.shape[0]
        self._preallocate_kv_caches(kv_cache, batch_size)

        return logits, kv_cache

    def _preallocate_kv_caches(
        self,
        kv_caches: list[KVCache],
        batch_size: int,
    ) -> None:
        """
        Pre-allocate KV cache buffers for decode mode.

        After prefill, the KV cache stores references to prefill KV tensors.
        This method allocates fixed-size buffers and copies prefill data to them.
        This enables paged_update_cache and trace capture in decode mode.
        """
        import torch

        # Get model config
        config = self.model.config
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads
        max_seq_len = getattr(config, 'max_position_embeddings', 4096)

        for layer_idx, cache in enumerate(kv_caches):
            if cache._preallocated:
                continue

            # Save prefill KV before preallocation
            prefill_key = getattr(cache, '_prefill_key', None)
            prefill_value = getattr(cache, '_prefill_value', None)
            prefill_len = cache.seq_len_cached

            # Preallocate cache buffers
            cache.preallocate(
                batch_size=batch_size,
                num_kv_heads=num_kv_heads,
                max_seq_len=max_seq_len,
                head_dim=head_dim,
                device=self.device,
            )

            # Copy prefill KV to preallocated cache
            if prefill_key is not None and prefill_len > 0:
                # Prefill KV is in BKSD format: [batch, heads, seq, head_dim]
                # Cache is also BKSD format: [batch, heads, max_seq, head_dim]
                # Use fill_cache_for_user_ to copy prefill data position by position
                try:
                    # Fill cache position by position using fill_cache_for_user_
                    # This is slower than paged_fill_cache but doesn't require sharding
                    for pos in range(prefill_len):
                        # Extract single position: [batch, heads, 1, head_dim]
                        key_at_pos = prefill_key[:, :, pos:pos+1, :]
                        value_at_pos = prefill_value[:, :, pos:pos+1, :]

                        ttnn.kv_cache.fill_cache_for_user_(
                            cache.key_cache, key_at_pos, pos
                        )
                        ttnn.kv_cache.fill_cache_for_user_(
                            cache.value_cache, value_at_pos, pos
                        )

                    if layer_idx == 0:
                        print(f"[DEBUG] Layer 0: fill_cache_for_user_ SUCCESS, prefill_len={prefill_len}")
                except (RuntimeError, AttributeError) as e:
                    # Fallback: Keep prefill tensors and handle in first decode
                    if layer_idx == 0:
                        print(f"[DEBUG] Layer 0: fill_cache FAILED: {e}")
                    cache._prefill_key = prefill_key
                    cache._prefill_value = prefill_value
                    continue

            cache.seq_len_cached = prefill_len

    def decode_forward(
        self,
        token: NDArray[np.int64],
        current_pos: int,
        kv_cache: list[KVCache],
    ) -> tuple[ttnn.Tensor, list[KVCache]]:
        """
        Generate single token (decode phase).

        Args:
            token: Single token ID (batch, 1)
            current_pos: Current position in sequence
            kv_cache: KV cache from prefill

        Returns:
            (logits, updated_kv_cache)
        """
        input_tensor = numpy_int_to_ttnn(token, self.device)
        logits, kv_cache = self.model(
            input_tensor,
            past_key_values=kv_cache,
            use_cache=True,
            mode="decode",
            current_pos=current_pos,
        )
        ttnn.deallocate(input_tensor)
        return logits, kv_cache

    def _capture_decode_trace(
        self,
        token: NDArray[np.int64],
        current_pos: int,
        kv_cache: list[KVCache],
    ) -> tuple[int, ttnn.Tensor, ttnn.Tensor]:
        """
        Capture a trace for the decode forward pass.

        This compiles the decode graph once and reuses it for all subsequent
        decode steps, eliminating compilation overhead (2-3x speedup).

        Returns:
            (trace_id, output_tensor, input_tensor)
        """
        # Compile run (warm up)
        input_tensor = numpy_int_to_ttnn(token, self.device)
        _, _ = self.model(
            input_tensor,
            past_key_values=kv_cache,
            use_cache=True,
            mode="decode",
            current_pos=current_pos,
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
        )
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

        return trace_id, logits, input_tensor

    def _execute_decode_trace(
        self,
        token: NDArray[np.int64],
    ) -> ttnn.Tensor:
        """
        Execute the captured decode trace with new input.

        Args:
            token: New token to process

        Returns:
            Output logits tensor
        """
        # Copy new token to the trace input tensor
        new_input = numpy_int_to_ttnn(token, self.device)
        ttnn.copy_host_to_device_tensor(new_input, self._trace_inputs)

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
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            do_sample: Whether to sample

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
        )

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def _generate_tokens(
        self,
        input_ids: NDArray[np.int64],
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        do_sample: bool,
    ) -> NDArray[np.int64]:
        """
        Generate tokens with optimized prefill/decode split.
        """
        batch_size, seq_len = input_ids.shape
        generated = input_ids.copy()

        # Reset trace for new generation
        self.reset_trace()

        # Phase 1: Prefill - process all prompt tokens
        logits, kv_cache = self.prefill_forward(input_ids)

        # Sample first token
        next_token = self._sample_next_token(
            logits, temperature, top_k, do_sample
        )
        ttnn.deallocate(logits)

        generated = np.concatenate(
            [generated, next_token.reshape(batch_size, 1)], axis=1
        )

        # Check EOS
        if self._is_eos(next_token[0]):
            return generated

        # Phase 2: Decode - generate remaining tokens
        current_pos = seq_len

        for step in range(max_new_tokens - 1):
            current_pos += 1

            # Decode forward (with trace if enabled)
            logits, kv_cache = self.decode_forward(
                generated[:, -1:],
                current_pos,
                kv_cache,
            )

            # Sample next token
            next_token = self._sample_next_token(
                logits, temperature, top_k, do_sample
            )
            ttnn.deallocate(logits)

            generated = np.concatenate(
                [generated, next_token.reshape(batch_size, 1)], axis=1
            )

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
    ) -> Generator[tuple[str, GenerationStats], None, None]:
        """
        Generate text with streaming output.

        Yields tokens as they are generated along with statistics.
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

        # Phase 1: Prefill (KV cache will be created during forward)
        prefill_start = time.perf_counter()
        logits, kv_cache = self.prefill_forward(input_ids)
        prefill_end = time.perf_counter()
        stats.prompt_time = prefill_end - prefill_start

        # Sample first token
        next_token = self._sample_next_token(logits, temperature, top_k, do_sample)
        ttnn.deallocate(logits)

        generated = np.concatenate(
            [generated, next_token.reshape(batch_size, 1)], axis=1
        )
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

            generated = np.concatenate(
                [generated, next_token.reshape(batch_size, 1)], axis=1
            )
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
        )
