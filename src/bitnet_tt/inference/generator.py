"""
Text generation utilities for BitNet using TT-NN.

This module provides a high-level interface for text generation
using BitNet models on Tenstorrent hardware with KV-Cache support.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator

import numpy as np
import ttnn
from numpy.typing import NDArray

from bitnet_tt.layers.attention import KVCache
from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn, ttnn_to_numpy


def _on_device_argmax(logits: ttnn.Tensor) -> NDArray[np.int64]:
    """
    Perform argmax on device and return only the token index.

    This avoids transferring the full logits tensor (128KB for vocab_size=128256)
    to CPU. Instead, only the resulting token ID(s) are transferred.

    Args:
        logits: TT-NN tensor of shape (batch, seq, vocab_size)

    Returns:
        Token indices of shape (batch,)
    """
    # Get last position logits: (batch, seq, vocab) -> (batch, 1, vocab)
    # Need ROW_MAJOR for slicing
    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    last_logits = logits_rm[:, -1:, :]  # (batch, 1, vocab_size)

    # Perform argmax on the last dimension (vocab_size)
    # Returns indices of shape (batch, 1)
    last_logits_tile = ttnn.to_layout(last_logits, ttnn.TILE_LAYOUT)
    token_indices = ttnn.argmax(last_logits_tile, dim=-1)

    # Transfer only the token indices back to CPU (4 bytes per batch element)
    indices_np = ttnn.to_torch(token_indices).numpy().astype(np.int64)
    return indices_np.flatten()


def _on_device_topk(
    logits: ttnn.Tensor, k: int
) -> tuple[NDArray[np.floating], NDArray[np.int64]]:
    """
    Perform top-k selection on device and return only the top-k values and indices.

    This avoids transferring the full logits tensor (128KB for vocab_size=128256)
    to CPU. Instead, only the top-k values and indices are transferred.

    Args:
        logits: TT-NN tensor of shape (batch, seq, vocab_size)
        k: Number of top elements to return

    Returns:
        Tuple of (top_k_values, top_k_indices) each of shape (batch, k)
    """
    # Get last position logits: (batch, seq, vocab) -> (batch, 1, vocab)
    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    last_logits = logits_rm[:, -1:, :]  # (batch, 1, vocab_size)
    last_logits_tile = ttnn.to_layout(last_logits, ttnn.TILE_LAYOUT)

    # Perform top-k on device
    # ttnn.topk returns (values, indices) tensors
    top_k_values, top_k_indices = ttnn.topk(last_logits_tile, k, dim=-1)

    # Transfer only top-k data to CPU (k * 8 bytes per batch element)
    values_np = ttnn.to_torch(top_k_values).float().numpy().squeeze(1)  # (batch, k)
    indices_np = ttnn.to_torch(top_k_indices).numpy().astype(np.int64).squeeze(1)

    return values_np, indices_np

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
    TT-NN native text generator for BitNet.

    Generates text on Tenstorrent hardware using autoregressive decoding
    with KV-Cache for efficient generation.
    """

    def __init__(
        self,
        model: "BitNetModel",
        tokenizer: Any = None,
    ) -> None:
        """
        Initialize the text generator.

        Args:
            model: BitNet model instance
            tokenizer: HuggingFace tokenizer (optional, will load default if not provided)
        """
        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer

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

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = 50,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (None to disable)
            do_sample: Whether to sample (False for greedy decoding)
            use_cache: Whether to use KV-Cache for efficient generation

        Returns:
            Generated text including the prompt
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Please provide a tokenizer.")

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids: NDArray[np.int64] = inputs["input_ids"]

        # Generate tokens
        generated_ids = self._generate_tokens(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            use_cache=use_cache,
        )

        # Decode to text
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return output_text

    def _generate_tokens(
        self,
        input_ids: NDArray[np.int64],
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        do_sample: bool,
        use_cache: bool = True,
    ) -> NDArray[np.int64]:
        """
        Generate tokens autoregressively with optional KV-Cache.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            do_sample: Whether to sample
            use_cache: Whether to use KV-Cache

        Returns:
            Generated token IDs of shape (batch, seq_len + max_new_tokens)
        """
        batch_size, seq_len = input_ids.shape
        generated = input_ids.copy()

        # Initialize KV-Cache
        past_key_values: list[KVCache] | None = None

        for step in range(max_new_tokens):
            if use_cache and past_key_values is not None:
                # Only process the new token (KV-Cache optimization)
                current_input = generated[:, -1:]
            else:
                # Process full sequence (first iteration or no cache)
                current_input = generated

            # Convert current input to TT-NN tensor
            input_tensor = numpy_int_to_ttnn(current_input, self.device)

            # Forward pass with KV-Cache
            logits, past_key_values = self.model(
                input_tensor,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

            # Deallocate input tensor to free device memory
            ttnn.deallocate(input_tensor)

            # Sample next token
            if not do_sample:
                # Greedy decoding: use on-device argmax (avoids 128KB transfer)
                next_token = _on_device_argmax(logits)
            elif top_k is not None and top_k > 0:
                # Top-k sampling: use on-device top-k (transfers only k values)
                top_k_values, top_k_indices = _on_device_topk(logits, top_k)
                next_token = self._sample_from_topk(
                    top_k_values, top_k_indices, temperature
                )
            else:
                # Full distribution sampling: need full logits on CPU
                logits_np = ttnn_to_numpy(logits)
                next_token_logits = logits_np[:, -1, :]  # (batch, vocab_size)
                next_token = self._sample_token(
                    next_token_logits,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=do_sample,
                )

            # Deallocate logits to free device memory (large tensor: 128K vocab)
            ttnn.deallocate(logits)

            # Append to generated sequence
            generated = np.concatenate(
                [generated, next_token.reshape(batch_size, 1)], axis=1
            )

            # Check for EOS token
            if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
                if next_token[0] == self.tokenizer.eos_token_id:
                    break

        return generated

    def _generate_tokens_no_cache(
        self,
        input_ids: NDArray[np.int64],
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        do_sample: bool,
    ) -> NDArray[np.int64]:
        """
        Generate tokens autoregressively without KV-Cache (legacy method).

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            do_sample: Whether to sample

        Returns:
            Generated token IDs of shape (batch, seq_len + max_new_tokens)
        """
        batch_size, seq_len = input_ids.shape
        generated = input_ids.copy()

        for _ in range(max_new_tokens):
            # Convert current sequence to TT-NN tensor
            input_tensor = numpy_int_to_ttnn(generated, self.device)

            # Forward pass without cache
            logits, _ = self.model(input_tensor, use_cache=False)

            # Deallocate input tensor to free device memory
            ttnn.deallocate(input_tensor)

            # Sample next token
            if not do_sample:
                # Greedy decoding: use on-device argmax (avoids 128KB transfer)
                next_token = _on_device_argmax(logits)
            elif top_k is not None and top_k > 0:
                # Top-k sampling: use on-device top-k (transfers only k values)
                top_k_values, top_k_indices = _on_device_topk(logits, top_k)
                next_token = self._sample_from_topk(
                    top_k_values, top_k_indices, temperature
                )
            else:
                # Full distribution sampling: need full logits on CPU
                logits_np = ttnn_to_numpy(logits)
                next_token_logits = logits_np[:, -1, :]
                next_token = self._sample_token(
                    next_token_logits,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=do_sample,
                )

            # Deallocate logits to free device memory (large tensor: 128K vocab)
            ttnn.deallocate(logits)

            # Append to generated sequence
            generated = np.concatenate(
                [generated, next_token.reshape(batch_size, 1)], axis=1
            )

            # Check for EOS token
            if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
                if next_token[0] == self.tokenizer.eos_token_id:
                    break

        return generated

    def _sample_from_topk(
        self,
        top_k_values: NDArray[np.floating],
        top_k_indices: NDArray[np.int64],
        temperature: float,
    ) -> NDArray[np.int64]:
        """
        Sample a token from pre-computed top-k values and indices.

        This is used with on-device top-k to avoid transferring full logits.

        Args:
            top_k_values: Top-k logit values of shape (batch, k)
            top_k_indices: Top-k token indices of shape (batch, k)
            temperature: Sampling temperature

        Returns:
            Sampled token IDs of shape (batch,)
        """
        batch_size = top_k_values.shape[0]
        k = top_k_values.shape[1]

        # Apply temperature
        if temperature != 1.0:
            top_k_values = top_k_values / temperature

        # Convert to probabilities
        probs = _softmax(top_k_values)

        # Sample from top-k
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
        """
        Sample a token from logits.

        Args:
            logits: Logits array of shape (batch, vocab_size)
            temperature: Sampling temperature
            top_k: Top-k filtering
            do_sample: Whether to sample

        Returns:
            Sampled token IDs of shape (batch,)
        """
        batch_size = logits.shape[0]

        if not do_sample:
            # Greedy decoding
            return np.argmax(logits, axis=-1)

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            # Get top-k indices and values
            top_k_indices = np.argpartition(logits, -top_k, axis=-1)[:, -top_k:]
            top_k_logits = np.take_along_axis(logits, top_k_indices, axis=-1)

            # Convert to probabilities
            probs = _softmax(top_k_logits)

            # Sample from top-k
            sampled_indices = np.array(
                [
                    np.random.choice(top_k, p=probs[i])
                    for i in range(batch_size)
                ]
            )
            return np.take_along_axis(
                top_k_indices, sampled_indices.reshape(-1, 1), axis=-1
            ).flatten()
        else:
            # Sample from full distribution
            probs = _softmax(logits)
            return np.array(
                [
                    np.random.choice(logits.shape[-1], p=probs[i])
                    for i in range(batch_size)
                ]
            )

    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = 50,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> Generator[tuple[str, GenerationStats], None, None]:
        """
        Generate text with streaming output.

        Yields tokens as they are generated along with statistics.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            do_sample: Whether to sample
            use_cache: Whether to use KV-Cache

        Yields:
            Tuple of (new_text, stats) for each generated token
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Please provide a tokenizer.")

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids: NDArray[np.int64] = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape

        stats = GenerationStats(prompt_tokens=seq_len)

        # Initialize
        generated = input_ids.copy()
        past_key_values: list[KVCache] | None = None
        prev_text_len = len(self.tokenizer.decode(generated[0], skip_special_tokens=True))

        for step in range(max_new_tokens):
            token_start = time.perf_counter()

            if use_cache and past_key_values is not None:
                current_input = generated[:, -1:]
            else:
                current_input = generated

            # Forward pass
            input_tensor = numpy_int_to_ttnn(current_input, self.device)
            logits, past_key_values = self.model(
                input_tensor,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

            # Deallocate input tensor to free device memory
            ttnn.deallocate(input_tensor)

            token_end = time.perf_counter()
            token_time = token_end - token_start

            # Record timing
            if step == 0:
                stats.prompt_time = token_time
            else:
                stats.token_times.append(token_time)
                stats.generation_time += token_time

            # Sample next token
            if not do_sample:
                # Greedy decoding: use on-device argmax (avoids 128KB transfer)
                next_token = _on_device_argmax(logits)
            elif top_k is not None and top_k > 0:
                # Top-k sampling: use on-device top-k (transfers only k values)
                top_k_values, top_k_indices = _on_device_topk(logits, top_k)
                next_token = self._sample_from_topk(
                    top_k_values, top_k_indices, temperature
                )
            else:
                # Full distribution sampling: need full logits on CPU
                logits_np = ttnn_to_numpy(logits)
                next_token_logits = logits_np[:, -1, :]
                next_token = self._sample_token(
                    next_token_logits,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=do_sample,
                )

            # Deallocate logits to free device memory (large tensor: 128K vocab)
            ttnn.deallocate(logits)

            # Append token
            generated = np.concatenate(
                [generated, next_token.reshape(batch_size, 1)], axis=1
            )
            stats.generated_tokens += 1

            # Decode new text
            current_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            new_text = current_text[prev_text_len:]
            prev_text_len = len(current_text)

            yield new_text, stats

            # Check for EOS
            if self.tokenizer.eos_token_id is not None:
                if next_token[0] == self.tokenizer.eos_token_id:
                    break

    def chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        use_cache: bool = True,
    ) -> str:
        """
        Generate a response in chat format.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_cache: Whether to use KV-Cache

        Returns:
            Assistant's response
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")

        # Format chat using tokenizer's chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate response
        response = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            use_cache=use_cache,
        )

        return response

    def chat_streaming(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        use_cache: bool = True,
    ) -> Generator[tuple[str, GenerationStats], None, None]:
        """
        Generate a chat response with streaming output.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_cache: Whether to use KV-Cache

        Yields:
            Tuple of (new_text, stats) for each generated token
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")

        # Format chat using tokenizer's chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Track prompt length to extract only new content
        prompt_text_len = len(prompt)

        # Generate with streaming
        for new_text, stats in self.generate_streaming(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            use_cache=use_cache,
        ):
            yield new_text, stats


def _softmax(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute softmax along the last axis."""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
