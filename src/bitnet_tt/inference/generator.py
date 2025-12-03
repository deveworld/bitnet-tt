"""
Text generation utilities for BitNet using TT-NN.

This module provides a high-level interface for text generation
using BitNet models on Tenstorrent hardware with KV-Cache support.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from bitnet_tt.layers.attention import KVCache
from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn, ttnn_to_numpy

if TYPE_CHECKING:
    from bitnet_tt.model.bitnet import BitNetModel


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

            # Get logits for the last position
            # logits shape: (batch, seq, vocab_size)
            logits_np = ttnn_to_numpy(logits)
            next_token_logits = logits_np[:, -1, :]  # (batch, vocab_size)

            # Sample next token
            next_token = self._sample_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
            )

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

            # Get logits for the last position
            logits_np = ttnn_to_numpy(logits)
            next_token_logits = logits_np[:, -1, :]

            # Sample next token
            next_token = self._sample_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
            )

            # Append to generated sequence
            generated = np.concatenate(
                [generated, next_token.reshape(batch_size, 1)], axis=1
            )

            # Check for EOS token
            if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
                if next_token[0] == self.tokenizer.eos_token_id:
                    break

        return generated

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


def _softmax(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute softmax along the last axis."""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
