#!/usr/bin/env python3
"""
BitNet-TT Demo

This script demonstrates running BitNet on Tenstorrent hardware using TT-NN.

Usage:
    python examples/demo.py [--mini]
    python examples/demo.py --full  # Load full 2B model
"""

import argparse
import time

import numpy as np
import ttnn

from bitnet_tt.config import BitNetMiniConfig
from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn, ttnn_to_numpy
from bitnet_tt.utils.device import device_context, is_ttnn_available


def run_mini_demo() -> None:
    """Run demo with mini model for testing."""
    print("=" * 60)
    print("BitNet-TT Mini Model Demo (TT-NN)")
    print("=" * 60)

    if not is_ttnn_available():
        print("\nError: TT-NN is not available.")
        print("Please install the Tenstorrent SDK and drivers.")
        return

    from bitnet_tt.model.bitnet import create_model

    with device_context() as device:
        # Create mini model
        print("\n[1] Creating mini BitNet model...")
        config = BitNetMiniConfig()
        model = create_model(config, device)

        print(f"    - Hidden size: {config.hidden_size}")
        print(f"    - Num layers: {config.num_layers}")
        print(f"    - Num heads: {config.num_attention_heads}")
        print(f"    - Vocab size: {config.vocab_size}")

        # Initialize with random weights for testing
        print("\n[2] Initializing with random weights...")
        _init_random_weights(model, config, device)
        print("    - Weights initialized")

        # Run inference without cache
        print("\n[3] Running inference (without KV-Cache)...")
        batch_size = 1
        seq_len = 32

        # Random input tokens
        input_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
        input_tensor = numpy_int_to_ttnn(input_ids, device)

        # Warmup
        logits, _ = model(input_tensor, use_cache=False)
        ttnn.synchronize_device(device)

        # Timed run
        start = time.perf_counter()
        logits, _ = model(input_tensor, use_cache=False)
        ttnn.synchronize_device(device)
        end = time.perf_counter()

        logits_np = ttnn_to_numpy(logits)
        print(f"    - Input shape: {input_ids.shape}")
        print(f"    - Output shape: {logits_np.shape}")
        print(f"    - Inference time: {(end - start) * 1000:.2f} ms")

        # Test with KV-Cache
        print("\n[4] Testing KV-Cache generation...")

        # First pass - process prompt
        prompt_ids = np.random.randint(0, config.vocab_size, (batch_size, 8))
        prompt_tensor = numpy_int_to_ttnn(prompt_ids, device)

        start = time.perf_counter()
        logits, past_key_values = model(prompt_tensor, use_cache=True)

        # Generate 5 tokens
        generated_tokens = []
        for i in range(5):
            # Get next token
            logits_np = ttnn_to_numpy(logits)
            next_token = np.argmax(logits_np[:, -1, :], axis=-1)
            generated_tokens.append(next_token[0])

            # Process next token with cache
            next_tensor = numpy_int_to_ttnn(next_token.reshape(1, 1), device)
            logits, past_key_values = model(
                next_tensor,
                past_key_values=past_key_values,
                use_cache=True,
            )

        ttnn.synchronize_device(device)
        end = time.perf_counter()

        print(f"    - Prompt length: {prompt_ids.shape[1]}")
        print(f"    - Generated {len(generated_tokens)} tokens: {generated_tokens}")
        print(f"    - Generation time: {(end - start) * 1000:.2f} ms")
        print(f"    - Time per token: {(end - start) * 1000 / (len(generated_tokens) + 1):.2f} ms")

        # Simple greedy decoding test
        print("\n[5] Testing greedy decoding...")
        next_token = np.argmax(logits_np[:, -1, :], axis=-1)
        print(f"    - Input tokens: {prompt_ids[0, :8].tolist()}")
        print(f"    - Next token prediction: {next_token[0]}")

        print("\n" + "=" * 60)
        print("Mini model demo completed successfully!")
        print("=" * 60)


def run_full_demo() -> None:
    """Run demo with full 2B model."""
    print("=" * 60)
    print("BitNet-TT Full Model Demo (2B-4T)")
    print("=" * 60)

    if not is_ttnn_available():
        print("\nError: TT-NN is not available.")
        print("Please install the Tenstorrent SDK and drivers.")
        return

    from bitnet_tt.inference.generator import TextGenerator
    from bitnet_tt.model.bitnet import create_model
    from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model

    with device_context() as device:
        # Load weights from HuggingFace
        print("\n[1] Loading weights from HuggingFace...")
        print("    This may take a while on first run...")

        try:
            # Use BF16 version (unpacked weights, easier to load)
            state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
        except Exception as e:
            print(f"\nError loading weights: {e}")
            print("Please ensure you have internet access and huggingface_hub installed.")
            return

        print(f"    - Loaded {len(state_dict)} weight tensors")
        print(f"    - Model config: {config.hidden_size}d, {config.num_layers}L")

        # Debug: print some keys to see structure
        print("    - Sample keys:")
        for i, key in enumerate(sorted(state_dict.keys())):
            if i < 10:
                print(f"      {key}: {state_dict[key].shape}")

        # Create model
        print("\n[2] Creating model...")
        model = create_model(config, device)

        # Load weights into model
        print("\n[3] Loading weights into model...")
        load_weights_to_model(model, state_dict)
        print("    - Weights loaded successfully")

        # Create generator
        print("\n[4] Creating text generator...")
        generator = TextGenerator(model)

        if generator.tokenizer is None:
            print("    Warning: Tokenizer not available. Skipping text generation.")
            return

        print("    - Generator ready")

        # Generate text with optimized 1BKD decode path
        print("\n[5] Generating text (with optimized decode)...")
        prompt = "Hello, I am"
        print(f"    Prompt: {prompt}")

        start = time.perf_counter()
        output = generator.generate(
            prompt,
            max_new_tokens=30,
            temperature=0.7,
            do_sample=True,
        )
        end = time.perf_counter()

        print(f"    Output: {output}")
        print(f"    Generation time: {(end - start):.2f}s")

        print("\n" + "=" * 60)
        print("Full model demo completed successfully!")
        print("=" * 60)


def _init_random_weights(model, config, device) -> None:
    """Initialize model with random weights for testing."""

    # Embedding weights
    embed_weight = np.random.randn(config.vocab_size, config.hidden_size).astype(
        np.float32
    ) * 0.02
    model.load_embedding_weights(embed_weight)

    # Layer weights
    for layer_idx in range(config.num_layers):
        layer_weights = {}

        head_dim = config.hidden_size // config.num_attention_heads

        # RMSNorm weights (initialized to 1)
        layer_weights["input_layernorm.weight"] = np.ones(config.hidden_size, dtype=np.float32)
        layer_weights["post_attention_layernorm.weight"] = np.ones(
            config.hidden_size, dtype=np.float32
        )

        # Attention weights (plain linear, no per-projection norms)
        layer_weights["self_attn.q_proj.weight"] = np.random.randn(
            config.num_attention_heads * head_dim, config.hidden_size
        ).astype(np.float32) * 0.02

        layer_weights["self_attn.k_proj.weight"] = np.random.randn(
            config.num_key_value_heads * head_dim, config.hidden_size
        ).astype(np.float32) * 0.02

        layer_weights["self_attn.v_proj.weight"] = np.random.randn(
            config.num_key_value_heads * head_dim, config.hidden_size
        ).astype(np.float32) * 0.02

        layer_weights["self_attn.o_proj.weight"] = np.random.randn(
            config.hidden_size, config.num_attention_heads * head_dim
        ).astype(np.float32) * 0.02

        # Attention sub-norm (applied after attention output, before o_proj)
        layer_weights["self_attn.attn_sub_norm.weight"] = np.ones(
            config.hidden_size, dtype=np.float32
        )

        # FFN weights (plain linear, no per-projection norms)
        layer_weights["mlp.gate_proj.weight"] = np.random.randn(
            config.intermediate_size, config.hidden_size
        ).astype(np.float32) * 0.02

        layer_weights["mlp.up_proj.weight"] = np.random.randn(
            config.intermediate_size, config.hidden_size
        ).astype(np.float32) * 0.02

        layer_weights["mlp.down_proj.weight"] = np.random.randn(
            config.hidden_size, config.intermediate_size
        ).astype(np.float32) * 0.02

        # FFN sub-norm (applied after gate*up, before down_proj)
        layer_weights["mlp.ffn_sub_norm.weight"] = np.ones(
            config.intermediate_size, dtype=np.float32
        )

        model.load_layer_weights(layer_idx, layer_weights)

    # Final norm and LM head
    norm_weight = np.ones(config.hidden_size, dtype=np.float32)
    lm_head_weight = np.random.randn(config.vocab_size, config.hidden_size).astype(
        np.float32
    ) * 0.02
    model.load_final_weights(norm_weight, lm_head_weight)


def main() -> None:
    parser = argparse.ArgumentParser(description="BitNet-TT Demo")
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Run with mini model (default)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run with full 2B model from HuggingFace",
    )
    args = parser.parse_args()

    if args.full:
        run_full_demo()
    else:
        run_mini_demo()


if __name__ == "__main__":
    main()
