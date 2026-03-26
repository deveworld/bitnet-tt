#!/usr/bin/env python3
"""
Test script for Batch32Generator.

Tests the batch-32 padded generation pipeline with Metal Trace support.
"""

import time
import torch
import ttnn

from bitnet_tt.config import BitNetConfig
from bitnet_tt.utils.device import device_context
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.model.bitnet import create_model
from bitnet_tt.inference.generator_batch32 import (
    Batch32Generator,
    Batch32RotarySetup,
    Batch32KVCache,
    PADDED_BATCH,
)


def test_batch32_rotary_setup():
    print("\n=== Test: Batch32RotarySetup ===\n")

    with device_context() as device:
        rotary = Batch32RotarySetup(
            device=device,
            head_dim=128,
            max_seq_len=4096,
            rope_theta=500000.0,
        )

        cos_s, sin_s = rotary.create_cos_sin_device_tensors(position=0)
        print(f"cos_s shape: {cos_s.shape}, memory: {cos_s.memory_config()}")
        print(f"sin_s shape: {sin_s.shape}, memory: {sin_s.memory_config()}")

        trans_mat = rotary.get_sharded_trans_mat()
        print(f"trans_mat shape: {trans_mat.shape}, memory: {trans_mat.memory_config()}")

        ttnn.deallocate(cos_s)
        ttnn.deallocate(sin_s)
        ttnn.deallocate(trans_mat)

        print("SUCCESS: Batch32RotarySetup works!")
        return True


def test_batch32_kv_cache():
    print("\n=== Test: Batch32KVCache ===\n")

    with device_context() as device:
        cache = Batch32KVCache(
            max_seq_len=1024,
            num_kv_heads=5,
            head_dim=128,
            device=device,
        )

        print(f"key_cache shape: {cache.key_cache.shape}")
        print(f"value_cache shape: {cache.value_cache.shape}")

        k, v = cache.get_for_attention(seq_len=10)
        print(f"k slice shape: {k.shape}")
        print(f"v slice shape: {v.shape}")

        print("SUCCESS: Batch32KVCache works!")
        return True


def test_batch32_decode_step():
    print("\n=== Test: Batch32Generator Decode Step ===\n")

    with device_context() as device:
        state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
        model = create_model(config, device)
        load_weights_to_model(model, state_dict)

        print(f"Model loaded: {config.num_layers} layers")

        generator = Batch32Generator(model, enable_trace=False)
        generator._kv_caches = generator._allocate_kv_caches(max_seq_len=256)

        embed_padded = torch.zeros((1, PADDED_BATCH, config.hidden_size), dtype=torch.bfloat16)
        embed_padded[0, 0, :] = generator._embedding_weight_host[1234]
        embeds = ttnn.from_torch(
            embed_padded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        pos_tensor = ttnn.from_torch(
            torch.zeros((PADDED_BATCH,), dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        cos_s, sin_s = generator.rotary_setup.create_cos_sin_device_tensors(position=0)

        logits = generator._decode_step_batch32(embeds, 0, pos_tensor, cos_s, sin_s)
        print(f"Decode logits shape: {logits.shape}")

        ttnn.deallocate(logits)
        ttnn.deallocate(embeds)
        ttnn.deallocate(pos_tensor)
        ttnn.deallocate(cos_s)
        ttnn.deallocate(sin_s)

        print("SUCCESS: Decode step through full batch-32 pipeline works!")
        return True


def test_batch32_generator():
    print("\n=== Test: Batch32Generator Full Generation ===\n")

    with device_context() as device:
        state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
        model = create_model(config, device)
        load_weights_to_model(model, state_dict)

        generator = Batch32Generator(model, enable_trace=False)

        prompt = "Hello, I am"
        print(f"Prompt: {prompt}")

        start = time.perf_counter()
        output = generator.generate(prompt, max_new_tokens=10, temperature=0.7)
        elapsed = time.perf_counter() - start

        print(f"Output: {output}")
        print(f"Time: {elapsed:.2f}s")

        return True


def test_batch32_streaming():
    print("\n=== Test: Batch32Generator Streaming ===\n")

    with device_context() as device:
        state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
        model = create_model(config, device)
        load_weights_to_model(model, state_dict)

        generator = Batch32Generator(model, enable_trace=False)

        prompt = "The capital of France is"
        print(f"Prompt: {prompt}")
        print("Output: ", end="", flush=True)

        for text, stats in generator.generate_streaming(prompt, max_new_tokens=20):
            print(text, end="", flush=True)

        print(f"\n{stats}")

        return True


def benchmark_batch32():
    print("\n=== Benchmark: Batch32Generator ===\n")

    with device_context() as device:
        state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
        model = create_model(config, device)
        load_weights_to_model(model, state_dict)

        generator = Batch32Generator(model, enable_trace=False)

        prompt = "Hello, world!"
        max_tokens = 30

        for text, stats in generator.generate_streaming(prompt, max_new_tokens=max_tokens):
            pass

        print(f"Prompt tokens: {stats.prompt_tokens}")
        print(f"Generated tokens: {stats.generated_tokens}")
        print(f"Prefill time: {stats.prompt_time * 1000:.1f}ms")
        print(f"Generation time: {stats.generation_time:.2f}s")
        print(f"Throughput: {stats.tokens_per_second:.2f} t/s")
        print(f"Avg token time: {stats.avg_token_time_ms:.2f}ms")

        return stats.tokens_per_second


def main():
    print("=" * 60)
    print("Batch-32 Generator Tests")
    print("=" * 60)

    tests = [
        ("Batch32RotarySetup", test_batch32_rotary_setup),
        ("Batch32KVCache", test_batch32_kv_cache),
        ("Decode Step", test_batch32_decode_step),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"FAILED: {name} - {e}")
            results[name] = False
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Results:")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
