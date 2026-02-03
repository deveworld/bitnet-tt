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
    create_height_sharded_config,
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

        cos_s, sin_s = rotary.get_sharded_cos_sin(position=0)
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

        layer0 = model.layers[0]

        rotary = Batch32RotarySetup(
            device=device,
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )

        token_padded = torch.zeros((PADDED_BATCH, 1), dtype=torch.int64)
        token_padded[0, 0] = 1234

        token_tt = ttnn.from_torch(
            token_padded,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        embeds = model.embed_tokens(token_tt)
        print(f"Embeddings shape: {embeds.shape}")

        hidden = layer0.input_layernorm(embeds)

        hidden_rm = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
        hidden_1bkd = ttnn.reshape(hidden_rm, (1, 1, PADDED_BATCH, config.hidden_size))
        hidden_1bkd = ttnn.to_layout(hidden_1bkd, ttnn.TILE_LAYOUT)
        print(f"Hidden 1BKD shape: {hidden_1bkd.shape}")

        attention = layer0.self_attn
        qkv_fused = ttnn.matmul(hidden_1bkd, attention.qkv_fused_weight)
        print(f"Fused QKV shape: {qkv_fused.shape}")

        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv_fused,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        print(f"Q heads shape: {q_heads.shape}")
        print(f"K heads shape: {k_heads.shape}")
        print(f"V heads shape: {v_heads.shape}")

        cos_s, sin_s = rotary.get_sharded_cos_sin(position=0)
        trans_mat = rotary.get_sharded_trans_mat()

        q_rot = ttnn.experimental.rotary_embedding_llama(
            q_heads, cos_s, sin_s, trans_mat, is_decode_mode=True
        )
        k_rot = ttnn.experimental.rotary_embedding_llama(
            k_heads, cos_s, sin_s, trans_mat, is_decode_mode=True
        )
        print(f"Q rotated shape: {q_rot.shape}")
        print(f"K rotated shape: {k_rot.shape}")

        print("SUCCESS: Decode step through layer 0 works!")
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
