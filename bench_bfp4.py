#!/usr/bin/env python3
"""
Benchmark: BFP4 vs BF16 weight dtype for BitNet ternary matmul.

Measures decode throughput (tokens/second) with different weight storage formats.
Runs on Tenstorrent p150a hardware.

Usage:
    python bench_bfp4.py
"""

import time
import sys
import numpy as np

def main():
    print("=" * 60)
    print("BitNet-TT: BFP4 vs BF16 Weight Dtype Benchmark")
    print("=" * 60)

    import ttnn
    from bitnet_tt.utils.device import get_device, close_device
    from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
    from bitnet_tt.config import BitNet2B4TConfig

    device = get_device()
    print(f"Device opened.\n")

    # Load weights once on CPU
    print("Loading weights from HuggingFace...")
    t0 = time.perf_counter()
    state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
    print(f"Weights loaded in {time.perf_counter() - t0:.1f}s")
    print(f"Model: {config.hidden_size}d, {config.num_layers}L, {config.vocab_size} vocab\n")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")

    prompt = "The meaning of life is"
    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
    seq_len = input_ids.shape[1]
    max_new_tokens = 32

    results = {}

    for dtype_name in ["bf16", "bfp4"]:
        print(f"\n{'─' * 50}")
        print(f"Testing weight_dtype = {dtype_name!r}")
        print(f"{'─' * 50}")

        # Create model with this dtype
        from bitnet_tt.model.bitnet import create_model
        t0 = time.perf_counter()
        model = create_model(config, device, weight_dtype=dtype_name)
        load_weights_to_model(model, state_dict)
        load_time = time.perf_counter() - t0
        print(f"Model loaded to device in {load_time:.1f}s")

        # Create generator
        from bitnet_tt.inference.generator import TextGenerator
        generator = TextGenerator(model, tokenizer, batch_size=1, enable_trace=False)

        # Warmup
        print("Warmup...")
        _ = generator.generate(prompt, max_new_tokens=4, do_sample=False)

        # Benchmark: greedy decode
        print(f"Generating {max_new_tokens} tokens (greedy)...")
        t_start = time.perf_counter()
        output = generator.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        t_total = time.perf_counter() - t_start

        # Collect token times from streaming
        token_times = []
        for token_text, stats in generator.generate_streaming(
            prompt, max_new_tokens=max_new_tokens, do_sample=False
        ):
            pass
        if stats:
            token_times = stats.token_times
            tps = stats.tokens_per_second
            ttft = stats.time_to_first_token
            gen_time = stats.generation_time
        else:
            tps = max_new_tokens / t_total
            ttft = 0
            gen_time = t_total

        print(f"\nOutput: {output[:120]}...")
        print(f"Tokens/s:        {tps:.2f}")
        print(f"TTFT:            {ttft * 1000:.1f}ms")
        print(f"Generation time: {gen_time:.2f}s")
        if token_times:
            avg_ms = np.mean(token_times) * 1000
            p50_ms = np.median(token_times) * 1000
            p99_ms = np.percentile(token_times, 99) * 1000
            print(f"Token latency:   avg={avg_ms:.1f}ms  p50={p50_ms:.1f}ms  p99={p99_ms:.1f}ms")

        results[dtype_name] = {
            "tps": tps,
            "ttft_ms": ttft * 1000,
            "gen_time": gen_time,
            "load_time": load_time,
            "avg_token_ms": np.mean(token_times) * 1000 if token_times else 0,
        }

        # Cleanup model to free device memory for next run
        del generator
        del model
        try:
            ttnn.dump_device_memory_state(device)
        except Exception:
            pass

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Metric':<25} {'bf16':>12} {'bfp4':>12} {'speedup':>10}")
    print(f"{'─' * 60}")

    bf16 = results.get("bf16", {})
    bfp4 = results.get("bfp4", {})

    for metric, unit in [("tps", "t/s"), ("ttft_ms", "ms"), ("avg_token_ms", "ms"), ("load_time", "s")]:
        v_bf16 = bf16.get(metric, 0)
        v_bfp4 = bfp4.get(metric, 0)
        if v_bf16 > 0 and v_bfp4 > 0:
            if metric == "tps":
                speedup = v_bfp4 / v_bf16
            else:
                speedup = v_bf16 / v_bfp4  # lower is better for latency
            print(f"{metric:<25} {v_bf16:>10.1f}{unit:>2} {v_bfp4:>10.1f}{unit:>2} {speedup:>9.2f}x")

    close_device()
    print("\nDone.")


if __name__ == "__main__":
    main()
