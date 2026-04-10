#!/usr/bin/env python3
"""Measure BitNet 2B4T decode tokens/sec on p150a via Batch32Generator."""

import argparse
import time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "bfp8", "bfp4"])
    ap.add_argument("--prompt", default="The meaning of life is")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--warmup-prompt-tokens", type=int, default=6)
    ap.add_argument("--warmup-new", type=int, default=32)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--lofi-mlp", action="store_true")
    args = ap.parse_args()

    print(f"=== BitNet-TT Batch32 decode benchmark ===")
    print(f"weight_dtype={args.dtype}  max_new={args.max_new}  trace={not args.no_trace}  lofi_mlp={args.lofi_mlp}")

    from bitnet_tt.utils.device import get_device, close_device
    from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
    from bitnet_tt.model.bitnet import create_model
    from bitnet_tt.inference.generator_batch32 import Batch32Generator
    from transformers import AutoTokenizer

    device = get_device()

    t0 = time.perf_counter()
    state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
    print(f"[load] hf state_dict -> numpy  {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    model = create_model(config, device, weight_dtype=args.dtype, use_lofi_mlp=args.lofi_mlp)
    load_weights_to_model(model, state_dict)
    print(f"[load] numpy -> device  {time.perf_counter()-t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")
    gen = Batch32Generator(model, tokenizer=tokenizer, enable_trace=not args.no_trace)

    # warmup (trace capture)
    print(f"[warmup] prompt_tokens={args.warmup_prompt_tokens} new={args.warmup_new}...")
    t0 = time.perf_counter()
    wstats = gen.warmup(prompt_tokens=args.warmup_prompt_tokens, max_new_tokens=args.warmup_new)
    print(f"[warmup] took {time.perf_counter()-t0:.1f}s  prompt={wstats.prompt_time:.2f}s  decode={wstats.decode_time:.2f}s  trace_captured={wstats.trace_captured}")

    # Timed streaming run
    print(f"[bench] generating {args.max_new} tokens (greedy)...")
    token_times = []
    prompt_time = 0.0
    first_token_time = None
    prev_t = None
    out_text_parts = []
    t_start = time.perf_counter()
    for new_text, stats in gen.generate_streaming(
        args.prompt,
        max_new_tokens=args.max_new,
        temperature=0.0,
        top_k=1,
    ):
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now - t_start
            prompt_time = stats.prompt_time
        if prev_t is not None:
            token_times.append(now - prev_t)
        prev_t = now
        out_text_parts.append(new_text)
    total = time.perf_counter() - t_start
    out_text = "".join(out_text_parts)

    gen_tokens = stats.generated_tokens
    decode_time = total - prompt_time
    decode_tps = (gen_tokens - 1) / decode_time if decode_time > 0 else 0.0
    overall_tps = gen_tokens / total

    print()
    print(f"Output (first 200 chars): {out_text[:200]}")
    print()
    print(f"[RESULT] prompt_time={prompt_time*1000:.1f}ms  TTFT={first_token_time*1000:.1f}ms")
    print(f"[RESULT] total={total:.2f}s  gen_tokens={gen_tokens}")
    print(f"[RESULT] decode_time={decode_time:.2f}s  decode_tps={decode_tps:.2f} t/s")
    print(f"[RESULT] overall_tps={overall_tps:.2f} t/s  avg_latency={1000*total/gen_tokens:.1f} ms/tok")
    if token_times:
        arr = np.array(token_times) * 1000
        print(f"[RESULT] token_latency  p50={np.median(arr):.1f}ms  p90={np.percentile(arr,90):.1f}ms  min={arr.min():.1f}ms  max={arr.max():.1f}ms")

    close_device()


if __name__ == "__main__":
    main()
