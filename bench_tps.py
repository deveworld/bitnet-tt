#!/usr/bin/env python3
"""Measure BitNet 2B4T decode tokens/sec on p150a."""

import argparse
import time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "bfp8", "bfp4"])
    ap.add_argument("--prompt", default="The meaning of life is")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    print(f"=== BitNet-TT decode benchmark ===")
    print(f"weight_dtype={args.dtype}  max_new={args.max_new}  trace={args.trace}")

    from bitnet_tt.utils.device import get_device, close_device
    from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
    from bitnet_tt.model.bitnet import create_model
    from bitnet_tt.inference.generator import TextGenerator
    from transformers import AutoTokenizer

    device = get_device()

    t0 = time.perf_counter()
    state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
    print(f"[load] hf state_dict -> numpy  {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    model = create_model(config, device, weight_dtype=args.dtype)
    load_weights_to_model(model, state_dict)
    print(f"[load] numpy -> device  {time.perf_counter()-t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")
    generator = TextGenerator(model, tokenizer, batch_size=1, enable_trace=args.trace)

    # warmup
    print(f"[warmup] {args.warmup} tokens...")
    _ = generator.generate(args.prompt, max_new_tokens=args.warmup, do_sample=False)

    # Timed run
    print(f"[bench] generating {args.max_new} tokens (greedy)...")
    t_start = time.perf_counter()
    output = generator.generate(args.prompt, max_new_tokens=args.max_new, do_sample=False)
    total = time.perf_counter() - t_start

    n_new = args.max_new
    tps = n_new / total
    print()
    print(f"Output: {output[:200]}")
    print()
    print(f"[RESULT] total={total:.2f}s  tokens={n_new}  tps={tps:.2f} t/s  latency={1000*total/n_new:.1f} ms/tok")

    close_device()


if __name__ == "__main__":
    main()
