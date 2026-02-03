#!/usr/bin/env python3
import time
import sys
import numpy as np

print("=" * 60, flush=True)
print("BitNet-TT Performance Test", flush=True)
print("=" * 60, flush=True)

from bitnet_tt.model.bitnet import create_model
from bitnet_tt.inference.generator import TextGenerator
from bitnet_tt.utils.device import device_context
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model

print("\n[1] Loading weights...", flush=True)
state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
print(f"    Config: {config.hidden_size}d, {config.num_layers}L", flush=True)

with device_context() as device:
    print("\n[2] Creating model...", flush=True)
    model = create_model(config, device)
    load_weights_to_model(model, state_dict)
    print("    Model loaded!", flush=True)

    print("\n[3] Creating generator (trace=False)...", flush=True)
    generator = TextGenerator(model, enable_trace=False)
    print("    Generator ready!", flush=True)

    print("\n[4] Running generation test...", flush=True)
    prompt = "Hello, I am"
    print(f"    Prompt: '{prompt}'", flush=True)

    start = time.perf_counter()
    output = ""
    final_stats = None
    token_count = 0

    for new_text, stats in generator.generate_streaming(
        prompt,
        max_new_tokens=20,
        temperature=0.7,
        use_optimized=False,
    ):
        output += new_text
        final_stats = stats
        token_count += 1
        sys.stdout.write(new_text)
        sys.stdout.flush()

    end = time.perf_counter()

    print(f"\n\n[5] Results:", flush=True)
    print(f"    Full output: {prompt}{output}", flush=True)
    print(f"    Tokens generated: {token_count}", flush=True)
    print(f"    Total time: {end - start:.2f}s", flush=True)
    if final_stats:
        print(f"    {final_stats}", flush=True)

print("\n" + "=" * 60, flush=True)
print("Test Complete!", flush=True)
print("=" * 60, flush=True)
