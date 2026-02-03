#!/usr/bin/env python3
"""Performance test script for BitNet-TT."""

import time
import sys
import numpy as np

print("Loading BitNet-TT (prefill + decode test)...", flush=True)
from bitnet_tt.model.bitnet import create_model
from bitnet_tt.inference.generator import TextGenerator
from bitnet_tt.utils.device import device_context
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.layers.bitlinear import ttnn_to_numpy

state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
print(f"Config: {config.hidden_size}d, {config.num_layers}L", flush=True)

with device_context() as device:
    print("Creating model...", flush=True)
    model = create_model(config, device)
    load_weights_to_model(model, state_dict)
    print("Model ready!", flush=True)

    generator = TextGenerator(model, enable_trace=False)

    # Test PREFILL ONLY first
    prompt = "Hello world"
    inputs = generator.tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    print(f"Input tokens: {input_ids.shape}", flush=True)

    start = time.time()
    logits, kv_cache = generator.prefill_forward(input_ids, use_preallocated=False)
    prefill_time = time.time() - start
    print(f"Prefill done in {prefill_time:.2f}s", flush=True)

    # Get next token
    logits_np = ttnn_to_numpy(logits)
    next_token_id = int(np.argmax(logits_np[0, -1, :]))
    next_token = generator.tokenizer.decode([next_token_id])
    print(f"Next token: {next_token_id} -> '{next_token}'", flush=True)

    # Now try ONE decode step
    print("Testing single decode step...", flush=True)
    current_pos = input_ids.shape[1]
    token_input = np.array([[next_token_id]], dtype=np.int64)

    start = time.time()
    logits2, kv_cache = generator.decode_forward(
        token_input, current_pos, kv_cache, use_optimized=False
    )
    decode_time = time.time() - start
    print(f"Decode step done in {decode_time:.2f}s", flush=True)

    logits2_np = ttnn_to_numpy(logits2)
    next_token_id2 = int(np.argmax(logits2_np[0, -1, :]))
    next_token2 = generator.tokenizer.decode([next_token_id2])
    print(f"Token 2: {next_token_id2} -> '{next_token2}'", flush=True)

    # Try a few more decode steps
    print("Testing 10 more decode steps...", flush=True)
    tokens_generated = [next_token_id, next_token_id2]
    decode_times = []
    for i in range(10):
        current_pos += 1
        token_input = np.array([[tokens_generated[-1]]], dtype=np.int64)
        start = time.time()
        logits_i, kv_cache = generator.decode_forward(
            token_input, current_pos, kv_cache, use_optimized=False
        )
        decode_time = time.time() - start
        decode_times.append(decode_time)
        logits_i_np = ttnn_to_numpy(logits_i)
        next_id = int(np.argmax(logits_i_np[0, -1, :]))
        tokens_generated.append(next_id)
        token_text = generator.tokenizer.decode([next_id])
        print(f"  Step {i + 1}: {decode_time:.3f}s -> '{token_text}'", flush=True)

    full_output = generator.tokenizer.decode(tokens_generated)
    avg_decode_time = sum(decode_times) / len(decode_times)
    tokens_per_sec = 1.0 / avg_decode_time if avg_decode_time > 0 else 0

    print(f"\n{'=' * 60}", flush=True)
    print(f"Generated text: {full_output}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Prefill time: {prefill_time:.2f}s", flush=True)
    print(f"Average decode time: {avg_decode_time:.3f}s/token", flush=True)
    print(f"Decode speed: {tokens_per_sec:.2f} tokens/sec", flush=True)
    print(f"{'=' * 60}", flush=True)

print("Done!", flush=True)
