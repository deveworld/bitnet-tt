#!/usr/bin/env python3
"""Test packed_ternary with simple (non-batch32) generator."""
import numpy as np
from bitnet_tt.utils.device import get_device, close_device
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.model.bitnet import create_model
from bitnet_tt.inference.generator import TextGenerator
from transformers import AutoTokenizer

device = get_device()
state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")

model = create_model(config, device, weight_dtype="packed_ternary", use_fused_rope=False)
load_weights_to_model(model, state_dict)
print("packed_ternary model loaded", flush=True)

tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")
gen = TextGenerator(model, tokenizer, batch_size=1, enable_trace=False)

prompt = "The capital of France is"
print(f"Prompt: {prompt}", flush=True)
out = gen.generate(prompt, max_new_tokens=16, do_sample=False)
print(f"Output: {out}", flush=True)

close_device()
print("DONE", flush=True)
