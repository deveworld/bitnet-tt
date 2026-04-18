#!/usr/bin/env python3
"""Compare TT prefill logits against bitnet.cpp reference logits (binary file)."""
import argparse, time, numpy as np, torch, sys

def pcc(a, b):
    a = a.astype(np.float64).flatten(); b = b.astype(np.float64).flatten()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if d == 0 else float(np.dot(a, b) / d)

def cos(a, b):
    a = a.astype(np.float64).flatten(); b = b.astype(np.float64).flatten()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if d == 0 else float(np.dot(a, b) / d)

ap = argparse.ArgumentParser()
ap.add_argument('--dtype', default='packed_ternary', choices=['bf16','bfp8','bfp4','packed_ternary'])
ap.add_argument('--prompt', default='The capital of France is')
ap.add_argument('--ref-logits', required=True, help='bitnet.cpp logits binary file')
args = ap.parse_args()

from transformers import AutoTokenizer
from bitnet_tt.utils.device import get_device
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.model.bitnet import create_model
from bitnet_tt.inference.generator_batch32 import Batch32Generator
import ttnn

MODEL = 'microsoft/bitnet-b1.58-2B-4T-bf16'
tok = AutoTokenizer.from_pretrained(MODEL)
enc = tok(args.prompt, return_tensors='pt')
ids = enc['input_ids'][0].tolist()
print(f'prompt_tokens={ids}')

device = get_device()
state_dict, config = load_bitnet_weights(MODEL)
model = create_model(config, device, weight_dtype=args.dtype)
load_weights_to_model(model, state_dict)
gen = Batch32Generator(model, tokenizer=tok, enable_trace=True)
gen._ensure_kv_caches(max_seq_len=256)


tt_logits_t, _ = gen._prefill_batch32(np.array(ids, dtype=np.int64).reshape(1,-1))
tt_logits_np = ttnn.to_torch(tt_logits_t).float().numpy(); tt_logits = (tt_logits_np[0,-1,:] if tt_logits_np.ndim==3 else tt_logits_np[-1,:])[:128256]
print(f'tt_logits.shape={tt_logits.shape} argmax={int(tt_logits.argmax())}')

ref = np.fromfile(args.ref_logits, dtype=np.float32)
print(f'ref_logits.shape={ref.shape} argmax={int(ref.argmax())}')

print(f'\n=== TT vs bitnet.cpp (official CPU inference) ===')
print(f'  PCC:          {pcc(tt_logits, ref):.6f}')
print(f'  cosine:       {cos(tt_logits, ref):.6f}')
print(f'  max|diff|:    {np.max(np.abs(tt_logits - ref)):.4f}')
print(f'  MSE:          {np.mean((tt_logits - ref)**2):.6f}')
print(f'  argmax match: {tt_logits.argmax() == ref.argmax()}')
for k in [1, 5, 10, 20]:
    tt_topk = set(np.argsort(-tt_logits)[:k].tolist())
    rf_topk = set(np.argsort(-ref)[:k].tolist())
    print(f'  top{k} overlap: {len(tt_topk & rf_topk)/k:.2f}')
