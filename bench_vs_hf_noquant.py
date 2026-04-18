#!/usr/bin/env python3
"""TT vs HF-bf16 with ActQuant DISABLED — measures TT's inherent bf16 faithfulness."""
import argparse, numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument('--dtype', default='packed_ternary')
ap.add_argument('--prompt', default='The capital of France is')
args = ap.parse_args()

def pcc(a,b):
    a=a.astype(np.float64).flatten(); b=b.astype(np.float64).flatten()
    a-=a.mean(); b-=b.mean()
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return 0.0 if d==0 else float(np.dot(a,b)/d)

# --- Patch HF ActQuant to identity ---
from transformers.integrations.bitnet import ActQuant
orig_apply = ActQuant.apply
ActQuant.apply = staticmethod(lambda x: x)
print('[patch] ActQuant.apply -> identity (no per-token INT8 quant)')

from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = 'microsoft/bitnet-b1.58-2B-4T-bf16'
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
m.eval()

enc = tok(args.prompt, return_tensors='pt')
ids = enc['input_ids']
with torch.no_grad():
    out = m(input_ids=ids)
hf_last = out.logits[0,-1,:].float().numpy()
print(f'HF-noquant argmax={int(hf_last.argmax())} ({tok.decode([int(hf_last.argmax())])!r})')

# --- TT prefill ---
from bitnet_tt.utils.device import get_device
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.model.bitnet import create_model
from bitnet_tt.inference.generator_batch32 import Batch32Generator
import ttnn
device = get_device()
state_dict, config = load_bitnet_weights(MODEL)
model = create_model(config, device, weight_dtype=args.dtype)
load_weights_to_model(model, state_dict)
gen = Batch32Generator(model, tokenizer=tok, enable_trace=True)
gen._ensure_kv_caches(max_seq_len=256)

ids_np = ids.numpy().astype(np.int64)
tt_logits_t, _ = gen._prefill_batch32(ids_np)
tt_np = ttnn.to_torch(tt_logits_t).float().numpy()
tt_last = (tt_np[0,-1,:] if tt_np.ndim==3 else tt_np[-1,:])[:128256]

print(f'\n=== TT vs HF-bf16 (ActQuant DISABLED) ===')
print(f'  PCC: {pcc(tt_last, hf_last):.6f}')
print(f'  argmax match: {tt_last.argmax() == hf_last.argmax()}')
print(f'  max|diff|: {np.max(np.abs(tt_last - hf_last)):.4f}')
