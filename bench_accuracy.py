#!/usr/bin/env python3
"""
Measure BitNet-TT accuracy against HuggingFace CPU reference.

Compares:
  - Last-token prefill logits (PCC, cosine, top-k overlap, greedy match)
  - First N greedy decode tokens (token-level match vs HF)

This is the "PCC >= 0.999" kind of check from the TT bringup guide.
"""

import argparse
import time
import numpy as np
import torch


def pearson_cc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ta = set(np.argsort(a.flatten())[-k:].tolist())
    tb = set(np.argsort(b.flatten())[-k:].tolist())
    return len(ta & tb) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "bfp8", "bfp4"])
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--decode-steps", type=int, default=16)
    args = ap.parse_args()

    MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"

    print(f"=== BitNet-TT accuracy vs HF ===")
    print(f"model={MODEL}  weight_dtype={args.dtype}  prompt={args.prompt!r}")

    # ---------- HF reference (CPU) ----------
    print("\n[HF] loading reference model on CPU...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    t0 = time.perf_counter()
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    hf_model.eval()
    print(f"[HF] loaded in {time.perf_counter()-t0:.1f}s")

    enc = tokenizer(args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"]  # [1, seq]
    seq_len = input_ids.shape[1]
    print(f"[HF] prompt token ids: {input_ids[0].tolist()}")
    print(f"[HF] prompt decoded:   {tokenizer.decode(input_ids[0])!r}")

    with torch.no_grad():
        out = hf_model(input_ids=input_ids)
        hf_logits_last = out.logits[0, -1, :].float().numpy()  # [vocab]

    # HF greedy decode
    hf_tokens = input_ids[0].tolist()
    hf_step_logits = []
    cur_ids = input_ids
    past = None
    with torch.no_grad():
        for step in range(args.decode_steps):
            out = hf_model(input_ids=cur_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            step_logits = out.logits[0, -1, :].float().numpy()
            hf_step_logits.append(step_logits)
            next_id = int(step_logits.argmax())
            hf_tokens.append(next_id)
            cur_ids = torch.tensor([[next_id]], dtype=torch.long)
    hf_gen_text = tokenizer.decode(hf_tokens, skip_special_tokens=True)
    print(f"[HF] greedy output ({args.decode_steps} new tok): {hf_gen_text!r}")

    del hf_model  # free RAM

    # ---------- TT model ----------
    print("\n[TT] opening device + loading model...")
    from bitnet_tt.utils.device import get_device, close_device
    from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
    from bitnet_tt.model.bitnet import create_model
    from bitnet_tt.inference.generator_batch32 import Batch32Generator
    import ttnn

    device = get_device()
    state_dict, config = load_bitnet_weights(MODEL)
    model = create_model(config, device, weight_dtype=args.dtype)
    load_weights_to_model(model, state_dict)

    gen = Batch32Generator(model, tokenizer=tokenizer, enable_trace=True)
    gen._ensure_kv_caches(max_seq_len=256)

    # Prefill with the exact same prompt
    input_ids_np = input_ids.numpy().astype(np.int64)
    t0 = time.perf_counter()
    tt_logits, tt_seq = gen._prefill_batch32(input_ids_np)
    print(f"[TT] prefill time: {(time.perf_counter()-t0)*1000:.1f}ms  logits.shape={tuple(tt_logits.shape)}")

    tt_logits_np = ttnn.to_torch(tt_logits).float().numpy()  # [1, seq, vocab] or similar
    # Shape may be [1, seq, vocab] — take last seq row
    # Sometimes the model slices to last token only
    if tt_logits_np.ndim == 3:
        tt_last = tt_logits_np[0, -1, :]
    elif tt_logits_np.ndim == 2:
        tt_last = tt_logits_np[-1, :]
    else:
        tt_last = tt_logits_np.flatten()[-config.vocab_size:]

    # Trim any padding in vocab dimension
    tt_last = tt_last[: config.vocab_size]
    hf_last = hf_logits_last[: config.vocab_size]

    print("\n=== Last-token prefill logits comparison ===")
    print(f"  HF argmax:      {int(hf_last.argmax())}  ({tokenizer.decode([int(hf_last.argmax())])!r})")
    print(f"  TT argmax:      {int(tt_last.argmax())}  ({tokenizer.decode([int(tt_last.argmax())])!r})")
    print(f"  PCC:            {pearson_cc(hf_last, tt_last):.6f}")
    print(f"  cosine:         {cosine(hf_last, tt_last):.6f}")
    print(f"  top1 match:     {int(hf_last.argmax()) == int(tt_last.argmax())}")
    print(f"  top5 overlap:   {topk_overlap(hf_last, tt_last, 5):.2f}")
    print(f"  top10 overlap:  {topk_overlap(hf_last, tt_last, 10):.2f}")
    print(f"  max |diff|:     {np.abs(hf_last - tt_last).max():.4f}")
    print(f"  MSE:            {float(np.mean((hf_last - tt_last) ** 2)):.6f}")

    # Release prefill logits, run TT greedy decode using HF's FORCED token path
    # We instead do independent TT greedy and compare token sequences
    ttnn.deallocate(tt_logits)

    print("\n=== TT greedy decode token comparison ===")
    gen.reset()
    gen._ensure_kv_caches(max_seq_len=256)
    tt_full = gen.generate(args.prompt, max_new_tokens=args.decode_steps, temperature=0.0, top_k=1)
    print(f"[TT] greedy output: {tt_full!r}")

    # Extract only generated tokens for both
    hf_gen_only = tokenizer.decode(hf_tokens[seq_len:], skip_special_tokens=True)
    tt_gen_only_ids = tokenizer(tt_full, return_tensors="np")["input_ids"][0][seq_len:]
    tt_gen_only = tokenizer.decode(tt_gen_only_ids.tolist(), skip_special_tokens=True)

    hf_toks = hf_tokens[seq_len : seq_len + args.decode_steps]
    tt_toks = tt_gen_only_ids[: args.decode_steps].tolist()
    matches = sum(1 for a, b in zip(hf_toks, tt_toks) if a == b)
    divergence_at = None
    for i, (a, b) in enumerate(zip(hf_toks, tt_toks)):
        if a != b:
            divergence_at = i
            break
    print(f"  HF greedy tokens: {hf_toks}")
    print(f"  TT greedy tokens: {tt_toks}")
    print(f"  match count:      {matches}/{len(hf_toks)}  ({100*matches/max(1,len(hf_toks)):.1f}%)")
    print(f"  first divergence: step {divergence_at}")

    close_device()


if __name__ == "__main__":
    main()
