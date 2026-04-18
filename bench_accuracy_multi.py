#!/usr/bin/env python3
"""
Multi-prompt accuracy validation against HuggingFace CPU reference.

Exercises the lm_head_dtype change (or any other path) across several prompts
to check robustness beyond the single "The capital of France is" probe in
bench_accuracy.py. Reports per-prompt PCC, top-k overlaps, and greedy match
count so a single outlier prompt cannot mask a regression.
"""

import argparse
import time
import numpy as np
import torch


PROMPTS = [
    "The capital of France is",
    "The meaning of life is",
    "In a hole in the ground there lived",
    "The quick brown fox",
    "Artificial intelligence will",
]


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).flatten() - a.astype(np.float64).mean()
    b = b.astype(np.float64).flatten() - b.astype(np.float64).mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ta = set(np.argsort(a.flatten())[-k:].tolist())
    tb = set(np.argsort(b.flatten())[-k:].tolist())
    return len(ta & tb) / k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="packed_ternary",
                    choices=["bf16", "bfp8", "bfp4", "packed_ternary"])
    ap.add_argument("--decode-steps", type=int, default=16)
    args = ap.parse_args()

    MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"
    print(f"=== multi-prompt accuracy: {MODEL} dtype={args.dtype} steps={args.decode_steps} ===")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    t0 = time.perf_counter()
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    hf_model.eval()
    print(f"[HF] loaded in {time.perf_counter() - t0:.1f}s")

    # Precompute HF reference results per prompt
    hf_results = []
    for prompt in PROMPTS:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        seq_len = input_ids.shape[1]
        with torch.no_grad():
            out = hf_model(input_ids=input_ids)
            logits_last = out.logits[0, -1, :].float().numpy()
        tokens = input_ids[0].tolist()
        past = None
        cur_ids = input_ids
        with torch.no_grad():
            for _ in range(args.decode_steps):
                out = hf_model(input_ids=cur_ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                step_logits = out.logits[0, -1, :].float().numpy()
                nxt = int(step_logits.argmax())
                tokens.append(nxt)
                cur_ids = torch.tensor([[nxt]], dtype=torch.long)
        hf_results.append({
            "input_ids": input_ids,
            "seq_len": seq_len,
            "logits_last": logits_last,
            "tokens": tokens,
        })
    del hf_model

    # TT model once, reused across prompts
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

    rows = []
    for prompt, ref in zip(PROMPTS, hf_results):
        gen.reset()
        gen._ensure_kv_caches(max_seq_len=256)
        input_ids_np = ref["input_ids"].numpy().astype(np.int64)
        tt_logits, _ = gen._prefill_batch32(input_ids_np)
        tt_np = ttnn.to_torch(tt_logits).float().numpy()
        if tt_np.ndim == 3:
            tt_last = tt_np[0, -1, :]
        elif tt_np.ndim == 2:
            tt_last = tt_np[-1, :]
        else:
            tt_last = tt_np.flatten()[-config.vocab_size:]
        tt_last = tt_last[: config.vocab_size]
        hf_last = ref["logits_last"][: config.vocab_size]
        ttnn.deallocate(tt_logits)

        gen.reset()
        gen._ensure_kv_caches(max_seq_len=256)
        tt_full = gen.generate(prompt, max_new_tokens=args.decode_steps,
                               temperature=0.0, top_k=1)
        tt_gen_ids = tokenizer(tt_full, return_tensors="np")["input_ids"][0][ref["seq_len"]:]
        hf_gen_ids = ref["tokens"][ref["seq_len"]: ref["seq_len"] + args.decode_steps]
        matches = sum(1 for a, b in zip(hf_gen_ids, tt_gen_ids[: args.decode_steps]) if a == b)

        rows.append({
            "prompt": prompt,
            "argmax_match": int(hf_last.argmax()) == int(tt_last.argmax()),
            "pcc": pcc(hf_last, tt_last),
            "top5": topk_overlap(hf_last, tt_last, 5),
            "top10": topk_overlap(hf_last, tt_last, 10),
            "greedy_match": f"{matches}/{args.decode_steps}",
        })

    print()
    print(f"{'prompt':40s} | argmax | pcc    | top5 | top10 | match")
    print("-" * 86)
    for r in rows:
        print(f"{r['prompt'][:40]:40s} | {str(r['argmax_match']):6s} | "
              f"{r['pcc']:.4f} | {r['top5']:.2f} | {r['top10']:.2f}  | {r['greedy_match']}")
    pccs = [r["pcc"] for r in rows]
    print()
    print(f"Aggregate: mean PCC={np.mean(pccs):.4f}, min={np.min(pccs):.4f}, max={np.max(pccs):.4f}")
    print(f"argmax match: {sum(r['argmax_match'] for r in rows)}/{len(rows)}")

    close_device()


if __name__ == "__main__":
    main()
