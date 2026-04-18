#!/usr/bin/env python3
"""Plan v4 Phase 1 — per-op RFE localization harness.

Captures intermediate tensors from HF bf16 and TT at identical layer
boundaries for a fixed set of capture layers, then ranks op
contributions to the PCC gap via relative Frobenius error.

Usage:
    python3 pcc_localize.py --capture        # run both TT + HF capture
    python3 pcc_localize.py --rank           # compute + print RFE table

See docs/session_8_localization_design.md for capture-point rationale.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "The answer to life is",
    "def fib(n):",
    "In 2024, we",
]
CAPTURE_LAYERS = [0, 5, 15, 25, 29]
CAPTURE_POINTS = [
    "block_input",
    "post_input_norm",
    "post_self_attn",
    "post_attn_residual",
    "post_post_attn_norm",
    "post_mlp",
    "block_output",
]
OUT_DIR = Path("/tmp/bitnet_localize")
OUT_DIR.mkdir(exist_ok=True, parents=True)

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"


def rfe(a: np.ndarray, b: np.ndarray) -> float:
    """Relative Frobenius error. Composes via triangle inequality."""
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    denom = np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(a - b) / denom)


def capture_hf(prompt: str, tag: str) -> dict[str, np.ndarray]:
    """Run HF bf16 model with forward hooks at capture boundaries."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.eval()

    captured: dict[str, np.ndarray] = {}
    handles = []

    def _mk_hook(name: str):
        def _hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach().float().cpu().numpy()
        return _hook

    def _mk_pre_hook(name: str):
        def _hook(_module, args):
            t = args[0] if len(args) > 0 else None
            if t is not None:
                captured[name] = t.detach().float().cpu().numpy()
        return _hook

    for li in CAPTURE_LAYERS:
        layer = model.model.layers[li]
        handles.append(layer.register_forward_pre_hook(_mk_pre_hook(f"L{li}.block_input")))
        handles.append(layer.input_layernorm.register_forward_hook(_mk_hook(f"L{li}.post_input_norm")))
        handles.append(layer.self_attn.register_forward_hook(_mk_hook(f"L{li}.post_self_attn")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(_mk_hook(f"L{li}.post_post_attn_norm")))
        handles.append(layer.mlp.register_forward_hook(_mk_hook(f"L{li}.post_mlp")))
        handles.append(layer.register_forward_hook(_mk_hook(f"L{li}.block_output")))

    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        model(input_ids=enc["input_ids"])

    for h in handles:
        h.remove()

    np.savez(OUT_DIR / f"hf_{tag}.npz", **captured)
    print(f"[HF] captured {len(captured)} tensors for tag={tag}")
    return captured


def capture_tt(prompt: str, tag: str) -> dict[str, np.ndarray]:
    """Run TT model with in-tree BITNET_LOCALIZE hook (transformer.py)."""
    # Force env flag BEFORE importing bitnet_tt so the module-level
    # _BITNET_LOCALIZE guard picks it up.
    os.environ["BITNET_LOCALIZE"] = "1"
    os.environ["BITNET_LOCALIZE_LAYERS"] = ",".join(str(i) for i in CAPTURE_LAYERS)

    import bitnet_tt  # noqa: F401
    import ttnn
    from bitnet_tt.utils.device import get_device
    from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
    from bitnet_tt.model.bitnet import create_model
    from bitnet_tt.inference.generator_batch32 import Batch32Generator
    from bitnet_tt.model import transformer as tr
    from transformers import AutoTokenizer

    tr._localize_captures.clear()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    device = get_device()
    state_dict, config = load_bitnet_weights(MODEL_ID)
    model = create_model(config, device, weight_dtype="packed_ternary")
    load_weights_to_model(model, state_dict)

    gen = Batch32Generator(model, tokenizer=tok, enable_trace=False)
    gen._ensure_kv_caches(max_seq_len=256)

    enc = tok(prompt, return_tensors="pt")
    ids_np = enc["input_ids"].numpy().astype(np.int64)
    _ = gen._prefill_batch32(ids_np)

    captured = {k: v for k, v in tr._localize_captures.items() if not k.endswith("__err")}
    errs = {k: v for k, v in tr._localize_captures.items() if k.endswith("__err")}
    np.savez(OUT_DIR / f"tt_{tag}.npz", **captured)
    print(f"[TT] captured {len(captured)} tensors for tag={tag} (errs={len(errs)})")
    return captured


def rank() -> None:
    """Load captured pairs + compute RFE table."""
    rows = []
    for k, prompt in enumerate(PROMPTS):
        tag = f"p{k}"
        hf_path = OUT_DIR / f"hf_{tag}.npz"
        tt_path = OUT_DIR / f"tt_{tag}.npz"
        if not hf_path.exists() or not tt_path.exists():
            print(f"[rank] missing capture for tag={tag}, skipping", file=sys.stderr)
            continue
        hf = np.load(hf_path, allow_pickle=True)
        tt = np.load(tt_path, allow_pickle=True)
        for layer_idx in CAPTURE_LAYERS:
            for cp in CAPTURE_POINTS:
                name = f"L{layer_idx}.{cp}"
                if name not in hf.files or name not in tt.files:
                    continue
                hf_t = hf[name]
                tt_t = tt[name]
                # shape fix: TT may have batch=32 pad, HF has batch=1 — take slice.
                if tt_t.shape != hf_t.shape:
                    # Try trimming leading-batch mismatch.
                    try:
                        tt_slice = tt_t
                        while tt_slice.ndim > hf_t.ndim:
                            tt_slice = tt_slice.squeeze(0)
                        if tt_slice.shape[0] != hf_t.shape[0]:
                            tt_slice = tt_slice[: hf_t.shape[0]]
                        # trim seq
                        if tt_slice.ndim >= 2 and tt_slice.shape[-2] != hf_t.shape[-2]:
                            tt_slice = tt_slice[..., : hf_t.shape[-2], :]
                        tt_t = tt_slice
                    except Exception:
                        pass
                if tt_t.shape != hf_t.shape:
                    print(f"[rank] shape mismatch {name}: tt={tt_t.shape} hf={hf_t.shape}", file=sys.stderr)
                    continue
                e = rfe(tt_t, hf_t)
                rows.append({"prompt": k, "layer": layer_idx, "capture": cp, "rfe": e})

    # Aggregate per (layer, capture) across prompts.
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["layer"], r["capture"])].append(r["rfe"])

    print()
    print("# RFE Localization Table (TT vs HF bf16)")
    print(f"{'layer':>5}  {'capture':>20}  {'mean':>10}  {'std':>10}  {'n':>3}")
    print("-" * 65)
    ranked = []
    for (layer, cp), vs in sorted(agg.items()):
        arr = np.array(vs)
        ranked.append((float(arr.mean()), layer, cp, arr))
    ranked.sort(reverse=True)
    for mean, layer, cp, arr in ranked:
        print(f"{layer:>5}  {cp:>20}  {mean:>10.6f}  {arr.std():>10.6f}  {len(arr):>3}")

    out_path = Path(__file__).parent.parent / "docs" / "session_8_rfe_ranking.md"
    with out_path.open("w") as f:
        f.write("# Session 8 — per-op RFE ranking\n\n")
        f.write(f"Prompts: {len(PROMPTS)}. Capture layers: {CAPTURE_LAYERS}.\n\n")
        f.write("| rank | layer | capture | mean RFE | std | n |\n")
        f.write("|-----:|------:|:--------|---------:|----:|--:|\n")
        for rank_i, (mean, layer, cp, arr) in enumerate(ranked, 1):
            f.write(f"| {rank_i} | {layer} | {cp} | {mean:.6f} | {arr.std():.6f} | {len(arr)} |\n")
    print(f"\n[rank] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true", help="run HF + TT captures for all prompts")
    ap.add_argument("--capture-hf-only", action="store_true")
    ap.add_argument("--capture-tt-only", action="store_true")
    ap.add_argument("--rank", action="store_true", help="compute RFE table from existing captures")
    args = ap.parse_args()

    if args.capture or args.capture_hf_only:
        for k, p in enumerate(PROMPTS):
            capture_hf(p, f"p{k}")
    if args.capture or args.capture_tt_only:
        for k, p in enumerate(PROMPTS):
            capture_tt(p, f"p{k}")
    if args.rank:
        rank()

    if not any([args.capture, args.capture_hf_only, args.capture_tt_only, args.rank]):
        ap.print_help()


if __name__ == "__main__":
    main()
