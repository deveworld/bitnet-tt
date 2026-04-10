#!/usr/bin/env python3
"""Analyze zero-tile distribution across all BitNet 2B4T weight matrices."""
import numpy as np
from bitnet_tt.utils.weights import load_bitnet_weights, get_layer_weights
from bitnet_tt.utils.ternary_analysis import summarize_weight

MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"
PROJECTIONS = [
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
]

def main():
    print(f"Loading {MODEL}...")
    state_dict, config = load_bitnet_weights(MODEL)
    print(f"Model: {config.num_layers} layers, {config.hidden_size}d\n")

    total_elements = 0
    total_zeros = 0
    total_tiles32 = 0
    total_zero_tiles32 = 0
    per_proj_stats = {}

    for layer_idx in range(config.num_layers):
        lw = get_layer_weights(state_dict, layer_idx)
        for proj in PROJECTIONS:
            if proj not in lw:
                continue
            w = lw[proj]
            s = summarize_weight(w)
            key = proj.split(".")[-2]  # q_proj, gate_proj, etc.
            if key not in per_proj_stats:
                per_proj_stats[key] = {"elements": 0, "zeros": 0, "tiles32": 0, "zero_tiles32": 0}

            n = s["elements"]
            nz = int(s["zero_frac"] * n)
            rows, cols = w.shape
            pr, pc = ((rows+31)//32)*32, ((cols+31)//32)*32
            nt = (pr//32) * (pc//32)
            nzt = int(s["tile32_all_zero_frac"] * nt)

            per_proj_stats[key]["elements"] += n
            per_proj_stats[key]["zeros"] += nz
            per_proj_stats[key]["tiles32"] += nt
            per_proj_stats[key]["zero_tiles32"] += nzt
            total_elements += n
            total_zeros += nz
            total_tiles32 += nt
            total_zero_tiles32 += nzt

    print(f"{'Projection':<12} {'Elements':>12} {'Zero%':>8} {'Tiles32':>10} {'ZeroTile%':>10} {'Skip Savings':>12}")
    print("-" * 70)
    for key, st in sorted(per_proj_stats.items()):
        zf = 100 * st["zeros"] / max(1, st["elements"])
        ztf = 100 * st["zero_tiles32"] / max(1, st["tiles32"])
        savings = st["zero_tiles32"] * 32 * 32 * 2 / 1e6  # MB saved if tiles skipped (bf16)
        print(f"{key:<12} {st['elements']:>12,} {zf:>7.1f}% {st['tiles32']:>10,} {ztf:>9.1f}% {savings:>10.1f} MB")

    print("-" * 70)
    zf = 100 * total_zeros / max(1, total_elements)
    ztf = 100 * total_zero_tiles32 / max(1, total_tiles32)
    print(f"{'TOTAL':<12} {total_elements:>12,} {zf:>7.1f}% {total_tiles32:>10,} {ztf:>9.1f}%")
    print(f"\nElement-wise zero rate: {zf:.1f}%")
    print(f"32x32 tile-wise all-zero rate: {ztf:.1f}%")
    print(f"Total zero tiles: {total_zero_tiles32:,} / {total_tiles32:,}")
    if ztf > 1:
        print(f"→ Track B (zero-tile skip) could skip {ztf:.1f}% of compute+bandwidth")
    else:
        print(f"→ Track B (zero-tile skip) has minimal benefit (<1% tiles are all-zero)")

if __name__ == "__main__":
    main()
