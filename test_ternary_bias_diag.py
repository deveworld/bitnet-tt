#!/usr/bin/env python3
"""
Diagnostic test: pinpoint the source of 8% mean bias in ternary_matmul.

Tests:
  1. Identity readback — verify unpacked weight values via I @ W = W
  2. K-scaling — measure bias vs number of K tiles (1, 2, 4, 8, 64, 216)
  3. All-ones weight — isolate activation rounding from weight errors
  4. Element-wise dump — compare exact values for single-tile case
"""

import numpy as np
import torch
import ttnn

from bitnet_tt.kernels.pack import pack_ternary_tilized
from bitnet_tt.utils.quantization import weight_quant_ternary


def make_packed_ttnn(weight_quant_int8, scale, device):
    """Pack ternary weight and create device tensor."""
    packed_bytes, _ = pack_ternary_tilized(weight_quant_int8, scale)
    packed_u32 = np.frombuffer(packed_bytes.flatten().tobytes(), dtype=np.uint32)
    num_tiles = packed_bytes.shape[0]
    packed_torch = torch.from_numpy(
        packed_u32.reshape(num_tiles, 64).astype(np.int32)
    ).to(torch.int32)
    return ttnn.from_torch(
        packed_torch,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


def make_act_ttnn(act_np, device):
    """Create bf16 TILE activation tensor on device."""
    return ttnn.from_torch(
        torch.from_numpy(act_np.astype(np.float32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def run_ternary_matmul(act_ttnn, packed_ttnn, M, N):
    """Run ternary_matmul and return numpy result."""
    out = ttnn.experimental.ternary_matmul(
        act_ttnn, packed_ttnn, use_packed_ternary=True,
    )
    result = ttnn.to_torch(out).float().numpy().reshape(M, N)
    ttnn.deallocate(out)
    return result


def run_bf16_matmul(act_ttnn, weight_t_np, device, M, N):
    """Run standard bf16 matmul for comparison."""
    w_ttnn = ttnn.from_torch(
        torch.from_numpy(weight_t_np.astype(np.float32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    out = ttnn.matmul(act_ttnn, w_ttnn)
    result = ttnn.to_torch(out).float().numpy().reshape(M, N)
    ttnn.deallocate(out)
    ttnn.deallocate(w_ttnn)
    return result


def run_bfp4_matmul(act_ttnn, weight_t_np, device, M, N):
    """Run bfp4 matmul (lossless for ternary) for comparison."""
    w_ttnn = ttnn.from_torch(
        torch.from_numpy(weight_t_np.astype(np.float32)),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    out = ttnn.matmul(act_ttnn, w_ttnn)
    result = ttnn.to_torch(out).float().numpy().reshape(M, N)
    ttnn.deallocate(out)
    ttnn.deallocate(w_ttnn)
    return result


def report(name, out, ref):
    """Print comparison statistics."""
    corr = np.corrcoef(out.flatten(), ref.flatten())[0, 1]
    out_mean = np.mean(np.abs(out))
    ref_mean = np.mean(np.abs(ref))
    max_err = np.max(np.abs(out - ref))
    rel_bias = (out_mean / ref_mean - 1) * 100 if ref_mean > 0 else float('nan')
    print(f"  {name:20s}: corr={corr:.6f}  mean={out_mean:.4f} vs {ref_mean:.4f}  "
          f"bias={rel_bias:+.2f}%  max_err={max_err:.4f}")
    return rel_bias


# ─────────────────────────────────────────────────────────────
# Test 1: Identity readback
# ─────────────────────────────────────────────────────────────
def test_identity_readback(device):
    """Multiply identity activation by packed weight to verify unpacking."""
    print("\n" + "=" * 60)
    print("TEST 1: Identity readback (I @ W = W)")
    print("=" * 60)

    M, K, N = 32, 32, 32  # single tile
    rng = np.random.default_rng(42)

    # Create known ternary weight
    weight_raw = rng.choice([-1, 0, 1], size=(N, K)).astype(np.int8)
    scale = 1.0  # no scaling for clarity
    weight_t = weight_raw.T.astype(np.float32)  # (K, N) = (32, 32)

    # Identity activation
    act = np.eye(M, K, dtype=np.float32)

    # Reference
    ref = act @ weight_t  # = weight_t (since act = I)

    # Ternary matmul
    act_ttnn = make_act_ttnn(act, device)
    packed_ttnn = make_packed_ttnn(weight_raw, scale, device)
    ternary_out = run_ternary_matmul(act_ttnn, packed_ttnn, M, N)

    # bf16 matmul
    bf16_out = run_bf16_matmul(act_ttnn, weight_t, device, M, N)

    report("ternary_matmul", ternary_out, ref)
    report("bf16_matmul", bf16_out, ref)

    # Element-wise comparison
    diff = ternary_out - ref
    nonzero_diff = np.count_nonzero(np.abs(diff) > 0.01)
    print(f"\n  Element-wise: {nonzero_diff}/{M*N} elements differ by >0.01")
    if nonzero_diff > 0 and nonzero_diff <= 20:
        indices = np.argwhere(np.abs(diff) > 0.01)
        for idx in indices[:10]:
            r, c = idx
            print(f"    [{r},{c}]: ternary={ternary_out[r,c]:.4f} ref={ref[r,c]:.4f} "
                  f"diff={diff[r,c]:.4f}")

    ttnn.deallocate(act_ttnn)
    ttnn.deallocate(packed_ttnn)
    return nonzero_diff


# ─────────────────────────────────────────────────────────────
# Test 2: K-scaling — bias vs number of K tiles
# ─────────────────────────────────────────────────────────────
def test_k_scaling(device):
    """Measure bias as a function of K dimension."""
    print("\n" + "=" * 60)
    print("TEST 2: Bias vs K tiles")
    print("=" * 60)

    M, N = 32, 32
    rng = np.random.default_rng(123)

    k_values = [32, 64, 128, 256, 512, 2048]
    print(f"  {'K':>6} {'Kt':>4} {'ternary_bias%':>14} {'bf16_bias%':>12} {'bfp4_bias%':>12}")
    print(f"  {'─'*6} {'─'*4} {'─'*14} {'─'*12} {'─'*12}")

    for K in k_values:
        Kt = K // 32

        # Random activation and ternary weight
        act = rng.standard_normal((M, K)).astype(np.float32)
        weight_raw = rng.choice([-1, 0, 1], size=(N, K)).astype(np.int8)
        scale = 1.0
        weight_t = weight_raw.T.astype(np.float32)

        ref = act @ weight_t

        act_ttnn = make_act_ttnn(act, device)
        packed_ttnn = make_packed_ttnn(weight_raw, scale, device)

        # Ternary
        ternary_out = run_ternary_matmul(act_ttnn, packed_ttnn, M, N)

        # bf16
        bf16_out = run_bf16_matmul(act_ttnn, weight_t, device, M, N)

        # bfp4
        bfp4_out = run_bfp4_matmul(act_ttnn, weight_t, device, M, N)

        ref_mean = np.mean(np.abs(ref))
        t_bias = (np.mean(np.abs(ternary_out)) / ref_mean - 1) * 100
        b16_bias = (np.mean(np.abs(bf16_out)) / ref_mean - 1) * 100
        b4_bias = (np.mean(np.abs(bfp4_out)) / ref_mean - 1) * 100

        print(f"  {K:>6} {Kt:>4} {t_bias:>+13.2f}% {b16_bias:>+11.2f}% {b4_bias:>+11.2f}%")

        ttnn.deallocate(act_ttnn)
        ttnn.deallocate(packed_ttnn)


# ─────────────────────────────────────────────────────────────
# Test 3: All-ones weight — isolate weight errors
# ─────────────────────────────────────────────────────────────
def test_all_ones_weight(device):
    """Use all-ones weight to isolate activation rounding effects."""
    print("\n" + "=" * 60)
    print("TEST 3: All-ones weight (isolate activation rounding)")
    print("=" * 60)

    M, K, N = 32, 256, 32
    rng = np.random.default_rng(456)

    act = rng.standard_normal((M, K)).astype(np.float32)
    weight_raw = np.ones((N, K), dtype=np.int8)  # all +1
    scale = 1.0
    weight_t = weight_raw.T.astype(np.float32)  # (K, N) all +1

    # ref: each output[i,j] = sum of act[i, :]
    ref = act @ weight_t

    act_ttnn = make_act_ttnn(act, device)
    packed_ttnn = make_packed_ttnn(weight_raw, scale, device)

    ternary_out = run_ternary_matmul(act_ttnn, packed_ttnn, M, N)
    bf16_out = run_bf16_matmul(act_ttnn, weight_t, device, M, N)
    bfp4_out = run_bfp4_matmul(act_ttnn, weight_t, device, M, N)

    report("ternary_matmul", ternary_out, ref)
    report("bf16_matmul", bf16_out, ref)
    report("bfp4_matmul", bfp4_out, ref)

    ttnn.deallocate(act_ttnn)
    ttnn.deallocate(packed_ttnn)


# ─────────────────────────────────────────────────────────────
# Test 4: Per-column bias — check if bias varies by N column
# ─────────────────────────────────────────────────────────────
def test_per_column_bias(device):
    """Check if bias varies by output column (N tile position)."""
    print("\n" + "=" * 60)
    print("TEST 4: Per-column bias analysis")
    print("=" * 60)

    M, K, N = 32, 2048, 256  # 8 N tiles
    rng = np.random.default_rng(789)

    act = rng.standard_normal((M, K)).astype(np.float32)
    weight_fp32 = rng.standard_normal((N, K)).astype(np.float32)
    weight_quant, scale = weight_quant_ternary(weight_fp32)
    weight_raw = np.round(weight_quant).astype(np.int8)
    weight_t = weight_raw.T.astype(np.float32)

    ref = act @ weight_t

    act_ttnn = make_act_ttnn(act, device)
    packed_ttnn = make_packed_ttnn(weight_raw, float(scale), device)

    ternary_out = run_ternary_matmul(act_ttnn, packed_ttnn, M, N)

    # Per-tile-column analysis
    print(f"  {'N_tile':>7} {'cols':>10} {'mean_bias%':>12} {'corr':>8}")
    print(f"  {'─'*7} {'─'*10} {'─'*12} {'─'*8}")
    for nt in range(N // 32):
        c_start = nt * 32
        c_end = (nt + 1) * 32
        ref_col = ref[:, c_start:c_end]
        out_col = ternary_out[:, c_start:c_end]
        ref_m = np.mean(np.abs(ref_col))
        out_m = np.mean(np.abs(out_col))
        bias = (out_m / ref_m - 1) * 100 if ref_m > 0 else 0
        corr = np.corrcoef(out_col.flatten(), ref_col.flatten())[0, 1]
        print(f"  {nt:>7} {c_start:>4}-{c_end-1:<4} {bias:>+11.2f}% {corr:>8.6f}")

    ttnn.deallocate(act_ttnn)
    ttnn.deallocate(packed_ttnn)


# ─────────────────────────────────────────────────────────────
# Test 5: FFN-size matmul (the actual failing case)
# ─────────────────────────────────────────────────────────────
def test_ffn_size(device):
    """Test at FFN dimensions: M=32, K=2560, N=6912 and K=6912, N=2560."""
    print("\n" + "=" * 60)
    print("TEST 5: FFN-size matmul")
    print("=" * 60)

    rng = np.random.default_rng(2024)

    for label, (M, K, N) in [("gate_up", (32, 2560, 6912)),
                               ("down", (32, 6912, 2560))]:
        print(f"\n  --- {label}: M={M}, K={K}, N={N} ---")

        act = rng.standard_normal((M, K)).astype(np.float32)
        weight_fp32 = rng.standard_normal((N, K)).astype(np.float32)
        weight_quant, scale = weight_quant_ternary(weight_fp32)
        weight_raw = np.round(weight_quant).astype(np.int8)
        weight_t = weight_raw.T.astype(np.float32)

        ref = act @ weight_t

        act_ttnn = make_act_ttnn(act, device)
        packed_ttnn = make_packed_ttnn(weight_raw, float(scale), device)

        ternary_out = run_ternary_matmul(act_ttnn, packed_ttnn, M, N)
        bfp4_out = run_bfp4_matmul(act_ttnn, weight_t, device, M, N)

        report("ternary_matmul", ternary_out, ref)
        report("bfp4_matmul", bfp4_out, ref)

        ttnn.deallocate(act_ttnn)
        ttnn.deallocate(packed_ttnn)


def main():
    print("=" * 60)
    print("Ternary Matmul Bias Diagnostic")
    print("=" * 60)

    from bitnet_tt.utils.device import get_device, close_device
    device = get_device()

    try:
        errors = test_identity_readback(device)
        test_k_scaling(device)
        test_all_ones_weight(device)
        test_per_column_bias(device)
        test_ffn_size(device)
    finally:
        close_device()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("\nInterpretation guide:")
    print("- Test 1 errors > 0: reader unpack or tile format issue")
    print("- Test 2 bias scales linearly with Kt: accumulation issue")
    print("- Test 2 bias constant regardless of Kt: per-tile issue")
    print("- Test 3 all-ones bias: activation rounding (HiFi2 BFP8)")
    print("- Test 4 bias varies by column: multi-core or dest issue")


if __name__ == "__main__":
    main()
