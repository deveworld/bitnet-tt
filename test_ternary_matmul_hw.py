#!/usr/bin/env python3
"""
Hardware test: verify ternary_matmul kernel correctness on p150a.

Compares:
  1. Standard bf16 matmul: activation @ (ternary_weight * scale)
  2. Custom ternary_matmul: 2-bit packed weights -> unpack in reader -> matmul -> scale

Both should produce identical results.
"""

import time
import numpy as np
import torch
import ttnn

from bitnet_tt.kernels.pack import pack_ternary_tilized, PACKED_TILE_BYTES
from bitnet_tt.utils.quantization import weight_quant_ternary


def test_ternary_matmul(device, M=32, K=64, N=32):
    """Test ternary_matmul correctness for a small matrix."""
    print(f"\n--- Test: M={M}, K={K}, N={N} ---")

    # Generate random ternary weights
    rng = np.random.default_rng(42)
    weight_fp32 = rng.standard_normal((N, K)).astype(np.float32)  # (out, in)
    weight_quant, scale = weight_quant_ternary(weight_fp32)
    scale = float(scale)
    print(f"Weight scale: {scale:.4f}")
    print(f"Ternary distribution: +1={np.mean(weight_quant==1):.1%}, 0={np.mean(weight_quant==0):.1%}, -1={np.mean(weight_quant==-1):.1%}")

    # Random activation
    act_np = rng.standard_normal((M, K)).astype(np.float32)

    # === Reference: standard bf16 matmul ===
    # weight_t = quantized weight transposed (K, N), scaled
    weight_t_scaled = (weight_quant.T.astype(np.float32)) * scale  # (K, N)
    ref_output = act_np @ weight_t_scaled  # (M, N)

    # === Test path 1: ttnn.matmul with bf16 ===
    act_ttnn = ttnn.from_torch(
        torch.from_numpy(act_np),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    weight_bf16 = ttnn.from_torch(
        torch.from_numpy(weight_t_scaled),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    out_bf16 = ttnn.matmul(act_ttnn, weight_bf16)
    out_bf16_np = ttnn.to_torch(out_bf16).float().numpy()
    ttnn.deallocate(out_bf16)
    ttnn.deallocate(weight_bf16)

    # Check bf16 vs numpy reference
    bf16_err = np.max(np.abs(ref_output - out_bf16_np))
    bf16_corr = np.corrcoef(ref_output.flatten(), out_bf16_np.flatten())[0, 1]
    print(f"bf16 matmul: max_err={bf16_err:.4f}, corr={bf16_corr:.6f}")

    # === Test path 2: packed ternary matmul via ttnn.experimental.ternary_matmul ===
    # True 2-bit storage: pack as BFP2_b then strip the constant exponent block
    # (256B mantissa = 64 uint32 per 32x32 tile). Device synthesizes the
    # exponent in L1.
    from ttnn._ttnn.bfp_utils import pack_bfp2

    w_kn = weight_quant.T.astype(np.float32)  # (K, N)
    Kt, Nt = K // 32, N // 32
    w_tiled = (
        w_kn.reshape(Kt, 32, Nt, 32)
        .transpose(0, 2, 1, 3)
        .reshape(Kt * Nt, 1024)
    )
    flat = w_tiled.reshape(-1).astype(np.float32)
    packed_full = np.asarray(pack_bfp2(flat, row_major_input=True, is_exp_a=False))
    num_tiles = Kt * Nt
    packed_full = packed_full.reshape(num_tiles, 80)
    mantissa_only = packed_full[:, 16:].copy()  # (num_tiles, 64)
    print(f"Packed: {num_tiles} tiles, {num_tiles * 256} bytes (vs {M*K*2} bf16 bytes)")
    packed_torch = torch.from_numpy(mantissa_only.astype(np.int32)).to(torch.int32)
    packed_ttnn = ttnn.from_torch(
        packed_torch,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    try:
        out_ternary = ttnn.experimental.ternary_matmul(
            act_ttnn, packed_ttnn,
            use_packed_ternary=True,
        )
        out_ternary_np = ttnn.to_torch(out_ternary).float().numpy().reshape(M, N)

        # Scale the output (packed weights are {-1,0,+1}, scale applied post-matmul)
        out_ternary_np *= scale

        ternary_err = np.max(np.abs(ref_output - out_ternary_np))
        ternary_corr = np.corrcoef(ref_output.flatten(), out_ternary_np.flatten())[0, 1]
        print(f"ternary_matmul: max_err={ternary_err:.4f}, corr={ternary_corr:.6f}")

        # Compare ternary vs bf16
        diff = np.max(np.abs(out_bf16_np - out_ternary_np))
        print(f"ternary vs bf16 diff: {diff:.4f}")

        ttnn.deallocate(out_ternary)
        print("PASSED" if ternary_corr > 0.99 else "FAILED")
    except Exception as e:
        print(f"ternary_matmul FAILED: {e}")
        import traceback
        traceback.print_exc()

    ttnn.deallocate(act_ttnn)
    ttnn.deallocate(packed_ttnn)


def main():
    print("=" * 60)
    print("Ternary Matmul Hardware Test")
    print("=" * 60)

    from bitnet_tt.utils.device import get_device, close_device
    device = get_device()

    # Small test first
    test_ternary_matmul(device, M=32, K=64, N=32)

    # Larger test
    test_ternary_matmul(device, M=32, K=128, N=64)

    close_device()
    print("\nDone.")


if __name__ == "__main__":
    main()
