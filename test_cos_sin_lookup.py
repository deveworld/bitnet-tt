#!/usr/bin/env python3
"""Verify that Batch32RotarySetup.lookup_decode_cos_sin matches the
existing _get_cos_sin_torch path numerically. If PCC >= 0.9999 for
every tested position, the lookup is a safe drop-in replacement for
the per-step H2D cos/sin copy."""

import numpy as np
import torch
import ttnn

from bitnet_tt.inference.generator_batch32 import Batch32RotarySetup, PADDED_BATCH
from bitnet_tt.utils.device import get_device, close_device


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).flatten() - a.astype(np.float64).mean()
    b = b.astype(np.float64).flatten() - b.astype(np.float64).mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def main() -> None:
    device = get_device()
    head_dim = 128
    max_seq_len = 4096
    rs = Batch32RotarySetup(
        device=device,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        rope_theta=500000.0,
        use_fused_rope=True,
    )

    failures = 0
    for pos in [0, 1, 5, 31, 64, 127, 512, 1023, 4095]:
        pos_tensor = ttnn.from_torch(
            torch.full((PADDED_BATCH,), pos, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        cos_lut, sin_lut = rs.lookup_decode_cos_sin(pos_tensor)
        cos_ref, sin_ref = rs._get_cos_sin_torch(pos)

        cos_lut_np = ttnn.to_torch(cos_lut).float().numpy().reshape(-1, head_dim)
        sin_lut_np = ttnn.to_torch(sin_lut).float().numpy().reshape(-1, head_dim)
        cos_ref_np = cos_ref.float().numpy().reshape(-1, head_dim)
        sin_ref_np = sin_ref.float().numpy().reshape(-1, head_dim)

        # Compare first row (all rows are same position)
        cos_pcc = pcc(cos_lut_np[0], cos_ref_np[0])
        sin_pcc = pcc(sin_lut_np[0], sin_ref_np[0])
        max_abs_cos = np.abs(cos_lut_np[0] - cos_ref_np[0]).max()
        max_abs_sin = np.abs(sin_lut_np[0] - sin_ref_np[0]).max()
        ok = cos_pcc > 0.9999 and sin_pcc > 0.9999
        status = "OK" if ok else "FAIL"
        print(f"  pos={pos:5d}  cos PCC={cos_pcc:.6f}  sin PCC={sin_pcc:.6f}  "
              f"max|Δ|={max(max_abs_cos, max_abs_sin):.5f}  [{status}]")
        if not ok:
            failures += 1
            print("    cos lut first 8:", cos_lut_np[0][:8])
            print("    cos ref first 8:", cos_ref_np[0][:8])

        ttnn.deallocate(cos_lut)
        ttnn.deallocate(sin_lut)
        ttnn.deallocate(pos_tensor)

    print(f"\n{failures} failures across 9 positions")
    close_device()


if __name__ == "__main__":
    main()
