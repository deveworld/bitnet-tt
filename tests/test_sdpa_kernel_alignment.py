"""Phase K1 unit harness for tt-metal SDPA kernel alignment.

Feeds a fixed (Q, K, V) triple -- captured at layer 0 post-RoPE from a
single prefill of "The capital of France is" via BITNET_LOCALIZE=1 -- through:

  1. ttnn.transformer.scaled_dot_product_attention (stock fused kernel)
  2. torch.nn.functional.scaled_dot_product_attention (PyTorch reference)
  3. Decomposed sub-ops:
     - Q @ K^T only (matmul-only RFE)
     - softmax(Q @ K^T * scale) (softmax-only RFE)
     - softmax(...) @ V  (attn*V-only RFE)

Reports per-op relative Frobenius error so Phase K2/K3/K4 kernel edits can
be gated on the right sub-op.

Run:
  pytest tests/test_sdpa_kernel_alignment.py -s
  # or, if the .npz capture is missing:
  BITNET_LOCALIZE=1 python3 scripts/pcc_localize.py --capture-tt-only
  pytest tests/test_sdpa_kernel_alignment.py -s

The test is skipped (not failed) if no capture and no device are available.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F


CAPTURE_DIR = Path("/tmp/bitnet_localize")
FIXED_PROMPT_TAG = "p0"

HEAD_DIM = 128
NUM_HEADS = 20
NUM_KV_HEADS = 4


def _rfe(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    denom = np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(a - b) / denom)


def _load_fixed_qkv() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load layer-0 post-RoPE Q/K/V from pcc_localize TT capture.

    Falls back to a deterministic synthetic Q/K/V if no capture is present --
    harness still runs, but the RFE numbers are no longer reproducible against
    session-8's 0.240.
    """
    tt_path = CAPTURE_DIR / f"tt_{FIXED_PROMPT_TAG}.npz"
    if tt_path.exists():
        npz = np.load(tt_path, allow_pickle=True)
        q_key = "L0.post_rope_query"
        k_key = "L0.post_rope_key"
        v_key = "L0.post_qkv_value"
        if q_key in npz.files and k_key in npz.files and v_key in npz.files:
            q = npz[q_key]
            k = npz[k_key]
            v = npz[v_key]
            return q, k, v
    rng = np.random.default_rng(0)
    b, s = 1, 6
    q = rng.standard_normal((b, NUM_HEADS, s, HEAD_DIM)).astype(np.float32) * 0.1
    k = rng.standard_normal((b, NUM_KV_HEADS, s, HEAD_DIM)).astype(np.float32) * 0.1
    v = rng.standard_normal((b, NUM_KV_HEADS, s, HEAD_DIM)).astype(np.float32) * 0.1
    return q, k, v


def _expand_kv_groups(kv: np.ndarray, groups: int) -> np.ndarray:
    """GQA: repeat each KV head `groups` times along head axis."""
    return np.repeat(kv, groups, axis=1)


def _torch_sdpa_decomposed(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float,
    dtype: torch.dtype = torch.float32,
) -> dict[str, np.ndarray]:
    """Run PyTorch reference SDPA and return intermediate sub-op outputs."""
    q_t = torch.from_numpy(q).to(dtype)
    k_t = torch.from_numpy(k).to(dtype)
    v_t = torch.from_numpy(v).to(dtype)

    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    s = scores.shape[-1]
    mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
    masked = scores.masked_fill(mask, float("-inf"))
    probs = F.softmax(masked, dim=-1)
    out = torch.matmul(probs, v_t)

    return {
        "scores": scores.to(torch.float32).numpy(),
        "masked_scores": masked.to(torch.float32).numpy(),
        "probs": probs.to(torch.float32).numpy(),
        "out": out.to(torch.float32).numpy(),
    }


def _run_ttnn_sdpa(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float
) -> np.ndarray | None:
    """Run stock ttnn.transformer.scaled_dot_product_attention.

    Returns None if no device is available (CI / host-only run).
    """
    try:
        import ttnn
        import bitnet_tt  # noqa: F401  -- uint-compat shim side-effects
        from bitnet_tt.utils.device import get_device
    except Exception as exc:
        pytest.skip(f"ttnn/bitnet_tt unavailable: {exc}")
        return None

    try:
        device = get_device()
    except Exception as exc:
        pytest.skip(f"no TT device: {exc}")
        return None

    def _to_ttnn(x: np.ndarray) -> "ttnn.Tensor":
        t = torch.from_numpy(x).to(torch.bfloat16)
        return ttnn.from_torch(
            t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

    q_tt = _to_ttnn(q)
    k_tt = _to_ttnn(k)
    v_tt = _to_ttnn(v)

    out_tt = ttnn.transformer.scaled_dot_product_attention(
        q_tt, k_tt, v_tt, attn_mask=None, is_causal=True, scale=scale
    )
    out_np = ttnn.to_torch(out_tt).to(torch.float32).numpy()

    for t in (q_tt, k_tt, v_tt, out_tt):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return out_np


def _run_ttnn_decomposed(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float
) -> dict[str, np.ndarray] | None:
    """Run Q*K^T, softmax, attn*V as separate ttnn primitives for sub-op RFE."""
    try:
        import ttnn
        import bitnet_tt  # noqa: F401
        from bitnet_tt.utils.device import get_device
    except Exception as exc:
        pytest.skip(f"ttnn unavailable: {exc}")
        return None
    try:
        device = get_device()
    except Exception as exc:
        pytest.skip(f"no TT device: {exc}")
        return None

    def _to_ttnn(x: np.ndarray) -> "ttnn.Tensor":
        t = torch.from_numpy(x).to(torch.bfloat16)
        return ttnn.from_torch(
            t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

    q_tt = _to_ttnn(q)
    k_tt = _to_ttnn(k)
    v_tt = _to_ttnn(v)

    k_t_tt = ttnn.transpose(k_tt, -1, -2)
    scores_tt = ttnn.matmul(q_tt, k_t_tt)
    scores_tt = ttnn.multiply(scores_tt, scale)
    scores_np = ttnn.to_torch(scores_tt).to(torch.float32).numpy()

    s = scores_np.shape[-1]
    mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
    mask_bf = torch.zeros(s, s, dtype=torch.bfloat16)
    mask_bf.masked_fill_(mask, float("-inf"))
    mask_tt = ttnn.from_torch(
        mask_bf.reshape(1, 1, s, s),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    scores_tt = ttnn.add(scores_tt, mask_tt)
    probs_tt = ttnn.softmax(scores_tt, dim=-1)
    probs_np = ttnn.to_torch(probs_tt).to(torch.float32).numpy()

    out_tt = ttnn.matmul(probs_tt, v_tt)
    out_np = ttnn.to_torch(out_tt).to(torch.float32).numpy()

    for t in (q_tt, k_tt, v_tt, k_t_tt, scores_tt, mask_tt, probs_tt, out_tt):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass

    return {"scores": scores_np, "probs": probs_np, "out": out_np}


def test_sdpa_fused_vs_pytorch_rfe(capsys):
    q, k, v = _load_fixed_qkv()
    if k.shape[1] != q.shape[1]:
        groups = q.shape[1] // k.shape[1]
        k_exp = _expand_kv_groups(k, groups)
        v_exp = _expand_kv_groups(v, groups)
    else:
        k_exp = k
        v_exp = v

    scale = 1.0 / math.sqrt(q.shape[-1])
    ref_fp32 = _torch_sdpa_decomposed(q, k_exp, v_exp, scale, torch.float32)
    ref_bf16 = _torch_sdpa_decomposed(q, k_exp, v_exp, scale, torch.bfloat16)

    tt_fused = _run_ttnn_sdpa(q, k_exp, v_exp, scale)
    tt_decomp = _run_ttnn_decomposed(q, k_exp, v_exp, scale)

    rfe_fused_fp32 = _rfe(tt_fused, ref_fp32["out"])
    rfe_fused_bf16 = _rfe(tt_fused, ref_bf16["out"])
    rfe_bf16_floor = _rfe(ref_bf16["out"], ref_fp32["out"])
    rfe_scores = _rfe(tt_decomp["scores"], ref_fp32["scores"])
    rfe_probs = _rfe(tt_decomp["probs"], ref_fp32["probs"])
    rfe_attnV = _rfe(tt_decomp["out"], ref_fp32["out"])

    print()
    print(f"# SDPA Kernel Alignment (Phase K1) — Q={q.shape} K={k_exp.shape} V={v_exp.shape}")
    print(f"  fused SDPA vs torch.F.sdpa(fp32)   RFE = {rfe_fused_fp32:.6f}")
    print(f"  fused SDPA vs torch.F.sdpa(bf16)   RFE = {rfe_fused_bf16:.6f}")
    print(f"  bf16 torch vs fp32 torch (floor)   RFE = {rfe_bf16_floor:.6f}")
    print(f"  Q@K^T matmul-only vs fp32           RFE = {rfe_scores:.6f}")
    print(f"  softmax-only vs fp32                RFE = {rfe_probs:.6f}")
    print(f"  attn@V decomposed end-to-end vs fp32 RFE = {rfe_attnV:.6f}")
    print(f"  NET drift attributable to TT kernels  = {max(0.0, rfe_fused_fp32 - rfe_bf16_floor):.6f}")

    assert np.isfinite(rfe_fused_fp32), "fused RFE is NaN/inf"
    assert rfe_fused_fp32 < 1.0, f"fused RFE {rfe_fused_fp32} too large — harness broken"


def test_env_variant_flag_plumbing():
    """BITNET_SDPA_KERNEL_VARIANT is read at module import and validated."""
    import bitnet_tt.layers.attention as att_mod

    assert hasattr(att_mod, "_BITNET_SDPA_KERNEL_VARIANT")
    assert att_mod._BITNET_SDPA_KERNEL_VARIANT in att_mod._SDPA_KERNEL_VARIANTS_KNOWN
