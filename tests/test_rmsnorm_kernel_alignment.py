"""Focused RMSNorm kernel alignment harness.

Feeds layer-0 post_self_attn output (a known 0.240-RFE-drifted TT tensor) to:
  1. ttnn.rms_norm (stock, default compute_kernel_config)
  2. ttnn.rms_norm with fp32_dest_acc_en=True (BITNET_RMSNORM_FP32_ACC=1)
  3. PyTorch RMSNorm reference (fp32 + bf16)

Measures output RFE to answer: does the attn_sub_norm amplifier go away
under fp32 accumulation? The K1 harness covered SDPA; this one covers the
other half of the attention chain.

Uses weight = ones(hidden_size) so result matches ||x|| / sqrt(mean(x^2)+eps).
For the real attn_sub_norm weight, load from HF state_dict when available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


CAPTURE_DIR = Path("/tmp/bitnet_localize")


def _rfe(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    denom = np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(a - b) / denom)


def _torch_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float,
                   dtype: torch.dtype = torch.float32) -> np.ndarray:
    x_t = torch.from_numpy(x).to(dtype)
    w_t = torch.from_numpy(weight).to(dtype)
    rms = torch.sqrt(x_t.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_t / rms * w_t).to(torch.float32).numpy()


def _load_attn_sub_norm_input() -> tuple[np.ndarray, np.ndarray] | None:
    """Pull the TT (drifted) and HF (clean) post_self_attn tensors for layer 0."""
    tt_path = CAPTURE_DIR / "tt_p0.npz"
    hf_path = CAPTURE_DIR / "hf_p0.npz"
    if not tt_path.exists() or not hf_path.exists():
        return None
    tt = np.load(tt_path, allow_pickle=True)
    hf = np.load(hf_path, allow_pickle=True)
    key = "L0.post_self_attn"
    if key not in tt.files or key not in hf.files:
        return None
    tt_x = tt[key]
    hf_x = hf[key]
    # TT has batch32 pad + potential shape mismatch; trim to HF seq_len + batch 0.
    while tt_x.ndim > hf_x.ndim:
        tt_x = tt_x.squeeze(0)
    tt_x = tt_x[: hf_x.shape[0]]
    if tt_x.shape[-2] != hf_x.shape[-2]:
        tt_x = tt_x[..., : hf_x.shape[-2], :]
    return tt_x.astype(np.float32), hf_x.astype(np.float32)


def _run_ttnn_rms_norm(x_np: np.ndarray, weight_np: np.ndarray,
                       eps: float, fp32_acc: bool) -> np.ndarray | None:
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

    compute_kernel_config = None
    if fp32_acc:
        # Mirror the config bitlinear.py:_rmsnorm_compute_kernel_config builds
        # when BITNET_RMSNORM_FP32_ACC=1. Duplicated (not imported) so this
        # harness runs independently of bitlinear's env-gated module state.
        try:
            from ttnn import BlackholeComputeKernelConfig, MathFidelity
            ckc_cls = BlackholeComputeKernelConfig
        except Exception:
            from ttnn import WormholeComputeKernelConfig, MathFidelity
            ckc_cls = WormholeComputeKernelConfig
        compute_kernel_config = ckc_cls(
            math_fidelity=MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    x_t = torch.from_numpy(x_np).to(torch.bfloat16)
    w_t = torch.from_numpy(weight_np.reshape(1, 1, 1, -1)).to(torch.bfloat16)

    x_tt = ttnn.from_torch(x_t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    w_tt = ttnn.from_torch(w_t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    kw = {"epsilon": eps, "weight": w_tt}
    if compute_kernel_config is not None:
        kw["compute_kernel_config"] = compute_kernel_config
    y_tt = ttnn.rms_norm(x_tt, **kw)
    y_np = ttnn.to_torch(y_tt).to(torch.float32).numpy()

    for t in (x_tt, w_tt, y_tt):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return y_np


def test_rmsnorm_kernel_alignment():
    pair = _load_attn_sub_norm_input()
    if pair is None:
        pytest.skip("no /tmp/bitnet_localize captures — run pcc_localize first")
    tt_x, hf_x = pair

    # Reshape into [1, 1, s, h] for ttnn.rms_norm (4D expected).
    if tt_x.ndim == 2:
        tt_x = tt_x[None, None]
        hf_x = hf_x[None, None]
    elif tt_x.ndim == 3:
        tt_x = tt_x[None]
        hf_x = hf_x[None]

    eps = 1e-5
    h = tt_x.shape[-1]
    weight = np.ones(h, dtype=np.float32)

    ref_fp32 = _torch_rmsnorm(hf_x, weight, eps, torch.float32)
    ref_bf16 = _torch_rmsnorm(hf_x, weight, eps, torch.bfloat16)

    # TT stock: feeds TT's drifted input through default rms_norm.
    tt_stock = _run_ttnn_rms_norm(tt_x, weight, eps, fp32_acc=False)
    tt_fp32 = _run_ttnn_rms_norm(tt_x, weight, eps, fp32_acc=True)

    rfe_input = _rfe(tt_x, hf_x)
    rfe_stock_out = _rfe(tt_stock, ref_fp32)
    rfe_fp32_out = _rfe(tt_fp32, ref_fp32)
    rfe_bf16_floor = _rfe(ref_bf16, ref_fp32)

    # Amplification = out_RFE / in_RFE. Plan hypothesis: ~1.75x for stock.
    amp_stock = rfe_stock_out / max(rfe_input, 1e-9)
    amp_fp32 = rfe_fp32_out / max(rfe_input, 1e-9)

    print()
    print(f"# RMSNorm kernel alignment — input shape {tt_x.shape}")
    print(f"  input drift (TT vs HF)         RFE = {rfe_input:.6f}")
    print(f"  TT stock rms_norm vs torch fp32 RFE = {rfe_stock_out:.6f}  (amp {amp_stock:.2f}x)")
    print(f"  TT fp32_acc rms_norm vs torch fp32 RFE = {rfe_fp32_out:.6f}  (amp {amp_fp32:.2f}x)")
    print(f"  bf16 torch floor               RFE = {rfe_bf16_floor:.6f}")
    print(f"  fp32_acc delta vs stock        = {rfe_stock_out - rfe_fp32_out:+.6f} "
          f"({(rfe_stock_out - rfe_fp32_out)/max(rfe_stock_out, 1e-9)*100:.1f}%)")

    assert np.isfinite(rfe_stock_out)
    assert np.isfinite(rfe_fp32_out)
