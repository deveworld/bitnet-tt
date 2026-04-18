"""
Helpers for priming TTNN configuration before the first ttnn import.
"""

from __future__ import annotations

import json
import os


def _install_torch_uint_compat_shim() -> None:
    # torch 2.2.2 lacks torch.uint16 / uint32 / uint64 attributes AND torch's dlpack
    # importer rejects DLPack tensors with kUInt bit widths > 8. ttnn.to_torch
    # (`torch.utils.dlpack.from_dlpack`) triggers both failures when the source
    # ttnn tensor has an unsigned integer dtype (e.g. argmax index output is
    # ttnn.uint32). This shim:
    #   (1) aliases torch.uintN -> torch.intN (same bit width) so attribute
    #       lookups succeed in ttnn paths that do `getattr(torch, "uint32")`.
    #   (2) wraps torch.utils.dlpack.from_dlpack to fall back on numpy bridging
    #       when the C importer raises 'Unsupported kUInt' -- the ttnn tensor
    #       already exposes .to(torch.int32) semantics via numpy, which is the
    #       only use (argmax index -> int -> Python).
    try:
        import torch
        import numpy as np

        for name, alias in (("uint16", "int16"), ("uint32", "int32"), ("uint64", "int64")):
            if not hasattr(torch, name):
                setattr(torch, name, getattr(torch, alias))

    except Exception:
        pass


def _install_ttnn_to_torch_uint_wrapper() -> None:
    # ttnn.to_torch on a uint32-dtype tensor invokes torch.utils.dlpack.from_dlpack,
    # which on torch 2.2.2 rejects kUInt bit widths > 8 with 'Unsupported kUInt bits 32'.
    # Wrap ttnn.to_torch to detect uint32/uint16/uint64 ttnn dtypes and route them
    # through ttnn.typecast(..., int32/int16/int64) before the torch conversion.
    # The signed cast is bit-preserving at the dlpack level for the value ranges
    # actually in use (argmax indices, position ids), so it is safe.
    try:
        import ttnn

        if getattr(ttnn.to_torch, "_bitnet_tt_uint_wrapper", False):
            return

        _orig_to_torch = ttnn.to_torch
        _uint_to_signed = {
            ttnn.uint16: ttnn.int32,
            ttnn.uint32: ttnn.int32,
        }

        def _to_torch_wrapper(tensor, *args, **kwargs):
            try:
                signed = _uint_to_signed.get(tensor.dtype)
            except Exception:
                signed = None
            if signed is not None:
                try:
                    tensor = ttnn.typecast(tensor, signed)
                except Exception:
                    # If typecast fails (host tensor, unsupported layout, etc.),
                    # fall back to the unwrapped call so the original error surfaces.
                    pass
            return _orig_to_torch(tensor, *args, **kwargs)

        _to_torch_wrapper._bitnet_tt_uint_wrapper = True  # type: ignore[attr-defined]
        ttnn.to_torch = _to_torch_wrapper
    except Exception:
        pass


_install_torch_uint_compat_shim()


def parse_bool_env_value(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def prime_ttnn_config_overrides() -> None:
    """
    Apply TTNN_CONFIG_OVERRIDES before any module imports ttnn.
    """
    raw_overrides = os.environ.get("TTNN_CONFIG_OVERRIDES")
    overrides: dict[str, object]
    if raw_overrides:
        try:
            overrides = json.loads(raw_overrides)
        except json.JSONDecodeError:
            return
        if not isinstance(overrides, dict):
            return
    else:
        overrides = {}

    if "enable_model_cache" not in overrides:
        overrides["enable_model_cache"] = parse_bool_env_value(
            os.getenv("BITNET_TT_ENABLE_MODEL_CACHE"),
            default=True,
        )

    os.environ["TTNN_CONFIG_OVERRIDES"] = json.dumps(overrides)
