"""
Utilities for analyzing BitNet ternary weight structure.

These helpers are intentionally lightweight so they can be reused in tests and
offline analysis scripts without pulling in TT-NN device state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from bitnet_tt.utils.quantization import pack_ternary_weights, weight_quant_ternary


def ternary_distribution(weight_quant: NDArray[np.integer]) -> dict[str, float]:
    """Return element-wise fractions for a quantized ternary weight tensor."""
    weight_quant = np.asarray(weight_quant, dtype=np.int8)
    total = int(weight_quant.size)
    if total == 0:
        return {"zero_frac": 0.0, "pos_frac": 0.0, "neg_frac": 0.0}
    return {
        "zero_frac": float(np.count_nonzero(weight_quant == 0) / total),
        "pos_frac": float(np.count_nonzero(weight_quant > 0) / total),
        "neg_frac": float(np.count_nonzero(weight_quant < 0) / total),
    }


def packed_ratio_vs_fp32(weight_quant: NDArray[np.integer]) -> float:
    """
    Return packed ternary size relative to dense fp32 storage for the same tensor.

    The current packer stores four 2-bit values per byte, so the theoretical best
    ratio is 1 / 16 relative to fp32.
    """
    weight_quant = np.asarray(weight_quant, dtype=np.int8)
    dense_fp32_bytes = float(weight_quant.astype(np.float32).nbytes)
    if dense_fp32_bytes == 0:
        return 0.0
    packed_bytes = float(pack_ternary_weights(weight_quant).nbytes)
    return packed_bytes / dense_fp32_bytes


def all_zero_tile_fraction(weight_quant: NDArray[np.integer], tile_size: int) -> float:
    """
    Return the fraction of tiles that are entirely zero after padding.

    This is the metric that matters for tile/block-sparse kernels. A high
    element-wise zero rate is not enough if no tile is fully zero.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    weight_quant = np.asarray(weight_quant, dtype=np.int8)
    if weight_quant.ndim != 2:
        raise ValueError("all_zero_tile_fraction expects a rank-2 weight matrix")

    height, width = weight_quant.shape
    padded_height = ((height + tile_size - 1) // tile_size) * tile_size
    padded_width = ((width + tile_size - 1) // tile_size) * tile_size

    if padded_height != height or padded_width != width:
        padded = np.zeros((padded_height, padded_width), dtype=np.int8)
        padded[:height, :width] = weight_quant
        weight_quant = padded

    tiles = weight_quant.reshape(
        padded_height // tile_size,
        tile_size,
        padded_width // tile_size,
        tile_size,
    ).transpose(0, 2, 1, 3)
    zero_tiles = np.all(tiles == 0, axis=(2, 3))
    return float(zero_tiles.mean())


def summarize_weight(weight: NDArray[np.floating], tile_sizes: tuple[int, ...] = (32, 64)) -> dict[str, Any]:
    """
    Quantize a float weight matrix and summarize the ternary structure.
    """
    weight_quant, _ = weight_quant_ternary(np.asarray(weight, dtype=np.float32))
    summary: dict[str, Any] = {
        "elements": int(weight_quant.size),
        **ternary_distribution(weight_quant),
        "pack_ratio_vs_fp32": packed_ratio_vs_fp32(weight_quant),
    }
    for tile_size in tile_sizes:
        summary[f"tile{tile_size}_all_zero_frac"] = all_zero_tile_fraction(weight_quant, tile_size)
    return summary
