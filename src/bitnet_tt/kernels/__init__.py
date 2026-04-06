"""
Custom kernels for ternary weight optimization on Tenstorrent hardware.

This module provides:
- 2-bit ternary weight packing (pack.py)
- TernaryLinear layer with BFP4/custom kernel support (ternary_linear.py)
- C++ kernel sources for custom ternary matmul (device/)
"""

from bitnet_tt.kernels.pack import (
    pack_ternary_tilized,
    unpack_ternary_tilized,
)
from bitnet_tt.kernels.ternary_linear import TernaryLinear

__all__ = [
    "pack_ternary_tilized",
    "unpack_ternary_tilized",
    "TernaryLinear",
]
