"""
BitNet-TT: BitNet b1.58 LLM implementation on Tenstorrent Blackhole p150a

This package provides an implementation of Microsoft's BitNet b1.58 architecture
optimized for Tenstorrent's Blackhole accelerator using the TT-NN library.
"""

from bitnet_tt.config import BitNet2B4TConfig, BitNetConfig, BitNetMiniConfig
from bitnet_tt.layers.attention import KVCache

__version__ = "0.1.0"
__all__ = [
    "BitNetConfig",
    "BitNetMiniConfig",
    "BitNet2B4TConfig",
    "KVCache",
]
