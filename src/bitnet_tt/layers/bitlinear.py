"""
BitLinear layer implementation for BitNet b1.58 using TT-NN.

This module provides TT-NN native implementation of the BitLinear layer,
which uses ternary weights {-1, 0, +1} and 8-bit activation quantization.
"""

import numpy as np
import torch
import ttnn
from numpy.typing import NDArray

from bitnet_tt.utils.quantization import weight_quant_ternary


def numpy_to_ttnn(
    array: NDArray[np.floating],
    device: ttnn.Device,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """
    Convert numpy array to ttnn tensor.

    Args:
        array: NumPy array (should be float32)
        device: TT-NN device
        layout: Tensor layout (default: TILE_LAYOUT)
        memory_config: Memory configuration (default: DRAM interleaved)

    Returns:
        ttnn.Tensor on device
    """
    if array.dtype != np.float32:
        array = array.astype(np.float32)

    torch_tensor = torch.from_numpy(array)
    tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )
    return tensor


def numpy_int_to_ttnn(
    array: NDArray[np.integer],
    device: ttnn.Device,
) -> ttnn.Tensor:
    """
    Convert integer numpy array to ttnn tensor for token IDs.

    Args:
        array: NumPy integer array (e.g., token IDs)
        device: TT-NN device

    Returns:
        ttnn.Tensor on device with uint32 dtype and ROW_MAJOR_LAYOUT
    """
    # Ensure int32 for torch compatibility (torch doesn't have uint32)
    if array.dtype != np.int32:
        array = array.astype(np.int32)

    # Convert via torch
    torch_tensor = torch.from_numpy(array)
    tensor = ttnn.from_torch(
        torch_tensor, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    return tensor


def quantize_and_transpose_weight(
    weight: NDArray[np.floating],
) -> tuple[NDArray[np.float32], float]:
    """
    Quantize weight to ternary {-1, 0, +1} and pre-transpose to (in, out).

    Returns pure ternary values (not scaled) so that BFP4 storage is lossless.
    The scale factor must be applied post-matmul.
    """
    weight_quant, scale = weight_quant_ternary(weight.astype(np.float32))
    return weight_quant.T.astype(np.float32).copy(), float(scale)


def ttnn_to_numpy(tensor: ttnn.Tensor) -> NDArray[np.floating]:
    """
    Convert ttnn tensor back to numpy array.

    Args:
        tensor: TT-NN tensor

    Returns:
        NumPy array
    """
    # Use to_torch then convert to numpy (bfloat16 -> float32 for numpy compatibility)
    torch_tensor = ttnn.to_torch(tensor)
    return torch_tensor.float().numpy()


class Linear:
    """
    TT-NN native Linear layer with weight quantization.

    For BF16 BitNet models: applies ternary weight quantization at load time.
    HuggingFace BitLinear applies weight_quant() which computes:
        s = 1.0 / weight.abs().mean()
        result = (weight * s).round().clamp(-1, 1) / s

    This quantizes weights to {-scale, 0, +scale} where scale = mean(|weight|).
    Quantization is pre-computed during load_weights() for efficiency.

    Optimization: Uses configurable compute kernel fidelity:
    - HiFi2 (default): ~2x speedup, good accuracy (BFP8)
    - LoFi: ~3.6x speedup, lower accuracy (BFP4) - suitable for MLP layers

    Weight dtype options:
    - "bf16": bfloat16 baseline (default, backward compatible)
    - "bfp4": bfloat4_b — 4x DRAM bandwidth reduction for ternary weights
    - "bfp8": bfloat8_b — 2x DRAM bandwidth reduction
    """

    # Map string names to ttnn data types
    _WEIGHT_DTYPE_MAP = {
        "bf16": ttnn.bfloat16,
        "bfp4": ttnn.bfloat4_b,
        "bfp8": ttnn.bfloat8_b,
    }

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: ttnn.Device,
        compute_fidelity: str = "hifi2",
        weight_dtype: str = "bfp4",
    ) -> None:
        """
        Initialize Linear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            device: TT-NN device
            compute_fidelity: One of "hifi4", "hifi2" (default), "lofi"
            weight_dtype: Weight storage format — "bf16", "bfp4", or "bfp8"
        """
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.weight: ttnn.Tensor | None = None
        self.weight_scale: float = 1.0
        self._fidelity = compute_fidelity
        self._weight_dtype_name = weight_dtype
        self._ttnn_weight_dtype = self._WEIGHT_DTYPE_MAP.get(weight_dtype, ttnn.bfloat16)

        # Initialize compute kernel config for faster matmul
        self._compute_kernel_config = None
        try:
            from bitnet_tt.config import get_compute_kernel_config

            self._compute_kernel_config = get_compute_kernel_config(compute_fidelity)
        except Exception:
            pass

    def load_weights(self, weight: NDArray[np.floating]) -> None:
        """
        Load and pre-quantize weights to device.

        Stores pure ternary {-1, 0, +1} values (lossless in BFP4) and
        applies the scale factor as a cheap post-matmul scalar multiply.

        Args:
            weight: Weight array of shape (out_features, in_features)
        """
        weight_t, scale = quantize_and_transpose_weight(weight)
        self.load_pretransposed_weight(weight_t, scale=scale)

    def load_pretransposed_weight(
        self, weight_t: NDArray[np.floating], scale: float = 1.0
    ) -> None:
        """
        Load a weight that is already quantized and transposed to (in, out).

        Args:
            weight_t: Pre-transposed weight array (in_features, out_features).
            scale: Post-matmul scale factor. Set to 1.0 for fused weights
                   whose scales are applied externally (e.g. fused gate_up).
        """
        self.weight_scale = scale
        torch_tensor = torch.from_numpy(weight_t.astype(np.float32))
        self.weight = ttnn.from_torch(
            torch_tensor,
            dtype=self._ttnn_weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass with pre-quantized and pre-transposed weights.

        Args:
            x: Input tensor of shape (batch, seq_len, in_features)

        Returns:
            Output tensor of shape (batch, seq_len, out_features)
        """
        if self.weight is None:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")

        if self._compute_kernel_config is not None:
            out = ttnn.matmul(
                x,
                self.weight,
                compute_kernel_config=self._compute_kernel_config,
            )
        else:
            out = ttnn.matmul(x, self.weight)

        if self.weight_scale != 1.0:
            out = ttnn.multiply(out, self.weight_scale)
        return out


class RMSNorm:
    """
    TT-NN native RMSNorm layer.
    """

    def __init__(
        self,
        hidden_size: int,
        device: ttnn.Device,
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize RMSNorm.

        Args:
            hidden_size: Hidden dimension size
            device: TT-NN device
            eps: Epsilon for numerical stability
        """
        self.hidden_size = hidden_size
        self.device = device
        self.eps = eps
        self.weight: ttnn.Tensor | None = None

    def load_weights(self, weight: NDArray[np.floating]) -> None:
        """
        Load norm weights to device.

        Args:
            weight: Weight array of shape (hidden_size,)
        """
        weight_2d = weight.reshape(1, 1, 1, -1).astype(np.float32)
        self.weight = numpy_to_ttnn(weight_2d, self.device)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight)
