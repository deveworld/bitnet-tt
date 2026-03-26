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


def quantize_weight_like_hf(weight: NDArray[np.floating]) -> NDArray[np.float32]:
    """
    Quantize a linear weight using the HuggingFace BitLinear rule.

    The returned values stay in float32 so they can be consumed by the current
    TT-NN matmul path without requiring a dedicated ternary kernel.
    """
    weight_fp32 = weight.astype(np.float32)
    scale = np.abs(weight_fp32).mean()
    s = 1.0 / max(scale, 1e-5)
    return np.clip(np.round(weight_fp32 * s), -1, 1).astype(np.float32) / s


def quantize_and_transpose_weight(weight: NDArray[np.floating]) -> NDArray[np.float32]:
    """Quantize a weight matrix and pre-transpose it to (in, out)."""
    return quantize_weight_like_hf(weight).T.copy()


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


class BitLinear:
    """
    TT-NN native BitLinear layer.

    This layer implements the quantized linear transformation used in BitNet:
    1. Apply RMSNorm to input
    2. Quantize activations to 8-bit (absmax per-token)
    3. Use pre-quantized ternary weights {-1, 0, +1}
    4. Perform matrix multiplication
    5. Dequantize output using stored scale factors
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: ttnn.Device,
        eps: float = 1e-5,
        skip_activation_quant: bool = False,
    ) -> None:
        """
        Initialize BitLinear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            device: TT-NN device
            eps: Epsilon for numerical stability
            skip_activation_quant: Skip activation quantization for speed
        """
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.eps = eps
        self.skip_activation_quant = skip_activation_quant

        # Weights (will be loaded later)
        self.weight: ttnn.Tensor | None = None
        self.weight_scale: float = 1.0
        self.norm_weight: ttnn.Tensor | None = None

    def load_weights(
        self,
        weight: NDArray[np.floating],
        norm_weight: NDArray[np.floating] | None = None,
    ) -> None:
        """
        Load and quantize weights to device.

        Weights are pre-transposed during load to avoid transpose per forward.

        Args:
            weight: Weight array of shape (out_features, in_features)
            norm_weight: Optional RMSNorm weight of shape (in_features,)
        """
        # Quantize weights to ternary
        weight_quant, scale = weight_quant_ternary(weight.astype(np.float32))
        self.weight_scale = float(scale)

        # Convert to float32 for TT-NN (ternary values as floats)
        weight_float = weight_quant.astype(np.float32)

        # Pre-transpose weights: (out, in) -> (in, out) to avoid runtime transpose
        weight_t = weight_float.T.copy()

        # Transfer to device (already transposed)
        self.weight = numpy_to_ttnn(weight_t, self.device)

        # Load norm weight if provided
        if norm_weight is not None:
            norm_weight_2d = norm_weight.reshape(1, 1, 1, -1).astype(np.float32)
            self.norm_weight = numpy_to_ttnn(norm_weight_2d, self.device)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch, seq_len, out_features)
        """
        if self.weight is None:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")

        # 1. RMSNorm
        x_norm = ttnn.rms_norm(x, epsilon=self.eps, weight=self.norm_weight)

        if self.skip_activation_quant:
            # Fast path: skip activation quantization
            weight_scaled = ttnn.multiply(self.weight, self.weight_scale)
            return ttnn.matmul(x_norm, weight_scaled)

        # 2. Activation quantization (simulated in bfloat16)
        # Compute max absolute value per token
        x_abs = ttnn.abs(x_norm)
        x_max = ttnn.max(x_abs, dim=-1, keepdim=True)
        scale_x = ttnn.multiply(ttnn.reciprocal(ttnn.add(x_max, self.eps)), 127.0)

        # Quantize activations
        x_scaled = ttnn.multiply(x_norm, scale_x)
        # Note: ttnn doesn't have round, using floor(x + 0.5) as approximation
        x_quant = ttnn.floor(ttnn.add(x_scaled, 0.5))
        x_quant = ttnn.clip(x_quant, -128.0, 127.0)

        # Dequantize for computation
        x_dequant = ttnn.multiply(x_quant, ttnn.reciprocal(scale_x))

        # 3. Matrix multiplication with pre-transposed weight
        # Weight is already transposed at load time: (in, out)
        weight_scaled = ttnn.multiply(self.weight, self.weight_scale)
        output = ttnn.matmul(x_dequant, weight_scaled)

        return output


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
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: ttnn.Device,
        compute_fidelity: str = "hifi2",
    ) -> None:
        """
        Initialize Linear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            device: TT-NN device
            compute_fidelity: One of "hifi4", "hifi2" (default), "lofi"
        """
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.weight: ttnn.Tensor | None = None
        self._fidelity = compute_fidelity

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

        Applies ternary weight quantization matching HuggingFace BitLinear:
            s = 1.0 / weight.abs().mean()
            result = (weight * s).round().clamp(-1, 1) / s

        Weights are pre-transposed to avoid transpose per forward.

        Args:
            weight: Weight array of shape (out_features, in_features)
        """
        self.load_pretransposed_weight(quantize_and_transpose_weight(weight))

    def load_pretransposed_weight(self, weight_t: NDArray[np.floating]) -> None:
        """
        Load a weight that is already quantized and transposed to (in, out).

        This lets higher-level fused layers concatenate separately quantized
        blocks and still reuse the standard TT matmul path.
        """
        self.weight = numpy_to_ttnn(weight_t.astype(np.float32), self.device)

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

        # Weight is already transposed at load time: (in, out)
        # Use HiFi2 compute kernel if available for ~2x speedup
        if self._compute_kernel_config is not None:
            return ttnn.matmul(
                x,
                self.weight,
                compute_kernel_config=self._compute_kernel_config,
            )
        else:
            return ttnn.matmul(x, self.weight)


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
