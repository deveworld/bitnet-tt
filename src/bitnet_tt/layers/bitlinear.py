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
) -> ttnn.Tensor:
    """
    Convert numpy array to ttnn tensor.

    Args:
        array: NumPy array (should be float32)
        device: TT-NN device
        layout: Tensor layout (default: TILE_LAYOUT)

    Returns:
        ttnn.Tensor on device
    """
    # Ensure float32 for compatibility
    if array.dtype != np.float32:
        array = array.astype(np.float32)

    # Convert via torch (ttnn.Tensor doesn't support bfloat16 from numpy)
    torch_tensor = torch.from_numpy(array)
    tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=layout, device=device)
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
    ) -> None:
        """
        Initialize BitLinear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            device: TT-NN device
            eps: Epsilon for numerical stability
        """
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.eps = eps

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
        output = ttnn.matmul(x_dequant, weight_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)

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
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: ttnn.Device,
    ) -> None:
        """
        Initialize Linear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            device: TT-NN device
        """
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.weight: ttnn.Tensor | None = None

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
        weight = weight.astype(np.float32)

        # Pre-compute ternary quantization (matching HF BitLinear weight_quant)
        # s = 1 / mean(|weight|), result = round(weight * s).clamp(-1, 1) / s
        scale = np.abs(weight).mean()
        s = 1.0 / max(scale, 1e-5)
        weight_quant = np.clip(np.round(weight * s), -1, 1) / s

        # Pre-transpose: (out, in) -> (in, out) to avoid runtime transpose
        weight_t = weight_quant.T.copy()

        # Store pre-quantized and pre-transposed weights
        self.weight = numpy_to_ttnn(weight_t, self.device)

    def __call__(self, x: ttnn.Tensor, memory_config: ttnn.MemoryConfig | None = None) -> ttnn.Tensor:
        """
        Forward pass with pre-quantized and pre-transposed weights.

        Args:
            x: Input tensor of shape (batch, seq_len, in_features)
            memory_config: Optional memory config for output (default: L1 for decode perf)

        Returns:
            Output tensor of shape (batch, seq_len, out_features)
        """
        if self.weight is None:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")

        # Weight is already transposed at load time: (in, out)
        # Use L1 by default for faster decode performance
        if memory_config is None:
            memory_config = ttnn.L1_MEMORY_CONFIG

        return ttnn.matmul(x, self.weight, memory_config=memory_config)


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
