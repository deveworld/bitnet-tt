"""
Quantization utilities for BitNet b1.58.

This module implements the quantization schemes used in BitNet using NumPy:
- Weight quantization: Ternary {-1, 0, +1} using absmean
- Activation quantization: 8-bit using absmax per-token

These implementations are used for preprocessing weights before TT-NN inference.
"""

import numpy as np
from numpy.typing import NDArray


def compute_weight_scale(weight: NDArray[np.floating], eps: float = 1e-5) -> np.floating:
    """
    Compute the scale factor (gamma) for weight quantization.

    The scale is the mean absolute value of the weight tensor.

    Args:
        weight: Weight array of shape (out_features, in_features)
        eps: Small epsilon for numerical stability

    Returns:
        Scale factor (scalar)
    """
    scale = np.mean(np.abs(weight))
    return np.maximum(scale, eps)


def weight_quant_ternary(
    weight: NDArray[np.floating],
    eps: float = 1e-5,
) -> tuple[NDArray[np.int8], np.floating]:
    """
    Quantize weights to ternary values {-1, 0, +1} using absmean quantization.

    This is the core quantization method for BitNet b1.58.

    Args:
        weight: Weight array of shape (out_features, in_features)
        eps: Small epsilon for numerical stability

    Returns:
        Tuple of (quantized_weight, scale_factor)
        - quantized_weight: INT8 array with values in {-1, 0, 1}
        - scale_factor: Scale factor for dequantization
    """
    # Compute scale factor (gamma = mean(|W|))
    scale = compute_weight_scale(weight, eps)

    # Scale and round to nearest integer, then clamp to {-1, 0, 1}
    weight_scaled = weight / scale
    weight_quant = np.clip(np.round(weight_scaled), -1, 1).astype(np.int8)

    return weight_quant, scale


def activation_quant_absmax(
    x: NDArray[np.floating],
    num_bits: int = 8,
    eps: float = 1e-5,
) -> tuple[NDArray[np.int8], NDArray[np.floating]]:
    """
    Quantize activations to 8-bit using absmax per-token quantization.

    Args:
        x: Activation array of shape (..., hidden_size)
        num_bits: Number of bits for quantization (default: 8)
        eps: Small epsilon for numerical stability

    Returns:
        Tuple of (quantized_activation, scale_factor)
        - quantized_activation: INT8 array with values in [-128, 127]
        - scale_factor: Per-token scale factors
    """
    # Compute max absolute value per token (last dimension)
    q_b = 2 ** (num_bits - 1) - 1  # 127 for 8-bit

    # Get max absolute value along the last dimension
    x_abs_max = np.max(np.abs(x), axis=-1, keepdims=True)
    x_abs_max = np.maximum(x_abs_max, eps)

    # Compute scale
    scale = q_b / x_abs_max

    # Quantize
    x_quant = np.clip(np.round(x * scale), -q_b - 1, q_b).astype(np.int8)

    return x_quant, scale


def dequantize_weight(
    weight_quant: NDArray[np.int8],
    scale: np.floating,
) -> NDArray[np.floating]:
    """
    Dequantize weight array.

    Args:
        weight_quant: Quantized weight array with values in {-1, 0, 1}
        scale: Scale factor used during quantization

    Returns:
        Dequantized array in float32
    """
    return weight_quant.astype(np.float32) * float(scale)


def pack_ternary_weights(weight_quant: NDArray[np.int8]) -> NDArray[np.uint8]:
    """
    Pack ternary weights into 2-bit representation.

    Maps: -1 -> 0b00, 0 -> 0b01, +1 -> 0b10

    Args:
        weight_quant: INT8 array with values in {-1, 0, 1}

    Returns:
        Packed array with 4 weights per byte
    """
    # Shift values from {-1, 0, 1} to {0, 1, 2}
    weight_shifted = weight_quant.astype(np.int32) + 1

    # Flatten and pad to multiple of 4
    flat = weight_shifted.flatten()
    pad_len = (4 - len(flat) % 4) % 4
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), constant_values=1)  # Pad with zeros (value 1 after shift)

    # Pack 4 values per byte
    flat = flat.reshape(-1, 4)
    packed = (
        (flat[:, 0] << 6) | (flat[:, 1] << 4) | (flat[:, 2] << 2) | flat[:, 3]
    ).astype(np.uint8)

    return packed


def unpack_ternary_weights(
    packed: NDArray[np.uint8],
    original_shape: tuple[int, ...],
) -> NDArray[np.int8]:
    """
    Unpack 2-bit ternary weights back to INT8.

    Args:
        packed: Packed array with 4 weights per byte
        original_shape: Original shape of the weight array

    Returns:
        INT8 array with values in {-1, 0, 1}
    """
    # Extract 4 values per byte
    unpacked = np.stack(
        [
            (packed >> 6) & 0x03,
            (packed >> 4) & 0x03,
            (packed >> 2) & 0x03,
            packed & 0x03,
        ],
        axis=-1,
    ).flatten()

    # Shift back from {0, 1, 2} to {-1, 0, 1}
    unpacked = unpacked.astype(np.int8) - 1

    # Reshape to original shape
    total_elements = 1
    for dim in original_shape:
        total_elements *= dim

    return unpacked[:total_elements].reshape(original_shape)
