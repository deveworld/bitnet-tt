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
        "packed_ternary": ttnn.uint32,  # 2-bit packed
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
        self._use_packed_ternary = (weight_dtype == "packed_ternary")

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

        For packed_ternary: stores 2-bit packed weight (8× smaller DRAM footprint).

        Args:
            weight: Weight array of shape (out_features, in_features)
        """
        if self._weight_dtype_name == "packed_ternary":
            # True 2-bit packed_ternary with HW unpack. We pack via tt-metal's
            # bfp2 packer, then DROP the shared-exponent section before storing
            # to DRAM. For ternary values {-1, 0, +1} the bfp2 exponent is
            # always 0x7F (= 127, scale 2^0=1) for any non-zero group, and
            # mantissa=0 makes all-zero groups correct regardless of exp.
            # The device-side reader synthesizes the 0x7F exponent block once
            # in cb_in1's L1 region, then DMAs only the 64 uint32 mantissa per
            # tile from DRAM. End-to-end storage is 256 B / 32x32 tile = pure
            # 2 bits per weight, while compute uses the Tensix HW unpacker.
            from ttnn._ttnn.bfp_utils import pack_bfp2

            weight_quant, scale = weight_quant_ternary(weight.astype(np.float32))
            w_kn = weight_quant.T.astype(np.float32)
            K, N = w_kn.shape
            assert K % 32 == 0 and N % 32 == 0, "K/N must be multiples of 32"
            Kt, Nt = K // 32, N // 32
            w_tiled = (
                w_kn.reshape(Kt, 32, Nt, 32)
                .transpose(0, 2, 1, 3)  # (Kt, Nt, 32, 32)
                .reshape(Kt * Nt, 1024)
            )
            flat = w_tiled.reshape(-1).astype(np.float32)
            packed_full = np.asarray(
                pack_bfp2(flat, row_major_input=True, is_exp_a=False)
            )
            num_tiles = Kt * Nt
            # Each bfp2 tile = 80 uint32 (16 exp + 64 mantissa). Drop exponent.
            packed_full = packed_full.reshape(num_tiles, 80)
            mantissa_only = packed_full[:, 16:].copy()  # (num_tiles, 64)
            packed_torch = torch.from_numpy(mantissa_only.astype(np.int32)).to(torch.int32)
            self.weight = ttnn.from_torch(
                packed_torch,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.weight_scale = float(scale)
            self._use_packed_ternary = True
            return

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

        if self._use_packed_ternary:
            # Pre-transposed weight: shape (K, N) with {-1, 0, +1} values.
            # Pack as BFP2_b tiles, drop the constant exponent block.
            from ttnn._ttnn.bfp_utils import pack_bfp2

            w_kn = weight_t.astype(np.float32)
            K, N = w_kn.shape
            assert K % 32 == 0 and N % 32 == 0
            Kt, Nt = K // 32, N // 32
            w_tiled = (
                w_kn.reshape(Kt, 32, Nt, 32)
                .transpose(0, 2, 1, 3)
                .reshape(Kt * Nt, 1024)
            )
            flat = w_tiled.reshape(-1).astype(np.float32)
            packed_full = np.asarray(
                pack_bfp2(flat, row_major_input=True, is_exp_a=False)
            )
            num_tiles = Kt * Nt
            packed_full = packed_full.reshape(num_tiles, 80)
            mantissa_only = packed_full[:, 16:].copy()  # (num_tiles, 64)
            packed_torch = torch.from_numpy(mantissa_only.astype(np.int32)).to(torch.int32)
            self.weight = ttnn.from_torch(
                packed_torch,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return

        torch_tensor = torch.from_numpy(weight_t.astype(np.float32))
        self.weight = ttnn.from_torch(
            torch_tensor,
            dtype=self._ttnn_weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(
        self,
        x: ttnn.Tensor,
        norm_weight: ttnn.Tensor | None = None,
        norm_epsilon: float | None = None,
    ) -> ttnn.Tensor:
        """
        Forward pass with pre-quantized and pre-transposed weights.

        Args:
            x: Input tensor of shape (batch, seq_len, in_features)
            norm_weight: Optional RMSNorm gamma tensor for fused norm+matmul.
                When provided together with norm_epsilon, the ternary matmul
                kernel computes RMSNorm(x) * gamma inline before the matmul,
                eliminating a separate kernel launch.
            norm_epsilon: Optional epsilon for the fused RMSNorm.

        Returns:
            Output tensor of shape (batch, seq_len, out_features)
        """
        if self.weight is None:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")

        if self._use_packed_ternary:
            out = ttnn.experimental.ternary_matmul(
                x, self.weight, use_packed_ternary=True,
                norm_weight=norm_weight,
                norm_epsilon=norm_epsilon,
            )
        elif self._compute_kernel_config is not None:
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

    def __call__(self, x: ttnn.Tensor, memory_config=None) -> ttnn.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            memory_config: Optional output memory config (e.g. L1_MEMORY_CONFIG
                           to keep output in L1 for a downstream matmul).

        Returns:
            Normalized tensor
        """
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight,
                             memory_config=memory_config)
