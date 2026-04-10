"""
Optimized Feed-Forward Network implementation using TT-NN.

This module implements the FFN with:
- Gated architecture (gate and up projections)
- Squared ReLU activation (ReLU^2)
- ffn_sub_norm applied after gate*up, before down projection
- Mode-aware memory configs (L1 for decode, DRAM for prefill)

Performance optimizations based on tt_transformers patterns.
"""

from typing import Optional

import numpy as np
import ttnn
from numpy.typing import NDArray

from bitnet_tt.layers.bitlinear import Linear, RMSNorm, quantize_and_transpose_weight


def build_fused_gate_up_weight(
    gate_weight: NDArray[np.floating],
    up_weight: NDArray[np.floating],
) -> tuple[NDArray[np.float32], float, float]:
    """
    Build a single pre-transposed gate+up matrix from separately quantized weights.

    Returns pure ternary {-1,0,+1} fused weight and per-projection scales.
    """
    gate_weight_t, gate_scale = quantize_and_transpose_weight(gate_weight)
    up_weight_t, up_scale = quantize_and_transpose_weight(up_weight)
    fused = np.concatenate([gate_weight_t, up_weight_t], axis=1)
    return fused, gate_scale, up_scale


class FeedForward:
    """
    Optimized TT-NN Feed-Forward Network with gated architecture and squared ReLU.

    Key optimizations:
    - Pre-transposed weights (no transpose per forward)
    - Mode-aware memory configs (L1 for decode, DRAM for prefill)
    - LoFi compute kernel for ~3.6x matmul speedup (configurable)
    - Fused gate+up projection to reduce decode dispatch overhead

    Architecture (matching HuggingFace Transformers):
        gate = SquaredReLU(gate_proj(x))  # Plain linear
        up = up_proj(x)                   # Plain linear
        intermediate = gate * up
        intermediate = ffn_sub_norm(intermediate)  # RMSNorm before down_proj
        output = down_proj(intermediate)  # Plain linear
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        device: ttnn.Device,
        eps: float = 1e-5,
        model_config: Optional[dict] = None,
        use_lofi: bool = False,
        weight_dtype: str = "bfp4",
    ) -> None:
        """
        Initialize FFN.

        Args:
            hidden_size: Input/output dimension
            intermediate_size: Hidden dimension
            device: TT-NN device
            eps: Epsilon for numerical stability
            model_config: Optional model config with memory configs
            use_lofi: Use LoFi compute kernel for ~3.6x speedup (default: False)
            weight_dtype: Weight storage format — "bf16", "bfp4", or "bfp8"
        """
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.device = device
        self.eps = eps
        self.model_config = model_config or {}
        self.use_lofi = use_lofi

        # Compute fidelity: LoFi for speed, HiFi2 for accuracy
        fidelity = "lofi" if use_lofi else "hifi2"

        # Fuse gate+up into one matmul while preserving per-matrix quantization.
        self.gate_up_proj = Linear(
            hidden_size,
            intermediate_size * 2,
            device,
            compute_fidelity=fidelity,
            weight_dtype=weight_dtype,
        )
        self.down_proj = Linear(
            intermediate_size, hidden_size, device,
            compute_fidelity=fidelity,
            weight_dtype=weight_dtype,
        )

        # Sub-norm applied after gate*up, before down_proj
        self.ffn_sub_norm = RMSNorm(intermediate_size, device, eps)

    def load_weights(
        self,
        gate_weight: NDArray[np.floating],
        up_weight: NDArray[np.floating],
        down_weight: NDArray[np.floating],
        ffn_sub_norm_weight: NDArray[np.floating] | None = None,
    ) -> None:
        """
        Load all projection weights.

        Args:
            gate_weight: Gate projection weight
            up_weight: Up projection weight
            down_weight: Down projection weight
            ffn_sub_norm_weight: Sub-norm weight (applied after gate*up, before down_proj)
        """
        fused_gate_up, self._gate_scale, self._up_scale = build_fused_gate_up_weight(
            gate_weight, up_weight
        )
        self.gate_up_proj.load_pretransposed_weight(fused_gate_up)
        self.down_proj.load_weights(down_weight)

        if ffn_sub_norm_weight is not None:
            self.ffn_sub_norm.load_weights(ffn_sub_norm_weight)

    def __call__(self, x: ttnn.Tensor, mode: str = "prefill") -> ttnn.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            mode: "prefill" or "decode" (kept for API compatibility)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        gate_up = self.gate_up_proj(x)  # scale=1.0, no post-matmul scale

        split_start = [0] * len(gate_up.shape)
        gate_end = list(gate_up.shape)
        gate_end[-1] = self.intermediate_size
        gate = ttnn.slice(gate_up, split_start, gate_end)

        up_start = [0] * len(gate_up.shape)
        up_start[-1] = self.intermediate_size
        up_end = list(gate_up.shape)
        up = ttnn.slice(gate_up, up_start, up_end)
        ttnn.deallocate(gate_up)

        # Apply per-projection scales from ternary quantization
        gate = ttnn.multiply(gate, self._gate_scale)
        up = ttnn.multiply(up, self._up_scale)

        # Gate with squared ReLU
        gate = ttnn.relu(gate)
        gate = ttnn.multiply(gate, gate)  # Squared ReLU

        # Element-wise multiplication
        hidden = ttnn.multiply(gate, up)

        # Deallocate intermediates to save memory
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # Apply sub-norm before down projection (BitNet-specific!)
        hidden = self.ffn_sub_norm(hidden)

        # Down projection
        return self.down_proj(hidden)
