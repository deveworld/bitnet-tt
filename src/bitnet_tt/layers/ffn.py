"""
Feed-Forward Network implementation using TT-NN.

This module implements the FFN with:
- Gated architecture (gate and up projections)
- Squared ReLU activation (ReLU^2)
- BitLinear layers for all projections
"""

import numpy as np
import ttnn
from numpy.typing import NDArray

from bitnet_tt.layers.bitlinear import BitLinear


class FeedForward:
    """
    TT-NN native Feed-Forward Network with gated architecture and squared ReLU.

    Architecture:
        gate = SquaredReLU(gate_proj(x))
        up = up_proj(x)
        output = down_proj(gate * up)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        device: ttnn.Device,
        eps: float = 1e-5,
    ) -> None:
        """
        Initialize FFN.

        Args:
            hidden_size: Input/output dimension
            intermediate_size: Hidden dimension
            device: TT-NN device
            eps: Epsilon for numerical stability
        """
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.device = device
        self.eps = eps

        # Projections
        self.gate_proj = BitLinear(hidden_size, intermediate_size, device, eps)
        self.up_proj = BitLinear(hidden_size, intermediate_size, device, eps)
        self.down_proj = BitLinear(intermediate_size, hidden_size, device, eps)

    def load_weights(
        self,
        gate_weight: NDArray[np.floating],
        gate_norm_weight: NDArray[np.floating],
        up_weight: NDArray[np.floating],
        up_norm_weight: NDArray[np.floating],
        down_weight: NDArray[np.floating],
        down_norm_weight: NDArray[np.floating],
    ) -> None:
        """
        Load all projection weights.

        Args:
            gate_weight: Gate projection weight
            gate_norm_weight: Gate projection norm weight
            up_weight: Up projection weight
            up_norm_weight: Up projection norm weight
            down_weight: Down projection weight
            down_norm_weight: Down projection norm weight
        """
        self.gate_proj.load_weights(gate_weight, gate_norm_weight)
        self.up_proj.load_weights(up_weight, up_norm_weight)
        self.down_proj.load_weights(down_weight, down_norm_weight)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        # Gate with squared ReLU
        gate = self.gate_proj(x)
        gate = ttnn.relu(gate)
        gate = ttnn.multiply(gate, gate)  # Squared ReLU

        # Up projection
        up = self.up_proj(x)

        # Element-wise multiplication
        hidden = ttnn.multiply(gate, up)

        # Down projection
        return self.down_proj(hidden)
