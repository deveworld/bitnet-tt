"""
Feed-Forward Network implementation using TT-NN.

This module implements the FFN with:
- Gated architecture (gate and up projections)
- Squared ReLU activation (ReLU^2)
- ffn_sub_norm applied after gate*up, before down projection
"""

import numpy as np
import ttnn
from numpy.typing import NDArray

from bitnet_tt.layers.bitlinear import Linear, RMSNorm


class FeedForward:
    """
    TT-NN native Feed-Forward Network with gated architecture and squared ReLU.

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

        # Projections (plain linear, no input norm)
        self.gate_proj = Linear(hidden_size, intermediate_size, device)
        self.up_proj = Linear(hidden_size, intermediate_size, device)
        self.down_proj = Linear(intermediate_size, hidden_size, device)

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
        self.gate_proj.load_weights(gate_weight)
        self.up_proj.load_weights(up_weight)
        self.down_proj.load_weights(down_weight)

        if ffn_sub_norm_weight is not None:
            self.ffn_sub_norm.load_weights(ffn_sub_norm_weight)

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

        # Apply sub-norm before down projection (key difference from standard FFN!)
        hidden = self.ffn_sub_norm(hidden)

        # Down projection
        return self.down_proj(hidden)
