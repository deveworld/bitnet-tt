"""
Transformer block implementation using TT-NN.

This module implements the Transformer decoder block with:
- Pre-normalization (RMSNorm before attention and FFN)
- Residual connections
- BitLinear-based attention and FFN
"""

import numpy as np
import ttnn
from numpy.typing import NDArray

from bitnet_tt.layers.attention import MultiHeadAttention
from bitnet_tt.layers.bitlinear import RMSNorm
from bitnet_tt.layers.ffn import FeedForward


class TransformerBlock:
    """
    TT-NN native Transformer decoder block.

    Architecture:
        x -> RMSNorm -> Attention -> + -> RMSNorm -> FFN -> + -> output
             |__________________|      |_______________|
                  residual                  residual
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        device: ttnn.Device,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        """
        Initialize transformer block.

        Args:
            hidden_size: Hidden dimension
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of KV heads for GQA
            intermediate_size: FFN intermediate dimension
            device: TT-NN device
            max_position_embeddings: Maximum sequence length
            rope_theta: RoPE base frequency
            rms_norm_eps: RMSNorm epsilon
        """
        self.hidden_size = hidden_size
        self.device = device

        # Pre-attention normalization
        self.input_layernorm = RMSNorm(hidden_size, device, rms_norm_eps)

        # Self-attention
        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            device=device,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            eps=rms_norm_eps,
        )

        # Pre-FFN normalization
        self.post_attention_layernorm = RMSNorm(hidden_size, device, rms_norm_eps)

        # Feed-forward network
        self.mlp = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=device,
            eps=rms_norm_eps,
        )

    def load_weights(
        self,
        input_layernorm_weight: NDArray[np.floating],
        post_attention_layernorm_weight: NDArray[np.floating],
        q_weight: NDArray[np.floating],
        q_norm_weight: NDArray[np.floating],
        k_weight: NDArray[np.floating],
        k_norm_weight: NDArray[np.floating],
        v_weight: NDArray[np.floating],
        v_norm_weight: NDArray[np.floating],
        o_weight: NDArray[np.floating],
        o_norm_weight: NDArray[np.floating],
        gate_weight: NDArray[np.floating],
        gate_norm_weight: NDArray[np.floating],
        up_weight: NDArray[np.floating],
        up_norm_weight: NDArray[np.floating],
        down_weight: NDArray[np.floating],
        down_norm_weight: NDArray[np.floating],
    ) -> None:
        """
        Load all weights for the transformer block.

        Args:
            input_layernorm_weight: Pre-attention norm weight
            post_attention_layernorm_weight: Pre-FFN norm weight
            q_weight: Query projection weight
            q_norm_weight: Query norm weight
            k_weight: Key projection weight
            k_norm_weight: Key norm weight
            v_weight: Value projection weight
            v_norm_weight: Value norm weight
            o_weight: Output projection weight
            o_norm_weight: Output norm weight
            gate_weight: Gate projection weight
            gate_norm_weight: Gate norm weight
            up_weight: Up projection weight
            up_norm_weight: Up norm weight
            down_weight: Down projection weight
            down_norm_weight: Down norm weight
        """
        self.input_layernorm.load_weights(input_layernorm_weight)
        self.post_attention_layernorm.load_weights(post_attention_layernorm_weight)

        self.self_attn.load_weights(
            q_weight,
            q_norm_weight,
            k_weight,
            k_norm_weight,
            v_weight,
            v_norm_weight,
            o_weight,
            o_norm_weight,
        )

        self.mlp.load_weights(
            gate_weight,
            gate_norm_weight,
            up_weight,
            up_norm_weight,
            down_weight,
            down_norm_weight,
        )

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        residual = hidden_states

        # Pre-attention norm
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        hidden_states = self.self_attn(hidden_states, attention_mask)

        # Residual connection
        hidden_states = ttnn.add(residual, hidden_states)

        # Pre-FFN norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # FFN
        hidden_states = self.mlp(hidden_states)

        # Residual connection
        hidden_states = ttnn.add(residual, hidden_states)

        return hidden_states
