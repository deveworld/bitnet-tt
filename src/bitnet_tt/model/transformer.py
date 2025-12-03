"""
Transformer block implementation using TT-NN.

This module implements the Transformer decoder block with:
- Pre-normalization (RMSNorm before attention and FFN)
- Residual connections
- BitNet-style attention and FFN with sub-norms
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

    Architecture (matching HuggingFace Transformers BitNet):
        x -> input_layernorm -> Attention (with attn_sub_norm) -> + -> post_attention_layernorm -> FFN (with ffn_sub_norm) -> + -> output
             |__________________________________________|           |___________________________________________________|
                              residual                                                   residual
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

        # Self-attention (includes attn_sub_norm internally)
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

        # Feed-forward network (includes ffn_sub_norm internally)
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
        k_weight: NDArray[np.floating],
        v_weight: NDArray[np.floating],
        o_weight: NDArray[np.floating],
        attn_sub_norm_weight: NDArray[np.floating] | None,
        gate_weight: NDArray[np.floating],
        up_weight: NDArray[np.floating],
        down_weight: NDArray[np.floating],
        ffn_sub_norm_weight: NDArray[np.floating] | None,
    ) -> None:
        """
        Load all weights for the transformer block.

        Args:
            input_layernorm_weight: Pre-attention norm weight
            post_attention_layernorm_weight: Pre-FFN norm weight
            q_weight: Query projection weight
            k_weight: Key projection weight
            v_weight: Value projection weight
            o_weight: Output projection weight
            attn_sub_norm_weight: Attention sub-norm weight (after attn, before o_proj)
            gate_weight: Gate projection weight
            up_weight: Up projection weight
            down_weight: Down projection weight
            ffn_sub_norm_weight: FFN sub-norm weight (after gate*up, before down_proj)
        """
        self.input_layernorm.load_weights(input_layernorm_weight)
        self.post_attention_layernorm.load_weights(post_attention_layernorm_weight)

        self.self_attn.load_weights(
            q_weight=q_weight,
            k_weight=k_weight,
            v_weight=v_weight,
            o_weight=o_weight,
            attn_sub_norm_weight=attn_sub_norm_weight,
        )

        self.mlp.load_weights(
            gate_weight=gate_weight,
            up_weight=up_weight,
            down_weight=down_weight,
            ffn_sub_norm_weight=ffn_sub_norm_weight,
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

        # Self-attention (includes attn_sub_norm before o_proj)
        hidden_states = self.self_attn(hidden_states, attention_mask)

        # Residual connection
        hidden_states = ttnn.add(residual, hidden_states)

        # Pre-FFN norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # FFN (includes ffn_sub_norm before down_proj)
        hidden_states = self.mlp(hidden_states)

        # Residual connection
        hidden_states = ttnn.add(residual, hidden_states)

        return hidden_states
