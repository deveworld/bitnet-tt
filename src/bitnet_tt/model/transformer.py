"""
Optimized Transformer block implementation using TT-NN.

This module implements the Transformer decoder block with:
- Pre-normalization (RMSNorm before attention and FFN)
- Residual connections
- BitNet-style attention and FFN with sub-norms
- KV-Cache support for efficient generation
- Mode-aware optimization (prefill vs decode)

Performance optimizations based on tt_transformers patterns.
"""

from typing import Optional

import numpy as np
import ttnn
from numpy.typing import NDArray

from bitnet_tt.layers.attention import KVCache, MultiHeadAttention
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
        layer_idx: int = 0,
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
            layer_idx: Layer index for cache management
        """
        self.hidden_size = hidden_size
        self.device = device
        self.layer_idx = layer_idx

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
            layer_idx=layer_idx,
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
        position_ids: NDArray[np.integer] | int | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        mode: str = "prefill",
        rot_mats: list | None = None,
        transformation_mat: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, Optional[KVCache]]:
        """
        Forward pass with mode-aware optimization.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Position indices for RoPE (int for decode mode)
            past_key_value: Optional KV-Cache from previous forward passes
            use_cache: Whether to return updated KV-Cache
            mode: "prefill" or "decode" - affects memory config and RoPE lookup
            rot_mats: [cos, sin] rotation matrices from RotarySetup (for optimized decode)
            transformation_mat: Transformation matrix for rotary_embedding_llama (for optimized decode)

        Returns:
            Tuple of (output tensor, updated KV-Cache if use_cache else None)
        """
        residual = hidden_states

        # Pre-attention norm
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention (includes attn_sub_norm before o_proj)
        hidden_states, updated_cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            mode=mode,
            rot_mats=rot_mats,
            transformation_mat=transformation_mat,
        )

        # Residual connection
        hidden_states = ttnn.add(residual, hidden_states)

        # Pre-FFN norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # FFN (includes ffn_sub_norm before down_proj)
        hidden_states = self.mlp(hidden_states, mode=mode)

        # Residual connection
        hidden_states = ttnn.add(residual, hidden_states)

        return hidden_states, updated_cache
