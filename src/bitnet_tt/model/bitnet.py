"""
BitNet model implementation using TT-NN.

This module provides the complete BitNet model for causal language modeling,
including:
- Token embedding
- Stack of Transformer blocks
- Final normalization
- Language model head
"""

import numpy as np
import ttnn
from numpy.typing import NDArray

from bitnet_tt.config import BitNetConfig
from bitnet_tt.layers.bitlinear import RMSNorm, numpy_to_ttnn
from bitnet_tt.layers.embedding import Embedding
from bitnet_tt.model.transformer import TransformerBlock


class BitNetModel:
    """
    TT-NN native BitNet Language Model.

    A decoder-only transformer model using BitLinear layers
    with ternary weights and 8-bit activations.
    """

    def __init__(
        self,
        config: BitNetConfig,
        device: ttnn.Device,
    ) -> None:
        """
        Initialize BitNet model.

        Args:
            config: Model configuration
            device: TT-NN device
        """
        self.config = config
        self.device = device

        # Token embedding (not quantized)
        self.embed_tokens = Embedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.hidden_size,
            device=device,
        )

        # Transformer layers
        self.layers: list[TransformerBlock] = []
        for _ in range(config.num_layers):
            self.layers.append(
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    intermediate_size=config.intermediate_size,
                    device=device,
                    max_position_embeddings=config.max_position_embeddings,
                    rope_theta=config.rope_theta,
                    rms_norm_eps=config.rms_norm_eps,
                )
            )

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, device, config.rms_norm_eps)

        # Language model head weight (will be loaded)
        self.lm_head_weight: ttnn.Tensor | None = None

    def load_embedding_weights(self, weight: NDArray[np.floating]) -> None:
        """
        Load embedding weights.

        Args:
            weight: Embedding weight array of shape (vocab_size, hidden_size)
        """
        self.embed_tokens.load_weights(weight)

    def load_layer_weights(self, layer_idx: int, weights: dict[str, NDArray[np.floating]]) -> None:
        """
        Load weights for a specific transformer layer.

        Args:
            layer_idx: Layer index
            weights: Dictionary of weight arrays
        """
        layer = self.layers[layer_idx]

        # BitNet b1.58 uses sub-norms:
        # - attn_sub_norm (hidden_size=2560): applied after attention output, before o_proj
        # - ffn_sub_norm (intermediate_size=6912): applied after gate*up, before down_proj
        attn_sub_norm = weights.get("self_attn.attn_sub_norm.weight")
        ffn_sub_norm = weights.get("mlp.ffn_sub_norm.weight")

        layer.load_weights(
            input_layernorm_weight=weights["input_layernorm.weight"],
            post_attention_layernorm_weight=weights["post_attention_layernorm.weight"],
            q_weight=weights["self_attn.q_proj.weight"],
            k_weight=weights["self_attn.k_proj.weight"],
            v_weight=weights["self_attn.v_proj.weight"],
            o_weight=weights["self_attn.o_proj.weight"],
            attn_sub_norm_weight=attn_sub_norm,
            gate_weight=weights["mlp.gate_proj.weight"],
            up_weight=weights["mlp.up_proj.weight"],
            down_weight=weights["mlp.down_proj.weight"],
            ffn_sub_norm_weight=ffn_sub_norm,
        )

    def load_final_weights(
        self,
        norm_weight: NDArray[np.floating],
        lm_head_weight: NDArray[np.floating],
    ) -> None:
        """
        Load final norm and LM head weights.

        Args:
            norm_weight: Final norm weight
            lm_head_weight: LM head weight
        """
        self.norm.load_weights(norm_weight)
        self.lm_head_weight = numpy_to_ttnn(lm_head_weight.astype(np.float32), self.device)

    def __call__(
        self,
        input_ids: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token indices tensor of shape (batch, seq_len)
            attention_mask: Optional attention mask

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size)
        """
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Language model head
        if self.lm_head_weight is None:
            raise RuntimeError("LM head weights not loaded.")

        logits = ttnn.matmul(hidden_states, ttnn.transpose(self.lm_head_weight, -2, -1))

        return logits


def create_model(
    config: BitNetConfig,
    device: ttnn.Device,
) -> BitNetModel:
    """
    Create a BitNet model.

    Args:
        config: Model configuration
        device: TT-NN device

    Returns:
        BitNetModel instance
    """
    return BitNetModel(config, device)
