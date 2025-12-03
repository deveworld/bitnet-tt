"""
BitNet model implementation using TT-NN.

This module provides the complete BitNet model for causal language modeling,
including:
- Token embedding
- Stack of Transformer blocks with KV-Cache support
- Final normalization
- Language model head
"""

from typing import Optional

import numpy as np
import ttnn
from numpy.typing import NDArray

from bitnet_tt.config import BitNetConfig
from bitnet_tt.layers.attention import KVCache
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
        for layer_idx in range(config.num_layers):
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
                    layer_idx=layer_idx,
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
        position_ids: NDArray[np.integer] | None = None,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> tuple[ttnn.Tensor, Optional[list[KVCache]]]:
        """
        Forward pass with optional KV-Cache support.

        Args:
            input_ids: Token indices tensor of shape (batch, seq_len)
            attention_mask: Optional attention mask
            position_ids: Position indices for RoPE (seq_len,) or None
            past_key_values: List of KV-Cache for each layer (from previous forward)
            use_cache: Whether to return updated KV-Cache

        Returns:
            Tuple of (logits tensor, updated KV-Cache list if use_cache else None)
        """
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Initialize cache list if using cache
        if use_cache and past_key_values is None:
            past_key_values = [None] * len(self.layers)

        updated_caches: list[KVCache] = [] if use_cache else None

        # Process through transformer layers
        for layer_idx, layer in enumerate(self.layers):
            past_kv = past_key_values[layer_idx] if past_key_values else None

            hidden_states, updated_cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            if use_cache:
                updated_caches.append(updated_cache)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Language model head
        if self.lm_head_weight is None:
            raise RuntimeError("LM head weights not loaded.")

        logits = ttnn.matmul(hidden_states, ttnn.transpose(self.lm_head_weight, -2, -1))

        return logits, updated_caches if use_cache else None

    def reset_cache(self, past_key_values: list[KVCache]) -> None:
        """
        Reset all KV-Caches.

        Args:
            past_key_values: List of KV-Cache to reset
        """
        for cache in past_key_values:
            if cache is not None:
                cache.reset()


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
