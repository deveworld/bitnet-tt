"""
BitNet model configurations.

This module defines configuration classes for different BitNet model sizes,
including a mini test model and the full BitNet b1.58 2B4T model.
"""

from dataclasses import dataclass


@dataclass
class BitNetConfig:
    """Base configuration for BitNet models."""

    # Model architecture
    hidden_size: int = 2560
    num_layers: int = 30
    num_attention_heads: int = 20
    num_key_value_heads: int = 5  # For Grouped Query Attention (GQA)
    intermediate_size: int = 6912
    vocab_size: int = 128256
    max_position_embeddings: int = 4096

    # Normalization
    rms_norm_eps: float = 1e-5

    # RoPE (Rotary Position Embedding)
    rope_theta: float = 500000.0

    # Quantization
    weight_bits: int = 2  # Ternary: {-1, 0, 1} -> ~1.58 bits
    activation_bits: int = 8

    # Activation function
    hidden_act: str = "relu2"  # Squared ReLU (ReLU^2)

    # Misc
    tie_word_embeddings: bool = True
    use_cache: bool = True
    attention_dropout: float = 0.0
    attention_bias: bool = False

    # Token IDs
    bos_token_id: int = 128000
    eos_token_id: int = 128001

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        """Alias for num_key_value_heads."""
        return self.num_key_value_heads


@dataclass
class BitNetMiniConfig(BitNetConfig):
    """
    Mini configuration for testing and development.

    A small model for validating the implementation pipeline
    before scaling up to the full model.
    """

    hidden_size: int = 256
    num_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    intermediate_size: int = 512
    vocab_size: int = 1000
    max_position_embeddings: int = 512


@dataclass
class BitNet2B4TConfig(BitNetConfig):
    """
    Configuration for BitNet b1.58 2B4T model.

    Based on microsoft/bitnet-b1.58-2B-4T from HuggingFace.
    ~2.4B parameters, trained on 4T tokens.

    Aligned with HuggingFace transformers BitNetConfig.
    """

    hidden_size: int = 2560
    num_layers: int = 30
    num_attention_heads: int = 20
    num_key_value_heads: int = 5
    intermediate_size: int = 6912
    vocab_size: int = 128256
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    hidden_act: str = "relu2"
    attention_dropout: float = 0.0
    attention_bias: bool = False
    bos_token_id: int = 128000
    eos_token_id: int = 128001


def get_config(model_name: str = "mini") -> BitNetConfig:
    """
    Get configuration by model name.

    Args:
        model_name: One of "mini", "2b4t", or "base"

    Returns:
        BitNetConfig instance
    """
    configs: dict[str, type[BitNetConfig]] = {
        "mini": BitNetMiniConfig,
        "2b4t": BitNet2B4TConfig,
        "base": BitNetConfig,
    }

    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(configs.keys())}")

    return configs[model_name]()
