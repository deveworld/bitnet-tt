"""
BitNet model configurations.

This module defines configuration classes for different BitNet model sizes,
including a mini test model and the full BitNet b1.58 2B4T model.
"""

from dataclasses import dataclass
from typing import Any


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

    def get_model_config(self, device: Any = None) -> dict:
        """
        Get model configuration dictionary with memory configs for TT-NN.

        This follows the pattern from tt_transformers/model_config.py for
        optimized memory layouts in decode/prefill modes.

        Args:
            device: TT-NN device (required for sharded memory configs)

        Returns:
            Dictionary containing memory configs and model parameters
        """
        import ttnn

        config = {}

        # Basic model parameters
        config["hidden_size"] = self.hidden_size
        config["num_layers"] = self.num_layers
        config["num_heads"] = self.num_attention_heads
        config["num_kv_heads"] = self.num_key_value_heads
        config["head_dim"] = self.head_dim
        config["intermediate_size"] = self.intermediate_size

        # Memory configurations
        # For single device (P150a), use simple L1/DRAM configs
        # L1 is ~10x faster than DRAM for intermediate tensors

        # Decode mode: use L1 for fast residual access
        config["DECODE_RESIDUAL_MEMCFG"] = ttnn.L1_MEMORY_CONFIG

        # Prefill mode: use DRAM (larger sequences)
        config["PREFILL_RESIDUAL_MEMCFG"] = ttnn.DRAM_MEMORY_CONFIG

        # Attention intermediates: L1 for decode, DRAM for prefill
        config["ATTN_L1_MEMCFG"] = ttnn.L1_MEMORY_CONFIG
        config["ATTN_DRAM_MEMCFG"] = ttnn.DRAM_MEMORY_CONFIG

        # MLP intermediates
        config["MLP_L1_MEMCFG"] = ttnn.L1_MEMORY_CONFIG
        config["MLP_DRAM_MEMCFG"] = ttnn.DRAM_MEMORY_CONFIG

        # If device is provided, create sharded configs for better performance
        if device is not None:
            try:
                # RoPE transformation matrix config (HEIGHT sharded)
                config["ROPE_TRANS_MAT_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
                    core_grid=ttnn.CoreGrid(y=1, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

                # RoPE cos/sin sharded config
                config["ROPE_COS_SIN_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, self.head_dim),
                    core_grid=ttnn.CoreGrid(y=1, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

            except Exception:
                # Fall back to non-sharded configs
                pass

        return config


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


# =============================================================================
# Performance Optimization: Compute Kernel Configuration
# =============================================================================

def get_compute_kernel_config(fidelity: str = "hifi2") -> "ttnn.WormholeComputeKernelConfig":
    """
    Get compute kernel configuration for matmul operations.

    Args:
        fidelity: One of "hifi4" (BF16, accurate), "hifi2" (BFP8, fast), "lofi" (BFP4, fastest)

    Returns:
        ttnn.WormholeComputeKernelConfig

    Note:
        - HiFi4: ~1x speed, best accuracy (BF16)
        - HiFi2: ~2x speed, good accuracy (BFP8) - recommended for decode
        - LoFi: ~3.6x speed, lower accuracy (BFP4) - for MLP if accuracy allows
    """
    import ttnn

    if fidelity == "hifi4":
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    elif fidelity == "hifi2":
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    elif fidelity == "lofi":
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    else:
        raise ValueError(f"Unknown fidelity: {fidelity}. Choose from hifi4, hifi2, lofi")
