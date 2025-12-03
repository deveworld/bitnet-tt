"""
Tests for BitNet model components.

These tests verify the basic functionality without requiring TT-NN hardware.
Hardware-specific tests are marked with skipif.
"""

import numpy as np
import pytest

from bitnet_tt.config import BitNet2B4TConfig, BitNetMiniConfig


class TestConfig:
    """Tests for model configuration."""

    def test_mini_config_defaults(self) -> None:
        """Test mini config default values."""
        config = BitNetMiniConfig()

        assert config.hidden_size == 256
        assert config.num_layers == 4
        assert config.num_attention_heads == 4
        assert config.num_key_value_heads == 2
        assert config.intermediate_size == 512
        assert config.vocab_size == 1000

    def test_full_config_defaults(self) -> None:
        """Test full 2B config default values."""
        config = BitNet2B4TConfig()

        assert config.hidden_size == 2560
        assert config.num_layers == 30
        assert config.num_attention_heads == 20
        assert config.num_key_value_heads == 5
        assert config.intermediate_size == 6912
        assert config.vocab_size == 128256

    def test_config_custom_values(self) -> None:
        """Test config with custom values."""
        config = BitNetMiniConfig(
            hidden_size=512,
            num_layers=8,
        )

        assert config.hidden_size == 512
        assert config.num_layers == 8


class TestRoPE:
    """Tests for Rotary Position Embeddings."""

    def test_precompute_freqs_shape(self) -> None:
        """Test shape of precomputed frequencies."""
        from bitnet_tt.layers.attention import precompute_freqs_cis

        dim = 64
        max_seq_len = 512

        cos, sin = precompute_freqs_cis(dim, max_seq_len)

        assert cos.shape == (max_seq_len, dim // 2)
        assert sin.shape == (max_seq_len, dim // 2)

    def test_precompute_freqs_values(self) -> None:
        """Test that frequencies are in valid range."""
        from bitnet_tt.layers.attention import precompute_freqs_cis

        cos, sin = precompute_freqs_cis(64, 512)

        # Cos and sin should be in [-1, 1]
        assert np.all(cos >= -1) and np.all(cos <= 1)
        assert np.all(sin >= -1) and np.all(sin <= 1)


class TestQuantizationIntegration:
    """Integration tests for quantization."""

    def test_weight_quantization_preserves_sign(self) -> None:
        """Test that sign is preserved in quantization."""
        from bitnet_tt.utils.quantization import weight_quant_ternary

        # Create weight with known sign pattern
        weight = np.array([[1.0, -1.0], [0.1, -0.1]], dtype=np.float32)
        weight_quant, _ = weight_quant_ternary(weight)

        # Check signs are preserved
        assert weight_quant[0, 0] == 1
        assert weight_quant[0, 1] == -1

    def test_activation_quantization_scale(self) -> None:
        """Test that activation scale is computed correctly."""
        from bitnet_tt.utils.quantization import activation_quant_absmax

        # Create activation with known max value
        x = np.zeros((1, 1, 10), dtype=np.float32)
        x[0, 0, 0] = 127.0  # Known max

        x_quant, scale = activation_quant_absmax(x)

        # Scale should map max to 127
        assert np.allclose(scale, 1.0, atol=0.01)


class TestWeightLoading:
    """Tests for weight loading utilities."""

    def test_convert_hf_state_dict(self) -> None:
        """Test HuggingFace state dict conversion."""
        from bitnet_tt.utils.weights import convert_hf_state_dict

        hf_state_dict = {
            "model.embed_tokens.weight": np.zeros((100, 256)),
            "model.layers.0.input_layernorm.weight": np.zeros((256,)),
            "model.norm.weight": np.zeros((256,)),
            "lm_head.weight": np.zeros((100, 256)),
        }

        converted = convert_hf_state_dict(hf_state_dict)

        assert "embed_tokens.weight" in converted
        assert "layers.0.input_layernorm.weight" in converted
        assert "norm.weight" in converted
        assert "lm_head.weight" in converted

    def test_get_layer_weights(self) -> None:
        """Test layer weight extraction."""
        from bitnet_tt.utils.weights import get_layer_weights

        state_dict = {
            "layers.0.input_layernorm.weight": np.zeros((256,)),
            "layers.0.self_attn.q_proj.weight": np.zeros((256, 256)),
            "layers.1.input_layernorm.weight": np.zeros((256,)),
        }

        layer_0_weights = get_layer_weights(state_dict, 0)

        assert "input_layernorm.weight" in layer_0_weights
        assert "self_attn.q_proj.weight" in layer_0_weights
        assert len(layer_0_weights) == 2


@pytest.mark.skipif(
    True,  # Skip by default, enable on hardware
    reason="Requires TT-NN hardware"
)
class TestModelTTNN:
    """Tests for TT-NN model (requires hardware)."""

    def test_model_creation(self) -> None:
        """Test model creation."""
        from bitnet_tt.model.bitnet import create_model
        from bitnet_tt.utils.device import close_device, get_device

        device = get_device()
        try:
            config = BitNetMiniConfig()
            model = create_model(config, device)

            assert model.config == config
            assert len(model.layers) == config.num_layers
        finally:
            close_device()

    def test_embedding_layer(self) -> None:
        """Test embedding layer."""
        from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn, ttnn_to_numpy
        from bitnet_tt.layers.embedding import Embedding
        from bitnet_tt.utils.device import close_device, get_device

        device = get_device()
        try:
            embed = Embedding(1000, 256, device)

            # Load weights
            weight = np.random.randn(1000, 256).astype(np.float32)
            embed.load_weights(weight)

            # Test forward
            input_ids = np.array([[1, 2, 3]], dtype=np.int32)
            input_tensor = numpy_int_to_ttnn(input_ids, device)

            output = embed(input_tensor)
            output_np = ttnn_to_numpy(output)

            assert output_np.shape == (1, 3, 256)
        finally:
            close_device()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
