"""
Tests for BitLinear layer and quantization utilities.

These tests verify the NumPy-based quantization functions work correctly.
TT-NN device tests require actual hardware and are marked accordingly.
"""

import numpy as np
import pytest

from bitnet_tt.utils.quantization import (
    activation_quant_absmax,
    compute_weight_scale,
    weight_quant_ternary,
)


class TestWeightQuantization:
    """Tests for weight quantization functions."""

    def test_ternary_values(self) -> None:
        """Test that quantized weights are ternary."""
        weight = np.random.randn(64, 128).astype(np.float32)
        weight_quant, scale = weight_quant_ternary(weight)

        unique_values = np.unique(weight_quant)
        assert len(unique_values) <= 3
        assert all(v in [-1, 0, 1] for v in unique_values.tolist())

    def test_scale_factor(self) -> None:
        """Test that scale factor is computed correctly."""
        weight = np.random.randn(64, 128).astype(np.float32)
        scale = compute_weight_scale(weight)

        expected_scale = np.mean(np.abs(weight))
        assert np.allclose(scale, expected_scale, atol=1e-5)

    def test_reconstruction_quality(self) -> None:
        """Test that dequantized weights are close to original."""
        weight = np.random.randn(64, 128).astype(np.float32)
        weight_quant, scale = weight_quant_ternary(weight)

        # Reconstruct
        weight_recon = weight_quant.astype(np.float32) * scale

        # Should have reasonable correlation
        correlation = np.corrcoef(weight.flatten(), weight_recon.flatten())[0, 1]
        assert correlation > 0.5  # Reasonable correlation

    def test_zero_weights(self) -> None:
        """Test handling of zero weights."""
        weight = np.zeros((64, 128), dtype=np.float32)
        weight_quant, scale = weight_quant_ternary(weight)

        assert np.all(weight_quant == 0)

    def test_large_weights(self) -> None:
        """Test handling of large weight values."""
        weight = np.random.randn(64, 128).astype(np.float32) * 100
        weight_quant, scale = weight_quant_ternary(weight)

        unique_values = np.unique(weight_quant)
        assert all(v in [-1, 0, 1] for v in unique_values.tolist())
        assert scale > 10  # Scale should be large


class TestActivationQuantization:
    """Tests for activation quantization functions."""

    def test_output_range(self) -> None:
        """Test that quantized activations are in valid range."""
        x = np.random.randn(2, 16, 256).astype(np.float32)
        x_quant, scale = activation_quant_absmax(x)

        assert x_quant.min() >= -128
        assert x_quant.max() <= 127

    def test_per_token_scale(self) -> None:
        """Test that scale is computed per-token."""
        batch_size, seq_len, hidden_size = 2, 16, 256
        x = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        x_quant, scale = activation_quant_absmax(x)

        # Scale should have shape (batch, seq, 1)
        assert scale.shape == (batch_size, seq_len, 1)

    def test_reconstruction(self) -> None:
        """Test activation reconstruction."""
        x = np.random.randn(2, 16, 256).astype(np.float32)
        x_quant, scale = activation_quant_absmax(x)

        # Reconstruct
        x_recon = x_quant * scale

        # Should be close (within quantization error)
        max_error = np.max(np.abs(x - x_recon))
        # Max error should be bounded by quantization step
        assert max_error < np.max(np.abs(x)) / 64  # Reasonable bound


class TestConfig:
    """Tests for configuration classes."""

    def test_mini_config(self) -> None:
        """Test mini config creation."""
        from bitnet_tt.config import BitNetMiniConfig

        config = BitNetMiniConfig()
        assert config.hidden_size == 256
        assert config.num_layers == 4
        assert config.num_attention_heads == 4

    def test_full_config(self) -> None:
        """Test full 2B config creation."""
        from bitnet_tt.config import BitNet2B4TConfig

        config = BitNet2B4TConfig()
        assert config.hidden_size == 2560
        assert config.num_layers == 30
        assert config.num_attention_heads == 20


@pytest.mark.skipif(
    True,  # Skip by default, enable on hardware
    reason="Requires TT-NN hardware"
)
class TestBitLinearTTNN:
    """Tests for TT-NN BitLinear layer (requires hardware)."""

    def test_forward_shape(self) -> None:
        """Test that output shape is correct."""
        from bitnet_tt.layers.bitlinear import BitLinear
        from bitnet_tt.utils.device import close_device, get_device

        device = get_device()
        try:
            in_features, out_features = 256, 512
            layer = BitLinear(in_features, out_features, device)

            # Load random weights
            weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
            norm_weight = np.ones(in_features, dtype=np.float32)
            layer.load_weights(weight, norm_weight)

            # Create input
            from bitnet_tt.layers.bitlinear import numpy_to_ttnn, ttnn_to_numpy

            x = np.random.randn(2, 16, in_features).astype(np.float32)
            x_ttnn = numpy_to_ttnn(x, device)

            # Forward
            output = layer(x_ttnn)
            output_np = ttnn_to_numpy(output)

            assert output_np.shape == (2, 16, out_features)
        finally:
            close_device()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
