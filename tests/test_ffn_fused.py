"""
Regression tests for the fused FFN gate+up projection helpers.
"""

import numpy as np

from bitnet_tt.layers.bitlinear import quantize_and_transpose_weight
from bitnet_tt.layers.ffn import build_fused_gate_up_weight


def test_build_fused_gate_up_weight_preserves_separate_quantization() -> None:
    gate_weight = np.array(
        [
            [1.2, -0.8, 0.1],
            [-0.7, 0.4, 0.0],
        ],
        dtype=np.float32,
    )
    up_weight = np.array(
        [
            [0.5, -1.5, 0.25],
            [2.0, -0.1, -0.3],
            [0.0, 0.8, -0.9],
        ],
        dtype=np.float32,
    )

    fused = build_fused_gate_up_weight(gate_weight, up_weight)
    gate_t = quantize_and_transpose_weight(gate_weight)
    up_t = quantize_and_transpose_weight(up_weight)

    assert fused.shape == (gate_weight.shape[1], gate_weight.shape[0] + up_weight.shape[0])
    np.testing.assert_allclose(fused[:, : gate_weight.shape[0]], gate_t)
    np.testing.assert_allclose(fused[:, gate_weight.shape[0] :], up_t)


def test_build_fused_gate_up_weight_matches_feature_axis_layout() -> None:
    hidden_size = 4
    intermediate_size = 6
    gate_weight = np.arange(intermediate_size * hidden_size, dtype=np.float32).reshape(
        intermediate_size, hidden_size
    )
    up_weight = (np.arange(intermediate_size * hidden_size, dtype=np.float32) + 100).reshape(
        intermediate_size, hidden_size
    )

    fused = build_fused_gate_up_weight(gate_weight, up_weight)

    assert fused.shape == (hidden_size, intermediate_size * 2)
