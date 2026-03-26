"""
Regression tests for fused projection host-side helpers.
"""

import numpy as np

from bitnet_tt.layers.attention import build_fused_qkv_weight
from bitnet_tt.layers.bitlinear import quantize_and_transpose_weight


def test_build_fused_qkv_weight_preserves_separate_quantization() -> None:
    q_weight = np.array(
        [
            [1.0, -0.5, 0.25],
            [0.0, 2.0, -1.0],
        ],
        dtype=np.float32,
    )
    k_weight = np.array(
        [
            [-1.25, 0.1, 0.7],
        ],
        dtype=np.float32,
    )
    v_weight = np.array(
        [
            [0.6, -0.2, 0.4],
        ],
        dtype=np.float32,
    )

    fused = build_fused_qkv_weight(q_weight, k_weight, v_weight)
    q_t = quantize_and_transpose_weight(q_weight)
    k_t = quantize_and_transpose_weight(k_weight)
    v_t = quantize_and_transpose_weight(v_weight)

    assert fused.shape == (q_weight.shape[1], q_weight.shape[0] + k_weight.shape[0] + v_weight.shape[0])
    np.testing.assert_allclose(fused[:, : q_weight.shape[0]], q_t)
    np.testing.assert_allclose(
        fused[:, q_weight.shape[0] : q_weight.shape[0] + k_weight.shape[0]],
        k_t,
    )
    np.testing.assert_allclose(fused[:, q_weight.shape[0] + k_weight.shape[0] :], v_t)
