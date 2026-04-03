import numpy as np
import pytest

from bitnet_tt.utils.ternary_analysis import (
    all_zero_tile_fraction,
    packed_ratio_vs_fp32,
    summarize_weight,
    ternary_distribution,
)


def test_ternary_distribution_counts_values() -> None:
    weight_quant = np.array(
        [
            [-1, 0, 1, 0],
            [1, 1, 0, -1],
        ],
        dtype=np.int8,
    )
    stats = ternary_distribution(weight_quant)
    assert stats["zero_frac"] == pytest.approx(3 / 8)
    assert stats["pos_frac"] == pytest.approx(3 / 8)
    assert stats["neg_frac"] == pytest.approx(2 / 8)


def test_packed_ratio_matches_two_bit_storage() -> None:
    weight_quant = np.array(
        [
            [-1, 0, 1, 0],
            [1, 1, 0, -1],
        ],
        dtype=np.int8,
    )
    assert packed_ratio_vs_fp32(weight_quant) == pytest.approx(1.0 / 16.0)


def test_all_zero_tile_fraction_uses_padded_tiles() -> None:
    weight_quant = np.array(
        [
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.int8,
    )
    assert all_zero_tile_fraction(weight_quant, tile_size=2) == pytest.approx(0.75)


def test_all_zero_tile_fraction_rejects_non_matrices() -> None:
    with pytest.raises(ValueError):
        all_zero_tile_fraction(np.zeros((2, 2, 2), dtype=np.int8), tile_size=2)


def test_summarize_weight_quantizes_and_reports_tile_zero_rates() -> None:
    weight = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    summary = summarize_weight(weight, tile_sizes=(2, 4))
    assert summary["elements"] == 16
    assert summary["zero_frac"] >= 0.9
    assert summary["tile2_all_zero_frac"] == pytest.approx(0.75)
    assert summary["tile4_all_zero_frac"] == pytest.approx(0.0)
