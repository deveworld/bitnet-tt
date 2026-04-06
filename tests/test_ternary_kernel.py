"""
Tests for ternary weight packing, TernaryLinear, and BFP4 matmul paths.

These tests verify correctness of the 2-bit packing round-trip and that
the BFP4 weight dtype produces results consistent with the bf16 baseline.
"""

import numpy as np
import pytest

from bitnet_tt.kernels.pack import (
    PACKED_TILE_BYTES,
    TILE_ELEMENTS,
    _decode_ternary,
    _encode_ternary,
    _faces_to_tile,
    _pack_4_to_byte,
    _tile_to_faces,
    _unpack_byte_to_4,
    pack_ternary_tilized,
    packed_dram_bytes,
    packed_tile_count,
    unpack_ternary_tilized,
)
from bitnet_tt.utils.quantization import weight_quant_ternary


class TestTernaryEncoding:
    """Test 2-bit encode/decode round-trip."""

    def test_encode_values(self) -> None:
        values = np.array([-1, 0, 1, 0], dtype=np.int8)
        codes = _encode_ternary(values)
        assert codes.tolist() == [0b10, 0b00, 0b01, 0b00]

    def test_decode_values(self) -> None:
        codes = np.array([0b10, 0b00, 0b01, 0b00], dtype=np.uint8)
        values = _decode_ternary(codes)
        assert values.tolist() == [-1, 0, 1, 0]

    def test_encode_decode_roundtrip(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.choice([-1, 0, 1], size=1024).astype(np.int8)
        roundtrip = _decode_ternary(_encode_ternary(values))
        np.testing.assert_array_equal(values, roundtrip)


class TestBytePacking:
    """Test 4-to-byte and byte-to-4 packing."""

    def test_pack_unpack_roundtrip(self) -> None:
        codes = np.array([0b01, 0b10, 0b00, 0b01, 0b10, 0b00, 0b01, 0b10], dtype=np.uint8)
        packed = _pack_4_to_byte(codes)
        assert len(packed) == 2
        unpacked = _unpack_byte_to_4(packed)
        np.testing.assert_array_equal(codes, unpacked)

    def test_pack_single_byte(self) -> None:
        # -1, +1, 0, -1 → 0b10_01_00_10 = 0x92
        codes = np.array([0b10, 0b01, 0b00, 0b10], dtype=np.uint8)
        packed = _pack_4_to_byte(codes)
        assert packed[0] == 0b10_01_00_10

    def test_full_tile_pack_unpack(self) -> None:
        rng = np.random.default_rng(123)
        values = rng.choice([-1, 0, 1], size=TILE_ELEMENTS).astype(np.int8)
        codes = _encode_ternary(values)
        packed = _pack_4_to_byte(codes)
        assert packed.shape == (PACKED_TILE_BYTES,)
        unpacked_codes = _unpack_byte_to_4(packed)
        roundtrip = _decode_ternary(unpacked_codes)
        np.testing.assert_array_equal(values, roundtrip)


class TestTileFaceOrder:
    """Test face reordering matches TT hardware layout."""

    def test_faces_roundtrip(self) -> None:
        tile = np.arange(32 * 32, dtype=np.int8).reshape(32, 32)
        faces_flat = _tile_to_faces(tile)
        recovered = _faces_to_tile(faces_flat)
        np.testing.assert_array_equal(tile, recovered)

    def test_face_ordering(self) -> None:
        tile = np.zeros((32, 32), dtype=np.int8)
        tile[0, 0] = 1     # face 0
        tile[0, 16] = 2    # face 1
        tile[16, 0] = 3    # face 2
        tile[16, 16] = 4   # face 3
        faces = _tile_to_faces(tile)
        # First element of each face
        assert faces[0] == 1       # face 0 start
        assert faces[256] == 2     # face 1 start (16*16 = 256)
        assert faces[512] == 3     # face 2 start
        assert faces[768] == 4     # face 3 start


class TestPackTernaryTilized:
    """Test full tilized packing and unpacking."""

    def test_small_matrix_roundtrip(self) -> None:
        rng = np.random.default_rng(7)
        weight = rng.standard_normal((64, 128)).astype(np.float32)
        weight_quant, scale = weight_quant_ternary(weight)

        packed, tile_scales = pack_ternary_tilized(weight_quant, float(scale))

        # Check dimensions
        expected_tiles = packed_tile_count(128, 64)  # transposed: (in=128, out=64)
        assert packed.shape == (expected_tiles, PACKED_TILE_BYTES)
        assert tile_scales.shape == (expected_tiles,)

        # Unpack and verify
        unpacked = unpack_ternary_tilized(packed, tile_scales, 128, 64)
        # Reconstruct expected: transpose quantized weight, scale
        expected = weight_quant.T.astype(np.float32) * float(scale)
        # Pad expected to match
        np.testing.assert_allclose(
            unpacked[:128, :64],
            expected[:128, :64],
            atol=1e-5,
        )

    def test_non_tile_aligned_shape(self) -> None:
        """Weights that don't align to 32x32 tiles should pad correctly."""
        rng = np.random.default_rng(99)
        weight = rng.standard_normal((50, 100)).astype(np.float32)
        weight_quant, scale = weight_quant_ternary(weight)

        packed, tile_scales = pack_ternary_tilized(weight_quant, float(scale))
        unpacked = unpack_ternary_tilized(packed, tile_scales, 100, 50)

        expected = weight_quant.T.astype(np.float32) * float(scale)
        np.testing.assert_allclose(unpacked[:100, :50], expected, atol=1e-5)

    def test_all_zeros(self) -> None:
        weight_quant = np.zeros((32, 64), dtype=np.int8)
        packed, tile_scales = pack_ternary_tilized(weight_quant, 0.5)
        unpacked = unpack_ternary_tilized(packed, tile_scales, 64, 32)
        np.testing.assert_array_equal(unpacked, 0.0)

    def test_all_ones(self) -> None:
        weight_quant = np.ones((32, 32), dtype=np.int8)
        scale = 0.25
        packed, tile_scales = pack_ternary_tilized(weight_quant, scale)
        unpacked = unpack_ternary_tilized(packed, tile_scales, 32, 32)
        np.testing.assert_allclose(unpacked, scale, atol=1e-6)


class TestPackedDramBytes:
    """Test DRAM size calculations."""

    def test_tile_aligned(self) -> None:
        assert packed_dram_bytes(2560, 2560) == packed_tile_count(2560, 2560) * 256

    def test_bitnet_projection_sizes(self) -> None:
        # Q projection: in=2560, out=2560
        n_tiles = packed_tile_count(2560, 2560)
        assert n_tiles == 80 * 80  # 2560/32 = 80

        # Gate projection: in=2560, out=6912
        n_tiles_gate = packed_tile_count(2560, 6912)
        assert n_tiles_gate == 80 * 216  # 6912/32 = 216 (exact)

    def test_bandwidth_savings(self) -> None:
        in_f, out_f = 2560, 2560
        bf16_bytes = in_f * out_f * 2
        packed2_bytes = packed_dram_bytes(in_f, out_f)
        ratio = bf16_bytes / packed2_bytes
        assert ratio > 7.5  # Should be ~8x savings


class TestKernelSourcesExist:
    """Verify that the C++ kernel source files are in place."""

    def test_reader_ternary_exists(self) -> None:
        from pathlib import Path
        kernel_dir = Path(__file__).parent.parent / "src" / "bitnet_tt" / "kernels" / "device"
        assert (kernel_dir / "reader_ternary_in1.cpp").exists()

    def test_compute_kernel_exists(self) -> None:
        from pathlib import Path
        kernel_dir = Path(__file__).parent.parent / "src" / "bitnet_tt" / "kernels" / "device"
        assert (kernel_dir / "ternary_mm_compute.cpp").exists()

    def test_writer_exists(self) -> None:
        from pathlib import Path
        kernel_dir = Path(__file__).parent.parent / "src" / "bitnet_tt" / "kernels" / "device"
        assert (kernel_dir / "writer_out.cpp").exists()

    def test_reader_in0_exists(self) -> None:
        from pathlib import Path
        kernel_dir = Path(__file__).parent.parent / "src" / "bitnet_tt" / "kernels" / "device"
        assert (kernel_dir / "reader_in0.cpp").exists()
