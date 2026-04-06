"""
2-bit ternary weight packing for TT-NN tile format.

Encoding: -1 -> 0b10, 0 -> 0b00, +1 -> 0b01  (2 bits per weight)
Packed MSB-first: 4 weights per byte.

A standard bfloat16 tile is 32x32 = 1024 elements = 2048 bytes.
A packed ternary tile is 1024 elements * 2 bits = 256 bytes (8x smaller).

Tile face order matches TT hardware: four 16x16 faces stored row-major.
Face layout within a 32x32 tile:
  Face 0: rows[ 0:16], cols[ 0:16]
  Face 1: rows[ 0:16], cols[16:32]
  Face 2: rows[16:32], cols[ 0:16]
  Face 3: rows[16:32], cols[16:32]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

TILE_H = 32
TILE_W = 32
TILE_ELEMENTS = TILE_H * TILE_W
FACE_H = 16
FACE_W = 16
FACE_ELEMENTS = FACE_H * FACE_W
PACKED_TILE_BYTES = TILE_ELEMENTS // 4  # 256 bytes

# 2-bit encoding
_ENC_NEG = np.uint8(0b10)  # -1
_ENC_ZERO = np.uint8(0b00)  # 0
_ENC_POS = np.uint8(0b01)  # +1


def _encode_ternary(values: NDArray[np.int8]) -> NDArray[np.uint8]:
    """Encode int8 ternary values {-1,0,+1} to 2-bit codes {0b10, 0b00, 0b01}."""
    codes = np.zeros_like(values, dtype=np.uint8)
    codes[values == 1] = _ENC_POS
    codes[values == -1] = _ENC_NEG
    return codes


def _decode_ternary(codes: NDArray[np.uint8]) -> NDArray[np.int8]:
    """Decode 2-bit codes back to int8 ternary values."""
    values = np.zeros_like(codes, dtype=np.int8)
    values[codes == _ENC_POS] = 1
    values[codes == _ENC_NEG] = -1
    return values


def _pack_4_to_byte(codes: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Pack groups of 4 x 2-bit codes into single bytes, MSB first."""
    assert codes.size % 4 == 0
    flat = codes.reshape(-1, 4)
    packed = (
        (flat[:, 0].astype(np.uint8) << 6)
        | (flat[:, 1].astype(np.uint8) << 4)
        | (flat[:, 2].astype(np.uint8) << 2)
        | flat[:, 3].astype(np.uint8)
    )
    return packed


def _unpack_byte_to_4(packed: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Unpack bytes into groups of 4 x 2-bit codes."""
    return np.stack(
        [
            (packed >> 6) & 0x03,
            (packed >> 4) & 0x03,
            (packed >> 2) & 0x03,
            packed & 0x03,
        ],
        axis=-1,
    ).reshape(-1)


def _tile_to_faces(tile: NDArray) -> NDArray:
    """Reorder a 32x32 tile into 4 row-major 16x16 faces (TT hardware order)."""
    assert tile.shape == (TILE_H, TILE_W)
    f0 = tile[:FACE_H, :FACE_W].ravel()
    f1 = tile[:FACE_H, FACE_W:].ravel()
    f2 = tile[FACE_H:, :FACE_W].ravel()
    f3 = tile[FACE_H:, FACE_W:].ravel()
    return np.concatenate([f0, f1, f2, f3])


def _faces_to_tile(faces_flat: NDArray) -> NDArray:
    """Reconstruct a 32x32 tile from 4 concatenated row-major 16x16 faces."""
    assert faces_flat.size == TILE_ELEMENTS
    f0 = faces_flat[0 * FACE_ELEMENTS : 1 * FACE_ELEMENTS].reshape(FACE_H, FACE_W)
    f1 = faces_flat[1 * FACE_ELEMENTS : 2 * FACE_ELEMENTS].reshape(FACE_H, FACE_W)
    f2 = faces_flat[2 * FACE_ELEMENTS : 3 * FACE_ELEMENTS].reshape(FACE_H, FACE_W)
    f3 = faces_flat[3 * FACE_ELEMENTS : 4 * FACE_ELEMENTS].reshape(FACE_H, FACE_W)
    top = np.concatenate([f0, f1], axis=1)
    bot = np.concatenate([f2, f3], axis=1)
    return np.concatenate([top, bot], axis=0)


def pack_ternary_tilized(
    weight_quant: NDArray[np.int8],
    scale: float,
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """
    Pack a quantized ternary weight matrix into 2-bit tilized format.

    Args:
        weight_quant: int8 array with values in {-1, 0, 1}, shape (out_features, in_features).
                      This is the RAW quantized weight before transposing.
        scale: Weight scale factor (mean |W|).

    Returns:
        packed_bytes: uint8 array, shape (num_tiles, PACKED_TILE_BYTES).
                      Tiles are stored in row-major order over the TRANSPOSED
                      weight (in_features, out_features), ready for matmul.
        tile_scales:  float32 array, shape (num_tiles,) — per-tile scale.
                      Currently uniform, but the format supports per-tile scales
                      for future block-quantization schemes.

    The weight is transposed to (in_features, out_features) before tiling,
    matching the pre-transposed layout used by Linear.load_weights().
    """
    out_f, in_f = weight_quant.shape

    # Transpose to (in_features, out_features) for matmul: x @ W_t
    wt = weight_quant.T.copy()  # (in_f, out_f)

    # Pad to tile boundaries
    pad_h = (TILE_H - wt.shape[0] % TILE_H) % TILE_H
    pad_w = (TILE_W - wt.shape[1] % TILE_W) % TILE_W
    if pad_h or pad_w:
        wt = np.pad(wt, ((0, pad_h), (0, pad_w)), constant_values=0)

    rows_t, cols_t = wt.shape
    n_tile_rows = rows_t // TILE_H
    n_tile_cols = cols_t // TILE_W
    num_tiles = n_tile_rows * n_tile_cols

    packed_bytes = np.empty((num_tiles, PACKED_TILE_BYTES), dtype=np.uint8)
    tile_scales = np.full(num_tiles, scale, dtype=np.float32)

    tile_idx = 0
    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            tile = wt[
                tr * TILE_H : (tr + 1) * TILE_H,
                tc * TILE_W : (tc + 1) * TILE_W,
            ]
            face_ordered = _tile_to_faces(tile)
            codes = _encode_ternary(face_ordered)
            packed_bytes[tile_idx] = _pack_4_to_byte(codes)
            tile_idx += 1

    return packed_bytes, tile_scales


def unpack_ternary_tilized(
    packed_bytes: NDArray[np.uint8],
    tile_scales: NDArray[np.float32],
    in_features: int,
    out_features: int,
) -> NDArray[np.float32]:
    """
    Unpack 2-bit tilized ternary weights back to a float32 (in_features, out_features) matrix.

    This is the inverse of pack_ternary_tilized. Used for verification and
    for creating the bfloat16/BFP4 device tensor from packed data.
    """
    # Padded dimensions
    rows_t = ((in_features + TILE_H - 1) // TILE_H) * TILE_H
    cols_t = ((out_features + TILE_W - 1) // TILE_W) * TILE_W
    n_tile_rows = rows_t // TILE_H
    n_tile_cols = cols_t // TILE_W

    wt = np.zeros((rows_t, cols_t), dtype=np.float32)

    tile_idx = 0
    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            codes = _unpack_byte_to_4(packed_bytes[tile_idx])
            values = _decode_ternary(codes)
            tile = _faces_to_tile(values).astype(np.float32)
            tile *= tile_scales[tile_idx]
            wt[
                tr * TILE_H : (tr + 1) * TILE_H,
                tc * TILE_W : (tc + 1) * TILE_W,
            ] = tile
            tile_idx += 1

    return wt[:in_features, :out_features]


def packed_tile_count(in_features: int, out_features: int) -> int:
    """Return the number of tiles for a weight matrix of the given shape."""
    rows_t = ((in_features + TILE_H - 1) // TILE_H) * TILE_H
    cols_t = ((out_features + TILE_W - 1) // TILE_W) * TILE_W
    return (rows_t // TILE_H) * (cols_t // TILE_W)


def packed_dram_bytes(in_features: int, out_features: int) -> int:
    """Total DRAM bytes for 2-bit packed weights (excluding scale array)."""
    return packed_tile_count(in_features, out_features) * PACKED_TILE_BYTES
