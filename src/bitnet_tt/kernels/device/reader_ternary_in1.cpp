// SPDX-License-Identifier: Apache-2.0
//
// reader_ternary_in1.cpp — Dataflow reader for 2-bit packed ternary weights.
//
// Reads packed ternary tiles (256 bytes each) from DRAM, unpacks to bfloat16
// tiles (2048 bytes) in L1 circular buffer.  The compute kernel then consumes
// these as ordinary bfloat16 weight tiles in a standard matmul.
//
// Encoding (2 bits per element, MSB-first, 4 elements per byte):
//   0b00 →  0.0   (bfloat16 0x0000)
//   0b01 → +1.0   (bfloat16 0x3F80)
//   0b10 → −1.0   (bfloat16 0xBF80)
//   0b11 →  reserved (treated as 0)
//
// Tile face order: four 16×16 row-major faces per 32×32 tile, matching
// the TT hardware tile layout.
//
// Runtime args:
//   arg 0: packed_weight_addr  — DRAM byte address of packed weight buffer
//   arg 1: num_tiles_K         — number of K-dimension tiles per output column
//   arg 2: num_tiles_N         — number of N-dimension tile columns
//   arg 3: Mt                  — current M-tile row (for reader synchronization)
//   arg 4: start_tile_id       — first tile index to read (for multi-core split)
//
// Compile-time args:
//   arg 0: num_tiles_per_core  — total tiles this core must produce

#include <cstdint>
#include "dataflow_api.h"

// BFloat16 bit patterns
constexpr uint16_t BF16_ZERO     = 0x0000u;
constexpr uint16_t BF16_POS_ONE  = 0x3F80u;
constexpr uint16_t BF16_NEG_ONE  = 0xBF80u;

// Tile geometry
constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;
constexpr uint32_t TILE_ELEMENTS = TILE_H * TILE_W;                  // 1024
constexpr uint32_t PACKED_TILE_BYTES = TILE_ELEMENTS / 4;            // 256
constexpr uint32_t BF16_TILE_BYTES = TILE_ELEMENTS * sizeof(uint16_t);  // 2048

// Lookup table for 2-bit → bfloat16 conversion (indexed by 2-bit code)
constexpr uint16_t TERNARY_LUT[4] = {
    BF16_ZERO,     // 0b00 → 0.0
    BF16_POS_ONE,  // 0b01 → +1.0
    BF16_NEG_ONE,  // 0b10 → −1.0
    BF16_ZERO,     // 0b11 → reserved, treat as 0
};

inline void unpack_byte(uint8_t packed, uint16_t* dst) {
    dst[0] = TERNARY_LUT[(packed >> 6) & 0x03];
    dst[1] = TERNARY_LUT[(packed >> 4) & 0x03];
    dst[2] = TERNARY_LUT[(packed >> 2) & 0x03];
    dst[3] = TERNARY_LUT[packed & 0x03];
}

void kernel_main() {
    // ── Runtime arguments ──────────────────────────────────────────────
    uint32_t packed_weight_addr = get_arg_val<uint32_t>(0);
    uint32_t Kt                = get_arg_val<uint32_t>(1);
    uint32_t Nt                = get_arg_val<uint32_t>(2);
    uint32_t Mt                = get_arg_val<uint32_t>(3);
    uint32_t start_tile_id     = get_arg_val<uint32_t>(4);

    // ── Circular buffer for unpacked bfloat16 weight tiles ─────────────
    // CB index 1 is conventionally used for the second matmul input (B/in1)
    constexpr uint32_t cb_id_in1 = 1;

    // Scratch buffer in L1 for one packed tile (256 bytes).
    // Aligned to 16 bytes for NoC transfer requirements.
    alignas(16) uint8_t packed_scratch[PACKED_TILE_BYTES];

    // ── Main loop: iterate over the same tile order as the compute kernel ──
    //
    // For a standard outer-product matmul the reader streams tiles in order:
    //   for mt in 0..Mt:
    //     for nt in 0..Nt:
    //       for kt in 0..Kt:
    //         read B[kt, nt]   ← this is what we do here
    //
    // The packed buffer stores tiles in row-major order: tile_id = kt * Nt + nt.

    uint32_t tile_id = start_tile_id;

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                // ── 1. Read packed tile from DRAM ──────────────────��───
                uint32_t packed_tile_id = kt * Nt + nt;
                uint32_t src_addr = packed_weight_addr + packed_tile_id * PACKED_TILE_BYTES;

                uint64_t src_noc_addr = get_noc_addr(src_addr);
                noc_async_read(src_noc_addr, reinterpret_cast<uint32_t>(packed_scratch), PACKED_TILE_BYTES);
                noc_async_read_barrier();

                // ── 2. Reserve space in CB and get L1 write pointer ───
                cb_reserve_back(cb_id_in1, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in1);
                auto* dst = reinterpret_cast<uint16_t*>(l1_write_addr);

                // ── 3. Unpack 256 packed bytes → 1024 bfloat16 values ─
                // The packed data is in face order (matching TT tile layout),
                // so we simply unpack sequentially.
                for (uint32_t byte_idx = 0; byte_idx < PACKED_TILE_BYTES; ++byte_idx) {
                    unpack_byte(packed_scratch[byte_idx], &dst[byte_idx * 4]);
                }

                // ── 4. Signal tile ready ──────────────────────────────
                cb_push_back(cb_id_in1, 1);
            }
        }
    }
}
