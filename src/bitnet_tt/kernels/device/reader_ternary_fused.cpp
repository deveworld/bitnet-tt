// SPDX-License-Identifier: Apache-2.0
// reader_ternary_fused.cpp - Fused reader for activation (bf16) and
// packed ternary weights (2-bit). Single RISCV_0 reader kernel.
//
// IMPORTANT: noc_async_read target must be an L1 address, NOT a stack address.
// We use CB2 as a scratch buffer to receive packed data via DMA, then
// unpack from that L1 region into CB1 (the bfloat16 weight CB).
//
// Runtime args:
//   arg0: act_addr     - DRAM address of activation tiles (bf16)
//   arg1: packed_addr  - DRAM address of packed weight buffer (2-bit)
//   arg2: Kt           - K tiles
//   arg3: Nt           - N tiles
//   arg4: Mt           - M tiles

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PACKED_TILE_BYTES = 256;  // 1024 elements * 2 bits / 8

constexpr uint16_t BF16_ZERO    = 0x0000u;
constexpr uint16_t BF16_POS_ONE = 0x3F80u;
constexpr uint16_t BF16_NEG_ONE = 0xBF80u;

static constexpr uint16_t LUT[4] = {
    BF16_ZERO, BF16_POS_ONE, BF16_NEG_ONE, BF16_ZERO,
};

inline void unpack_byte(uint8_t packed, uint16_t* dst) {
    dst[0] = LUT[(packed >> 6) & 0x03];
    dst[1] = LUT[(packed >> 4) & 0x03];
    dst[2] = LUT[(packed >> 2) & 0x03];
    dst[3] = LUT[packed & 0x03];
}

void kernel_main() {
    uint32_t act_addr    = get_arg_val<uint32_t>(0);
    uint32_t packed_addr = get_arg_val<uint32_t>(1);
    uint32_t Kt          = get_arg_val<uint32_t>(2);
    uint32_t Nt          = get_arg_val<uint32_t>(3);
    uint32_t Mt          = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in0    = 0;  // activation (bf16 tiles)
    constexpr uint32_t cb_in1    = 1;  // unpacked weight (bf16 tiles)
    constexpr uint32_t cb_scratch = 2; // scratch for packed data DMA (L1)

    const uint32_t act_tile_bytes = get_tile_size(cb_in0);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                // --- Read activation tile A[mt, kt] from DRAM ---
                uint32_t act_tile_id = mt * Kt + kt;
                cb_reserve_back(cb_in0, 1);
                uint32_t l1_act = get_write_ptr(cb_in0);
                uint64_t act_noc = get_noc_addr(act_addr + act_tile_id * act_tile_bytes);
                noc_async_read(act_noc, l1_act, act_tile_bytes);

                // --- Read packed weight tile B[kt, nt] into L1 scratch ---
                uint32_t w_tile_id = kt * Nt + nt;
                cb_reserve_back(cb_scratch, 1);
                uint32_t l1_scratch = get_write_ptr(cb_scratch);
                uint64_t w_noc = get_noc_addr(packed_addr + w_tile_id * PACKED_TILE_BYTES);
                noc_async_read(w_noc, l1_scratch, PACKED_TILE_BYTES);

                // Wait for both DMA reads to complete
                noc_async_read_barrier();

                // Activation tile ready
                cb_push_back(cb_in0, 1);

                // --- Unpack 256 bytes → 2048 bytes (1024 bf16 values) ---
                cb_reserve_back(cb_in1, 1);
                uint32_t l1_weight = get_write_ptr(cb_in1);
                const uint8_t* src = reinterpret_cast<const uint8_t*>(l1_scratch);
                uint16_t* dst = reinterpret_cast<uint16_t*>(l1_weight);

                for (uint32_t i = 0; i < PACKED_TILE_BYTES; ++i) {
                    unpack_byte(src[i], &dst[i * 4]);
                }

                // Release scratch, push unpacked weight
                cb_pop_front(cb_scratch, 1);
                cb_push_back(cb_in1, 1);
            }
        }
    }
}
