// SPDX-License-Identifier: Apache-2.0
// reader_ternary_fused.cpp - Fused reader for activation (bf16 TILE) and
// packed ternary weights (2-bit). Single RISCV_0 reader kernel.
//
// Both tensors use TensorAccessor for bank-aware DRAM addressing.
// Packed weight is stored as a bf16 TILE tensor (raw bytes reinterpreted).
//
// Compile-time args:
//   [0..1]: TensorAccessorArgs for activation tensor
//   [2..3]: TensorAccessorArgs for packed weight tensor
//
// Runtime args:
//   arg 0: act_addr     - DRAM address of activation tensor
//   arg 1: packed_addr  - DRAM address of packed weight tensor
//   arg 2: Kt           - K tiles
//   arg 3: Nt           - N tiles
//   arg 4: Mt           - M tiles

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

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

    constexpr auto act_accessor_args = TensorAccessorArgs<0>();
    constexpr auto packed_accessor_args = TensorAccessorArgs<2>();

    constexpr uint32_t cb_in0    = 0;  // activation (bf16 tiles)
    constexpr uint32_t cb_in1    = 1;  // unpacked weight (bf16 tiles)
    constexpr uint32_t cb_scratch = 2; // scratch for packed data (L1)

    const uint32_t act_page_bytes = get_local_cb_interface(cb_in0).fifo_page_size;
    const uint32_t scratch_page_bytes = get_local_cb_interface(cb_scratch).fifo_page_size;

    const auto act_tensor = TensorAccessor(act_accessor_args, act_addr, act_page_bytes);
    // Packed weight stored as bf16 TILE — each "page" holds one bf16 tile
    // but only the first PACKED_TILE_BYTES bytes contain actual packed data.
    const auto packed_tensor = TensorAccessor(packed_accessor_args, packed_addr, scratch_page_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb0(cb_in0);
    experimental::CircularBuffer cb1(cb_in1);

    // Use scratch CB just for its L1 address — we manage it manually.
    // Reserve once to get a stable L1 pointer, then reuse across iterations.
    cb_reserve_back(cb_scratch, 1);
    uint32_t l1_scratch = get_local_cb_interface(cb_scratch).fifo_wr_ptr;

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                // --- Read activation tile A[mt, kt] ---
                uint32_t act_tile_id = mt * Kt + kt;
                cb0.reserve_back(1);
                noc.async_read(act_tensor, cb0, act_page_bytes,
                               {.page_id = act_tile_id}, {.offset_bytes = 0});

                // --- Read packed weight tile B[kt, nt] into scratch L1 ---
                uint32_t w_tile_id = kt * Nt + nt;
                // Use raw noc_async_read to scratch L1 (not through CB protocol)
                uint64_t w_noc = packed_tensor.get_noc_addr(w_tile_id, 0);
                noc_async_read(w_noc, l1_scratch, PACKED_TILE_BYTES);

                // Wait for both DMA reads
                noc.async_read_barrier();
                cb0.push_back(1);

                // --- Unpack packed data → bf16 tile ---
                cb1.reserve_back(1);
                uint32_t l1_weight = get_local_cb_interface(cb_in1).fifo_wr_ptr;
                const uint8_t* src = reinterpret_cast<const uint8_t*>(l1_scratch);
                uint16_t* dst = reinterpret_cast<uint16_t*>(l1_weight);

                for (uint32_t i = 0; i < PACKED_TILE_BYTES; ++i) {
                    unpack_byte(src[i], &dst[i * 4]);
                }

                cb1.push_back(1);
            }
        }
    }
}
