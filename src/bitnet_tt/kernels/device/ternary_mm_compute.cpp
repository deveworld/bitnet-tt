// SPDX-License-Identifier: Apache-2.0
//
// ternary_mm_compute.cpp — Blocked matmul compute kernel for ternary weights.
//
// By the time data reaches this kernel, both inputs are ordinary bfloat16 tiles
// (the reader kernel handles 2-bit → bfloat16 unpacking).
//
// Compile-time args:
//   arg 0: Mt — number of M-dimension tile rows
//   arg 1: Kt — number of K-dimension tiles (inner dimension)
//   arg 2: Nt — number of N-dimension tile columns

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"

#ifndef ARCH_QUASAR
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // activation tiles (bfloat16)
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // unpacked weight tiles (bfloat16)
    constexpr auto cb_out = tt::CBIndex::c_16;   // output tiles

    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();

            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);

                matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);

            tile_regs_release();
        }
    }
}
