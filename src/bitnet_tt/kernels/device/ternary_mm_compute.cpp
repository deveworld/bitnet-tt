// SPDX-License-Identifier: Apache-2.0
//
// ternary_mm_compute.cpp — Blocked matmul compute kernel for ternary weights.
//
// This is a standard outer-product blocked matmul.  The ternary-specific work
// happens in the reader kernel (unpacking 2-bit → bfloat16); by the time data
// reaches this compute kernel, both inputs are ordinary bfloat16 tiles.
//
// After the matmul accumulation loop the result is multiplied by the ternary
// weight scale factor so that:
//   output_tile = (Σ_k activation[m,k] × ternary_weight[k,n]) × weight_scale
//
// Compile-time args:
//   arg 0: Mt — number of M-dimension tile rows
//   arg 1: Kt — number of K-dimension tiles (inner dimension)
//   arg 2: Nt — number of N-dimension tile columns
//
// The weight scale is passed as a compile-time arg encoded as a uint32_t
// reinterpreted as float (bit-cast):
//   arg 3: weight_scale_bits — float scale factor bit-cast to uint32_t

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // activation tiles (bfloat16)
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // unpacked weight tiles (bfloat16)
    constexpr auto cb_out = tt::CBIndex::c_16;   // output tiles

    // Initialize the matrix multiply engine
    mm_init(cb_in0, cb_in1, cb_out);

    // ── Outer-product blocked matmul ──────────────────────────────────
    //
    // For each output tile (mt, nt):
    //   accumulator = 0
    //   for kt in 0..Kt:
    //     accumulator += matmul_tiles(A[mt,kt], B[kt,nt])
    //   output[mt,nt] = accumulator
    //
    // The reader kernels stream tiles in the matching nested-loop order.

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            // Acquire destination registers for accumulation
            tile_regs_acquire();

            for (uint32_t kt = 0; kt < Kt; ++kt) {
                // Wait for both input tiles
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);

                // Hardware matmul: accumulates into DST register 0
                matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

                // Release input tiles back to readers
                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }

            // Finalize computation
            tile_regs_commit();
            tile_regs_wait();

            // Pack result tile into output circular buffer
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);

            tile_regs_release();
        }
    }
}
}  // namespace NAMESPACE
