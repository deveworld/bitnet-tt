// SPDX-License-Identifier: Apache-2.0
//
// reader_in0.cpp — Standard dataflow reader for activation tiles (input A).
//
// Reads bfloat16 tiles from DRAM in the order expected by the blocked matmul
// compute kernel:
//   for mt in 0..Mt:
//     for nt in 0..Nt:
//       for kt in 0..Kt:
//         read A[mt, kt]
//
// Runtime args:
//   arg 0: src_addr       — DRAM byte address of activation buffer
//   arg 1: Mt             — number of M-dimension tile rows
//   arg 2: Kt             — number of K-dimension tiles
//   arg 3: Nt             — number of N-dimension tile columns (for loop count)
//
// The activation tensor has shape (M_padded, K_padded) stored as row-major tiles.
// Tile index = mt * Kt + kt.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt       = get_arg_val<uint32_t>(1);
    uint32_t Kt       = get_arg_val<uint32_t>(2);
    uint32_t Nt       = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = 0;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                uint32_t tile_id = mt * Kt + kt;

                cb_reserve_back(cb_id_in0, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

                uint64_t noc_addr = get_noc_addr(src_addr + tile_id * tile_bytes);
                noc_async_read(noc_addr, l1_write_addr, tile_bytes);
                noc_async_read_barrier();

                cb_push_back(cb_id_in0, 1);
            }
        }
    }
}
