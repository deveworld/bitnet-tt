// SPDX-License-Identifier: Apache-2.0
//
// writer_out.cpp — Standard dataflow writer for matmul output tiles.
//
// Writes bfloat16 output tiles to DRAM in row-major tile order:
//   tile_id = mt * Nt + nt
//
// Runtime args:
//   arg 0: dst_addr  — DRAM byte address of output buffer
//   arg 1: Mt        — number of M-dimension tile rows
//   arg 2: Nt        — number of N-dimension tile columns

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt       = get_arg_val<uint32_t>(1);
    uint32_t Nt       = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out0 = 16;  // tt::CBIndex::c_16
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            uint32_t tile_id = mt * Nt + nt;

            cb_wait_front(cb_id_out0, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

            uint64_t noc_addr = get_noc_addr(dst_addr + tile_id * tile_bytes);
            noc_async_write(l1_read_addr, noc_addr, tile_bytes);
            noc_async_write_barrier();

            cb_pop_front(cb_id_out0, 1);
        }
    }
}
