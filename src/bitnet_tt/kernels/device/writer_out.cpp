// SPDX-License-Identifier: Apache-2.0
//
// writer_out.cpp — Standard dataflow writer for matmul output tiles.
//
// Compile-time args:
//   [0]: output CB index
//   [1..2]: TensorAccessorArgs for output tensor
//
// Runtime args:
//   arg 0: dst_addr  — DRAM byte address of output buffer
//   arg 1: Mt        — number of M-dimension tile rows
//   arg 2: Nt        — number of N-dimension tile columns

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt       = get_arg_val<uint32_t>(1);
    uint32_t Nt       = get_arg_val<uint32_t>(2);

    constexpr uint32_t out_cb_idx = get_compile_time_arg_val(0);
    constexpr auto out_accessor_args = TensorAccessorArgs<1>();

    const uint32_t page_bytes = get_local_cb_interface(out_cb_idx).fifo_page_size;
    const auto out_tensor = TensorAccessor(out_accessor_args, dst_addr, page_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb_out(out_cb_idx);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            uint32_t tile_id = mt * Nt + nt;

            cb_out.wait_front(1);
            noc.async_write(cb_out, out_tensor, page_bytes,
                            {}, {.page_id = tile_id});
            noc.async_writes_flushed();
            cb_out.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
