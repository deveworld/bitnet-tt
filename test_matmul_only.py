#!/usr/bin/env python3
"""Test: reader reads activation → CB0, writes zeros → CB1, compute does matmul."""
import os, numpy as np, torch, ttnn
from bitnet_tt.utils.device import get_device, close_device
KERNEL_DIR = os.path.join(os.path.dirname(__file__), "src", "bitnet_tt", "kernels", "device")

def main():
    device = get_device()
    M, K, N = 32, 32, 32
    BF16_TILE = 2048

    # Activation = all 1.0, Weight = all 1.0 → output = 32.0 everywhere
    act = np.ones((M, K), dtype=np.float32)
    act_tt = ttnn.from_torch(torch.from_numpy(act), dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # Weight stored normally as bf16 tile
    wt = np.ones((K, N), dtype=np.float32)
    wt_tt = ttnn.from_torch(torch.from_numpy(wt), dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([M, N]), ttnn.bfloat16,
        ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)

    core = ttnn.CoreCoord(0, 0)
    cr = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    cb0 = ttnn.CBDescriptor(total_size=BF16_TILE*2, core_ranges=cr,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=BF16_TILE)])
    cb1 = ttnn.CBDescriptor(total_size=BF16_TILE*2, core_ranges=cr,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=BF16_TILE)])
    cb16 = ttnn.CBDescriptor(total_size=BF16_TILE*2, core_ranges=cr,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.bfloat16, page_size=BF16_TILE)])

    # Fused reader: reads act to CB0, weight to CB1 (both bf16 tiles)
    READER = """
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
void kernel_main() {
    uint32_t act_addr = get_arg_val<uint32_t>(0);
    uint32_t wt_addr = get_arg_val<uint32_t>(1);
    constexpr auto act_args = TensorAccessorArgs<0>();
    constexpr auto wt_args = TensorAccessorArgs<2>();
    const uint32_t page = get_local_cb_interface(0).fifo_page_size;
    const auto a = TensorAccessor(act_args, act_addr, page);
    const auto w = TensorAccessor(wt_args, wt_addr, page);
    experimental::Noc noc;
    experimental::CircularBuffer cb0(0);
    experimental::CircularBuffer cb1(1);
    cb0.reserve_back(1);
    noc.async_read(a, cb0, page, {.page_id=0}, {.offset_bytes=0});
    noc.async_read_barrier();
    cb0.push_back(1);
    cb1.reserve_back(1);
    noc.async_read(w, cb1, page, {.page_id=0}, {.offset_bytes=0});
    noc.async_read_barrier();
    cb1.push_back(1);
}
"""
    rct = list(ttnn.TensorAccessorArgs(act_tt).get_compile_time_args()) + \
          list(ttnn.TensorAccessorArgs(wt_tt).get_compile_time_args())
    rrt = ttnn.RuntimeArgs()
    rrt[0][0] = [act_tt.buffer_address(), wt_tt.buffer_address()]
    rk = ttnn.KernelDescriptor(kernel_source=READER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cr, compile_time_args=rct, runtime_args=rrt,
        config=ttnn.ReaderConfigDescriptor())

    # Compute: simple copy from CB0 → CB16 (skip matmul, skip CB1)
    COMPUTE = """
#include "api/compute/tile_move_copy.h"
#ifndef ARCH_QUASAR
#include "experimental/circular_buffer.h"
#endif
void kernel_main() {
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    copy_tile_init(cb_in);
    tile_regs_acquire();
    cb_wait_front(cb_in, 1);
    copy_tile(cb_in, 0, 0);
    cb_pop_front(cb_in, 1);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
}
"""
    ck = ttnn.KernelDescriptor(kernel_source=COMPUTE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cr, compile_time_args=[], runtime_args=[],
        config=ttnn.ComputeConfigDescriptor())

    # Writer
    WRITER = """
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
void kernel_main() {
    uint32_t dst = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    const uint32_t page = get_local_cb_interface(cb_id).fifo_page_size;
    const auto d = TensorAccessor(dst_args, dst, page);
    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id);
    cb.wait_front(1);
    noc.async_write(cb, d, page, {}, {.page_id=0});
    noc.async_writes_flushed();
    cb.pop_front(1);
    noc.async_write_barrier();
}
"""
    wct = [16] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    wrt = ttnn.RuntimeArgs()
    wrt[0][0] = [out.buffer_address()]
    wk = ttnn.KernelDescriptor(kernel_source=WRITER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cr, compile_time_args=wct, runtime_args=wrt,
        config=ttnn.WriterConfigDescriptor())

    prog = ttnn.ProgramDescriptor(kernels=[rk, wk, ck], semaphores=[], cbs=[cb0, cb1, cb16])
    print("Running matmul test...")
    output = ttnn.generic_op([act_tt, wt_tt, out], prog)
    ttnn.synchronize_device(device)
    print("Done.")
    out_np = ttnn.to_torch(output).float().numpy().reshape(M, N)
    print(f"Expected: 32.0 everywhere")
    print(f"Output first 4: {out_np[0,:4]}")
    print(f"Max error: {np.max(np.abs(out_np - 32.0)):.4f}")
    print(f"PASS: {np.max(np.abs(out_np - 32.0)) < 1.0}")
    close_device()

if __name__ == "__main__":
    main()
