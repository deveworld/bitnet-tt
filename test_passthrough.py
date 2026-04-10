#!/usr/bin/env python3
"""Minimal pass-through: reader reads 1 tile → CB0 → writer writes to output.
No compute, no packed weight. Tests TensorAccessor + CB pipeline."""
import os, numpy as np, torch, ttnn
from bitnet_tt.utils.device import get_device, close_device

KERNEL_DIR = os.path.join(os.path.dirname(__file__), "src", "bitnet_tt", "kernels", "device")

def main():
    device = get_device()
    M, N = 32, 32
    BF16_TILE = 2048
    CB_SIZE = BF16_TILE * 2

    data = np.arange(M * N, dtype=np.float32).reshape(M, N) * 0.01
    inp = ttnn.from_torch(torch.from_numpy(data), dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=device,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([M, N]), ttnn.bfloat16,
                                         ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)

    core = ttnn.CoreCoord(0, 0)
    cr = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    cb0 = ttnn.CBDescriptor(total_size=CB_SIZE, core_ranges=cr,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0,
            data_format=ttnn.bfloat16, page_size=BF16_TILE)])

    # Simple reader: read 1 tile from input tensor
    READER_SRC = """
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_id = 0;
    const uint32_t page_bytes = get_local_cb_interface(cb_id).fifo_page_size;
    const auto s = TensorAccessor(src_args, src_addr, page_bytes);
    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id);
    cb.reserve_back(1);
    noc.async_read(s, cb, page_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb.push_back(1);
}
"""
    reader_ct = list(ttnn.TensorAccessorArgs(inp).get_compile_time_args())
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [inp.buffer_address()]

    reader_k = ttnn.KernelDescriptor(
        kernel_source=READER_SRC,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cr, compile_time_args=reader_ct,
        runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor())

    # Simple writer: write 1 tile to output tensor
    WRITER_SRC = """
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    const uint32_t page_bytes = get_local_cb_interface(cb_id).fifo_page_size;
    const auto d = TensorAccessor(dst_args, dst_addr, page_bytes);
    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id);
    cb.wait_front(1);
    noc.async_write(cb, d, page_bytes, {}, {.page_id = 0});
    noc.async_writes_flushed();
    cb.pop_front(1);
    noc.async_write_barrier();
}
"""
    writer_ct = [0] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [out.buffer_address()]

    writer_k = ttnn.KernelDescriptor(
        kernel_source=WRITER_SRC,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cr, compile_time_args=writer_ct,
        runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor())

    prog = ttnn.ProgramDescriptor(kernels=[reader_k, writer_k], semaphores=[], cbs=[cb0])

    print("Running pass-through...")
    output = ttnn.generic_op([inp, out], prog)
    ttnn.synchronize_device(device)
    print("Done.")

    out_np = ttnn.to_torch(output).float().numpy().reshape(M, N)
    ref = data.astype(np.float32)  # bf16 rounding
    max_err = np.max(np.abs(ref - out_np))
    print(f"Input first 4:  {ref[0,:4]}")
    print(f"Output first 4: {out_np[0,:4]}")
    print(f"Max error: {max_err:.6f}")
    print(f"PASS: {max_err < 0.1}")

    close_device()

if __name__ == "__main__":
    main()
