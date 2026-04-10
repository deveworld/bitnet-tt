#!/usr/bin/env python3
"""Test: fused reader unpacks packed weight to CB1 → writer dumps CB1 to output.
No compute. Verifies the 2-bit unpack logic produces correct bf16 tiles."""
import os, numpy as np, torch, ttnn
from bitnet_tt.kernels.pack import pack_ternary_tilized, PACKED_TILE_BYTES
from bitnet_tt.utils.quantization import weight_quant_ternary
from bitnet_tt.utils.device import get_device, close_device

KERNEL_DIR = os.path.join(os.path.dirname(__file__), "src", "bitnet_tt", "kernels", "device")

def main():
    device = get_device()
    BF16_TILE = 2048
    CB_SIZE = BF16_TILE * 2

    # Create known ternary weight: all +1
    weight_quant = np.ones((32, 32), dtype=np.int8)
    scale = 1.0
    packed_bytes, _ = pack_ternary_tilized(weight_quant, scale)
    print(f"Packed: {packed_bytes.shape[0]} tiles, {packed_bytes.nbytes} bytes")

    # Pack as uint32 ROW_MAJOR — each page = 64 uint32 = 256 bytes = 1 packed tile
    packed_flat = packed_bytes.flatten()  # [256] uint8 for 1 tile
    packed_i32 = np.frombuffer(packed_flat.tobytes(), dtype=np.int32)  # [64]
    packed_tt = ttnn.from_torch(
        torch.from_numpy(packed_i32.copy()).unsqueeze(0),  # [1, 64]
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Output for unpacked weight tile
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([32, 32]), ttnn.bfloat16,
        ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)

    core = ttnn.CoreCoord(0, 0)
    cr = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # CB1 for unpacked weight, CB2 for scratch
    cb1 = ttnn.CBDescriptor(total_size=CB_SIZE, core_ranges=cr,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1,
            data_format=ttnn.bfloat16, page_size=BF16_TILE)])
    # Scratch CB: page_size = 256 bytes = PACKED_TILE_BYTES
    # Use uint32 format, page_size=256
    cb_scratch = ttnn.CBDescriptor(total_size=512, core_ranges=cr,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=2,
            data_format=ttnn.uint32, page_size=256)])

    # Reader: only read packed weight, unpack to CB1
    # (skip activation entirely)
    READER_SRC = """
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

constexpr uint32_t PACKED_TILE_BYTES = 256;
constexpr uint16_t LUT[4] = {0x0000u, 0x3F80u, 0xBF80u, 0x0000u};

inline void unpack_byte(uint8_t packed, uint16_t* dst) {
    dst[0] = LUT[(packed >> 6) & 0x03];
    dst[1] = LUT[(packed >> 4) & 0x03];
    dst[2] = LUT[(packed >> 2) & 0x03];
    dst[3] = LUT[packed & 0x03];
}

void kernel_main() {
    uint32_t packed_addr = get_arg_val<uint32_t>(0);
    constexpr auto packed_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t cb_scratch = 2;

    const uint32_t scratch_page = get_local_cb_interface(cb_scratch).fifo_page_size;
    const auto packed_tensor = TensorAccessor(packed_args, packed_addr, scratch_page);

    experimental::Noc noc;
    experimental::CircularBuffer cb_s(cb_scratch);
    experimental::CircularBuffer cb1_buf(cb_in1);

    // Read packed tile into scratch via TensorAccessor
    cb_s.reserve_back(1);
    noc.async_read(packed_tensor, cb_s, scratch_page, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_s.push_back(1);

    // Unpack from scratch to CB1
    cb_s.wait_front(1);
    cb1_buf.reserve_back(1);
    uint32_t l1_scratch = get_local_cb_interface(cb_scratch).fifo_rd_ptr;
    uint32_t l1_weight = get_local_cb_interface(cb_in1).fifo_wr_ptr;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(l1_scratch);
    uint16_t* dst = reinterpret_cast<uint16_t*>(l1_weight);
    for (uint32_t i = 0; i < PACKED_TILE_BYTES; ++i) {
        unpack_byte(src[i], &dst[i * 4]);
    }
    cb_s.pop_front(1);
    cb1_buf.push_back(1);
}
"""
    reader_ct = list(ttnn.TensorAccessorArgs(packed_tt).get_compile_time_args())
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [packed_tt.buffer_address()]

    reader_k = ttnn.KernelDescriptor(
        kernel_source=READER_SRC,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cr, compile_time_args=reader_ct,
        runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor())

    # Writer: read from CB1, write to output
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
    writer_ct = [1] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [out.buffer_address()]

    writer_k = ttnn.KernelDescriptor(
        kernel_source=WRITER_SRC,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cr, compile_time_args=writer_ct,
        runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor())

    prog = ttnn.ProgramDescriptor(kernels=[reader_k, writer_k], semaphores=[],
                                  cbs=[cb1, cb_scratch])

    print("Running unpack test...")
    output = ttnn.generic_op([packed_tt, out], prog)
    ttnn.synchronize_device(device)
    print("Done.")

    out_np = ttnn.to_torch(output).float().numpy().reshape(32, 32)
    # Expected: all +1.0 (ternary weight was all +1)
    expected = np.ones((32, 32), dtype=np.float32)
    max_err = np.max(np.abs(expected - out_np))
    nonone = np.sum(out_np != 1.0)
    print(f"Expected: all 1.0")
    print(f"Output first row:  {out_np[0, :16]}")
    print(f"Output row 16:     {out_np[16, :16]}")
    print(f"Non-1.0 elements:  {nonone}/1024")
    print(f"Unique values:     {np.unique(out_np).tolist()}")
    print(f"Max error: {max_err:.6f}")
    print(f"PASS: {max_err < 0.01}")

    close_device()

if __name__ == "__main__":
    main()
