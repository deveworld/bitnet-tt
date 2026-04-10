#!/usr/bin/env python3
"""
Track A validation: run packed ternary reader + dense matmul compute via generic_op.

Tests the existing C++ kernels (reader_in0, reader_ternary_in1, ternary_mm_compute,
writer_out) through ttnn.generic_op to verify correctness on p150a hardware.

This is a single-core, single-tile-output test (M=32, K=32, N=32) as the simplest
possible validation before multi-core scaling.
"""
import os
import numpy as np
import torch
import ttnn

from bitnet_tt.kernels.pack import pack_ternary_tilized, PACKED_TILE_BYTES
from bitnet_tt.utils.quantization import weight_quant_ternary
from bitnet_tt.utils.device import get_device, close_device

# Kernel source paths (absolute)
KERNEL_DIR = os.path.join(os.path.dirname(__file__), "src", "bitnet_tt", "kernels", "device")


def test_single_tile_ternary_matmul():
    """M=32, K=32, N=32 → 1 output tile, single core."""
    M, K, N = 32, 32, 32
    Mt, Kt, Nt = M // 32, K // 32, N // 32  # 1, 1, 1

    device = get_device()

    # --- Host data ---
    # Use known simple pattern: all-ones weight → result = row sums of activation
    act_np = np.ones((M, K), dtype=np.float32) * 0.5  # constant activation
    weight_quant = np.ones((N, K), dtype=np.int8)  # all +1 ternary
    scale = 1.0  # no scaling

    # Reference: act @ weight_T → each output element = sum of act row = K * 0.5
    weight_t_scaled = weight_quant.T.astype(np.float32) * scale  # (K, N)
    ref_output = act_np @ weight_t_scaled  # (M, N) — should be all 16.0

    print(f"Reference expected value: {K * 0.5} (all elements)")
    print(f"Reference first 4: {ref_output[0, :4]}")

    # Pack ternary weights (already transposed in pack_ternary_tilized)
    packed_bytes, tile_scales = pack_ternary_tilized(weight_quant, scale)
    print(f"Packed: {packed_bytes.shape[0]} tiles, {packed_bytes.nbytes} bytes")

    # --- Device tensors ---
    # Activation: bf16 TILE_LAYOUT
    act_tt = ttnn.from_torch(
        torch.from_numpy(act_np),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Output: preallocated bf16 TILE_LAYOUT
    out_tt = ttnn.allocate_tensor_on_device(
        ttnn.Shape([M, N]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    # Packed weight: stored as bf16 TILE tensor for TensorAccessor compatibility.
    # Each tile "page" (2048 bytes bf16) holds one packed ternary tile
    # (256 bytes actual data in the first 256 bytes, rest padding).
    packed_flat = packed_bytes.flatten()
    num_packed_tiles = packed_bytes.shape[0]
    # Each packed tile → one bf16 tile page (2048 bytes). Pad packed data
    # to fill a full bf16 tile, then store as bf16 TILE_LAYOUT.
    packed_pages = np.zeros((num_packed_tiles, 32, 32), dtype=np.float32)
    for t in range(num_packed_tiles):
        # Store 256 packed bytes as 128 float32 values (reinterpreted)
        tile_bytes = packed_bytes[t]  # [256] uint8
        # Put raw bytes into the first 256 bytes of 4096-byte float32 tile
        raw_f32 = np.frombuffer(tile_bytes.tobytes(), dtype=np.float32)  # [64] f32
        packed_pages[t, :2, :] = raw_f32.reshape(2, 32)  # first 2 rows × 32 cols
    # Shape for ttnn: [num_tiles, 32, 32]
    packed_torch = torch.from_numpy(packed_pages)
    if num_packed_tiles == 1:
        packed_torch = packed_torch.unsqueeze(0)  # [1, 1, 32, 32]
    packed_tt = ttnn.from_torch(
        packed_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # --- Kernel setup ---
    BF16_TILE_BYTES = 2 * 1024  # 32*32 * 2 bytes
    CB_DOUBLE_BUFFER = 2 * BF16_TILE_BYTES

    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # Circular buffers
    # cb0: activation tiles (bf16)
    # cb1: unpacked weight tiles (bf16, written by fused reader)
    # cb2: scratch for packed DMA (256 bytes per packed tile)
    # cb16: output tiles (bf16)
    # Scratch CB: needs at least PACKED_TILE_BYTES (256) per page.
    # Use a full bf16 tile page (2048) to satisfy CB alignment — only
    # the first 256 bytes of each page are used by the reader.
    SCRATCH_PAGE_SIZE = BF16_TILE_BYTES  # 2048 (oversized but safe)
    SCRATCH_SIZE = SCRATCH_PAGE_SIZE * 2  # double-buffer
    cb_in0 = ttnn.CBDescriptor(
        total_size=CB_DOUBLE_BUFFER,
        core_ranges=core_range,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=0,
            data_format=ttnn.bfloat16,
            page_size=BF16_TILE_BYTES,
        )],
    )
    cb_in1 = ttnn.CBDescriptor(
        total_size=CB_DOUBLE_BUFFER,
        core_ranges=core_range,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=1,
            data_format=ttnn.bfloat16,
            page_size=BF16_TILE_BYTES,
        )],
    )
    cb_scratch = ttnn.CBDescriptor(
        total_size=SCRATCH_SIZE,
        core_ranges=core_range,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=2,
            data_format=ttnn.bfloat16,
            page_size=SCRATCH_PAGE_SIZE,
        )],
    )
    cb_out = ttnn.CBDescriptor(
        total_size=CB_DOUBLE_BUFFER,
        core_ranges=core_range,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=16,
            data_format=ttnn.bfloat16,
            page_size=BF16_TILE_BYTES,
        )],
    )

    # Fused reader: reads activation + packed weight, unpacks to cb1
    act_ct = list(ttnn.TensorAccessorArgs(act_tt).get_compile_time_args())
    packed_ct = list(ttnn.TensorAccessorArgs(packed_tt).get_compile_time_args())
    reader_ct_args = act_ct + packed_ct
    print(f"Reader CT args: act={act_ct} packed={packed_ct}")

    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [act_tt.buffer_address(), packed_tt.buffer_address(), Kt, Nt, Mt]
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=os.path.join(KERNEL_DIR, "reader_ternary_fused.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # Compute: ternary matmul
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=os.path.join(KERNEL_DIR, "ternary_mm_compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=[Mt, Kt, Nt],
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    # Writer — compile-time args: [out_cb_idx, TensorAccessorArgs...]
    out_cb_idx = 16
    writer_ct_args = [out_cb_idx] + list(ttnn.TensorAccessorArgs(out_tt).get_compile_time_args())
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [out_tt.buffer_address(), Mt, Nt]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=os.path.join(KERNEL_DIR, "writer_out.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Program descriptor
    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_in0, cb_in1, cb_scratch, cb_out],
    )

    print("Running generic_op...")
    output = ttnn.generic_op([act_tt, packed_tt, out_tt], program)
    ttnn.synchronize_device(device)
    print("Synchronized.")

    # Compare
    out_np = ttnn.to_torch(output).float().numpy().reshape(M, N)
    max_err = np.max(np.abs(ref_output - out_np))
    corr = np.corrcoef(ref_output.flatten(), out_np.flatten())[0, 1]

    print(f"\nReference (first 4): {ref_output[0, :4]}")
    print(f"TT output (first 4): {out_np[0, :4]}")
    print(f"Max error:  {max_err:.4f}")
    print(f"Correlation: {corr:.6f}")
    print(f"PASS: {corr > 0.99}" if not np.isnan(corr) else "FAIL: NaN correlation")

    close_device()


if __name__ == "__main__":
    test_single_tile_ternary_matmul()
