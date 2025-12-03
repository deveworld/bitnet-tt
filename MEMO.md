# TT-NN API & Usage Guide

Based on official documentation and tutorials.

## 1. Device Management

Start and stop the device connection. Always close the device to prevent hangs.

```python
import ttnn

# Basic usage
device_id = 0
device = ttnn.open_device(device_id=device_id)
# ... operations ...
ttnn.close_device(device)

# With L1 memory reservation (e.g., for Conv2d)
device = ttnn.open_device(device_id=0, l1_small_size=8192)
```

## 2. Tensor Management

### 2.1 Creation & Conversion

*   **From PyTorch**: The primary way to move data to device.
*   **From NumPy**: Not directly supported; convert to PyTorch first or use `ttnn.Tensor` (limited).
*   **Direct Creation**: `zeros`, `ones`, `full`, `rand`, `arange`.

```python
import torch
import ttnn

# From PyTorch (Recommended)
torch_tensor = torch.randn(32, 32)
tt_tensor = ttnn.from_torch(
    torch_tensor,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG
)

# To PyTorch (for debugging/output)
output_torch = ttnn.to_torch(tt_tensor)
```

### 2.2 Layouts

*   **`ttnn.ROW_MAJOR_LAYOUT`**:
    *   Standard row-major order.
    *   Best for: Data loading, non-compute-intensive ops, or when required by specific ops (e.g., embedding inputs).
    *   Alignment: Width usually requires alignment (e.g., % 2 for bfloat16).
*   **`ttnn.TILE_LAYOUT`** (Crucial for Performance):
    *   Data organized in 32x32 tiles.
    *   **Required** for `matmul` and most compute-intensive ops.
    *   Automatic padding if dimensions aren't multiples of 32.
    *   Inner structure: 16x16 faces.

### 2.3 Data Types (Dtype)

*   **`ttnn.float32`**: Standard FP32.
*   **`ttnn.bfloat16`**: Standard BF16. Recommended for general inference.
*   **`ttnn.bfloat8_b`**: Block floating point (8-bit).
    *   Significant memory/bandwidth savings (2x over BF16).
    *   Must use `TILE_LAYOUT`.
    *   Good for weights and intermediate activations in large models (like LLaMA, BitNet).

### 2.4 Memory Configurations

*   **`ttnn.DRAM_MEMORY_CONFIG`**: Stores tensor in DRAM (Global Memory). Safe default for large tensors.
*   **`ttnn.L1_MEMORY_CONFIG`**: Stores tensor in L1 SRAM (Core Cache).
    *   **Fastest access**.
    *   Limited capacity (~1MB per core).
    *   Use for frequent intermediate results in a compute chain (e.g., inside a Transformer block).
    *   **Manual Management**: Use `ttnn.deallocate(tensor)` to free L1 space if needed.

## 3. Core Operations

### 3.1 Matrix Multiplication

Highly optimized for `TILE_LAYOUT`.

```python
# Basic
output = ttnn.matmul(a, b)
output = a @ b

# Optimized (L1 output, Core Grid)
output = ttnn.matmul(
    a, b,
    memory_config=ttnn.L1_MEMORY_CONFIG,
    # core_grid=ttnn.CoreGrid(y=8, x=8) # Explicit grid often not needed in high-level API
)

# Linear (Matmul + Bias)
# Weight must be pre-transposed usually: (out_features, in_features) -> (in, out)
output = ttnn.linear(input, weight, bias=bias)
```

### 3.2 Element-wise (Pointwise)

*   **Unary**: `relu`, `gelu`, `silu`, `exp`, `log`, `tanh`, `sigmoid`, `reciprocal`.
*   **Binary**: `add`, `sub`, `mul`, `div` (supports broadcasting).

```python
z = ttnn.add(x, y)
z = x + y
z = ttnn.mul(x, y)
```

### 3.3 Normalization

*   **`ttnn.rms_norm`**: Root Mean Square Layer Norm (common in LLMs).
*   **`ttnn.layer_norm`**: Standard Layer Norm.
*   **`ttnn.softmax`**: Softmax (usually on dim=-1).

```python
# RMS Norm
out = ttnn.rms_norm(x, weight=scale, epsilon=1e-5)
```

## 4. Advanced Model Components

### 4.1 Transformer Ops

TT-NN provides fused/optimized kernels for Transformers.

*   **`ttnn.transformer.rotary_embedding`**: Apply RoPE.
*   **`ttnn.transformer.scaled_dot_product_attention`**: Fused SDPA.
    *   Supports causal masking.
    *   Optimized for flash-attention-like performance.

### 4.2 Convolution (Conv2d)

*   **Input Layout**: Requires `NHWC` (Batch, Height, Width, Channels).
    *   PyTorch is usually `NCHW`, so `permute(0, 2, 3, 1)` is needed.
*   **Config**: Use `ttnn.Conv2dConfig`.

```python
# Prepare input (B, C, H, W) -> (B, H, W, C)
input_permuted = ttnn.permute(input_tensor, (0, 2, 3, 1))

# Run Conv2d
output = ttnn.conv2d(
    input_tensor=input_permuted,
    weight_tensor=weight,
    bias_tensor=bias,
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=(3, 3),
    stride=(1, 1),
    padding=(1, 1),
    # ... specific config ...
)
```

### 4.3 Embedding

*   Input indices should be `uint32`.
*   Weights in `ROW_MAJOR` usually worked best in older versions, check if `TILE` is supported now.

## 5. Optimization & Best Practices

1.  **Layout is King**: Keep tensors in `TILE_LAYOUT` as much as possible. Only convert to `ROW_MAJOR` if an op strictly requires it (e.g., some data movements or embeddings) or for final output.
2.  **Minimize Host-Device Transfer**: `from_torch` and `to_torch` are slow. Load weights once, keep activations on device.
3.  **Program Caching**: TT-NN compiles kernels (programs) on the first run. Subsequent runs with the same shapes are much faster.
    *   Use static shapes where possible.
    *   Warmup iterations are real: run your model once before measuring performance.
4.  **L1 Memory usage**: For deep networks, chaining ops with `L1_MEMORY_CONFIG` avoids round-trips to DRAM, significantly boosting throughput.
5.  **Async Execution**: TT-NN operations are asynchronous on host. `to_torch` or `synchronize_device` acts as a barrier.

## 6. Debugging

*   **`ttnn.set_printoptions(profile="short")`**: Control tensor printing.
*   **Comparison**: Always verify against a PyTorch "Golden" reference (like in `debug_compare.py`).
*   **Visualizer**: Tenstorrent provides a graph visualizer (via `tt_metal` tools) to inspect the execution graph.

## 7. Specific BitNet-TT Notes (Project Context)

*   **BitLinear**:
    *   The project implements manual quantization in `BitLinear`.
    *   Weights: {-1, 0, 1}. Stored as float/bfloat16 for compute currently.
    *   Activations: 8-bit quantization.
*   **Current Status**:
    *   Basic layers implemented (`BitLinear`, `RMSNorm`, `Attention`).
    *   Mini model and Full 2B model demo scripts exist.
    *   Needs validation on actual hardware to confirm `bfloat8_b` or integer math capabilities for further optimization.