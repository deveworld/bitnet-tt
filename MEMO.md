# TT-NN BitNet-TT Core Memo (Compressed)

Summary of key patterns and currently working configurations. Removed unnecessary repetition/failure logs. Goal: Stable inference, throughput improvement.

## 0. Current Status & Goals

- Stable path: **Alpha baseline (`cd995ba`) + HiFi2 compute kernel (`4b50353`)** → **~8.5 t/s** (8.02-9.69 t/s range) on batch=32.
- Single-user target: ≥33 t/s (based on tt_transformers Llama 3.1 8B per-user).
- Focus on decode which is DRAM-bandwidth bound. Utilize Trace/Multi-CQ, DRAM-sharded matmul, lightweight kernels (HiFi2/LoFi).

---

## 1. Device & Tensor Basics

- Device open/close

  ```python
  import ttnn
  with ttnn.manage_device(0, l1_small_size=8192) as device:
      ...
  ```

- PyTorch ↔ TT-NN

  ```python
  tt = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16,
                       layout=ttnn.TILE_LAYOUT, device=device,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
  torch_out = ttnn.to_torch(tt)  # sync barrier
  ```

- Direct create (device): `ttnn.zeros/ones/full/rand/arange(...)`
- Async default: Most ops are async. Synchronize with `to_torch()` or `ttnn.synchronize_device(device)`.
- Manual L1 release: `ttnn.deallocate(tensor)` (frees intermediate L1 space).

## 2. Layout / Dtype / Memory Summary

- Layout
  - `ROW_MAJOR_LAYOUT`: Loading/simple ops/embedding input. Width alignment required.
  - `TILE_LAYOUT`: Performance critical. **Required for matmul, linear, attention**. Auto-pads (32x32 tile).
- Dtype (recommended)
  - `bfloat16`: Default.
  - `bfloat8_b` (BFP8): TILE required. Memory savings for weights/intermediates, accuracy OK.
  - `bfloat4_b` (BFP4): TILE required. Speed/memory priority, caution in accuracy-sensitive regions.
  - `uint32`: For index/position (embedding, KV cache idx).
- MemoryConfig
  - `DRAM_MEMORY_CONFIG`: Default for weights/large tensors.
  - `L1_MEMORY_CONFIG`: Small intermediate results. Use `ttnn.deallocate` when needed.
  - Sharded: `HEIGHT_SHARDED` (decode QKV/attention), `WIDTH_SHARDED` (matmul weight). Conversion: `interleaved_to_sharded`, `sharded_to_interleaved`, `ttnn.to_memory_config`.

## 3. Core Ops Patterns

- Matmul / Linear

  ```python
  kernel_hifi2 = ttnn.WormholeComputeKernelConfig(
      math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
      fp32_dest_acc_en=False, packer_l1_acc=False)

  out = ttnn.linear(
      x, weight, bias=None,
      transpose_a=False, transpose_b=True,            # built-in weight transpose
      memory_config=ttnn.L1_MEMORY_CONFIG,            # or DRAM/SHARDED
      core_grid=ttnn.CoreGrid(y=8, x=8),              # adjust for batch/model
      compute_kernel_config=kernel_hifi2)
  ```

- Attention SDPA
  - Prefill: `ttnn.transformer.scaled_dot_product_attention`.
  - Decode: `ttnn.transformer.paged_scaled_dot_product_attention_decode` (paged KV cache + page_table).
- Others: `ttnn.concat`, `ttnn.reshape`, `ttnn.unsqueeze_to_4D`, `ttnn.permute_dim` (minimize usage).

## 4. Transformer / Attention Flow (BitNet-TT)

**Prefill (compute-bound, DRAM interleaved)**  

1) Embedding (row-major input → TILE)  
2) QKV matmul (BF16/HiFi4)  
3) `split_query_key_value_and_split_heads` (prefill-specific helper)  
4) RoPE: pre-uploaded cos/sin → `ttnn.experimental.rotary_embedding_llama`  
5) SDPA (`scaled_dot_product_attention`)  
6) Concat heads → Output proj  
7) MLP (gate+up/down)

**Decode (DRAM-bandwidth-bound, L1/SHARDED)**  

1) QKV projection: DRAM-sharded weight, output L1 WIDTH_SHARDED  
2) `nlp_create_qkv_heads_decode(..., memory_config=HEIGHT_SHARDED)`  
3) RoPE decode (HEIGHT_SHARDED input required, uses trans_mat)  
4) In-place KV cache update: `ttnn.experimental.paged_update_cache` or `kv_cache.update_cache_for_token_` (pass cur_pos tensor)  
5) SDPA decode (paged)  
6) `nlp_concat_heads_decode`  
7) Output proj (DRAM-sharded)  
8) LM head: matmul by vocab chunks → concat. For prefill, compute only last token slice to reduce cost.

### KV Cache & RoPE Tips

- Use in-place updates instead of concat for KV cache to avoid reallocation.
- RoPE: Upload cos/sin matrix once and query per-position via `ttnn.embedding` then shard.
- Trace-friendly: Pass `cur_pos`/`update_idxs` as device tensor instead of list.

## 5. BitNet b1.58 Model Core

- Config (2B-4T): vocab 128,256 | hidden 2,560 | ff hidden 6,912 | layers 30 | heads 20 | kv_heads 5 | head_dim 128 | act `relu2` | rope_theta 500000 | rms_eps 1e-5 | attention_bias False | tie_word_embeddings False.
- BitLinear (weight ternary {-1,0,1} scaled):

  ```python
  # Weight quant (load-time)
  scale_w = 1.0 / weight.abs().mean().clamp(min=1e-5)
  w_q = (weight * scale_w).round().clamp(-1, 1) / scale_w

  # Activation quant (per token)
  scale_x = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
  x_q = (x * scale_x).round().clamp(-128, 127) / scale_x
  ```

- Attention specificity: `attn_sub_norm`(RMSNorm) **after** attention before O-proj. MLP also has `ffn_sub_norm`.
- HuggingFace has separate Q/K/V, official GPU has fused `wqkv`; consider fusing in TT-NN (performance↑).

## 6. Compute / Memory Detailed Settings (Verified Working)

- Layout: Compute path uses TILE. Embedding input/position index uses ROW_MAJOR.
- Memory: Weights in DRAM (DRAM_sharded if needed), intermediates in L1. Decode QKV/attn requires **HEIGHT_SHARDED** input. Output proj/LM head weights use DRAM_sharded for bandwidth↑.
- DRAM-Sharded Matmul: **~26%** improvement vs interleaved (190 → 240 GB/s). Apply to decode matmul.
- CoreGrid: Default 8x8 (32 cores) for decode. Adjust y=batch chunk, x=cores per model/batch.
- Math fidelity recommendation
  - Prefill QKV/SDPA: HiFi4 + BF16 (accuracy).
  - Decode QKV/SDPA: HiFi2 + BFP8 (speed/accuracy balance).
  - MLP FF1/FF3: LoFi + BFP4 (speed), FF2: HiFi2 + BFP8.
- LM Head: Matmul by vocab chunks + concat; slice to last token only for prefill.
- Remove transpose cost: Use `ttnn.linear(..., transpose_b=True)`.

## 7. Trace & Multi-CQ (Performance Critical)

- After one compile run with same input shape, capture trace → repeat with `ttnn.execute_trace(..., blocking=False)`.
- Keep input tensors on device, only update host values in-place (`copy_host_to_device`).
- CQ usage example: CQ0=compute, CQ1=H2D copy for overlap. Greatest benefit when combined with Trace.

## 8. Results/Attempts Summary (Condensed for <1000 lines)

| Attempt | Commit | Result | Speed/Notes |
| --- | --- | --- | --- |
| Alpha baseline | `cd995ba` | ✅ | **8.68 t/s** |
| `_forward_simple` (batch=32) | `fd7a1d2` | ✅ Works | 0.80 t/s (slow) |
| HEIGHT_SHARDED decode | `105f828` etc | ❌ | Failure cause: BitNet `num_heads=20` mismatches CoreGrid 4x8 (32 cores) requirement → shard height mismatch |
| Fused QKV + LM Head slice | `0e97f48` | ❌ | `ttnn.slice` argument error corrupted output |
| **HiFi2 kernel applied** | `4b50353` | ✅ | **8.02-9.69 t/s**, quality normal |

Current status: Using Alpha path + HiFi2 compute kernel as default, DRAM-sharded matmul/LM head slice/trace are additional candidates.

## 9. Future Items to Verify/Apply

1) Apply DRAM-Sharded matmul to entire decode path (QKV, O-proj, LM head).  
2) Compute kernel tuning: HiFi2 default, experiment LoFi for MLP only.  
3) LM head optimization: Prefill last token slice, weight chunk parallelization.  
4) Fused QKV/gate: Re-experiment with `ttnn.slice` keyword arguments (`slice_start/slice_end`).  
5) Fully apply Trace + Multi-CQ to decode loop (tensor reuse).  
6) HEIGHT_SHARDED alternative: Hard to match head count to 32, evaluate custom shard geometry for BitNet.

## 10. Quick API Reference (Core Only)

- Conversion: `ttnn.to_layout(t, layout)`, `ttnn.to_memory_config(t, mem_cfg)`, `interleaved_to_sharded`, `sharded_to_interleaved`.
- Slice: `ttnn.slice(t, slice_start=[...], slice_end=[...])` (keywords required, match rank).
- Trace: `begin_trace_capture(device, cq_id) → end_trace_capture(...) → execute_trace(..., blocking=False) → release_trace(...)`.
- Tensor management: `ttnn.deallocate(t)`, `ttnn.copy_host_to_device_tensor(host, device_tensor)`, `ttnn.synchronize_device(device)`.
- Print/Debug: `ttnn.set_printoptions(profile="short")`, compare against PyTorch golden.

---

## 11. Custom Kernel Development Notes

### 11.1 Blackhole p150a Hardware Characteristics

| Item | Wormhole N150 | Blackhole p150a |
| --- | --- | --- |
| Tensix Cores | 8x10 (8x8 compute) | 14x10 (**13x10 compute = 130 cores**) |
| L1 (per core) | 1464 KB | 1464 KB + **Data Cache** (4×16B) |
| DRAM | 12 banks × 1GB | **8 banks × ~4GB = ~32GB** |
| NoC Alignment | R:32B, W:16B | R:**64B**, W:16B |
| Multicast | Rectangular | Rectangular + **Strided + L-shaped** |

**L1 Data Cache (BH only)**:

```cpp
// Enable/disable in kernel
set_l1_data_cache<true>();   // Enable
invalidate_l1_cache();       // Explicit invalidation (recommended)
set_l1_data_cache<false>();  // Disable before kernel exit
```

### 11.2 TT-Metal Kernel Structure

```text
Program (single Op)
├── Reader Kernel (RISC0) - Read data from DRAM/L1
├── Writer Kernel (RISC1) - Write data to L1/DRAM
└── Compute Kernel (RISC2,3,4) - Tile-based compute

Parallel execution on each Tensix core, synchronized via Circular Buffer
```

**Circular Buffer & Double Buffering**:

```cpp
// Reader: Read data and push to CB
cb_reserve_back(cb_id, num_tiles);
// ... noc_async_read ...
noc_async_read_barrier();
cb_push_back(cb_id, num_tiles);

// Compute: Pop from CB and compute
cb_wait_front(cb_id, num_tiles);
// ... compute ...
cb_pop_front(cb_id, num_tiles);
```

### 11.3 BitNet INT8×INT2 Kernel Pattern (Official GPU Implementation)

**Core Operation**:

```text
Output = (INT8_activation × INT2_weight) × scale_activation × scale_weight
```

**Weight Packing (2-bit → INT8)**:

```python
# Pack 4 2-bit values into 1 int8
# weight: [-1, 0, 1] → [1, 2, 3] (offset by 2)
# Compression: N×K → N×(K/4)
packed = w0 | (w1 << 2) | (w2 << 4) | (w3 << 6)
```

**GPU Kernel Structure** (ladder_int8xint2_kernel):

```cpp
// 1. Weight decode: INT2 → INT8 (per thread)
decode_i2s_to_i8s(B_reshape_local, B_decode_local, 16);
// LUT-based conversion: {1,2,3} → {-1,0,1}

// 2. Dot product accumulation
for (int k = 0; k < 4; k++) {
    acc = __dp4a(A_local[k*4], B_decode_local[k*4], acc);
}

// 3. Warp reduction
for (int offset = K_block/2; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(mask, acc, offset);
}

// 4. Scale and output
output[idx] = (bf16)((float)acc / scale_act * scale_weight);
```

### 11.4 TT-Metal Optimization Techniques

**DRAM-Sharded Matmul** (for decode, ~26% bandwidth improvement):

```python
# Shard weight across DRAM banks
memory_config = create_dram_sharded_mem_config(k=K, n=N)
program_config = dram_matmul_config(m=M, k=K, n=N, num_cores=cores)

output = ttnn.linear(
    activation, weights,
    compute_kernel_config=kernel_hifi2,
    program_config=program_config,
    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
)
```

**Math Fidelity**:

| Fidelity | Precision | Speed | Use Case |
| --- | --- | --- | --- |
| HiFi4 | BF16 | 1x | Prefill, accuracy-critical |
| HiFi2 | BFP8 | **2x** | Decode default |
| LoFi | BFP4 | **3.6x** | MLP (when accuracy permits) |

### 11.5 FlashDecode Pattern (SDPA Decode)

```python
# Query shape transform: Place heads in y dimension
# [bsz, num_q_heads, 1, head_dim] → [1, bsz, num_q_heads, head_dim]
query = query.view(1, bsz, num_q_heads, head_dim)

# Distribute bsz * n_kv_heads across cores
# n_qh_per_kvh (= num_q_heads // num_kv_heads) processed within core
attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
    query, keys, values,
    cur_pos_tensor=current_pos,  # causal masking (no explicit mask needed)
)
```

**Performance**: Up to 180 GB/s memory bandwidth on Wormhole (~70% of DRAM 288 GB/s)

### 11.6 BitNet-TT Custom Kernel Design Direction

**Problem**: `num_heads=20` mismatches HEIGHT_SHARDED (32 cores) requirement

**Solutions**:

1. **Ternary Matmul Kernel** (INT8×TERNARY):
   - Store weight as 2-bit packed (8x memory savings: FP32 → 2bit)
   - Activation as INT8 quantized
   - Alleviates DRAM bandwidth bottleneck

2. **Custom QKV Heads Decode**:
   - Shard geometry matching `num_heads=20`
   - CoreGrid: 20 cores (4x5 or 5x4) or pad to 32

3. **Fused Ops**:
   - Fuse QKV projection + split heads
   - Fuse RMSNorm + Linear (sub_norm pattern)

### 11.7 Trace Capture Pattern (2-3x Speed Improvement)

```python
# 1. Compile run (first execution)
output = model_forward(input_tensor, pos_tensor)

# 2. Trace capture
ttnn.synchronize_device(device)
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
output = model_forward(input_tensor, pos_tensor)  # same shape
trace_id = ttnn.end_trace_capture(device, trace_id, cq_id=0)

# 3. Execute trace (repeat)
ttnn.copy_host_to_device_tensor(new_input_host, input_tensor)
ttnn.copy_host_to_device_tensor(new_pos_host, pos_tensor)
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
```

**Constraints**:

- No tensor alloc/dealloc during trace → **in-place KV cache required**
- Same shape only → apply to decode only (not prefill)

### 11.8 Reference Material Paths

| Material | Path |
| --- | --- |
| LLM Optimization Guide | `~/tt-metal/tech_reports/LLMs/llms.md` |
| FlashDecode | `~/tt-metal/tech_reports/FlashAttention/FlashDecode.md` |
| FlashAttention | `~/tt-metal/tech_reports/FlashAttention/FlashAttention.md` |
| Blackhole Guide | `~/tt-metal/tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md` |
| BitNet GPU Kernels | `~/BitNet/gpu/bitnet_kernels/` |
| tt_transformers | `~/tt-metal/models/tt_transformers/` |
| SDPA Decode Source | `~/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/` |

---

## 12. Custom Kernel Development Plan

### 12.1 Current Analysis

| Item | Current | Target | Bottleneck |
| --- | --- | --- | --- |
| Speed | 8.5 t/s | **30+ t/s** | DRAM bandwidth, kernel dispatch |
| Optimization Path | `_forward_simple` (concat) | Trace + In-place | concat blocks trace |
| HEIGHT_SHARDED | ❌ (num_heads=20) | Custom geometry | 32-core assumption |
| Weight format | BF16 (fp16) | **2-bit packed** | Memory bandwidth |

### 12.2 Development Priorities

#### Phase 1: In-place KV Cache (Enable Trace) - **Most Effective**

**Goal**: Enable Trace by switching from concat to index-based update (2-3x speed improvement)

**Changes**:

```python
# Current (concat-based - Trace impossible)
new_key = ttnn.concat([self.key_cache, key], dim=2)  # allocates new memory

# Target (index-based - Trace possible)
# Write directly to pre-allocated cache
ttnn.fill_cache(self.key_cache, key, update_idx=current_pos)
```

**Implementation**:

1. Use simple `fill_cache` instead of `paged_update_cache` (no HEIGHT_SHARDED needed)
2. Pre-allocate KVCache to max_seq_len size
3. Specify write position with `current_pos` tensor

**Expected Effect**: **8.5 → 17-25 t/s** (2-3x)

#### Phase 2: Ternary Matmul Kernel - **Memory Bandwidth Improvement**

**Goal**: 8x reduction in DRAM reads with 2-bit weight matmul

**Core Idea**:

```text
Current: BF16 weight (16 bit/param) → DRAM read = 2.4B params × 2 bytes = 4.8 GB
Target: 2-bit weight (2 bit/param) → DRAM read = 2.4B params × 0.25 bytes = 0.6 GB
```

**TT-Metal Kernel Structure**:

```text
ttnn/cpp/ttnn/operations/matmul/ternary_matmul/
├── ternary_matmul.hpp/.cpp          # Python API binding
├── device/
│   ├── ternary_matmul_device_op.hpp/.cpp
│   ├── ternary_matmul_program_factory.hpp/.cpp
│   └── kernels/
│       ├── dataflow/reader_ternary.cpp  # 2-bit unpack
│       ├── dataflow/writer_ternary.cpp
│       └── compute/ternary_matmul.cpp   # INT8 × {-1,0,1}
```

**Compute Flow**:

1. Reader: Read packed 2-bit weight from DRAM
2. Reader: Unpack 4 2-bit → 4 INT8 (lookup table)
3. Compute: INT8 activation × INT8 weight (ternary) accumulate
4. Compute: Apply scale (activation_scale × weight_scale)
5. Writer: Write BF16 output

**Expected Effect**: 50-70% speed improvement for decode matmul

#### Phase 3: Custom QKV Decode (num_heads=20 Support)

**Goal**: Enable HEIGHT_SHARDED optimization path

##### Method A: Padding (Simple)

```python
# num_heads=20 → 32 (add 12 dummy heads)
# CoreGrid(y=4, x=8) = 32 cores
padded_heads = 32
# padding overhead: 32/20 = 1.6x compute increase
```

##### Method B: Custom Shard Geometry (Optimal)

```python
# CoreGrid matching num_heads=20
# Option 1: 20 cores (4x5 or 5x4)
# Option 2: 40 cores (2 heads/core) - 5x8
# Requires modifying rotary_embedding_llama, sdpa_decode
```

##### Expected Effect: Additional 30-50% improvement after Phase 1+2

### 12.3 Implementation Roadmap

```text
Week 1: Phase 1 - In-place KV Cache
├── Day 1-2: Implement fill_cache/update_cache (Python level)
├── Day 3-4: Integrate Trace capture
└── Day 5: Test and benchmark

Week 2-3: Phase 2 - Ternary Matmul Kernel
├── Day 1-3: Implement 2-bit weight packing/unpacking
├── Day 4-7: Write TT-Metal kernel (reader/compute/writer)
├── Day 8-10: Python binding and ttnn op registration
└── Day 11-14: BitNet-TT integration and testing

Week 4: Phase 3 - Custom QKV Decode (Optional)
├── Option A: Padding approach (1-2 days)
└── Option B: Custom geometry (1 week)
```

### 12.4 Expected Performance

| Phase | Implementation | Speed | vs Current |
| --- | --- | --- | --- |
| Current | concat + BF16 | 8.5 t/s | 1x |
| Phase 1 | In-place + Trace | 17-25 t/s | **2-3x** |
| Phase 2 | + Ternary Matmul | 25-35 t/s | **3-4x** |
| Phase 3 | + Custom QKV | 30-45 t/s | **4-5x** |

### 12.5 Risks and Alternatives

| Risk | Probability | Alternative |
| --- | --- | --- |
| fill_cache doesn't exist | Medium | Workaround with concat → slice or implement directly |
| Ternary kernel accuracy | Low | Maintain HiFi2 math fidelity |
| QKV geometry conflict | High | Fallback to padding approach |
| Trace instability | Medium | Apply only non-trace optimizations |

### 12.6 Immediately Actionable Items

1. **Investigate ttnn.fill_cache / ttnn.update_cache API**

   ```bash
   grep -r "fill_cache\|update_cache" ~/tt-metal/ttnn/
   ```

2. **Check for existing ternary matmul**

   ```bash
   grep -r "ternary\|int2\|2bit" ~/tt-metal/ttnn/cpp/
   ```

3. **Test simple index-based cache update**

   ```python
   # Test ttnn.copy or slice-based update
   cache[:, :, pos:pos+1, :] = new_kv
   ```

---

## 13. TT-NN LLM Optimization Patterns (Based on Official Tech Report)

> **Source**: `~/tt-metal/tech_reports/LLMs/llms.md` (1871 lines), `AdvancedPerformanceOptimizationsForModels.md`, HuggingFace BitNet implementation, official BitNet GPU kernels

### 13.1 Performance Benchmarks (Blackhole p150)

| Model | Hardware | Batch | t/s/u | t/s | Notes |
| --- | --- | --- | --- | --- | --- |
| **Llama 3.1 8B** | p150 | 32 | **33.1** | 1059.2 | **BitNet target reference** |
| Llama 3.1 8B | p100 | 32 | 29.0 | 928.0 | |
| Llama 3.2 1B | n150 (WH) | 32 | 80.5 | 2576.0 | Small model |
| Mistral 7B | n150 (WH) | 32 | 28.7 | 918.4 | |

**BitNet-TT Current**: ~8.5 t/s → **Target 33.1 t/s (3.9x improvement needed)**

### 13.2 Key Optimization Techniques Summary

#### 13.2.1 Prefill vs Decode Separation

| Item | Prefill | Decode |
| --- | --- | --- |
| **Bottleneck** | Compute-bound | **DRAM bandwidth-bound** |
| **Memory** | DRAM interleaved | L1 sharded (**HEIGHT_SHARDED**) |
| **Batch** | Single user, full sequence | Multi-user (≤32), 1 token each |
| **Optimization** | Matmul 2D | **DRAM-sharded matmul** |

#### 13.2.2 Attention Implementation Patterns

**Prefill (scaled_dot_product_attention)**:

```python
# 1. Fused QKV
xqkv_fused = ttnn.linear(x, wqkv, dtype=ttnn.bfloat16)

# 2. Split heads
Q, K, V = ttnn.experimental.nlp_create_qkv_heads(
    xqkv_fused, num_heads=n_q_heads, num_kv_heads=n_kv_heads,
    transpose_k_heads=False
)

# 3. RoPE (fused op)
Q = ttnn.experimental.rotary_embedding_llama(Q, cos, sin, trans_mat, is_decode_mode=False)
K = ttnn.experimental.rotary_embedding_llama(K, cos, sin, trans_mat, is_decode_mode=False)

# 4. KV Cache fill
ttnn.fill_cache(K_cache, K, batch_idx)
ttnn.fill_cache(V_cache, V, batch_idx)

# 5. SDPA (causal)
attn_out = ttnn.transformer.scaled_dot_product_attention(Q, K_cache, V_cache, is_causal=True)

# 6. Concat heads + output proj
attn_out = ttnn.experimental.nlp_concat_heads(attn_out)
output = ttnn.linear(attn_out, wo)
```

**Decode (paged_scaled_dot_product_attention_decode)**:

```python
# 1. Fused QKV (DRAM-sharded matmul)
xqkv_fused = ttnn.linear(x, wqkv, program_config=dram_sharded_config)

# 2. Split heads (HEIGHT_SHARDED output) ← BitNet issue: num_heads=20
Q, K, V = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused, num_heads=20, num_kv_heads=5,
    memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
)

# 3. RoPE decode (HEIGHT_SHARDED input required)
Q = ttnn.experimental.rotary_embedding_llama(Q, cos, sin, trans_mat, is_decode_mode=True)
K = ttnn.experimental.rotary_embedding_llama(K, cos, sin, trans_mat, is_decode_mode=True)

# 4. In-place KV Cache update (Trace compatible) ← Key!
ttnn.experimental.paged_update_cache(keys, K, update_idxs_tensor=cur_pos_tensor, page_table=page_table)
ttnn.experimental.paged_update_cache(values, V, update_idxs_tensor=cur_pos_tensor, page_table=page_table)

# 5. SDPA Decode (cur_pos_tensor for causal masking)
attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    Q, keys, values,
    cur_pos_tensor=cur_pos_tensor,
    page_table=page_table
)

# 6. Concat heads + output proj
attn_out = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=20)
output = ttnn.linear(attn_out, wo)
```

#### 13.2.3 MLP Optimization Pattern

```python
# Decode mode: DRAM-sharded matmul + L1 sharded output
# FF1/FF3 (column parallel)
w1_out = ttnn.linear(ff_in, w1, program_config=dram_sharded_config,
                     memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
w3_out = ttnn.linear(ff_in, w3, program_config=dram_sharded_config,
                     memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

# Fused SiLU + multiply (BitNet uses squared_relu)
w2_in = ttnn.multiply(w1_out, w3_out, input_tensor_a_activation=ttnn.UnaryOpType.SILU)

# FF2 (row parallel) + reduce
w2_out = ttnn.linear(w2_in, w2, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
```

**BitNet Differences**:

- `squared_relu` instead of `SiLU`: `ttnn.relu(x) ** 2`
- Additional `ffn_sub_norm`: RMSNorm after gate*up, before down_proj

#### 13.2.4 Complete Trace Capture Pattern

```python
# 1. Pre-allocate persistent tensors
input_dram_tensor = ttnn.allocate_tensor_on_device(spec, device, dram_sharded_config)
op_event = ttnn.record_event(device, 0)  # Dummy event

# 2. Compile run
ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=1)
input_l1 = ttnn.to_memory_config(input_dram_tensor, l1_sharded_config)
output = run_model(input_l1)

# 3. Trace capture
write_event = ttnn.record_event(device, 1)
ttnn.wait_for_event(0, write_event)
input_l1 = ttnn.to_memory_config(input_dram_tensor, l1_sharded_config)
op_event = ttnn.record_event(device, 0)

input_trace_addr = input_l1.buffer_address()
output.deallocate(force=True)  # Free for input reallocation

tid = ttnn.begin_trace_capture(device, cq_id=0)
output = run_model(input_l1)
input_l1 = ttnn.allocate_tensor_on_device(input_l1.spec, device)
assert input_l1.buffer_address() == input_trace_addr  # Verify address
ttnn.end_trace_capture(device, tid, cq_id=0)

# 4. Execute loop (Multi-CQ: CQ0=ops, CQ1=writes)
for _ in iterations:
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(new_host, input_dram_tensor, cq_id=1)
    write_event = ttnn.record_event(device, 1)

    ttnn.wait_for_event(0, write_event)
    input_l1 = ttnn.reshard(input_dram_tensor, l1_sharded_config, input_l1)
    op_event = ttnn.record_event(device, 0)

    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    outputs.append(output.cpu(blocking=False))

ttnn.synchronize_device(device)
```

### 13.3 HuggingFace BitNet Implementation Analysis

**modular_bitnet.py / modeling_bitnet.py**:

```python
class BitNetMLP(GemmaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.ffn_sub_norm = BitNetRMSNorm(config.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, x):
        # gate * up → ffn_sub_norm → down
        down_proj = self.down_proj(self.ffn_sub_norm(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
        return down_proj

class BitNetAttention(LlamaAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.attn_sub_norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, ...):
        # ... standard attention ...
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.attn_sub_norm(attn_output)  # ← BitNet-specific structure
        attn_output = self.o_proj(attn_output)
        return attn_output
```

**configuration_bitnet.py Key Parameters**:

```python
vocab_size=128256, hidden_size=2560, intermediate_size=6912,
num_hidden_layers=30, num_attention_heads=20, num_key_value_heads=5,
hidden_act="relu2",  # squared ReLU
rope_theta=500000.0, rms_norm_eps=1e-5,
tie_word_embeddings=False
```

### 13.4 Official BitNet GPU Kernel Analysis

**bitnet_kernels.cu** (INT8 × INT2 matmul):

```cpp
// M=1 (decode), specialized kernel for various N/K
extern "C" void bitlinear_int8xint2(
    int8_t* input0,      // INT8 quantized activation
    int8_t* input1,      // INT2 packed weight (4 weights per byte)
    __nv_bfloat16* output0,
    __nv_bfloat16* s,    // activation scale
    __nv_bfloat16* ws,   // weight scale
    int M, int N, int K,
    cudaStream_t stream
);

// BitNet 2B-4T specific shapes:
if (M == 1 && N == 3840 && K == 2560)  // wqkv (n_heads=20, head_dim=128, 3*hidden)
if (M == 1 && N == 2560 && K == 2560)  // wo, input norm
if (M == 1 && N == 13824 && K == 2560) // w13 (gate+up fused, 2*intermediate)
if (M == 1 && N == 2560 && K == 6912)  // w2 (down_proj)
```

**model.py** (Official BitNet GPU Implementation):

```python
class BitLinearKernel(nn.Module):
    def __init__(self, in_features, out_features):
        # weight: [out, in//4] INT8 (4 weights packed)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features//4, dtype=torch.int8))
        self.weight_scale = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))

    @torch.compile
    def quant_input(self, input):
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127).to(torch.int8), s

    def forward(self, input):
        input, s = self.quant_input(input)
        return bitnet_int8xint2_linear(input, self.weight, s, self.weight_scale)

# Attention structure
class Attention(nn.Module):
    def __init__(self, ...):
        Linear = BitLinearKernel if use_kernel else BitLinear
        self.wqkv = Linear(dim, (n_q_heads + 2*n_kv_heads) * head_dim)  # Fused!
        self.wo = Linear(n_q_heads * head_dim, dim)
        self.attn_sub_norm = RMSNorm(dim, norm_eps)

    def forward(self, x, cache, attn_bias):
        xqkv = self.wqkv(x)
        # Split → RoPE → KV cache → attention → sub_norm → wo
        output = self.attn_sub_norm(output)
        output = self.wo(output)
        return output

# FFN structure
class FeedForward(nn.Module):
    def __init__(self, ...):
        self.w13 = Linear(dim, 2 * hidden_dim)  # gate+up fused
        self.w2 = Linear(hidden_dim, dim)
        self.ffn_sub_norm = RMSNorm(hidden_dim, norm_eps)

    def forward(self, x):
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, -1)
        inner = self.ffn_sub_norm(squared_relu(x1) * x3)  # squared_relu!
        return self.w2(inner)
```

### 13.5 BitNet-TT Implementation Improvement Direction

#### Problem Analysis

| Current Implementation | Official Implementation | Improvement Direction |
| --- | --- | --- |
| Separate Q/K/V projection | Fused wqkv | Use `wqkv` fused matmul |
| Separate gate/up | Fused w13 | Use `w13` fused matmul |
| BF16 weights | INT2 packed | Ternary matmul kernel |
| concat KV cache | In-place update | `paged_update_cache` |
| No trace | Trace | Apply Trace after in-place |

#### Implementation Priorities

1. **Fused Projections** (Immediately feasible):

   ```python
   # Current: 3 matmuls
   q = ttnn.linear(x, w_q)
   k = ttnn.linear(x, w_k)
   v = ttnn.linear(x, w_v)

   # Improved: 1 fused matmul + split
   qkv = ttnn.linear(x, w_qkv)  # [batch, seq, (q+k+v)]
   Q, K, V = ttnn.experimental.nlp_create_qkv_heads(qkv, ...)
   ```

2. **In-place KV Cache** (Prerequisite for Trace):

   ```python
   # Pre-allocate
   k_cache = ttnn.allocate_tensor_on_device([batch, kv_heads, max_seq, head_dim], device)

   # Update (not concat!)
   ttnn.experimental.paged_update_cache(k_cache, k_new, update_idxs_tensor=pos_tensor)
   ```

3. **Ternary Matmul** (8x memory savings):
   - Requires TT-Metal custom kernel
   - INT8 activation × INT2 weight → BF16 output

### 13.6 Performance Optimization Checklist

- [ ] Fused QKV projection (`wqkv`)
- [ ] Fused gate+up projection (`w13`)
- [ ] In-place KV cache update (`paged_update_cache`)
- [ ] Trace capture for decode loop
- [ ] Multi-CQ (CQ0=ops, CQ1=IO)
- [ ] DRAM-sharded matmul for decode
- [ ] HEIGHT_SHARDED for attention (need to resolve num_heads=20)
- [ ] Ternary matmul kernel (Phase 2)

---

This document is a core summary with duplicates/failure logs removed. When adding new experiments, record successful configurations in detail, leave only one-line cause for failures.
