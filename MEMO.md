# TT-NN BitNet-TT Core Memo (Compressed)

Summary of key patterns and currently working configurations. Removed unnecessary repetition/failure logs. Goal: Stable inference, throughput improvement.

## 0. Current Status & Goals

- Current perf (2026-04-18, batch32 + trace + fused RoPE + packed_ternary,
  reconciled in `docs/session_7_baseline_reconciled.md`):
  **p50 12.0 ms / decode_tps 71.05 t/s (128-tok bench)**,
  p50 11.5 ms / decode_tps 68.93 t/s (64-tok bench).
  Previous MEMO figure (p50 17.5 ms / 57.1 t/s) was stale — predates the
  split-lm-head / sharded-rmsnorm / multicore-argmax / cos-sin-lookup /
  fused-QKV-norm stack.
- Historical Alpha baseline (`cd995ba` + HiFi2 kernel ≈ 8.5 t/s) kept
  in tables below for context only — current stable path is the
  packed_ternary + fused-QKV-norm pipeline documented in README / TODO.
- Focus now: attention data-path core-range split
  (`docs/plan_attention_core_range_split.md`) and anything else not
  already dispatcher-overhead-limited.

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

> NOTE: The table below records the *early* bf16/bfp8 attempts. The
> packed_ternary + fused-QKV-norm pipeline described at the top of
> this file supersedes all of these. For the current perf-evolution
> ladder (baseline → 57 t/s p50) see `TODO.md`.

| Attempt | Commit | Result | Speed/Notes |
| --- | --- | --- | --- |
| Alpha baseline | `cd995ba` | ✅ | **8.68 t/s** (historical) |
| `_forward_simple` (batch=32) | `fd7a1d2` | ✅ Works | 0.80 t/s (slow) |
| HEIGHT_SHARDED decode | `105f828` etc | ❌ | Failure cause: BitNet `num_heads=20` mismatches CoreGrid 4x8 (32 cores) requirement → shard height mismatch |
| Fused QKV + LM Head slice | `0e97f48` | ❌ | `ttnn.slice` argument error corrupted output |
| **HiFi2 kernel applied** | `4b50353` | ✅ | **8.02-9.69 t/s** (historical), quality normal |

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

> **NOTE (2026-04-18, session 5):** This section describes the **reference
> GPU kernel** from the BitNet paper's CUDA implementation. It is **not**
> what the TT-Metal `ternary_matmul` kernel does. The TT kernel keeps
> activations in bf16 end-to-end, decodes 2-bit weights into a bf16 LUT
> `{0, +1, -1}`, and runs `matmul_block` on bf16 × bf16 with
> `fp32_dest_acc_en = true`. There is no runtime INT8 activation
> quantization step on the TT path. See `docs/session_5_correct_attribution.md`
> for the ablation evidence and the corrected PCC-error attribution.

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

> Historical snapshot from the early development phase. All four items
> below have been addressed — the Trace-mode packed_ternary pipeline
> runs at p50 17.5 ms / ~57 t/s. Kept here to show the original
> problem statement; for current optimization targets see Track D / E
> in `TODO.md`.

| Item | Historic | Current | Resolved by |
| --- | --- | --- | --- |
| Speed | 8.5 t/s | **57 t/s p50** | Packed-ternary + trace + L1-norm + fused-QKV-norm |
| Optimization Path | `_forward_simple` (concat) | Trace (batch32) + in-place paged cache | In-trace argmax, streaming decode |
| HEIGHT_SHARDED | ❌ (num_heads=20) | ✅ via `nlp_create_qkv_heads_decode` | Decode-specific head-split op |
| Weight format | BF16 (fp16) | **2-bit packed (BFP2_b)** | Custom `ternary_matmul` op |

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

### 12.4 Phase 1 Results (COMPLETED)

**Implementation**: `update_decode_simple()` using `ttnn.kv_cache.update_cache_for_token_`

```python
# New method in KVCache class (attention.py)
def update_decode_simple(self, key_states, value_states, current_pos, num_kv_groups=1):
    # Uses kv_cache API that accepts [batch, heads, 1, head_dim] directly
    self.key_cache = ttnn.kv_cache.update_cache_for_token_(
        self.key_cache, key_states, update_index=current_pos, batch_offset=0
    )
    self.value_cache = ttnn.kv_cache.update_cache_for_token_(
        self.value_cache, value_states, update_index=current_pos, batch_offset=0
    )
    # ... slice and GQA expansion
```

**Test Results**:
- Performance: **5.74 t/s** (baseline ~5.5 t/s, minimal improvement)
- Accuracy: **0.990** logits correlation with HuggingFace
- Text quality: Coherent output ("The capital of France is Paris...")
- Commit: `744bc69` - "feat: Phase 1 - use kv_cache.update_cache_for_token_ for in-place decode"

**Why Minimal Speed Improvement**:
- `update_cache_for_token_` uses integer position (not tensor), so **Trace capture still blocked**
- The concat→in-place change saves allocation but kernel dispatch overhead dominates
- Real 2-3x speedup requires Trace, which needs `paged_update_cache` with HEIGHT_SHARDED

**Blocking Issue for Trace**:
- `paged_update_cache` requires HEIGHT_SHARDED input
- HEIGHT_SHARDED requires shard height divisible by 32 (core grid constraint)
- `num_heads=20` (BitNet) is incompatible with 32-core grid requirement
- Options: (A) Pad to 32 heads (1.6x compute overhead), (B) Custom shard geometry

### 12.5 Expected Performance (Updated — historic plan, all shipped)

> All phases below are done. The actual final number (p50 17.5 ms ≈
> 57 t/s, decode_tps ~50) is well past the original Phase 3 target.

| Phase | Implementation | Planned speed | Status |
| --- | --- | --- | --- |
| Current (historic) | concat + BF16 | 8.5 t/s | baseline |
| Phase 1 | In-place (no Trace) | 5.7 t/s | ✅ done (superseded) |
| Phase 1b | In-place + Trace | 17-25 t/s | ✅ done |
| Phase 2 | + Ternary Matmul | 25-35 t/s | ✅ done |
| Phase 3 | + Custom QKV | 30-45 t/s | ✅ done |
| Extras (2026-04-17) | + L1 norm/SDPA + fused-QKV-norm | — | ✅ **57 t/s p50** |

### 12.6 Risks and Alternatives

| Risk | Probability | Alternative |
| --- | --- | --- |
| fill_cache doesn't exist | Medium | Workaround with concat → slice or implement directly |
| Ternary kernel accuracy | Low | Maintain HiFi2 math fidelity |
| QKV geometry conflict | High | Fallback to padding approach |
| Trace instability | Medium | Apply only non-trace optimizations |

### 12.7 Immediately Actionable Items

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

**BitNet-TT Current (2026-04-17)**: **57 t/s p50 / 50.4 t/s decode avg**
(packed_ternary, batch32, Metal Trace). The original 33.1 t/s Llama target
has been exceeded by 72 % on a 2.4B model with 2-bit weights. No further
"fill the gap to 33 t/s" optimization needed — remaining work is
incremental (attention core-range split, ~0.7 ms → ~60 t/s).

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
- [x] In-place KV cache update (`update_cache_for_token_`) - Phase 1 complete
- [ ] In-place KV cache update (`paged_update_cache`) - Phase 1b blocked by TT-NN bug
- [ ] Trace capture for decode loop - blocked by TT-NN bug
- [ ] Multi-CQ (CQ0=ops, CQ1=IO)
- [ ] DRAM-sharded matmul for decode
- [ ] HEIGHT_SHARDED for attention (need to resolve num_heads=20)
- [ ] Ternary matmul kernel (Phase 2)

---

## 14. Phase 1b Results: Metal Trace Implementation (BLOCKED)

### 14.1 Implementation Summary

**Goal**: Enable Metal Trace for 2x+ decode speedup using `paged_update_cache` with 32-head padding.

**Changes Made**:
1. `attention.py`:
   - `KVCache.preallocate()`: Added `use_paged` parameter for 32-head padded cache
   - `KVCache.update_decode_paged()`: Uses `paged_update_cache` with HEIGHT_SHARDED + 32-head padding
   - `_forward_simple()`: Added `scaled_dot_product_attention_decode` path for trace-compatible decode

2. `generator.py`:
   - `_capture_decode_trace()`: Creates `pos_tensor_int32` (int32 for KV cache) and `pos_tensor` (uint32 for RoPE)
   - `_execute_decode_trace()`: Updates position tensors via `copy_host_to_device_tensor`

3. `model/bitnet.py` and `transformer.py`:
   - Pass `current_pos_tensor` (int32) and `pos_tensor` (uint32) through model layers

### 14.2 TT-NN Bug Discovered

**Critical Bug**: `ttnn.experimental.paged_update_cache()` corrupts its `update_idxs_tensor` parameter.

**Reproduction**:
- Running 30-layer transformer forward with paged decode
- Using same `pos_tensor` across layers for KV cache updates
- First call (key cache) preserves tensor value
- Second call (value cache) **corrupts** the tensor
- Corruption is cumulative - manifests around layer 14-27 depending on sync timing

**Evidence**:
```
Layer 27: pos_tensor_int32 before paged_update_cache(value): 6
Layer 27: pos_tensor_int32 after paged_update_cache(value): 1068416405 (garbage)
```

### 14.3 Workaround Applied

Modified `update_decode_paged()` to create fresh position tensors for each cache update:

```python
def update_decode_paged(self, key_states, value_states, current_pos, current_pos_tensor, num_kv_groups=1):
    # ... setup code ...
    
    # Fresh tensor for key cache (don't use caller's tensor)
    key_pos_tensor = ttnn.from_torch(
        torch.tensor([current_pos], dtype=torch.int32),
        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
        device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.experimental.paged_update_cache(self.key_cache, key_sharded, update_idxs_tensor=key_pos_tensor)
    ttnn.deallocate(key_pos_tensor)
    
    # Fresh tensor for value cache
    value_pos_tensor = ttnn.from_torch(...)
    ttnn.experimental.paged_update_cache(self.value_cache, value_sharded, update_idxs_tensor=value_pos_tensor)
    ttnn.deallocate(value_pos_tensor)
```

**Result**: Protects caller's tensor but **breaks Metal Trace compatibility** (Trace requires reusing same tensor across iterations).

### 14.4 Current Status

| Path | Status | Performance | Notes |
|------|--------|-------------|-------|
| Non-paged decode (`update_decode_simple`) | ✅ Working | ~5.74 t/s | Default, stable |
| Paged decode (`update_decode_paged`) | ⚠️ Disabled | N/A | Manual transforms cause corruption |
| Metal Trace | ⏳ Pending | N/A | Requires nlp_create_qkv_heads_decode refactor |
| `scaled_dot_product_attention_decode` | ⏳ Pending | N/A | Works with proper 1BKD format |

### 14.5 Recommended Next Steps

1. **Continue using non-paged decode** (`enable_trace=False`):
   - Working, stable ~5.74 t/s
   - No immediate refactoring needed
   
2. **Phase 2 optimization** (when performance boost needed):
   - Refactor to use `nlp_create_qkv_heads_decode()` for HEAD_SHARDED tensors
   - Enable proper Trace-compatible decode path
   - Target: 15+ t/s with Metal Trace

3. **No TT-NN bug report needed**:
   - Issue was our implementation, not TT-NN
   - `nlp_create_qkv_heads_decode` → `paged_update_cache` pattern works correctly

### 14.6 Files Modified (NOT committed)

- `/src/bitnet_tt/layers/attention.py`:
  - `update_decode_paged()`: Uses fresh position tensors
  - `_forward_simple()`: `use_decode_sdpa = False` (disabled buggy path)
  
- `/src/bitnet_tt/inference/generator.py`:
  - Trace capture/execute logic (implemented but not usable)

### 14.7 API Research: paged_update_cache (2025-01-09)

**Official API Signature** (from TT-Metal source):
```python
ttnn.experimental.paged_update_cache(
    cache_tensor: ttnn.Tensor,      # [B, 1, kv_len, head_dim] - modified IN-PLACE
    input_tensor: ttnn.Tensor,      # [1, B, 1[32], head_dim] - HEIGHT_SHARDED on B cores
    update_idxs: List[int] = [],    # XOR with update_idxs_tensor
    update_idxs_tensor: ttnn.Tensor = None,  # INT32, ROW_MAJOR
    share_cache: bool = None,       # Not supported with paged cache
    page_table: ttnn.Tensor = None, # INT32 (interleaved) or UINT16 (sharded)
    batch_offset: int = 0,          # Must be 0
)
```

**`update_idxs_tensor` Requirements**:
- Dtype: `ttnn.int32` (strict)
- Layout: `ROW_MAJOR_LAYOUT` (strict)
- Memory: DRAM-INTERLEAVED or L1-HEIGHT_SHARDED
- Shape: `[batch_size]` for DRAM, `[num_cores, batch_size]` for L1-sharded
- **API Documentation says it's READ-ONLY** (not supposed to be modified)

**Comparison with tt-transformers Pattern**:

| Aspect | tt-transformers | BitNet-TT |
|--------|-----------------|-----------|
| Tensor creation | `ttnn.from_torch(..., device=None)` + `copy_host_to_device_tensor` | `ttnn.from_torch(..., device=device)` |
| Memory config | Default (no explicit) | `DRAM_MEMORY_CONFIG` |
| Tensor reuse | Reused + `ttnn.plus_one()` for increment | Single tensor reused across layers |
| Protection | `skip_negative_entries=True` + `torch.maximum(pos, 0)` | None |

**Key Difference Found**:
- tt-transformers uses `device=None` initially, then `copy_host_to_device_tensor()`
- BitNet-TT creates tensor directly on device with `device=device`
- This may affect buffer allocation/aliasing behavior

**Known Blackhole Issue**:
- GitHub Issue #16674: `ttnn.experimental.paged_update_cache` consistently hanging on Blackhole
- Our p150a is Blackhole hardware - may be related

### 14.8 Hypothesis Test Results (2025-01-09)

| Hypothesis | Test | Result | Interpretation |
|------------|------|--------|----------------|
| H1: Async issue | Sync after every op | ❌ Layer 27 corrupt | NOT async issue |
| H2: Memory aliasing | 32-element buffer | ⛔ API rejects | API requires buffer size == batch size |
| H3: DRAM-specific | L1_MEMORY_CONFIG | ⛔ API rejects | API requires DRAM for INTERLEAVED |
| H4: paged_update_cache internal | 60 isolated calls | ✅ All OK | NOT paged_update_cache alone |
| H5: Other model ops | Model without paged | ✅ 30 layers OK | Confirms paged is the cause |

### 14.9 Root Cause Analysis (CORRECTED 2025-01-09)

**IMPORTANT CORRECTION**: This is **NOT a TT-NN bug**. The corruption was caused by our **improper tensor transformation sequence** before calling `paged_update_cache`.

**The Problem - Our Manual Transformations**:
```python
# BAD - BitNet-TT pattern (causes corruption)
key_rm = ttnn.to_layout(key_states, ttnn.ROW_MAJOR_LAYOUT)
key_1bkd = ttnn.permute(key_rm, (2, 0, 1, 3))  # BKSD -> 1BKD
key_1bkd = ttnn.to_layout(key_1bkd, ttnn.TILE_LAYOUT)
key_padded = ttnn.pad(key_1bkd, [(0, 0), (0, 0), (0, pad_amount), (0, 0)], 0.0)  # Pad to 32 heads
key_sharded = ttnn.to_memory_config(key_padded, shard_config)  # Manual sharding
ttnn.experimental.paged_update_cache(cache.key_cache, key_sharded, update_idxs_tensor=pos_tensor)
# → pos_tensor gets corrupted after ~27 layers!
```

**The Solution - tt-transformers Pattern**:
```python
# GOOD - tt-transformers pattern (no corruption)
qkv_fused = ttnn.concat([q, k, v], dim=-1)
qkv_fused = ttnn.reshape(qkv_fused, (1, 1, batch, qkv_dim))
q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
    qkv_fused, num_heads=20, num_kv_heads=5,
    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
)
# nlp_create_qkv_heads_decode outputs correctly sharded tensors
ttnn.experimental.paged_update_cache(cache.key_cache, k_heads, update_idxs_tensor=pos_tensor)
# → pos_tensor remains valid!
```

**Verification Test Results**:
| Pattern | Result | Interpretation |
|---------|--------|----------------|
| tt-transformers pattern (no model, 60 calls) | ✅ All passed | nlp_create_qkv_heads_decode + paged_update_cache works |
| tt-transformers pattern (with model, 30 layers) | ✅ All passed | Full integration works |
| Our manual transforms (with model, 30 layers) | ❌ Layer 27 corrupt | Manual reshape/transpose/permute/pad causes issue |

**Key Insight**: The `nlp_create_qkv_heads_decode` function handles internal tensor transformations and sharding correctly. Our manual approach (reshape → transpose → permute → pad → shard) somehow causes memory corruption when combined with `paged_update_cache`.

### 14.10 Current State & Path Forward

**Current Working Path** (default, `enable_trace=False`):
- Uses `update_decode_simple()` with `kv_cache.update_cache_for_token_()`
- Works correctly at ~5.74 t/s
- No corruption issues
- NOT Trace-compatible (uses Python int for position)

**Trace Path** (`enable_trace=True`):
- Uses `_forward_decode_optimized()` with `paged_update_cache`
- Requires HEIGHT_SHARDED input for K/V tensors
- Status: Implementation complete, awaiting hardware test

### 14.11 paged_update_cache Sharding Fix (2025-01-09)

**Problem**: `paged_update_cache` requires HEIGHT_SHARDED input, but after manual RoPE the tensors are interleaved.

**Solution**: Re-shard K/V after RoPE transformation:

```python
# After RoPE, tensors are in interleaved L1 memory
k_heads_1bkd = ttnn.permute(k_bksd, (2, 0, 1, 3))  # Back to 1BKD format

# Create HEIGHT_SHARDED config for paged_update_cache
kv_shard_config = ttnn.create_sharded_memory_config(
    shape=(32, self.head_dim),      # TILE_SIZE x head_dim
    core_grid=ttnn.CoreGrid(y=1, x=1),  # Single core for batch=1
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

# Convert to HEIGHT_SHARDED
k_sharded = ttnn.to_memory_config(k_heads_1bkd, kv_shard_config)
v_sharded = ttnn.to_memory_config(v_heads_1bkd, kv_shard_config)

# Now paged_update_cache accepts the sharded tensors
ttnn.experimental.paged_update_cache(cache.key_cache, k_sharded, update_idxs_tensor=pos_tensor)
ttnn.experimental.paged_update_cache(cache.value_cache, v_sharded, update_idxs_tensor=pos_tensor)
```

**Key Constraints**:
| Constraint | Value | Reason |
|------------|-------|--------|
| Shard shape | `(32, head_dim)` | TILE_SIZE requirement |
| Core grid | `CoreGrid(y=1, x=1)` | Single core for batch=1 |
| Strategy | `HEIGHT` | Required by paged_update_cache |
| Orientation | `ROW_MAJOR` | Required by paged_update_cache |

**tt-transformers Pattern Comparison**:
| Aspect | tt-transformers | BitNet-TT |
|--------|-----------------|-----------|
| Batch size | ≥32 | 1 |
| Core grid | `CoreGrid(y=4, x=8)` = 32 cores | `CoreGrid(y=1, x=1)` = 1 core |
| RoPE | `rotary_embedding_llama` (preserves sharding) | Manual RoPE (requires re-sharding) |
| Sharding after RoPE | Not needed | Required via `to_memory_config` |

### 14.12 Metal Trace Limitation for batch=1 (2025-01-09)

**CONFIRMED LIMITATION**: Metal Trace cannot work for batch=1 due to TT-NN sharding requirements.

**Root Cause Chain**:
1. Metal Trace requires NO buffer allocations during trace execution
2. `rotary_embedding_llama` is the only RoPE op that doesn't allocate (preserves sharding)
3. `rotary_embedding_llama` REQUIRES HEIGHT_SHARDED inputs
4. `nlp_create_qkv_heads_decode` with `memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG` doesn't produce HEIGHT_SHARDED outputs for batch=1
5. Manual RoPE requires permute/concat ops which ALLOCATE buffers
6. **Conclusion**: For batch=1, trace-compatible decode is impossible with current TT-NN

**Error When Attempting**:
```
TT_FATAL: Sharded inputs for RoPE must be HEIGHT_SHARDED.
```

**Current Status**:
| Mode | Status | Speed |
|------|--------|-------|
| NO TRACE (batch=1) | ✅ Working | ~10.4 t/s |
| WITH TRACE (batch=1) | ⚠️ Runs but corrupted output | ~19-22 t/s |
| WITH TRACE (batch≥32) | ✅ Supported by TT | 15+ t/s |

**Trace Experiment Results (2025-01-14)**:
- Trace captures and executes successfully, achieving 2x speedup
- However, allocations during trace execution corrupt output quality
- Tested approaches:
  1. `update_cache_for_token_` with int index → Index frozen in trace
  2. `paged_update_cache` with tensor index → Format conversions allocate
  3. SDPA decode with Q format conversion → Allocates during permute/to_layout
- All paths require layout conversions that allocate new tensors

**Path Forward**:
- For single-user inference (batch=1): Use non-trace optimizations (see below)
- For batch inference (batch≥32): Implement batched generation to enable trace and achieve 15+ t/s
- Alternative: Wait for TT-NN to support non-sharded `rotary_embedding_llama`

### 14.13 Non-Trace Optimization Roadmap (batch=1)

Since trace is not viable for batch=1, focus on these optimizations:

| Optimization | Expected Gain | Difficulty | Status |
|--------------|---------------|------------|--------|
| DRAM-sharded matmul | ~26% | Medium | Not started |
| LoFi compute kernel (MLP) | ~3.6x for MLP ops | Low | Not started |
| Fused QKV projection | ~10-15% | Medium | Not started |
| Pre-computed attention scale | Minimal | Low | Not started |
| Async tensor transfers | ~5-10% | Medium | Not started |

**DRAM-Sharded Matmul**:
- For weight tensors > L1 capacity, use DRAM sharding
- Reduces memory transfer overhead for large projections
- Reference: tt-transformers uses this for all linear layers

**LoFi Compute Kernel**:
- Use BFP4 precision for MLP layers (gate, up, down projections)
- Theoretical 3.6x speedup over HiFi2 (BFP8)
- May have accuracy impact - needs testing

**Fused QKV Projection**:
- Combine q_proj, k_proj, v_proj into single matmul
- Reduces kernel launch overhead
- Already have `_forward_fused_qkv` implementation - needs optimization

**Target**: Achieve 15-20 t/s without trace using these optimizations.

### 14.14 TT-Metal Trace Analysis (2025-01-14)

**Deep Investigation of tt-metal source code reveals the exact trace requirements:**

#### Trace Allocation Warning Source
Location: `tt_metal/impl/allocator/allocator.cpp:102-105`
```cpp
if (allocations_unsafe_ and not warning_generated) {
    log_warning(tt::LogMetal,
        "Allocating device buffers is unsafe due to the existence of an active trace...");
}
```
- `allocations_unsafe_` is set TRUE in `mesh_device.cpp:1009` after `end_mesh_trace()`
- Any buffer allocation after trace capture triggers this warning

#### How TT-Transformers Avoids Allocations

Looking at `models/tt_transformers/tt/attention.py:450-493`:
1. **Uses `nlp_create_qkv_heads_decode`** with `memory_config=CREATE_QKV_DECODE_SHARD`
2. **Uses `rotary_embedding_llama`** which preserves HEIGHT_SHARDED state
3. **Uses `paged_update_cache`** directly on sharded tensors (no format conversion needed!)

The key is **HEIGHT_SHARDED throughout** - no `to_layout`, `permute`, or `pad` needed.

#### HEIGHT_SHARDED Requirements (Critical)
From `models/tt_transformers/tt/model_config.py:944-954`:
```python
self.model_config["CREATE_QKV_DECODE_SHARD"] = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, self.head_dim),  # TILE_SIZE = 32!
    core_grid=ttnn.CoreGrid(y=4, x=8),      # 32 cores
    strategy=ttnn.ShardStrategy.HEIGHT,
    ...
)
```
- Shard shape is `(32, head_dim)` 
- Requires 32 elements in height dimension to fill grid
- For decode: height = batch × num_heads

#### TT-Transformers Batch Padding Strategy
From `models/tt_transformers/tt/model_config.py:1506-1509`:
```python
if batch < 32:
    zeros = torch.zeros(1, seq_len, 32, self.dim)
    zeros[:, :, :batch, :] = x
    x = zeros
```
**Small batches are padded to 32!** This is how they enable HEIGHT_SHARDED for batch<32.

#### BitNet vs TT-Transformers Comparison

| Aspect | TT-Transformers | BitNet |
|--------|-----------------|--------|
| Batch size | Padded to 32 | 1 (no padding) |
| num_heads | 32+ (divisible by 32) | 20 (not divisible) |
| num_kv_heads | 8+ (divisible by 32) | 5 (not divisible) |
| QKV creation | nlp_create_qkv_heads_decode | Manual projection |
| RoPE | rotary_embedding_llama (sharded) | Manual (allocates) |
| Format conversions | None needed | to_layout, permute, pad |
| Trace compatible | ✅ Yes | ❌ No |

#### Root Cause Summary

BitNet cannot use trace because:
1. **Head counts (20, 5) not divisible by 32** → Cannot use HEIGHT_SHARDED ops
2. **Batch=1 without padding** → Cannot fill 32-core grid
3. **No sharded RoPE** → Manual ops require format conversion → allocations

#### Potential Solutions

1. **Pad batch to 32** (like TT-transformers)
   - Pros: Enables HEIGHT_SHARDED, trace works
   - Cons: 32x memory for KV cache, 32x compute waste

2. **Pad heads to 32**
   - Already doing this for cache (32 heads)
   - But QKV projection and RoPE still need sharding

3. **Wait for non-sharded rotary_embedding_llama**
   - TT-NN feature request

4. **Use batch≥32 for production**
   - Best solution for multi-user serving

#### Relevant GitHub Issues (tenstorrent/tt-metal)

| Issue | Title | Relevance |
|-------|-------|-----------|
| [#35498](https://github.com/tenstorrent/tt-metal/issues/35498) | Multi-query attention for N<32 tokens | **Exact problem** - need MQA without padding to 32 |
| [#35100](https://github.com/tenstorrent/tt-metal/issues/35100) | Garbage outputs for batch != 1 with trace | Trace memory corruption issue |
| [#30418](https://github.com/tenstorrent/tt-metal/issues/30418) | nlp_create_qkv_heads wrong shapes for head_dim < 32 | Integer division bug in tile calculation |
| [#24926](https://github.com/tenstorrent/tt-metal/issues/24926) | rotary_embedding_llama failing for non-standard shapes | RoPE op limitations |
| [#35259](https://github.com/tenstorrent/tt-metal/issues/35259) | trace_region_size allocation mismatch | Trace memory allocation bug |

**Key Quote from #35498:**
> "The current 'fast' attention kernels are tile-based... This enforces chunk sizes in 32×32 tiles... getting correct reuse within one call is easiest only at 32-token granularity."

This confirms our finding: TT-NN's tile-based design fundamentally requires 32-alignment for efficient trace-compatible operations.

### 14.15 Trace Investigation Conclusion (2025-01-15)

#### Final Trace Test Results

| Mode | Speed | Output Quality | Status |
|------|-------|----------------|--------|
| No Trace (baseline) | ~10.4 t/s | ✅ Correct | **Working** |
| With Trace | ~18-22 t/s | ❌ Garbage (memory corruption) | **Blocked** |

#### Confirmed Root Cause

Trace mode is **fundamentally incompatible** with BitNet batch=1 due to TT-NN architecture:

1. **BitNet dimensions not 32-aligned**:
   - `num_heads = 20` (not divisible by 32)
   - `num_kv_heads = 5` (not divisible by 32)
   - `batch = 1` (cannot fill 32-core grid)

2. **TT-NN sharded ops require 32-alignment**:
   - `nlp_create_qkv_heads_decode` → requires `batch × heads >= 32`
   - `rotary_embedding_llama` → requires HEIGHT_SHARDED input
   - `paged_update_cache` → requires HEIGHT_SHARDED input

3. **Manual format conversions allocate during trace**:
   - `to_layout()` → allocates new buffer
   - `permute()` → allocates new buffer
   - `pad()` → allocates new buffer
   - Allocations during trace execution corrupt output

#### Implementation Completed (Not Usable)

- `KVCache.update_decode_trace()` in `attention.py:299-351`
- `TextGenerator._capture_decode_trace()` in `generator.py:347-423`
- `TextGenerator._execute_decode_trace()` in `generator.py:425+`

These work mechanically but produce garbage output due to allocation corruption.

#### Promising Future: PR #35111

TT-Metal PR [#35111](https://github.com/tenstorrent/tt-metal/pull/35111) introduces fused ops that claim batch=1 support:
- `RotaryEmbeddingLlamaFusedQKDeviceOperation`
- `PagedFusedUpdateCacheDeviceOperation`
- **Status**: OPEN (not merged as of 2025-01-15)
- **Branch**: `aling/fused-pcu-ttt`
- **Claimed speedup**: 1.85x (46% reduction in kernel time)

#### Decision: Focus on Non-Trace Optimizations

Since trace requires TT-NN changes beyond our control, proceeding with non-trace optimizations:

| Optimization | Expected Gain | Complexity |
|--------------|---------------|------------|
| DRAM-sharded matmul | ~26% | Medium |
| LoFi compute kernel (MLP) | Up to 3.6x for MLP | Low |
| Fused QKV projection | ~10-15% | Medium |

**Target**: Achieve 15-20 t/s without trace.

---

## 15. Non-Trace Optimization Implementation (2025-01-15)

### 15.1 Optimization Roadmap

| Priority | Optimization | Expected Gain | Status |
|----------|--------------|---------------|--------|
| 1 | DRAM-sharded matmul for decode | ~26% | Not started |
| 2 | LoFi compute kernel for MLP | ~50-100% for MLP ops | Not started |
| 3 | Fused QKV projection | ~10-15% | Not started |

### 15.2 DRAM-Sharded Matmul

**Concept**: For large weight matrices that don't fit in L1, use DRAM sharding to reduce memory transfer overhead.

**Reference**: tt-transformers uses `DRAM_MEMCFG` for all linear layers during decode.

**Implementation Plan**:
```python
# Current (interleaved)
output = ttnn.linear(x, weight)

# Optimized (DRAM-sharded)
weight_dram_sharded = ttnn.to_memory_config(weight, ttnn.DRAM_MEMORY_CONFIG)
output = ttnn.linear(x, weight_dram_sharded, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

### 15.3 LoFi Compute Kernel for MLP

**Concept**: Use BFP4 precision for MLP layers where accuracy tolerance is higher.

**Fidelity Comparison**:
| Fidelity | Precision | Relative Speed | Use Case |
|----------|-----------|----------------|----------|
| HiFi4 | BF16 | 1.0x | Prefill, accuracy-critical |
| HiFi2 | BFP8 | ~2.0x | Current decode (all layers) |
| LoFi | BFP4 | ~3.6x | MLP layers (experimental) |

**Implementation Plan**:
```python
from bitnet_tt.config import get_compute_kernel_config

# MLP forward with LoFi
lofi_config = get_compute_kernel_config("lofi")
gate_output = ttnn.linear(x, w_gate, compute_kernel_config=lofi_config)
up_output = ttnn.linear(x, w_up, compute_kernel_config=lofi_config)
```

### 15.4 Fused QKV Projection

**Concept**: Combine Q, K, V projections into single matmul to reduce kernel launch overhead.

**Status**: Already implemented in `attention.py` (lines 924-941) - `qkv_fused_weight` is created during `load_weights()`.

**Analysis** (2025-01-15):
- Fused QKV saves 2 matmuls per layer (3→1)
- Per-matmul time: 0.096ms
- Savings: 2 × 0.096ms × 30 layers = **5.76ms** (6% of 95ms decode)
- However, requires significant code changes to use the fused weight
- The optimized decode path (`_forward_decode_optimized`) is disabled due to HEIGHT_SHARDED constraints

**Decision**: SKIP - savings insufficient to justify complexity. The fused weight is prepared but not actively used.

**Current Implementation** (used, 3 separate matmuls):
```python
q = ttnn.linear(hidden, w_q)
k = ttnn.linear(hidden, w_k)
v = ttnn.linear(hidden, w_v)
```

**Prepared but unused** (1 fused matmul):
```python
qkv = ttnn.matmul(hidden, self.qkv_fused_weight)  # Already created in load_weights
# Splitting code in _forward_decode_optimized is disabled
```

---
This document is a core summary with duplicates/failure logs removed. When adding new experiments, record successful configurations in detail, leave only one-line cause for failures.

---

### 15.5 LoFi MLP Test Results (2025-01-15)

| Mode | Speed | Output Quality |
|------|-------|----------------|
| HiFi2 (baseline) | 9.22 t/s | ✅ Coherent |
| LoFi MLP | 9.11 t/s | ✅ Coherent |

**Conclusion**: LoFi provides **NO improvement**. Compute is NOT the bottleneck.

### 15.6 Operation Profiling Results (2025-01-15)

#### Microbenchmark (isolated operations on P150a)

| Operation | Time | Notes |
|-----------|------|-------|
| Matmul (2560×6912) | 0.096ms | Very fast |
| RMSNorm (2560) | 0.032ms | |
| ReLU (6912) | 0.029ms | |
| Multiply (6912) | 0.046ms | |
| Add (2560) | 0.045ms | |
| to_layout (TILE→RM) | 0.030ms | |
| to_layout (RM→TILE) | 0.035ms | |
| concat (KV cache) | 0.114ms | **Slow** |
| repeat_interleave (GQA) | 0.107ms | **Slow** |
| softmax | 0.076ms | |

**Estimated per-layer (isolated)**: ~1.84ms
**Estimated 30 layers**: ~55ms → **18 t/s theoretical**

#### Actual Decode Step

| Metric | Value |
|--------|-------|
| Actual decode time | 92-96ms |
| Actual speed | ~10 t/s |
| Theoretical vs actual | 55ms vs 95ms (**1.7× gap**) |

**Gap Analysis (~40ms unknown overhead)**:
- Python dispatch overhead (loop, function calls)
- `ttnn.deallocate()` calls (many per layer)
- Hidden layout conversions in attention
- Memory allocation/deallocation per operation

### 15.7 Optimization Priority (Revised)

Based on profiling, **compute optimizations (LoFi, DRAM-sharded) will NOT help significantly**.

The bottleneck is **dispatch overhead**, which requires **Metal Trace** to eliminate.

| Optimization | Expected Gain | Reality |
|--------------|---------------|---------|
| LoFi compute kernel | ~3.6x for MLP | **0%** (tested) |
| DRAM-sharded matmul | ~26% | Unlikely (matmul already fast) |
| Fused QKV | ~10-15% | Small (3→1 kernel launches) |
| **Metal Trace** | **2-3×** | **Blocked for batch=1** |

### 15.8 Conclusion

For batch=1 inference without trace, **~10 t/s is near the limit** with current architecture.

To achieve 30+ t/s, need one of:
1. **Wait for TT-NN PR #35111** (fused ops for batch=1)
2. **Implement batch≥32** (enables trace + sharded ops)
3. **Wait for TT-NN improvements** to batch=1 decode path

---

## 16. Batch-32 Padding Implementation (2025-01-15)

### 16.1 Key Discovery: Batch-32 Enables HEIGHT_SHARDED Operations

Following the tt-transformers pattern, padding batch to 32 unlocks trace-compatible decode:

| Component | Batch=1 | Batch=32 (Padded) |
|-----------|---------|-------------------|
| nlp_create_qkv_heads_decode | Works but not sharded | HEIGHT_SHARDED output |
| rotary_embedding_llama | FAILS (needs HEIGHT_SHARDED) | Works with sharded inputs |
| paged_update_cache | Needs format conversion | Works with sharded inputs |
| Metal Trace | BLOCKED (allocations) | ENABLED (no allocations) |

### 16.2 Implementation Pattern

From `~/tt-metal/models/tt_transformers/tt/model_config.py:1506-1509`:
```python
if batch < 32:
    zeros = torch.zeros(1, seq_len, 32, self.dim)
    zeros[:, :, :batch, :] = x
    x = zeros
```

### 16.3 Required Tensor Formats

All inputs to RoPE decode must be HEIGHT_SHARDED:

| Tensor | Shape | Memory Config |
|--------|-------|---------------|
| Q heads | [1, 32, 20, 128] | HEIGHT_SHARDED (32 cores, 32×128 per core) |
| K heads | [1, 32, 5, 128] | HEIGHT_SHARDED |
| cos | [1, 32, 1, 128] | HEIGHT_SHARDED |
| sin | [1, 32, 1, 128] | HEIGHT_SHARDED |
| trans_mat | [1, 1, 1024, 32] | HEIGHT_SHARDED |

### 16.4 Sharding Configuration

```python
batch_grid = ttnn.CoreGrid(y=4, x=8)  # 32 cores

cos_sin_mem_config = ttnn.create_sharded_memory_config(
    shape=(TILE_SIZE, HEAD_DIM),  # (32, 128)
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

trans_mat_mem_config = ttnn.create_sharded_memory_config(
    shape=(TILE_SIZE, TILE_SIZE),  # (32, 32)
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    ...
)
```

### 16.5 Cos/Sin Creation (from tt-transformers)

```python
# rot_idxs: [1, batch] containing position indices
rot_idxs = ttnn.from_torch(
    torch.full((1, BATCH_SIZE), position, dtype=torch.int32),
    dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
)

# Embedding lookup from pre-computed cache
cos = ttnn.embedding(rot_idxs, cos_matrix, layout=ttnn.TILE_LAYOUT)  # [1, batch, head_dim]
sin = ttnn.embedding(rot_idxs, sin_matrix, layout=ttnn.TILE_LAYOUT)

# Reshape and shard
cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, batch, head_dim]
cos = ttnn.transpose(cos, 1, 2)  # [1, batch, 1, head_dim]
cos = ttnn.interleaved_to_sharded(cos, cos_sin_mem_config)  # HEIGHT_SHARDED
```

### 16.6 Benchmark Results (2025-01-15)

| Operation | Time (batch=32) |
|-----------|-----------------|
| nlp_create_qkv_heads_decode | ~0.1ms |
| rotary_embedding_llama (Q) | 0.089ms |
| rotary_embedding_llama (K) | 0.089ms |
| **Total RoPE (Q+K) × 30 layers** | **5.3ms** |

Compare to batch=1 (no sharding):
- Per-layer overhead: ~3ms (including layout conversions)
- 30 layers: ~90ms

**Batch-32 with sharding is ~17× faster for RoPE operations!**

### 16.7 Next Steps

1. **Implement full batch-32 decode path** in `generator.py`:
   - Pad input tokens to batch=32
   - Use fused QKV + nlp_create_qkv_heads_decode
   - Shard cos/sin per tt-transformers pattern
   - Use paged_update_cache for KV cache (already HEIGHT_SHARDED compatible)
   - Use scaled_dot_product_attention_decode

2. **Enable Metal Trace**:
   - With all ops HEIGHT_SHARDED, no allocations during trace
   - Expected 2-3× additional speedup

3. **Extract batch=1 output**:
   - After generation, slice `output[:, 0, ...]` to get real batch result
   - Other 31 batch slots are padding (ignored)

### 16.8 Trade-offs

| Aspect | Impact |
|--------|--------|
| KV Cache Memory | 32× larger (but still fits in 32GB) |
| Compute | 32× more ops (but heavily parallelized) |
| Per-token Speed | Expected 20-30 t/s (vs 10 t/s for batch=1) |
| Throughput | 32 users × speed = high throughput |
