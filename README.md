# BitNet-TT

A native TT-NN implementation for running Microsoft's **BitNet b1.58 2B-4T** model on Tenstorrent Blackhole p150a.

## Key Features

- **True 2-bit Weights**: Custom `ternary_matmul` op with BFP2_b HW unpack — **~600 MB** model size (vs 1.2 GB bfp4, 4.8 GB bf16)
- **74.1 t/s Decode** (p50, batch32 + Metal Trace + fused RoPE + multicore argmax + sharded rms_norm; peak 78.1 t/s at min latency)
- **+129% faster than bfp4** at half the storage
- **Fused RMSNorm + ternary matmul** for QKV projection — RMSNorm runs inline inside the matmul kernel
- **HuggingFace Compatible**: Direct loading of `microsoft/bitnet-b1.58-2B-4T-bf16` weights
- **In-trace Greedy Argmax (multicore)**: Argmax runs inside Metal Trace with `use_multicore=True` — vocab-dim reduction parallelised across 110 cores (1.76 → 0.08 ms/step)
- **Accuracy Preserved**: PCC 0.975 vs HuggingFace CPU reference (packed_ternary inherent limit)

## Quick Start

```bash
# On Tenstorrent p150a server
cd ~/bitnet-tt && source ~/.tenstorrent-venv/bin/activate

# Benchmark (128-token greedy decode)
TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS=BR,NC,TR,ER \
BITNET_TT_TRACE_REGION_SIZE=200000000 \
python bench_batch32.py --dtype packed_ternary --max-new 128

# Accuracy comparison vs HuggingFace
python bench_accuracy.py --dtype packed_ternary --decode-steps 32

# Interactive (non-trace, slower)
python main.py --chat
```

## Performance

Measured on Tenstorrent Blackhole p150a (110 Tensix, harvesting mask 0x2080).

### Decode Throughput (batch32 + trace + fused RoPE)

| dtype | p50 ms | p50 t/s | min ms | min t/s | decode_tps | model size |
|---|---:|---:|---:|---:|---:|---:|
| **packed_ternary** (2026-04-17 +sharded rms_norm, 64 tok) | **13.5** | **74.1** | **12.8** | **78.1** | **62.66** | **~600 MB** |
| packed_ternary (2026-04-17 +multicore argmax, 64 tok)     | 15.5 | 64.5 | 15.0 | 66.7 | 55.83 | ~600 MB |
| packed_ternary (2026-04-17 fused QKV-norm, 64 tok)        | 17.5 | 57.1 | 16.9 | 59.2 | 50.4 | ~600 MB |
| bfp4 (production baseline)                                | 31   | 32.3 | —    | —    | —    | ~1.2 GB |
| bf16                                                      | ~62  | ~16  | —    | —    | —    | ~4.8 GB |

### Performance Evolution

| Step | t/s | Key change |
|---|---:|---|
| 0. baseline (1-core scalar) | ~0.9 | — |
| 1-5. multi-core + BFP2_b HW unpack | ~21 | 108 Tensix cores |
| 8. dual-NoC RISC split | ~30 | BRISC/NOC_0 + NCRISC/NOC_1 |
| 10. activation multicast | 32.41 | rectangular mcast layout |
| 12. K-block pipelining | 34.81 | cb0/cb1 double-buffer |
| 13. cos/sin hoist | 35.38 | 60 ops/step eliminated |
| 14. QKV scale fold | 35.94 | q*k → SDPA scale, v → RMSNorm absorbed |
| 15. qkv_layout skip | 37.01 | 4D TILE stays, no roundtrip |
| 16. kv_update_prep skip | 37.52 | paged_update_cache accepts RoPE output |
| 17. **in-trace greedy argmax** | **41.39** | sample_token 6.37→0.16 ms |
| 18. **lm_head bfp4** | **43.17** | 1.81→0.75 ms DRAM BW halved |
| 19. **ROW_MAJOR argmax** | **47.68** | TILE 4.39→RM 2.26 ms |
| 20. **streaming decode O(1)** | **48.18** | tokenizer.decode per-token |
| 21. **L1 norm→matmul + SDPA L1** | **51.58** | norm/SDPA outputs in L1, DRAM round-trip eliminated |
| 22. **fused RMSNorm + ternary_matmul (QKV)** | **57.1 p50** | In-kernel RMSNorm for the QKV path + BFP2 exp init overlapped with first weight-block DMA + gamma DMA batched into one barrier + Phase 1a tile-regs merged |
| 23. **multicore argmax (trace-primed)** | **64.5 p50 / 55.83 decode_tps** | `ttnn.argmax(use_multicore=True)` parallelises vocab-dim reduction across 110 cores; isolated cost 1.99→0.076 ms (26×). First call writes are rejected inside `begin_trace_capture`, so the warmup run primes the op before capture. |
| 24. **sharded rms_norm (width-sharded multicore)** | **74.1 p50 / 62.66 decode_tps** | `ttnn.rms_norm` auto-dispatches a multicore kernel on width-sharded input; isolated cost 56→33 us/call. All 91 non-fused norms reshard input → run → convert back, via `RMSNorm.enable_sharded_fast_path()`. |

### Accuracy (vs HuggingFace CPU reference)

| Metric | packed_ternary | bfp4 |
|---|---|---|
| Prefill PCC | 0.975 | 0.993 |
| Top-1 match | True | True |
| Top-5 overlap | 80% | 90% |
| Greedy divergence | step 1 | step 3 |

Note: PCC 0.975 is the inherent limit of 2-bit weight quantization. The lm_head dtype (bfp4) adds negligible additional error (PCC 0.9992 vs bf16 lm_head).

## Architecture

### Model Structure

```text
BitNetModel (2.4B params, ~600 MB packed_ternary)
├── Embedding (128256 vocab, 2560 dim)
├── TransformerBlock × 30
│   ├── RMSNorm (input) — FUSED into QKV matmul kernel (decode only)
│   ├── MultiHeadAttention
│   │   ├── Fused QKV Projection (ternary_matmul + inline RMSNorm, 2-bit)
│   │   │   ├── QKV scales folded: q*k → SDPA, v → attn_sub_norm
│   │   │   └── In-kernel RMSNorm (REDUCE_ROW + bcast_cols + gamma bcast_rows)
│   │   ├── nlp_create_qkv_heads_decode (L1 HEIGHT_SHARDED)
│   │   ├── RoPE (rotary_embedding_llama, fused)
│   │   ├── paged_update_cache (K, V)
│   │   ├── SDPA decode (with folded q*k scale)
│   │   ├── Attention Sub-Norm (RMSNorm, BitNet-specific)
│   │   └── O Projection (ternary_matmul, 2-bit)
│   ├── Residual Add
│   ├── RMSNorm (post-attention)
│   └── FFN
│       ├── Fused Gate+Up Projection (ternary_matmul, 2-bit)
│       ├── Squared ReLU × Up (fused mul with stacked activations)
│       ├── FFN Sub-Norm (RMSNorm, BitNet-specific)
│       └── Down Projection (ternary_matmul, 2-bit)
├── RMSNorm (final)
├── LM Head (bfp4, standard matmul)
└── Argmax (in-trace, ROW_MAJOR fast path)
```

### Packed Ternary Storage (BFP2_b)

```
DRAM per tile:  256 bytes mantissa only (64 uint32)
L1 per tile:    320 bytes (256B mantissa + 64B synthesized 0x7F exponent)
Tensix HW:      UNPACKER decodes BFP2_b → bf16 on the fly
Storage:        2 bits/weight × 2.4B params ≈ 600 MB
```

### Decode Pipeline (traced)

```
Per step (17.5 ms p50, min 16.9 ms):
  copy_inputs [host→device: embed + pos + cos + sin]     0.66 ms
  execute_trace [CQ dispatch]                             0.01 ms
  ┌── trace kernel ──────────────────────────────────┐
  │  30 × (FUSED_QKV(norm+matmul) → heads → RoPE →   │
  │        cache_update → SDPA(→L1) → reshape →       │
  │        sub_norm(→L1) → o_proj → residual →        │  ~13.7 ms
  │        norm(→L1) → gate_up → relu²×up →           │
  │        sub_norm(→L1) → down_proj → residual)      │
  │  final_norm → lm_head(bfp4) → to_layout(RM)      │  +1.8 ms
  │  → argmax(RM, branchless bf16)                    │
  └──────────────────────────────────────────────────┘
  sync_device [wait for trace kernel]                    ~16.2 ms
  sample_token [read 4-byte int32 from device]            0.16 ms
```

## Project Structure

```text
bitnet-tt/
├── src/bitnet_tt/
│   ├── config.py                  # Model config + compute kernel config
│   ├── layers/
│   │   ├── attention.py           # GQA + KV-Cache + fused RoPE + QKV scale fold
│   │   ├── bitlinear.py           # Linear (ternary_matmul), RMSNorm
│   │   ├── embedding.py           # Token embedding
│   │   └── ffn.py                 # FFN (fused gate+up, stacked relu²×up)
│   ├── model/
│   │   ├── bitnet.py              # BitNetModel (bfp4 lm_head)
│   │   └── transformer.py         # TransformerBlock
│   ├── inference/
│   │   ├── generator.py           # Single-batch generator
│   │   └── generator_batch32.py   # Batch-32 + Metal Trace + in-trace argmax
│   └── utils/
│       ├── device.py              # TT-NN device management
│       ├── quantization.py        # weight_quant_ternary
│       └── weights.py             # HuggingFace weight loading
├── bench_batch32.py               # Decode throughput benchmark
├── bench_accuracy.py              # PCC + greedy match vs HuggingFace
├── profile_decode.py              # Category-level non-trace profiler
├── main.py                        # CLI entry point
├── TODO.md                        # Detailed optimization log
└── MEMO.md                        # Technical reference memo
```

## API Usage

### High-Performance Batch-32 Decode

```python
from bitnet_tt.utils.device import get_device, close_device
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.model.bitnet import create_model
from bitnet_tt.inference.generator_batch32 import Batch32Generator
from transformers import AutoTokenizer

device = get_device()
state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
model = create_model(config, device, weight_dtype="packed_ternary",
                     use_fused_rope=True)
load_weights_to_model(model, state_dict)

tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")
gen = Batch32Generator(model, tokenizer=tokenizer, enable_trace=True)

# Streaming generation (greedy decode uses in-trace argmax)
for text, stats in gen.generate_streaming(
    "The meaning of life is",
    max_new_tokens=100,
    temperature=0.0,
    top_k=1,
):
    print(text, end="", flush=True)
print(f"\n[{stats.tokens_per_second:.1f} t/s]")

close_device()
```

## Hardware Requirements

### Tenstorrent Blackhole p150a

| Spec | Value |
|---|---|
| Tensix Cores | 140 (108 active after harvesting) |
| L1 SRAM | 1.5 MB per core |
| DRAM | 32 GB GDDR6 (~512 GB/s) |

### Environment

```bash
# Required env vars
export TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS=BR,NC,TR,ER
export BITNET_TT_TRACE_REGION_SIZE=200000000

# Software
# - Source-built tt-metal with custom ternary_matmul op
# - Python 3.10, ttnn (editable install), torch 2.11+cpu
```

## Custom tt-metal Op: `ternary_matmul`

Located at `ttnn/cpp/ttnn/operations/experimental/ternary_matmul/` in our tt-metal fork.

Key design:
- **BFP2_b HW unpack**: Tensix UNPACKER decodes 2-bit mantissa + shared exponent → bf16
- **True 2-bit DRAM**: 256 B mantissa per tile, 64 B exponent synthesized in L1
- **BFP2 exp init overlapped with first weight-block DMA**: the ~4 µs of L1 stores hide behind the first K-block's ~30 µs DMA latency
- **Activation multicast**: rectangular core layout for down_proj/o_proj (5×8), L-shape for gate_up
- **RISC split**: BRISC/NOC_0 = activation reader + mcast, NCRISC/NOC_1 = weight + output + exp init
- **K-block pipelining**: `in0_block_w = Kt/2` for automatic double-buffering
- **Fused RMSNorm variant** (`fused_norm_ternary_mm_compute.cpp` + mcast norm sender/receiver): the QKV call computes `rsqrt(mean(x²)+eps) · gamma · x` inline before the matmul, using REDUCE_ROW + `mul_tiles_bcast_cols` + `mul_tiles_bcast_rows`. Gamma DMA is batched into a single barrier; Phase 1a uses `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>` to accumulate the running sum in-register (one acquire/release per Kt tile instead of two)

## References

- [BitNet b1.58 Paper](https://arxiv.org/abs/2402.17764)
- [BitNet b1.58 2B4T (HuggingFace)](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- [TT-Metal](https://github.com/tenstorrent/tt-metal)

## License

MIT License
