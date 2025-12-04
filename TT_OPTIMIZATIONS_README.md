# Tenstorrent tt_transformers Optimization Analysis for BitNet-TT

This directory contains a comprehensive analysis of Tenstorrent's optimization patterns from their official `tt_transformers` implementation, achieving **33 tokens/second per user** on Llama 3.1 8B.

## Files in This Analysis

### 1. **EXECUTIVE_SUMMARY.txt** (Start here!)
Quick overview of:
- All 6 key optimizations identified
- Priority ranking for implementation
- Expected performance gains per phase
- Critical code references in original

**Read time**: 10 minutes

### 2. **TT_TRANSFORMERS_ANALYSIS.md** (Deep dive)
Detailed technical breakdown:
- Trace capture architecture (2-3x speedup)
- Paged attention with KV cache (100x KV reduction)
- Memory hierarchy optimization (5-10% speedup)
- RoPE pre-upload (5% speedup)
- Split sampling trace (33x I/O reduction)
- Multi-device topology patterns
- Performance metrics and calculations

**Read time**: 30 minutes

### 3. **INTEGRATION_GUIDE.md** (Implementation roadmap)
Step-by-step guide:
- Exact code patterns to apply
- BitNet-specific integration points
- 5-phase implementation checklist
- Common pitfalls and solutions
- Validation script

**Read time**: 25 minutes

## Quick Reference: Optimization Impact

| Phase | Speedup | KV Memory | Effort | Priority |
|-------|---------|-----------|--------|----------|
| Trace Capture | 2-3x | 8.5 GB | 4h | Critical |
| Paged Attention | - | 64 MB | 8h | Critical |
| Memory Configs | 5-10% | 64 MB | 4h | High |
| RoPE Pre-upload | 5% | 64 MB | 2h | Medium |
| Split Sampling | 5% | 64 MB | 3h | Medium |

**Total Expected**: 2.6-3.1x cumulative speedup + 135x KV reduction

## Where to Start

1. **For management/overview**: Read `EXECUTIVE_SUMMARY.txt` (10 min)
2. **For technical understanding**: Read `TT_TRANSFORMERS_ANALYSIS.md` sections 1-4 (20 min)
3. **For implementation**: Use `INTEGRATION_GUIDE.md` as checklist while coding

## Source Code References

Original implementations analyzed:
- `~/tt-metal/models/tt_transformers/tt/generator.py` (1,731 lines)
- `~/tt-metal/models/tt_transformers/tt/attention.py` (942 lines)
- `~/tt-metal/models/tt_transformers/tt/rope.py` (539 lines)
- `~/tt-metal/models/tt_transformers/tt/model_config.py` (2,700+ lines)

## Key Insights Extracted

### 1. Trace Capture Pattern (Lines 161-171, 626-637)
The most critical optimization: Eliminates compilation overhead by capturing the entire decode forward+sampling in a trace that's replayed each iteration. Only changed inputs (tokens, positions) are copied.

### 2. Paged Attention (Lines 333-349)
Reduces KV cache from 8.5 GB to 64 MB by allocating in blocks instead of full sequences. Uses `ttnn.experimental.paged_update_cache` for in-place updates.

### 3. Memory Hierarchy (Model Config)
Strategic placement of tensors: L1 for reused outputs (QKV, concat heads), DRAM for large intermediates (SDPA, FFN). Precision tuning: BFP8 for weights, HIFI2 for decode, HIFI4 for prefill.

### 4. RoPE Pre-upload (Lines 400-408, 511-523)
Pre-allocate cos/sin matrices for entire max_seq_len, use embedding lookup instead of per-token computation. 100ms runtime savings for 64MB memory cost.

### 5. Split Sampling (Lines 472-473, 620-641)
Separate on-device sampling from logits computation. Reduces data transfer from 256 KB to 8 bytes per token.

## Integration Roadmap

### Phase 1: Trace Capture (Week 1)
- Modify `TextGenerator.__init__` to use dict-based trace caching
- Implement `_capture_decode_trace()` with compile + capture + cache
- Implement `_decode_with_trace()` with non-blocking execute
- Expected: 2-3x immediate speedup

### Phase 2: Paged Attention (Week 2-3)
- Create `PagedKVCache` dataclass with block-based allocation
- Replace full pre-allocation with block-based version
- Use `paged_scaled_dot_product_attention_decode`
- Expected: 100x KV reduction (enables long sequences)

### Phase 3: Memory Configs (Week 3)
- Define L1_WIDTH_SHARDED for QKV, concat heads
- Define DRAM for SDPA output, FFN intermediates
- Add BFP8 KV cache dtype
- Expected: 5-10% speedup + lower L1 pressure

### Phase 4: RoPE Pre-upload (Week 4)
- Create `RotarySetup` with pre-allocated cos/sin
- Use embedding lookup instead of per-token computation
- Expected: 5% speedup

### Phase 5: Split Sampling (Week 5, optional)
- Separate sampling into second trace
- Use on-device argmax/topk
- Expected: 33x I/O reduction

## Common Pitfalls

1. **Trace hangs**: Weights not on device before capture
2. **Page table errors**: Block indices exceed max_num_blocks
3. **L1 overflow**: Too many WIDTH_SHARDED configs
4. **RoPE shape mismatch**: Input indices not formatted correctly
5. **Sampling instability**: Different batch sizes between traces

Solutions provided in INTEGRATION_GUIDE.md

## Performance Expectations

After all optimizations:
- Decode latency: ~250-350 ms per token (from ~1000ms baseline)
- KV cache: 64 MB (from 8.5 GB)
- Cumulative speedup: 2.6-3.1x
- Note: 33 t/s requires Llama 3.1 8B; BitNet 1B-3B may exceed this

## Files Generated From Analysis

These documents were created by reading and analyzing:
- 1,731 lines of generator.py
- 942 lines of attention.py
- 539 lines of rope.py
- 2,700+ lines of model_config.py

Total analysis: 5,900+ lines of source code distilled into 3 implementation guides.

## Questions?

Refer to INTEGRATION_GUIDE.md for:
- Exact code examples for your BitNet implementation
- Per-phase checklist with specific line numbers to change
- Validation script to verify optimizations work
- Troubleshooting guide for common errors

---

Last updated: 2025-12-05
Analysis source: ~/tt-metal/models/tt_transformers/tt/
