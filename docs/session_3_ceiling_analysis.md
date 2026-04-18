# Session 3 ceiling analysis (80 t/s + PCC > 0.99)

## Measured reality

Baseline at commit `f0a07af` on Tenstorrent p150a:
- decode_tps: **73.90 t/s**, p50 = **11.7 ms**, prefill PCC = **0.9807**,
  greedy match = 6/32 on the "The capital of France is" probe

Target asked for: decode_tps ≥ 80 t/s AND prefill PCC > 0.99
(i.e. +6.1 t/s AND +0.0093 PCC simultaneously).

## Experiments run this session

### (1) bf16 weights on transformer layer 0 only

Set `BITNET_BF16_LAYERS=0` and measured on p150a:

| metric | all-ternary | bf16@layer0 | Δ |
|---|---:|---:|---:|
| decode_tps | 73.90 | 72.18 | -1.72 |
| p50 ms | 11.7 | 11.9 | +0.2 |
| prefill PCC | 0.9807 | 0.9815 | +0.0008 |
| greedy match / 32 | 6 | 5 | -1 |

One bf16 layer buys +0.0008 PCC for -1.72 t/s. Extrapolating to the 0.01 PCC
gap the stretch goal requires: roughly **11–12 layers in bf16**, i.e. a
speed cost of **~20 t/s**. That puts the speed side at ~54 t/s, which
is incompatible with the ≥ 80 t/s target. Mixed precision cannot close
both gaps simultaneously on this architecture.

### (2) Device-side embed lookup (US-203)

Replaced the 160 KB/step H2D copy of the pre-built embed tensor with an
inlined `ttnn.embedding` + reshape chain inside the captured trace.

| metric | H2D embed | device-side lookup | Δ |
|---|---:|---:|---:|
| decode_tps | 73.90 | 74.21 | +0.31 |
| p50 ms | 11.7 | 11.8 | +0.1 |
| prefill PCC | 0.9807 | 0.9807 | 0 |
| greedy match / 32 | 6 | 1 | -5 |

Speed gain is real but tiny (the 160 KB H2D was already overlapping
dispatch well). The decode trajectory shifted because the embedding
weight is stored on device as fp32 while the old host tensor was
round-tripped through bf16 before H2D — one extra bf16 quantisation
step per token in the baseline path turned out to be load-bearing for
the greedy decode output. Code stays in place behind
`BITNET_DECODE_EMBED_LOOKUP` (default off) so future work can re-enable
it after casting the embed output to bf16 explicitly.

### (3) Boltz heads drop-in (US-102 carry-over from session 2)

`nlp_create_qkv_heads_boltz` expects a prefill-shape `[B, 1, S, 3·H·Nh]`
tensor and pre-transposes K; the decode path feeds `[1, 1, 32, qkv_dim]`
to `nlp_create_qkv_heads_decode` and consumes the sharded output
directly. Not a viable drop-in — any reshape to match the boltz shape
cancels the kernel savings.

## Why the stretch target cannot be hit on this architecture

Two independent reasons:

1. **Speed ceiling.** Every Python-level lever has been tuned (split
   lm_head, sharded rms_norm, multicore argmax, cos/sin device lookup,
   nlp_concat_heads_decode, fused-QKV-norm). The measured p50 is
   ~11.7 ms with the trace kernel consuming ~10.5 ms of that — the
   remaining ~1.2 ms is dispatcher/sync overhead. +6 t/s would need
   trace-kernel compute to drop below ~10 ms, which requires tt-metal
   kernel changes (the fused-norm extension for o_proj / gate_up /
   down_proj was already tried at commit `9e86763` and reverted at
   `289e260` after measuring a steady-state regression from 18.2 ms to
   19.5 ms — the L1 norm→matmul path already eliminated the DRAM hop
   the fused kernel was meant to save).
2. **PCC floor.** The 2-bit ternary weight quantisation is the
   dominant error source. Each bf16 layer buys ~0.0008 PCC at -1.7 t/s.
   Closing the 0.0093 gap takes ~11 layers → ~-18 t/s → the speed
   axis drops below 56 t/s. Any simultaneous 80 t/s target is
   physically incompatible.

## What is reachable

Giving up one axis at a time:

- **Speed only** (PCC stays at 0.98): realistic near-term ceiling is
  ~76–78 t/s with tt-metal kernel work on the trace hot-path. 80 t/s
  is possible only after a measured kernel breakthrough — candidates
  are a fused residual+RMSNorm op for the four non-QKV matmuls that
  does not regress in trace mode, or a dispatcher-overhead reduction
  inside tt-metal itself.
- **PCC only** (speed budget opened): bf16 attention on layers 0 + 29
  + final_norm path can probably reach ~0.983–0.985 while staying
  near 70 t/s. 0.99 requires either QAT / calibration re-training or
  dropping to ~55 t/s.

## Closing position for this session

Current HEAD (`f0a07af`) still holds the session-2 baseline of
**decode_tps 73.90 / p50 11.7 ms / PCC 0.9807** intact. All session-3
experiments that regressed accuracy (embed lookup) or did not net a
win (bf16@layer0 alone, boltz heads) have been reverted behind feature
flags so they are preserved for follow-up research without hurting
the default configuration.

The 80 t/s + PCC > 0.99 joint target is not reachable with the
current BitNet b1.58 2-bit weight architecture + Python-level
tt-metal ops; the next step would be quantisation-aware training or
a new fused kernel that breaks the 10 ms trace-kernel floor, both of
which are out of scope for a single optimisation session.
