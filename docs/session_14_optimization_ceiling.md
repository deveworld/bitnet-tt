# Session 14 — Optimization Ceiling Reference

> Consolidates Phase K (SDPA kernel alignment harness + FP32_ACC default)
> and Phase L (bfp8 KV + ttnn.split dead-ends) Ralph cycles into a single
> reference. Updated 2026-04-19, HEAD `75a65b5`.

## Calibrated baseline (bench_batch32 --dtype packed_ternary --max-new 128)

| Metric | Value |
|:---|---:|
| decode_tps | ~74.1 t/s |
| p50 | 11.6 ms |
| min | 10.7 ms |
| p90 | 12.4 ms |
| PCC vs HF fp32 | 0.982045 |
| PCC vs HF bf16 | 0.9813 |
| PCC vs bitnet.cpp | 0.9699 |
| greedy argmax ("The capital of France is" → " Paris") | match |

Variance across 5 back-to-back runs on HEAD `75a65b5`:
`73.82, 74.14, 74.30, 73.99, 73.97` t/s. Mean **74.04 t/s**, stddev
**0.17**, min-max spread **0.48**. Any single-run speed delta
< +0.5 t/s is noise, not signal. A stable +0.5 t/s requires
≥ 3 stddev = +0.51, so report kernel wins only when a new mean
from ≥ 3 runs beats the old mean by > 0.5 t/s.

## What's been tried and rejected

### SDPA kernel fork (Phase K plan K2-K4) — REDIRECTED
K1 harness measured fused SDPA RFE vs torch.F.sdpa(fp32) = 0.0257. bf16
torch rounding floor alone = 0.0025. Net TT-kernel SDPA drift = 0.023,
well under the plan's 0.05 redirect gate. The SDPA compute kernel is
already aligned with PyTorch semantics to within bf16 rounding; forking
it for softmax/QK/attnV alignment would yield diminishing returns.

### attn_sub_norm amplification hypothesis (Phase KR1/KR2) — FALSIFIED
RMSNorm kernel alignment harness on layer-0 drifted post_self_attn:
  input drift RFE 0.235 → TT rms_norm output RFE 0.220 (amp 0.93x)
The TT rms_norm kernel does NOT amplify input drift. Session 8's high
0.420 at `post_attn_sub_norm` is a comparison-order artifact (TT-output
vs HF-output where each end-state reflects the whole cumulative chain),
not kernel amplification.

### BITNET_RMSNORM_FP32_ACC default flip — LANDED (marginal)
Flipped to default=ON at commit 97cb9bc. Speed delta +0.35 t/s (within
noise). PCC delta +0.00005 vs HF, +0.002 vs bitnet.cpp. Local per-op
RFE at post_input_norm: 0.0089 → 0.0028 (-68%). Small but real local
win; end-to-end gain absorbed by downstream Q/K/V ternary matmul drift.

### bfp8 KV cache (Phase L1) — DEAD END
BITNET_KV_CACHE_DTYPE=bfp8 regresses decode_tps 74.35 → 24.57 (3×
slower) with max token latency 1005 ms. The bfp8 dtype adds a
quantize/dequantize step on every DRAM-L1 cache fetch in SDPA_decode
and paged_update_cache, swamping the bandwidth saving.

### ttnn.split replacing 2 ttnn.slice (Phase L2) — NEUTRAL
Replacing the ffn.gate_up dual-slice with one ttnn.split call measured
74.08 t/s (-0.27 vs 74.35 baseline, within noise). The 131 us/call figure
in the non-trace profile was dispatch overhead absorbed by trace replay;
no wall-clock win is available from this path.

### LoFi on decode-path RoPE (Phase O) — PURE REGRESSION
`ttnn.experimental.rotary_embedding_llama` accepts a `compute_kernel_config`.
Phase O tried passing a LoFi math_fidelity config, hypothesizing that RoPE's
cos/sin multiplication has bounded inputs and wouldn't need HiFi2 precision.
Result: decode_tps 72.93 t/s (-1.11 vs 74.04 mean, outside 0.5 noise band)
and the greedy output string diverged from baseline — likely both slower AND
less accurate. Reverted fully (env flag + call-site) since LoFi neither
picked faster code paths nor preserved PCC. The fused RoPE kernel is already
at its tuned default. Third kernel-config lever to be falsified after N.

### HiFi4 + fp32_dest_acc on ternary_matmul (Phase N) — PURE REGRESSION
`ttnn.experimental.ternary_matmul` accepts a `compute_kernel_config` but
bitlinear.py was never passing one. Phase N wired `BITNET_TERNARY_MATMUL_HIFI4=1`
to pass HiFi4 + fp32_dest_acc_en — the same config the RMSNorm path uses.
Result: decode_tps 71.69 t/s (-2.35 vs 74.04 mean), PCC vs HF fp32 0.982045
(IDENTICAL), PCC vs bitnet.cpp 0.969858 (IDENTICAL). The ternary {-1, 0, +1}
accumulation already produces exact partial sums in bf16 (the 8-bit
mantissa never truncates), so HiFi4 only adds cycles. Flag kept default OFF
so future investigation can re-test without code changes, but this closes
the "ternary_matmul kernel precision" door.

## Real drift engine

K1 + KR1/KR2 localized the 0.019 PCC gap vs HF bf16 to the Q/K/V
`ttnn.experimental.ternary_matmul` chain feeding SDPA. Sub-evidence:

- L0 post_input_norm RFE = 0.009 (clean, RMSNorm is near-bit-identical).
- L0 post_self_attn RFE = 0.235 (compounded drift from
  q_proj + k_proj + v_proj + RoPE + fused SDPA).
- Stock SDPA adds only 0.023 to whatever input drift it receives.
- Stock attn_sub_norm preserves or slightly reduces its input drift.

So the drift enters in the three ternary matmul projections and
the RoPE application, not in SDPA or RMSNorm. **Phase N falsified the
hypothesis that ternary_matmul kernel precision is the lever**: passing
HiFi4+fp32_dest_acc yielded identical PCC (0.982045 vs HF, 0.9699 vs
bitnet.cpp) with -2.35 t/s cost. Ternary accumulation is already exact
in bf16. The remaining drift is structural — per-layer ternary
quantization noise compounding across 30 layers — not a kernel fix.

## Decision tree for the next Ralph session

```
User re-greenlights kernel-fork cycle?
│
├─ YES, targeting accuracy → edit ternary_matmul compute kernel
│   (e.g., add fp32_dest_acc option to matmul_block calls in
│    ~/tt-metal/ttnn/cpp/ttnn/operations/experimental/ternary_matmul/
│    device/kernels/ternary_mm_compute.cpp). Requires tt-metal rebuild
│    (~30 min). Gate on unit harness (tests/test_sdpa_kernel_alignment.py
│    pattern, adapted to ternary_matmul) showing sub-op RFE drop before
│    committing.
│
├─ YES, targeting speed → trace-level profiling first
│   (tt-metal tracy profiler or device-side perf counter dump) to find
│   the actual trace-mode hot op. Non-trace category profiling has
│   been exhausted; all non-trace top ops are already optimized or
│   dispatch-overhead-bound.
│
└─ NO → session closed; the 74 t/s / PCC 0.982 / PCC 0.9699 vs bitnet.cpp
         ceiling is the current practical limit for the packed_ternary
         + fused-QKV-norm + trace + multicore-argmax + sharded-rmsnorm
         stack without kernel forks.
```

## Files from these cycles

- `tests/test_sdpa_kernel_alignment.py` — K1 SDPA harness + env-flag test
- `tests/test_rmsnorm_kernel_alignment.py` — KR2 RMSNorm harness
- `src/bitnet_tt/layers/attention.py` — BITNET_SDPA_KERNEL_VARIANT plumbing
  ({stock, pytorch_ref}; aligned_* reserved for future kernel-fork cycle)
- `src/bitnet_tt/layers/bitlinear.py` — BITNET_RMSNORM_FP32_ACC default ON
- `.omc/prd.json` — K/L/M cycle story closure records
- `MEMO.md` — baseline header refresh

## Commits (2026-04-19)

```
949e2ff Phase K1: SDPA kernel-variant env + unit harness
d920016 Phase K1 follow-up: RMSNorm kernel alignment harness
97cb9bc Speed+accuracy: enable BITNET_RMSNORM_FP32_ACC by default
fac7b94 MEMO: refresh baseline to 97cb9bc (74.18 t/s, PCC 0.9699 vs bitnet.cpp)
2d67946 PRD: Phase K1 session closure — all 7 stories marked passes=true
7b282d8 Deslop Phase K1 landing (post-architect)
75a65b5 PRD: Phase L cycle — bfp8 KV and ttnn.split both falsified
1323f6f Session 14: Phase K+L+M optimization-ceiling reference
032f5f6 Phase N: ternary_matmul HiFi4 env flag + finding (false lever)
43aefc4 Phase O: LoFi decode-path RoPE — pure regression
```

## Session closure — K through O verdict table

| Cycle | Lever | Verdict | Net effect on main |
|:---|:---|:---|:---|
| K1 | SDPA compute_kernel_config + harness | REDIRECTED (RFE 0.023, below gate) | Env flag + 2 test files landed |
| KR1 | BITNET_RMSNORM_FP32_ACC default flip | LANDED (marginal, in noise) | bitlinear.py default=ON |
| KR2 | RMSNorm amplifier hypothesis | FALSIFIED (amp 0.93x, not amp) | Harness landed, no code change |
| L1 | bfp8 KV cache | DEAD END (3x regression) | reverted |
| L2 | ttnn.split vs 2 ttnn.slice | NEUTRAL (within noise) | reverted |
| M | 5-run variance characterization | STDDEV 0.17, floor 0.5 t/s | doc landed |
| N | ternary_matmul HiFi4 + fp32_acc | FALSE LEVER (PCC identical, -2.35 t/s) | Env flag default OFF, helper refactor |
| O | RoPE LoFi math_fidelity | FALSE LEVER (-1.11 t/s + output divergence) | Fully reverted |

**Cumulative net production code delta**: +2 test harnesses, bitlinear.py FP32_ACC default flipped, BITNET_SDPA_KERNEL_VARIANT + BITNET_TERNARY_MATMUL_HIFI4 opt-in env flags, `_hifi4_fp32_kernel_config()` shared helper, docs refresh. Zero algorithmic change to the production trace-replay path.

**Prerequisite for any further progress**: tt-metal C++ source edit in
`~/tt-metal/ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/ternary_mm_compute.cpp`
(accuracy axis) or
`~/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp`
(speed axis, K2 softmax max-reduce alignment in the original plan).
Each attempt costs ~30 min tt-metal rebuild + risk of a broken device
state that needs PCI reset. That is the boundary for the next session
to cross; this one ends at the API/config layer boundary.
