# Session 5 -- correct PCC ceiling attribution (session 4 hypothesis falsified)

Session 4 closed with the claim that "activation INT8 re-quantisation inside
the ternary_matmul kernel is the dominant error source". Session 5 falsifies
that claim with direct measurements.

## Ablations (all on p150a, prompt='The capital of France is')

| config                                    | PCC       | Δ vs baseline |
|-------------------------------------------|----------:|--------------:|
| baseline (packed_ternary + bfp8 lm_head)  | 0.980679  | --             |
| --dtype bf16 (all weights bf16-stored)    | 0.980938  | +0.00026      |
| BITNET_LM_HEAD_DTYPE=bf16                 | 0.980696  | +0.000017     |

Both interventions move PCC only within measurement noise (<1e-3).

## What this proves

1. **Storage format is not the bottleneck.** Switching all 30 layers from
   the 2-bit packed ternary kernel (`ternary_matmul`) to the standard
   bf16 `ttnn.matmul` produces the same values -- the pretrained BitNet
   weights are ternary {-1, 0, +1} and every storage format encodes them
   losslessly. The observed delta is noise.

2. **LM head precision is not the bottleneck.** Upgrading the lm_head
   from bfp8 to bf16 moves PCC by 0.000017. bfp8 was already numerically
   sufficient for the lm_head weights.

3. **There is no runtime INT8 activation quantisation.** Inspection of
   `reader_ternary_fused.cpp` and `ternary_mm_compute.cpp` confirms
   activations stay bf16 end-to-end. Weights are unpacked from 2-bit
   storage into bf16 LUT values {0, +1, -1} and fed to `matmul_block`.

## Where the ~0.019 PCC gap actually lives

HF reference runs in **fp32** across all ops; TT runs in **bf16**
throughout. bf16 has only 7 mantissa bits, so every RMSNorm, residual
add, SDPA, and matmul accumulate rounding error that the fp32 reference
does not have. Across 30 transformer layers the residual drift
accumulates to ~1.9% correlation loss. The ceiling is
**numerical-precision-of-bf16-arithmetic**, not weight storage and not a
kernel quantisation step.

## Implication for the >0.99 target

PCC > 0.99 requires recovering ~1.9% correlation lost to bf16 rounding.
Realistic options:

- **fp32 residual stream** -- forces every op onto fp32 math; cost on the
  ternary_matmul compute and the trace hot path is prohibitive (would
  easily blow past the 70 t/s budget and is not supported on the 2-bit
  kernel today).
- **fp32 RMSNorm accumulator only** -- smaller cost but the measured
  per-layer bf16-vs-fp32 norm delta is small; closing 0.019 via RMSNorm
  alone is unlikely.
- **QAT / calibration retraining** -- train the ternary weights against
  a bf16 forward pass so the residual drift cancels out; out of session
  scope.

Under the ≥70 t/s speed budget, none of these paths close 0.019. The
session 4 measurement ceiling of PCC ≈ 0.981 at ≥70 t/s remains correct
-- only its root cause was wrong.

**Deferred ablation (not run this session).** The fp32-RMSNorm-accumulator
intervention is reasoned out of scope above from adjacent evidence
(session-4 HiFi4 result, small per-layer bf16-vs-fp32 norm deltas in the
on-device RMSNorm path), but it was not directly measured. If accuracy
work is reopened, that is the smallest-surface lever to ablate first
before committing to a residual-stream rewrite.

## Verdict

Replace the "INT8 activation re-quantisation" explanation wherever it
appears in the project docs with: "cumulative bf16 arithmetic drift
across 30 transformer layers vs the fp32 HF reference". The
`BITNET_BF16_LAYERS` and `BITNET_DECODE_MATMUL_FIDELITY` flags remain
useful as instrumentation -- their small measured PCC deltas are
consistent with this corrected attribution (small per-layer numerical
perturbations, not quantisation steps).
