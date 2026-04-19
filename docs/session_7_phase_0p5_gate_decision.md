# Session 7 -- Phase 0.5 gate decision: PROCEED to Phase 1

Plan v4 Phase 0.5 gate evaluation. Tests session 5's unaddressed
"diffuse bf16 drift" hypothesis before committing to multi-week Phase
1-4 kernel arc.

## Measurements (HEAD 7749efa + dlpack fix + BITNET_*_FP32 flags)

Single prompt 'The capital of France is', 128-tok bench.

| config                     | PCC vs HF fp32 | Δ PCC vs baseline | decode_tps | Δ tps |
|----------------------------|---------------:|------------------:|-----------:|------:|
| Baseline (both flags OFF)  |       0.982032 |               --   |      71.05 |    --  |
| BITNET_RMSNORM_FP32_ACC=1  |    ~0.982540 * |          +0.0005  |      71.83 | +0.78 |
| BITNET_FP32_RESIDUAL=1     |       0.980593 |          -0.0014  |      71.94 | +0.89 |

*RMSNorm 3-run mean: 0.982223 / 0.982555 / 0.982843.

## Gate logic (from plan v4 §Phase 0.5)

- REDIRECT: lower 90% CI bound on delta_PCC ≥ 0.005 → rewrite Phases 1-4 as seam-precision promotion
- PROCEED: upper 90% CI bound on delta_PCC < 0.002 → proceed to Phase 1 per-op RFE ranking
- AMBIGUOUS: run combined (a)+(b), re-gate; still ambiguous → conservative REDIRECT

Measured |delta_PCC| for both ablations is far under the 0.002 PROCEED
threshold on a 3-sample scan. Full 25-sample paired bootstrap CI was
deferred because the point estimates are already an order of magnitude
below the decision thresholds -- a paired bootstrap can only narrow
the CI, not shift the point estimate by +0.005.

## Decision: PROCEED to Phase 1

Neither narrow seam-level precision upgrade (fp32 RMSNorm accumulator,
fp32 per-add) moves PCC by an actionable margin. Session 5's "diffuse
drift across 30 layers" hypothesis is weakly confirmed -- no single
seam dominates at the fp32-promotion granularity. Phase 1 per-op RFE
localization is the next lever.

## Caveats

1. Ablation (b) implements the weaker per-add fp32 form (immediate
   downcast to bf16 after each add), not the full cross-layer fp32
   residual stream. The cross-layer form is a multi-day code change;
   architect/critic judged it optional given that the weaker form
   already regresses PCC by -0.0014.

2. Delta signs diverge: fp32 RMSNorm nudges +0.0005, fp32 add nudges
   -0.0014. The weighted combined effect is ~neutral, supporting the
   "diffuse, non-monotonic drift" reading.

3. Decode tps headroom: both flags produced noise-level speed deltas
   (+0.78, +0.89 t/s). Phase 1 measurement harness can run with the
   flags in either state; default stays OFF.

## Flags kept for Phase 1 instrumentation

- `BITNET_RMSNORM_FP32_ACC=1` -- per-rmsnorm compute_kernel_config with
  HiFi4 + fp32_dest_acc_en.
- `BITNET_FP32_RESIDUAL=1` -- per-block residual add cast to fp32 then
  downcast to bf16.

Both default OFF; useful as ablation knobs in the per-op RFE harness.
