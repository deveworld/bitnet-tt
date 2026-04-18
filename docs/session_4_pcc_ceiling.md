# Session 4 — PCC ceiling at >= 70 t/s

User target: `decode_tps >= 70 AND prefill PCC > 0.99`.

Starting point (commit `6114dbf`): decode_tps = 74.25 t/s, p50 = 11.8 ms,
PCC = 0.9807. Speed already satisfies the ≥ 70 budget; the open
question is whether the remaining 4.25 t/s headroom is enough to buy
+0.0093 PCC.

## Measurement table

All runs on Tenstorrent Blackhole p150a, bench_batch32 at 128 tokens
and bench_accuracy at 32 steps against the HuggingFace CPU reference
for the "The capital of France is" probe. Default lm_head dtype is
bfp8, default cos/sin path is the in-trace lookup.

| config                    | decode_tps | Δ t/s | prefill PCC | Δ PCC   | match/32 |
|---------------------------|-----------:|------:|------------:|--------:|---------:|
| baseline                  |      74.25 |   —   | 0.980679    |   —     | 6        |
| HiFi4 decode matmul       |      73.84 | -0.41 | **0.980679**| **0.0** | 6        |
| bf16 @ layer 0            |      72.18 | -2.07 | 0.981537    | +0.00086| 5        |
| bf16 @ layer 29           |      71.42 | -2.83 | 0.980837    | +0.00016| 6        |
| bf16 @ layer 0 + layer 1  |      70.70 | -3.55 | 0.981304    | +0.00063| 3        |

## What the data says

1. **Math-fidelity lever is dead in this pipeline.** `HiFi4` on the
   decode matmul path leaves prefill PCC bit-identical to `HiFi2`
   (0.980679 in both runs). The only matmul this knob controls is the
   lm_head chunks — and the lm_head weight is already bfp8, so the
   higher math fidelity has nothing numerically new to accumulate. The
   ternary_matmul kernel has its own internal fidelity that is not
   reachable from Python; flipping it would require a tt-metal kernel
   edit.

2. **Layer asymmetry is real but the early-layer wins are tiny.**
   Layer 0 is 5.4× more effective at raising PCC than layer 29, which
   matches the intuition that quantisation error seeded at layer 0
   gets amplified through the 29 downstream residual paths. But the
   best single-layer PCC gain is only +0.00086. Closing the +0.0093
   gap at that rate takes ~11 bf16 layers, and even at the best
   measured cost (-2.07 t/s/layer) that is ≈ -23 t/s — decode drops
   below ~51 t/s, blowing through the ≥ 70 gate.

3. **Multi-layer combos do not compound.** Layer 0 + layer 1 together
   give +0.00063 PCC, *less* than layer 0 alone. The PCC shifts are
   near the measurement noise floor and the layer-to-layer
   interactions are not simply additive. No supralinear combination
   surfaced in the experiments run.

4. **Match count is decoupled from prefill PCC.** Every config keeps
   the HF argmax on the prefill probe (Top-1 match = True
   everywhere), but the per-step greedy decode drifts a different way
   under each mix. This is the expected behaviour once PCC settles
   around 0.98 — the argmax is robust but the logit distribution's
   lower bits are effectively noise.

## Interpretation

The remaining prefill PCC error is dominated by **activation
quantisation** inside the ternary_matmul kernel (per-token INT8
with max-rescale, documented in `MEMO.md` sec. 5), not by weight
storage precision. Swapping a few weight-matrices to bf16 doesn't
help because the activation is re-quantised at every projection
regardless of the weight dtype. That is why bf16 @ layer 0 only
moves PCC by a thousandth — the compute path immediately loses the
extra precision back to the INT8 activation side.

Closing the 0.0093 PCC gap therefore requires one of:

- **activation path upgrade**: replace the INT8/token rescale inside
  `ternary_matmul` (or bypass it for selected layers) — this is a
  tt-metal kernel change with unknown speed cost;
- **QAT / calibration retraining**: keep ternary weights but
  condition them against a higher-precision activation path during
  training so PCC recovers — out of scope for an optimisation session;
- **full bf16 fallback on a large contiguous layer block**, which the
  measurements above already rule out on the speed side.

## Closing position

The session's 70 t/s speed gate is already cleared by the current
HEAD (74.25 t/s). The 0.99 prefill PCC gate is **not achievable**
inside that speed budget on this architecture — the ceiling landed
at around **PCC 0.981** for any config that stays above 70 t/s.

`BITNET_BF16_LAYERS` and `BITNET_DECODE_MATMUL_FIDELITY` env flags are
left in place so future sessions (kernel-level or retraining) can
re-run the same measurement harness without any additional code.

No further changes to the default configuration — current HEAD
remains decode_tps 74.25 / p50 11.8 ms / PCC 0.9807 / argmax match
True.
