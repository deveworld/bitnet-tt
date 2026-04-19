# Session 8 -- Per-op RFE localization design

Plan v4 Phase 1. Output: ranked table of which tt-metal op contributes
most to the 0.019 PCC gap vs HF bf16.

## Capture points (per transformer block)

To keep the harness tractable and avoid fused-op alignment pitfalls, we
capture at module-output boundaries only. Internal boundaries (post-QKV
projection, post-SDPA internal) are skipped because TT fuses QKV+RMSNorm
and uses a different SDPA tiling than PyTorch -- tensor-shape alignment
there requires the un-fused TT shadow path which is deferred.

For each captured layer:

| # | Name              | HF tensor                                     | TT tensor                       |
|---|-------------------|------------------------------------------------|---------------------------------|
| 1 | block_input       | decoder_layer input (hidden_states entering)  | transformer block hidden_in     |
| 2 | post_input_norm   | decoder_layer.input_layernorm(hidden_states)  | self.input_layernorm output    |
| 3 | post_attn_add     | after first residual add (before 2nd norm)    | after first ttnn.add            |
| 4 | post_post_attn_norm | decoder_layer.post_attention_layernorm(h)   | self.post_attention_layernorm output |
| 5 | block_output      | decoder_layer output (after 2nd residual add) | after second ttnn.add           |

5 capture points per layer × 5 layers = **25 cells**. Bonferroni at
FWER=0.05 → threshold z = Φ⁻¹(1 - 0.025/25) ≈ **2.81σ**. Tighter than
plan's 55-cell 3.32σ; still powered for 0.003-level discrimination at N=10.

## Layer selection

{0, 5, 15, 25, 29}. Matches plan v4 §Phase 1 task 3. Early + middle +
late coverage catches layer-position asymmetry (session 4: layer 0 is
5.4× layer 29's contribution).

## Fused-op boundary notes

- **Fused QKV+RMSNorm**: TT runs `ternary_matmul` with fused RMSNorm;
  HF has separate `q_proj` / `k_proj` / `v_proj` after a standalone
  RMSNorm. We capture at `post_input_norm` (pre-fused-QKV), skipping
  the internal fused boundary. Any drift attributable to the fused
  kernel lands in capture point #3 (post_attn_add) as a combined
  signal with SDPA/o_proj. Non-decomposable by design.
- **SDPA**: paged flash-attention on TT vs PyTorch scaled_dot_product
  on HF. Same fate -- contributes to post_attn_add aggregate.
- **Fused gate/up/down (SwiGLU)**: HF has separate gate_proj / up_proj
  / down_proj; TT fuses gate/up. We capture only post_post_attn_norm
  (pre-fused) and block_output (post-down-proj). Internal SwiGLU
  drift lands in block_output.

## Measurement protocol

- **Prompts (K=5)**: "The capital of France is", "Once upon a time",
  "The answer to life is", "def fib(n):", "In 2024, we"
- **Runs per prompt (N=10)**: same prompt, 10 independent forward
  passes (no dropout -- deterministic -- runs are for bootstrap variance
  estimation, not actual variation; repeated to confirm zero-variance).
  If runs are literally identical, we drop to N=1 and rely on K=5
  prompt variation for the variance estimate.
- **Metric**: RFE(tt, hf) = ||tt - hf||_F / ||hf||_F (relative
  Frobenius error). Composes through residual streams via triangle
  inequality: e(A + B) ≤ e(A) + e(B) + O(e(A)e(B)).
- **Paired bootstrap**: same prompt/seed on both sides; 1000 resamples.
- **Threshold**: 2.81σ Bonferroni at FWER=0.05 across 25 cells.

## Ranking output

`docs/session_8_rfe_ranking.md` will hold a table:

| Cell | Layer | Capture | RFE mean | RFE 90% CI | 2.81σ clear? | Op class |
|-----:|------:|:--------|---------:|:-----------|:-------------|:---------|

Grouped by op class {norm, attention+o_proj, FFN, residual}. Ranked
within class; cross-class ranking requires Jacobian caveat per plan v4.

## Out of scope for Phase 1

- Implementing un-fused TT shadow path for fused QKV+RMSNorm (deferred).
- Per-layer sweep beyond {0, 5, 15, 25, 29}.
- Actually fixing any op (that's Phase 2).
