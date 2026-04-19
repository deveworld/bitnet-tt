# Sessions 8-10 -- Phase 2 cheap-lever exhaustion

Plan v4 Phase 2 goal: identify and fix the dominant tt-metal op whose
bf16 numerics diverge from PyTorch, to raise TT prefill PCC past 0.99.

## Localization result (session 8)

Per-op RFE ranking on prompt "The capital of France is" showed:

- L0.post_input_norm: RFE 0.009 (near bit-identical to HF)
- L0.post_attn_sub_norm: RFE **0.420** (local max)
- L0.post_self_attn (post o_proj): RFE 0.240
- L0.block_output: RFE 0.139

Drift enters during QKV→RoPE→SDPA→attn_sub_norm→o_proj chain in layer 0
and compounds through subsequent layers. RMSNorm itself is near
bit-identical -- drift comes from the attention math.

## Cheap levers attempted (sessions 7-10, all failed)

| Intervention | PCC delta | Notes |
|:---|---:|:---|
| fp32 RMSNorm acc (`BITNET_RMSNORM_FP32_ACC=1`) | +0.0005 | noise |
| fp32 residual add (`BITNET_FP32_RESIDUAL=1`) | −0.0014 | regression |
| SDPA HiFi4 (`BITNET_SDPA_HIFI4=1`) | −0.0005 | regression |
| Manual RoPE (`use_fused_rope=False`) | −0.0007 | regression |
| Manual SDPA (`BITNET_MANUAL_SDPA=1`) | −0.0018 | regression |

All five levers either stay within measurement noise or regress PCC.
None moves the needle toward the +0.019 gap required to cross PCC 0.99.

## Structural conclusion

tt-metal's bf16 op implementations diverge from PyTorch bf16 in a
**distributed** way -- not concentrated in one op that can be swapped
or tuned. Every op (matmul, rms_norm, softmax, transpose-reshape,
add) carries a small per-op bf16-arithmetic delta; the deltas compound
through 30 layers into the observed 0.019 gap.

Swapping one op's compute_kernel_config to HiFi4/fp32-acc does not
help because:

1. The pretrained weights are ternary -- HiFi4's extra mantissa bits
   have nothing to compute with.
2. fp32 dest accumulator is already on for the 2-bit kernel (session 5).
3. The fused SDPA kernel is already close-to-optimal bf16 attention
   math -- replacing it with ttnn primitives *worsens* the numerics
   because primitives add extra bf16 intermediate rounding events.

## Path forward (out of cheap-lever scope)

True alignment would require either:

- **Op-library-wide bf16-arithmetic audit and rewrite** to match
  PyTorch's exact reduction order / rounding semantics across
  matmul, rms_norm, softmax, transpose, repeat, add. Multi-week
  tt-metal kernel work. This is the "Phase 3-4" scope that plan v4
  hedged on.
- **Accept the practical ceiling** at PCC ≈ 0.982 vs HF fp32 / 0.981
  vs HF bf16 / 0.968 vs bitnet.cpp, and shift evaluation to task-level
  metrics (argmax match, top-k overlap, downstream benchmark scores)
  that are less sensitive to sub-per-mille logit-correlation drift.

## Current HEAD

`48b4052` -- decode_tps 71.05 t/s, p50 12.0 ms, PCC 0.982032 vs HF fp32.
All Phase 2 env flags committed, all default OFF.

## Kept env flags for future use

- `BITNET_RMSNORM_FP32_ACC` -- per-rmsnorm HiFi4 + fp32 dest acc
- `BITNET_FP32_RESIDUAL` -- per-block residual adds via fp32
- `BITNET_SDPA_HIFI4` -- prefill SDPA HiFi4 compute config
- `BITNET_MANUAL_SDPA` -- prefill SDPA via ttnn primitives
- `BITNET_LOCALIZE` + `BITNET_LOCALIZE_LAYERS` -- per-op RFE capture
- `BITNET_DECODE_MATMUL_FIDELITY` -- decode-path lm_head fidelity (session 4)
- `BITNET_BF16_LAYERS` -- per-layer weight bf16 mix (session 4)
- `BITNET_LM_HEAD_DTYPE` -- lm_head dtype override (session 5)

## Tooling added

- `scripts/pcc_localize.py` -- per-op RFE harness (HF + TT paired)
- `scripts/bench-smoke.sh` -- one-shot env gate (bench_batch32 + bench_accuracy)
- `bench_vs_bitnetcpp.py` -- TT vs bitnet.cpp CPU reference
- `bench_vs_hf_noquant.py` -- TT vs HF bf16 with ActQuant disabled
- `extract_logits.cpp` -- bitnet.cpp prefill logit dumper

All committed to origin/main, TT server synced.
