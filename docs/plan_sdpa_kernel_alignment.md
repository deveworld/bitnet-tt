# tt-metal SDPA Kernel Alignment Plan

Goal: close the ~0.019 prefill PCC gap (TT vs HF bf16) by modifying
tt-metal's prefill SDPA compute kernel to match PyTorch bf16 reduction
semantics. Speed must stay at decode_tps ≥ 70 t/s.

## Target identification (from sessions 8 + 9 evidence)

Per-op RFE localization (prompt "The capital of France is", L0):

- L0.post_input_norm: **0.009** (near bit-identical to HF)
- L0.post_attn_sub_norm: **0.420** (local max)
- L0.post_o_proj: 0.240

Drift enters the residual between `post_input_norm` and
`post_attn_sub_norm`. Chain is: QKV proj → RoPE → fused SDPA → transpose
→ reshape → attn_sub_norm. Cheap levers all tested negative:

- SDPA HiFi4 compute_kernel_config: −0.0005
- Manual RoPE: −0.0007
- Manual primitive SDPA (ttnn matmul+softmax+matmul): −0.0018
- fp32 RMSNorm accumulator: +0.0005

The fused prefill SDPA kernel is the dominant drift source and is not
addressable via host-side flags. A kernel-side rewrite is the remaining
lever.

## Kernel files in scope

- `~/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp`
- `~/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`
- `~/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`

Primary suspects inside these:

1. Softmax max-subtract + exp + normalize reduction order. PyTorch
   computes row-max → subtract → exp → sum → divide in that exact
   sequence on the full row. tt-metal tile-based softmax may accumulate
   partial maxes per tile and reduce across tiles, producing a
   slightly different max (and thus different exp inputs).
2. Q · Kᵀ matmul accumulation order. Tile-inner-K reduction on TT
   vs. warp-tree reduction on CUDA produces different bf16 rounding
   paths even at HiFi4 / fp32_acc.
3. attn · V matmul: same story.

## Phase structure

Runs over 4-6 Ralph sessions. Each session is gated on (a) speed floor
not regressed and (b) PCC delta directionally correct or neutral.

### Phase K1 -- Baseline + harness on kernel sweeps (1 session)

- Wire `BITNET_SDPA_KERNEL_VARIANT=<tag>` env through
  `_forward_prefill_with_preallocated_cache` to select between the
  stock SDPA and each variant kernel we build.
- Build a unit harness `tests/test_sdpa_kernel_alignment.py` that
  feeds a fixed (Q, K, V) triple through (i) stock SDPA and (ii)
  PyTorch `F.scaled_dot_product_attention` and reports per-head
  relative Frobenius error. Use a single layer-0 Q/K/V capture
  saved from `pcc_localize.py` as the fixed test input.
- **Acceptance**: harness reproduces the sub-op RFE 0.240 for the
  stock kernel; env knob selects a stock-equivalent variant without
  measurable delta.

### Phase K2 -- Softmax max-reduce alignment (1 session)

- Fork `sdpa.cpp` compute kernel to a `sdpa_aligned_softmax.cpp`.
- Change the max-reduce so each tile's row-max is fully reduced
  across the full K dimension *before* the exp, matching PyTorch's
  `torch.max(attn_scores, dim=-1, keepdim=True)` single global max.
- Keep accumulator fp32, math HiFi4.
- **Acceptance**: unit harness RFE drops below 0.15 (halves the
  current 0.24). End-to-end bench_accuracy PCC delta ≥ +0.005 vs
  the 0.982 baseline. decode_tps ≥ 70 t/s. If the global-max
  reduce regresses speed below 70, revert and attempt Phase K3
  before dropping further.

### Phase K3 -- Q · Kᵀ matmul reduction alignment (1 session)

- Fork the K-loop in `sdpa.cpp` to walk K in contiguous chunks of
  `head_dim` matching PyTorch's reduction dimension, with fp32
  accumulation across tiles (`pack_tile_block` held in fp32 dest
  until the final pack).
- This is invasive; expect +0.5-1.0 ms / step on prefill and a
  proportional decode-time hit because the same factory is hit on
  every layer.
- **Acceptance**: unit harness RFE drops below 0.05. End-to-end
  bench_accuracy PCC delta ≥ +0.010. decode_tps ≥ 70 t/s.

### Phase K4 -- attn · V matmul reduction alignment (1 session)

- Same treatment to the attn·V matmul in the second half of the
  SDPA kernel.
- **Acceptance**: unit harness RFE drops below 0.02. End-to-end
  bench_accuracy PCC ≥ 0.99. decode_tps ≥ 70 t/s.

### Phase K5 -- Joint validation + 64-prompt robustness (1 session)

- Run the new kernel combined on bench_vs_bitnetcpp.py and
  bench_vs_hf_noquant.py for 64 prompts.
- Min / median / p90 PCC recorded.
- **Acceptance**: median PCC ≥ 0.99 across 64 prompts; min PCC ≥ 0.97;
  decode_tps ≥ 70 on 128-tok bench.

### Phase K6 -- Upstream + cleanup (optional, 1 session)

- Prepare a tt-metal PR with the three compute kernel variants gated
  behind a per-op config flag (keeps stock kernel for non-BitNet
  users).
- Add kernel selection plumbing to ttnn's `scaled_dot_product_attention`
  Python signature.
- Replace our env-flag wrapping with the upstreamed config path.

## Speed budget tracking

Current HEAD (1345ab8):

- decode_tps 74.28 t/s / p50 11.7 ms
- Floor: 70 t/s → p50 ≤ 14.29 ms → **2.6 ms of headroom** across the
  combined Phase K2 + K3 + K4 kernel changes.

Rough per-change cost estimate:

- K2 softmax max-reduce alignment: +0.3-0.6 ms (30 layers × ~20 μs)
- K3 Q·Kᵀ reduction alignment: +0.8-1.5 ms (dominant matmul)
- K4 attn·V reduction alignment: +0.5-1.0 ms

Total estimated cost ceiling: 2.5-3 ms -- **right at the floor**. If
K3 alone blows past 1.5 ms, the remaining K4 budget is negative and
the plan must either:

(a) Accept sub-70 t/s and document the tradeoff,
(b) Skip the attn·V alignment (K4) and accept partial PCC gain, or
(c) Invest in a Phase K7 speed-recovery pass (parallelism refactor).

## Risks + mitigations

| Risk | Prob | Mitigation |
|:---|:---|:---|
| Per-phase RFE delta is smaller than estimated | High | Unit harness in K1 runs BEFORE any kernel edits; if stock→PyTorch RFE is already small for a sub-kernel, skip it |
| Kernel rewrite introduces numerical bug | High | Unit harness is the regression gate; no edits ship without it green |
| Speed regresses past 70 floor | Medium | Per-phase rollback criteria; Phase K6 holds stock kernel as fallback |
| Compilation + tt-metal build-environment issues | Medium | Check tt-metal build toolchain works before K2 kernel edit |
| Out-of-scope ops still contribute drift (RMSNorm, RoPE) | High | Even if SDPA alignment drops sub-op RFE to 0.02, full-stack PCC may only reach 0.99-floor if RMSNorm/RoPE drift stays -- prepare fallback Phase K4.5 to repeat the same treatment on RMSNorm kernel |

## Out of scope

- Decode-path SDPA (`ttnn.transformer.scaled_dot_product_attention_decode`).
  RFE localization was on the prefill path; decode correctness is
  argmax-tracked and already acceptable.
- SwiGLU / FFN kernel alignment. L15.post_mlp RFE 0.273 is second-
  ranked but after the Phase K arc, if PCC still < 0.99, this is the
  next target.
- bitnet.cpp reference parity. bitnet.cpp does INT8×INT2 with its own
  quant semantics; aligning to HF bf16 is the practical target.

## Decision points

- **After K1 harness**: if the unit harness shows stock-vs-PyTorch
  RFE is already ≤ 0.05 per-sub-op, the session 8 end-to-end 0.240
  is due to compounded drift from surrounding ops (layer norm input,
  RoPE output), not SDPA itself. Redirect the arc to those ops.
- **After K2 softmax**: if delta_PCC ≥ +0.005 and speed is intact,
  the softmax path is the dominant contributor -- prioritize K3/K4.
  If delta_PCC is near zero, softmax is not the issue -- skip to K3.
- **After K3**: largest planned speed hit. If decode_tps drops below
  72, gate K4 behind an explicit user decision.

## Effort estimate

- 4-6 Ralph sessions @ ~2-4 hours TT-side work each + review/commit.
- Total ~10-20 hours of directed tt-metal kernel work.
- Does NOT include learning-curve time on the LLK / matmul_block API
  or debugging a broken kernel. First-time tt-metal compute kernel
  authors should budget 2× for Phase K2.

## Current status

- Plan drafted. Phase K1 (harness + env-flag plumbing) is the first
  Ralph session to execute when the user greenlights the arc.
- All foundations from sessions 7-13 (per-op RFE harness, env flags,
  speed baseline reconciled) are in place to support measurement at
  each phase boundary.
