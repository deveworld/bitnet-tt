# BitNet-TT — Current Status & Roadmap

> Consolidates sessions 3–13 findings + the Phase K (tt-metal kernel
> alignment) roadmap into a single reference. Updated 2026-04-19.

---

## 1. Current HEAD snapshot

| Property | Value |
|:---|:---|
| Commit | `82d551a` |
| Device | Tenstorrent Blackhole p150a (`ssh TT` → `e09cs04`) |
| Bench | `python3 bench_batch32.py --dtype packed_ternary --max-new 128` |
| decode_tps | **74.28 t/s** |
| p50 | **11.7 ms** |
| p90 | 12.3 ms |
| min | 10.7 ms |
| PCC vs HF fp32 | **0.9820** |
| PCC vs HF bf16 | 0.9813 |
| PCC vs HF bf16 (no ActQuant) | 0.9801 |
| PCC vs bitnet.cpp (INT8×INT2) | 0.9679 |
| argmax match | True (prompt `"The capital of France is"` → ` Paris`) |
| greedy 16-tok match vs HF | 6/16 (37.5%) |

---

## 2. Two-axis summary

### Speed axis: shipped

Session 11+12 recovered the dlpack regression (71.05 → **74.28 t/s**, +3.23)
by replacing 4 uint32 argmax readout sites:

- `to_layout(TILE) + typecast(int32) + to_torch` = 0.282 ms/call
- `ttnn.Tensor.cpu().to_list()` = 0.047 ms/call (5.5× cheaper)

Current decode_tps exceeds the pre-dlpack-regression baseline of 74.21
t/s. Speed floor is 70 t/s; headroom is **~2.6 ms / step**.

### Accuracy axis: plateaued at 0.982 vs HF fp32

Five cheap-lever attempts, all falsified:

| Intervention | Δ PCC | Verdict |
|:---|---:|:---|
| `BITNET_RMSNORM_FP32_ACC=1` (fp32 RMSNorm acc) | +0.0005 | noise |
| `BITNET_FP32_RESIDUAL=1` (fp32 per-block residual add) | −0.0014 | regression |
| `BITNET_SDPA_HIFI4=1` (HiFi4 + fp32_acc on prefill SDPA) | −0.0005 | noise/neg |
| `use_fused_rope=False` (manual RoPE) | −0.0007 | regression |
| `BITNET_MANUAL_SDPA=1` (primitive matmul + softmax + matmul) | −0.0018 | regression |
| `BITNET_LM_HEAD_DTYPE=bf16` (session 5) | +0.000017 | noise |
| `BITNET_BF16_LAYERS=0` / `=0,1` / `=29` (session 4) | +0.00016 ~ +0.00086 | noise |

Every intervention that stays at the host / env-flag / ttnn-primitive
level fails to close the 0.019 gap. No single op's compute_kernel_config
swap moves the needle.

### Why: session 5 + 6 root cause

The 0.019 gap is **not** caused by:

- Weight storage format (packed_ternary vs bfp4/bfp8/bf16) — session 5 (±0.001)
- LM head precision (bfp8 vs bf16) — session 5 (+0.000017)
- INT8 activation quantization (HF AutoBitLinear ActQuant toggle) —
  session 6 (+0.0012 noise)
- Math fidelity on matmul kernels (HiFi2 vs HiFi4) — session 4/9 (noise)

The root cause, confirmed by session 8 per-op RFE localization, is
**distributed op-level numerical drift between tt-metal bf16 and
PyTorch/CUDA bf16**. Every op (matmul, softmax, transpose, add, norm)
carries a small per-op delta; these compound through 30 layers.

### Where the drift enters (session 8)

Per-op Relative Frobenius Error (RFE), prompt `"The capital of France is"`:

| Capture point | L0 RFE | interpretation |
|:---|---:|:---|
| `block_input` | **0.000** | bf16 embedding matches HF bit-for-bit |
| `post_input_norm` | **0.009** | RMSNorm is near-identical |
| `post_self_attn` (= `post_o_proj`) | 0.240 | +23% in one attention block |
| `post_attn_sub_norm` | **0.420** | local max inside the attention chain |
| `post_post_attn_norm` | 0.177 | |
| `post_mlp` | 0.139 | |
| `block_output` | 0.139 | |

Drift enters the **fused SDPA + RoPE chain** between `post_input_norm`
and `post_self_attn`. RMSNorm itself is not the culprit.

---

## 3. Measured references and their relationships

```
HF fp32        (0.9820)  ←── our primary reference (bench_accuracy.py)
     │                     ▲
     │ bf16 drift          │ 0.9820
     ▼                     │
HF bf16        (0.9813)    │
     │                     │ 0.9813
     │ ActQuant on/off     │
     ▼                     │
HF bf16 no ActQuant (0.9801)          TT packed_ternary (bf16 runtime)
                                                │
                                                │ 0.9679
                                                │
                                       bitnet.cpp (INT8×INT2, CPU)
```

- TT matches HF bf16 **with** ActQuant better (0.9813) than without
  (0.9801), ruling out ActQuant as the gap driver.
- TT is **closer to HF than to bitnet.cpp**. bitnet.cpp's INT8×INT2
  pipeline is farther from both than they are from each other.
- No single reference is "the ground truth" — all four disagree with
  each other at the 0.01–0.03 level.

---

## 4. File inventory

### New tools added sessions 7–13

| File | Purpose |
|:---|:---|
| `scripts/pcc_localize.py` | Per-op RFE harness (HF + TT paired capture + ranked table) |
| `scripts/bench-smoke.sh` | One-shot env gate (bench_batch32 + bench_accuracy) |
| `bench_vs_bitnetcpp.py` | TT vs bitnet.cpp logit comparison |
| `bench_vs_hf_noquant.py` | TT vs HF bf16 with ActQuant disabled |
| `extract_logits.cpp` | Binary that dumps bitnet.cpp prefill logits to a .bin file |

### Design + decision docs

| Doc | Covers |
|:---|:---|
| `docs/session_3_ceiling_analysis.md` | 80 t/s + PCC>0.99 infeasibility |
| `docs/session_4_pcc_ceiling.md` | PCC ceiling at PCC 0.981 ≥ 70 t/s |
| `docs/session_5_correct_attribution.md` | INT8 activation hypothesis falsified |
| `docs/session_6_int8_act.md` | ActQuant toggle evidence, env regression note |
| `docs/session_7_baseline_reconciled.md` | dlpack fix + reconciled 71.05 t/s |
| `docs/session_7_phase_0p5_gate_decision.md` | PROCEED to Phase 1 |
| `docs/session_8_localization_design.md` | Per-op capture point design |
| `docs/session_8_rfe_ranking.md` | First RFE ranked table |
| `docs/session_10_phase2_closure.md` | Cheap-lever scope exhausted |
| `docs/plan_sdpa_kernel_alignment.md` | **Next-step plan (Phase K)** |
| `.omc/plans/pcc_above_099_multisession.md` | Plan v4 (7-phase roadmap) |

### Env flags (all default OFF)

| Flag | Effect |
|:---|:---|
| `BITNET_LOCALIZE` | Enable per-op tensor capture in forward |
| `BITNET_LOCALIZE_LAYERS` | Comma-separated layer indices to capture (default 0,5,15,25,29) |
| `BITNET_BF16_LAYERS` | Per-layer bf16 weight override |
| `BITNET_LM_HEAD_DTYPE` | lm_head dtype override |
| `BITNET_DECODE_MATMUL_FIDELITY` | lm_head decode-path math fidelity |
| `BITNET_RMSNORM_FP32_ACC` | HiFi4 + fp32_dest_acc on every rms_norm call |
| `BITNET_FP32_RESIDUAL` | Cast each block's residual add through fp32 |
| `BITNET_SDPA_HIFI4` | HiFi4 + fp32_acc on the prefill SDPA call |
| `BITNET_MANUAL_SDPA` | Replace fused SDPA with ttnn primitives (matmul + softmax + matmul) |

### dlpack compat shim

`src/bitnet_tt/_ttnn_config.py:_install_torch_uint_compat_shim` aliases
`torch.uint{16,32,64}` → `torch.int{16,32,64}` and wraps `ttnn.to_torch`
to typecast uint tensors via ttnn.typecast → int32 before the torch
conversion. Required because torch 2.2.2 dlpack rejects kUInt bit
widths > 8. Loaded from `bitnet_tt/__init__.py:_install_ttnn_to_torch_uint_wrapper`.

---

## 5. Localization infrastructure (session 8+9)

`scripts/pcc_localize.py` produces a 55-cell ranked table:

- Capture points (per layer): `block_input`, `post_input_norm`,
  `post_qkv_{query,key,value}`, `post_attn_sub_norm`, `post_o_proj`,
  `post_self_attn`, `post_attn_residual`, `post_post_attn_norm`,
  `post_mlp`, `block_output`
- Captured layers: `{0, 5, 15, 25, 29}`
- TT captures: live via `BITNET_LOCALIZE=1` env flag + inline hooks in
  `transformer.py` (block-level) and `attention.py` (sub-op level)
- HF captures: `register_forward_hook` on module outputs for the same
  boundaries
- Metric: relative Frobenius error (RFE) = `||tt - hf||_F / ||hf||_F`
- Output: `docs/session_8_rfe_ranking.md` + `/tmp/bitnet_localize/*.npz`

Known gaps:

- `post_attn_residual` TT-only (no HF hook — residual-add output is
  not a module output in HF)
- Some fused-op boundaries are non-decomposable; RFE at the fused
  output is an aggregate, not a per-op signal

---

## 6. Phase K — the tt-metal kernel alignment arc

Full plan: `docs/plan_sdpa_kernel_alignment.md`.

| Phase | Scope | Expected Δ PCC | Expected Δ speed |
|:---|:---|---:|---:|
| **K1** | Unit harness + env flag plumbing (no kernel edit) | — | 0 |
| **K2** | Softmax max-reduce alignment (single global max) | +0.005 | −0.3~0.6 ms |
| **K3** | Q·Kᵀ matmul reduction alignment (fp32 accum, contig K) | +0.010 | −0.8~1.5 ms |
| **K4** | attn·V matmul reduction alignment | ≥ +0.005 | −0.5~1.0 ms |
| **K5** | 64-prompt joint validation | median ≥ 0.99 | — |
| **K6** | tt-metal upstream PR (optional) | — | — |

**Speed budget**: 2.6 ms headroom vs K2+K3+K4 ceiling 2.5–3 ms. Tight.
K3 alone > 1.5 ms forces K4 to be reconsidered.

**Files in scope**:

- `~/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp`
- `~/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`
- `~/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`

**Decision gates**:

- After K1 harness: if stock-vs-PyTorch sub-op RFE ≤ 0.05, SDPA is not
  the dominant contributor — redirect to RMSNorm / RoPE alignment.
- After K2 softmax: if Δ PCC ≈ 0, softmax is not the issue — skip to K3.
- After K3: if decode_tps drops below 72, gate K4 behind explicit
  user re-approval.

---

## 7. Evidence package (links to raw measurements)

| Question | Evidence |
|:---|:---|
| "Is INT8 activation quantisation the cause?" | `docs/session_6_int8_act.md` — toggle Δ +0.0012 |
| "Does bf16 storage matter?" | `docs/session_5_correct_attribution.md` — Δ +0.00026 |
| "Does lm_head precision matter?" | `docs/session_5_correct_attribution.md` — Δ +0.000017 |
| "Are layer 0 and layer 29 asymmetric?" | `docs/session_4_pcc_ceiling.md` — 5.4× asymmetry |
| "Which op dominates drift?" | `docs/session_8_rfe_ranking.md` — `post_attn_sub_norm` L0 RFE 0.420 |
| "Does tt-metal SDPA match its ttnn-primitive decomposition?" | `session 10 commit 61a3644` — fused is *better* than primitives |
| "How much speed headroom is there?" | Current 74.28 t/s, floor 70 → 2.6 ms |

---

## 8. Decision tree for the next Ralph session

```
Goal: maintain or raise PCC at decode_tps ≥ 70 t/s
│
├─ User greenlights Phase K arc?
│   ├─ YES → start Phase K1 (unit harness + env-flag plumbing)
│   │         — no kernel edits yet; establishes go/no-go signal
│   │
│   └─ NO  → session is closed; no further cheap levers known
│
└─ Phase K1 harness shows stock SDPA RFE ≤ 0.05 against PyTorch-F.sdpa?
    ├─ YES → SDPA is not the dominant op; re-run pcc_localize.py
    │         on finer boundaries to find the real culprit
    │         (candidates: attn_sub_norm, post_mlp, residual-add
    │         compounded drift)
    │
    └─ NO  → proceed K2 → K3 → K4, gating each on (Δ PCC ≥ 0,
              speed ≥ 70 t/s)
```

---

## 9. What a single Phase K1 Ralph session looks like

**Goal**: prove the unit harness can reproduce session 8's 0.240 RFE
for the stock kernel and report baseline sub-op RFE for Q·Kᵀ, softmax,
and attn·V in isolation.

**Concrete steps**:

1. Capture (Q, K, V) tensors at layer 0 post-RoPE from a single
   prefill via `BITNET_LOCALIZE=1 python3 scripts/pcc_localize.py`
   and save as `/tmp/sdpa_fixed_qkv.pt`.
2. Write `tests/test_sdpa_kernel_alignment.py` that:
   - Loads (Q, K, V) from the .pt file.
   - Runs them through `ttnn.transformer.scaled_dot_product_attention`.
   - Runs them through `torch.nn.functional.scaled_dot_product_attention`.
   - Reports per-head RFE and the decomposed softmax-only and matmul-
     only sub-RFEs.
3. Land the env flag `BITNET_SDPA_KERNEL_VARIANT=<tag>` plumbing
   through `_forward_prefill_with_preallocated_cache` in
   `src/bitnet_tt/layers/attention.py`. No kernel variants yet —
   the switch just passes through to the stock kernel.
4. Commit + push. Close session. Hand off to K2 with an honest
   assessment of whether the stock kernel reproduces the 0.240 signal
   or whether the drift is compound (surfaces only after multiple
   layers).

**No kernel edits in K1.** That session's deliverable is entirely
Python-side harness + one env-flag plumbing touch.

---

## 10. Quick-reference cheat sheet

```
# Current benchmark
ssh TT "cd ~/bitnet-tt && python3 bench_batch32.py --dtype packed_ternary"
# → 74.28 t/s / p50 11.7 ms / PCC 0.982

# Per-op RFE (single-prompt)
ssh TT "cd ~/bitnet-tt && BITNET_LOCALIZE=1 python3 scripts/pcc_localize.py --capture --rank"
# → docs/session_8_rfe_ranking.md

# Compare against bitnet.cpp
ssh TT "cd ~/bitnet-tt && python3 bench_vs_bitnetcpp.py --dtype packed_ternary --ref-logits /tmp/bitnet_cpp_logits.bin"

# Smoke gate
ssh TT "cd ~/bitnet-tt && bash scripts/bench-smoke.sh"

# Rebuild bitnet.cpp logit reference (if prompt changes)
ssh TT "~/bitnet.cpp/extract_logits ~/bitnet.cpp/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf 'NEW PROMPT' /tmp/bitnet_cpp_logits.bin"
```

---

## 11. Open questions (non-blocking)

- Does the Phase K arc's expected +0.015 Δ PCC actually land, or is
  the drift too diffuse? Phase K1 is the first evidence point.
- If K1 shows stock SDPA is already aligned with PyTorch per-op,
  what's the real dominant contributor? Candidates include RMSNorm
  accumulator (even though L0 post_input_norm RFE is 0.009, the
  amplifier at `post_attn_sub_norm` 0.420 suggests the normalization
  nonlinearity magnifies small earlier drift).
- Is there a speed-neutral kernel variant that closes the gap? K2's
  softmax max-reduce alignment is the cheapest candidate; if it costs
  more than 0.6 ms, alternatives are thin.
- Can we get PCC > 0.99 and keep decode_tps ≥ 70, or is the speed
  budget structural? Answered only after K3 is measured.

---

*Maintained by the session-N workflow. Update after each Ralph
session or planning round.*
