# Plan: BitNet-TT Prefill PCC > 0.99 vs HF bf16 — Multi-Session Kernel Work (v4)

> ## Execution status — 2026-04-19 (HEAD 503c164)
>
> | Phase | Scope | Status |
> |:---|:---|:---|
> | **Phase 0** | dlpack env restore + baseline reconciliation | ✅ **DONE** (session 7) — torch.uintN shim + ttnn.to_torch wrapper; reconciled baseline 71.05 t/s / PCC 0.982. `docs/session_7_baseline_reconciled.md`. |
> | **Phase 0.5** | fp32 RMSNorm acc + fp32 residual ablations | ✅ **DONE** (session 7) — both |delta_PCC| < 0.002 → **PROCEED**. `docs/session_7_phase_0p5_gate_decision.md`. |
> | **Phase 1** | Per-op RFE localization harness | ✅ **DONE** (session 8) — `scripts/pcc_localize.py` + `BITNET_LOCALIZE` env. L0.post_input_norm RFE 0.009, L0.post_self_attn 0.240, L0.post_attn_sub_norm 0.420. Dominant drift chain = SDPA path. |
> | **Phase 2** | Top-1 op kernel alignment (cheap levers) | ⚠️ **CLOSED NEGATIVE** (sessions 9–10) — 5 cheap levers tested (fp32 RMSNorm acc, fp32 residual, SDPA HiFi4, manual RoPE, manual primitive SDPA), all ≤ |0.002| PCC delta. `docs/session_10_phase2_closure.md`. |
> | **Phase 3–4** | Multi-op alignment / speed recovery | ↻ **REPLACED by Phase K arc** — see `docs/plan_sdpa_kernel_alignment.md`. Cheap-lever exhaustion forced the pivot into direct tt-metal C++ kernel edits. |
> | **Phase 5** | 64-prompt joint validation | ⏳ **Deferred** — runs after Phase K3 at earliest. |
> | **Phase 6** | Upstream PR | ⏳ **Optional** — gated on Phase K landing a material win. |
>
> **Speed-side bonus (out-of-plan):** sessions 11–12 recovered +3.23 decode_tps via the `cpu().to_list()` argmax readout fix. Current HEAD 74.28 t/s / PCC 0.982, exceeding pre-dlpack-regression baseline.
>
> **Next step:** user greenlight → Phase K1 (unit harness + env-flag plumbing, no kernel edits yet). See `docs/plan_sdpa_kernel_alignment.md` §"Phase K1" and `docs/STATUS.md` §9 for the concrete Ralph-session sketch.
>
> ---

**Mode:** DELIBERATE consensus (RALPLAN-DR) — high-risk, multi-week, multi-session kernel rewrite
**Base commit:** d37798d (plan-of-record) — speed baseline to be RECONCILED in Phase 0 against MEMO.md-reported state (p50 17.5 ms / ~57 t/s / decode_tps ~50.4).
**Estimated scope:** 5-7 weeks across ~7 Ralph sessions (added Phase 0.5)
**Partial-progress is an acceptable outcome.** Reaching PCC 0.990 after 3 kernel rewrites is a legitimate stopping point even if the full 0.019 gap does not close.

**v4 revision summary (vs v3) — Critic ITERATE verdict, 4 MAJOR fixes:**
- **FIX 1 (paired bootstrap):** Phase 0.5 bootstrap now explicitly paired across conditions — 25 (run, prompt, seed) tuples evaluated under both ablation and HEAD with identical seed/prompt; resample per-tuple deltas, not marginals. Tightens CI by √2 and makes the 0.002/0.005 gate bands decision-relevant.
- **FIX 2 (multiple-comparisons correction):** Phase 1's 3σ bar replaced with **3.32σ (Bonferroni at FWER=0.05 across 55 cells)**. Benjamini-Hochberg at FDR=0.10 considered and rejected (binary per-op Phase-2 commitment makes Bonferroni's stricter FWER target more appropriate). Measurement-plan threshold updated to match.
- **FIX 3 (wall-clock abort):** Phase 0.5 ablation (b) gains a 5×-normal-prefill wall-clock trigger that drops samples to 3×3=9 and widens CI to 80%, with hard abort if 9-sample run exceeds 1 day.
- **FIX 4 (REDIRECT feasibility catalog):** Before committing Phases 1-4 to REDIRECT scope, a ~1-day tt-metal-feasibility catalog enumerates fp32_dest_acc_en coverage across every seam; <80% coverage triggers a hybrid Python-carry + targeted-rewrite fallback (partial reversion toward Option A).
- **Polish:** 0.002 / 0.01 thresholds explicitly labeled (within-class ranking noise floor vs cross-class actionable signal); combined (a)+(b) inherits (b)'s speed-budget exemption; task-flow gate text references CI bounds rather than point estimates.

**v3 revision summary (vs v2):**
- **REQUIRED FIX 1 — Phase 0.5 gate is now CI-aware.** Ablation measurements use 5 runs × 5 prompts = 25 samples with 90% percentile-bootstrap CIs. REDIRECT gate is `lower CI bound on delta >= 0.005`; PROCEED gate is `upper CI bound on delta < 0.002`; between-state runs the combined (a)+(b) ablation and re-gates on combined CI. Removes the 3-run-median noise-flip failure mode identified by Architect.
- **REQUIRED FIX 2 — Ablation (b) redesigned.** v2 ablation (b) cast residual to fp32 only at each residual-add and downcast immediately; this is numerically equivalent to the bf16 add (deferred rounding by one op) and did NOT test session 5's diffuse-across-layers fp32 accumulation hypothesis. v3 keeps residual in fp32 ACROSS all 30 layers and only downcasts at the final pre-lm_head boundary. Falsification section and gate cross-references updated.
- **Polish:** Phase 1 variance N=5→N=10 with a 3σ separability bar (tightened from 2σ) to survive multiple-comparisons across 55 (layer, capture-point) cells. RFE triangle-inequality caveat added: cross-class ranking (linear vs norm vs softmax/attention) requires Jacobian-aware reweighting before kernel commitment. Phase 0.5 speed budget made prescriptive (0.5 ms/step is the budget, not an example), with ablation (b) exempt because cross-layer fp32 carry is a measurement-only path.
- **v2 revision summary (for historical context):** baseline reconciliation gate added to Phase 0; new Phase 0.5 runs the deferred session-5 residual/accumulator ablations BEFORE committing to the multi-week harness; Phase 1 measurement replaced with relative Frobenius error (RFE) normalized by tensor RMS plus variance estimation and multi-layer capture from day one; Principles 2 and 4 rewritten for internal consistency; explicit falsification section added for session-5 hypothesis; Scenario 1 upgraded MEDIUM -> HIGH with Phase 0.5 as primary mitigation; explicit tradeoff-resolution statement added (speed-preserving numerical improvement wins over bit-identity when they conflict).

---

## RALPLAN-DR Summary

### Principles (5)

1. **Measurement before mutation.** Any kernel change must be preceded by an intermediate-tensor numerical localization (RFE/RMS, see Phase 1) that quantifies that op's contribution. No speculative rewrites.
2. **Speed floor: >= 70 t/s OR explicit documented regression with rollback criterion.** If reconciled HEAD is already below 70 t/s (see Phase 0), the floor becomes "no further regression vs reconciled-HEAD baseline" and any phase that drops below that reverts immediately unless an explicit speed/accuracy tradeoff is documented AND user-approved.
3. **One kernel per session.** A Ralph session rewrites exactly one op, lands measurement, and stops. Multi-op diffs can't be bisected when PCC moves.
4. **Speed-preserving numerical improvement over bit-identity.** The goal is reducing bf16 rounding error in a speed-compatible way (fp32 accumulators, better reduction order, pre-scaling) — NOT matching PyTorch/CUDA bf16 byte-for-byte. tt-metal's 32x32 tile-reduction topology cannot be bit-identical to CUDA's warp-shuffle tree with fp32 accum; we accept that. When bit-identity conflicts with Principle 2, **speed-preserving improvement wins.**
5. **Partial progress is victory.** PCC 0.985 -> 0.990 after two kernels is worth shipping and documenting. The "chase to 0.99" can stop mid-plan without regret.

### Decision Drivers (top 3)

1. **Is the 0.019 gap diffuse cumulative drift (session 5's prior) or localized to a few ops (plan's Phase 1 hypothesis)?** Phase 0.5 ablations decide this before committing multi-week harness work.
2. **Where does the 0.019 gap actually live?** Conditional on Phase 0.5 not falsifying the per-op hypothesis, Phase 1 must produce a ranked op-contribution table with variance bars before Phase 2+ commits kernel time.
3. **Is the measurement harness itself reliable?** bench_batch32 must run AND the speed baseline must be reconciled against current MEMO.md state. A flaky harness OR an orphaned baseline invalidates every later measurement.

### Viable Options

#### Option A: Measurement-first, cheap-kernels-first (RECOMMENDED, gated by Phase 0.5)

Phase 0 (env + baseline reconcile) -> Phase 0.5 (session-5 ablations) -> Phase 1 (localize) -> rank ops by (contribution / rewrite-cost) -> rewrite cheapest-payoff first.

**Pros:**
- Phase 0.5 provides a fast kill-switch if session 5's diffuse-drift prior is correct
- Fastest signal per week of kernel work if per-op hypothesis holds
- Early wins de-risk the multi-week arc
- Rollback is per-op, so a failed rewrite costs one session not the plan

**Cons:**
- Might plateau at 0.988 if the biggest contributor is also the most expensive and we defer it
- User may need to accept "partial" outcome

#### Option B: Measurement-first, biggest-contributor-first

Phase 0 -> Phase 0.5 -> Phase 1 -> rewrite the single op with largest RFE contribution regardless of cost.

**Pros:**
- Highest chance of crossing 0.99 if one op dominates
- Cleaner story if successful ("we fixed SDPA")

**Cons:**
- If Phase 1 shows SDPA is top contributor, Phase 2 becomes a 3-4 week kernel rewrite with no intermediate checkpoints
- Rollback kills the whole arc if SDPA rewrite fails numerically

#### Option C: Bypass the problem — emulator-as-reference

**INVALIDATED.** User rejected in session 6. Included per RALPLAN-DR for completeness. Invalidation rationale: does not address real numerical drift; future BitNet users hit the same ceiling.

**Decision:** Option A with the new Phase 0.5 gate. Option B becomes a fallback if Phase 1 shows a single op contributes >0.015 of the 0.019 gap. Option C is explicitly rejected.

---

## Pre-Mortem (3 failure scenarios)

### Scenario 1: Phase 1 localization shows diffuse contributions (every op ~0.002-0.003)

**Probability:** **HIGH (upgraded from MEDIUM in v1).** This is *the architect-identified prior* from session 5: "cumulative bf16 arithmetic drift across 30 transformer layers." The plan must treat this as the default hypothesis, not a tail risk.

**Symptom:** Ranked contribution table has flat top-5, no op > 0.005 RFE contribution beyond the noise floor.

**Primary mitigation (NEW in v2): Phase 0.5 runs the session-5-deferred fp32-accumulator and fp32-residual ablations FIRST.** If those ablations close >0.005 of the gap cheaply, the plan redirects to "residual+accumulator promotion across all ops" (smaller surface, higher expected value) and skips the Phase 1 per-op harness. If they close <0.002, session 5's diffuse-drift hypothesis is materially weakened and the Phase 1 multi-week arc earns its cost.

**Secondary mitigation:** Plan still allows stopping after Phase 4 with three kernels rewritten. Commit Phase 1 harness as lasting infrastructure.

### Scenario 2: Top-contributor kernel rewrite drops decode speed below reconciled baseline

**Probability:** HIGH for SDPA, MEDIUM for RMSNorm, LOW for RoPE/residual.

**Symptom:** New kernel matches reference but adds >0.5 ms/step beyond Phase 0 reconciled HEAD.

**Mitigation:** Each phase has an explicit rollback criterion on tps (tied to reconciled-HEAD baseline, not to the 74.21 figure from v1). Hybrid path: keep fast kernel for decode, use accurate kernel for prefill-only PCC measurement. Do not silently regress.

### Scenario 3: Phase 0 env fix is non-durable (dlpack kUInt32 resurfaces)

**Probability:** MEDIUM. Any pip install in the venv can re-break it.

**Mitigation:** pinned requirements lockfile, `uv sync --frozen`, `make bench-smoke` gate before every later phase.

---

## Context

**Reported HEAD (d37798d) state — to be RECONCILED in Phase 0:**

Two internally inconsistent speed figures exist in the repo. Phase 0 must resolve which is current:

| Source | decode_tps | p50 | bench used |
|---|---|---|---|
| v1 plan baseline (stale?) | 74.21 | 11.7 ms | bench_batch32 (historic) |
| MEMO.md:8 (fused-RMSNorm work) | 57.1 (p50) / 50.4 (decode avg) | 17.5 ms | 64-tok bench |
| MEMO.md:586 | 57 t/s p50 / 50.4 t/s decode avg | — | — |

**Working assumption until Phase 0 resolves it:** HEAD is ~57 t/s p50 / 50.4 decode_tps, p50 17.5 ms. That puts HEAD *already below* the 70 t/s Principle-2 floor by about -3.2 ms. Downstream phases must therefore be **speed-neutral or speed-positive**, not "spend tps budget."

**Accuracy baseline (d37798d, unchanged):**
- PCC vs HF fp32: 0.980679
- PCC vs HF bf16 (ActQuant on): 0.981319
- PCC vs HF bf16 (ActQuant off): 0.980143
- PCC vs bitnet.cpp (INT8xINT2 CPU): 0.967860
- **Gap:** ~0.019 to 0.999 target, ~0.008 to cross 0.99.

**What has been ruled out (sessions 3-6):** weight format, LM head bfp8 vs bf16, HiFi2/HiFi4, multi-layer bf16 swaps, ActQuant, attention core-range split, fused-norm extensions. See v1 for details.

**Root cause (diagnosed session 6):** op-level numerical drift between tt-metal bf16 and PyTorch/CUDA bf16. Hardware CAN produce bit-close bf16; this is a software kernel choice. **But note:** session 5 has a competing prior — the drift is *diffuse cumulative* across 30 layers, not localized to a few ops. Phase 0.5 is designed to arbitrate between these priors.

**Unranked candidate contributors:** SDPA, RMSNorm, RoPE fused, residual-add ordering, ternary_matmul internal reduction, residual-stream accumulator precision.

## Falsification Section — Which Phase 1 Outcome Validates Session 5 vs Falsifies It

**Session 5 hypothesis:** diffuse cumulative bf16 drift across 30 layers; no small set of ops dominates.

**Phase 1 outcome that CONFIRMS session 5:**
- Top-5 per-op RFE contributions all within 0.002 of each other, none > 0.005
- Layer-position asymmetry persists (layer-0 contribution ~5x layer-29) but is explained by accumulated residual-stream magnitude, not any single op

**Phase 1 outcome that FALSIFIES session 5:**
- A single op contributes >= 0.010 RFE (clearly separable from noise)
- OR top-2 ops contribute >= 0.015 combined with clear gap to top-3

**Phase 0.5 outcome that REDIRECTS the plan (bypasses Phase 1 per-op work):**
- fp32 accumulator promotion (ablation a) 90% CI lower bound on delta_PCC >= 0.005, OR
- cross-layer fp32 residual-stream promotion (ablation b) 90% CI lower bound on delta_PCC >= 0.005
- Either result means "promote precision at well-defined seams" beats "rewrite individual kernels."
- Note: ablation (b) now tests the full session-5 hypothesis (fp32 carry across ALL 30 layers), not a per-op downcast-every-layer weak form. See Phase 0.5 for the redesign rationale.

## Work Objectives

1. Restore a reliable measurement harness AND reconcile the speed baseline (Phase 0).
2. **NEW:** Test the deferred session-5 residual / fp32-accumulator ablation hypotheses before building per-op harness (Phase 0.5).
3. Produce a ranked op-contribution table for the 0.019 PCC gap using RFE-vs-reference, with variance bars (Phase 1).
4. Rewrite the top-3 contributing ops, one per Ralph session, each speed-neutral or speed-positive vs reconciled HEAD (Phases 2-4).
5. Run joint validation + multi-prompt robustness against HF bf16 (Phase 5).
6. Optionally upstream kernel fixes to tt-metal (Phase 6).

## Guardrails

**Must have:**
- Every phase gated by `bench-smoke` + accuracy regression check against prior-phase baseline
- Speed rollback tied to the **reconciled** HEAD baseline, not the 74.21 figure from v1
- All phase results logged to `docs/session_N_<phase>.md` with raw bench output
- Rollback criterion stated BEFORE kernel edits start in each phase
- ADR entry appended per phase to `docs/ADR_pcc_kernel_rewrites.md`

**Must NOT have:**
- Skipping Phase 0.5 because "per-op work is more interesting"
- Batching multiple op rewrites in one Ralph session
- Emulator-as-reference (Option C)
- Silent speed regressions
- Chasing bit-identity with PyTorch/CUDA bf16 when Principle 4 says speed-preserving improvement wins

## Task Flow

```
Phase 0: Environment restore + SPEED BASELINE RECONCILIATION (1 session, ~1 day)
     |
     v
Phase 0.5 (NEW): Session-5 deferred ablations — fp32 accum & fp32 residual (1 session, ~1-2 days)
     |
     | GATE (CI-aware): if lower 90% CI bound on delta >= 0.005, REDIRECT; if upper 90% CI bound on delta < 0.002, PROCEED to Phase 1; else AMBIGUOUS (run combined ablation)
     v
Phase 1: Per-op RFE localization harness, multi-layer, with variance (1-2 sessions, ~3-5 days)
     |
     | produces: ranked op-contribution table (RFE, normalized by tensor RMS, with variance bars)
     v
Phase 2: Rewrite op #1 (1 session, ~3-7 days)
     |
     v
Phase 3: Rewrite op #2 (1 session, ~3-7 days)
     |
     v
Phase 4: Rewrite op #3 (1 session, ~3-7 days)
     |
     v
Phase 5: Joint validation + multi-prompt robustness (1 session, ~2 days)
     |
     v
Phase 6 (optional): Upstream to tt-metal (open-ended)
```

---

## Phase 0 — Restore Measurement Harness AND Reconcile Speed Baseline

**Goal:** `bench_batch32.py`, `bench_accuracy.py`, `bench_vs_hf_noquant.py`, `bench_vs_bitnetcpp.py` all run to completion on current HEAD. **Additionally:** reconcile the two reported-speed figures (74.21 t/s vs 57 t/s p50) and lock a single authoritative HEAD speed baseline before any other phase runs.

**Task breakdown:**

1. **Diagnose dlpack kUInt32 failure.** (unchanged from v1)
2. **Pick lowest-risk fix** between torch upgrade vs cast workaround vs separate bitnet.cpp venv. (unchanged from v1)
3. **Pin and gate.** (unchanged from v1)
4. **NEW — Speed baseline reconciliation.** On reconciled HEAD run the same bench used to produce each historical figure:
   - (a) bench_batch32 with 64-tok warmup + 256-tok measure (MEMO.md:8 protocol) -> expect ~p50 17.5 ms
   - (b) bench_batch32 with the v1-plan protocol (whichever produced 11.7 ms / 74.21 t/s) -> determine whether this is reproducible or orphaned
   - If (a) and (b) give different numbers, document which bench is authoritative for this plan. Default to (a) if (b) cannot be reproduced.
   - **Output:** `docs/session_N_speed_baseline_reconciled.md` with the single authoritative tuple `(decode_tps, p50_ms, bench_protocol, commit)` that all downstream phases gate against.

**Acceptance criteria:**
- All four benches run to completion on reconciled HEAD
- `scripts/bench-smoke.sh` exits 0 in under 60 s
- **NEW:** `docs/session_N_speed_baseline_reconciled.md` records authoritative `(decode_tps, p50_ms, bench_protocol)` tuple. MEMO.md updated if reconciliation reveals either figure was stale.
- **NEW:** Principle 2's speed floor is restated as a concrete number (e.g. "reconciled HEAD is 17.5 ms; no phase may exceed 18.0 ms without documented tradeoff") in `docs/ADR_pcc_kernel_rewrites.md`.

**Rollback criterion:** If reconciliation cannot reproduce EITHER figure, do not proceed to Phase 0.5. File a bug, freeze plan.

**Files affected:** as in v1, plus `docs/session_N_speed_baseline_reconciled.md` (new) and `docs/ADR_pcc_kernel_rewrites.md` (new, with speed-floor entry).

---

## Phase 0.5 — Session-5 Deferred Ablations (NEW)

**Goal:** Before committing to a multi-week per-op harness, cheaply test the two session-5-deferred hypotheses. Redirect the plan if either closes a material chunk of the 0.019 gap.

**Rationale:** Session 5 concluded the gap is "cumulative bf16 drift across 30 layers." If true, per-op rewrites are the wrong surface; promoting precision at a *seam* (accumulator or residual stream) is cheaper and higher-value. This phase arbitrates between priors in ~1-2 days.

**Task breakdown:**

1. **Ablation (a): fp32-RMSNorm-accumulator-only**
   - Enable `compute_kernel_config.fp32_dest_acc_en=True` (or the closest equivalent flag) ONLY on rmsnorm call sites
   - Leave everything else at current bf16 config
   - Measure PCC vs HF bf16 with `bench_vs_hf_noquant` using **5 runs × 5 prompts = 25 samples** (matches Phase 1 discipline; see "variance" below)
   - Measure decode_tps with `bench_batch32` (5 runs, median + IQR)
   - Record `(delta_PCC_mean, delta_PCC_90CI_lower, delta_PCC_90CI_upper, delta_tps)` in `docs/session_N_ablation_a_rmsnorm_fp32acc.md`

2. **Ablation (b): fp32-residual-stream-across-layers (REDESIGNED in v3)**
   - **Keep residual-stream tensor in fp32 ACROSS LAYERS** — no intra-layer downcast. Accumulate residual in fp32 from the embedding output through all 30 transformer blocks; downcast to bf16 ONLY at the final pre-lm_head projection boundary.
   - Op inputs are still bf16 (we cast bf16 op-output back up to fp32 before the residual add); op internal compute is unchanged. This isolates the "diffuse fp32 residual accumulator across layers" hypothesis from session 5, not just per-op rounding.
   - Every other op unchanged (still bf16 compute).
   - Measure same three metrics as (a) with the same 5×5 discipline.
   - Record in `docs/session_N_ablation_b_residual_fp32.md`.
   - **Note:** this is intentionally a measurement-only path; the Python-side fp32 carry will be slower. Speed is gated via Principle 2 only as a directional check, not as a pass/fail (see "Speed check" below). Combined run (a)+(b) inherits ablation (b)'s exemption from the speed budget.
   - **Wall-clock abort (v4):** If ablation (b) per-sample wall-clock exceeds 5× normal prefill (measured in Phase 0), automatically drop sample count to 3 prompts × 3 runs = 9 samples and widen bootstrap CI to 80%. Document the reduced statistical power in the decision doc. Hard abort if even 9-sample run exceeds 1 day total.

3. **Combined ablation (a)+(b)** always runs when either individual ablation lands in the AMBIGUOUS band (below), otherwise optional
   - Measure interaction effect with the same 5×5 discipline
   - Record in `docs/session_N_ablation_ab_combined.md`

**Variance protocol (v3):** ablation PCC uses 5 runs × 5 prompts = 25 samples; report mean, 90% CI (percentile bootstrap, 1000 resamples), and the per-prompt std. Gate decisions use the 90% CI bounds on `delta_PCC` (ablation minus reconciled HEAD), not a single point estimate.

**Paired-bootstrap protocol (v4):** Samples are PAIRED across conditions — each of the 25 (run, prompt, seed) tuples is evaluated under both (ablation) and (HEAD) with identical seed and identical prompt. Bootstrap resamples the 25 per-tuple delta_i = PCC(ablation)_i − PCC(HEAD)_i values, NOT the two marginal distributions independently. Paired bootstrap tightens the CI by √2 and is required for the 0.002/0.005 gate bands to be decision-relevant.

**Gate criteria (primary plan decision point) — CI-AWARE (v3):**

Given σ ≈ 0.002 noise on PCC measurements, a 3-run median can easily flip REDIRECT ↔ PROCEED around a true delta of 0.004. Gate on CI bounds, not point estimates.

Let `delta = PCC(ablation) - PCC(reconciled HEAD)` for ablation (a) or (b). Compute a 90% CI from 25 samples via percentile bootstrap.

- **REDIRECT:** if (a) lower 90% CI bound `>= 0.005` OR (b) lower 90% CI bound `>= 0.005`. Rewrite Phases 1-4 as a single "promote precision at seams across all ops" arc. Phase 1 becomes "map every accumulator/residual-stream seam in the forward pass"; Phases 2-4 become "promote (a) all accumulators, (b) all residual adds, (c) any remaining seam identified in Phase 1 map." **Feasibility catalog (v4):** Before committing Phases 1-4 to REDIRECT scope, insert a ~1 day tt-metal-feasibility catalog step: enumerate fp32_dest_acc_en / fp32-accumulator support coverage across every residual, norm, and matmul seam in the forward pass. If kernel coverage < 80% of seams: REDIRECT fallback becomes a hybrid — Python-side fp32 carry for measurement-only seams + targeted kernel rewrites for uncovered seams (partial reversion toward Option A scoping).
- **PROCEED:** if (a) upper 90% CI bound `< 0.002` AND (b) upper 90% CI bound `< 0.002`. Session 5's diffuse-drift-at-seams hypothesis is materially weakened at the 90% confidence level. Per-op hypothesis earns the multi-week harness cost. Proceed to Phase 1 as specified.
- **AMBIGUOUS:** any configuration between the above (e.g. CI straddles the 0.002-0.005 band, or one ablation REDIRECTs and the other PROCEEDs). Run combined (a)+(b) ablation with the same 5×5 discipline and re-gate on the combined CI using the same REDIRECT/PROCEED rules above. If combined is still AMBIGUOUS, treat as REDIRECT (conservative: session 5's prior is stronger than a non-separable null).

**Speed check (Principle 2) — PRESCRIPTIVE (v3):**
- **0.5 ms/step is the budget, not an example.** Ablation (a) must stay within +0.5 ms/step of reconciled-HEAD p50; ablation (b) is exempt from the budget because the cross-layer fp32 carry is a measurement-only path whose speed is not the thing being tested. Record both numbers either way.
- If (a) exceeds the budget, note it and still use the PCC delta to decide the gate — but flag in the decision doc that the REDIRECTed plan will need kernel-level fp32 accumulator support, not just a Python toggle.

**Risk + mitigation:**
- *Risk:* tt-metal's fp32_dest_acc_en flag has incomplete op coverage and (a) fails to compile. *Mitigation:* fall back to a Python-side fp32 cast around rmsnorm input/output for the measurement only.
- *Risk:* (b) Python-side fp32 residual cast is much slower and confounds the PCC measurement. *Mitigation:* this is a measurement-only pass; we tolerate the slowdown to get the PCC delta.

**Files affected:**
- `src/bitnet_tt/model/transformer.py` (ablation toggles, env-var gated)
- `src/bitnet_tt/layers/attention.py` (residual-add fp32 cast, env-var gated)
- `docs/session_N_ablation_a_rmsnorm_fp32acc.md`
- `docs/session_N_ablation_b_residual_fp32.md`
- `docs/session_N_ablation_ab_combined.md` (if needed)

---

## Phase 1 — Per-Op Numerical Localization Harness (REWRITTEN)

**Goal:** Produce a table ranking each op's contribution to the 0.019 PCC gap using a metric that composes correctly through residual streams, with variance estimation sufficient to distinguish ops that differ by <= 0.002 (**within-class ranking noise floor**; the **cross-class actionable signal threshold** is 0.01 — see Phase 1 measurement plan).

**Metric change (vs v1):**

v1 used "incremental delta-PCC per op, sum to end-to-end gap ± 0.005." PCC is **non-additive** through nested transforms, so that sum-check was mathematically unsound.

v2 uses **relative Frobenius error (RFE) normalized by tensor RMS**:

```
RFE(tt_T, hf_T) = ||tt_T - hf_T||_F / (||hf_T||_F + eps)   # relative Frobenius
```

RFE composes through residual streams via the triangle inequality: if op A adds error e_A and op B adds error e_B, the post-B error is bounded by e_A + e_B + O(e_A * e_B). This makes per-op contribution **meaningfully rankable** by RFE-delta between adjacent capture points.

**Caveat (v3):** the triangle inequality is *loose across nonlinear ops* (RMSNorm, softmax). Two ops with equal RFE-delta can have very different downstream amplification factors because their local Jacobians differ. Ranking is therefore only directly comparable between ops with **similar local Jacobian scales** — e.g. two linear matmuls, or two residual-adds. A big RFE on RMSNorm is not directly comparable to a big RFE on a residual-add without Jacobian-aware reweighting. Phase 1 flags this in the output table by grouping ops into {linear-ish, norm, softmax/attention} classes; cross-class ranking is annotated "requires Jacobian correction before kernel commitment."

**Task breakdown:**

1. **Instrument the TT forward pass for intermediate tensor capture** (capture points unchanged from v1, same 11 points per layer).
2. **Build a parallel HF bf16 reference capture** in `bench_vs_hf_noquant.py` with forward hooks.
3. **Multi-layer capture from day one (CHANGED).** v1 proposed single-layer bootstrap. v2 requires layers {0, 5, 15, 25, 29} from the first capture run because session 4 showed layer-0 contribution is ~5x layer-29; a single-layer bootstrap misranks ops.
4. **HF-boundary alignment for fused QKV + RMSNorm (NEW).** Fused-kernel drift is non-decomposable because the fused output has no reference layout to compare against. Provide ONE of:
   - (a) an un-fused TT shadow path: a debug-only code path that runs unfused QKV and unfused post-RMSNorm, purely for boundary capture. Used ONLY in Phase 1, not in production decode.
   - (b) explicit documentation in `docs/pcc_alignment.md` that fused-kernel capture points are "combined RFE only — non-decomposable into per-sub-op contribution." Fused-kernel rewrites in Phase 2+ are scoped by combined RFE, not sub-op breakdown.
   - Phase 1 picks (a) if the shadow path is < 2 days of work; else (b).
5. **Variance estimation (UPDATED v3).** For each (layer, capture_point) pair, run the capture **N=10** times across the same prompt and across K=5 different prompts (was N=5 — underpowered at a 2σ bar). Compute:
   - within-prompt std of RFE
   - across-prompt std of RFE
   - report `RFE_delta +/- max(within_std, across_std)` per op
   - An op is "statistically separable" only if `RFE_delta > 3.32 * sigma` (v4: Bonferroni correction at FWER=0.05 across 55 comparison cells — 11 capture points × 5 layers. Per-cell two-sided z-threshold = Φ⁻¹(1 − 0.025/55) ≈ 3.32σ. **Target: FWER=0.05 (family-wise error rate).** Alternative Benjamini-Hochberg at FDR=0.10 was considered but Bonferroni chosen because the Phase-2 target commitment is binary per-op and false-positives cost a full kernel-rewrite session.). Ops that do not clear this bar are grouped into a "noise floor" bucket and not ranked individually.
   - If 50 samples × 11 capture points × 5 layers proves too expensive in wall-clock, fall back to N=5 with the same 3.32σ Bonferroni bar documented as the weaker power — but keep the 3.32σ threshold regardless of N.
6. **Ranking harness** `scripts/pcc_localize.py` outputs CSV and markdown table sorted by RFE-delta with variance bars.
7. **Produce final ranked table** + decision doc `docs/session_N_pcc_localization.md` recommending Phase 2 target.

**Acceptance criteria (REWRITTEN):**
- Capture infrastructure adds < 2 s to a single prefill call (not used in bench_batch32)
- Multi-layer, multi-prompt RFE table with variance bars is produced
- "Statistically separable" top-K ops identified (K may be 0, 1, 2, or more)
- Fused-kernel alignment resolution (shadow path or documented non-decomposability) is committed
- Decision doc names Phase 2 target OR states "no statistically separable op dominates — session 5 confirmed; plan may stop."

**Measurement plan (v3):**
- 5 prompts × 5 layers × 11 capture points × N=10 runs for variance (fallback N=5 if wall-clock forbids)
- Expected signal size: if any op dominates, its RFE-delta is > 3.32*sigma (Bonferroni FWER=0.05 across 55 cells) AND > 0.01 relative (cross-class actionable signal threshold; 0.002 is the within-class ranking noise floor)
- If no op clears the 3.32*sigma Bonferroni bar, Scenario 1 is confirmed and plan exits to Phase 5 joint validation on reconciled HEAD
- Cross-class (linear vs norm vs softmax) rankings carry the "Jacobian-correction-required" annotation before being used to pick a Phase 2 target

**Rollback criterion:** If capture infrastructure itself perturbs PCC (captured run differs from uncaptured run beyond 0.0001), freeze capture impl and restart. PCC capture harness must be a pure observer.

**Risk + mitigation:**
- *Risk:* HF reference module boundaries don't align 1:1 with TT (fused QKV). *Mitigation:* addressed explicitly in task (4) above.
- *Risk:* variance estimation shows noise floor swamps all per-op signals. *Mitigation:* this IS a valid Phase 1 outcome; plan exits cleanly.
- *Risk:* capture costs balloon. *Mitigation:* capture is opt-in via env var; bench_batch32 never runs it.

**Files affected:**
- `src/bitnet_tt/model/transformer.py`
- `src/bitnet_tt/layers/attention.py`
- `src/bitnet_tt/layers/ffn.py`
- `bench_vs_hf_noquant.py` (new hooks)
- `scripts/pcc_localize.py` (new)
- `docs/session_N_pcc_localization.md`
- `docs/pcc_alignment.md` (fused-kernel decomposability note)
- `src/bitnet_tt/debug/unfused_shadow.py` (new, if task 4(a) selected)

---

## Phase 2 — Rewrite Op #1 (Top Contribution / Cost Ratio)

**Goal:** Rewrite the op ranked #1 by Phase 1 to reduce RFE in a **speed-preserving** way (Principle 4) while holding decode speed at or above reconciled-HEAD baseline (Principle 2).

(Placeholder target ordering unchanged from v1; speed budget now expressed in ms/step delta vs reconciled HEAD, not against 11.7 ms.)

**Generic task breakdown:** unchanged from v1 except:

- Step 1 uses RFE (not PCC) for op characterization
- Step 5 "Decide" becomes:
  - **PASS:** decode speed at or above reconciled-HEAD baseline (no regression), AND end-to-end PCC improved by at least 30% of Phase 1's predicted RFE gain for this op (was 50% in v1 — relaxed because fused-kernel alignment may give us only combined-RFE targets)
  - **FAIL:** revert, document

**Acceptance criteria (REWRITTEN for internal consistency):**
- decode speed: no regression vs reconciled-HEAD baseline (hard gate)
- end-to-end PCC vs HF bf16: improved by >= 30% of Phase 1 predicted gain for this op (was 50%)
- argmax match on 64 test prompts >= current baseline
- Zero regression on `bench_accuracy` vs HF fp32
- Full kernel diff committed with `docs/session_N_<op>_rewrite.md` and ADR entry
- **NEW:** ADR entry explicitly states whether the fix is "bit-closer to CUDA bf16" (Principle 4 aligned) or "numerically cleaner but differs from CUDA" (Principle 4 explicitly chosen over bit-identity)

**Measurement plan / Rollback / Risk:** unchanged from v1 except speed comparisons gate against reconciled-HEAD, not 74.21 t/s.

---

## Phase 3 — Rewrite Op #2

(Unchanged from v1 structure; baselines now chain against reconciled HEAD.)

## Phase 4 — Rewrite Op #3

(Unchanged from v1 structure.)

**Partial-success exit:** If PCC >= 0.990 after any of Phases 2-4, user may elect to skip remaining phases.

---

## Phase 5 — Joint Validation + Multi-Prompt Robustness

(Structure unchanged from v1; acceptance thresholds for speed now reference reconciled HEAD rather than 70.0 t/s absolute.)

**Acceptance criteria (REWRITTEN):**
- Mean PCC vs HF bf16 across 64 prompts: > 0.99 (goal) OR > 0.985 (partial-success fallback)
- Minimum PCC across 64 prompts: > 0.97
- decode speed: no regression vs reconciled-HEAD baseline, OR explicit documented tradeoff with user sign-off
- argmax match rate >= baseline

---

## Phase 6 — Upstream to tt-metal (OPTIONAL)

(Unchanged from v1.)

---

## Success Criteria (whole plan, REWRITTEN)

- decode speed: no regression vs reconciled-HEAD baseline at final commit (or documented tradeoff)
- PCC vs HF bf16 >= 0.990 on mean-of-64-prompts (partial: >= 0.985)
- RFE localization harness is lasting infrastructure
- Phase 0.5 ablation results logged even if they redirect the plan
- Every phase has a documented result in `docs/session_N_*.md`
- Cumulative ADR in `docs/ADR_pcc_kernel_rewrites.md` records every kernel change with Principle-4 alignment statement

---

## ADR — PCC > 0.99 Multi-Session Kernel Plan (v4)

**Decision:** Adopt Option A (measurement-first, cheap-kernels-first, one-kernel-per-session) with a **new Phase 0.5 gate** that tests session-5's deferred fp32-accumulator and fp32-residual hypotheses before committing to per-op harness work. Speed baseline is reconciled in Phase 0 before any numerical target is defined.

**Drivers:**
1. Session 5 already posited diffuse cumulative drift; Phase 0.5 arbitrates this prior cheaply (~1-2 days) and can redirect the plan before multi-week commitment.
2. RFE is the mathematically correct composable metric; PCC-delta summation (v1) was unsound.
3. Speed baseline must be reconciled because v1's 74.21 t/s figure conflicts with MEMO.md's 57 t/s — every downstream speed gate depends on knowing which is current.
4. Ralph session cadence favors atomic, revertible per-op commits over multi-week monoliths.
5. **(v3) The Phase 0.5 gate is the most consequential decision in the whole plan** (REDIRECT or PROCEED), so it must be gated by confidence bounds, not point estimates; and ablation (b) must actually test the hypothesis it claims to test (diffuse fp32 accumulation across layers), not a numerically-equivalent weaker form.

**Alternatives considered:**
- **Option A without Phase 0.5 (v1 plan):** Rejected. Session 5's diffuse-drift prior is too strong to bypass with an expensive harness build. Phase 0.5 is cheap insurance.
- **Option B (biggest-contributor-first):** Conditional fallback if Phase 1 shows a single op contributes > 0.015.
- **Option C (emulator-as-reference):** Rejected; invalidation rationale per RALPLAN-DR.
- **Do-nothing:** Legitimate fallback after Phase 0.5 if ablations close the gap cheaply, or after Phase 1 if drift is too diffuse.
- **(v3) Architect-suggested simplification — run only ablation (a) in Phase 0.5, defer (b) to Phase 1 RFE ranking:** Considered and rejected. Session 5's hypothesis is specifically about cross-layer residual-stream accumulation, which Phase 1's per-op RFE harness does NOT naturally surface (residual-add RFE can look small per-op while accumulation dominates end-to-end). Keeping the cross-layer fp32 residual carry as an explicit Phase 0.5 ablation (b) preserves a cheap falsification path for the session-5 prior; dropping it would force us to either run a multi-week harness to discover the same thing or leave the prior un-falsified. Ablation (a) alone does not substitute.
- **(v3) Retaining v2's per-op-downcast ablation (b):** Rejected. Casting to fp32 at the add and immediately back to bf16 is mathematically equivalent to a bf16 add (rounding is merely deferred one op); it does not test the cross-layer hypothesis and would produce a near-zero delta_PCC that would spuriously trigger PROCEED.

**Why chosen:**
- Phase 0.5 arbitrates the session-5 prior before harness cost is sunk
- RFE metric composes correctly through residual streams (triangle inequality)
- Reconciled speed baseline prevents downstream phases from chasing a stale target
- Principle 4's "speed-preserving numerical improvement over bit-identity" explicitly resolves the tt-metal-vs-CUDA tile-topology tradeoff

**Consequences:**
- Adds Phase 0.5 (~1-2 days) to the arc; total 5-7 weeks vs v1's 4-6
- Requires ablation infrastructure (fp32 cast toggles gated by env var)
- Commits to "speed-preserving improvement" framing; will not pursue bit-identity with CUDA bf16
- Accepts that Phase 0.5 may redirect the entire plan away from per-op rewrites (this is a feature, not a cost)
- **(v3)** Phase 0.5 wall-clock grows from ~1-2 days to ~2-3 days because CI-aware gating requires 25 samples per ablation (vs 3-run median); the combined ablation is no longer conditional-on-delta but conditional-on-CI-straddling-bands. Net schedule impact: +~1 day at most.
- **(v3)** Ablation (b) requires a cross-layer fp32-residual Python path (env-var gated in `src/bitnet_tt/model/transformer.py`) that is slower than production; this is accepted as measurement-only and exempt from the 0.5 ms/step budget.
- **(v3)** Phase 1 variance grows from N=5 to N=10 (fallback N=5 with the same 3σ bar). Wall-clock of the localization harness approximately doubles; accepted because the bar's statistical meaning is primary.
- **(v4)** Phase 0.5 bootstrap is paired across conditions (tightens CI by √2) — prerequisite for the 0.002/0.005 gate bands to be decision-relevant.
- **(v4)** Phase 1 separability threshold is 3.32σ (Bonferroni FWER=0.05 across 55 cells) rather than raw 3σ; a small number of borderline ops that would have cleared 3σ will now be binned into the noise floor. Accepted because false positives cost a full kernel-rewrite session.
- **(v4)** Phase 0.5 ablation (b) has a wall-clock safety valve: 5× normal prefill triggers a fallback to 3×3=9 samples with 80% CI, and a hard 1-day abort. Worst-case Phase 0.5 wall-clock is bounded.
- **(v4)** REDIRECT path adds a ~1-day tt-metal-feasibility catalog before Phase 1-4 rescope. If fp32-accumulator coverage <80% of seams, REDIRECT becomes a hybrid Python-carry + targeted-rewrite plan rather than pure seam promotion. Total schedule impact under REDIRECT: +~1 day.

**Follow-ups:**
- Phase 0.5 outcome documented as `docs/session_N_ablation_*.md` and summarized in ADR
- If Phase 0.5 REDIRECTS, rewrite Phase 1-4 scope before starting Phase 1
- Each phase's ADR entry appends to `docs/ADR_pcc_kernel_rewrites.md` with Principle-4 alignment statement

---

## Open Questions

- Which bench protocol produced the v1 74.21 t/s figure, and is it reproducible on HEAD? (Phase 0 resolves.)
- Does tt-metal's `fp32_dest_acc_en` flag have coverage on rmsnorm kernel, or is a Python-side cast required? (Phase 0.5 task 1 resolves.)
- Is the un-fused TT shadow path (Phase 1 task 4(a)) < 2 days of work, or do we commit to non-decomposable fused-kernel RFE (4(b))? (Phase 1 resolves during scoping.)

Any further open questions identified during execution will be appended to `.omc/plans/open-questions.md`.
