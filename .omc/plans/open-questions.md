# Open Questions — BitNet-TT PCC Plans

## pcc_above_099_multisession — 2026-04-18

- [ ] Which dlpack-kUInt32 workaround does the user prefer (torch upgrade vs cast workaround vs separate bitnet.cpp venv)? — Determines Phase 0 scope and risk profile.
- [ ] If Phase 1 shows SDPA contributes >0.015, does user want Option B fallback (SDPA-first, 3+ week single phase) or stay on Option A (cheaper ops first, accept partial outcome)? — Branching point after Phase 1.
- [ ] Is "partial success" (PCC 0.985-0.990) an acceptable shipping outcome, or must the plan continue until 0.99 is crossed even if it means Phase 6+ work? — Affects per-phase stop criteria.
- [ ] Is a hybrid fast-decode + accurate-prefill kernel path acceptable if a kernel rewrite tps-regresses below 70 for decode but improves prefill PCC? — Affects Phase 2-4 rollback criteria.
- [ ] Phase 6 (upstreaming) — user priority? Open-ended review cycles vs keeping patches local in fork indefinitely. — Affects whether Phase 6 is scheduled or left as "someday".
- [ ] Should Phase 1 capture harness also instrument bitnet.cpp as a third reference (alongside HF bf16 and HF fp32)? — Widens diagnostic value but adds env coupling to a flaky bitnet.cpp install.
