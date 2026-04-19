# Open Questions: BitNet-TT PCC Plans

## pcc_above_099_multisession: resolved in sessions 7-13

- [x] **dlpack-kUInt32 workaround (Phase 0)**: RESOLVED session 7. Chose the in-tree compat shim: alias `torch.uint{16,32,64}` → `int` + `ttnn.to_torch` wrapper that typecasts uint tensors via `ttnn.typecast(..., int32)`. No torch upgrade needed. Separate bitnet.cpp venv not required, bitnet.cpp lives in `~/bitnet.cpp` and uses its own pip env. Commit 1843133, later refined at 403377c, 959845a.
- [x] **Option A vs Option B after Phase 1**: RESOLVED session 9-10. Phase 1 per-op RFE (session 8) located drift in the SDPA chain (L0.post_self_attn RFE 0.240, L0.post_attn_sub_norm 0.420). Sessions 9-10 tested five cheap levers (fp32 RMSNorm/residual, SDPA HiFi4, manual RoPE, manual primitive SDPA), all ≤ |0.002| PCC delta, i.e. below PROCEED threshold. Pivoted to Phase K (tt-metal kernel rewrite) plan: `docs/plan_sdpa_kernel_alignment.md`.
- [x] **Partial success at PCC 0.985-0.990 acceptable?**: DEFERRED, not yet needed. Cheap levers delivered no partial progress (still at 0.982); the question becomes live only if Phase K2/K3 partially close the gap.
- [x] **Phase 1 capture harness instrument bitnet.cpp?**: RESOLVED session 7/10. Decoupled: `bench_vs_bitnetcpp.py` does TT-vs-bitnet.cpp directly on prefill logits via `extract_logits.cpp` binary dump. Not integrated into `pcc_localize.py` because bitnet.cpp has no per-op boundary we can match TT at, only end-to-end logits.

## Still open: gates for Phase K execution

- [ ] **Hybrid fast-decode + accurate-prefill kernel path acceptable** if a Phase K kernel rewrite raises prefill PCC but regresses decode tps below 70? Relevant for K3/K4 gating. Decode path uses a different SDPA variant (`scaled_dot_product_attention_decode`), so prefill-only kernel changes won't touch decode directly, but the fused compute kernel may be shared.
- [ ] **Phase 6 upstreaming priority**: schedule an upstream tt-metal PR vs keep patches in fork. Not relevant until at least Phase K3 lands with a measurable win.
- [ ] **Phase K1 harness expected signal**: if stock SDPA vs `torch.nn.functional.scaled_dot_product_attention` RFE is already ≤ 0.05, SDPA itself is not the bottleneck and Phase K redirects to RMSNorm / RoPE op alignment. Cannot predict without running the harness.
- [ ] **Kernel budget trade-off if decode_tps < 72 after K3**: current headroom is 2.6 ms vs 70 t/s floor; Phase K2+K3+K4 ceiling is 2.5-3 ms. If K3 alone consumes > 1.5 ms, K4 must either be skipped or ship at sub-70 decode_tps. User decision required at that gate.
