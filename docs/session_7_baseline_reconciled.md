# Session 7 — Phase 0 baseline reconciliation + dlpack env fix

Plan v4 Phase 0 deliverable.

## dlpack kUInt regression (US-701)

After the session-6 bitnet.cpp pip install, `bench_batch32.py` started
failing with `RuntimeError: Unsupported kUInt bits 32` inside
`ttnn.to_torch` → `torch.utils.dlpack.from_dlpack`.

Root cause: torch 2.2.2 rejects DLPack capsules whose dtype is
`kUInt{16,32,64}`. ttnn's argmax op returns a `ttnn.uint32` tensor;
`ttnn.to_torch` on that tensor routes through dlpack and dies.

Fix (`src/bitnet_tt/_ttnn_config.py` + `src/bitnet_tt/__init__.py`):
1. `_install_torch_uint_compat_shim` — adds `torch.uint{16,32,64}` as
   aliases of `torch.int{16,32,64}` so any attribute-based dispatch in
   ttnn succeeds.
2. `_install_ttnn_to_torch_uint_wrapper` — intercepts `ttnn.to_torch`
   and routes `uint32`-dtyped tensors through `ttnn.typecast(..., int32)`
   before the torch conversion. Falls back cleanly if typecast is not
   applicable.
3. Single call-site fix in `_sample_token` (`inference/generator_batch32.py:1362`):
   for the trace-embedded argmax tensor (shape `[1]`, ROW_MAJOR), the
   typecast path additionally requires a TILE-layout pad first, so the
   readout uses
   `ttnn.to_torch(ttnn.typecast(ttnn.to_layout(t, TILE_LAYOUT), int32))`.

## Baseline reconciliation (US-702)

`bench_batch32.py --dtype packed_ternary` on HEAD (after the dlpack fix):

| max-new | decode_tps | overall_tps | p50 ms | p90 ms | min ms |
|--------:|-----------:|------------:|-------:|-------:|-------:|
|    128  |     71.05  |      69.00  |  12.0  |  12.7  |  11.0  |
|     64  |     68.93  |      65.23  |  11.5  |  11.9  |  10.9  |

Compare with MEMO.md:8 which reads:
> 2026-04-17, batch32 + trace + fused RoPE + packed_ternary:
> p50 17.5 ms = 57.1 t/s, min 16.9 ms = 59.2 t/s peak, decode_tps ≈ 50.4 (64-tok bench).

MEMO.md:8 is **stale**. It predates the split-lm-head / sharded-rmsnorm
/ multicore-argmax / cos-sin-lookup / fused-QKV-norm stack and never
got refreshed. Current 64-tok run is 68.93 t/s / p50 11.5 ms, not
57.1 t/s / p50 17.5 ms.

## Authoritative Phase-0 baseline

For Phase 0.5 onward:

| key                 | value         |
|--------------------:|:--------------|
| commit              | d37798d (+ dlpack fix)           |
| bench               | `bench_batch32.py --dtype packed_ternary --max-new 128` |
| decode_tps          | 71.05 t/s     |
| overall_tps         | 69.00 t/s     |
| p50 ms              | 12.0          |
| p90 ms              | 12.7          |
| min ms              | 11.0          |

Speed floor is still **≥ 70 t/s on decode_tps**. Current baseline clears
by +1.05 t/s; headroom before the floor is ~0.2 ms / step across any
Phase 0.5 / 1 / 2-4 additions. That is tight.

## Next phase

Phase 0.5 stories will be written to `.omc/prd.json` in US-704:
fp32-RMSNorm-accumulator ablation + cross-layer fp32 residual stream
ablation, both measured with paired 25-sample bootstrap CI against the
reconciled HEAD baseline above.
