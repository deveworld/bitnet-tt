# Attention Core-Range Split for Fused Q/K + K/V Ops

## Status: CLOSED (measured, 2026-04-18)

Microbench with proper tt_transformers-style shard setup (Q on (0,0)-(7,3),
K on (0,4)-(7,7), cos/sin/trans_mat on (0,0)-(7,7) = 64 cores):

- `2x rotary_embedding_llama` in trace: **26.27 us / layer**
- `rotary_embedding_llama_fused_qk` in trace: **31.05 us / layer** (*slower*)

The fused op coordinates 2× more cos/sin/trans_mat data across 64 cores
vs 32, which outweighs the saving from one kernel launch vs two. On top
of that, the fused path needs the pre-reshard of Q and K onto disjoint
cores (~64 us / layer) which we don't pay today. Net per-layer delta:
`(31 + 64) - 26 = +69 us / layer * 30 layers = +2.0 ms / step`.

`paged_fused_update_cache` has the same story: ~10 us saving vs 2x
paged_update_cache, but K and V must go to disjoint cores -- costing
one extra reshard (~32 us / layer) to accommodate. Net negative.

The "~0.7 ms combined win" estimate in the original plan (below) was a
back-of-envelope based on non-trace op categorisation. Trace-level
measurements reveal the ops are fundamentally wider than narrower when
distributed across double the cores, and the reshard overhead eats
whatever launch-fusion savings exist.

**Do not revisit unless the fused ops themselves change on tt-metal or
we find a way to get Q and K onto disjoint cores for free (e.g.
`nlp_create_qkv_heads_decode` with `overlap_qk_coregrid=False`, but
that also requires sharded input -- another reshard).**

---

## Goal (original, historical)

Enable two tt-metal fused ops that are currently gated behind a sharding
constraint:

1. `ttnn.experimental.rotary_embedding_llama_fused_qk` -- applies RoPE to Q
   and K in a **single kernel** instead of the two separate
   `rotary_embedding_llama` calls we do today.
2. `ttnn.experimental.paged_fused_update_cache` -- writes K and V slots to
   their respective paged caches in a **single kernel** instead of two
   back-to-back `paged_update_cache` calls.

Both require their two input tensors to sit on **non-overlapping core
ranges** so the fused kernel can read them concurrently without racing.

## Expected gain

From the documented trace-kernel breakdown (20.8 ms total, matmul 40 %,
norm 17 %, other 43 %) and the non-trace profile at the end of
`project_decode_hot_ops.md`:

| Op                       | Current (approx, trace) | Target | Savings |
| ------------------------ | ----------------------- | ------ | ------- |
| `rotary_embedding_llama` × 2 | ~0.8 ms / step          | 1 call | ~0.4 ms |
| `paged_update_cache`  × 2 | ~0.6 ms / step          | 1 call | ~0.3 ms |
| **Combined**             |                         |        | **~0.7 ms** |

On the current p50 of 17.5 ms, that is ~4 % -- closer to 60 t/s p50.

## What blocks it today

`src/bitnet_tt/layers/attention.py:_forward_decode_1bkd` creates heads via

```python
q_heads_1bkd, k_heads_1bkd, v_heads_1bkd = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused,
    num_heads=self.num_heads,
    num_kv_heads=self.num_kv_heads,
    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
)
```

`nlp_create_qkv_heads_decode` uses a **single** core range for all three
outputs -- the exact situation that blocks the fused RoPE + fused cache
update ops (explicit comment at `attention.py:1039` and
`attention.py:1082-1085`).

## Design options

### Option A -- post-split re-shard (lowest risk)

Keep `nlp_create_qkv_heads_decode` as-is, then **re-shard Q, K, V onto
disjoint core ranges** immediately after. This is a dispatch-only
rearrangement: no kernel changes, just `ttnn.to_memory_config` with
custom `ShardSpec`s.

Core grid on Blackhole after harvest is 11×10 = 110 cores. Suggested
split (num_heads = 32, num_kv_heads = 8 on BitNet 2B):

- Q heads → a `q_grid` with ≥ 32 cores, e.g. 4 rows × 8 cols
- K heads → a `k_grid` disjoint from `q_grid`, ≥ 8 cores
- V heads → a `v_grid` disjoint from both, ≥ 8 cores

Example layout:
```
Q: (0,0)-(7,3)     32 cores
K: (0,4)-(7,4)      8 cores
V: (0,5)-(7,5)      8 cores
```

**Risk:** re-shard is itself a dispatched op; its cost must be less than
the 0.7 ms we are trying to save. Measure first with a microbench
before wiring into the decode path.

### Option B -- extend `nlp_create_qkv_heads_decode` to accept split grids

Larger change -- either a new op or a new arg. Cleaner long-term but
touches tt-metal. Keep as a follow-up if Option A lands a measurable
win and we want to shave the re-shard cost.

### Option C -- compose from lower-level heads ops

There are separate `nlp_create_q_heads` / `nlp_create_kv_heads` paths in
tt-metal. Using those would give per-tensor core ranges directly but
may cost more than the current decode-tuned combined op.

**Start with Option A.** It's the one that fits cleanly in the Python
layer without any tt-metal work.

## Implementation steps

1. **Microbench the re-shard cost.** Standalone test: create a fake
   1BKD Q tensor, `to_memory_config` it to a smaller sharded spec,
   measure μs. If it is already > ~300 μs, Option A may not pay off --
   escalate to Option C before spending more time.
2. **Microbench fused RoPE and fused cache update in isolation.** Build
   disjoint-sharded Q/K and K/V, call the fused ops, verify PCC vs. the
   two-call baseline, and measure per-call latency. Budget: match or
   beat 2 × current-call time by ≥ 200 μs.
3. **Wire into `_forward_decode_1bkd`.** Gate on a `use_fused_qk_rope`
   flag (mirror the existing `use_fused_rope` plumbing). Re-shard
   after `nlp_create_qkv_heads_decode`, call the fused ops, reshard
   back (or directly into what the next op wants) before
   `paged_update_cache` / SDPA.
4. **End-to-end trace benchmark.** The only number that matters is the
   p50 on `bench_batch32.py --dtype packed_ternary --max-new 64`. Must
   improve over the current 17.5 ms to ship.
5. **Numerical check.** Full-model greedy decode output must match the
   non-fused path for the first ≥ 32 tokens (same greedy trajectory).

## Files to touch

- `src/bitnet_tt/layers/attention.py`
  - `_forward_decode_1bkd` -- add re-shard + fused ops
  - Potentially pre-compute the Q/K/V shard specs in `__init__` so they
    are not rebuilt per decode step
- `src/bitnet_tt/inference/generator_batch32.py`
  - Pass a flag or just rely on attention's internal decision
- No tt-metal changes for Option A

## Pass / fail criteria

- **PCC** (fused vs. non-fused) ≥ 0.999 on the o_proj matmul output
  (downstream of both fused ops -- exercises the full chain).
- **p50 decode latency** on the full benchmark drops by ≥ 0.3 ms
  (enough to justify the added code path).
- **No new L1 overflow** in the trace-mode program. Check with
  `TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS=BR,NC,TR,ER` stays valid.

If either of the latency or PCC criterion fails, revert the wiring --
keep the standalone sharding helpers in case a future attention
refactor wants them.

## Non-goals for this plan

- Further fused-norm extensions (o_proj / gate_up / down_proj) -- already
  tested and regressed at trace level, see
  `project_fused_norm_result.md`.
- Any SDPA-internal optimization -- treat SDPA as a black box.
- Multi-device sharding (all-gather variants). Irrelevant for p150a.
