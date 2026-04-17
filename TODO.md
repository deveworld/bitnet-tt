# TODO

## 현재 상태 (2026-04-17)

### 달성 — p50 11.3 ms / 71.0 t/s (min 10.7 ms, 64 tok), bfp4 대비 +135%
- **p50 11.3 ms, min 10.7 ms, decode_tps ≈ 70.98 t/s** (batch32 + trace + fused RoPE + multicore argmax + tuned sharded rms_norm + split lm_head with L1 chunks + nlp_concat_heads_decode, 64-tok) — Blackhole p150a
- **nlp_concat_heads_decode**: SDPA 이후 `ttnn.reshape` 대신 전용 kernel 사용. GQA SDPA는 sharded 출력 불가로 reshard 필요하지만, 전체 pass가 reshape 38→concat 29 μs (-9 μs/layer).
- **LM head chunks L1**: split lm_head의 4 chunk 출력을 L1에 유지 → concat/to_layout/argmax 체인 DRAM 라운드트립 제거. -40 μs/step.
- **Split lm_head matmul (4-way)**: 단일 matmul이 vocab=128256 전체에 맞춰 per_core config를 잡아 ~2.2 ms/step 소요되던 것을 4개 chunk로 쪼개 각각 tight per-core config 적용. 측정: 2197 → 722 μs (-67%). trace kernel 11.9 → 10.4 ms.
- **Shard grid tuning**: H=2560은 y=2,x=8 (16 cores), H=6912은 y=3,x=8 (24 cores)로 per-core width 축소 → 멀티코어 reduction이 per-core sum 비용 dominant. 28 μs/call (H=2560), 34 μs/call (H=6912). -0.3 ms/step.
- **Sharded rms_norm**: `ttnn.rms_norm`이 width-sharded 입력에 대해 자동으로 multicore 커널 디스패치. `RMSNorm.enable_sharded_fast_path()`로 91개 non-fused norm 모두에 적용 (reshard → rms_norm → sharded_to_interleaved). 56 → 33 μs/call (-41%), trace kernel 14.7 → 12.1 ms (-2.6 ms).
- **Multicore argmax (trace-primed)**: `ttnn.argmax(..., use_multicore=True)`로 vocab-dim reduction을 코어 그리드 전체로 병렬화. 첫 호출 시 write가 trace 내에서 거부되므로 warmup 경로에서 한 번 호출해 상태 초기화. 1.99 → 0.076 ms (26×), 단계당 -1.68 ms.
- **Fused RMSNorm + ternary_matmul (QKV)**: RMSNorm이 matmul 커널 안에서 실행됨. QKV만 적용 (o_proj/gate_up/down_proj는 trace-level 회귀로 기본 L1-norm 경로 유지)
- **BFP2 exp init을 첫 블록 DMA와 병렬화**: ~4 µs의 L1 stores를 ~30 µs DMA latency 뒤에 숨김
- **Gamma DMA 단일 배치**: fused reader들이 Kt 번의 barrier를 1번으로 축소
- **Phase 1a tile-regs 병합**: square + accumulate를 `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>`로 한 cycle에 처리
- **bfp4 production (30.15) 대비 +89%**, storage는 절반 (~600MB vs ~1.2GB)
- **L1 norm→matmul 경로**: 121개 RMSNorm 중 QKV 제외한 ~97개 출력이 L1에 유지
- **In-trace greedy argmax**: sample_token 6.37→0.16 ms
- **lm_head bfp4**: DRAM BW 절반 (1.81→0.75 ms, PCC 0.9992 vs bf16)
- **ROW_MAJOR argmax**: TILE 4.39→RM 2.26 ms
- **QKV/O_proj/FFN 전부 packed_ternary**
- **K-block pipelining**: in0_block_w = Kt/2, cb0/cb1 double-buffer
- **True 2-bit DRAM** storage (BFP2_b + L1 exp 합성)
- **Activation multicast** 경로 동작 (sender/receiver on BRISC/NOC_0). fused-norm mcast 변형 `reader_ternary_norm_sender.cpp` / `_receiver.cpp` 도 작동.
- **정확도:** PCC 0.975 vs HF, fused vs 비-fused 경로 PCC > 0.9997 (max_diff 4–6 at std≈26)
- **Dual RoPE:** fused / manual, `--no-fused-rope` 옵션

### 성능 진화 요약
| 단계 | t/s | 주요 변화 |
|---|---:|---|
| 0. baseline (1 core scalar) | ~0.9 | — |
| 1. 2D core grid (108) | ~7 | — |
| 2-5. LUT+Bfp2_b+matmul_block | ~21 | HW unpack |
| 7. true 2-bit DRAM | ~21 | 256B/tile + L1 exp |
| 8. dual-NoC split | ~30 | — |
| 9. cb1 exp probe 버그 fix + nt_per_core≥2 | 31.4 | NaN 버그 해결 |
| 10. activation multicast + RISC swap | 32.41 | sender on BRISC/NOC_0 |
| 11. fused QKV + attn projections → packed_ternary | 33.61 | 전체 attention 경로 2-bit |
| 12. in0_block_w=Kt/2 K-block pipelining | 34.81 | cb0/cb1 자동 double-buffer, BRISC sender가 상쇄 제거 |
| 13. Phase 1a: cos/sin hoist 밖으로 | 35.38 | 60 ops/step 제거 (to_memory_config 30×2→1×2) |
| 14. QKV scale fold (B') | 35.94 | 3 multiplies/layer 제거. q*k scale → SDPA, v scale → attn_sub_norm RMSNorm 흡수 (90 ops/step) |
| 15. qkv_layout_reshape skip | 37.01 | decode 입력이 이미 4D TILE → to_layout(RM) + to_layout(TILE) 건너뛰기 (60 ops/step) |
| 16. kv_update_prep 제거 | 37.52 | paged_update_cache가 RoPE 출력을 직접 수용 → typecast + to_memory_config 전부 제거 (60-120 ops/step) |
| 17. **in-trace greedy argmax** | **41.39** | sample_token 6.37→0.16 ms. argmax를 trace 안에 포함, host round-trip 제거 |
| 18. **lm_head bfp4** | **43.17** | bf16→bfp4 (657→164 MB). DRAM BW 절반, PCC 0.9992 vs bf16 무손실 |
| 19. **ROW_MAJOR argmax** | **47.68** | to_layout(RM) + argmax(RM). TILE 4.39→RM 2.26 ms. branchless bf16 비교 커널 |
| 20. **streaming decode O(1)** | **48.18** | tokenizer.decode 전체 시퀀스 재디코딩 → per-token O(1) 변경 |
| 21. **L1 norm→matmul + SDPA L1** | **51.58** | 120 norm→matmul 중간값 + SDPA 출력을 L1 유지. DRAM 왕복 제거 |
| 22. **Fused RMSNorm + ternary_matmul (QKV)** | **~50.4 (p50 17.5ms = 57.1 t/s)** | QKV 경로의 RMSNorm이 matmul 커널 안에서 실행. BFP2 exp init을 첫 블록 DMA와 병렬화. Gamma DMA 단일 배치. Phase 1a tile-regs 병합. |
| 23. **Multicore argmax (trace-primed)** | **55.83 (p50 15.5ms)** | `ttnn.argmax(use_multicore=True)` vocab reduction을 110 cores로 병렬화. 첫 호출 write가 trace capture 내에서 거부되므로 warmup 경로에서 호출해 상태 초기화. 1.99→0.076 ms (26×) 단독 측정, 전체 -1.68 ms. |
| 24. **Sharded multicore rms_norm** | **62.66 (p50 13.5ms)** | `ttnn.rms_norm`이 width-sharded 입력에 대해 자동 multicore 경로. 91 non-fused norms 모두 sharded 경로 사용 (reshard→rms→desharded). 56→33 μs/call, trace kernel -2.6 ms. |
| 25. **Shard grid tuning** | **63.46 (p50 13.2ms)** | H=2560 y=2x=8 (16 cores), H=6912 y=3x=8 (24 cores). 측정 기반 코어 그리드 최적화. -0.3 ms/step. |
| 26. **Split lm_head (4-way)** | **68.57 (p50 11.7ms)** | vocab matmul 4 chunk으로 쪼개 per-core config 최적화. 2197→722 μs (-67%), trace kernel -1.5 ms. |
| 27. **LM head chunks L1** | **70.51 (p50 11.6ms)** | split lm_head chunk 출력을 L1 유지. -40 μs/step. |
| 28. **nlp_concat_heads_decode** | **70.98 (p50 11.3ms)** | SDPA 뒤 `ttnn.reshape`를 전용 kernel로 대체. reshard 포함해도 38→29 μs/layer (-0.27 ms/step). |

---

## Track A: Packed Ternary Matmul — ✅ DONE

### 완료
- [x] 2-bit pack/unpack roundtrip 검증
- [x] BFP2_b HW unpack (Tensix UNPACKER)
- [x] True 2-bit DRAM (mantissa-only, L1-synthesized exp)
- [x] 2D multi-core (108 cores gate_up, 40 cores mcast for down_proj/o_proj)
- [x] Dual-NoC split (BRISC activation, NCRISC weight+output)
- [x] `matmul_block` compute (ct_dim=nt_per_core)
- [x] `nt_per_core ∈ [2, 8]` heuristic (amortises call overhead)
- [x] **Activation multicast** via rectangular layout + sender/receiver
- [x] `SetCommonRuntimeArgs` for shared DRAM addresses
- [x] FFN 통합: gate_up, down_proj, o_proj 모두 `ttnn.experimental.ternary_matmul` 사용
- [x] Attention 통합: q/k/v/o_proj + fused QKV 모두 packed_ternary
- [x] **K-block pipelining** (`in0_block_w = Kt/2`) — cb0/cb1 자동 double-buffer
- [x] bench_batch32 end-to-end 검증

### 설계 메모
- **RISC 배치**: activation reader/sender/receiver → **BRISC / NOC_0**
  (production `reader_bmm_tile_layout_in0_sender_padding` 패턴).
  NCRISC에서 시작하는 multicast는 Blackhole에서 dispatcher CQ를 손상시킴.
- **Writer (weight reads + cb1 exp init + output writes)** → NCRISC / NOC_1
- **cb1 exp init**: 매 launch마다 재초기화 (L1 probe cache 불가 — Blackhole
  un-inited L1이 0x7F 패턴과 충돌 가능)
- **mcast 활성 조건**: rectangular layout의 core count ≥ L-shape core count
  AND ≥ 2. gate_up은 108 L-shape 유지, down_proj/o_proj는 40-core 5×8 rect.
- **K-block pipelining**: cb0/cb1은 Kt 크기 유지, in0_block_w만 Kt/2로 줄이면
  자동으로 2-block double-buffer가 됨. BRISC sender 전환 이전에는 perf-neutral
  이었는데, dual-NoC가 실제로 병렬 동작하게 되면서 DMA/compute overlap이 발현됨.
  Kt/4는 sync 오버헤드로 Kt/2와 noise 내 동등.

---

## Track B: Zero-Tile / Sparsity Skip — ✗ 불가 (실측 검증 완료)

### 조사한 각도
1. **All-zero 32×32 tiles**: 전 projection 0%. 값 분포가 1/3:1/3:1/3 균일이라
   1024개 값이 모두 0일 확률이 (1/3)^1024 ≈ 0.
2. **All-zero BFP2 groups (16v)**: gate/up/down/o_proj에서 4.7%, q/k/v에서
   0~3%. Tensix `matmul_block`이 tile 단위라 group skip 불가능.
3. **Dead 32-row/col slabs**: 전부 0/…. 개별 dead rows/cols는 있지만 32
   경계에 정렬되지 않음.
4. **FFN 구조적 sparsity (lossless 조건)**:
   `(gate_row_dead ∧ up_row_dead) ∨ down_col_dead`인 intermediate 채널.
   - 30 layers 평균: 334 channels (4.8%)
   - **Layer 1: 3016 channels (43.5%)** — 놀라운 값, 학습 중 early layer가
     intermediate의 절반을 사용하지 않도록 압축됨
   - Layer 2/3도 27%/15%
   - 전 레이어 공통 dead 인덱스: 0 → per-layer 독립 pruning 필요

### 구현 시도 결과 (2026-04-15)
`compute_ffn_prune_mask` + per-layer `intermediate_size` 축소 구현, tile=128
(Nt가 8의 배수 되도록) alignment, gate_up/down_proj/ffn_sub_norm 재생성.

| 지표 | Baseline | With FFN pruning | Δ |
|---|---:|---:|---:|
| 7-run mean | 35.00 t/s | 35.05 t/s | +0.05 |
| 7-run trimmed | 34.81 t/s | 35.08 t/s | +0.27 |
| Trace capture | ~3s | ~20s | +17s (per-layer shapes → program cache miss ×60) |

**결론: noise 수준 (σ ≈ 0.7 t/s)**. 출력은 bfp4와 token-level 일치 → lossless
검증됨. 수학적으론 정확하지만 현재 decode가 core-count-limited도 DRAM-BW-
bound도 아니라서 Nt 축소가 wall-clock에 전환되지 않음.

**분석 스크립트** (재현용, 미커밋 local):
- `analyze_sparsity.py` — tile/group/row/col-level zero 통계
- `analyze_dead_indices.py` — per-layer lossless dead index count + 32 정렬

**재검토 조건:** decode가 BW-bound가 되는 shape(더 큰 batch, 더 긴 context)이
나 hardware(더 낮은 DRAM bandwidth)로 전환될 경우 재고 가치 있음.

---

## Track C: multi_core_reuse_optimized 포팅 — ⚠️ Reduced priority
**목표:** Activation + weight 둘 다 multicast + 2D core grid reuse 패턴으로
매트멀 팩토리 재작성. DRAM 대역폭을 `Kt × Nt` → `Kt + Nt` 수준으로 축소.

**예상 이득 (재평가):** +1~2 t/s (L1 norm이 activation DRAM 증폭을 이미 해결.
mcast heuristic 실험에서 gate_up 72 rect → 50.77 t/s vs 108 L-shape 51.12 확인.
남은 이득은 weight column sharing만으로 제한됨.)

**범위:**
- tt-metal 프로덕션 `matmul_multi_core_reuse_optimized` 약 3300줄 참고
- ternary(BFP2_b) 포맷 + mantissa-only DRAM 레이아웃 유지
- 2D mcast: in0는 row-wise mcast, in1(weight)는 column-wise mcast
- 기존 Track A의 단순 팩토리와 공존 (shape 기반 선택)
- Blackhole NCRISC mcast 제약 재확인 필요 (receivers도 BRISC?)

**규모:** 독립 프로젝트 (며칠~주 단위), Track A와는 분리된 작업.
구현 가이드는 별도 설계 문서 필요.

**선행 조건 / 리스크:**
- p150a harvesting (2 Tensix reserved) 하에서 row/column mcast 사각형 유효성
- Weight mcast sender도 BRISC에 배치 필요 (NCRISC mcast CQ 손상)
- 현재 팩토리의 core heuristic (`nt_per_core ∈ [2, 8]`)과 재구성된 core
  grid의 호환성 — 특히 non-divisor shapes 처리

---

## Track D: Fused RMSNorm + ternary_matmul — ✅ DONE (QKV only)

### 완료 (2026-04-17)
- [x] Fused compute kernel `fused_norm_ternary_mm_compute.cpp`
  - Phase 1 RMSNorm: `REDUCE_ROW` + `mul_tiles_bcast_cols` (scalar 위치 문제
    해결), gamma는 `mul_tiles_bcast_rows` (batch-32 행 브로드캐스트)
  - Phase 1a tile-regs 병합: square+accumulate 1 cycle
  - Phase 2 normalize + gamma
  - Phase 3 matmul (기존 matmul_block)
- [x] Unicast fused reader `reader_ternary_norm_act.cpp`
- [x] Mcast fused sender/receiver `reader_ternary_norm_{sender,receiver}.cpp`
  (cb_raw 대상, 모든 Kt 타일을 한 번의 mcast로)
- [x] BFP2 exp init을 첫 weight-block DMA와 병렬화 (writer)
- [x] Gamma DMA 단일 배치 + 단일 barrier (Kt≤128 제약: cb_gamma = Kt tiles)
- [x] Program factory: fuse_norm 경로, Mt/Kt/Nt runtime args, gamma accessor
- [x] `ttnn.experimental.ternary_matmul`에 `norm_weight`/`norm_epsilon` 추가
- [x] `Linear.__call__`이 norm_weight 통과
- [x] `generator_batch32.py`: QKV에서 fused 경로 사용

### 설계 메모
- o_proj/gate_up/down_proj는 trace-level에서 회귀. baseline이 이미
  L1 norm→matmul 경로라서 fused 커널의 단일 launch 이득이 상쇄됨.
  QKV는 residual stream이 DRAM에서 오므로 ($cb\_raw$ DMA는 어차피 필요)
  fused가 net positive. 자세한 실험 결과는 `memory/project_fused_norm_result.md`.
- down_proj (K=6912, Kt=216)는 cb_raw + cb_gamma + cb_in0 = 432KB × 3 = 1.3MB로
  L1 초과. Python 레이어에서 `Kt ≤ 128` 가드. 현재는 사용 안 함.

---

## Track E: Attention core-range 분할 — 🚧 계획 단계

### 목표
두 개의 tt-metal fused op가 K/V (또는 Q/K)가 겹치지 않는 코어 범위에 있길
요구합니다:
- `ttnn.experimental.rotary_embedding_llama_fused_qk` (~0.4 ms 예상)
- `ttnn.experimental.paged_fused_update_cache` (~0.3 ms 예상)

결합 ~0.7 ms/step = p50 17.5 → ~16.8 ms = ~60 t/s 도달 가능.

### 차단 원인
`nlp_create_qkv_heads_decode`가 Q/K/V를 모두 같은 코어 그리드에 내보냄
(`attention.py:1039`, `attention.py:1082-1085` 주석 참조).

### 권장 접근
**Option A (post-split re-shard)**: `nlp_create_qkv_heads_decode` 이후
`to_memory_config`로 Q, K, V를 disjoint shard spec에 재배치. Python-only
변경 — tt-metal 수정 없음.

전체 설계 및 성공 기준: `docs/plan_attention_core_range_split.md`.

---

## RoPE
- [x] `rotary_embedding_llama` decode mode 통합 (fused RoPE)
- [x] Q/K weight permutation (HF half-split → TT adjacent-pair)
- [x] Interleaved cos/sin format
- [x] HEIGHT_SHARDED cos/sin inside trace
- [x] 128×128 rotation matrix (prefill adjacent-pair)
- [x] `--no-fused-rope` fallback

---

## bitnet-tt 통합 상태
- [x] **FFN**: gate/up/down → ternary_matmul
- [x] **Attention q/k/v/o_proj**: ternary_matmul
- [x] **Fused QKV (decode)**: ternary_matmul (BFP2_b mantissa-only 저장)
- [x] **Prefill**: HF 출력 일치

---

## 성능 참조 (최종)
| dtype | p50 ms | p50 t/s | min ms | min t/s | decode_tps | storage (2.4B) |
|---|---:|---:|---:|---:|---:|---:|
| **packed_ternary (current, 2026-04-17 +concat_heads_decode, 64 tok)** | **11.3** | **88.5** | **10.7** | **93.5** | **70.98** | **~600 MB** |
| packed_ternary (2026-04-17 +L1 chunks, 64 tok) | 11.6 | 86.2 | 10.9 | 91.7 | 70.51 | ~600 MB |
| packed_ternary (2026-04-17 +split lm_head, 64 tok) | 11.7 | 85.5 | 11.0 | 90.9 | 68.57 | ~600 MB |
| packed_ternary (2026-04-17 +tuned shard grid, 64 tok) | 13.2 | 75.8 | 12.4 | 80.6 | 63.46 | ~600 MB |
| packed_ternary (2026-04-17 +sharded rms_norm, 64 tok) | 13.5 | 74.1 | 12.8 | 78.1 | 62.66 | ~600 MB |
| packed_ternary (2026-04-17 +multicore argmax, 64 tok) | 15.5 | 64.5 | 15.0 | 66.7 | 55.83 | ~600 MB |
| packed_ternary (2026-04-17 fused QKV-norm, 64 tok) | 17.5 | 57.1 | 16.9 | 59.2 | 50.4 | ~600 MB |
| packed_ternary (2026-04-16, 128 tok)             | 17.8    | 56.2    | —       | —       | — | ~600 MB |
| packed_ternary (pre-L1)                          | 19.3    | 51.8    | —       | —       | ~600 MB |
| packed_ternary (pre-session)                     | 25      | 40      | —       | —       | ~600 MB |
| packed_ternary (Track A final)                   | 26      | 38.5    | —       | —       | ~600 MB |
| bfp4 production                                  | 31      | 32.3    | —       | —       | ~1.2 GB |
| bf16                                             | ~62     | ~16     | —       | —       | ~4.8 GB |

측정 조건: batch32 + trace + fused RoPE.

### Trace kernel 내부 분해 (실측 10.4 ms, split lm_head 적용 후)
| 항목 | ms | % |
|---|---:|---:|
| RMSNorm (91× sharded multicore) | ~3.0 | 25% |
| ternary matmul (120×, QKV는 norm 포함) | ~4.0 | 33% |
| argmax RM (to_layout + argmax multicore) | 0.08 | 1% |
| lm_head bfp4 (4-way split) | 0.72 | 6% |
| glue (SDPA, heads, RoPE, slice, add, sharded_to_interleaved) | ~2.5 | 21% |
| non-trace overhead | ~1.6 | 13% |

이 분해는 추정치. 실측은 category-sorted 비-trace profile (`profile_decode.py`)에서
rope/create_qkv_heads/slice_gate_up이 각각 top 3 (~180μs in non-trace).
trace-mode에서는 모든 카테고리가 ~86μs 수준으로 평평함 — 단일 병목 없음.

### L1 메모리 최적화 실험 결과 (2026-04-16)
| 변경 | 결과 | 효과 |
|---|---:|---|
| **norm→matmul 중간값 L1** | **51.12** | **+6.1%** — 120 norm 출력이 L1에서 matmul 입력으로 직접 전달 |
| **+ SDPA 출력 L1** | **51.58** | **+7.1%** — SDPA→reshape→sub_norm 체인이 L1에서 동작 |
| ttnn.add(residual) L1 | 47.58 | **-1.2%** — DRAM+L1 혼합 입력으로 역효과 |
| final_norm L1 | 47.41 | **-1.6%** — slice(L1)→matmul(lm_head) 경로 역효과 |
| gate_up mcast heuristic (72 cores) | 50.77 | **-0.7%** — L1 norm이 이미 activation 증폭 해결 |
| lm_head matmul 출력 L1 | FAIL | trace 중 L1 할당 초과 (8.2MB TILE output) |

### 미래 커널 레벨 최적화 (C++ 필요)
| 최적화 | 예상 효과 | 상태 |
|---|---|---|
| RMSNorm+matmul 커널 퓨전 | -0.3ms (QKV) / 나머지 회귀 | ✅ QKV 적용 완료 (Track D) |
| Attention core-range 분할 → `rotary_embedding_llama_fused_qk` + `paged_fused_update_cache` | -0.7ms 예상 | 🚧 계획 단계 (Track E, 설계 문서: `docs/plan_attention_core_range_split.md`) |
| lm_head+argmax 커널 퓨전 | -1~2ms | ✅ 사실상 해결 (multicore argmax로 1.68 ms 회수) |
| Track C weight column sharing | +1~2 t/s (재평가) | 재평가 — L1 norm이 activation 증폭을 이미 해결 |

---

## 디버깅 참조

### Device / build
- Device reset: `~/.tenstorrent-venv/bin/tt-smi -r 0` (BH 동작 확인됨)
  또는 `echo 1 | sudo tee /sys/bus/pci/devices/0000:c1:00.0/{remove,rescan}`
- `_ttnn.so` 복사 필수: `cp build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so`
- 증분 빌드: `cmake --build build_Release -j$(nproc)`
- SDPA grid: CoreCoord(8,4) for BH + kv_heads=5 + batch32

### Blackhole mcast 주의사항
1. **Multicast sender는 반드시 BRISC/NOC_0**. NCRISC에서 `noc_async_write_multicast`
   또는 `noc_semaphore_set_multicast`를 호출하면 dispatcher CQ가 손상됩니다
   (rectangle 크기/연속성과 무관). Production matmul이 in0 sender를 BRISC에
   두는 이유.
2. **cb1 exp init probe 사용 금지**. Un-inited L1이 우연히 0x7F 패턴과 같을 수
   있어 false-positive skip → 다른 슬롯이 garbage → NaN.
3. **`worker_core_from_logical_core`는 2 corner로 사용 가능** — p150a의
   harvesting (x=8,9 reserved)은 multicast 레벨에서 올바르게 처리됩니다.

### 실행
```bash
ssh TT
cd ~/bitnet-tt && source ~/.tenstorrent-venv/bin/activate
TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS=BR,NC,TR,ER \
BITNET_TT_TRACE_REGION_SIZE=200000000 \
python bench_batch32.py --dtype packed_ternary --max-new 128
```
