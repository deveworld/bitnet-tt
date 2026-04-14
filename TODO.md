# TODO

## 현재 상태 (2026-04-15)

### 달성 — Track A 완료
- **34.81 t/s** decode (batch32 + trace + fused RoPE, 7-run trimmed avg) — Blackhole p150a
- **bfp4 production (30.15) 대비 +15.5%**, storage는 절반 (~600MB vs ~1.2GB)
- **QKV/O_proj/FFN 전부 packed_ternary**: attention 경로도 2-bit
- **K-block pipelining**: in0_block_w = Kt/2로 cb0/cb1 double-buffer → DMA/compute overlap
- **True 2-bit DRAM** storage (BFP2_b 포맷 + L1 exp 합성)
- **Activation multicast** 경로 동작 (sender/receiver on BRISC/NOC_0)
- **Prefill 정확도:** HF reference 일치
- **matmul 정확도:** 모든 shape corr > 0.9999 vs bf16
- **Dual RoPE:** fused (~32 t/s) / manual (~21 t/s), `--no-fused-rope` 옵션

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
| 12. **in0_block_w=Kt/2 K-block pipelining** | **34.81** | cb0/cb1 자동 double-buffer, BRISC sender가 상쇄 제거 |

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

## Track B: Zero-Tile Skip — ✗ 불가
- 0% all-zero tiles (분석 완료, 구현 가치 없음)

---

## Track C: multi_core_reuse_optimized 포팅 — 🔄 Future
**목표:** Activation + weight 둘 다 multicast + 2D core grid reuse 패턴으로
매트멀 팩토리 재작성. DRAM 대역폭을 `Kt × Nt` → `Kt + Nt` 수준으로 축소.

**예상 이득:** +2~4 t/s (추정, 대역폭 축소 비율 × 현재 DMA 비중)

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
| dtype | avg t/s | p50 ms | storage (2.4B) |
|---|---:|---:|---:|
| **packed_ternary (K-pipe + full attn)** | **34.81** | **26** | **~600 MB** |
| bfp4 production | 30.15 | 31 | ~1.2 GB |
| bf16 | ~16 | ~62 | ~4.8 GB |

측정 조건: batch32 + trace + fused RoPE + 128 tokens + LoFi.

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
