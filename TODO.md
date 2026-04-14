# TODO

## 현재 상태 (2026-04-14)

### 달성 — Track A 완료
- **32.41 t/s** decode (batch32 + trace + fused RoPE, 7-run avg) — Blackhole p150a
- **bfp4 production (30.08) 대비 +7.7%**, storage는 절반 (~600MB vs ~1.2GB)
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
| 10. **activation multicast + RISC swap** | **32.41** | sender on BRISC/NOC_0 |

---

## Track A: Packed Ternary Matmul — ✅ DONE

### 완료
- [x] 2-bit pack/unpack roundtrip 검증
- [x] BFP2_b HW unpack (Tensix UNPACKER)
- [x] True 2-bit DRAM (mantissa-only, L1-synthesized exp)
- [x] 2D multi-core (108 cores gate_up, 40 cores mcast for down_proj/o_proj)
- [x] Dual-NoC split (BRISC activation, NCRISC weight+output)
- [x] `matmul_block` compute (ct_dim=nt_per_core, single K block)
- [x] `nt_per_core ∈ [2, 8]` heuristic (amortises call overhead)
- [x] **Activation multicast** via rectangular layout + sender/receiver
- [x] `SetCommonRuntimeArgs` for shared DRAM addresses
- [x] FFN 통합: gate_up, down_proj, o_proj 모두 `ttnn.experimental.ternary_matmul` 사용
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

### 향후 여지 (선택)
- [ ] PACKER_L1_ACC multi-K-block pipelining (이전에 deadlock)
- [ ] `multi_core_reuse_optimized` factory 포팅 (3300줄 재작성)
- [ ] QKV 경로 packed_ternary 전환 (현재 fused는 bfp4 사용)

---

## Track B: Zero-Tile Skip
- **불가:** 0% all-zero tiles (분석 완료)

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
- [x] **O_proj**: ternary_matmul
- [x] **Prefill**: HF 출력 일치
- [ ] **QKV (decode fused)**: 현재 bfp4. `uint32` TILE로 fuse pack이 안 되어
  ternary 전환 시 per-proj 경로가 필요 (trade-off 있음)

---

## 성능 참조 (최종)
| dtype | avg t/s | p50 ms | storage (2.4B) |
|---|---:|---:|---:|
| **packed_ternary (mcast)** | **32.41** | **29** | **~600 MB** |
| bfp4 production | 30.08 | 31 | ~1.2 GB |
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
