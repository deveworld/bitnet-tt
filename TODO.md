# TODO

## 현재 상태 (2026-04-12)

### 달성
- **30.1 t/s** decode (batch32 + trace + fused RoPE) — Blackhole p150a
- **Prefill 정확도:** HF reference와 일치 ("Paris. Paris is the largest city in France...")
- **ternary_matmul op:** multi-core 4.5ms (10 cores), corr=0.999 — single-core 850ms에서 190× 속도 향상
- **Dual RoPE 지원:** fused (30 t/s) / manual (16 t/s), `--no-fused-rope` 옵션
- **bfp4 weights:** lossless ternary 저장, post-matmul scale

### 미해결
- ternary_matmul multi-core (4.5ms) vs bf16 matmul (0.1ms) — 45× gap
- decode quality: ~8 token 후 drift (2.4B 모델 + bf16 precision 한계, 버그 아님)

---

## 방향

- `ttnn.experimental.ternary_matmul`을 multi-core로 최적화하여 bf16 matmul에 근접시킨다.
- 최적화된 ternary_matmul을 bitnet-tt FFN 경로에 통합한다.
- batch32 decode pipeline의 수치 정밀도를 개선한다 (sdpa_decode verified correct, issue is accumulated bf16 noise).

## Track A: Packed Ternary Matmul

### 완료
- [x] 2-bit pack/unpack roundtrip 검증
- [x] reader_ternary_fused.cpp 하드웨어 검증 (BH)
- [x] ttnn.experimental.ternary_matmul 등록 + nanobind
- [x] Simple program factory (single-core) 정확도 검증
- [x] Multi-core program factory (N-dimension 병렬화, 10 cores)
- [x] BitNet 실제 크기 테스트 (32×2560×2560, 32×2560×6912)

### 다음
- [ ] 2D core grid (M×N 병렬화)
- [ ] Activation multicast (1 core가 A 읽어서 broadcast)
- [ ] Double buffering (DRAM read와 compute overlap)
- [ ] FFN 통합: ternary_matmul → bitnet-tt Linear 교체

### 성능 참조
| 구현 | 32×2560×2560 | cores | corr |
|------|-------------|-------|------|
| bf16 matmul | 0.1ms | 110 | baseline |
| ternary_mc | 4.5ms | 10 | 0.999743 |
| ternary single | 850ms | 1 | 0.999743 |

### 병목 분석
- 10 cores vs 110 cores → 11× gap
- reader per-core: 전체 K dimension DRAM read + unpack → core간 중복
- compute pipeline stall: read와 compute가 순차적 (no overlap)

## Track B: Zero-Tile Skip
- **불가:** 0% all-zero tiles (분석 완료)

## RoPE 최적화
- [x] rotary_embedding_llama decode mode 통합 (fused RoPE)
- [x] Q/K weight permutation (HF half-split → TT adjacent-pair)
- [x] Interleaved cos/sin format
- [x] HEIGHT_SHARDED cos/sin → inside trace
- [x] 128×128 rotation matrix for prefill adjacent-pair RoPE
- [x] `--no-fused-rope` fallback 옵션

## bitnet-tt 통합 우선순위

1. **FFN (최우선):** gate/up/down projection에 ternary_matmul 적용
2. **prefill QKV:** 긴 sequence에서 packed ternary 비교
3. **decode QKV:** trace + paged cache 호환성 확인 필요

## 디버깅 참조

- Device reset: PCI remove + rescan (`/sys/bus/pci/devices/0000:c1:00.0/`)
- CoreRange: harvested core 회피 — 개별 CoreRange per core 사용
- `_ttnn.so` 복사 필수: `cp build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so`
- SDPA grid: CoreCoord(8,4) for BH + kv_heads=5 + batch32
