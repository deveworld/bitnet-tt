# TT-NN BitNet-TT Core Memo (Compressed)

핵심 패턴과 현재 동작하는 설정만 정리. 불필요한 반복/실패 로그 제거. 목표: 안정 추론, Throughput 개선.

## 0. 현황 & 목표
- Stable 경로: **Alpha baseline (`cd995ba`) + HiFi2 compute kernel (`4b50353`)** → **~8.5 t/s** (8.02-9.69 t/s 범위) on batch=32.
- 단일 사용자 타겟: ≥33 t/s (tt_transformers Llama 3.1 8B per-user 기준).
- DRAM-bandwidth가 병목인 decode에 집중. Trace/Multi-CQ, DRAM-sharded matmul, 경량 커널(HiFi2/LoFi) 활용.

---

## 1. Device & Tensor 기본
- Device open/close
  ```python
  import ttnn
  with ttnn.manage_device(0, l1_small_size=8192) as device:
      ...
  ```
- PyTorch ↔ TT-NN
  ```python
  tt = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16,
                       layout=ttnn.TILE_LAYOUT, device=device,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
  torch_out = ttnn.to_torch(tt)  # sync barrier
  ```
- Direct create (device): `ttnn.zeros/ones/full/rand/arange(...)`
- Async 기본: 대부분 op은 비동기. `to_torch()` or `ttnn.synchronize_device(device)`로 동기화.
- L1 수동 해제: `ttnn.deallocate(tensor)` (intermediate L1 공간 확보).

## 2. Layout / Dtype / Memory 요약
- Layout
  - `ROW_MAJOR_LAYOUT`: 로딩/간단 연산/embedding 입력. 폭 정렬 필요.
  - `TILE_LAYOUT`: 성능 핵심. **matmul, linear, attention 필수**. 자동 pad(32x32 tile).
- Dtype (권장)
  - `bfloat16`: 기본.
  - `bfloat8_b` (BFP8): TILE 필수. weight/중간에 메모리 절약, 정확도 OK.
  - `bfloat4_b` (BFP4): TILE 필수. 속도/메모리 최우선, 정확도 민감한 구간 주의.
  - `uint32`: index/position용 (embedding, KV cache idx).
- MemoryConfig
  - `DRAM_MEMORY_CONFIG`: weight/큰 텐서 기본.
  - `L1_MEMORY_CONFIG`: 작은 중간 결과. 필요 시 `ttnn.deallocate`.
  - Sharded: `HEIGHT_SHARDED`(decode QKV/attention), `WIDTH_SHARDED`(matmul weight). 변환: `interleaved_to_sharded`, `sharded_to_interleaved`, `ttnn.to_memory_config`.

## 3. Core Ops 패턴
- Matmul / Linear
  ```python
  kernel_hifi2 = ttnn.WormholeComputeKernelConfig(
      math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
      fp32_dest_acc_en=False, packer_l1_acc=False)

  out = ttnn.linear(
      x, weight, bias=None,
      transpose_a=False, transpose_b=True,            # weight 내장 transpose
      memory_config=ttnn.L1_MEMORY_CONFIG,            # 또는 DRAM/SHARDED
      core_grid=ttnn.CoreGrid(y=8, x=8),              # 배치/모델에 맞춰 조정
      compute_kernel_config=kernel_hifi2)
  ```
- Attention SDPA
  - Prefill: `ttnn.transformer.scaled_dot_product_attention`.
  - Decode: `ttnn.transformer.paged_scaled_dot_product_attention_decode` (paged KV cache + page_table).
- 기타: `ttnn.concat`, `ttnn.reshape`, `ttnn.unsqueeze_to_4D`, `ttnn.permute_dim` (필요 최소).

## 4. Transformer / Attention 흐름 (BitNet-TT)
**Prefill (compute-bound, DRAM interleaved)**  
1) Embedding (row-major input → TILE)  
2) QKV matmul (BF16/HiFi4)  
3) `split_query_key_value_and_split_heads` (prefill 전용 helper)  
4) RoPE: pre-upload된 cos/sin → `ttnn.experimental.rotary_embedding_llama`  
5) SDPA (`scaled_dot_product_attention`)  
6) Concat heads → Output proj  
7) MLP (gate+up/down)

**Decode (DRAM-bandwidth-bound, L1/SHARDED)**  
1) QKV projection: DRAM-sharded weight, output L1 WIDTH_SHARDED  
2) `nlp_create_qkv_heads_decode(..., memory_config=HEIGHT_SHARDED)`  
3) RoPE decode (HEIGHT_SHARDED 입력 필수, trans_mat 사용)  
4) KV cache update in-place: `ttnn.experimental.paged_update_cache` 또는 `kv_cache.update_cache_for_token_` (cur_pos tensor 전달)  
5) SDPA decode (paged)  
6) `nlp_concat_heads_decode`  
7) Output proj (DRAM-sharded)  
8) LM head: vocab 조각별 matmul → concat. Prefill 시 마지막 토큰 slice만 계산해 비용 절감.

**KV Cache & RoPE 팁**
- KV cache는 concat 대신 in-place 업데이트로 재할당 방지.
- RoPE: cos/sin matrix를 한 번만 업로드하고 `ttnn.embedding`으로 위치별 조회 후 sharding.
- Trace-friendly: `cur_pos`/`update_idxs`는 리스트 대신 device tensor로 전달.

## 5. BitNet b1.58 모델 핵심
- Config (2B-4T): vocab 128,256 | hidden 2,560 | ff hidden 6,912 | layers 30 | heads 20 | kv_heads 5 | head_dim 128 | act `relu2` | rope_theta 500000 | rms_eps 1e-5 | attention_bias False | tie_word_embeddings False.
- BitLinear (weight ternary {-1,0,1} scaled):
  ```python
  # Weight quant (load-time)
  scale_w = 1.0 / weight.abs().mean().clamp(min=1e-5)
  w_q = (weight * scale_w).round().clamp(-1, 1) / scale_w

  # Activation quant (per token)
  scale_x = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
  x_q = (x * scale_x).round().clamp(-128, 127) / scale_x
  ```
- Attention 특이점: `attn_sub_norm`(RMSNorm) **after** attention before O-proj. MLP도 `ffn_sub_norm`.
- HuggingFace는 분리된 Q/K/V, 공식 GPU는 fused `wqkv`; TT-NN에서도 fuse 고려(성능↑).

## 6. Compute / Memory 세부 설정 (동작 검증됨)
- Layout: compute 경로는 TILE. Embedding 입력/포지션 index는 ROW_MAJOR.
- Memory: weight DRAM(필요 시 DRAM_sharded), 중간 L1. Decode QKV/attn은 **HEIGHT_SHARDED** 입력 필수. Output proj/LM head weight는 DRAM_sharded로 bandwidth↑.
- DRAM-Sharded Matmul: interleaved 대비 **~26%** 개선(190 → 240 GB/s). Decode matmul에 적용.
- CoreGrid: decode 일반 8x8(32 cores) 기준. 모델/배치에 맞춰 y=batch chunk, x=cores.
- Math fidelity 권장
  - Prefill QKV/SDPA: HiFi4 + BF16 (정확도).
  - Decode QKV/SDPA: HiFi2 + BFP8 (속도/정확도 균형).
  - MLP FF1/FF3: LoFi + BFP4 (속도), FF2: HiFi2 + BFP8.
- LM Head: vocab 조각별 matmul + concat; prefill에서 마지막 토큰만 slice 처리.
- Transpose 비용 제거: `ttnn.linear(..., transpose_b=True)` 사용.

## 7. Trace & Multi-CQ (성능 핵심)
- 1회 컴파일 실행 후 같은 입력 shape로 trace capture → 이후 `ttnn.execute_trace(..., blocking=False)` 반복.
- 입력 텐서는 device에 유지, host 값만 in-place 갱신(`copy_host_to_device`).
- CQ 사용 예: CQ0=compute, CQ1=H2D copy 로 overlap. Trace와 병행 시 가장 큰 이득.

## 8. 결과/시도 요약 (1000줄 이하 유지용 간략화)
| 시도 | 커밋 | 결과 | 속도/비고 |
|------|------|------|-----------|
| Alpha baseline | `cd995ba` | ✅ | **8.68 t/s** |
| `_forward_simple` (batch=32) | `fd7a1d2` | ✅ 작동 | 0.80 t/s (느림) |
| HEIGHT_SHARDED decode | `105f828` 등 | ❌ | 실패 원인: BitNet `num_heads=20`가 CoreGrid 4x8(32 cores) 요구와 불일치 → shard height mismatch |
| Fused QKV + LM Head slice | `0e97f48` | ❌ | `ttnn.slice` 인수 오류로 출력 손상 |
| **HiFi2 kernel 적용** | `4b50353` | ✅ | **8.02-9.69 t/s**, 품질 정상 |

현재 상태: Alpha 경로 + HiFi2 compute kernel을 기본으로 사용, DRAM-sharded matmul/LM head slice/trace는 추가 적용 대상.

## 9. 앞으로 확인/적용
1) DRAM-Sharded matmul 전체 decode 경로에 적용 (QKV, O-proj, LM head).  
2) Compute kernel 튜닝: HiFi2 기본, LoFi는 MLP만 실험.  
3) LM head 최적화: prefill 마지막 토큰 slice, weight chunk 병렬화.  
4) Fused QKV/게이트: `ttnn.slice` 키워드 인수(`slice_start/slice_end`)로 재실험.  
5) Trace + Multi-CQ를 decode loop에 온전히 적용(입력 tensor 재사용).  
6) HEIGHT_SHARDED 대안: head 수 32 맞추기 어려우므로 BitNet 전용 shard geometry 설계 여부 검토.

## 10. 빠른 API 참조 (핵심만)
- 변환: `ttnn.to_layout(t, layout)`, `ttnn.to_memory_config(t, mem_cfg)`, `interleaved_to_sharded`, `sharded_to_interleaved`.
- Slice: `ttnn.slice(t, slice_start=[...], slice_end=[...])` (키워드 필수, rank 맞추기).
- Trace: `begin_trace_capture(device, cq_id) → end_trace_capture(...) → execute_trace(..., blocking=False) → release_trace(...)`.
- Tensor 관리: `ttnn.deallocate(t)`, `ttnn.copy_host_to_device_tensor(host, device_tensor)`, `ttnn.synchronize_device(device)`.
- Print/Debug: `ttnn.set_printoptions(profile="short")`, 비교는 PyTorch golden과 대조.

---

이 문서는 중복/실패 로그를 제거한 핵심 요약본이다. 새로운 실험 추가 시 성공 설정은 상세 기록, 실패는 원인 한 줄만 남길 것.
