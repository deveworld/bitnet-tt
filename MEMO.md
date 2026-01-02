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

### KV Cache & RoPE 팁

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
| --- | --- | --- | --- |
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

## 11. 커스텀 커널 개발 참고사항

### 11.1 Blackhole p150a 하드웨어 특성

| 항목 | Wormhole N150 | Blackhole p150a |
| --- | --- | --- |
| Tensix 코어 | 8x10 (8x8 compute) | 14x10 (**13x10 compute = 130 cores**) |
| L1 (per core) | 1464 KB | 1464 KB + **Data Cache** (4×16B) |
| DRAM | 12 banks × 1GB | **8 banks × ~4GB = ~32GB** |
| NoC Alignment | R:32B, W:16B | R:**64B**, W:16B |
| Multicast | Rectangular | Rectangular + **Strided + L-shaped** |

**L1 Data Cache (BH only)**:

```cpp
// 커널에서 활성화/비활성화
set_l1_data_cache<true>();   // 활성화
invalidate_l1_cache();       // 명시적 무효화 (권장)
set_l1_data_cache<false>();  // 커널 종료 전 비활성화
```

### 11.2 TT-Metal 커널 구조

```text
Program (하나의 Op)
├── Reader Kernel (RISC0) - DRAM/L1에서 데이터 읽기
├── Writer Kernel (RISC1) - L1/DRAM에 데이터 쓰기
└── Compute Kernel (RISC2,3,4) - Tile 기반 연산

각 Tensix 코어에서 병렬 실행, Circular Buffer로 동기화
```

**Circular Buffer & Double Buffering**:

```cpp
// Reader: 데이터 읽고 CB에 push
cb_reserve_back(cb_id, num_tiles);
// ... noc_async_read ...
noc_async_read_barrier();
cb_push_back(cb_id, num_tiles);

// Compute: CB에서 pop하고 연산
cb_wait_front(cb_id, num_tiles);
// ... compute ...
cb_pop_front(cb_id, num_tiles);
```

### 11.3 BitNet INT8×INT2 커널 패턴 (공식 GPU 구현)

**핵심 연산**:

```text
Output = (INT8_activation × INT2_weight) × scale_activation × scale_weight
```

**Weight Packing (2-bit → INT8)**:

```python
# 4개 2-bit 값을 1개 int8에 저장
# weight: [-1, 0, 1] → [1, 2, 3] (offset by 2)
# 압축: N×K → N×(K/4)
packed = w0 | (w1 << 2) | (w2 << 4) | (w3 << 6)
```

**GPU 커널 구조** (ladder_int8xint2_kernel):

```cpp
// 1. Weight decode: INT2 → INT8 (per thread)
decode_i2s_to_i8s(B_reshape_local, B_decode_local, 16);
// LUT 기반 변환: {1,2,3} → {-1,0,1}

// 2. Dot product accumulation
for (int k = 0; k < 4; k++) {
    acc = __dp4a(A_local[k*4], B_decode_local[k*4], acc);
}

// 3. Warp reduction
for (int offset = K_block/2; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(mask, acc, offset);
}

// 4. Scale and output
output[idx] = (bf16)((float)acc / scale_act * scale_weight);
```

### 11.4 TT-Metal 최적화 기법

**DRAM-Sharded Matmul** (decode용, ~26% 대역폭 향상):

```python
# Weight를 DRAM bank에 sharding
memory_config = create_dram_sharded_mem_config(k=K, n=N)
program_config = dram_matmul_config(m=M, k=K, n=N, num_cores=cores)

output = ttnn.linear(
    activation, weights,
    compute_kernel_config=kernel_hifi2,
    program_config=program_config,
    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
)
```

**Math Fidelity**:

| Fidelity | Precision | Speed | 용도 |
| --- | --- | --- | --- |
| HiFi4 | BF16 | 1x | Prefill, 정확도 중요 |
| HiFi2 | BFP8 | **2x** | Decode 기본 |
| LoFi | BFP4 | **3.6x** | MLP (정확도 허용 시) |

### 11.5 FlashDecode 패턴 (SDPA Decode)

```python
# Query shape 변환: heads를 y dimension에 배치
# [bsz, num_q_heads, 1, head_dim] → [1, bsz, num_q_heads, head_dim]
query = query.view(1, bsz, num_q_heads, head_dim)

# bsz * n_kv_heads를 코어에 분산
# n_qh_per_kvh (= num_q_heads // num_kv_heads)는 코어 내에서 처리
attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
    query, keys, values,
    cur_pos_tensor=current_pos,  # causal masking (explicit mask 불필요)
)
```

**성능**: Wormhole에서 최대 180 GB/s 메모리 대역폭 (DRAM 288 GB/s의 ~70%)

### 11.6 BitNet-TT 커스텀 커널 설계 방향

**문제**: `num_heads=20`이 HEIGHT_SHARDED (32 cores) 요구사항과 불일치

**해결 방안**:

1. **Ternary Matmul 커널** (INT8×TERNARY):
   - Weight를 2-bit packed로 저장 (메모리 8x 절감: FP32 → 2bit)
   - Activation은 INT8 quantized
   - DRAM bandwidth 병목 완화

2. **커스텀 QKV Heads Decode**:
   - `num_heads=20`에 맞는 shard geometry
   - CoreGrid: 20 cores (4x5 or 5x4) 또는 padding하여 32 사용

3. **Fused Ops**:
   - QKV projection + split heads 융합
   - RMSNorm + Linear 융합 (sub_norm 패턴)

### 11.7 Trace Capture 패턴 (2-3x 속도 향상)

```python
# 1. Compile run (첫 실행)
output = model_forward(input_tensor, pos_tensor)

# 2. Trace capture
ttnn.synchronize_device(device)
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
output = model_forward(input_tensor, pos_tensor)  # 같은 shape
trace_id = ttnn.end_trace_capture(device, trace_id, cq_id=0)

# 3. Execute trace (반복)
ttnn.copy_host_to_device_tensor(new_input_host, input_tensor)
ttnn.copy_host_to_device_tensor(new_pos_host, pos_tensor)
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
```

**제약사항**:

- Trace 중 텐서 할당/해제 불가 → **in-place KV cache 필수**
- 동일 shape만 가능 → decode에만 적용 (prefill 불가)

### 11.8 참고 자료 경로

| 자료 | 경로 |
| --- | --- |
| LLM 최적화 가이드 | `~/tt-metal/tech_reports/LLMs/llms.md` |
| FlashDecode | `~/tt-metal/tech_reports/FlashAttention/FlashDecode.md` |
| FlashAttention | `~/tt-metal/tech_reports/FlashAttention/FlashAttention.md` |
| Blackhole 가이드 | `~/tt-metal/tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md` |
| BitNet GPU 커널 | `~/BitNet/gpu/bitnet_kernels/` |
| tt_transformers | `~/tt-metal/models/tt_transformers/` |
| SDPA Decode 소스 | `~/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/` |

---

## 12. 커스텀 커널 개발 계획

### 12.1 현황 분석

| 항목 | 현재 | 목표 | 병목 |
| --- | --- | --- | --- |
| 속도 | 8.5 t/s | **30+ t/s** | DRAM bandwidth, kernel dispatch |
| 최적화 경로 | `_forward_simple` (concat) | Trace + In-place | concat이 trace 차단 |
| HEIGHT_SHARDED | ❌ (num_heads=20) | 커스텀 geometry | 32-core 가정 |
| Weight format | BF16 (fp16) | **2-bit packed** | 메모리 대역폭 |

### 12.2 개발 우선순위

#### Phase 1: In-place KV Cache (Trace 활성화) - **가장 효과적**

**목표**: concat → index 기반 업데이트로 Trace 활성화 (2-3x 속도 향상)

**변경 사항**:

```python
# 현재 (concat 기반 - Trace 불가)
new_key = ttnn.concat([self.key_cache, key], dim=2)  # 새 메모리 할당

# 목표 (index 기반 - Trace 가능)
# Pre-allocated cache에 직접 write
ttnn.fill_cache(self.key_cache, key, update_idx=current_pos)
```

**구현 방법**:

1. `paged_update_cache` 대신 간단한 `fill_cache` 사용 (HEIGHT_SHARDED 불필요)
2. KVCache를 max_seq_len 크기로 pre-allocate
3. `current_pos` tensor로 write 위치 지정

**예상 효과**: **8.5 → 17-25 t/s** (2-3x)

#### Phase 2: Ternary Matmul 커널 - **메모리 대역폭 개선**

**목표**: 2-bit weight matmul로 DRAM 읽기 8x 감소

**핵심 아이디어**:

```text
현재: BF16 weight (16 bit/param) → DRAM read = 2.4B params × 2 bytes = 4.8 GB
목표: 2-bit weight (2 bit/param) → DRAM read = 2.4B params × 0.25 bytes = 0.6 GB
```

**TT-Metal 커널 구조**:

```text
ttnn/cpp/ttnn/operations/matmul/ternary_matmul/
├── ternary_matmul.hpp/.cpp          # Python API binding
├── device/
│   ├── ternary_matmul_device_op.hpp/.cpp
│   ├── ternary_matmul_program_factory.hpp/.cpp
│   └── kernels/
│       ├── dataflow/reader_ternary.cpp  # 2-bit unpack
│       ├── dataflow/writer_ternary.cpp
│       └── compute/ternary_matmul.cpp   # INT8 × {-1,0,1}
```

**연산 흐름**:

1. Reader: DRAM에서 packed 2-bit weight 읽기
2. Reader: 4개 2-bit → 4개 INT8 unpack (lookup table)
3. Compute: INT8 activation × INT8 weight (ternary) accumulate
4. Compute: Scale 적용 (activation_scale × weight_scale)
5. Writer: BF16 output 쓰기

**예상 효과**: decode matmul 50-70% 속도 향상

#### Phase 3: 커스텀 QKV Decode (num_heads=20 지원)

**목표**: HEIGHT_SHARDED 최적화 경로 활성화

##### 방법 A: Padding (간단)

```python
# num_heads=20 → 32 (12 dummy heads 추가)
# CoreGrid(y=4, x=8) = 32 cores
padded_heads = 32
# padding overhead: 32/20 = 1.6x 연산량 증가
```

##### 방법 B: 커스텀 Shard Geometry (최적)

```python
# num_heads=20에 맞는 CoreGrid
# 옵션 1: 20 cores (4x5 or 5x4)
# 옵션 2: 40 cores (2 heads/core) - 5x8
# rotary_embedding_llama, sdpa_decode 수정 필요
```

##### 예상 효과**: Phase 1+2 이후 추가 30-50% 향상

### 12.3 구현 로드맵

```text
Week 1: Phase 1 - In-place KV Cache
├── Day 1-2: fill_cache/update_cache 구현 (Python level)
├── Day 3-4: Trace capture 통합
└── Day 5: 테스트 및 벤치마크

Week 2-3: Phase 2 - Ternary Matmul 커널
├── Day 1-3: 2-bit weight packing/unpacking 구현
├── Day 4-7: TT-Metal 커널 작성 (reader/compute/writer)
├── Day 8-10: Python binding 및 ttnn op 등록
└── Day 11-14: BitNet-TT 통합 및 테스트

Week 4: Phase 3 - 커스텀 QKV Decode (선택적)
├── Option A: Padding 방식 (1-2일)
└── Option B: 커스텀 geometry (1주)
```

### 12.4 예상 성능

| Phase | 구현 | 속도 | 대비 현재 |
| --- | --- | --- | --- |
| 현재 | concat + BF16 | 8.5 t/s | 1x |
| Phase 1 | In-place + Trace | 17-25 t/s | **2-3x** |
| Phase 2 | + Ternary Matmul | 25-35 t/s | **3-4x** |
| Phase 3 | + Custom QKV | 30-45 t/s | **4-5x** |

### 12.5 리스크 및 대안

| 리스크 | 확률 | 대안 |
| --- | --- | --- |
| fill_cache가 없음 | 중 | concat → slice로 우회 또는 직접 구현 |
| Ternary 커널 정확도 | 낮 | HiFi2 math fidelity 유지 |
| QKV geometry 충돌 | 높 | Padding 방식으로 fallback |
| Trace 불안정 | 중 | Non-trace 최적화만 적용 |

### 12.6 즉시 시작 가능한 작업

1. **ttnn.fill_cache / ttnn.update_cache API 조사**

   ```bash
   grep -r "fill_cache\|update_cache" ~/tt-metal/ttnn/
   ```

2. **기존 ternary matmul 존재 여부 확인**

   ```bash
   grep -r "ternary\|int2\|2bit" ~/tt-metal/ttnn/cpp/
   ```

3. **Simple index-based cache update 테스트**

   ```python
   # ttnn.copy 또는 slice-based update 테스트
   cache[:, :, pos:pos+1, :] = new_kv
   ```

---

## 13. TT-NN LLM 최적화 패턴 (공식 Tech Report 기반)

> **출처**: `~/tt-metal/tech_reports/LLMs/llms.md` (1871줄), `AdvancedPerformanceOptimizationsForModels.md`, HuggingFace BitNet 구현, 공식 BitNet GPU 커널

### 13.1 성능 벤치마크 (Blackhole p150)

| 모델 | Hardware | Batch | t/s/u | t/s | 비고 |
| --- | --- | --- | --- | --- | --- |
| **Llama 3.1 8B** | p150 | 32 | **33.1** | 1059.2 | **BitNet 목표 기준** |
| Llama 3.1 8B | p100 | 32 | 29.0 | 928.0 | |
| Llama 3.2 1B | n150 (WH) | 32 | 80.5 | 2576.0 | 작은 모델 |
| Mistral 7B | n150 (WH) | 32 | 28.7 | 918.4 | |

**BitNet-TT 현재**: ~8.5 t/s → **목표 33.1 t/s (3.9x 개선 필요)**

### 13.2 핵심 최적화 기법 요약

#### 13.2.1 Prefill vs Decode 분리

| 항목 | Prefill | Decode |
| --- | --- | --- |
| **병목** | Compute-bound | **DRAM bandwidth-bound** |
| **메모리** | DRAM interleaved | L1 sharded (**HEIGHT_SHARDED**) |
| **배치** | 단일 사용자, 전체 시퀀스 | 다중 사용자 (≤32), 토큰 1개씩 |
| **최적화** | Matmul 2D | **DRAM-sharded matmul** |

#### 13.2.2 Attention 구현 패턴

**Prefill (scaled_dot_product_attention)**:

```python
# 1. Fused QKV
xqkv_fused = ttnn.linear(x, wqkv, dtype=ttnn.bfloat16)

# 2. Split heads
Q, K, V = ttnn.experimental.nlp_create_qkv_heads(
    xqkv_fused, num_heads=n_q_heads, num_kv_heads=n_kv_heads,
    transpose_k_heads=False
)

# 3. RoPE (fused op)
Q = ttnn.experimental.rotary_embedding_llama(Q, cos, sin, trans_mat, is_decode_mode=False)
K = ttnn.experimental.rotary_embedding_llama(K, cos, sin, trans_mat, is_decode_mode=False)

# 4. KV Cache fill
ttnn.fill_cache(K_cache, K, batch_idx)
ttnn.fill_cache(V_cache, V, batch_idx)

# 5. SDPA (causal)
attn_out = ttnn.transformer.scaled_dot_product_attention(Q, K_cache, V_cache, is_causal=True)

# 6. Concat heads + output proj
attn_out = ttnn.experimental.nlp_concat_heads(attn_out)
output = ttnn.linear(attn_out, wo)
```

**Decode (paged_scaled_dot_product_attention_decode)**:

```python
# 1. Fused QKV (DRAM-sharded matmul)
xqkv_fused = ttnn.linear(x, wqkv, program_config=dram_sharded_config)

# 2. Split heads (HEIGHT_SHARDED output) ← BitNet 문제점: num_heads=20
Q, K, V = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused, num_heads=20, num_kv_heads=5,
    memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
)

# 3. RoPE decode (HEIGHT_SHARDED 입력 필수)
Q = ttnn.experimental.rotary_embedding_llama(Q, cos, sin, trans_mat, is_decode_mode=True)
K = ttnn.experimental.rotary_embedding_llama(K, cos, sin, trans_mat, is_decode_mode=True)

# 4. In-place KV Cache update (Trace 호환) ← 핵심!
ttnn.experimental.paged_update_cache(keys, K, update_idxs_tensor=cur_pos_tensor, page_table=page_table)
ttnn.experimental.paged_update_cache(values, V, update_idxs_tensor=cur_pos_tensor, page_table=page_table)

# 5. SDPA Decode (cur_pos_tensor로 causal masking)
attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    Q, keys, values,
    cur_pos_tensor=cur_pos_tensor,
    page_table=page_table
)

# 6. Concat heads + output proj
attn_out = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=20)
output = ttnn.linear(attn_out, wo)
```

#### 13.2.3 MLP 최적화 패턴

```python
# Decode mode: DRAM-sharded matmul + L1 sharded output
# FF1/FF3 (column parallel)
w1_out = ttnn.linear(ff_in, w1, program_config=dram_sharded_config,
                     memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
w3_out = ttnn.linear(ff_in, w3, program_config=dram_sharded_config,
                     memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

# Fused SiLU + multiply (BitNet은 squared_relu)
w2_in = ttnn.multiply(w1_out, w3_out, input_tensor_a_activation=ttnn.UnaryOpType.SILU)

# FF2 (row parallel) + reduce
w2_out = ttnn.linear(w2_in, w2, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
```

**BitNet 차이점**:

- `SiLU` 대신 `squared_relu`: `ttnn.relu(x) ** 2`
- `ffn_sub_norm` 추가: gate*up 이후, down_proj 이전에 RMSNorm 적용

#### 13.2.4 Trace Capture 완전 패턴

```python
# 1. Pre-allocate persistent tensors
input_dram_tensor = ttnn.allocate_tensor_on_device(spec, device, dram_sharded_config)
op_event = ttnn.record_event(device, 0)  # Dummy event

# 2. Compile run
ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=1)
input_l1 = ttnn.to_memory_config(input_dram_tensor, l1_sharded_config)
output = run_model(input_l1)

# 3. Trace capture
write_event = ttnn.record_event(device, 1)
ttnn.wait_for_event(0, write_event)
input_l1 = ttnn.to_memory_config(input_dram_tensor, l1_sharded_config)
op_event = ttnn.record_event(device, 0)

input_trace_addr = input_l1.buffer_address()
output.deallocate(force=True)  # Free for input reallocation

tid = ttnn.begin_trace_capture(device, cq_id=0)
output = run_model(input_l1)
input_l1 = ttnn.allocate_tensor_on_device(input_l1.spec, device)
assert input_l1.buffer_address() == input_trace_addr  # 주소 검증
ttnn.end_trace_capture(device, tid, cq_id=0)

# 4. Execute loop (Multi-CQ: CQ0=ops, CQ1=writes)
for _ in iterations:
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(new_host, input_dram_tensor, cq_id=1)
    write_event = ttnn.record_event(device, 1)

    ttnn.wait_for_event(0, write_event)
    input_l1 = ttnn.reshard(input_dram_tensor, l1_sharded_config, input_l1)
    op_event = ttnn.record_event(device, 0)

    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    outputs.append(output.cpu(blocking=False))

ttnn.synchronize_device(device)
```

### 13.3 HuggingFace BitNet 구현 분석

**modular_bitnet.py / modeling_bitnet.py**:

```python
class BitNetMLP(GemmaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.ffn_sub_norm = BitNetRMSNorm(config.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, x):
        # gate * up → ffn_sub_norm → down
        down_proj = self.down_proj(self.ffn_sub_norm(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
        return down_proj

class BitNetAttention(LlamaAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.attn_sub_norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, ...):
        # ... standard attention ...
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.attn_sub_norm(attn_output)  # ← BitNet 특이 구조
        attn_output = self.o_proj(attn_output)
        return attn_output
```

**configuration_bitnet.py 핵심 파라미터**:

```python
vocab_size=128256, hidden_size=2560, intermediate_size=6912,
num_hidden_layers=30, num_attention_heads=20, num_key_value_heads=5,
hidden_act="relu2",  # squared ReLU
rope_theta=500000.0, rms_norm_eps=1e-5,
tie_word_embeddings=False
```

### 13.4 공식 BitNet GPU 커널 분석

**bitnet_kernels.cu** (INT8 × INT2 matmul):

```cpp
// M=1 (decode), 다양한 N/K에 대해 특화된 커널
extern "C" void bitlinear_int8xint2(
    int8_t* input0,      // INT8 quantized activation
    int8_t* input1,      // INT2 packed weight (4 weights per byte)
    __nv_bfloat16* output0,
    __nv_bfloat16* s,    // activation scale
    __nv_bfloat16* ws,   // weight scale
    int M, int N, int K,
    cudaStream_t stream
);

// BitNet 2B-4T 전용 shape들:
if (M == 1 && N == 3840 && K == 2560)  // wqkv (n_heads=20, head_dim=128, 3*hidden)
if (M == 1 && N == 2560 && K == 2560)  // wo, input norm
if (M == 1 && N == 13824 && K == 2560) // w13 (gate+up fused, 2*intermediate)
if (M == 1 && N == 2560 && K == 6912)  // w2 (down_proj)
```

**model.py** (공식 BitNet GPU 구현):

```python
class BitLinearKernel(nn.Module):
    def __init__(self, in_features, out_features):
        # weight: [out, in//4] INT8 (4 weights packed)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features//4, dtype=torch.int8))
        self.weight_scale = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))

    @torch.compile
    def quant_input(self, input):
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127).to(torch.int8), s

    def forward(self, input):
        input, s = self.quant_input(input)
        return bitnet_int8xint2_linear(input, self.weight, s, self.weight_scale)

# Attention 구조
class Attention(nn.Module):
    def __init__(self, ...):
        Linear = BitLinearKernel if use_kernel else BitLinear
        self.wqkv = Linear(dim, (n_q_heads + 2*n_kv_heads) * head_dim)  # Fused!
        self.wo = Linear(n_q_heads * head_dim, dim)
        self.attn_sub_norm = RMSNorm(dim, norm_eps)

    def forward(self, x, cache, attn_bias):
        xqkv = self.wqkv(x)
        # Split → RoPE → KV cache → attention → sub_norm → wo
        output = self.attn_sub_norm(output)
        output = self.wo(output)
        return output

# FFN 구조
class FeedForward(nn.Module):
    def __init__(self, ...):
        self.w13 = Linear(dim, 2 * hidden_dim)  # gate+up fused
        self.w2 = Linear(hidden_dim, dim)
        self.ffn_sub_norm = RMSNorm(hidden_dim, norm_eps)

    def forward(self, x):
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, -1)
        inner = self.ffn_sub_norm(squared_relu(x1) * x3)  # squared_relu!
        return self.w2(inner)
```

### 13.5 BitNet-TT 구현 개선 방향

#### 문제점 분석

| 현재 구현 | 공식 구현 | 개선 방향 |
| --- | --- | --- |
| 분리된 Q/K/V projection | Fused wqkv | `wqkv` fused matmul 사용 |
| 분리된 gate/up | Fused w13 | `w13` fused matmul 사용 |
| BF16 weights | INT2 packed | Ternary matmul 커널 |
| concat KV cache | In-place update | `paged_update_cache` |
| No trace | Trace | In-place 후 Trace 적용 |

#### 구현 우선순위

1. **Fused Projections** (즉시 가능):

   ```python
   # 현재: 3개 matmul
   q = ttnn.linear(x, w_q)
   k = ttnn.linear(x, w_k)
   v = ttnn.linear(x, w_v)

   # 개선: 1개 fused matmul + split
   qkv = ttnn.linear(x, w_qkv)  # [batch, seq, (q+k+v)]
   Q, K, V = ttnn.experimental.nlp_create_qkv_heads(qkv, ...)
   ```

2. **In-place KV Cache** (Trace 전제조건):

   ```python
   # Pre-allocate
   k_cache = ttnn.allocate_tensor_on_device([batch, kv_heads, max_seq, head_dim], device)

   # Update (not concat!)
   ttnn.experimental.paged_update_cache(k_cache, k_new, update_idxs_tensor=pos_tensor)
   ```

3. **Ternary Matmul** (메모리 8x 절감):
   - TT-Metal 커스텀 커널 필요
   - INT8 activation × INT2 weight → BF16 output

### 13.6 성능 최적화 체크리스트

- [ ] Fused QKV projection (`wqkv`)
- [ ] Fused gate+up projection (`w13`)
- [ ] In-place KV cache update (`paged_update_cache`)
- [ ] Trace capture for decode loop
- [ ] Multi-CQ (CQ0=ops, CQ1=IO)
- [ ] DRAM-sharded matmul for decode
- [ ] HEIGHT_SHARDED for attention (num_heads=20 해결 필요)
- [ ] Ternary matmul kernel (Phase 2)

---

이 문서는 중복/실패 로그를 제거한 핵심 요약본이다. 새로운 실험 추가 시 성공 설정은 상세 기록, 실패는 원인 한 줄만 남길 것.
