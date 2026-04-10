# TODO

## 방향

- `ttnn.experimental.ternary_matmul` 계열의 별도 native op를 만든다.
- `minimal_matmul`은 packed ternary weight GEMM의 기반으로 쓰지 않는다.
  - 현재 `minimal_matmul`의 "fused ternary"는 ternary weight matmul이 아니라 addcmul fusion이다.
- 초기 bring-up은 `generic_op` 또는 `patchable_generic_op`로 구조를 검증하고, 이후 정식 `ttnn.experimental` op로 승격한다.
- `bitnet-tt` 최적화는 generic batch-1 decode를 더 복잡하게 만드는 대신, 공식 decode primitive와 sharding discipline에 맞춘 경로를 중심으로 진행한다.

## 목표

- packed ternary weight를 DRAM에서 더 효율적으로 읽는 경로를 확보한다.
- `bitnet-tt`의 FFN, prefill QKV, decode QKV에서 기존 BFP4/BFP8 경로와 비교 가능한 대체 경로를 만든다.
- trace, program cache, sharded memory config, paged cache 경로와 충돌하지 않는 구현을 만든다.
- correctness, trace compatibility, 성능 병목 위치를 단계적으로 분리해서 검증한다.

## 기본 설계 원칙

- op 이름은 `ttnn.experimental.ternary_matmul`로 분리한다.
- packed weight는 새 dtype을 즉시 도입하지 않고, 우선 `uint32` 또는 row-major packed buffer tensor로 표현한다.
- activation은 기존 TT-NN matmul이 기대하는 텐서 형식과 최대한 맞춘다.
- reader에서 packed ternary를 해석해 `cb_in1`로 공급하고, compute는 가능한 한 단순하게 시작한다.
- validation은 초기에 강하게 건다.
  - device resident 여부
  - supported layout
  - tile alignment
  - packed weight buffer shape 및 stride 규약
  - scale tensor shape 규약
  - sharded / interleaved 조합 제약

## API 초안

- Python API 초안:
  - `ttnn.experimental.ternary_matmul(input_tensor, packed_weight_tensor, scale_tensor=None, *, config=None, memory_config=None, dtype=None, compute_kernel_config=None, zero_tile_mask=None)`
- 설계 의도:
  - `input_tensor`: 일반 activation tensor
  - `packed_weight_tensor`: packed 2-bit ternary weight buffer
  - `scale_tensor`: global 또는 tile-wise scale
  - `config`: block size, core grid, packed layout metadata
  - `zero_tile_mask`: zero tile skip용 optional tensor

## 디렉터리 / 파일 구조

- `tt-metal` 내부에 다음 디렉터리를 추가한다.
  - `ttnn/cpp/ttnn/operations/experimental/ternary_matmul/`
- 추가 대상 파일:
  - `ternary_matmul.hpp`
  - `ternary_matmul.cpp`
  - `ternary_matmul_nanobind.hpp`
  - `ternary_matmul_nanobind.cpp`
  - `CMakeLists.txt`
  - `device/ternary_matmul_device_operation.hpp`
  - `device/ternary_matmul_device_operation.cpp`
  - `device/ternary_matmul_device_operation_types.hpp`
  - `device/ternary_matmul_program_factory.cpp`
  - `device/kernels/dataflow/...`
  - `device/kernels/compute/...`
  - 필요 시 `device/kernels/writer/...`
- 상위 등록 지점:
  - `ttnn/CMakeLists.txt`
  - `ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp`

## 프로토타입 경로

### generic_op 기반 검증

- `ttnn.generic_op` 또는 `ttnn.experimental.patchable_generic_op`로 먼저 커널 구조를 검증한다.
- 목적:
  - packed reader의 shape / runtime arg / CB wiring 검증
  - packed weight buffer 주소 patching 검증
  - tile order와 face order 검증
  - output correctness 검증
- 구성:
  - input tensor
  - packed weight tensor
  - preallocated output tensor
  - program descriptor
  - reader / compute / writer kernel descriptor

### 초기 커널 구성

- Reader:
  - packed ternary tile을 DRAM에서 읽는다.
  - TT tile face order를 유지한 채 unpack한다.
  - unpack 결과를 `cb_in1`에 쓴다.
- Compute:
  - 우선 dense matmul 스타일로 시작한다.
  - packed ternary 특화 계산은 아직 넣지 않는다.
- Writer:
  - 기본 output write-back만 수행한다.

## 정식 native op 경로

### C++ wrapper / nanobind

- `ttnn.experimental.ternary_matmul(...)` wrapper를 추가한다.
- nanobind docstring에는 다음 제약을 명시한다.
  - packed weight tensor 포맷
  - supported input dtype / layout
  - scale tensor 규약
  - zero tile mask 규약
  - sharded decode use case 여부

### device operation

- `operation_attributes_t`에 포함할 후보:
  - block sizes
  - packed format metadata
  - use_scale
  - use_zero_tile_mask
  - compute kernel config
  - output memory config
- `tensor_args_t`에 포함할 후보:
  - activation tensor
  - packed weight tensor
  - optional scale tensor
  - optional zero tile mask tensor
- validation에 포함할 항목:
  - buffer allocated 여부
  - packed buffer alignment
  - tile-aligned logical / padded shape
  - supported dtype
  - supported memory layout
  - shard spec consistency

### program factory

- `minimal_matmul_program_factory.cpp`를 참고하되, packed weight path를 별도 취급한다.
- DRAM reader placement는 bank-aware 전략을 따른다.
- interleaved weight보다 DRAM-sharded packed weight가 우선 경로가 되도록 설계한다.
- runtime arg override가 빈번하면 `patchable_generic_op` 스타일의 주소 patching 아이디어도 검토한다.

## 구현 단계

### Track A: packed reader + dense compute

- packed 2-bit ternary를 reader에서 unpack하여 `cb_in1`로 공급한다.
- compute는 dense BF16/BFP style matmul과 최대한 유사하게 둔다.
- scale 적용은 우선 post-matmul scalar 또는 tile-wise multiply로 단순화한다.
- 목적:
  - correctness 확보
  - DRAM traffic 절감 효과 측정
  - trace 및 program cache 호환성 확인

### Track B: zero-tile skip

- `zero_tile_mask`를 추가한다.
- 32x32 all-zero tile은 reader 또는 compute에서 완전히 skip한다.
- `bitnet-tt`의 `ternary_analysis.py`를 활용해 실제 zero tile 분포를 먼저 측정한다.
- 목표:
  - packed reader만으로 부족한 경우 compute 낭비를 줄인다.

### Track C: direct packed compute

- compute kernel이 unpacked BF16 tile이 아니라 packed ternary를 직접 소비하도록 확장한다.
- 이 단계는 앞선 track에서 병목이 compute에 남아 있을 때만 진행한다.
- 핵심 검토 항목:
  - packed format에서 matrix engine feed 방식
  - accumulator 정밀도
  - scale 적용 방식
  - zero value 처리 비용

## bitnet-tt 통합 우선순위

- 1순위: FFN
  - `gate/up/down` projection에 ternary-native op 적용
  - matmul 횟수와 weight traffic 측면에서 ROI가 가장 높다.
- 2순위: prefill QKV
  - 긴 sequence prefill에서 기존 `minimal_matmul` / `linear`와 비교
- 3순위: decode QKV
  - 공식 decode primitive와 trace 경로를 해치지 않는 범위에서만 적용
- 주의:
  - attention decode는 이미 공식 stack이 DRAM-sharded + paged cache + traced decode discipline에 강하게 의존한다.
  - 여기서는 native ternary op보다 layout / program_config 정렬이 우선이다.

## bitnet-tt 쪽 병행 정리 항목

- `Batch32Generator`를 기본 decode 엔진 후보로 유지한다.
- generic batch-1 decode 경로는 기능 fallback으로만 둔다.
- official decode primitive 체인과 맞지 않는 `to_layout`, `slice`, `to_memory_config` churn을 줄인다.
- prefill은 긴 sequence에서 `minimal_matmul` 도입 여부를 별도로 검토한다.

## 테스트 계획

### unit test

- `tt-metal`에 `test_ternary_matmul.py`를 추가한다.
- 검증 항목:
  - torch reference와 PCC / RMSE 비교
  - padded / unpadded shape
  - small / large M, K, N
  - BF16 / BFP8 / BFP4 activation path
  - scale on / off
  - zero tile mask on / off
  - interleaved / sharded memory config

### traced test

- trace capture / replay 가능한 static shape 조합을 별도로 검증한다.
- persistent DRAM input, L1 reshard, output buffer 재사용을 확인한다.
- decode 스타일 static batch / static block count 조합을 우선 검증한다.

### integration test

- `bitnet-tt` FFN 단위 테스트
- `bitnet-tt` attention prefill 경로 비교 테스트
- `bitnet-tt` batch32 decode smoke test
- end-to-end generation 비교

### accuracy 기준

- submodule test는 높은 PCC 기준을 유지한다.
- layer / model 수준은 reference 대비 상대 허용폭을 따로 둔다.
- zero tile skip 또는 packed direct compute를 켠 경우, baseline 대비 accuracy regression을 별도로 기록한다.

## 성능 분석 계획

- Tracy / Device Profiler / perf report를 함께 사용한다.
- 비교 대상:
  - 기존 BF16
  - 기존 BFP8
  - 기존 BFP4
  - ternary packed reader path
  - ternary packed + zero tile skip path
- 확인할 병목:
  - host dispatch
  - device dispatch gap
  - DRAM bandwidth saturation
  - compute utilization
  - CB stall
  - runtime arg patch 비용

## 디버깅 체크리스트

- `TT_METAL_WATCHER=1`
- `TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1`
- 필요 시 DPRINT로 CB tile dump
- hang 발생 시 단일 op, 단일 device, 작은 core grid로 축소
- program cache를 끄고 재현 여부 확인
- fixed input 값으로 단순화
- shard spec mismatch 여부 점검
- packed tile face order와 unpack order를 torch 기준으로 재검증

## 아키텍처 / 하드웨어 주의점

- Wormhole / Blackhole 차이를 염두에 둔다.
- Blackhole에서는 64B DRAM read alignment를 확인한다.
- NoC write 이후 barrier 필요 여부를 커널별로 확인한다.
- L1 cache가 켜진 환경에서는 invalidate 누락 여부를 점검한다.
- padded shard는 피하고, 가능한 경우 core grid를 tensor shape에 맞춘다.

## 성공 조건

- ternary-native 경로가 최소한 일부 핵심 matmul에서 기존 경로와 비교 가능한 결과를 만든다.
- FFN 경로에서 weight traffic 절감 효과가 실제 wall-clock 개선으로 이어진다.
- trace, paged cache, batch32 decode와 충돌하지 않는다.
- 정확도 regression이 허용 범위 안에 있다.
- packed reader만으로 성능 이득이 불충분하면 zero-tile skip 또는 direct packed compute로 자연스럽게 확장 가능하다.
