# TT-NN API 정리

## 핵심 발견사항

**`ttnn.from_numpy()`는 존재하지 않음!**

텐서 생성 방법:
1. `ttnn.from_torch()` - PyTorch 텐서에서 변환
2. `ttnn.Tensor(numpy_array, device=device, layout=ttnn.TILE_LAYOUT)` - NumPy 배열에서 직접 생성
3. `ttnn.full()`, `ttnn.zeros()`, `ttnn.ones()`, `ttnn.rand()` - 직접 생성

---

## 1. 디바이스 관리

```python
import ttnn

# 디바이스 열기
device = ttnn.open_device(device_id=0)

# L1 메모리 설정 (convolution 등)
device = ttnn.open_device(device_id=0, l1_small_size=8192)

# 디바이스 닫기
ttnn.close_device(device)
```

---

## 2. 텐서 생성

### 2.1 PyTorch에서 변환 (권장)

```python
import torch
import ttnn

torch_tensor = torch.randn(32, 32)

tt_tensor = ttnn.from_torch(
    torch_tensor,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device
)
```

### 2.2 NumPy에서 직접 생성

```python
import numpy as np
import ttnn

numpy_array = np.random.randn(32, 32).astype(np.float32)

tt_tensor = ttnn.Tensor(
    numpy_array,
    device=device,
    layout=ttnn.TILE_LAYOUT
)
```

### 2.3 직접 생성 함수

```python
# 특정 값으로 채우기
tt_tensor = ttnn.full(
    shape=(32, 32),
    fill_value=1.0,
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    device=device
)

# 0으로 초기화
tt_zeros = ttnn.zeros(
    shape=(32, 32),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device
)

# 1로 초기화
tt_ones = ttnn.ones(
    shape=(32, 32),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device
)

# 랜덤 초기화
tt_rand = ttnn.rand(
    (1024, 1024),
    dtype=ttnn.bfloat16,
    device=device,
    layout=ttnn.TILE_LAYOUT
)

# 메모리 설정 포함
tt_rand = ttnn.rand(
    (1024, 1024),
    dtype=ttnn.bfloat16,
    device=device,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.L1_MEMORY_CONFIG
)
```

---

## 3. 텐서 변환

### 3.1 TT-NN → PyTorch

```python
result_torch = ttnn.to_torch(tt_tensor)
```

### 3.2 레이아웃 변환

```python
# TILE → ROW_MAJOR
tt_tensor = ttnn.to_layout(tt_tensor, ttnn.ROW_MAJOR_LAYOUT)

# ROW_MAJOR → TILE
tt_tensor = ttnn.to_layout(tt_tensor, ttnn.TILE_LAYOUT)
```

---

## 4. 데이터 타입 (dtype)

| dtype | 설명 | 정렬 요구사항 (ROW_MAJOR) |
|-------|------|--------------------------|
| `ttnn.uint16` | 16비트 정수 | width % 2 == 0 |
| `ttnn.uint32` | 32비트 정수 | width % 1 == 0 |
| `ttnn.float32` | 32비트 부동소수점 | width % 1 == 0 |
| `ttnn.bfloat16` | Brain Float 16 | width % 2 == 0 |
| `ttnn.bfloat8_b` | 8비트 Brain Float | TILE_LAYOUT 필수, width % 32 == 0 |

---

## 5. 레이아웃 (Layout)

### 5.1 ROW_MAJOR_LAYOUT

- 행 우선 저장
- 일반적인 CPU/GPU 레이아웃
- 데이터 전처리/후처리에 사용

### 5.2 TILE_LAYOUT (권장)

- 32x32 타일 단위 저장
- 내부적으로 16x16 face로 분할
- **매트릭스 연산에 최적화**
- 32의 배수가 아닌 차원은 자동 패딩

```python
# 타일 레이아웃 권장 크기
shape = (32, 32)      # 완벽한 타일
shape = (1, 32, 64)   # 자동 패딩
shape = (batch, seq_len, hidden_size)  # hidden_size % 32 == 0 권장
```

---

## 6. 메모리 설정 (Memory Config)

```python
# DRAM (기본값, 큰 텐서용)
memory_config = ttnn.DRAM_MEMORY_CONFIG

# L1 (빠른 접근, 작은 텐서용)
memory_config = ttnn.L1_MEMORY_CONFIG

# 샤딩 (분산 저장)
memory_config = ttnn.create_sharded_memory_config(...)
```

---

## 7. 기본 연산

### 7.1 산술 연산

```python
# 덧셈
result = ttnn.add(a, b)
result = a + b

# 곱셈 (element-wise)
result = ttnn.mul(a, b)
result = a * b

# 행렬 곱셈
result = ttnn.matmul(a, b)
result = a @ b

# 메모리/코어 설정 포함 matmul
result = ttnn.matmul(
    a, b,
    memory_config=ttnn.L1_MEMORY_CONFIG,
    core_grid=ttnn.CoreGrid(y=8, x=8)
)
```

### 7.2 활성화 함수

```python
result = ttnn.relu(x)
result = ttnn.softmax(x, dim=-1)
result = ttnn.exp(x)
```

### 7.3 정규화

```python
result = ttnn.rms_norm(x, epsilon=1e-6, weight=gamma)
```

### 7.4 형상 변환

```python
# Reshape
result = ttnn.reshape(x, (batch, seq, hidden))

# Transpose
result = ttnn.transpose(x, -2, -1)

# Permute
result = ttnn.permute(x, (0, 2, 1, 3))
```

---

## 8. Linear Layer 구현 패턴

```python
# PyTorch weight를 TT-NN으로 변환
weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=device)

# Weight transpose (필수!)
weight_t = ttnn.transpose(weight, -2, -1)

# Bias reshape
bias = ttnn.reshape(torch_bias, [1, -1])

# Linear 연산
output = ttnn.linear(input, weight_t, bias=bias)
```

---

## 9. Multi-Head Attention 패턴

```python
# Q, K, V 프로젝션
query = hidden_states @ query_weight
query = query + query_bias

# Reshape: (batch, seq, hidden) → (batch, seq, num_heads, head_size)
query = ttnn.to_layout(query, layout=ttnn.ROW_MAJOR_LAYOUT)
query = ttnn.reshape(query, (batch_size, sequence_size, num_heads, head_size))
query = ttnn.to_layout(query, layout=ttnn.TILE_LAYOUT)

# Permute: (batch, seq, heads, head_dim) → (batch, heads, seq, head_dim)
query = ttnn.permute(query, (0, 2, 1, 3))
key = ttnn.permute(key, (0, 2, 3, 1))  # Key는 마지막 두 차원 전치

# Attention score
attention_scores = query @ key
attention_scores = attention_scores * (1 / (head_size ** 0.5))
attention_scores = attention_scores + attention_mask

# Softmax
attention_probs = ttnn.softmax(attention_scores, dim=-1)

# Context
context = attention_probs @ value
```

---

## 10. 최적화 팁

### 10.1 메모리 관리

```python
# 사용 후 메모리 해제
ttnn.deallocate(tensor)

# 메모리 재할당
tensor = ttnn.reallocate(tensor)
```

### 10.2 성능 최적화

```python
# Core grid 지정 (병렬화)
result = ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=8, x=8))

# L1 메모리 사용 (빠른 접근)
result = ttnn.matmul(a, b, memory_config=ttnn.L1_MEMORY_CONFIG)

# 낮은 정밀도 사용 (bfloat8_b)
tensor = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, ...)
```

### 10.3 프로그램 캐싱

- 첫 실행: 커널 컴파일 (느림)
- 이후 실행: 캐시된 프로그램 사용 (빠름)
- "Two orders of magnitude faster" 성능 향상

---

## 11. Storage 타입

| Storage | 설명 |
|---------|------|
| `OWNED_HOST_STORAGE` | TT-NN이 관리하는 호스트 버퍼 |
| `BORROWED_HOST_STORAGE` | 외부(torch/numpy) 버퍼 참조 |
| `DEVICE_STORAGE` | TT-NN이 관리하는 디바이스 버퍼 |

---

## 12. 주의사항

1. **TILE_LAYOUT 권장**: 대부분의 연산에서 TILE_LAYOUT이 훨씬 빠름
2. **32의 배수**: TILE_LAYOUT은 32x32 타일 기반, 32의 배수 권장
3. **Weight 전치**: Linear layer에서 weight는 transpose 필요
4. **Reshape 전 레이아웃**: reshape 전에 ROW_MAJOR로 변환 권장
5. **bfloat16 기본**: 대부분의 연산에서 bfloat16 사용
6. **디바이스 닫기**: 프로그램 종료 전 반드시 `ttnn.close_device()`

---

## 13. BitNet 구현 시 수정 필요 사항

### 기존 코드 (잘못됨)
```python
self.weight = ttnn.from_numpy(weight_float, ...)  # 존재하지 않는 함수!
```

### 수정 코드 옵션 1: torch 경유
```python
import torch
weight_torch = torch.from_numpy(weight_float)
self.weight = ttnn.from_torch(weight_torch, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=device)
```

### 수정 코드 옵션 2: ttnn.Tensor 직접 생성
```python
self.weight = ttnn.Tensor(weight_float, device=device, layout=ttnn.TILE_LAYOUT)
```

**참고**: 옵션 2가 torch 의존성 없이 가능하지만, dtype 지정이 제한적일 수 있음.
실제 하드웨어에서 테스트하여 작동 여부 확인 필요.
