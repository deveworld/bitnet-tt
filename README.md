# BitNet-TT

Tenstorrent Blackhole p150a에서 BitNet LLM을 구현하기 위한 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 Microsoft Research의 **BitNet b1.58** 아키텍처를 Tenstorrent의 **Blackhole p150a** AI 가속기에서 실행하는 것을 목표로 합니다. BitNet의 1.58-bit 삼진 가중치({-1, 0, 1})와 Tenstorrent의 Tensix 코어 아키텍처를 결합하여 초저전력 고효율 LLM 추론을 구현합니다.

## 하드웨어: Tenstorrent Blackhole p150a

### 사양

| 항목 | 사양 |
|------|------|
| Tensix 코어 | 140개 |
| RISC-V 코어 | 16개 |
| SRAM | 210MB (코어당 1.5MB) |
| 메모리 | 32GB GDDR6 |
| 메모리 대역폭 | 높은 대역폭 |
| TDP | 최대 300W |
| 네트워킹 | 4x QSFP-DD 800Gbps (멀티카드 연결 지원) |
| 폼팩터 | 듀얼슬롯, 액티브 쿨러 |
| 전원 | 12+4핀 12V-2x6 커넥터 |

### 주요 특징

- **Tensix 코어**: 32-wide FPU 벡터 유닛과 32x32 행렬 가속기 탑재
- **소프트웨어 관리 메모리**: 캐시 기반이 아닌 스크래치패드 SRAM + DMA 방식
- **고속 토러스 인터커넥트**: 코어 간 빠른 데이터 전송
- **멀티칩 스케일링**: QSFP-DD 포트를 통한 다중 카드 연결로 메모리 풀링 가능

## 소프트웨어 스택

Tenstorrent는 세 가지 레벨의 소프트웨어 스택을 제공합니다:

### 1. TT-Forge (High-Level)

MLIR 기반 컴파일러로, PyTorch, JAX, TensorFlow 모델을 Tenstorrent 하드웨어용으로 컴파일합니다.

```python
# PyTorch 모델을 TT-Forge로 컴파일하는 예시
import tt_forge
model = tt_forge.compile(pytorch_model)
```

### 2. TT-NN (Mid-Level)

Python & C++ 뉴럴 네트워크 연산 라이브러리입니다. 친숙한 고수준 API로 모델을 실행할 수 있습니다.

```python
import ttnn

# 디바이스 열기
device = ttnn.open_device(device_id=0)

# 텐서 생성 및 연산
a = ttnn.from_torch(torch_tensor, device=device)
b = ttnn.matmul(a, weights)

# 디바이스 닫기
ttnn.close_device(device)
```

**주요 연산**:
- `ttnn.matmul`: 행렬 곱셈
- `ttnn.add`, `ttnn.sub`, `ttnn.mul`: 요소별 연산
- `ttnn.softmax`, `ttnn.relu`, `ttnn.gelu`: 활성화 함수
- `ttnn.layer_norm`, `ttnn.rms_norm`: 정규화
- `ttnn.embedding`: 임베딩 룩업

### 3. TT-Metalium (Low-Level)

저수준 SDK로 커스텀 커널을 직접 개발할 수 있습니다.

**커널 타입**:
- **Data Movement 커널**: 메모리 간 데이터 이동
- **Compute 커널**: SFPU(Scalar Floating Point Unit)를 사용한 연산
- **Ethernet 커널**: 멀티칩 통신

## BitNet b1.58 아키텍처

### 개요

BitNet b1.58은 Microsoft Research에서 개발한 1-bit LLM 아키텍처입니다. 모든 가중치가 삼진 값 **{-1, 0, 1}**로 표현됩니다.

### 핵심 컴포넌트: BitLinear

```
Input -> LayerNorm -> Absmax Quantization -> 1-bit Weights -> Dequantization -> Output
                           |
                     beta (scale factor)
```

**BitLinear 연산**:
1. 활성화를 8-bit 정수로 양자화 (absmax 양자화)
2. 가중치를 삼진 값 {-1, 0, 1}로 양자화 (absmean 양자화)
3. 정수 행렬 곱셈 수행
4. 결과를 역양자화

### 장점

| 항목 | BitNet b1.58 | FP16 모델 |
|------|--------------|----------|
| 가중치 크기 | ~1.58 bits/param | 16 bits/param |
| 메모리 사용량 | ~10x 감소 | 기준 |
| 에너지 소비 | 크게 감소 | 기준 |
| 추론 속도 | 향상 | 기준 |
| 정확도 | 동등 (3B+ 모델) | 기준 |

### 참조 모델

- **BitNet b1.58 2B4T**: 2B 파라미터, 4T 토큰으로 학습된 최초의 오픈소스 네이티브 1-bit LLM
- HuggingFace: `microsoft/bitnet-b1.58-2B-4T`

## 설치 및 환경 설정

### 사전 요구사항

- Tenstorrent Blackhole p150a 카드
- Ubuntu 20.04/22.04 LTS
- Python 3.12+

### 프로젝트 설치

```bash
git clone https://github.com/your-username/bitnet-tt.git
cd bitnet-tt
uv sync
```

## 사용법

```python
from bitnet_tt import BitNetModel

# 모델 로드
model = BitNetModel.from_pretrained("microsoft/bitnet-b1.58-2B-4T")

# 추론
output = model.generate("Hello, world!")
```

## 참고 자료

### Tenstorrent
- [공식 문서](https://docs.tenstorrent.com)
- [TT-Metal GitHub](https://github.com/tenstorrent/tt-metal)
- [Blackhole 사양](https://docs.tenstorrent.com/aibs/blackhole/)

### BitNet
- [BitNet 논문 (arXiv:2310.11453)](https://arxiv.org/abs/2310.11453)
- [BitNet b1.58 논문 (arXiv:2402.17764)](https://arxiv.org/abs/2402.17764)
- [BitNet b1.58 2B4T (HuggingFace)](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- [bitnet.cpp (추론 라이브러리)](https://github.com/microsoft/BitNet)

## 라이선스

MIT License
