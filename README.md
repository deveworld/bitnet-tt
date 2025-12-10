# BitNet-TT

Tenstorrent Blackhole p150a에서 Microsoft의 **BitNet b1.58 2B-4T** 모델을 실행하는 TT-NN 네이티브 구현입니다.

## 주요 기능

- **HuggingFace 호환**: `microsoft/bitnet-b1.58-2B-4T-bf16` 가중치 직접 로드
- **TT-NN 네이티브**: Tenstorrent 하드웨어에 최적화된 구현
- **KV-Cache 지원**: 효율적인 자기회귀 생성
- **HiFi2 Compute Kernel**: BFP8 연산으로 matmul 가속
- **높은 정확도**: HuggingFace 구현과 correlation 0.99+ 달성

## 빠른 시작

```bash
# 설치
git clone https://github.com/deveworld/bitnet-tt.git
cd bitnet-tt
uv sync

# 테스트
python main.py --test

# Mini 모델 데모
python main.py

# Full 2B 모델 데모
python main.py --full

# 대화형 채팅
python main.py --chat
```

## 성능

Tenstorrent Blackhole p150a에서 측정:

| 모드 | 속도 | 비고 |
|------|------|------|
| Chat (Streaming) | **8.0 - 9.7 t/s** | HiFi2 적용, batch_size=1 |
| Full Demo | ~5.5 t/s | 30 tokens 생성 |

### 최적화 현황

| 최적화 | 상태 | 효과 |
|--------|------|------|
| HiFi2 Compute Kernel | ✅ 적용 | matmul ~2x 가속 (이론) |
| KV-Cache | ✅ 적용 | concat 기반 |
| Pre-transposed Weights | ✅ 적용 | transpose 오버헤드 제거 |
| RoPE Pre-upload | ✅ 적용 | cos/sin 재계산 방지 |
| HEIGHT_SHARDED Decode | ❌ 미적용 | num_heads=20 호환 문제 |
| Trace Capture | ❌ 미적용 | concat 캐시와 비호환 |

### 목표 성능

| 참조 모델 | Hardware | 속도 |
|-----------|----------|------|
| Llama 3.1 8B | p150a | 33.1 t/s/u |
| Llama 3.2 3B | n150 | 46.6 t/s/u |
| **BitNet 2B (목표)** | p150a | **30+ t/s** |

## 검증 결과

HuggingFace 공식 구현과의 비교:

| 지표 | 결과 |
|------|------|
| Logits Correlation | 0.988 ~ 0.999 |
| Top-1 Prediction Match | 100% |
| Max Logit Difference | < 2.5 |

```bash
# 검증 스크립트 실행
python examples/debug_compare.py      # 레이어별 비교
python examples/debug_full_compare.py  # 전체 모델 비교
```

## 아키텍처

### 모델 구조

```
BitNetModel (2.4B params)
├── Embedding (128256 vocab, 2560 dim)
├── TransformerBlock x 30
│   ├── RMSNorm (input)
│   ├── MultiHeadAttention
│   │   ├── Q/K/V Projection (Linear, ternary weights)
│   │   ├── RoPE (θ=500000)
│   │   ├── Grouped Query Attention (20 Q heads, 5 KV heads)
│   │   ├── Attention Sub-Norm (BitNet 고유)
│   │   └── O Projection (Linear)
│   ├── RMSNorm (post-attention)
│   └── FFN
│       ├── Gate/Up Projection (Linear)
│       ├── Squared ReLU (relu2)
│       ├── FFN Sub-Norm (BitNet 고유)
│       └── Down Projection (Linear)
├── RMSNorm (final)
└── LM Head
```

### BitLinear 가중치 양자화

HuggingFace BitLinear와 동일한 양자화 적용:

```python
# 가중치 양자화 공식 (로드 시 적용)
s = 1.0 / weight.abs().mean()
weight_quant = (weight * s).round().clamp(-1, 1) / s
# 결과: {-scale, 0, +scale} 삼진 값
```

### 주요 컴포넌트

| 컴포넌트 | 파일 | 설명 |
|----------|------|------|
| Config | `config.py` | 모델 설정 + HiFi2 커널 설정 |
| Embedding | `layers/embedding.py` | 토큰 임베딩 |
| RMSNorm | `layers/bitlinear.py` | Root Mean Square Normalization |
| Linear | `layers/bitlinear.py` | 삼진 가중치 + HiFi2 matmul |
| Attention | `layers/attention.py` | GQA + RoPE + KV-Cache |
| FFN | `layers/ffn.py` | Squared ReLU |
| Transformer | `model/transformer.py` | 트랜스포머 블록 |
| BitNetModel | `model/bitnet.py` | 전체 모델 |
| Generator | `inference/generator.py` | 텍스트 생성 + 스트리밍 |

## 프로젝트 구조

```
bitnet-tt/
├── src/bitnet_tt/
│   ├── config.py              # 모델 설정 + compute kernel config
│   ├── layers/
│   │   ├── attention.py       # Multi-Head Attention + KV-Cache
│   │   ├── bitlinear.py       # Linear (HiFi2), RMSNorm
│   │   ├── embedding.py       # Embedding layer
│   │   ├── ffn.py             # Feed-Forward Network
│   │   └── rope_optimized.py  # RoPE pre-upload
│   ├── model/
│   │   ├── bitnet.py          # BitNetModel
│   │   └── transformer.py     # TransformerBlock
│   ├── inference/
│   │   └── generator.py       # TextGenerator (streaming)
│   └── utils/
│       ├── device.py          # TT-NN device 관리
│       ├── quantization.py    # 양자화 유틸리티
│       └── weights.py         # HuggingFace 가중치 로딩
├── examples/
│   ├── demo.py                # 데모 스크립트
│   ├── debug_compare.py       # 레이어별 비교
│   └── debug_full_compare.py  # 전체 모델 비교
├── main.py                    # CLI 엔트리포인트
└── MEMO.md                    # 기술 메모 (최적화 기록)
```

## API 사용법

### 기본 추론

```python
from bitnet_tt.model.bitnet import create_model
from bitnet_tt.inference.generator import TextGenerator
from bitnet_tt.utils.device import device_context
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model

with device_context() as device:
    # 모델 로드
    state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
    model = create_model(config, device)
    load_weights_to_model(model, state_dict)

    # 텍스트 생성
    generator = TextGenerator(model)
    output = generator.generate(
        "Hello, I am",
        max_new_tokens=50,
        temperature=0.7,
    )
    print(output)
```

### 스트리밍 채팅

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")
generator = TextGenerator(model, tokenizer)

# 스트리밍 출력
for text, stats in generator.chat_streaming(
    "Hello! How are you?",
    max_new_tokens=100,
    temperature=0.7,
):
    print(text, end="", flush=True)

print(f"\n[Speed: {stats.tokens_per_second:.2f} t/s]")
```

## 하드웨어 요구사항

### Tenstorrent Blackhole p150a

| 항목 | 사양 |
|------|------|
| Tensix 코어 | 140개 |
| SRAM | 210MB (코어당 1.5MB) |
| 메모리 | 32GB GDDR6 |
| TDP | 최대 300W |

### 소프트웨어 요구사항

- Ubuntu 20.04/22.04 LTS
- Python 3.10 (ttnn 호환)
- TT-NN SDK
- PyTorch 2.0+
- Transformers 4.40+

## 구현 세부사항

### Compute Kernel Configuration

```python
# HiFi2: BFP8 연산, ~2x 속도 향상 (정확도 유지)
from bitnet_tt.config import get_compute_kernel_config

kernel_config = get_compute_kernel_config("hifi2")
# ttnn.matmul(..., compute_kernel_config=kernel_config)
```

| Fidelity | 정밀도 | 속도 | 용도 |
|----------|--------|------|------|
| HiFi4 | BF16 | 1x | Prefill, 정확도 중요 |
| HiFi2 | BFP8 | ~2x | Decode (현재 사용) |
| LoFi | BFP4 | ~3.6x | MLP (실험 필요) |

### TT-NN 텐서 레이아웃

```python
# TILE_LAYOUT: 연산에 최적화 (matmul 필수)
# ROW_MAJOR_LAYOUT: reshape/embedding 입력

# reshape/permute 전 레이아웃 변환
x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
x = ttnn.reshape(x, new_shape)
x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
```

## 알려진 제한사항

1. **HEIGHT_SHARDED 미지원**: BitNet의 `num_heads=20`이 tt_transformers의 32-core grid와 호환되지 않음
2. **Trace Capture 미적용**: concat 기반 KV-Cache가 trace와 비호환
3. **batch_size=1 전용**: batch>1은 HEIGHT_SHARDED 없이 비효율적

## 참고 자료

### Tenstorrent
- [공식 문서](https://docs.tenstorrent.com)
- [TT-Metal GitHub](https://github.com/tenstorrent/tt-metal)
- [tt_transformers](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)

### BitNet
- [BitNet 논문 (arXiv:2310.11453)](https://arxiv.org/abs/2310.11453)
- [BitNet b1.58 논문 (arXiv:2402.17764)](https://arxiv.org/abs/2402.17764)
- [BitNet b1.58 2B4T (HuggingFace)](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)

## 라이선스

MIT License
