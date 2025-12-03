# BitNet-TT

Tenstorrent Blackhole p150a에서 Microsoft의 **BitNet b1.58 2B-4T** 모델을 실행하는 TT-NN 네이티브 구현입니다.

## 주요 기능

- **HuggingFace 호환**: `microsoft/bitnet-b1.58-2B-4T-bf16` 가중치 직접 로드
- **TT-NN 네이티브**: Tenstorrent 하드웨어에 최적화된 구현
- **KV-Cache 지원**: 효율적인 자기회귀 생성
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

## 성능

Tenstorrent Blackhole p150a에서 측정:

| 모드 | 시간 | 속도 |
|------|------|------|
| Without KV-Cache | 6.53s / 30 tokens | ~217 ms/token |
| With KV-Cache | 3.84s / 30 tokens | ~128 ms/token |

**KV-Cache 속도 향상: 1.7x**

## 아키텍처

### 모델 구조

```
BitNetModel
├── Embedding (128256 vocab, 2560 dim)
├── TransformerBlock x 30
│   ├── RMSNorm (input)
│   ├── MultiHeadAttention
│   │   ├── Q/K/V Projection (BitLinear)
│   │   ├── RoPE (θ=500000)
│   │   ├── Grouped Query Attention (20 Q heads, 5 KV heads)
│   │   ├── Attention Sub-Norm
│   │   └── O Projection (BitLinear)
│   ├── RMSNorm (post-attention)
│   └── FFN
│       ├── Gate/Up Projection (BitLinear)
│       ├── SiLU + Squared ReLU
│       ├── FFN Sub-Norm
│       └── Down Projection (BitLinear)
├── RMSNorm (final)
└── LM Head (tie_word_embeddings=True)
```

### BitLinear 가중치 양자화

HuggingFace BitLinear와 동일한 양자화 적용:

```python
# 가중치 양자화 공식
s = 1.0 / weight.abs().mean()
weight_quant = (weight * s).round().clamp(-1, 1) / s
# 결과: {-scale, 0, +scale} 삼진 값
```

### 주요 컴포넌트

| 컴포넌트 | 파일 | 설명 |
|----------|------|------|
| Embedding | `layers/embedding.py` | 토큰 임베딩 |
| RMSNorm | `layers/bitlinear.py` | Root Mean Square Normalization |
| Linear | `layers/bitlinear.py` | 삼진 가중치 양자화 Linear |
| Attention | `layers/attention.py` | GQA + RoPE + KV-Cache |
| FFN | `layers/ffn.py` | SwiGLU + Squared ReLU |
| Transformer | `model/transformer.py` | 트랜스포머 블록 |
| BitNetModel | `model/bitnet.py` | 전체 모델 |

## 프로젝트 구조

```
bitnet-tt/
├── src/bitnet_tt/
│   ├── config.py              # 모델 설정
│   ├── layers/
│   │   ├── attention.py       # Multi-Head Attention + KV-Cache
│   │   ├── bitlinear.py       # BitLinear, RMSNorm, Linear
│   │   ├── embedding.py       # Embedding layer
│   │   └── ffn.py             # Feed-Forward Network
│   ├── model/
│   │   ├── bitnet.py          # BitNetModel
│   │   └── transformer.py     # TransformerBlock
│   ├── inference/
│   │   └── generator.py       # Text generation
│   └── utils/
│       ├── device.py          # TT-NN device 관리
│       ├── quantization.py    # 양자화 유틸리티
│       └── weights.py         # HuggingFace 가중치 로딩
├── examples/
│   ├── demo.py                # 데모 스크립트
│   ├── debug_compare.py       # 레이어별 비교
│   └── debug_full_compare.py  # 전체 모델 비교
└── main.py                    # CLI 엔트리포인트
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
        use_cache=True,
    )
    print(output)
```

### KV-Cache 사용

```python
from bitnet_tt.layers.attention import KVCache

# 첫 번째 forward (프롬프트 처리)
logits, past_key_values = model(input_ids, use_cache=True)

# 후속 forward (토큰 생성)
for _ in range(max_tokens):
    next_token = sample(logits)
    logits, past_key_values = model(
        next_token,
        past_key_values=past_key_values,
        use_cache=True,
    )
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
- Python 3.10+
- TT-NN SDK
- PyTorch 2.0+
- Transformers 4.40+

## 구현 세부사항

### TT-NN 텐서 레이아웃

```python
# TILE_LAYOUT: 연산에 최적화 (기본값)
# ROW_MAJOR_LAYOUT: reshape/permute 전에 변환 필요

# reshape/permute 전 레이아웃 변환
x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
x = ttnn.reshape(x, new_shape)
x = ttnn.permute(x, dims)
x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
```

### 가중치 매핑

HuggingFace → TT-NN 가중치 매핑:

```python
layer_mapping = {
    "input_layernorm.weight": "input_layernorm.weight",
    "self_attn.q_proj.weight": "self_attn.q_proj.weight",
    "self_attn.k_proj.weight": "self_attn.k_proj.weight",
    "self_attn.v_proj.weight": "self_attn.v_proj.weight",
    "self_attn.o_proj.weight": "self_attn.o_proj.weight",
    "self_attn.attn_sub_norm.weight": "self_attn.attn_sub_norm.weight",  # BitNet 전용
    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "mlp.up_proj.weight": "mlp.up_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
    "mlp.ffn_sub_norm.weight": "mlp.ffn_sub_norm.weight",  # BitNet 전용
    "post_attention_layernorm.weight": "post_attention_layernorm.weight",
}
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
- [HuggingFace Transformers BitNet](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bitnet)

## 라이선스

MIT License
