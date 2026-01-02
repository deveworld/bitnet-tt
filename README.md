# BitNet-TT

A native TT-NN implementation for running Microsoft's **BitNet b1.58 2B-4T** model on Tenstorrent Blackhole p150a.

## Key Features

- **HuggingFace Compatible**: Direct loading of `microsoft/bitnet-b1.58-2B-4T-bf16` weights
- **TT-NN Native**: Implementation optimized for Tenstorrent hardware
- **KV-Cache Support**: Efficient autoregressive generation
- **HiFi2 Compute Kernel**: Accelerated matmul with BFP8 operations
- **High Accuracy**: Achieves correlation 0.99+ with HuggingFace implementation

## Quick Start

```bash
# Installation
git clone https://github.com/deveworld/bitnet-tt.git
cd bitnet-tt
uv sync

# Test
python main.py --test

# Mini model demo
python main.py

# Full 2B model demo
python main.py --full

# Interactive chat
python main.py --chat
```

## Performance

Measured on Tenstorrent Blackhole p150a:

| Mode | Speed | Notes |
|------|-------|-------|
| Chat (Streaming) | **8.0 - 9.7 t/s** | HiFi2 applied, batch_size=1 |
| Full Demo | ~5.5 t/s | 30 tokens generation |

### Optimization Status

| Optimization | Status | Effect |
|--------------|--------|--------|
| HiFi2 Compute Kernel | ✅ Applied | ~2x matmul acceleration (theoretical) |
| KV-Cache | ✅ Applied | Concat-based |
| Pre-transposed Weights | ✅ Applied | Transpose overhead removed |
| RoPE Pre-upload | ✅ Applied | Prevents cos/sin recomputation |
| HEIGHT_SHARDED Decode | ❌ Not Applied | num_heads=20 compatibility issue |
| Trace Capture | ❌ Not Applied | Incompatible with concat cache |

### Target Performance

| Reference Model | Hardware | Speed |
|-----------------|----------|-------|
| Llama 3.1 8B | p150a | 33.1 t/s/u |
| Llama 3.2 3B | n150 | 46.6 t/s/u |
| **BitNet 2B (Target)** | p150a | **30+ t/s** |

## Validation Results

Comparison with HuggingFace official implementation:

| Metric | Result |
|--------|--------|
| Logits Correlation | 0.988 ~ 0.999 |
| Top-1 Prediction Match | 100% |
| Max Logit Difference | < 2.5 |

```bash
# Run validation scripts
python examples/debug_compare.py      # Layer-by-layer comparison
python examples/debug_full_compare.py  # Full model comparison
```

## Architecture

### Model Structure

```
BitNetModel (2.4B params)
├── Embedding (128256 vocab, 2560 dim)
├── TransformerBlock x 30
│   ├── RMSNorm (input)
│   ├── MultiHeadAttention
│   │   ├── Q/K/V Projection (Linear, ternary weights)
│   │   ├── RoPE (θ=500000)
│   │   ├── Grouped Query Attention (20 Q heads, 5 KV heads)
│   │   ├── Attention Sub-Norm (BitNet-specific)
│   │   └── O Projection (Linear)
│   ├── RMSNorm (post-attention)
│   └── FFN
│       ├── Gate/Up Projection (Linear)
│       ├── Squared ReLU (relu2)
│       ├── FFN Sub-Norm (BitNet-specific)
│       └── Down Projection (Linear)
├── RMSNorm (final)
└── LM Head
```

### BitLinear Weight Quantization

Same quantization as HuggingFace BitLinear:

```python
# Weight quantization formula (applied at load time)
s = 1.0 / weight.abs().mean()
weight_quant = (weight * s).round().clamp(-1, 1) / s
# Result: ternary values {-scale, 0, +scale}
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| Config | `config.py` | Model configuration + HiFi2 kernel settings |
| Embedding | `layers/embedding.py` | Token embedding |
| RMSNorm | `layers/bitlinear.py` | Root Mean Square Normalization |
| Linear | `layers/bitlinear.py` | Ternary weights + HiFi2 matmul |
| Attention | `layers/attention.py` | GQA + RoPE + KV-Cache |
| FFN | `layers/ffn.py` | Squared ReLU |
| Transformer | `model/transformer.py` | Transformer block |
| BitNetModel | `model/bitnet.py` | Full model |
| Generator | `inference/generator.py` | Text generation + streaming |

## Project Structure

```
bitnet-tt/
├── src/bitnet_tt/
│   ├── config.py              # Model configuration + compute kernel config
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
│       ├── device.py          # TT-NN device management
│       ├── quantization.py    # Quantization utilities
│       └── weights.py         # HuggingFace weight loading
├── examples/
│   ├── demo.py                # Demo script
│   ├── debug_compare.py       # Layer-by-layer comparison
│   └── debug_full_compare.py  # Full model comparison
├── main.py                    # CLI entry point
└── MEMO.md                    # Technical memo (optimization records)
```

## API Usage

### Basic Inference

```python
from bitnet_tt.model.bitnet import create_model
from bitnet_tt.inference.generator import TextGenerator
from bitnet_tt.utils.device import device_context
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model

with device_context() as device:
    # Load model
    state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
    model = create_model(config, device)
    load_weights_to_model(model, state_dict)

    # Generate text
    generator = TextGenerator(model)
    output = generator.generate(
        "Hello, I am",
        max_new_tokens=50,
        temperature=0.7,
    )
    print(output)
```

### Streaming Chat

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")
generator = TextGenerator(model, tokenizer)

# Streaming output
for text, stats in generator.chat_streaming(
    "Hello! How are you?",
    max_new_tokens=100,
    temperature=0.7,
):
    print(text, end="", flush=True)

print(f"\n[Speed: {stats.tokens_per_second:.2f} t/s]")
```

## Hardware Requirements

### Tenstorrent Blackhole p150a

| Specification | Value |
|---------------|-------|
| Tensix Cores | 140 |
| SRAM | 210MB (1.5MB per core) |
| Memory | 32GB GDDR6 |
| TDP | Up to 300W |

### Software Requirements

- Ubuntu 20.04/22.04 LTS
- Python 3.10 (ttnn compatible)
- TT-NN SDK
- PyTorch 2.0+
- Transformers 4.40+

## Implementation Details

### Compute Kernel Configuration

```python
# HiFi2: BFP8 operations, ~2x speed improvement (accuracy maintained)
from bitnet_tt.config import get_compute_kernel_config

kernel_config = get_compute_kernel_config("hifi2")
# ttnn.matmul(..., compute_kernel_config=kernel_config)
```

| Fidelity | Precision | Speed | Use Case |
|----------|-----------|-------|----------|
| HiFi4 | BF16 | 1x | Prefill, accuracy-critical |
| HiFi2 | BFP8 | ~2x | Decode (currently used) |
| LoFi | BFP4 | ~3.6x | MLP (experimental) |

### TT-NN Tensor Layout

```python
# TILE_LAYOUT: Optimized for compute (required for matmul)
# ROW_MAJOR_LAYOUT: reshape/embedding input

# Layout conversion before reshape/permute
x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
x = ttnn.reshape(x, new_shape)
x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
```

## Known Limitations

1. **HEIGHT_SHARDED Not Supported**: BitNet's `num_heads=20` is incompatible with tt_transformers' 32-core grid
2. **Trace Capture Not Applied**: Concat-based KV-Cache is incompatible with trace
3. **batch_size=1 Only**: batch>1 is inefficient without HEIGHT_SHARDED

## References

### Tenstorrent

- [Official Documentation](https://docs.tenstorrent.com)
- [TT-Metal GitHub](https://github.com/tenstorrent/tt-metal)
- [tt_transformers](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)

### BitNet

- [BitNet Paper (arXiv:2310.11453)](https://arxiv.org/abs/2310.11453)
- [BitNet b1.58 Paper (arXiv:2402.17764)](https://arxiv.org/abs/2402.17764)
- [BitNet b1.58 2B4T (HuggingFace)](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)

## License

MIT License
