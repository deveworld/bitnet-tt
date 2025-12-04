# TT-NN API & Usage Guide

Based on official documentation and tutorials.

## 1. Device Management

Start and stop the device connection. Always close the device to prevent hangs.

```python
import ttnn

# Basic usage
device_id = 0
device = ttnn.open_device(device_id=device_id)
# ... operations ...
ttnn.close_device(device)

# With L1 memory reservation (e.g., for Conv2d)
device = ttnn.open_device(device_id=0, l1_small_size=8192)

# Context manager (recommended)
with ttnn.manage_device(0) as device:
    # ... operations ...
```

## 2. Tensor Management

### 2.1 Creation & Conversion

*   **From PyTorch**: The primary way to move data to device.
*   **From NumPy**: Not directly supported; convert to PyTorch first.
*   **Direct Creation**: `zeros`, `ones`, `full`, `rand`, `arange`.

```python
import torch
import ttnn

# From PyTorch (Recommended)
torch_tensor = torch.randn(32, 32)
tt_tensor = ttnn.from_torch(
    torch_tensor,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG
)

# To PyTorch (for debugging/output)
output_torch = ttnn.to_torch(tt_tensor)

# Direct creation on device (preferred - avoids host-device transfer)
tt_tensor = ttnn.zeros(
    shape=(32, 32),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device
)
```

### 2.2 Layouts

*   **`ttnn.ROW_MAJOR_LAYOUT`**:
    *   Standard row-major order.
    *   Best for: Data loading, non-compute-intensive ops, or when required by specific ops (e.g., embedding inputs).
    *   Alignment: Width usually requires alignment (e.g., % 2 for bfloat16).
*   **`ttnn.TILE_LAYOUT`** (Crucial for Performance):
    *   Data organized in 32x32 tiles.
    *   **Required** for `matmul` and most compute-intensive ops.
    *   Automatic padding if dimensions aren't multiples of 32.
    *   Inner structure: 16x16 faces (matrix engine natively multiplies 16x16 matrices).

### 2.3 Data Types (Dtype)

| Dtype | Description | Width Multiple | Notes |
|-------|-------------|----------------|-------|
| `ttnn.uint32` | Unsigned 32-bit int | 1 | For indices (embedding, etc.) |
| `ttnn.float32` | Standard FP32 | 1 | |
| `ttnn.bfloat16` | Standard BF16 | 2 | Recommended for general inference |
| `ttnn.bfloat8_b` | Block floating point | 32 | **Requires TILE_LAYOUT**. 2x memory savings over BF16. Good for weights and intermediates in large models. |
| `ttnn.bfloat4_b` | Block floating point | 32 | Even more compressed, lower precision |

**Note on `bfloat8_b`**: Uses block-floating point where 16 numbers share one exponent. Magnitude variance within groups causes precision loss.

### 2.4 Memory Configurations

*   **`ttnn.DRAM_MEMORY_CONFIG`**: Stores tensor in DRAM (Global Memory). Safe default for large tensors.
*   **`ttnn.L1_MEMORY_CONFIG`**: Stores tensor in L1 SRAM (Core Cache).
    *   **Fastest access**.
    *   Limited capacity (~1.5MB per core, 210MB total on Blackhole).
    *   Use for frequent intermediate results in a compute chain.
    *   **Manual Management**: Use `ttnn.deallocate(tensor)` to free L1 space.
*   **Sharded Memory**: Distributes tensors across multiple cores' L1 via height, width, or block sharding.

## 3. Core Operations

### 3.1 Matrix Multiplication & Linear

```python
# Basic matmul
output = ttnn.matmul(a, b)
output = a @ b

# Optimized matmul (L1 output, Core Grid)
output = ttnn.matmul(
    a, b,
    memory_config=ttnn.L1_MEMORY_CONFIG,
    core_grid=ttnn.CoreGrid(y=8, x=8)
)

# Linear with transpose option (IMPORTANT: avoids manual transpose!)
output = ttnn.linear(
    input_tensor,
    weight,                          # Shape: (out_features, in_features)
    bias=bias,                       # Optional, must be on device
    transpose_a=False,
    transpose_b=True,                # Transposes weight internally!
    memory_config=ttnn.L1_MEMORY_CONFIG,
    dtype=ttnn.bfloat16,
    core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x)
)
```

**Key Point**: Use `ttnn.linear(transpose_b=True)` instead of manually transposing weights every forward pass.

### 3.2 Element-wise (Pointwise)

*   **Unary**: `relu`, `gelu`, `silu`, `exp`, `log`, `tanh`, `sigmoid`, `reciprocal`, `neg`, `abs`.
*   **Binary**: `add`, `sub`, `mul`, `div` (supports broadcasting).

```python
z = ttnn.add(x, y)
z = x + y
z = ttnn.mul(x, y)
```

### 3.3 Normalization

```python
# RMS Norm
out = ttnn.rms_norm(x, weight=scale, epsilon=1e-5)

# Layer Norm
out = ttnn.layer_norm(x, weight=gamma, bias=beta, epsilon=1e-5)

# Softmax
out = ttnn.softmax(x, dim=-1)
```

## 4. Transformer Operations (Critical for LLM Performance)

TT-NN provides highly optimized fused kernels for Transformers.

### 4.1 Rotary Position Embedding (RoPE)

**Use `ttnn.experimental.rotary_embedding` instead of manual implementation!**

```python
# Pre-compute and upload cos/sin cache ONCE during initialization
cos_cache = ttnn.from_torch(cos_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
sin_cache = ttnn.from_torch(sin_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Apply RoPE on device (prefill mode)
q_rotated = ttnn.experimental.rotary_embedding(
    q,              # Input tensor
    cos_cache,      # Cosine cache (pre-uploaded)
    sin_cache,      # Sine cache (pre-uploaded)
    memory_config=ttnn.L1_MEMORY_CONFIG
)

# Apply RoPE on device (decode mode - single token)
# When token_idx is passed, assumes input is [seq_len=1, 1, B, head_dim]
q_rotated = ttnn.experimental.rotary_embedding(
    q,
    cos_cache,
    sin_cache,
    token_index=current_position  # Position index for decode
)
```

### 4.2 KV Cache Management

**Use native KV cache ops for in-place updates instead of concat!**

```python
# Pre-allocate cache tensor
key_cache = ttnn.zeros(
    shape=(batch_size, num_kv_heads, max_seq_len, head_dim),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device
)

# Prefill: Populate cache with prompt's K/V
ttnn.kv_cache.fill_cache_for_user_(
    cache=key_cache,
    input_tensor=key_states,  # Keys from prompt processing
    batch_index=0
)

# Decode: Update cache with single new token's K/V (in-place!)
ttnn.kv_cache.update_cache_for_token_(
    cache=key_cache,
    token=new_key,           # New key for current token
    update_index=seq_position,  # Position in sequence
    batch_offset=0
)
```

### 4.3 Scaled Dot-Product Attention

```python
# Standard SDPA (for prefill)
# Q: [b, nqh, s, dh], K: [b, nkv, s, dh], V: [b, nkv, s, dh]
output = ttnn.transformer.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,          # Optional [b, 1, s, s]
    is_causal=True,          # Causal masking
    scale=1.0/math.sqrt(head_dim),
    memory_config=ttnn.L1_MEMORY_CONFIG
)

# Decode-specific SDPA (Flash-Decode, optimized for single-token generation)
# Q: [1, b, nh, dh], K: [b, nkv, s, dh], V: [b, nkv, s, dh]
output = ttnn.transformer.scaled_dot_product_attention_decode(
    q, k, v,
    cur_pos=[current_position],  # List of positions per batch
    is_causal=True,
    memory_config=ttnn.L1_MEMORY_CONFIG
)
```

### 4.4 Fused QKV Operations

**Optimized pattern from official tutorial - significantly reduces layout conversions!**

```python
# Option 1: Fused QKV projection + split + head reshape
# Input: hidden_states [batch, seq, hidden_size]
# Fused weight: [hidden_size, 3 * hidden_size]
fused_qkv_output = ttnn.linear(
    hidden_states,
    fused_qkv_weight,
    bias=fused_qkv_bias,
    memory_config=ttnn.L1_MEMORY_CONFIG,
    dtype=ttnn.bfloat8_b,
    core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
)

# Split into Q, K, V and reshape heads in ONE operation
query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
    fused_qkv_output,
    memory_config=ttnn.L1_MEMORY_CONFIG,
    num_heads=num_heads,
)
ttnn.deallocate(fused_qkv_output)

# After attention, concatenate heads back
# Input: [batch, num_heads, seq, head_dim]
# Output: [batch, seq, hidden_size]
context_layer = ttnn.transformer.concatenate_heads(
    attention_output,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
```

### 4.5 In-place Attention Softmax

```python
# In-place softmax with scaling and optional mask
attention_probs = ttnn.transformer.attention_softmax_(
    attention_scores,           # Modified in-place!
    attention_mask=attn_mask,   # Optional
    head_size=head_dim          # For scaling by 1/sqrt(head_dim)
)
```

## 5. Optimized Multi-Head Attention Pattern

Complete optimized implementation from official tutorial:

```python
def optimized_multi_head_attention(
    hidden_states,
    attention_mask,
    fused_qkv_weight,
    fused_qkv_bias,
    output_weight,
    output_bias,
    *,
    num_heads,
    num_cores_x=12,
):
    batch_size, _, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    # 1. Fused QKV projection
    fused_qkv = ttnn.linear(
        hidden_states,
        fused_qkv_weight,
        bias=fused_qkv_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
    )

    # 2. Split Q/K/V and reshape heads
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        fused_qkv,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(fused_qkv)

    # 3. Attention scores
    attention_scores = ttnn.matmul(
        query, key,  # key is automatically transposed
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    # 4. Softmax (in-place)
    attention_probs = ttnn.transformer.attention_softmax_(
        attention_scores,
        attention_mask=attention_mask,
        head_size=head_size
    )

    # 5. Attention output
    context = ttnn.matmul(
        attention_probs, value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    # 6. Concatenate heads
    context = ttnn.transformer.concatenate_heads(
        context,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # 7. Output projection
    output = ttnn.linear(
        context,
        output_weight,
        bias=output_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(context)

    return output
```

## 6. Embedding

```python
# Create embedding weight
weight = ttnn.from_torch(
    embedding_weight,
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,  # Embedding weights often ROW_MAJOR
    device=device
)

# Input indices must be uint32
indices = ttnn.from_torch(
    token_ids,
    dtype=ttnn.uint32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device
)

# Lookup
output = ttnn.embedding(indices, weight, layout=ttnn.ROW_MAJOR_LAYOUT)
output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)  # Convert for subsequent ops
```

## 7. Optimization Best Practices

### 7.1 Performance Hierarchy

1.  **Minimize Host-Device Transfer**: `from_torch`/`to_torch` are slow. Upload weights once, keep activations on device.
2.  **Use Native Transformer Ops**: `rotary_embedding`, `scaled_dot_product_attention_decode`, `split_query_key_value_and_split_heads`, etc.
3.  **Use L1 Memory**: Chain ops with `L1_MEMORY_CONFIG` to avoid DRAM round-trips.
4.  **Avoid Layout Conversions**: Keep tensors in `TILE_LAYOUT`. Use ops that work natively on tiles.
5.  **Use `ttnn.linear(transpose_b=True)`**: Avoids manual weight transpose per forward pass.
6.  **Explicit Deallocation**: Call `ttnn.deallocate()` on large intermediate tensors to free L1 space.

### 7.2 Program Caching

TT-NN compiles kernels on first run. Subsequent runs with same shapes are much faster.
*   Use static shapes where possible.
*   Warmup iterations are real: run model once before measuring.

### 7.3 Core Grid Configuration

```python
# Specify compute grid for parallel execution
output = ttnn.matmul(
    a, b,
    core_grid=ttnn.CoreGrid(y=8, x=8)
)
```

### 7.4 Async Execution

TT-NN ops are asynchronous. `to_torch()` or `ttnn.synchronize_device(device)` acts as barrier.

## 8. Debugging

*   **Print Options**: `ttnn.set_printoptions(profile="short")`
*   **Comparison**: Verify against PyTorch "Golden" reference.
*   **Graph Tracing**: Use `ttnn.graph.begin_graph_capture()` for visualization.
*   **Environment**: Set `TTNN_CONFIG_OVERRIDES` JSON for logging control.

## 9. BitNet-TT Optimization Notes

### Current Bottlenecks (Identified)

| Issue | Current Implementation | Optimized Solution |
|-------|----------------------|-------------------|
| RoPE upload every token | `numpy_to_ttnn()` per token | `ttnn.experimental.rotary_embedding` with pre-uploaded cache |
| KV Cache reallocation | `ttnn.concat()` creates new tensors | `ttnn.kv_cache.update_cache_for_token_()` in-place |
| Weight transpose | `ttnn.transpose()` every forward | `ttnn.linear(transpose_b=True)` |
| Layout conversions | ~200 conversions/token | Use `split_query_key_value_and_split_heads`, `concatenate_heads` |
| Memory config | Default DRAM | `L1_MEMORY_CONFIG` for intermediates |
| Decode attention | General SDPA | `scaled_dot_product_attention_decode` |

### Target Performance

*   Current: ~7.58 tokens/s
*   Target: ~50+ tokens/s (matching bitnet.cpp on CPU)

---

## 10. BitNet b1.58 Architecture Reference

Based on HuggingFace Transformers and Official Microsoft BitNet GPU implementation.

### 10.1 Model Configuration (BitNet b1.58 2B-4T)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `vocab_size` | 128,256 | Large vocabulary |
| `hidden_size` | 2,560 | Model dimension |
| `intermediate_size` | 6,912 | FFN hidden dim |
| `num_hidden_layers` | 30 | Transformer blocks |
| `num_attention_heads` | 20 | Query heads |
| `num_key_value_heads` | 5 | KV heads (GQA, 4:1 ratio) |
| `head_dim` | 128 | = hidden_size / num_attention_heads |
| `hidden_act` | `relu2` | Squared ReLU: `relu(x)²` |
| `max_position_embeddings` | 4,096 | (HF default 2,048, but 4,096 in practice) |
| `rms_norm_eps` | 1e-5 | Normalization epsilon |
| `rope_theta` | 500,000.0 | RoPE base frequency |
| `tie_word_embeddings` | False | lm_head ≠ embed_tokens |
| `attention_bias` | False | No bias in Q/K/V/O projections |

### 10.2 BitLinear Layer

BitNet's core innovation - ternary weights {-1, 0, +1} with 8-bit activations.

#### Weight Quantization (Pre-computed at load time)

```python
# Ternary quantization using absmean
def weight_quant(weight):
    scale = 1.0 / weight.abs().mean().clamp_(min=1e-5)
    weight_quant = (weight * scale).round().clamp(-1, 1)
    # Result: ternary values {-1, 0, +1} scaled back
    return weight_quant / scale  # For BF16 inference
    # OR return weight_quant.to(int8), 1/scale  # For INT8xINT2 kernel
```

**Key insight**: After quantization, weights become `{-scale, 0, +scale}` where `scale = mean(|W|)`.

#### Activation Quantization (Per-token, runtime)

```python
# 8-bit absmax quantization per token
def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    x_quant = (x * scale).round().clamp(-128, 127)
    return x_quant / scale  # Dequantized for BF16 compute
    # OR return x_quant.to(int8), scale  # For INT8xINT2 kernel
```

#### Forward Pass (BF16 mode - current implementation)

```python
def forward(self, x):
    # 1. Quantize activations (simulated in BF16)
    x = activation_quant(x)
    # 2. Linear with pre-quantized weights (already ternary-scaled)
    return F.linear(x, self.weight)  # weight is pre-quantized
```

### 10.3 Attention Architecture (BitNet-specific)

**Critical difference from LLaMA**: `attn_sub_norm` after attention, before output projection.

```
Input
  ↓
input_layernorm (RMSNorm)
  ↓
Q, K, V projections (BitLinear, NO input norm inside)
  ↓
RoPE on Q, K
  ↓
Scaled Dot-Product Attention
  ↓
attn_sub_norm (RMSNorm) ← BitNet-specific!
  ↓
O projection (BitLinear, NO input norm inside)
  ↓
+ Residual
```

#### Official Implementation (Fused QKV)

```python
class Attention:
    def __init__(self, ...):
        # Fused Q, K, V projection
        self.wqkv = BitLinear(dim, (n_heads + 2*n_kv_heads) * head_dim)
        self.wo = BitLinear(n_heads * head_dim, dim)
        self.attn_sub_norm = RMSNorm(dim)  # Applied to attention output

    def forward(self, x, cache, attn_bias):
        # Fused QKV projection
        xqkv = self.wqkv(x)

        # Split Q, K, V
        xq = xqkv[:, :n_heads * head_dim]
        xk = xqkv[:, n_heads * head_dim:(n_heads + n_kv_heads) * head_dim]
        xv = xqkv[:, (n_heads + n_kv_heads) * head_dim:]

        # Reshape for attention
        xq = xq.view(1, seq, n_kv_heads, heads_per_group, head_dim)
        xk = xk.view(1, seq, n_kv_heads, 1, head_dim)
        xv = xv.view(1, seq, n_kv_heads, 1, head_dim)

        # RoPE + KV Cache update + Attention (using xformers)
        xq = rope_padded(xq, xk, xv, cache_k, cache_v, attn_bias, theta=rope_theta)
        output = fmha.memory_efficient_attention_forward(xq, cache_k, cache_v, attn_bias)

        # Sub-norm before output projection
        output = self.attn_sub_norm(output)
        output = self.wo(output)
        return output
```

### 10.4 FFN Architecture (BitNet-specific)

**Critical difference**: `ffn_sub_norm` after gating, before down projection.

```
Input
  ↓
post_attention_layernorm (RMSNorm)
  ↓
┌─────────────┬─────────────┐
│ gate_proj   │  up_proj    │  (Can be fused as w13)
│ (BitLinear) │ (BitLinear) │
└──────┬──────┴──────┬──────┘
       ↓             ↓
   squared_relu      │
       ↓             ↓
       └─────*───────┘  (element-wise multiply)
             ↓
      ffn_sub_norm (RMSNorm) ← BitNet-specific!
             ↓
        down_proj (BitLinear)
             ↓
        + Residual
```

#### Official Implementation (Fused Gate/Up)

```python
class FeedForward:
    def __init__(self, dim, hidden_dim, norm_eps):
        # Fused gate + up projection
        self.w13 = BitLinear(dim, 2 * hidden_dim)
        self.w2 = BitLinear(hidden_dim, dim)
        self.ffn_sub_norm = RMSNorm(hidden_dim)

    def forward(self, x):
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, dim=-1)

        # Squared ReLU + gating
        inner = squared_relu(x1) * x3

        # Sub-norm before down projection
        inner = self.ffn_sub_norm(inner)
        return self.w2(inner)

def squared_relu(x):
    return F.relu(x) ** 2
```

### 10.5 KV Cache Shape (Official)

```python
# Shape: (1, max_seq_len, n_kv_heads, 1, head_dim)
# The extra dimensions are for GQA expansion
cache_shape = (1, length, n_kv_heads, 1, head_dim)
# Expanded for GQA: (1, length, n_kv_heads, heads_per_group, head_dim)
expansion = (-1, -1, -1, heads_per_group, -1)
```

### 10.6 INT8xINT2 Kernel (GPU Reference)

For ultimate performance, Microsoft's GPU implementation uses:

1. **Activation**: INT8 (per-token absmax quantization)
2. **Weight**: INT2 (ternary {-1,0,+1} packed 4 per byte)
3. **Compute**: CUDA `__dp4a` instruction (dot product of 4 int8 values)

```cpp
// Kernel signature
void bitlinear_int8xint2(
    int8_t* activations,   // [M, K] INT8
    int8_t* weights,       // [N, K/4] packed INT2
    bfloat16* output,      // [M, N] BF16
    bfloat16* act_scale,   // Per-token activation scale
    bfloat16* weight_scale // Per-layer weight scale
);

// Weight packing: 4 ternary values per byte
// {-1,0,+1} → {0,1,2} → packed as 2-bit values
// Special memory layout for WMMA (16x32 tiles)
```

### 10.7 Optimization Opportunities for TT-NN

Based on official implementations, key optimizations for TT-NN:

| Feature | Official GPU Impl | TT-NN Approach |
|---------|------------------|----------------|
| Fused QKV | `wqkv` single projection | Use `ttnn.linear` → `split_query_key_value_and_split_heads` |
| Fused Gate/Up | `w13` single projection | Combine gate_proj + up_proj weights |
| RoPE | `xformers.rope_padded` | `ttnn.experimental.rotary_embedding` |
| Attention | Flash Attention (`fmha`) | `ttnn.transformer.scaled_dot_product_attention_decode` |
| Sub-norms | RMSNorm between ops | Keep using `ttnn.rms_norm` |
| INT quantization | INT8xINT2 kernel | BF16 simulation (or future INT support) |

### 10.8 Weight Loading Notes

HuggingFace safetensors keys → Internal mapping:

```python
# Embedding & Output
"model.embed_tokens.weight" → embed_tokens
"model.norm.weight" → final_norm
"lm_head.weight" → lm_head (NOT tied to embed_tokens)

# Per-layer (layer_idx = 0..29)
"model.layers.{i}.input_layernorm.weight"
"model.layers.{i}.post_attention_layernorm.weight"
"model.layers.{i}.self_attn.q_proj.weight"
"model.layers.{i}.self_attn.k_proj.weight"
"model.layers.{i}.self_attn.v_proj.weight"
"model.layers.{i}.self_attn.o_proj.weight"
"model.layers.{i}.self_attn.attn_sub_norm.weight"  # BitNet-specific
"model.layers.{i}.mlp.gate_proj.weight"
"model.layers.{i}.mlp.up_proj.weight"
"model.layers.{i}.mlp.down_proj.weight"
"model.layers.{i}.mlp.ffn_sub_norm.weight"  # BitNet-specific
```

**Note**: HuggingFace uses separate Q/K/V projections, while official impl uses fused `wqkv`. Consider fusing for TT-NN optimization.

---

## 11. TT-Transformers Reference (Official Llama 3.1 Implementation)

Based on `~/tt-metal/models/tt_transformers/` - Achieves **1059.2 t/s (33.1 t/s/u)** for Llama 3.1 8B on p150.

### 11.1 Performance Results (README.md)

| Model | Device | Batch | Throughput | Per-User |
|-------|--------|-------|------------|----------|
| Llama 3.1 8B | p150a | 32 | **1059.2 t/s** | 33.1 t/s/u |
| Llama 3.2 1B | p150a | 32 | 2182.5 t/s | 68.2 t/s/u |
| Llama 3.2 3B | p150a | 32 | 1497.4 t/s | 46.8 t/s/u |

**Key insight**: Batching is critical for throughput. Single-user performance ~33 t/s for 8B model.

### 11.2 Trace Capture Pattern (Critical for Decode Performance)

```python
# 1. Compile run (first pass compiles kernels)
self._decode_forward_no_trace(tokens, current_pos, page_table, kv_cache)
logger.info("Done Compiling Model")

# 2. Prepare inputs on device (before trace)
device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)

# 3. Capture trace
trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
tt_out = model.ttnn_decode_forward(*device_inputs, kv_cache=kv_cache)
ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
logger.info("Done Capturing Decode Trace")

# 4. Execute trace (subsequent iterations - FAST!)
for step in range(max_tokens):
    # Update input tensors in-place
    copy_host_to_device(new_host_inputs, device_tensors=device_inputs)
    # Execute pre-captured trace
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
```

**Benefits**:
- Eliminates kernel compilation overhead per step
- Reuses memory allocations
- Enables async execution with `blocking=False`

### 11.3 RoPE Optimization (RotarySetup Class)

```python
class RotarySetup:
    def __init__(self, device, batch_size, head_dim, max_seq_len, rope_theta, ...):
        # Pre-compute and upload cos/sin matrices ONCE
        self.cos_matrix, self.sin_matrix = get_rot_mats(
            head_dim=head_dim,
            device=device,
            seq_len=max_seq_len,
            theta=rope_theta,
        )
        # Pre-allocate transformation matrix for decode
        self.transformation_mat = ttnn.from_torch(
            trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=trans_mat_mem_config,  # Sharded!
        )

    def get_rot_mats(self, position_idxs):
        """Get rotation matrices for given positions - uses ttnn.embedding!"""
        rot_idxs = ttnn.as_tensor(position_idxs, dtype=ttnn.uint32, device=device)

        # Use embedding lookup instead of slicing!
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)

        # Reshape and shard for decode
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)
        cos = ttnn.interleaved_to_sharded(cos, sharded_mem_config)
        sin = ttnn.interleaved_to_sharded(sin, sharded_mem_config)

        return [cos, sin]
```

**Key technique**: Use `ttnn.embedding()` to index into pre-uploaded cos/sin matrices instead of slicing on host.

### 11.4 Attention Implementation (attention.py)

```python
class Attention:
    def __init__(self, ...):
        # Separate Q/K/V weights (not fused in this implementation)
        self.wq = ttnn.as_tensor(wq, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.wk = ttnn.as_tensor(wk, dtype=dtype, ...)
        self.wv = ttnn.as_tensor(wv, dtype=dtype, ...)
        self.wo = ttnn.as_tensor(wo, dtype=dtype, ...)

        # Pre-allocated KV cache (Paged attention)
        # Shape: [num_layers, batch, max_context, n_kv_heads, head_dim]

    def forward(self, x, current_pos, rot_mats, page_table=None, kv_cache=None):
        # 1. QKV projections with fused activation
        xq = ttnn.linear(x, self.wq, dtype=ttnn.bfloat16, memory_config=L1_MEMCFG)
        xk = ttnn.linear(x, self.wk, dtype=ttnn.bfloat16, memory_config=L1_MEMCFG)
        xv = ttnn.linear(x, self.wv, dtype=ttnn.bfloat16, memory_config=L1_MEMCFG)

        # 2. Create QKV heads (optimized ops for decode)
        if mode == "decode":
            xq = ttnn.experimental.nlp_create_qkv_heads_decode(
                xq, num_heads=n_heads, num_kv_heads=n_kv_heads, head_dim=head_dim
            )
            xk = ttnn.experimental.nlp_create_qkv_heads_decode(...)
            xv = ttnn.experimental.nlp_create_qkv_heads_decode(...)
        else:  # prefill
            xq, xk, xv = ttnn.experimental.nlp_create_qkv_heads(
                xq, xk, xv, num_heads=n_heads, num_kv_heads=n_kv_heads
            )

        # 3. RoPE (using pre-computed rotation matrices)
        xq = ttnn.experimental.rotary_embedding_llama(xq, rot_mats[0], rot_mats[1], trans_mat)
        xk = ttnn.experimental.rotary_embedding_llama(xk, rot_mats[0], rot_mats[1], trans_mat)

        # 4. Update KV cache (paged, in-place)
        ttnn.experimental.paged_update_cache(kv_cache[0], xk, update_idxs=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(kv_cache[1], xv, update_idxs=current_pos, page_table=page_table)

        # 5. Scaled Dot-Product Attention (decode optimized)
        if mode == "decode":
            attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
                xq, keys, values,
                cur_pos=current_pos,
                memory_config=L1_MEMCFG,
                compute_kernel_config=compute_kernel_config,
            )
        else:  # prefill
            attn_output = ttnn.transformer.scaled_dot_product_attention(
                xq, keys, values,
                is_causal=True,
                memory_config=L1_MEMCFG,
            )

        # 6. Concat heads (optimized ops)
        if mode == "decode":
            attn_output = ttnn.experimental.nlp_concat_heads_decode(attn_output, num_heads=n_heads)
        else:
            attn_output = ttnn.experimental.nlp_concat_heads(attn_output)

        # 7. Output projection
        return ttnn.linear(attn_output, self.wo, memory_config=L1_MEMCFG)
```

### 11.5 MLP Implementation (mlp.py)

```python
class MLP:
    def __init__(self, ...):
        # Fused FF1 (gate) + FF3 (up) weights
        self.w1 = ttnn.as_tensor(w1, dtype=dtype, ...)  # gate_proj
        self.w3 = ttnn.as_tensor(w3, dtype=dtype, ...)  # up_proj
        self.w2 = ttnn.as_tensor(w2, dtype=dtype, ...)  # down_proj

    def forward(self, x, mode="decode"):
        # Option 1: Separate projections with fused multiply
        ff1_out = ttnn.linear(
            x, self.w1,
            memory_config=L1_MEMCFG,
            dtype=ttnn.bfloat8_b,
            activation="silu",  # Fused activation!
        )
        ff3_out = ttnn.linear(x, self.w3, memory_config=L1_MEMCFG, dtype=ttnn.bfloat8_b)

        # Multiply (gating)
        hidden = ttnn.mul(ff1_out, ff3_out, memory_config=L1_MEMCFG)
        ttnn.deallocate(ff1_out)
        ttnn.deallocate(ff3_out)

        # Down projection
        return ttnn.linear(hidden, self.w2, memory_config=L1_MEMCFG, dtype=ttnn.bfloat16)
```

### 11.6 Memory Configurations (Sharding)

```python
# Decode mode: WIDTH sharding across cores
DECODE_RESIDUAL_MEMCFG = ttnn.create_sharded_memory_config(
    shape=(tile_size, hidden_size // num_cores),  # Per-core shard
    core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

# Prefill mode: DRAM (larger tensors)
PREFILL_MEMCFG = ttnn.DRAM_MEMORY_CONFIG

# L1 for intermediate activations
L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
```

### 11.7 Model Optimizations (Precision Settings)

```python
class ModelOptimizations:
    @classmethod
    def performance(cls, model_name):
        """BFP4 for FF1/FF3, LOFI math fidelity"""
        return cls({
            "TensorPrecision": {TensorGroup.FF1_FF3: PrecisionSetting.BFP4},
            "OpFidelity": {OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI},
        })

    @classmethod
    def accuracy(cls, model_name):
        """BF16 weights, HIFI4 math fidelity"""
        return cls({
            "TensorPrecision": {
                TensorGroup.WQKV: PrecisionSetting.BF16,
                TensorGroup.KV_CACHE: PrecisionSetting.BF16,
            },
            "OpFidelity": {
                OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI4,
            },
        })
```

### 11.8 Generator Pattern (Decode Loop)

```python
class Generator:
    def decode_forward_text(self, tokens, start_pos, page_table, kv_cache, enable_trace=True):
        if enable_trace:
            # First call: capture trace
            if self.trace_id is None:
                self.trace_id, self.trace_output, *self.trace_inputs = self._capture_trace(...)

            # Subsequent calls: update inputs and execute trace
            copy_host_to_device(new_host_inputs, device_tensors=self.trace_inputs)
            ttnn.execute_trace(mesh_device, self.trace_id, cq_id=0, blocking=False)
            return self.trace_output
        else:
            return self._decode_forward_no_trace(...)

    def _capture_trace(self, tokens, current_pos, ...):
        # 1. Compile
        self._decode_forward_no_trace(...)

        # 2. Prepare device inputs
        device_inputs = copy_host_to_device(host_inputs)

        # 3. Capture
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output = model.ttnn_decode_forward(*device_inputs)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

        return trace_id, output, *device_inputs
```

### 11.9 Key Optimizations Summary for BitNet-TT

Based on tt_transformers, the following optimizations should be applied:

| Feature | tt_transformers Pattern | BitNet-TT Action |
|---------|------------------------|------------------|
| **RoPE** | `RotarySetup` + `ttnn.embedding` lookup | Pre-upload cos/sin, use embedding indexing |
| **KV Cache** | `ttnn.experimental.paged_update_cache` | Use paged cache with in-place updates |
| **Attention** | `nlp_create_qkv_heads_decode` + `nlp_concat_heads_decode` | Use optimized head create/concat ops |
| **SDPA** | `scaled_dot_product_attention_decode` | Use decode-specific SDPA |
| **MLP** | Fused activation in `ttnn.linear` | Use `activation="silu"` parameter |
| **Trace** | `begin_trace_capture` / `execute_trace` | Capture decode loop, execute repeatedly |
| **Memory** | Sharded configs for decode | Use `create_sharded_memory_config` |
| **Precision** | BFP4/BFP8 for weights | Consider `bfloat8_b` for attention weights |
| **Input transfer** | `copy_host_to_device` with reuse | Reuse device buffers, update in-place |

### 11.10 Expected Performance Impact

Applying these optimizations to BitNet-TT:

- **Current**: ~7.58 tokens/s (single user)
- **With trace capture**: ~2-3x improvement (eliminate kernel compilation)
- **With optimized RoPE**: ~1.5x improvement (eliminate host transfers)
- **With paged KV cache**: ~1.5x improvement (eliminate allocations)
- **With SDPA decode**: ~1.2x improvement (optimized attention kernel)
- **Combined target**: **30-50+ tokens/s** (matching bitnet.cpp)
