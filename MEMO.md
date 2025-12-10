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

---

## 12. Implementation Guide

### 12.1 Optimization Priority & Impact Summary

| Phase | Optimization | Speedup | KV Memory | Effort | Priority |
|-------|--------------|---------|-----------|--------|----------|
| 1 | Trace Capture | 2-3x | - | 4h | Critical |
| 2 | Paged Attention | - | 8.5GB → 64MB | 8h | Critical |
| 3 | Memory Configs | 5-10% | - | 4h | High |
| 4 | RoPE Pre-upload | 5% | - | 2h | Medium |
| 5 | Split Sampling | 5% (33x I/O) | - | 3h | Medium |

**Total Expected**: 2.6-3.1x cumulative speedup + 135x KV reduction

### 12.2 Paged Attention Implementation

Block-based KV cache allocation (reduces 8.5GB → 64MB):

```python
@dataclass
class PagedKVCache:
    """Block-based KV cache for memory efficiency."""
    block_size: int = 128  # Tokens per block
    max_num_blocks: int = 256  # For ~32K max length
    key_cache: ttnn.Tensor | None = None
    value_cache: ttnn.Tensor | None = None
    page_table: ttnn.Tensor | None = None

    def preallocate(self, batch_size, num_kv_heads, head_dim, device):
        """Allocate block-based cache."""
        # Shape: [max_blocks, kv_heads, block_size, head_dim]
        # = 256 * 8 * 128 * 128 * 2 bytes = 64 MB (vs 8.5 GB full allocation!)
        cache_shape = (self.max_num_blocks, num_kv_heads, self.block_size, head_dim)
        zeros = torch.zeros(cache_shape, dtype=torch.bfloat16)

        self.key_cache = ttnn.from_torch(
            zeros, device=device, layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.value_cache = ttnn.from_torch(
            zeros, device=device, layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def update_decode(self, key_states, value_states, seq_position):
        """In-place block update using paged_update_cache."""
        ttnn.experimental.paged_update_cache(
            self.key_cache, key_states,
            update_idxs_tensor=ttnn.from_torch(
                torch.tensor([seq_position]), device=self.key_cache.device
            ),
            page_table=self.page_table,
        )
        ttnn.experimental.paged_update_cache(
            self.value_cache, value_states,
            update_idxs_tensor=ttnn.from_torch(
                torch.tensor([seq_position]), device=self.value_cache.device
            ),
            page_table=self.page_table,
        )
        return self.key_cache, self.value_cache
```

### 12.3 Memory Config Decision Tree

```
Output size < 1 MB?           → L1_WIDTH_SHARDED
Output reused immediately?    → L1_WIDTH_SHARDED
Output is reduction/SDPA?     → DRAM_MEMORY_CONFIG
Large intermediate (FFN1)?    → DRAM_MEMORY_CONFIG
```

**Practical pattern:**
```python
class BitNetAttention:
    def __init__(self, ...):
        # QKV output goes to L1 (reused immediately)
        self.xqkv_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

        # SDPA output goes to DRAM (large reduction)
        self.sdpa_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Concat heads output back to L1
        self.concat_heads_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

class BitNetMLP:
    def __init__(self, ...):
        # FFN1 output to DRAM (large intermediate)
        self.ff1_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # FFN2 output back to L1 if possible
        self.ff2_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
```

### 12.4 Common Pitfalls & Solutions

#### Pitfall 1: Trace compilation hangs
**Symptom**: Program stuck on `ttnn.begin_trace_capture()`
**Cause**: Weights not on device, undefined shapes
**Solution**:
```python
# Before capturing trace, ensure:
assert self.wqkv.device == self.device
assert self.kv_cache.key_cache.device == self.device
# Run forward once without trace first
_ = self.model.ttnn_decode_forward(dummy_input)
ttnn.synchronize_device(self.device)
```

#### Pitfall 2: Page table index out of bounds
**Symptom**: Segfault in `paged_update_cache()`
**Cause**: Block indices exceed `max_num_blocks`
**Solution**:
```python
def update_decode(self, k, v, seq_position):
    block_idx = seq_position // self.block_size
    if block_idx >= self.max_num_blocks:
        raise ValueError(f"Sequence too long: {seq_position} exceeds "
                        f"{self.max_num_blocks * self.block_size}")
```

#### Pitfall 3: L1 memory overflow
**Symptom**: `ttnn.linear()` fails with "out of L1 memory"
**Cause**: Too many WIDTH_SHARDED configs
**Solution**:
```python
# Don't shard outputs that aren't immediately reused
qkv = ttnn.linear(x, self.wqkv, memory_config=ttnn.DRAM_MEMORY_CONFIG)
# Then shard on-demand if needed
qkv_l1 = ttnn.to_memory_config(qkv, ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
```

#### Pitfall 4: Embedding lookup shape mismatch
**Symptom**: RoPE `embedding()` returns wrong shape
**Cause**: Input indices not properly formatted
**Solution**:
```python
# Embedding expects [batch, seq_len] for 2D lookup
position_idxs = torch.tensor([seq_position], dtype=torch.uint32)
position_idxs = position_idxs.reshape(1, 1)  # [1, 1] for single position
cos = ttnn.embedding(position_idxs, self.cos_matrix)
assert cos.shape[-1] == self.head_dim  # Verify!
```

### 12.5 Validation Script

```python
def validate_optimizations(model, generator, test_prompt, num_tokens=10):
    """Validate all optimizations work correctly."""

    # 1. Test trace capture
    assert generator.enable_trace
    output = generator.generate(test_prompt, max_tokens=num_tokens)
    assert generator._trace_captured
    print("✓ Trace capture working")

    # 2. Test paged attention
    assert isinstance(generator.model.kv_cache, PagedKVCache)
    assert generator.model.kv_cache.max_num_blocks == 256
    print("✓ Paged attention working")

    # 3. Test memory configs (verify via ttnn.Device memory usage)
    initial_dram = measure_dram_usage(generator.device)
    output = generator.generate(test_prompt, max_tokens=1)
    final_dram = measure_dram_usage(generator.device)
    assert final_dram - initial_dram < 100_000_000  # < 100 MB per token
    print(f"✓ Memory configs working (DRAM delta: {(final_dram - initial_dram) / 1e6:.1f} MB)")

    # 4. Test RoPE pre-upload
    assert hasattr(generator.model.attention, 'rope')
    cos, sin = generator.model.attention.rope.get_rot_mats(torch.tensor([0]))
    assert cos.shape[-1] == generator.model.head_dim
    print("✓ RoPE pre-upload working")

    print("\nAll validations passed!")
```

### 12.6 Performance Metrics Reference (Llama 3.1 8B on p150)

**33 t/s/u breakdown:**
- Trace overhead: Amortized to <1% per token
- SDPA decode: ~10ms (bottleneck)
- QKV linear: ~5ms (width-sharded)
- All-reduce: ~2ms (ring topology)
- **Total decode loop: ~30ms = 33 t/s**

**KV Cache footprint comparison:**
| Mode | Per-layer | Total (32 layers) |
|------|-----------|-------------------|
| Full pre-allocated | 8.5 GB | 272 GB |
| Paged attention | 64 MB | 2 GB |
| **Reduction** | **135x** | **135x** |

---

## 13. Metal Trace 심화 가이드

TT-Metal의 Metal Trace는 모델 성능 최적화에 **가장 중요한** 기능입니다.

### 13.1 개요

Metal Trace는 호스트의 연산 디스패칭 오버헤드를 제거합니다:
- **문제**: 호스트가 각 연산을 구성하고 디스패치하는 시간이 디바이스 연산 시간보다 길면 디바이스가 대기 상태
- **해결**: 연산 디스패치 명령을 DRAM 버퍼에 기록 → 이후 재실행 시 즉시 실행

```
호스트 바운드 (트레이스 없음):
[Host: op1 디스패치] → [Device: op1 대기...실행] → [Host: op2 디스패치] → [Device: op2 대기...실행]
                        ^^^^^^^^ 긴 대기 ^^^^^^^^

트레이스 사용:
[Host: 트레이스 실행] → [Device: op1→op2→op3→...] (거의 대기 없음)
```

### 13.2 핵심 API

```python
# 1. 디바이스 생성 시 trace_region_size 지정
device = ttnn.open_device(device_id=0, trace_region_size=800768)

# 2. 트레이스 캡처 시작
trace_id = ttnn.begin_trace_capture(device, cq_id=0)

# 3. 연산 실행 (캡처됨)
output = model.forward(input)

# 4. 트레이스 캡처 종료
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# 5. 트레이스 실행 (빠름!)
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
```

### 13.3 영구 DRAM 입력 패턴 (권장)

```python
# 1. 영구 입력 텐서 할당
input_dram_tensor = ttnn.allocate_tensor_on_device(
    tensor_spec, device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)

# 2. 컴파일 런
ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=0)
input_l1_tensor = ttnn.to_memory_config(input_dram_tensor, L1_MEMCFG)
output_tensor = model.forward(input_l1_tensor)

# 3. 트레이스 캡처
ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=0)
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
input_l1_tensor = ttnn.to_memory_config(input_dram_tensor, L1_MEMCFG)
output_tensor = model.forward(input_l1_tensor)  # 이 output 참조 유지 필수!
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# 4. 트레이스 실행 (반복)
for _ in range(iterations):
    ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    host_output = output_tensor.cpu(blocking=False)
ttnn.synchronize_device(device)
```

### 13.4 비영구 L1 입력 패턴 (고급)

모델이 L1에 입력 텐서를 영구 보관할 메모리가 없을 때:

```python
# 컴파일 런
input_l1 = host_tensor.to(device, L1_MEMCFG)
output = model.forward(input_l1)

# 트레이스 캡처 시 주소 기록
input_l1 = host_tensor.to(device, L1_MEMCFG)
input_trace_addr = input_l1.buffer_address()  # 주소 기록
spec = input_l1.spec

output.deallocate(force=True)  # 이전 출력 해제로 같은 주소 확보

trace_id = ttnn.begin_trace_capture(device, cq_id=0)
output = model.forward(input_l1)

# 트레이스 종료 전 같은 주소에 입력 할당
input_l1 = ttnn.allocate_tensor_on_device(spec, device)
assert input_trace_addr == input_l1.buffer_address()  # 주소 검증!

ttnn.end_trace_capture(device, trace_id, cq_id=0)

# 실행
ttnn.copy_host_to_device_tensor(host_tensor, input_l1, cq_id=0)
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
```

### 13.5 트레이스 제한사항

1. **정적 형태만 지원**: 입출력 텐서 크기/주소가 캡처 시점에 고정
2. **이벤트 캡처 불가**: `record_event`, `wait_for_event`는 트레이스 외부에서
3. **I/O 캡처 불가**: `copy_host_to_device`, `cpu()` 등은 트레이스 외부에서
4. **프로그램 캐시 필수**: 트레이스 전에 컴파일 런 필요

---

## 14. 다중 커맨드 큐 (Multiple CQs)

### 14.1 개요

2개의 독립적인 커맨드 큐로 I/O와 연산을 병렬화:
- **CQ 0**: 연산 디스패치 + 출력 읽기
- **CQ 1**: 입력 쓰기

```
단일 CQ (I/O 바운드):
[Write] → [Ops...] → [Read] → [Write] → [Ops...] → [Read]
          ^^^^^^^^ 모델 대기 ^^^^^^^^

다중 CQ (오버랩):
CQ 0: [Ops...] → [Ops...] → [Ops...]
CQ 1: [Write] → [Write] → [Write]
```

### 14.2 핵심 API

```python
# 디바이스 생성 시 2개 CQ 요청
device = ttnn.open_device(device_id=0, num_command_queues=2)

# 이벤트 기록 (해당 CQ의 현재 명령 완료 후 기록)
event = ttnn.record_event(device, cq_id=0)

# 이벤트 대기 (해당 CQ가 이벤트 발생까지 대기)
ttnn.wait_for_event(cq_id=1, event=event)
```

### 14.3 연산 + 출력: CQ0, 입력 쓰기: CQ1 패턴

```python
# 영구 입력 텐서 할당
input_dram = ttnn.allocate_tensor_on_device(spec, device, DRAM_MEMCFG)

# 초기 더미 이벤트 (루프 시작용)
op_event = ttnn.record_event(device, 0)

outputs = []
for _ in range(iterations):
    # CQ1: 이전 연산 완료 대기 후 입력 쓰기
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(host_tensor, input_dram, cq_id=1)
    write_event = ttnn.record_event(device, 1)

    # CQ0: 쓰기 완료 대기 후 연산
    ttnn.wait_for_event(0, write_event)
    input_l1 = ttnn.to_memory_config(input_dram, L1_MEMCFG)
    op_event = ttnn.record_event(device, 0)  # 입력 소비 완료 신호

    output = model.forward(input_l1)
    outputs.append(output.cpu(blocking=False))

ttnn.synchronize_device(device)
```

### 14.4 4-이벤트 패턴 (CQ1에서 읽기+쓰기)

```python
# 이벤트: first_op (입력 소비), write (쓰기 완료), last_op (출력 생성), read (읽기 완료)

input_dram = ttnn.allocate_tensor_on_device(input_spec, device, DRAM_MEMCFG)
output_dram = ttnn.allocate_tensor_on_device(output_spec, device, DRAM_MEMCFG)

first_op_event = ttnn.record_event(device, 0)
read_event = ttnn.record_event(device, 1)

# 첫 입력 미리 쓰기
ttnn.wait_for_event(1, first_op_event)
ttnn.copy_host_to_device_tensor(host_tensor, input_dram, cq_id=1)
write_event = ttnn.record_event(device, 1)

for _ in range(iterations):
    # CQ0: 쓰기 완료 대기 → 연산
    ttnn.wait_for_event(0, write_event)
    input_l1 = ttnn.to_memory_config(input_dram, L1_MEMCFG)
    first_op_event = ttnn.record_event(device, 0)

    output = model.forward(input_l1)

    ttnn.wait_for_event(0, read_event)
    output_dram = ttnn.reshard(output, OUTPUT_DRAM_MEMCFG, output_dram)
    last_op_event = ttnn.record_event(device, 0)

    # CQ1: 다음 입력 쓰기 + 이전 출력 읽기
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(next_host_tensor, input_dram, cq_id=1)
    write_event = ttnn.record_event(device, 1)

    ttnn.wait_for_event(1, last_op_event)
    outputs.append(output_dram.cpu(blocking=False, cq_id=1))
    read_event = ttnn.record_event(device, 1)
```

---

## 15. Trace + Multi-CQ 결합 패턴 (최고 성능)

### 15.1 개요

두 최적화를 결합하면:
- 호스트가 디바이스보다 **훨씬 앞서** 명령 전송
- 디바이스는 연산 간 **거의 대기 없이** 연속 실행

### 15.2 결합 패턴 구현

```python
# 설정
device = ttnn.open_device(
    device_id=0,
    trace_region_size=800768,
    num_command_queues=2,
)
input_dram = ttnn.allocate_tensor_on_device(spec, device, DRAM_MEMCFG)

op_event = ttnn.record_event(device, 0)

# 컴파일 런 (트레이스 없이)
ttnn.wait_for_event(1, op_event)
ttnn.copy_host_to_device_tensor(host_tensor, input_dram, cq_id=1)
write_event = ttnn.record_event(device, 1)
ttnn.wait_for_event(0, write_event)
input_l1 = ttnn.to_memory_config(input_dram, L1_MEMCFG)
op_event = ttnn.record_event(device, 0)
output = model.forward(input_l1)

# 트레이스 캡처 (중요: 이벤트는 캡처 외부)
ttnn.wait_for_event(1, op_event)
ttnn.copy_host_to_device_tensor(host_tensor, input_dram, cq_id=1)
write_event = ttnn.record_event(device, 1)
ttnn.wait_for_event(0, write_event)
input_l1 = ttnn.to_memory_config(input_dram, L1_MEMCFG)
op_event = ttnn.record_event(device, 0)

# 트레이스 입력 주소 기록
input_trace_addr = input_l1.buffer_address()
spec = input_l1.spec
output.deallocate(force=True)

# 트레이스 시작 (연산만 캡처)
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
output = model.forward(input_l1)

# 같은 주소에 입력 재할당
input_l1 = ttnn.allocate_tensor_on_device(spec, device)
assert input_trace_addr == input_l1.buffer_address()
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# 트레이스 실행 루프
outputs = []
for _ in range(iterations):
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(host_tensor, input_dram, cq_id=1)
    write_event = ttnn.record_event(device, 1)

    ttnn.wait_for_event(0, write_event)
    input_l1 = ttnn.reshard(input_dram, L1_MEMCFG, input_l1)  # 제자리 리샤드
    op_event = ttnn.record_event(device, 0)

    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    outputs.append(output.cpu(blocking=False))

ttnn.synchronize_device(device)
```

### 15.3 핵심 주의사항

1. **이벤트는 트레이스 외부**: `record_event`, `wait_for_event`는 캡처 대상 아님
2. **입력 소비자 연산 분리**: 트레이스 입력의 첫 연산 후 바로 이벤트 기록
3. **주소 일관성 검증**: 트레이스 종료 전 입력 텐서 주소 확인
4. **리샤드로 제자리 업데이트**: `ttnn.reshard(src, config, dst)`로 기존 텐서에 쓰기

---

## 16. TT-NN LLM 구현 패턴 (llms.md 요약)

### 16.1 RoPE 구현 (공식 패턴)

```python
class TtLlamaRotarySetup:
    def __init__(self, device, head_dim, max_seq_len, rope_theta):
        # 1. cos/sin 행렬 사전 계산
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2) / head_dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        cos_cache = torch.cos(freqs)
        sin_cache = torch.sin(freqs)

        # 2. 디바이스에 업로드 (한 번만!)
        self.cos_matrix = ttnn.from_torch(
            cos_cache, device=device, layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.sin_matrix = ttnn.from_torch(
            sin_cache, device=device, layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def get_rot_mats(self, position_idxs: torch.Tensor):
        """위치 인덱스로 cos/sin 조회 (ttnn.embedding 사용!)"""
        pos_tensor = ttnn.from_torch(
            position_idxs.to(torch.int32),
            device=self.device, dtype=ttnn.uint32
        )
        # 임베딩 룩업으로 빠른 인덱싱
        cos = ttnn.embedding(pos_tensor, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(pos_tensor, self.sin_matrix, layout=ttnn.TILE_LAYOUT)
        return cos, sin
```

### 16.2 Attention 구현 (Prefill vs Decode)

```python
class TtLlamaAttention:
    def forward(self, x, current_pos, rot_mats, kv_cache, mode="decode"):
        # QKV 프로젝션
        xq = ttnn.linear(x, self.wq, memory_config=L1_MEMCFG)
        xk = ttnn.linear(x, self.wk, memory_config=L1_MEMCFG)
        xv = ttnn.linear(x, self.wv, memory_config=L1_MEMCFG)

        if mode == "decode":
            # 1. 헤드 생성 (디코드 최적화)
            xq = ttnn.experimental.nlp_create_qkv_heads_decode(
                xq, num_heads=self.n_heads,
                num_kv_heads=self.n_kv_heads, head_dim=self.head_dim
            )
            # ... xk, xv도 동일

            # 2. RoPE 적용 (융합 연산)
            xq = ttnn.experimental.rotary_embedding_llama(
                xq, rot_mats[0], rot_mats[1], self.trans_mat
            )
            xk = ttnn.experimental.rotary_embedding_llama(
                xk, rot_mats[0], rot_mats[1], self.trans_mat
            )

            # 3. KV 캐시 업데이트 (페이지드, 제자리)
            ttnn.experimental.paged_update_cache(
                kv_cache[0], xk, update_idxs=current_pos, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                kv_cache[1], xv, update_idxs=current_pos, page_table=page_table
            )

            # 4. SDPA (디코드 최적화)
            attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
                xq, kv_cache[0], kv_cache[1], cur_pos=current_pos
            )

            # 5. 헤드 결합
            attn_output = ttnn.experimental.nlp_concat_heads_decode(
                attn_output, num_heads=self.n_heads
            )

        else:  # prefill
            xq, xk, xv = ttnn.experimental.nlp_create_qkv_heads(
                xq, xk, xv, num_heads=self.n_heads, num_kv_heads=self.n_kv_heads
            )
            # RoPE (프리필용)
            xq = ttnn.experimental.rotary_embedding(xq, rot_mats[0], rot_mats[1])
            xk = ttnn.experimental.rotary_embedding(xk, rot_mats[0], rot_mats[1])

            # KV 캐시 채우기
            ttnn.experimental.paged_fill_cache(kv_cache[0], xk, page_table)
            ttnn.experimental.paged_fill_cache(kv_cache[1], xv, page_table)

            # SDPA (프리필)
            attn_output = ttnn.transformer.scaled_dot_product_attention(
                xq, kv_cache[0], kv_cache[1], is_causal=True
            )

            # 헤드 결합
            attn_output = ttnn.experimental.nlp_concat_heads(attn_output)

        return ttnn.linear(attn_output, self.wo, memory_config=L1_MEMCFG)
```

### 16.3 MLP 구현 (DRAM-Sharded Matmul)

대형 FFN (intermediate_size > 8K)은 DRAM 샤딩 필요:

```python
class TtLlamaMLP:
    def __init__(self, args, device):
        # 가중치를 DRAM에 샤딩
        self.w1 = ttnn.as_tensor(
            w1, device=device, dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG  # 큰 가중치는 DRAM
        )

    def forward(self, x):
        # gate와 up을 별도 연산 (또는 fused w13)
        ff1 = ttnn.linear(
            x, self.w1,
            activation="silu",  # 융합 활성화!
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # 큰 출력은 DRAM
            dtype=ttnn.bfloat8_b,
        )
        ff3 = ttnn.linear(x, self.w3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # 게이팅
        hidden = ttnn.mul(ff1, ff3, memory_config=L1_MEMCFG)

        # 다운 프로젝션
        return ttnn.linear(hidden, self.w2, memory_config=L1_MEMCFG)
```

### 16.4 프로그램 설정 (Program Configs)

최적 코어 활용을 위한 설정:

```python
# 디코드 모드: 시퀀스 길이 1이므로 배치 축으로 병렬화
DECODE_MATMUL_CONFIG = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),  # 8x8 코어 그리드
    in0_block_w=4,   # 입력 블록 너비
    out_subblock_h=1,
    out_subblock_w=4,
    per_core_M=1,    # 디코드: 배치당 1 타일
    per_core_N=4,    # 출력 차원 분할
)

# 프리필 모드: 시퀀스 길이가 길어서 시퀀스 축으로도 병렬화
PREFILL_MATMUL_CONFIG = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=2,
    out_subblock_h=4,
    out_subblock_w=2,
    per_core_M=seq_len // 32 // 8,  # 시퀀스 분할
    per_core_N=hidden_dim // 32 // 8,
)
```

### 16.5 메모리 설정 결정 트리 (상세)

```python
def get_memory_config(tensor_role, tensor_size_mb, mode):
    """텐서 역할과 크기에 따른 메모리 설정 결정."""

    if mode == "prefill":
        # 프리필은 큰 텐서 → DRAM
        return ttnn.DRAM_MEMORY_CONFIG

    # 디코드 모드
    if tensor_role in ["qkv_output", "attn_scores"]:
        # 즉시 재사용 → L1 샤딩
        if tensor_size_mb < 1.0:
            return ttnn.create_sharded_memory_config(
                shape=(32, hidden_size // num_cores),
                core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
        else:
            return ttnn.L1_MEMORY_CONFIG

    elif tensor_role == "ffn_intermediate":
        # 큰 중간 결과 → DRAM
        return ttnn.DRAM_MEMORY_CONFIG

    elif tensor_role == "residual":
        # 레이어 간 전달 → L1 샤딩
        return ttnn.create_sharded_memory_config(...)

    else:
        # 기본: L1
        return ttnn.L1_MEMORY_CONFIG
```

---

## 17. 성능 벤치마크 (TT-Metal 공식)

### 17.1 LLM 성능 (2025년 11월 기준)

| Model | Device | Batch | t/s/u | t/s | TTFT |
|-------|--------|-------|-------|-----|------|
| **Llama 3.1 8B** | p150 (Blackhole) | 32 | **33.1** | **1059.2** | 57ms |
| Llama 3.1 8B | p100 (Blackhole) | 32 | 29.0 | 928.0 | 61ms |
| Llama 3.1 8B | n150 (Wormhole) | 32 | 26.0 | 832.0 | 104ms |
| Llama 3.2 3B | n150 | 32 | 46.6 | 1491.2 | 52ms |
| Llama 3.2 1B | n150 | 32 | 80.5 | 2576.0 | 23ms |
| Llama 3.1 70B | QuietBox 8x WH | 32 | 15.9 | 508.8 | 159ms |

### 17.2 BitNet-TT 목표 성능

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Decode (t/s/u) | ~7.8 | 30+ | 4x |
| Prefill (t/s) | ~4.6 | 15+ | 3x |
| TTFT | ~217ms | <100ms | 2x |
| Memory/token | 8.5GB | 64MB | 135x |

### 17.3 최적화 영향 추정

| 최적화 | 속도 향상 | 메모리 절감 | 복잡도 |
|--------|----------|------------|--------|
| Trace Capture | 2-3x | - | 중 |
| Paged Attention | 1.5x | 135x | 상 |
| Optimized Decode Ops | 1.5x | - | 중 |
| RoPE Pre-upload | 1.2x | - | 하 |
| Memory Sharding | 1.1x | 2x | 중 |
| Multi-CQ | 1.2x | - | 상 |
| **총 누적** | **5-8x** | **270x** | |

---

## 18. BitNet INT8×INT2 커널 참조 (공식 GPU 구현)

### 18.1 양자화 전략

```python
# 가중치: 삼진 {-1, 0, +1} → INT2 (4개/바이트)
def pack_ternary_weights(weight):
    scale = weight.abs().mean()
    quant = (weight / scale).round().clamp(-1, 1)  # {-1, 0, +1}
    # 인코딩: -1→0, 0→1, +1→2
    encoded = (quant + 1).to(torch.uint8)
    # 4개씩 묶어 1바이트로 패킹
    packed = (encoded[..., 0::4] |
              (encoded[..., 1::4] << 2) |
              (encoded[..., 2::4] << 4) |
              (encoded[..., 3::4] << 6))
    return packed, scale

# 활성화: INT8 (per-token absmax)
def quant_activation(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    x_int8 = (x * scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale
```

### 18.2 CUDA 커널 구조

```cpp
// bitnet_kernels.cu - 특화된 GEMM 커널
template<int M, int N, int K, int BLOCKS, int WARP_Y, int WARP_X>
__global__ void ladder_int8xint2_kernel(
    int8_t* __restrict__ activations,  // [M, K] INT8
    int8_t* __restrict__ weights,       // [N, K/4] packed INT2
    __nv_bfloat16* __restrict__ output, // [M, N] BF16
    __nv_bfloat16* __restrict__ act_scale,   // Per-token
    __nv_bfloat16* __restrict__ weight_scale // Per-layer
);

// 특화 커널 디스패치 (크기별)
void bitlinear_int8xint2(int8_t* a, int8_t* w, bf16* out, ...) {
    if (M == 1 && N == 3840 && K == 2560) {
        // QKV 프로젝션 (2560 → 3840)
        ladder_int8xint2_kernel<1, 3840, 2560, 3, 8, 16><<<...>>>;
    }
    else if (M == 1 && N == 2560 && K == 2560) {
        // O 프로젝션 (2560 → 2560)
        ladder_int8xint2_kernel<1, 2560, 2560, 1, 8, 16><<<...>>>;
    }
    // ... 다른 크기들
}
```

### 18.3 TT-NN에서의 BitLinear 구현 (BF16 시뮬레이션)

현재 TT-NN은 INT8×INT2 커널이 없으므로 BF16으로 시뮬레이션:

```python
class TTBitLinear:
    def __init__(self, in_features, out_features, device):
        # 가중치: 삼진 양자화 후 BF16으로 저장
        weight_quant = ternary_quantize(weight)  # {-s, 0, +s}
        self.weight = ttnn.from_torch(
            weight_quant.T,  # 미리 전치!
            device=device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT
        )

    def forward(self, x):
        # 활성화: per-token absmax 양자화 (BF16에서 시뮬레이션)
        scale = 127.0 / ttnn.max(ttnn.abs(x), dim=-1, keepdim=True)
        x_quant = ttnn.round(x * scale)
        x_quant = ttnn.clip(x_quant, -128, 127) / scale

        # 선형 연산 (가중치 이미 전치됨)
        return ttnn.linear(x_quant, self.weight, transpose_b=False)
```

### 18.4 향후 최적화 가능성

TT-NN에서 INT 커널 지원 시:
1. **INT8 활성화**: `ttnn.experimental.int8_matmul` (있다면)
2. **패킹된 가중치**: 2-bit 가중치 직접 지원
3. **융합 양자화**: absmax + linear 융합

---

## 19. BitNet-TT 구현 체크리스트

### Phase 1: 기본 최적화 (현재)
- [x] 가중치 미리 전치 (`transpose_b` 불필요)
- [x] RoPE cos/sin 디바이스 캐시
- [x] KV 캐시 사전 할당
- [x] Paged Attention 구현
- [ ] Trace Capture 적용

### Phase 2: 고급 최적화
- [ ] `nlp_create_qkv_heads_decode` 사용
- [ ] `rotary_embedding_llama` 사용
- [ ] `nlp_concat_heads_decode` 사용
- [ ] Multi-CQ 도입

### Phase 3: 메모리 최적화
- [ ] L1 샤딩 설정
- [ ] BFP8 가중치 (정확도 검증 후)
- [ ] 중간 결과 DRAM/L1 분리

### Phase 4: 검증
- [ ] HuggingFace 대비 Logits 일치 (correlation > 0.99)
- [ ] Top-1 토큰 일치율 100%
- [ ] 30+ t/s/u 달성

---

## 20. TT-Metal 공식 Attention 구현 상세 (tt_transformers/attention.py)

### 20.1 핵심 구조

공식 Llama 구현에서의 Attention 클래스 구조:

```python
class Attention(LightweightModule):
    def __init__(self, ...):
        # 1. Fused QKV 가중치 (모든 Q, K, V를 하나로)
        qkv_list = []
        for i in range(num_devices):
            wq = state_dict[f"layer.{l}.wq.weight"].chunk(num_devices)[i].T
            wk = state_dict[f"layer.{l}.wk.weight"].chunk(num_devices)[i].T
            wv = state_dict[f"layer.{l}.wv.weight"].chunk(num_devices)[i].T
            qkv_list.append(torch.cat([wq, wk, wv], dim=-1))

        # 디바이스에 샤딩
        self.wqkv = ttnn.as_tensor(
            torch.cat(qkv_list, dim=-1),
            dtype=self.wqkv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=wqkv_mem_config,  # DRAM 샤딩
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2)),
        )

        # 2. KV 캐시 (Paged Attention)
        if paged_attention_config:
            cache_shape = (max_num_blocks, n_kv_heads, block_size, head_dim)
        else:
            cache_shape = (batch_size, n_kv_heads, max_seq_len, head_dim)

        self.layer_past = [
            ttnn.as_tensor(torch.zeros(cache_shape), device=mesh_device, ...)
            for _ in [cache_k, cache_v]
        ]
```

### 20.2 Decode Forward 상세

```python
def forward_decode(self, x, current_pos, rot_mats, page_table, kv_cache):
    # 1. QKV Matmul (DRAM 샤딩)
    xqkv_fused_sharded = ttnn.linear(
        x, self.wqkv,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        program_config=self.model_config["XQKV_DECODE_PROGCFG"],
        compute_kernel_config=self.li_qkv_decode_compute_kernel_cfg,
        dtype=ttnn.bfloat16,
    )

    # 2. All Reduce (멀티 디바이스)
    xqkv_fused = tt_all_reduce(
        xqkv_fused_sharded, mesh_device, cluster_axis=1,
        memory_config=self.model_config["QKV_OUT_GATHERED_MEMCFG"],
    )

    # 3. 헤드 분리 (최적화된 연산)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv_fused,
        num_heads=self.n_local_heads,
        num_kv_heads=self.n_local_kv_heads,
        memory_config=self.model_config["CREATE_QKV_DECODE_SHARD"],
    )

    # 4. RoPE (융합 연산)
    q = ttnn.experimental.rotary_embedding_llama(
        q, rot_mats[0], rot_mats[1],
        self.transformation_mats["decode"],
        is_decode_mode=True
    )
    k = ttnn.experimental.rotary_embedding_llama(
        k, rot_mats[0], rot_mats[1],
        self.transformation_mats["decode"],
        is_decode_mode=True
    )

    # 5. KV 캐시 업데이트 (페이지드, 제자리)
    ttnn.experimental.paged_update_cache(
        keys, k, update_idxs_tensor=current_pos, page_table=page_table
    )
    ttnn.experimental.paged_update_cache(
        values, v, update_idxs_tensor=current_pos, page_table=page_table
    )

    # 6. SDPA (디코드 최적화 또는 페이지드)
    if page_table:
        attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q, keys, values,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            scale=self.scale,
            program_config=self.model_config["SDPA_DECODE_PROGCFG"],
        )
    else:
        attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
            q, keys, values,
            cur_pos_tensor=current_pos,
            scale=self.scale,
            program_config=self.model_config["SDPA_DECODE_PROGCFG"],
        )

    # 7. 헤드 결합 (최적화된 연산)
    attn_output = ttnn.experimental.nlp_concat_heads_decode(
        attn_output, num_heads=self.n_local_heads
    )

    # 8. Output projection + All Reduce
    if self.use_fused_all_gather_matmul:
        _, dense_out = ttnn.experimental.all_gather_matmul_async(
            attn_output, self.wo, dim=3, ...
        )
    else:
        dense_out = ttnn.linear(attn_output, self.wo, ...)
        dense_out = tt_all_reduce(dense_out, ...)

    return dense_out
```

### 20.3 Prefill Forward 상세

```python
def forward_prefill(self, x, rot_mats, user_id, page_table, kv_cache):
    seq_len = x.shape[-2]

    # 1. 긴 시퀀스 리쉐이프 (메모리 제한)
    if seq_len > self.MAX_QKV_MM_SEQ_LEN:
        x = ttnn.reshape(x, [1, seq_len // MAX_QKV_MM_SEQ_LEN, MAX_QKV_MM_SEQ_LEN, -1])

    # 2. QKV Matmul (DRAM)
    xqkv = ttnn.linear(
        x, self.wqkv,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=self.model_config["XQKV_PREFILL_PROGCFG"](seq_len),
    )

    # 3. All Reduce
    xqkv = tt_all_reduce(xqkv, ...)

    # 4. 헤드 분리 (프리필용)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        xqkv,
        num_heads=self.n_local_heads,
        num_kv_heads=self.n_local_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # 5. RoPE (프리필용)
    q = ttnn.experimental.rotary_embedding_llama(
        q, rot_mats[0], rot_mats[1],
        self.transformation_mats["prefill"],
        is_decode_mode=False,
    )
    k = ttnn.experimental.rotary_embedding_llama(k, ...)

    # 6. KV 캐시 채우기
    if page_table:
        ttnn.experimental.paged_fill_cache(keys, k, page_table, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(values, v, page_table, batch_idx=user_id)
    else:
        ttnn.fill_cache(keys, k, user_id)
        ttnn.fill_cache(values, v, user_id)

    # 7. SDPA (프리필)
    attn_output = ttnn.transformer.scaled_dot_product_attention(
        q, k, v,
        is_causal=True,
        scale=self.scale,
        program_config=self.model_config["SDPA_PROGCFG"](seq_len),
    )

    # 8. 헤드 결합
    attn_output = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # 9. Output projection
    output = ttnn.linear(attn_output, self.wo, ...)

    # 10. All Reduce (멀티 디바이스)
    output = tt_all_reduce(output, ...)

    return output
```

### 20.4 핵심 최적화 연산 정리

| 연산 | Decode | Prefill | 설명 |
|------|--------|---------|------|
| QKV 헤드 생성 | `nlp_create_qkv_heads_decode` | `nlp_create_qkv_heads` | 융합 Q/K/V 분리 + 헤드 리쉐이프 |
| RoPE | `rotary_embedding_llama` | `rotary_embedding_llama` | cos/sin 캐시 + transformation_mat |
| KV 캐시 | `paged_update_cache` | `paged_fill_cache` / `fill_cache` | 제자리 업데이트 |
| SDPA | `paged_scaled_dot_product_attention_decode` | `scaled_dot_product_attention` | 디코드용 최적화 |
| 헤드 결합 | `nlp_concat_heads_decode` | `nlp_concat_heads` | 융합 헤드 결합 |

---

## 21. TT-NN Matmul 설정 상세 (llms.md 기반)

### 21.1 Matmul 종류

1. **Matmul 2D** (`MatmulMultiCoreReuseMultiCastProgramConfig`)
   - M, N 차원으로 2D 병렬화
   - 프리필 모드에 적합 (M, N >= 256)
   - DRAM interleaved 입출력

2. **DRAM-Sharded Matmul**
   - 디코드 모드에 최적 (작은 활성화, 큰 가중치)
   - 가중치를 DRAM 뱅크에 샤딩
   - ~240 GB/s 대역폭 (vs interleaved ~190 GB/s)

3. **Matmul 1D** (`MatmulMultiCoreReuseMultiCast1DProgramConfig`)
   - N 차원으로만 병렬화
   - 활성화/출력 L1 width-sharded
   - 가중치 DRAM interleaved

### 21.2 Program Config 예시

```python
# Matmul 2D (Prefill)
PREFILL_PROGCFG = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),  # 코어 그리드
    in0_block_w=1,      # K 타일 블록 (높을수록 좋음, L1 제한)
    out_subblock_h=1,   # M 타일 서브블록
    out_subblock_w=1,   # N 타일 서브블록 (h*w <= DST 크기)
    per_core_M=seq_len // 32 // 8,  # 코어당 M 타일
    per_core_N=hidden_dim // 32 // 8,  # 코어당 N 타일
    transpose_mcast=False,
    fused_activation=None,
    fuse_batch=False,  # DRAM 입력 시 False
)

# DRAM-Sharded (Decode)
DECODE_PROGCFG = dram_matmul_config(
    m=batch_size,
    k=hidden_dim,
    n=output_dim,
    num_cores=core_grid.num_cores,
)
```

### 21.3 Memory Config 예시

```python
# DRAM-Sharded 가중치 메모리 설정
weights_mem_config = create_dram_sharded_mem_config(k=hidden_dim, n=output_dim)

# L1 Width-Sharded 활성화 메모리 설정
input_memcfg = ttnn.create_sharded_memory_config(
    shape=(batch_size, hidden_dim // core_grid.num_cores),
    core_grid=core_grid,
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
```

### 21.4 Compute Kernel Config

```python
# HiFi2 (BFP8 가중치용, 권장)
compute_kernel_hifi2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,  # 2x 빠름 vs HiFi4
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# HiFi4 (BF16 가중치용, 정확도 중요 시)
compute_kernel_hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,  # 더 정확
    packer_l1_acc=True,
)

# LoFi (BFP4 가중치용, 3.6x 빠름)
compute_kernel_lofi = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
```

---

## 22. 공식 BitNet GPU 커널 상세 (~/BitNet/gpu/)

### 22.1 W2A8 GEMV 최적화

| 최적화 | 설명 |
|--------|------|
| **Weight Permutation** | 16×32 블록으로 메모리 접근 패턴 최적화 |
| **Fast Decoding** | 16개 2-bit 값을 특수 인터리빙으로 32-bit에 패킹 |
| **dp4a Instruction** | 4개 INT8 값의 dot product 가속 |

### 22.2 인터리빙 패턴

```
[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
```

이 패턴으로 4개 값을 한 번에 INT8로 추출 가능.

### 22.3 커널 성능 (A100)

| Shape (N×K) | W2A8 (us) | BF16 (us) | Speedup |
|-------------|-----------|-----------|---------|
| 2560×2560 | 13.32 | 18.32 | 1.38x |
| 13824×2560 | 18.75 | 59.51 | **3.17x** |
| 2560×6912 | 14.49 | 37.78 | **2.61x** |

### 22.4 E2E 생성 성능

| Input | Output | BF16 (ms) | W2A8 (ms) | Speedup |
|-------|--------|-----------|-----------|---------|
| 64 | 64 | 683.23 | 221.08 | **3.09x** |
| 512 | 64 | 709.65 | 231.82 | **3.06x** |

---

## 23. BitNet-TT 최종 성능 목표

### 23.1 단계별 목표

| Phase | Target t/s/u | 핵심 최적화 |
|-------|--------------|-------------|
| 현재 | ~7.8 | 기본 구현 |
| Phase 1 | 15-20 | Trace Capture |
| Phase 2 | 25-30 | Optimized Decode Ops |
| Phase 3 | 30+ | Multi-CQ + Memory Sharding |

### 23.2 참조 성능 비교

| 구현 | Hardware | t/s/u | 비고 |
|------|----------|-------|------|
| bitnet.cpp | CPU (ARM) | ~50 | 공식 최적화 |
| BitNet GPU | A100 | ~45 (추정) | W2A8 커널 |
| Llama 3.1 8B | p150a | 33.1 | TT-Metal 공식 |
| **BitNet-TT Target** | **p150a** | **30+** | **목표** |

### 23.3 BitNet vs Llama 크기 비교

| Model | Params | Hidden | Layers | Heads | KV Heads |
|-------|--------|--------|--------|-------|----------|
| BitNet 2B-4T | 2B | 2560 | 30 | 20 | 5 |
| Llama 3.1 8B | 8B | 4096 | 32 | 32 | 8 |
| **Ratio** | **4x 작음** | **1.6x 작음** | 유사 | 유사 | 유사 |

BitNet이 4배 작으므로, 최적화 시 Llama 8B 대비 더 높은 t/s/u 달성 가능.

---

## 24. TT-NN LLM 최적화 핵심 패턴 (llms.md 기반)

### 24.1 Prefill vs Decode 메모리 전략

| 모드 | Activation 위치 | Weight 위치 | Bottleneck |
|------|----------------|-------------|------------|
| Prefill | DRAM interleaved | DRAM | Compute-bound |
| Decode | L1 sharded | DRAM sharded | DRAM-bandwidth-bound |

**핵심**: Prefill은 큰 seq_len으로 compute-bound, Decode는 batch 병렬화로 DRAM-bound.

### 24.2 Attention Decode 최적화 흐름

```python
# 1. QKV Projection (DRAM-sharded matmul)
xqkv_fused = ttnn.linear(x, wqkv,
    program_config=DRAM_SHARDED_PROGCFG,
    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

# 2. Split QKV heads (HEIGHT_SHARDED 필수!)
Q, K, V = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused,
    num_heads=n_q_heads,
    num_kv_heads=n_kv_heads,
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1
    )
)

# 3. RoPE (HEIGHT_SHARDED 입력 필요)
Q = ttnn.experimental.rotary_embedding_llama(Q, cos, sin, trans_mat)
K = ttnn.experimental.rotary_embedding_llama(K, cos, sin, trans_mat)

# 4. KV Cache Update (paged)
ttnn.experimental.paged_update_cache(keys, K,
    update_idxs_tensor=cur_pos, page_table=page_table)

# 5. SDPA Decode
attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    Q, keys, values,
    cur_pos_tensor=cur_pos,
    page_table_tensor=page_table,
    scale=scale)

# 6. Concat heads
output = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=n_q_heads)

# 7. Output projection
output = ttnn.linear(output, wo)
```

### 24.3 DRAM-Sharded Matmul 설정

Decode 모드의 모든 matmul에 사용 (DRAM bandwidth 최적화):

```python
# Weight memory config
weights_memory_config = create_dram_sharded_mem_config(k=K, n=N)

# Program config
pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
    in0_block_w=math.ceil(k / (tile_size * num_cores)),
    per_core_M=math.ceil(m / tile_size),
    per_core_N=math.ceil(n / (tile_size * num_cores)),
    fused_activation=None,
)
```

**성능**: Interleaved ~190 GB/s → DRAM-Sharded ~240 GB/s (**26% 향상**)

### 24.4 Matmul 2D (Prefill용)

```python
pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(cores_x, cores_y),
    in0_block_w=1,  # K 차원 분할
    out_subblock_h=1,
    out_subblock_w=1,
    per_core_M=math.ceil(M / 32 / cores_y),
    per_core_N=math.ceil(N / 32 / cores_x),
    transpose_mcast=False,
)
```

### 24.5 트레이싱 필수 조건

```python
# ✅ 트레이싱 가능: 정적 shape, cur_pos를 텐서로 전달
cur_pos_tensor = ttnn.from_torch(...)  # Device tensor
ttnn.experimental.paged_update_cache(..., update_idxs_tensor=cur_pos_tensor)

# ❌ 트레이싱 불가: 동적 shape, cur_pos를 리스트로 전달
ttnn.experimental.paged_update_cache(..., update_idxs=cur_pos_list)
```

### 24.6 LM Head 최적화

vocab_size가 크므로 weight 분할 필요:

```python
# Prefill: 마지막 토큰만 계산
if mode == "prefill":
    x = ttnn.slice(x, (0, 0, last_token, 0), (1, 1, last_token + 32, dim))

# Weight splitting (vocab_size=128256 → 여러 조각)
for weight_chunk, pc in zip(output_weights, program_configs):
    out = ttnn.linear(x, weight_chunk, program_config=pc)
    outputs.append(out)
output = ttnn.concat(outputs, dim=-1)
```

---

## 25. tt_transformers Attention 구현 상세

### 25.1 Decode Forward 핵심 코드

```python
# QKV projection + All-Reduce (multi-device)
xqkv_fused_sharded = ttnn.linear(x, self.wqkv,
    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    program_config=self.model_config["XQKV_DECODE_PROGCFG"])

xqkv_fused = tt_all_reduce(xqkv_fused_sharded, ...)

# Create QKV heads - HEIGHT_SHARDED 메모리 설정 중요!
Q, K, V = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused,
    num_heads=self.n_local_heads,
    num_kv_heads=self.n_local_kv_heads,
    memory_config=self.model_config["CREATE_QKV_DECODE_SHARD"]  # HEIGHT_SHARDED
)

# RoPE with transformation matrices
Q = ttnn.experimental.rotary_embedding_llama(
    Q, rot_mats[0], rot_mats[1],
    self.transformation_mats["decode"],
    is_decode_mode=True)
```

### 25.2 CREATE_QKV_DECODE_SHARD 설정

**BitNet-TT 현재 문제점**: L1_MEMORY_CONFIG 사용 → HEIGHT_SHARDED 필요

```python
# 올바른 설정 (tt_transformers 참조)
CREATE_QKV_DECODE_SHARD = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1
)

# 잘못된 설정 (BitNet-TT 현재)
memory_config=ttnn.L1_MEMORY_CONFIG  # ❌ rotary_embedding_llama 실패
```

### 25.3 Fused All-Gather Matmul

Output projection에서 AllGather와 Matmul을 융합:

```python
_, dense_out_sharded = ttnn.experimental.all_gather_matmul_async(
    attn_output_cat,
    self.wo,
    dim=3,
    all_gather_core_grid_offset=(0, 4),
    num_links=1,
    memory_config_mm=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    program_config=all_gather_matmul_progcfg,
)
```

---

## 26. 공식 모델 성능 벤치마크 (2025.11)

### 26.1 Blackhole p150 성능

| Model | Batch | ttft (ms) | t/s/u | t/s |
|-------|-------|-----------|-------|-----|
| Llama 3.1 8B | 32 | 57 | **33.1** | 1059.2 |
| Llama 3.1 70B (TP=4) | 32 | 188 | 14.9 | 476.5 |
| Whisper distil-large-v3 | 1 | 113 | 101.5 | 101.5 |

### 26.2 Wormhole n150 성능

| Model | Batch | ttft (ms) | t/s/u | t/s |
|-------|-------|-----------|-------|-----|
| Llama 3.1 8B | 32 | 104 | 26.0 | 832.0 |
| Llama 3.2 1B | 32 | 23 | 80.5 | 2576.0 |
| Llama 3.2 3B | 32 | 52 | 46.6 | 1491.2 |
| Mistral 7B | 32 | 99 | 28.7 | 918.4 |

### 26.3 BitNet-TT 목표 재설정

**Llama 3.2 3B (46.6 t/s/u)를 참조 모델로**:
- BitNet 2B-4T는 Llama 3.2 3B보다 작음
- 목표: **50+ t/s/u** (최적화 완료 시)

---

## 27. 메모리 관리 Best Practices

### 27.1 텐서 해제 패턴

```python
def forward_decode(self, x, ...):
    xqkv = ttnn.linear(x, self.wqkv, ...)
    ttnn.deallocate(x)  # 즉시 해제

    Q, K, V = ttnn.experimental.nlp_create_qkv_heads_decode(xqkv, ...)
    ttnn.deallocate(xqkv)

    Q_rot = ttnn.experimental.rotary_embedding_llama(Q, ...)
    ttnn.deallocate(Q)  # pre-rot 해제

    # ... 계속
```

### 27.2 L1 vs DRAM 선택 기준

| 조건 | 메모리 | 이유 |
|------|--------|------|
| seq_len > 2048 | DRAM | L1에 안 맞음 |
| Decode activation | L1 sharded | 빠른 접근 |
| Weights (항상) | DRAM sharded | 크기가 큼 |
| KV Cache | DRAM | 긴 시퀀스 지원 |

### 27.3 Sharding 불일치 방지

```python
# ❌ 피해야 할 패턴: 불필요한 변환
x = ttnn.sharded_to_interleaved(x_sharded, ...)
x = ttnn.interleaved_to_sharded(x, ...)

# ✅ 권장: 연속 op에서 동일한 sharding 유지
output = ttnn.linear(x_sharded, w,
    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
```

---

## 28. BitNet-TT 수정 우선순위

### 28.1 즉시 수정 필요

1. **HEIGHT_SHARDED 메모리 설정**
   - 위치: `attention.py:844-849`
   - 변경: `L1_MEMORY_CONFIG` → `HEIGHT_SHARDED`

2. **Pre-allocated KV Cache GQA 확장**
   - 현재: `[batch, num_kv_heads, max_seq, head_dim]`
   - 필요: `[batch, num_heads, max_seq, head_dim]`

### 28.2 성능 최적화 (Phase 1)

1. **트레이싱 활성화**
   - `cur_pos`를 텐서로 변환
   - `_capture_decode_trace()` 활성화

2. **DRAM-Sharded Matmul**
   - 모든 decode matmul에 적용
   - weight 메모리 설정 변경

### 28.3 고급 최적화 (Phase 2-3)

1. **Fused Projections**
   - wqkv fused (Q+K+V projection 통합)
   - w13 fused (gate+up projection 통합)

2. **Multi-CQ 파이프라인**
   - 입력 전송과 계산 오버랩

3. **BFP8 Weight Quantization**
   - 메모리 50% 절약, 미미한 정확도 손실

---

## 29. tt_transformers 핵심 API 상세

### 29.1 nlp_create_qkv_heads_decode (Decode용 QKV 분리)

**목적**: Fused QKV 텐서를 Q, K, V heads로 분리 (decode용)

```python
q_heads_1BQD, k_heads_1BKD, v_heads_1BKD = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused,  # [1, 1, batch, qkv_dim]
    num_heads=n_local_heads,       # Q heads 수
    num_kv_heads=n_local_kv_heads, # KV heads 수 (GQA)
    memory_config=CREATE_QKV_DECODE_SHARD,  # ⚠️ HEIGHT_SHARDED 필수!
)
```

**출력 형식**: `1BKD` = `[1, batch, heads, head_dim]`

**⚠️ 중요**: `rotary_embedding_llama`가 HEIGHT_SHARDED 입력을 요구하므로 `memory_config`는 반드시 HEIGHT_SHARDED이어야 함.

### 29.2 nlp_create_qkv_heads (Prefill용 QKV 분리)

```python
q_heads_1QSD, k_heads_1KSD, v_heads_1VSD = ttnn.experimental.nlp_create_qkv_heads(
    xqkv_fused,  # [1, 1, seq_len, qkv_dim]
    num_heads=n_local_heads,
    num_kv_heads=n_local_kv_heads,
    transpose_k_heads=False,  # K 전치 여부
    memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Prefill은 DRAM 사용
)
```

**출력 형식**: `1QSD` = `[1, num_heads, seq_len, head_dim]`

### 29.3 rotary_embedding_llama (RoPE 적용)

```python
q_rotated = ttnn.experimental.rotary_embedding_llama(
    q_heads_pre_rot,      # HEIGHT_SHARDED 텐서
    rot_cos,              # cos 행렬 [1, 1, max_seq, head_dim]
    rot_sin,              # sin 행렬 [1, 1, max_seq, head_dim]
    transformation_mat,   # decode 또는 prefill용
    is_decode_mode=True,  # True: decode, False: prefill
)
```

**⚠️ 핵심 요구사항**:
- `TT_FATAL: Sharded inputs for RoPE must be HEIGHT_SHARDED`
- 입력 텐서가 HEIGHT_SHARDED가 아니면 에러 발생

### 29.4 paged_update_cache (Decode KV 캐시 업데이트)

```python
ttnn.experimental.paged_update_cache(
    cache_tensor,         # [max_blocks, kv_heads, block_size, head_dim] 또는 [batch, kv_heads, max_seq, head_dim]
    new_kv_tensor,        # [1, batch, kv_heads, head_dim] - 1BKD 형식!
    update_idxs_tensor=current_pos,  # 위치 텐서
    page_table=page_table,           # 페이지 테이블 (옵션)
)
```

**⚠️ 핵심 요구사항**:
- `TT_FATAL: Expect input_tensor to be sharded`
- `new_kv_tensor`가 반드시 **HEIGHT_SHARDED**이어야 함
- Interleaved 텐서를 전달하면 에러 발생

**1BKD 형식 변환**:
```python
# BKSD [batch, heads, seq=1, dim] → 1BKD [1, batch, heads, dim]
k_heads_1BKD = ttnn.permute(k_heads_BKSD, (2, 0, 1, 3))  # 이것만으론 부족!
k_heads_1BKD = ttnn.to_memory_config(k_heads_1BKD, HEIGHT_SHARDED_CONFIG)  # sharding 필수
```

### 29.5 paged_fill_cache (Prefill KV 캐시 채우기)

```python
ttnn.experimental.paged_fill_cache(
    cache_tensor,         # [max_blocks, kv_heads, block_size, head_dim]
    kv_tensor,            # [1, kv_heads, seq_len, head_dim]
    page_table,           # 페이지 테이블
    batch_idx=user_id,    # 사용자 ID
)
```

### 29.6 scaled_dot_product_attention_decode

```python
attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
    q_heads,              # [1, batch, n_heads, head_dim]
    keys_cache,           # [batch, kv_heads, max_seq, head_dim]
    values_cache,         # [batch, kv_heads, max_seq, head_dim]
    cur_pos_tensor=current_pos,
    scale=1.0 / math.sqrt(head_dim),
    program_config=sdpa_decode_progcfg,
    compute_kernel_config=compute_cfg,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

### 29.7 nlp_concat_heads_decode

```python
attn_output_concat = ttnn.experimental.nlp_concat_heads_decode(
    attn_output_1G4D,     # [1, 1, batch, n_heads * head_dim]
    num_heads=n_local_heads,
)
```

---

## 30. HEIGHT_SHARDED 메모리 설정

### 30.1 CREATE_QKV_DECODE_SHARD 정의 (tt_transformers 방식)

```python
# Blackhole용
CREATE_QKV_DECODE_SHARD = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, head_dim),  # (32, 128) for head_dim=128
    core_grid=ttnn.CoreGrid(y=4, x=8),  # 32 cores
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

# Wormhole용
CREATE_QKV_DECODE_SHARD = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
```

### 30.2 Interleaved → Sharded 변환

```python
# ❌ 잘못된 방법: permute 후 바로 사용
k_heads = ttnn.permute(k_heads_pre, (2, 0, 1, 3))  # 여전히 interleaved

# ✅ 올바른 방법: memory config 변환 필요
k_heads_sharded = ttnn.to_memory_config(
    k_heads,
    memory_config=CREATE_QKV_DECODE_SHARD,  # HEIGHT_SHARDED로 변환
)
```

### 30.3 batch 크기에 따른 동적 설정

```python
def get_scores_memcfg(batch_size, n_heads, head_dim):
    return ttnn.create_sharded_memory_config(
        shape=(math.ceil(n_heads / 32) * 32, head_dim),
        core_grid=ttnn.CoreRangeSet({num_to_corerange(batch_size)}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
```

---

## 31. DRAM-Sharded Matmul 설정

### 31.1 DRAM Sharded 메모리 설정 생성

```python
def create_dram_sharded_mem_config(k_dim, n_dim):
    """DRAM에서 weight를 12개 bank에 분산"""
    # 12는 DRAM bank 수
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
        [k_dim, n_dim // 12],  # 각 bank당 shard 크기
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )
```

### 31.2 DRAM-Sharded Matmul Program Config

```python
def dram_matmul_config(m, k, n, num_cores):
    """DRAM-sharded weight와의 matmul 설정"""
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=k // 32 // num_cores,  # K를 core 수로 나눔
        per_core_M=m // 32,
        per_core_N=n // 32 // num_cores,
        fused_activation=None,
    )
```

### 31.3 사용 예시 (attention QKV)

```python
# Weight 메모리 설정
wqkv_mem_config = create_dram_sharded_mem_config(dim, qkv_size // num_devices)

# Weight 로드
self.wqkv = ttnn.as_tensor(
    qkv_weight,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=wqkv_mem_config,  # DRAM-sharded
)

# Matmul 실행
xqkv = ttnn.linear(
    x,  # [1, 1, batch, dim]
    self.wqkv,
    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,  # 출력은 L1 sharded
    program_config=dram_matmul_config(batch, dim, qkv_size, num_cores),
    compute_kernel_config=hifi2_config,
)
```

---

## 32. Trace Capture 패턴 (tt_transformers 방식)

### 32.1 기본 Trace 캡처

```python
class Generator:
    def __init__(self, ...):
        self.trace_id = None
        self.trace_inputs = None
        self.trace_output = None

    def _capture_decode_trace(self, input_tokens, current_pos, kv_cache):
        # 1. 먼저 한 번 실행하여 컴파일
        host_inputs = self.prepare_inputs(input_tokens, current_pos)
        device_inputs = copy_host_to_device(host_inputs, self.device)
        output = self.model.forward_decode(*device_inputs, kv_cache=kv_cache)
        ttnn.synchronize_device(self.device)

        # 2. Trace 캡처 시작
        device_inputs = copy_host_to_device(host_inputs, self.device)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)

        output = self.model.forward_decode(*device_inputs, kv_cache=kv_cache)

        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

        return trace_id, output, device_inputs
```

### 32.2 Trace 실행

```python
def _execute_decode_trace(self, input_tokens, current_pos):
    # Host에서 새 입력 준비
    host_inputs = self.prepare_inputs(input_tokens, current_pos)

    # Device 텐서에 새 값 복사 (메모리 주소 유지!)
    copy_host_to_device(
        host_inputs,
        device_tensors=self.trace_inputs,  # 기존 디바이스 텐서 재사용
        mesh_device=self.device,
    )

    # Trace 실행 (blocking=False로 비동기)
    ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)

    return self.trace_output  # 미리 캡처된 출력 텐서
```

### 32.3 copy_host_to_device 패턴

```python
def copy_host_to_device(host_tensors, device_tensors=None, mesh_device=None):
    """Host 텐서를 device로 복사. device_tensors 제공 시 기존 메모리에 덮어씀"""
    if device_tensors is None:
        # 새 device 텐서 생성
        return tuple(
            ttnn.from_torch(t, device=mesh_device, ...)
            for t in host_tensors
        )
    else:
        # 기존 device 텐서에 복사 (trace용 - 메모리 주소 유지)
        for host_t, device_t in zip(host_tensors, device_tensors):
            ttnn.copy_host_to_device_tensor(host_t, device_t)
        return device_tensors
```

### 32.4 Prefill Trace 지원

```python
def _easy_trace_prefill(self, prefill_ids, page_table, kv_cache, prefill_seq_len, model_id):
    trace_key = f"{prefill_seq_len}_{model_id}"

    if self.trace_id_prefill[trace_key] is None:
        # 첫 호출: trace 캡처
        trace_id, output, *inputs = self._capture_trace_prefill(
            prefill_ids, page_table, kv_cache, model_id
        )
        self.trace_id_prefill[trace_key] = trace_id
        self.trace_inputs_prefill[trace_key] = inputs
        self.trace_output_prefill[trace_key] = output

    # Trace 실행
    return self._prefill_forward_trace(
        self.trace_id_prefill[trace_key],
        self.trace_inputs_prefill[trace_key],
        self.trace_output_prefill[trace_key],
        prefill_ids,
        page_table,
        model_id,
    )
```

---

## 33. Compute Kernel Configuration

### 33.1 Math Fidelity 설정

```python
# HiFi4: 최고 정확도 (BF16)
compute_kernel_hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# HiFi2: 빠른 연산 (BFP8)
compute_kernel_hifi2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)

# LoFi: 최고 속도 (BFP4)
compute_kernel_lofi = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)
```

### 33.2 모델별 권장 설정

| 연산 | 권장 Fidelity | dtype | 이유 |
|------|---------------|-------|------|
| QKV Decode | HiFi2 | BFP8 | DRAM-bound, 정확도 덜 민감 |
| SDPA Decode | HiFi2 | BFP8 | 속도 최적화 |
| QKV Prefill | HiFi4 | BF16 | 정확도 중요 |
| SDPA Prefill | HiFi4 | BF16 | 정확도 중요 |
| MLP FF1/FF3 | LoFi | BFP4 | 속도 최우선 |
| MLP FF2 | HiFi2 | BFP8 | 균형 |

---

## 34. BitNet-TT 최적화 적용 가이드

### 34.1 현재 문제점 요약

| 문제 | 파일:라인 | 원인 | 해결책 |
|------|-----------|------|--------|
| rotary_embedding 실패 | attention.py | nlp_create_qkv_heads_decode가 L1_MEMORY_CONFIG 사용 | HEIGHT_SHARDED 메모리 설정 |
| paged_update_cache 실패 | attention.py | interleaved 입력 전달 | to_memory_config로 sharded 변환 |
| Trace 비활성화 | generator.py | 현재 구현 미완성 | tt_transformers 패턴 적용 |

### 34.2 수정 순서

**1단계: HEIGHT_SHARDED 설정 추가**

```python
# config.py에 추가
def get_create_qkv_decode_shard(head_dim=128, batch_size=32):
    return ttnn.create_sharded_memory_config(
        shape=(32, head_dim),  # TILE_SIZE, head_dim
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
```

**2단계: nlp_create_qkv_heads_decode 수정**

```python
# attention.py의 _forward_decode_optimized 수정
Q, K, V = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv,
    num_heads=self.num_heads,
    num_kv_heads=self.num_kv_heads,
    memory_config=self.create_qkv_decode_shard,  # ✅ HEIGHT_SHARDED
)
```

**3단계: paged_update_cache 입력 변환**

```python
# K, V를 sharded로 변환 후 캐시 업데이트
k_sharded = ttnn.to_memory_config(k_heads, self.create_qkv_decode_shard)
ttnn.experimental.paged_update_cache(
    self.key_cache,
    k_sharded,
    update_idxs_tensor=current_pos,
    page_table=page_table,
)
```

**4단계: Trace 캡처 활성화**

```python
# generator.py
def _generate_tokens(self, ...):
    if self.enable_trace and self._trace_id is None:
        # 첫 토큰: trace 캡처
        self._trace_id, self._trace_output, self._trace_inputs = \
            self._capture_decode_trace(token, pos, kv_cache)

    if self._trace_id:
        # trace 실행
        logits = self._execute_decode_trace(token, pos)
    else:
        # 일반 실행
        logits = self.decode_forward(token, pos, kv_cache)
```

### 34.3 검증 순서

1. `python main.py --test` - 기본 동작 확인
2. `python main.py --full` - 전체 모델 추론 (optimized path 테스트)
3. `python main.py --chat` - 연속 생성 (trace 동작 확인)

### 34.4 예상 성능 향상

| 최적화 | 현재 | 적용 후 | 향상 |
|--------|------|---------|------|
| HEIGHT_SHARDED | 8.68 t/s | 9-10 t/s | ~15% |
| + Trace | - | 18-26 t/s | 2-3x |
| + DRAM-Sharded | - | 25-35 t/s | +20% |
| 최종 목표 | - | 50+ t/s | 5-6x |

---

## 35. ttnn 핵심 함수 빠른 참조

### 35.1 메모리 설정 함수

```python
# Interleaved (기본)
ttnn.DRAM_MEMORY_CONFIG
ttnn.L1_MEMORY_CONFIG

# Sharded
ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

# 커스텀 sharded
ttnn.create_sharded_memory_config(
    shape=(height, width),
    core_grid=ttnn.CoreGrid(y=rows, x=cols),
    strategy=ttnn.ShardStrategy.HEIGHT,  # or WIDTH, BLOCK
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
```

### 35.2 메모리 변환

```python
# Layout 변환
ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)

# Memory config 변환
ttnn.to_memory_config(tensor, new_memory_config)

# Sharding 변환
ttnn.interleaved_to_sharded(tensor, sharded_memory_config)
ttnn.sharded_to_interleaved(tensor, interleaved_memory_config)
```

### 35.3 Trace 관련

```python
# 캡처
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
# ... 연산들 ...
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# 실행
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)

# 해제
ttnn.release_trace(device, trace_id)
```

### 35.4 텐서 관리

```python
# 해제 (L1 메모리 확보)
ttnn.deallocate(tensor)

# 동기화
ttnn.synchronize_device(device)

# 복사
ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)
```

---

## 36. 최적화 시도 기록 (2025-12-10)

### 36.1 시도 결과 요약

| 시도 | 커밋 | 결과 | 속도 |
|------|------|------|------|
| Alpha baseline | `cd995ba` | ✅ 성공 | **8.68 t/s** |
| HEIGHT_SHARDED (batch=1) | `105f828` | ❌ 실패 | - |
| L1_HEIGHT_SHARDED_MEMORY_CONFIG | `de8494a` | ❌ 실패 | - |
| to_memory_config() 변환 | `3776f79` | ❌ 실패 | - |
| 단일 코어 CoreGrid(1,1) | `4302980` | ❌ 실패 | - |
| batch_size=32 HEIGHT_SHARDED | `7c4f765`~`72e5044` | ❌ 실패 | - |
| batch_size=32 _forward_simple | `fd7a1d2` | ✅ 작동 | **0.80 t/s** (10x 느림) |
| **Alpha로 롤백** | `82f238c` | ✅ 현재 | **8.68 t/s** |

### 36.2 HEIGHT_SHARDED 실패 원인 분석

**근본 원인**: tt_transformers의 `rotary_embedding_llama`는 특정 shard geometry를 요구함

```
BitNet: num_heads=20, batch_size=32
→ Q tensor shape: [1, 32, 20, 128]
→ Shard height: 1 × 32 × 20 = 640

tt_transformers 기대값 (Llama):
→ CoreGrid(y=4, x=8) = 32 cores
→ Shard height per core: 32 (TILE_SIZE)
→ Total expected: 32 × 32 = 1024

Mismatch: 640 ≠ 1024
```

**결론**: BitNet의 `num_heads=20`이 32코어 그리드와 호환되지 않음. Llama는 `num_heads=32, 64, 128...` (2의 거듭제곱)을 사용하여 문제없음.

### 36.3 시도하지 않은 최적화 목록

| 최적화 | 섹션 | HEIGHT_SHARDED 의존 | 비고 |
|--------|------|---------------------|------|
| DRAM-Sharded Matmul | 31 | ❌ 독립적 | ~26% 향상 예상 |
| Compute Kernel Config | 33 | ❌ 독립적 | HiFi2/LoFi로 2-3x |
| LM Head 최적화 | 24.6 | ❌ 독립적 | 마지막 토큰만 계산 |
| Fused Projections | 28.3 | ❌ 독립적 | QKV, gate+up 통합 |
| Multi-CQ 파이프라인 | 28.3 | ⚠️ Trace 필요 | 보류 |

### 36.4 다음 시도 계획

1. **DRAM-Sharded Matmul**: decode 시 weight를 DRAM 12개 bank에 분산
2. **Compute Kernel Config**: HiFi2 또는 LoFi 적용으로 matmul 가속
3. **LM Head 최적화**: prefill 시 마지막 토큰만 계산
4. **Fused QKV**: Q, K, V projection을 단일 matmul로 통합
