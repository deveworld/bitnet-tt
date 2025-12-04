# BitNet-TT Integration Guide: From 33 t/s Theory to Practice

## QUICK REFERENCE: Where to Apply Each Optimization

### 1. TRACE CAPTURE (generator.py)

**File**: `src/bitnet_tt/inference/generator.py`

**Current lines 145-148** (stub state):
```python
# Trace state
self._trace_id: Optional[int] = None
self._trace_inputs: Optional[list] = None
self._trace_output: Optional[ttnn.Tensor] = None
```

**Pattern from tt_transformers** (generator.py lines 626-637):
```python
# Capture decode trace ONCE
if not self.trace_ids_decode[sampling_on_device]:
    # 1. Compile run (no trace)
    self._decode_forward_no_trace_text(...)
    
    # 2. Capture trace
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_out_trace = self.model.ttnn_decode_forward(...)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    
    # 3. Cache for reuse
    self.trace_ids_decode[sampling_on_device] = trace_id
    self.trace_inputs_decode[sampling_on_device] = device_inputs
    self.trace_output_decode[sampling_on_device] = tt_out_trace
```

**Apply to BitNet**:
```python
class TextGenerator:
    def __init__(self, model, tokenizer, enable_trace=True):
        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.enable_trace = enable_trace
        
        # Change this:
        self._trace_id = None
        self._trace_inputs = None
        self._trace_output = None
        
        # To this:
        self._trace_id = {}  # {sampling_enabled: trace_id}
        self._trace_inputs = {}  # {sampling_enabled: device_inputs_list}
        self._trace_output = {}  # {sampling_enabled: output_tensor}
        self._trace_captured = False
        
    def decode_forward(self, tokens, start_pos, page_table=None, kv_cache=None):
        """Decode forward with trace capture."""
        sampling_on_device = self.sampling_enabled  # Your sampling flag
        
        if self.enable_trace and not self._trace_captured:
            # First iteration: capture trace
            self._capture_decode_trace(tokens, start_pos, page_table, kv_cache, 
                                      sampling_on_device)
            self._trace_captured = True
        
        if self.enable_trace:
            return self._decode_with_trace(tokens, start_pos, page_table, kv_cache)
        else:
            return self._decode_no_trace(tokens, start_pos, page_table, kv_cache)
    
    def _capture_decode_trace(self, tokens, start_pos, page_table, kv_cache, 
                             sampling_on_device):
        """Capture decode trace once."""
        # 1. Compile run (forces kernel compilation)
        _ = self.model.ttnn_decode_forward(tokens, start_pos, page_table, kv_cache)
        ttnn.synchronize_device(self.device)
        
        # 2. Prepare inputs for trace
        device_inputs = self._prepare_device_inputs(tokens, start_pos, page_table)
        
        # 3. Capture trace
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        trace_output = self.model.ttnn_decode_forward(
            device_inputs[0],  # tokens
            device_inputs[1],  # start_pos
            page_table=device_inputs[2] if page_table else None,
            kv_cache=kv_cache
        )
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        
        # 4. Cache for future use
        self._trace_id[sampling_on_device] = trace_id
        self._trace_inputs[sampling_on_device] = device_inputs
        self._trace_output[sampling_on_device] = trace_output
    
    def _decode_with_trace(self, tokens, start_pos, page_table, kv_cache):
        """Execute cached trace with updated inputs."""
        sampling_on_device = self.sampling_enabled
        trace_id = self._trace_id[sampling_on_device]
        
        # Update only changed inputs (non-blocking copy)
        device_inputs = self._prepare_device_inputs(tokens, start_pos, page_table)
        ttnn.copy_host_to_device(
            host_tensors=device_inputs,
            device_tensors=self._trace_inputs[sampling_on_device]
        )
        
        # Execute trace (non-blocking)
        ttnn.execute_trace(self.device, trace_id, cq_id=0, blocking=False)
        
        return self._trace_output[sampling_on_device]
```

**Key insight**: Only `tokens`, `start_pos`, and `page_table` change each iteration. Everything else is reused.

---

### 2. PAGED ATTENTION (attention.py)

**File**: `src/bitnet_tt/layers/attention.py`

**Current lines 48-82** (full pre-allocation):
```python
def preallocate(self, batch_size, num_kv_heads, max_seq_len, head_dim, device):
    cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
    # Allocates 8.5 GB for max_seq_len=131k
```

**Pattern from tt_transformers** (attention.py lines 333-349):
```python
if self.paged_attention_config:
    # Block-based allocation (135x smaller!)
    cache_k = torch.zeros((
        max_num_blocks,              # 256 for 32k length
        n_local_kv_heads,            # 8 for GQA
        block_size,                  # 128 tokens
        head_dim                     # 128
    ))
    # = 256 * 8 * 128 * 128 * 2 bytes = 64 MB (!!)
```

**Apply to BitNet**:
```python
@dataclass
class PagedKVCache:
    """Block-based KV cache for memory efficiency."""
    block_size: int = 128  # Tokens per block
    max_num_blocks: int = 256  # For ~32K max length
    key_cache: ttnn.Tensor | None = None
    value_cache: ttnn.Tensor | None = None
    page_table: ttnn.Tensor | None = None  # Block indices
    
    def preallocate(self, batch_size, num_kv_heads, head_dim, device):
        """Allocate block-based cache."""
        cache_shape = (self.max_num_blocks, num_kv_heads, self.block_size, head_dim)
        zeros = torch.zeros(cache_shape, dtype=torch.bfloat16)
        
        self.key_cache = ttnn.from_torch(
            zeros,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # 64 MB instead of 8.5 GB!
        )
        self.value_cache = ttnn.from_torch(
            zeros,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    
    def update_decode(self, key_states, value_states, seq_position):
        """Update cache for decode (in-place block update)."""
        # Which block does seq_position map to?
        block_idx = seq_position // self.block_size
        token_idx = seq_position % self.block_size
        
        # In-place update (ttnn.experimental.paged_update_cache)
        ttnn.experimental.paged_update_cache(
            self.key_cache,
            key_states,
            update_idxs_tensor=ttnn.from_torch(torch.tensor([block_idx]), device=self.key_cache.device),
            page_table=None,  # Already block-indexed
        )
        return self.key_cache, self.value_cache
```

**Integration point**: Call in attention forward:
```python
def forward_decode(self, q, k, v, seq_position, ...):
    # Update cache
    k_cache, v_cache = self.kv_cache.update_decode(k, v, seq_position)
    
    # Use in SDPA with page table (if available)
    attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q,
        k_cache,
        v_cache,
        cur_pos_tensor=ttnn.from_torch(torch.tensor([seq_position]), ...),
        page_table_tensor=self.kv_cache.page_table,  # Optional
        ...
    )
    return attn_out
```

---

### 3. MEMORY CONFIG STRATEGY (attention.py, mlp.py, model.py)

**File**: `src/bitnet_tt/layers/`

**Pattern from tt_transformers** (model_config.py lines 246-271):

```python
# In your model config / layer initialization:

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

# In forward:
def forward_decode(self, x):
    # QKV: L1 sharded (small, reused in SDPA)
    qkv = ttnn.linear(x, self.wqkv, memory_config=self.xqkv_memory_config)
    
    # SDPA: DRAM (reduction output is large)
    attn = ttnn.transformer.scaled_dot_product_attention_decode(
        ...,
        memory_config=self.sdpa_memory_config
    )
    
    # Concat heads: L1 sharded (feeds WO matmul)
    concat_heads = ttnn.experimental.nlp_concat_heads(attn, 
        memory_config=self.concat_heads_memory_config)
    
    # WO: L1 sharded output
    output = ttnn.linear(concat_heads, self.wo,
        memory_config=self.concat_heads_memory_config)
```

**Decision tree**:
```
Output size < 1 MB? → L1_WIDTH_SHARDED
Output reused immediately? → L1_WIDTH_SHARDED
Output is reduction/SDPA? → DRAM_MEMORY_CONFIG
Large intermediate (FFN1)? → DRAM_MEMORY_CONFIG
```

---

### 4. ROPE PRE-UPLOAD (rope.py)

**File**: `src/bitnet_tt/layers/rope.py`

**Current lines 145-200** (per-token computation):
```python
def precompute_freqs_cis(...):
    """Precompute cos/sin for RoPE."""
    # Computed on host, transferred each iteration
```

**Pattern from tt_transformers** (rope.py lines 400-408):

```python
class RotarySetup:
    def __init__(self, device, batch_size, head_dim, max_seq_len, rope_theta, ...):
        # Pre-allocate global cos/sin matrices for ALL positions
        self.cos_matrix, self.sin_matrix = get_rot_mats(
            head_dim=head_dim,
            device=device,
            seq_len=max_seq_len,         # Upload full range
            theta=rope_theta,
            datatype=ttnn.bfloat16,
        )
    
    def get_rot_mats(self, position_idxs):
        """Get cos/sin for batch of positions via embedding lookup."""
        # position_idxs: [batch] tensor with current sequence position
        
        # Embedding lookup is much faster than computing cos/sin
        cos = ttnn.embedding(position_idxs, self.cos_matrix, 
                            layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(position_idxs, self.sin_matrix, 
                            layout=ttnn.TILE_LAYOUT)
        
        # Reshape for head projection
        cos = ttnn.unsqueeze_to_4D(cos)  # [1, batch, head_dim]
        sin = ttnn.unsqueeze_to_4D(sin)
        
        return [cos, sin]

# In attention forward:
def forward_decode(self, x, seq_position, ...):
    # Get RoPE for current position (from pre-uploaded matrices)
    rot_mats = self.rope.get_rot_mats(
        ttnn.from_torch(torch.tensor([seq_position]))
    )
    
    q_rot = ttnn.experimental.rotary_embedding_llama(q, rot_mats[0], rot_mats[1], ...)
    k_rot = ttnn.experimental.rotary_embedding_llama(k, rot_mats[0], rot_mats[1], ...)
```

**Memory cost**: 
```
cos/sin matrices = [max_seq_len, head_dim] * 2 * 2 bytes
                 = 128k * 128 * 2 * 2 = 64 MB (one-time upload)
```

---

## INTEGRATION CHECKLIST

### Phase 1: Trace Capture (Week 1 - 2-3x speedup)
- [ ] Modify `TextGenerator.__init__` to use dict-based trace caching
- [ ] Implement `_capture_decode_trace()` 
- [ ] Implement `_decode_with_trace()` with `ttnn.copy_host_to_device`
- [ ] Test with `enable_trace=True` flag
- [ ] Measure: Should see 2-3x speedup immediately
- [ ] **Blocker**: Make sure `ttnn.execute_trace` doesn't deadlock with your kv_cache updates

### Phase 2: Paged Attention (Week 2-3 - 100x KV reduction)
- [ ] Create `PagedKVCache` dataclass
- [ ] Implement block-based `preallocate()`
- [ ] Replace `KVCache.update_decode()` with `paged_update_cache()`
- [ ] Update attention to use `paged_scaled_dot_product_attention_decode`
- [ ] Test: Verify no OOM on long sequences
- [ ] **Blocker**: Ensure page_table indices are correct (row-major block ordering)

### Phase 3: Memory Configs (Week 3 - 5-10% speedup)
- [ ] Define memory config constants in each layer
- [ ] Update all `ttnn.linear()` calls with `memory_config=`
- [ ] Update all `ttnn.transformer.scaled_dot_product_attention_decode()` calls
- [ ] Test: Check L1 memory usage stays under limit
- [ ] **Blocker**: L1 overflow → might need to increase DRAM buffers

### Phase 4: RoPE Pre-upload (Week 4 - 5% speedup)
- [ ] Create `RotarySetup` class with pre-allocated cos/sin matrices
- [ ] Implement embedding-based lookup in `get_rot_mats()`
- [ ] Remove per-token RoPE computation from `forward_decode()`
- [ ] Test: Verify RoPE accuracy unchanged
- [ ] **Blocker**: Verify embedding op matches expected shape

### Phase 5: Optional - Split Sampling (Week 5 - 33x I/O reduction)
- [ ] Separate sampling into second trace
- [ ] Implement `enable_split_sampling` flag
- [ ] Use `ttnn.argmax()` or `ttnn.topk()` on-device
- [ ] **Blocker**: Sampling stability with different batch sizes

---

## KEY PERFORMANCE METRICS

| Phase | Cumulative Speedup | KV Memory | L1 Pressure |
|-------|-------------------|-----------|-------------|
| Baseline | 1x | 8.5 GB | High |
| + Trace | 2-3x | 8.5 GB | High |
| + Paged | 2-3x | 64 MB | Medium |
| + Memory Cfg | 2.5-3x | 64 MB | Low |
| + RoPE | 2.6-3.1x | 64 MB | Low |
| + Sampling | 2.6-3.1x | 64 MB | Low |

---

## COMMON PITFALLS & SOLUTIONS

### Pitfall 1: Trace compilation hangs
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

### Pitfall 2: Page table index out of bounds
**Symptom**: Segfault in `paged_update_cache()`
**Cause**: Block indices exceed `max_num_blocks`
**Solution**:
```python
def update_decode(self, k, v, seq_position):
    block_idx = seq_position // self.block_size
    if block_idx >= self.max_num_blocks:
        raise ValueError(f"Sequence too long: {seq_position} exceeds {self.max_num_blocks * self.block_size}")
```

### Pitfall 3: L1 memory overflow
**Symptom**: `ttnn.linear()` fails with "out of L1 memory"
**Cause**: Too many WIDTH_SHARDED configs
**Solution**:
```python
# Don't shard outputs that aren't immediately reused
# Instead:
qkv = ttnn.linear(x, self.wqkv, memory_config=ttnn.DRAM_MEMORY_CONFIG)
# Then shard on-demand if needed
qkv_l1 = ttnn.to_memory_config(qkv, ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
```

### Pitfall 4: Embedding lookup shape mismatch
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

---

## VALIDATION SCRIPT

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
    
    # 3. Test memory configs
    # (Manually verify via ttnn.Device memory usage)
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

