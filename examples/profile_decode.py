#!/usr/bin/env python3
"""Profile decode step to identify bottlenecks."""

import time
import numpy as np
import ttnn

# Patch attention to add timing
def profile_attention():
    """Profile attention operations."""
    from bitnet_tt.layers.attention import MultiHeadAttention

    original_call = MultiHeadAttention.__call__

    def timed_call(self, hidden_states, **kwargs):
        timings = {}

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # QKV projections
        t0 = time.perf_counter()
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        ttnn.synchronize_device(self.device)
        timings['qkv_proj'] = time.perf_counter() - t0

        # Layout conversion for reshape
        t0 = time.perf_counter()
        query = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
        key = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.synchronize_device(self.device)
        timings['to_row_major'] = time.perf_counter() - t0

        # Reshape
        t0 = time.perf_counter()
        query = ttnn.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = ttnn.reshape(key, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        value = ttnn.reshape(value, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.permute(value, (0, 2, 1, 3))
        ttnn.synchronize_device(self.device)
        timings['reshape_permute'] = time.perf_counter() - t0

        # Back to tile
        t0 = time.perf_counter()
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        ttnn.synchronize_device(self.device)
        timings['to_tile'] = time.perf_counter() - t0

        # RoPE
        from bitnet_tt.layers.bitlinear import numpy_to_ttnn
        mode = kwargs.get('mode', 'prefill')
        position_ids = kwargs.get('position_ids')
        past_key_value = kwargs.get('past_key_value')

        t0 = time.perf_counter()
        if mode == "decode" and isinstance(position_ids, int):
            pos_ids = np.array([position_ids], dtype=np.int64)
        elif past_key_value is not None and past_key_value.seq_len_cached > 0:
            start_pos = past_key_value.seq_len_cached
            pos_ids = np.arange(start_pos, start_pos + seq_len, dtype=np.int64)
        else:
            pos_ids = np.arange(seq_len, dtype=np.int64)

        cos = self.cos_np[:, :, pos_ids, :]
        sin = self.sin_np[:, :, pos_ids, :]
        cos_ttnn = numpy_to_ttnn(cos.astype(np.float32), self.device)
        sin_ttnn = numpy_to_ttnn(sin.astype(np.float32), self.device)
        ttnn.synchronize_device(self.device)
        timings['rope_upload'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        query = self._apply_rope(query, cos_ttnn, sin_ttnn)
        key = self._apply_rope(key, cos_ttnn, sin_ttnn)
        ttnn.synchronize_device(self.device)
        timings['rope_apply'] = time.perf_counter() - t0

        # KV-Cache
        t0 = time.perf_counter()
        use_cache = kwargs.get('use_cache', False)
        updated_cache = None
        if use_cache:
            from bitnet_tt.layers.attention import KVCache
            if past_key_value is None:
                past_key_value = KVCache()
            key, value = past_key_value.update(key, value)
            updated_cache = past_key_value
        ttnn.synchronize_device(self.device)
        timings['kv_cache'] = time.perf_counter() - t0

        # GQA expand
        t0 = time.perf_counter()
        if self.num_kv_groups > 1:
            key = ttnn.repeat_interleave(key, self.num_kv_groups, dim=1)
            value = ttnn.repeat_interleave(value, self.num_kv_groups, dim=1)
        ttnn.synchronize_device(self.device)
        timings['gqa_expand'] = time.perf_counter() - t0

        # SDPA
        t0 = time.perf_counter()
        is_causal = past_key_value is None or past_key_value.seq_len_cached == seq_len
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query, key, value,
            attn_mask=kwargs.get('attention_mask'),
            is_causal=is_causal,
            scale=self.scale,
        )
        ttnn.synchronize_device(self.device)
        timings['sdpa'] = time.perf_counter() - t0

        # Reshape back
        t0 = time.perf_counter()
        attn_output = ttnn.to_layout(attn_output, ttnn.ROW_MAJOR_LAYOUT)
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.hidden_size))
        attn_output = ttnn.to_layout(attn_output, ttnn.TILE_LAYOUT)
        ttnn.synchronize_device(self.device)
        timings['reshape_back'] = time.perf_counter() - t0

        # Sub-norm and o_proj
        t0 = time.perf_counter()
        attn_output = self.attn_sub_norm(attn_output)
        output = self.o_proj(attn_output)
        ttnn.synchronize_device(self.device)
        timings['subnorm_oproj'] = time.perf_counter() - t0

        # Store timings for analysis
        if not hasattr(MultiHeadAttention, '_timings'):
            MultiHeadAttention._timings = []
        MultiHeadAttention._timings.append(timings)

        return output, updated_cache

    MultiHeadAttention.__call__ = timed_call
    return MultiHeadAttention


def main():
    from bitnet_tt.model.bitnet import BitNetModel
    from bitnet_tt.config import BitNetConfig
    from bitnet_tt.utils.weight_loader import load_weights_from_hf

    print("Profiling BitNet decode step...")
    print("=" * 60)

    # Patch attention
    profile_attention()

    # Initialize
    device = ttnn.open_device(device_id=0)
    config = BitNetConfig()
    model = BitNetModel(config, device)

    print("Loading weights...")
    load_weights_from_hf(model, "microsoft/bitnet-b1.58-2B-4T")

    # Tokenize
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T")

    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]

    from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn
    from bitnet_tt.layers.attention import MultiHeadAttention

    # Prefill
    print("\n--- Prefill ---")
    input_tensor = numpy_int_to_ttnn(input_ids, device)
    logits, kv_cache = model(input_tensor, use_cache=True, mode="prefill")
    ttnn.synchronize_device(device)
    ttnn.deallocate(input_tensor)
    ttnn.deallocate(logits)

    # Print prefill timings
    if hasattr(MultiHeadAttention, '_timings'):
        prefill_timings = MultiHeadAttention._timings
        print(f"Layers profiled: {len(prefill_timings)}")

        # Aggregate
        agg = {}
        for t in prefill_timings:
            for k, v in t.items():
                agg[k] = agg.get(k, 0) + v

        total = sum(agg.values())
        print(f"Total attention time: {total*1000:.1f}ms")
        for k, v in sorted(agg.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v*1000:.1f}ms ({v/total*100:.1f}%)")

        MultiHeadAttention._timings = []

    # Decode - single token
    print("\n--- Decode (1 token) ---")
    next_token = np.array([[1]], dtype=np.int64)  # dummy token

    for i in range(3):  # Run 3 decode steps
        input_tensor = numpy_int_to_ttnn(next_token, device)

        t0 = time.perf_counter()
        logits, kv_cache = model(
            input_tensor,
            past_key_values=kv_cache,
            use_cache=True,
            mode="decode",
            current_pos=input_ids.shape[1] + i,
        )
        ttnn.synchronize_device(device)
        decode_time = time.perf_counter() - t0

        ttnn.deallocate(input_tensor)
        ttnn.deallocate(logits)

        print(f"Decode step {i+1}: {decode_time*1000:.1f}ms ({1/decode_time:.1f} tokens/s)")

    # Print decode timings
    if hasattr(MultiHeadAttention, '_timings'):
        # Last decode step timings (30 layers)
        decode_timings = MultiHeadAttention._timings[-30:]

        agg = {}
        for t in decode_timings:
            for k, v in t.items():
                agg[k] = agg.get(k, 0) + v

        total = sum(agg.values())
        print(f"\nDecode attention breakdown (30 layers):")
        print(f"Total attention time: {total*1000:.1f}ms")
        for k, v in sorted(agg.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v*1000:.1f}ms ({v/total*100:.1f}%)")

    ttnn.close_device(device)
    print("\nDone!")


if __name__ == "__main__":
    main()
