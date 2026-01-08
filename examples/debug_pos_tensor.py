"""
Debug script to isolate the current_pos_tensor modification issue.

Run on TT server:
    cd ~/bitnet-tt && source .venv/bin/activate
    python3 examples/debug_pos_tensor.py
"""

import torch
import ttnn
import numpy as np

from bitnet_tt.model.bitnet import create_model
from bitnet_tt.utils.device import device_context
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.layers.attention import KVCache


def test_pos_tensor_isolation():
    """Test if paged_update_cache or SDPA decode modifies the position tensor."""

    with device_context() as device:
        print("=" * 60)
        print("Test 1: Check if paged_update_cache modifies pos tensor")
        print("=" * 60)

        # Create test tensors
        batch_size = 1
        num_kv_heads = 5
        padded_heads = 32
        max_seq_len = 64
        head_dim = 128
        current_pos = 6

        # Create KV cache
        cache_shape = (batch_size, padded_heads, max_seq_len, head_dim)
        k_cache = ttnn.from_torch(
            torch.zeros(cache_shape, dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_cache = ttnn.from_torch(
            torch.zeros(cache_shape, dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Create new KV to insert (in 1BKD format, padded to 32 heads)
        kv_1bkd = torch.randn(1, batch_size, padded_heads, head_dim, dtype=torch.bfloat16)
        k_new = ttnn.from_torch(
            kv_1bkd,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        v_new = ttnn.from_torch(
            kv_1bkd,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        # Create position tensor (int32)
        pos_tensor = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        print(f"pos_tensor BEFORE paged_update_cache: {ttnn.to_torch(pos_tensor).numpy()}")

        # Create shard config for 32 heads
        shard_config = ttnn.create_sharded_memory_config(
            shape=(32, head_dim),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        k_sharded = ttnn.to_memory_config(k_new, shard_config)
        v_sharded = ttnn.to_memory_config(v_new, shard_config)

        # Call paged_update_cache
        ttnn.experimental.paged_update_cache(k_cache, k_sharded, update_idxs_tensor=pos_tensor)
        ttnn.experimental.paged_update_cache(v_cache, v_sharded, update_idxs_tensor=pos_tensor)

        ttnn.synchronize_device(device)

        print(f"pos_tensor AFTER paged_update_cache: {ttnn.to_torch(pos_tensor).numpy()}")

        print()
        print("=" * 60)
        print("Test 2: Check if SDPA decode modifies pos tensor")
        print("=" * 60)

        # Create fresh position tensor
        pos_tensor2 = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # Create Q tensor in 1BKD format [1, batch, num_heads, head_dim]
        num_heads = 20
        q_1bkd = torch.randn(1, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
        q_tensor = ttnn.from_torch(
            q_1bkd,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        # Use full cache (expanded to num_heads)
        k_expanded = k_cache[:, :num_heads, :, :]
        v_expanded = v_cache[:, :num_heads, :, :]

        # Expand for GQA (num_heads/num_kv_heads = 4)
        k_expanded = ttnn.repeat_interleave(k_cache[:, :num_kv_heads, :, :], 4, dim=1)
        v_expanded = ttnn.repeat_interleave(v_cache[:, :num_kv_heads, :, :], 4, dim=1)

        print(f"pos_tensor2 BEFORE sdpa_decode: {ttnn.to_torch(pos_tensor2).numpy()}")
        print(f"Q shape: {q_tensor.shape}")
        print(f"K shape: {k_expanded.shape}")
        print(f"V shape: {v_expanded.shape}")

        # Call scaled_dot_product_attention_decode
        try:
            attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
                q_tensor,
                k_expanded,
                v_expanded,
                cur_pos_tensor=pos_tensor2,
                scale=1.0 / (head_dim**0.5),
            )
            ttnn.synchronize_device(device)
            print(f"pos_tensor2 AFTER sdpa_decode: {ttnn.to_torch(pos_tensor2).numpy()}")
            print(f"SDPA output shape: {attn_output.shape}")
        except Exception as e:
            print(f"SDPA decode failed: {e}")

        print()
        print("=" * 60)
        print("Test 3: Full model forward pass")
        print("=" * 60)

        # Load full model
        print("Loading BitNet model...")
        state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
        model = create_model(config, device)
        load_weights_to_model(model, state_dict)

        # Pre-allocate caches with paged option
        caches = []
        for _ in range(config.num_layers):
            cache = KVCache()
            cache.preallocate(
                batch_size=1,
                num_kv_heads=config.num_key_value_heads,
                max_seq_len=64,
                head_dim=config.head_dim,
                device=device,
                use_paged=True,
            )
            caches.append(cache)

        # Prefill with 6 tokens
        from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn

        prefill_tokens = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
        input_tensor = numpy_int_to_ttnn(prefill_tokens, device)

        print("Running prefill...")
        logits, caches = model(
            input_tensor,
            past_key_values=caches,
            use_cache=True,
            mode="prefill",
        )
        ttnn.deallocate(input_tensor)
        ttnn.deallocate(logits)
        ttnn.synchronize_device(device)

        print(f"After prefill, cache seq_len: {caches[0].seq_len_cached}")

        # Now decode with position 6
        decode_token = np.array([[7]], dtype=np.int64)
        input_tensor = numpy_int_to_ttnn(decode_token, device)
        current_pos = 6

        # Create position tensors
        pos_tensor_uint32 = ttnn.from_torch(
            torch.tensor([[current_pos]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        pos_tensor_int32 = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        print(f"pos_tensor_int32 BEFORE model forward: {ttnn.to_torch(pos_tensor_int32).numpy()}")

        # Run decode
        print("Running decode...")
        logits, caches = model(
            input_tensor,
            past_key_values=caches,
            use_cache=True,
            mode="decode",
            current_pos=current_pos,
            pos_tensor=pos_tensor_uint32,
            current_pos_tensor=pos_tensor_int32,
        )
        ttnn.synchronize_device(device)

        print(f"pos_tensor_int32 AFTER model forward: {ttnn.to_torch(pos_tensor_int32).numpy()}")

        # Check outputs
        logits_np = ttnn.to_torch(logits).float().numpy()
        print(f"Output logits shape: {logits_np.shape}")
        print(f"Output logits sample: {logits_np[0, 0, :5]}")

        # Cleanup
        ttnn.deallocate(input_tensor)
        ttnn.deallocate(logits)

        print()
        print("=" * 60)
        print("DEBUG COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    test_pos_tensor_isolation()
