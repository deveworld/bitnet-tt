"""Comprehensive test to verify the actual cause of pos_tensor corruption."""

import torch
import ttnn
import numpy as np
import time

from bitnet_tt.model.bitnet import create_model
from bitnet_tt.utils.device import device_context
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.layers.attention import KVCache
from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn


def check_tensor(tensor, expected=6):
    ttnn.synchronize_device(tensor.device())
    val = ttnn.to_torch(tensor).numpy().flatten()[0]
    return val, val == expected


def create_pos_tensor(current_pos, device, memory_config, size=1):
    if size == 1:
        shape = [current_pos]
    else:
        shape = [current_pos] + [0] * (size - 1)
    return ttnn.from_torch(
        torch.tensor(shape, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def run_paged_updates(
    cache, key, value, current_pos, pos_tensor, num_kv_groups, sync_after_each=False
):
    kv_heads = key.shape[1]
    padded_heads = 32
    pad_amount = padded_heads - kv_heads
    device = key.device()

    key_rm = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
    key_1bkd = ttnn.permute(key_rm, (2, 0, 1, 3))
    key_1bkd = ttnn.to_layout(key_1bkd, ttnn.TILE_LAYOUT)

    value_rm = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
    value_1bkd = ttnn.permute(value_rm, (2, 0, 1, 3))
    value_1bkd = ttnn.to_layout(value_1bkd, ttnn.TILE_LAYOUT)

    key_padded = ttnn.pad(key_1bkd, [(0, 0), (0, 0), (0, pad_amount), (0, 0)], 0.0)
    value_padded = ttnn.pad(value_1bkd, [(0, 0), (0, 0), (0, pad_amount), (0, 0)], 0.0)

    shard_config = ttnn.create_sharded_memory_config(
        shape=(32, cache.head_dim),
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    key_sharded = ttnn.to_memory_config(key_padded, shard_config)
    value_sharded = ttnn.to_memory_config(value_padded, shard_config)

    if sync_after_each:
        ttnn.synchronize_device(device)

    ttnn.experimental.paged_update_cache(
        cache.key_cache, key_sharded, update_idxs_tensor=pos_tensor
    )

    if sync_after_each:
        ttnn.synchronize_device(device)

    ttnn.experimental.paged_update_cache(
        cache.value_cache, value_sharded, update_idxs_tensor=pos_tensor
    )

    if sync_after_each:
        ttnn.synchronize_device(device)

    ttnn.deallocate(key_1bkd)
    ttnn.deallocate(value_1bkd)
    ttnn.deallocate(key_padded)
    ttnn.deallocate(value_padded)
    ttnn.deallocate(key_sharded)
    ttnn.deallocate(value_sharded)


def test_hypothesis_1_sync():
    """Test: Is this an async execution issue?"""
    print("\n" + "=" * 60)
    print("HYPOTHESIS 1: Async execution issue")
    print("Test: Add synchronization after every paged_update_cache call")
    print("=" * 60)

    with device_context() as device:
        state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
        model = create_model(config, device)
        load_weights_to_model(model, state_dict)

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

        prefill_tokens = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
        input_tensor = numpy_int_to_ttnn(prefill_tokens, device)
        logits, caches = model(input_tensor, past_key_values=caches, use_cache=True, mode="prefill")
        ttnn.deallocate(input_tensor)
        ttnn.deallocate(logits)
        ttnn.synchronize_device(device)

        current_pos = 6
        pos_tensor = create_pos_tensor(current_pos, device, ttnn.DRAM_MEMORY_CONFIG)

        decode_token = np.array([[7]], dtype=np.int64)
        input_ids = numpy_int_to_ttnn(decode_token, device)
        hidden = model.embed_tokens(input_ids)

        corruption_layer = None
        for layer_idx in range(30):
            layer = model.layers[layer_idx]
            attn = layer.self_attn
            cache = caches[layer_idx]

            hidden_norm = layer.input_layernorm(hidden)

            query = attn.q_proj(hidden_norm)
            key = attn.k_proj(hidden_norm)
            value = attn.v_proj(hidden_norm)

            query = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
            key = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
            value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)

            query = ttnn.reshape(query, (1, 1, attn.num_heads, attn.head_dim))
            key = ttnn.reshape(key, (1, 1, attn.num_kv_heads, attn.head_dim))
            value = ttnn.reshape(value, (1, 1, attn.num_kv_heads, attn.head_dim))

            query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
            key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
            value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

            query = ttnn.transpose(query, 1, 2)
            key = ttnn.transpose(key, 1, 2)
            value = ttnn.transpose(value, 1, 2)

            run_paged_updates(
                cache, key, value, current_pos, pos_tensor, attn.num_kv_groups, sync_after_each=True
            )

            val, ok = check_tensor(pos_tensor)
            if not ok:
                corruption_layer = layer_idx
                print(f"  Layer {layer_idx}: CORRUPTED to {val}")
                break

            # Deallocate intermediate tensors
            ttnn.deallocate(hidden_norm)
            ttnn.deallocate(query)
            ttnn.deallocate(key)
            ttnn.deallocate(value)
            ttnn.synchronize_device(device)

        if corruption_layer is None:
            print("  Result: All 30 layers passed with sync - NOT an async issue")
            return True
        else:
            print(
                f"  Result: Corruption at layer {corruption_layer} even WITH sync - NOT purely async issue"
            )
            return False


def test_hypothesis_2_buffer_size():
    """Test: Is this a memory aliasing issue due to small buffer?"""
    print("\n" + "=" * 60)
    print("HYPOTHESIS 2: Memory aliasing (small buffer reuse)")
    print("Test: Use larger buffer (32 elements instead of 1)")
    print("=" * 60)

    with device_context() as device:
        state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
        model = create_model(config, device)
        load_weights_to_model(model, state_dict)

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

        prefill_tokens = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
        input_tensor = numpy_int_to_ttnn(prefill_tokens, device)
        logits, caches = model(input_tensor, past_key_values=caches, use_cache=True, mode="prefill")
        ttnn.deallocate(input_tensor)
        ttnn.deallocate(logits)
        ttnn.synchronize_device(device)

        current_pos = 6
        # Use 32-element buffer instead of 1
        pos_tensor = ttnn.from_torch(
            torch.tensor([current_pos] + [0] * 31, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        decode_token = np.array([[7]], dtype=np.int64)
        input_ids = numpy_int_to_ttnn(decode_token, device)
        hidden = model.embed_tokens(input_ids)

        corruption_layer = None
        for layer_idx in range(30):
            layer = model.layers[layer_idx]
            attn = layer.self_attn
            cache = caches[layer_idx]

            hidden_norm = layer.input_layernorm(hidden)

            query = attn.q_proj(hidden_norm)
            key = attn.k_proj(hidden_norm)
            value = attn.v_proj(hidden_norm)

            query = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
            key = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
            value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)

            query = ttnn.reshape(query, (1, 1, attn.num_heads, attn.head_dim))
            key = ttnn.reshape(key, (1, 1, attn.num_kv_heads, attn.head_dim))
            value = ttnn.reshape(value, (1, 1, attn.num_kv_heads, attn.head_dim))

            query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
            key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
            value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

            query = ttnn.transpose(query, 1, 2)
            key = ttnn.transpose(key, 1, 2)
            value = ttnn.transpose(value, 1, 2)

            run_paged_updates(
                cache,
                key,
                value,
                current_pos,
                pos_tensor,
                attn.num_kv_groups,
                sync_after_each=False,
            )

            ttnn.synchronize_device(device)
            val = ttnn.to_torch(pos_tensor).numpy()[0]
            if val != 6:
                corruption_layer = layer_idx
                print(f"  Layer {layer_idx}: CORRUPTED to {val}")
                break

            ttnn.deallocate(hidden_norm)
            ttnn.deallocate(query)
            ttnn.deallocate(key)
            ttnn.deallocate(value)
            ttnn.synchronize_device(device)

        if corruption_layer is None:
            print(
                "  Result: All 30 layers passed with 32-element buffer - memory aliasing issue CONFIRMED"
            )
            return True
        else:
            print(
                f"  Result: Corruption at layer {corruption_layer} even with 32-element buffer - NOT memory aliasing"
            )
            return False


def test_hypothesis_3_l1_memory():
    """Test: Is this a DRAM-specific issue? Try L1."""
    print("\n" + "=" * 60)
    print("HYPOTHESIS 3: DRAM-specific issue")
    print("Test: Use L1_MEMORY_CONFIG instead of DRAM_MEMORY_CONFIG")
    print("=" * 60)

    with device_context() as device:
        state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
        model = create_model(config, device)
        load_weights_to_model(model, state_dict)

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

        prefill_tokens = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
        input_tensor = numpy_int_to_ttnn(prefill_tokens, device)
        logits, caches = model(input_tensor, past_key_values=caches, use_cache=True, mode="prefill")
        ttnn.deallocate(input_tensor)
        ttnn.deallocate(logits)
        ttnn.synchronize_device(device)

        current_pos = 6
        # Use L1 memory instead of DRAM
        pos_tensor = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        decode_token = np.array([[7]], dtype=np.int64)
        input_ids = numpy_int_to_ttnn(decode_token, device)
        hidden = model.embed_tokens(input_ids)

        corruption_layer = None
        for layer_idx in range(30):
            layer = model.layers[layer_idx]
            attn = layer.self_attn
            cache = caches[layer_idx]

            hidden_norm = layer.input_layernorm(hidden)

            query = attn.q_proj(hidden_norm)
            key = attn.k_proj(hidden_norm)
            value = attn.v_proj(hidden_norm)

            query = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
            key = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
            value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)

            query = ttnn.reshape(query, (1, 1, attn.num_heads, attn.head_dim))
            key = ttnn.reshape(key, (1, 1, attn.num_kv_heads, attn.head_dim))
            value = ttnn.reshape(value, (1, 1, attn.num_kv_heads, attn.head_dim))

            query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
            key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
            value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

            query = ttnn.transpose(query, 1, 2)
            key = ttnn.transpose(key, 1, 2)
            value = ttnn.transpose(value, 1, 2)

            run_paged_updates(
                cache,
                key,
                value,
                current_pos,
                pos_tensor,
                attn.num_kv_groups,
                sync_after_each=False,
            )

            ttnn.synchronize_device(device)
            val = ttnn.to_torch(pos_tensor).numpy()[0]
            if val != 6:
                corruption_layer = layer_idx
                print(f"  Layer {layer_idx}: CORRUPTED to {val}")
                break

            ttnn.deallocate(hidden_norm)
            ttnn.deallocate(query)
            ttnn.deallocate(key)
            ttnn.deallocate(value)
            ttnn.synchronize_device(device)

        if corruption_layer is None:
            print("  Result: All 30 layers passed with L1 memory - DRAM issue CONFIRMED")
            return True
        else:
            print(
                f"  Result: Corruption at layer {corruption_layer} even with L1 - NOT DRAM-specific"
            )
            return False


def test_hypothesis_4_paged_update_internal():
    """Test: Does paged_update_cache internally modify the index tensor?"""
    print("\n" + "=" * 60)
    print("HYPOTHESIS 4: paged_update_cache modifies index tensor internally")
    print("Test: Call paged_update_cache 60 times in isolation (no model)")
    print("=" * 60)

    with device_context() as device:
        cache = KVCache()
        cache.preallocate(
            batch_size=1,
            num_kv_heads=5,
            max_seq_len=64,
            head_dim=128,
            device=device,
            use_paged=True,
        )

        current_pos = 6
        pos_tensor = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        k_data = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16)
        v_data = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16)

        shard_config = ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        corruption_call = None
        for i in range(60):
            k_tt = ttnn.from_torch(
                k_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            v_tt = ttnn.from_torch(
                v_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

            k_sharded = ttnn.to_memory_config(k_tt, shard_config)
            v_sharded = ttnn.to_memory_config(v_tt, shard_config)

            ttnn.experimental.paged_update_cache(
                cache.key_cache, k_sharded, update_idxs_tensor=pos_tensor
            )
            ttnn.experimental.paged_update_cache(
                cache.value_cache, v_sharded, update_idxs_tensor=pos_tensor
            )

            ttnn.deallocate(k_tt)
            ttnn.deallocate(v_tt)
            ttnn.deallocate(k_sharded)
            ttnn.deallocate(v_sharded)

            ttnn.synchronize_device(device)
            val = ttnn.to_torch(pos_tensor).numpy()[0]
            if val != 6:
                corruption_call = i
                print(f"  Call {i}: CORRUPTED to {val}")
                break
            elif i % 10 == 0:
                print(f"  Call {i}: OK")

        if corruption_call is None:
            print("  Result: All 60 calls passed in isolation - NOT paged_update_cache alone")
            return False
        else:
            print(
                f"  Result: Corruption at call {corruption_call} - paged_update_cache DOES corrupt"
            )
            return True


def test_hypothesis_5_model_ops():
    """Test: Is it caused by other model operations (matmul, reshape, etc.)?"""
    print("\n" + "=" * 60)
    print("HYPOTHESIS 5: Other model operations corrupt the tensor")
    print("Test: Run model layers WITHOUT paged_update_cache")
    print("=" * 60)

    with device_context() as device:
        state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
        model = create_model(config, device)
        load_weights_to_model(model, state_dict)

        caches = []
        for _ in range(config.num_layers):
            cache = KVCache()
            cache.preallocate(
                batch_size=1,
                num_kv_heads=config.num_key_value_heads,
                max_seq_len=64,
                head_dim=config.head_dim,
                device=device,
                use_paged=False,  # Use simple cache, no paged_update_cache
            )
            caches.append(cache)

        prefill_tokens = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
        input_tensor = numpy_int_to_ttnn(prefill_tokens, device)
        logits, caches = model(input_tensor, past_key_values=caches, use_cache=True, mode="prefill")
        ttnn.deallocate(input_tensor)
        ttnn.deallocate(logits)
        ttnn.synchronize_device(device)

        current_pos = 6
        pos_tensor = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        decode_token = np.array([[7]], dtype=np.int64)
        input_ids = numpy_int_to_ttnn(decode_token, device)
        hidden = model.embed_tokens(input_ids)

        corruption_layer = None
        for layer_idx in range(30):
            layer = model.layers[layer_idx]
            attn = layer.self_attn
            cache = caches[layer_idx]

            hidden_norm = layer.input_layernorm(hidden)

            query = attn.q_proj(hidden_norm)
            key = attn.k_proj(hidden_norm)
            value = attn.v_proj(hidden_norm)

            query = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
            key = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
            value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)

            query = ttnn.reshape(query, (1, 1, attn.num_heads, attn.head_dim))
            key = ttnn.reshape(key, (1, 1, attn.num_kv_heads, attn.head_dim))
            value = ttnn.reshape(value, (1, 1, attn.num_kv_heads, attn.head_dim))

            query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
            key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
            value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

            query = ttnn.transpose(query, 1, 2)
            key = ttnn.transpose(key, 1, 2)
            value = ttnn.transpose(value, 1, 2)

            # Use simple cache update (no paged_update_cache)
            cache.update_decode_simple(key, value, current_pos, attn.num_kv_groups)

            ttnn.synchronize_device(device)
            val = ttnn.to_torch(pos_tensor).numpy()[0]
            if val != 6:
                corruption_layer = layer_idx
                print(f"  Layer {layer_idx}: CORRUPTED to {val}")
                break

            ttnn.deallocate(hidden_norm)
            ttnn.deallocate(query)
            ttnn.deallocate(key)
            ttnn.deallocate(value)
            ttnn.synchronize_device(device)

        if corruption_layer is None:
            print(
                "  Result: All 30 layers passed WITHOUT paged_update_cache - confirms paged_update_cache is the cause"
            )
            return True
        else:
            print(
                f"  Result: Corruption at layer {corruption_layer} even WITHOUT paged_update_cache - other ops also corrupt"
            )
            return False


if __name__ == "__main__":
    print("=" * 60)
    print("CORRUPTION CAUSE VERIFICATION")
    print("=" * 60)

    results = {}

    print("\nLoading model (this takes a while)...")

    results["hypothesis_1_sync"] = test_hypothesis_1_sync()
    results["hypothesis_2_buffer"] = test_hypothesis_2_buffer_size()
    results["hypothesis_3_l1"] = test_hypothesis_3_l1_memory()
    results["hypothesis_4_isolated"] = test_hypothesis_4_paged_update_internal()
    results["hypothesis_5_other_ops"] = test_hypothesis_5_model_ops()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "PASSED (confirms hypothesis)" if result else "FAILED (rules out hypothesis)"
        print(f"  {name}: {status}")

    print("\nDone")
