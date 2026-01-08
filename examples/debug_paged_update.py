"""Debug update_decode_paged step by step."""

import torch
import ttnn
import numpy as np

from bitnet_tt.model.bitnet import create_model
from bitnet_tt.utils.device import device_context
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.layers.attention import KVCache
from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn


def check_pos(pos_tensor, label):
    ttnn.synchronize_device(pos_tensor.device())
    val = ttnn.to_torch(pos_tensor).numpy()[0]
    print(f"  {label}: pos_tensor = {val}")
    return val


def test_paged_update_detailed():
    with device_context() as device:
        print("Loading model...")
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

        logits, caches = model(
            input_tensor,
            past_key_values=caches,
            use_cache=True,
            mode="prefill",
        )
        ttnn.deallocate(input_tensor)
        ttnn.deallocate(logits)
        ttnn.synchronize_device(device)

        current_pos = 6

        pos_tensor_int32 = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        key_states = torch.randn(1, 5, 1, 128, dtype=torch.bfloat16)
        value_states = torch.randn(1, 5, 1, 128, dtype=torch.bfloat16)

        key_tt = ttnn.from_torch(
            key_states, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        value_tt = ttnn.from_torch(
            value_states, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

        past_kv = caches[23]

        print("\n=== Step-by-step update_decode_paged ===")
        check_pos(pos_tensor_int32, "Initial")

        kv_heads = key_tt.shape[1]
        padded_heads = 32
        pad_amount = padded_heads - kv_heads

        key_rm = ttnn.to_layout(key_tt, ttnn.ROW_MAJOR_LAYOUT)
        check_pos(pos_tensor_int32, "After key to_layout ROW_MAJOR")

        key_1bkd = ttnn.permute(key_rm, (2, 0, 1, 3))
        check_pos(pos_tensor_int32, "After key permute")

        key_1bkd = ttnn.to_layout(key_1bkd, ttnn.TILE_LAYOUT)
        check_pos(pos_tensor_int32, "After key to_layout TILE")

        value_rm = ttnn.to_layout(value_tt, ttnn.ROW_MAJOR_LAYOUT)
        value_1bkd = ttnn.permute(value_rm, (2, 0, 1, 3))
        value_1bkd = ttnn.to_layout(value_1bkd, ttnn.TILE_LAYOUT)
        check_pos(pos_tensor_int32, "After value transforms")

        key_padded = ttnn.pad(key_1bkd, [(0, 0), (0, 0), (0, pad_amount), (0, 0)], 0.0)
        check_pos(pos_tensor_int32, "After key pad")

        value_padded = ttnn.pad(value_1bkd, [(0, 0), (0, 0), (0, pad_amount), (0, 0)], 0.0)
        check_pos(pos_tensor_int32, "After value pad")

        shard_config = ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        key_sharded = ttnn.to_memory_config(key_padded, shard_config)
        check_pos(pos_tensor_int32, "After key to_memory_config (sharded)")

        value_sharded = ttnn.to_memory_config(value_padded, shard_config)
        check_pos(pos_tensor_int32, "After value to_memory_config (sharded)")

        print("\n=== Before paged_update_cache ===")
        check_pos(pos_tensor_int32, "Before paged_update_cache")

        ttnn.experimental.paged_update_cache(
            past_kv.key_cache, key_sharded, update_idxs_tensor=pos_tensor_int32
        )
        check_pos(pos_tensor_int32, "After FIRST paged_update_cache (key)")

        ttnn.experimental.paged_update_cache(
            past_kv.value_cache, value_sharded, update_idxs_tensor=pos_tensor_int32
        )
        check_pos(pos_tensor_int32, "After SECOND paged_update_cache (value)")

        print("\n=== Cleanup ===")
        ttnn.deallocate(key_1bkd)
        check_pos(pos_tensor_int32, "After deallocate key_1bkd")

        ttnn.deallocate(value_1bkd)
        check_pos(pos_tensor_int32, "After deallocate value_1bkd")

        ttnn.deallocate(key_padded)
        check_pos(pos_tensor_int32, "After deallocate key_padded")

        ttnn.deallocate(value_padded)
        check_pos(pos_tensor_int32, "After deallocate value_padded")

        ttnn.deallocate(key_sharded)
        check_pos(pos_tensor_int32, "After deallocate key_sharded")

        ttnn.deallocate(value_sharded)
        check_pos(pos_tensor_int32, "After deallocate value_sharded")

        print("\nDone")


if __name__ == "__main__":
    test_paged_update_detailed()
