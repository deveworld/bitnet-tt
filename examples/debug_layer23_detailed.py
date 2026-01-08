"""Debug layer 23 with buffer addresses."""

import torch
import ttnn
import numpy as np

from bitnet_tt.model.bitnet import create_model
from bitnet_tt.utils.device import device_context
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.layers.attention import KVCache
from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn


def check_pos(pos_tensor, label, show_addr=False):
    ttnn.synchronize_device(pos_tensor.device())
    val = ttnn.to_torch(pos_tensor).numpy()[0]
    addr = pos_tensor.buffer_address() if show_addr else ""
    addr_str = f" [addr={addr}]" if show_addr else ""
    print(f"  {label}: pos_tensor = {val}{addr_str}")
    return val


def test_layer23_detailed():
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pos_addr = pos_tensor_int32.buffer_address()
        print(f"\npos_tensor_int32 buffer_address: {pos_addr}")

        pos_tensor_uint32 = ttnn.from_torch(
            torch.tensor([[current_pos]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        decode_token = np.array([[7]], dtype=np.int64)
        input_ids = numpy_int_to_ttnn(decode_token, device)
        hidden_states = model.embed_tokens(input_ids)

        for layer_idx in range(23):
            hidden_states, _ = model.layers[layer_idx](
                hidden_states,
                attention_mask=None,
                position_ids=current_pos,
                past_key_value=caches[layer_idx],
                use_cache=True,
                mode="decode",
                rot_mats=None,
                transformation_mat=None,
                current_pos_tensor=pos_tensor_int32,
                pos_tensor=pos_tensor_uint32,
            )
            caches[layer_idx] = _

        print(f"\nBefore layer 23:")
        check_pos(pos_tensor_int32, "Initial", True)

        layer = model.layers[23]
        past_kv = caches[23]
        attn = layer.self_attn
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        check_pos(pos_tensor_int32, "After input_layernorm")

        query = attn.q_proj(hidden_states)
        check_pos(pos_tensor_int32, "After q_proj")

        key = attn.k_proj(hidden_states)
        check_pos(pos_tensor_int32, "After k_proj")

        value = attn.v_proj(hidden_states)
        check_pos(pos_tensor_int32, "After v_proj")

        query = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
        key = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
        check_pos(pos_tensor_int32, "After to_layout ROW_MAJOR")

        query = ttnn.reshape(query, (batch_size, seq_len, attn.num_heads, attn.head_dim))
        check_pos(pos_tensor_int32, "After query reshape")

        key = ttnn.reshape(key, (batch_size, seq_len, attn.num_kv_heads, attn.head_dim))
        check_pos(pos_tensor_int32, "After key reshape")

        value = ttnn.reshape(value, (batch_size, seq_len, attn.num_kv_heads, attn.head_dim))
        check_pos(pos_tensor_int32, "After value reshape")

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        check_pos(pos_tensor_int32, "After query to_layout TILE")

        key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
        check_pos(pos_tensor_int32, "After key to_layout TILE")

        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        check_pos(pos_tensor_int32, "After value to_layout TILE")

        query = ttnn.transpose(query, 1, 2)
        check_pos(pos_tensor_int32, "After query transpose")

        key = ttnn.transpose(key, 1, 2)
        check_pos(pos_tensor_int32, "After key transpose")

        value = ttnn.transpose(value, 1, 2)
        check_pos(pos_tensor_int32, "After value transpose")

        print("\n=== RoPE ===")
        query_before_rope = query
        key_before_rope = key
        query, key = attn._apply_rope_manual(query, key, current_pos, seq_len, pos_tensor_uint32)
        check_pos(pos_tensor_int32, "After RoPE")

        print("\n=== update_decode_paged internal ===")

        kv_heads = key.shape[1]
        padded_heads = 32
        pad_amount = padded_heads - kv_heads

        key_rm = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
        check_pos(pos_tensor_int32, "After key to_layout ROW_MAJOR")

        key_1bkd = ttnn.permute(key_rm, (2, 0, 1, 3))
        check_pos(pos_tensor_int32, "After key permute")

        key_1bkd = ttnn.to_layout(key_1bkd, ttnn.TILE_LAYOUT)
        check_pos(pos_tensor_int32, "After key to_layout TILE")

        value_rm = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
        value_1bkd = ttnn.permute(value_rm, (2, 0, 1, 3))
        value_1bkd = ttnn.to_layout(value_1bkd, ttnn.TILE_LAYOUT)
        check_pos(pos_tensor_int32, "After value transforms")

        key_padded = ttnn.pad(key_1bkd, [(0, 0), (0, 0), (0, pad_amount), (0, 0)], 0.0)
        check_pos(pos_tensor_int32, "After key pad")

        value_padded = ttnn.pad(value_1bkd, [(0, 0), (0, 0), (0, pad_amount), (0, 0)], 0.0)
        check_pos(pos_tensor_int32, "After value pad")

        shard_config = ttnn.create_sharded_memory_config(
            shape=(32, attn.head_dim),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        key_sharded = ttnn.to_memory_config(key_padded, shard_config)
        check_pos(pos_tensor_int32, "After key to_memory_config sharded")

        value_sharded = ttnn.to_memory_config(value_padded, shard_config)
        check_pos(pos_tensor_int32, "After value to_memory_config sharded")

        print("\n=== paged_update_cache ===")
        print(f"  pos_tensor buffer: {pos_tensor_int32.buffer_address()}")
        print(f"  key_cache buffer: {past_kv.key_cache.buffer_address()}")
        print(f"  key_sharded buffer: {key_sharded.buffer_address()}")

        check_pos(pos_tensor_int32, "Before paged_update_cache (key)", True)
        ttnn.experimental.paged_update_cache(
            past_kv.key_cache, key_sharded, update_idxs_tensor=pos_tensor_int32
        )
        check_pos(pos_tensor_int32, "After paged_update_cache (key)", True)

        ttnn.experimental.paged_update_cache(
            past_kv.value_cache, value_sharded, update_idxs_tensor=pos_tensor_int32
        )
        check_pos(pos_tensor_int32, "After paged_update_cache (value)", True)

        print("\nDone")


if __name__ == "__main__":
    test_layer23_detailed()
