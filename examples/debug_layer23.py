"""Debug layer 23 to find which op modifies pos tensor."""

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


def test_layer23():
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

        decode_token = np.array([[7]], dtype=np.int64)
        current_pos = 6

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
        check_pos(pos_tensor_int32, "Initial")

        layer = model.layers[23]
        past_kv = caches[23]

        residual = hidden_states
        check_pos(pos_tensor_int32, "After residual assignment")

        hidden_states = layer.input_layernorm(hidden_states)
        check_pos(pos_tensor_int32, "After input_layernorm")

        print("\n  === ATTENTION START ===")
        attn = layer.self_attn
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

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
        key = ttnn.reshape(key, (batch_size, seq_len, attn.num_kv_heads, attn.head_dim))
        value = ttnn.reshape(value, (batch_size, seq_len, attn.num_kv_heads, attn.head_dim))
        check_pos(pos_tensor_int32, "After reshape")

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        check_pos(pos_tensor_int32, "After to_layout TILE")

        query = ttnn.transpose(query, 1, 2)
        key = ttnn.transpose(key, 1, 2)
        value = ttnn.transpose(value, 1, 2)
        check_pos(pos_tensor_int32, "After transpose")

        query, key = attn._apply_rope_manual(query, key, current_pos, seq_len, pos_tensor_uint32)
        check_pos(pos_tensor_int32, "After RoPE")

        print("\n  === KV CACHE UPDATE ===")
        key_exp, value_exp = past_kv.update_decode_paged(
            key, value, current_pos, pos_tensor_int32, attn.num_kv_groups
        )
        check_pos(pos_tensor_int32, "After update_decode_paged")

        print("\n  === SDPA ===")
        q_rm = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
        q_1bkd = ttnn.permute(q_rm, (2, 0, 1, 3))
        q_1bkd = ttnn.to_layout(q_1bkd, ttnn.TILE_LAYOUT)
        check_pos(pos_tensor_int32, "After Q permute to 1BKD")

        attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
            q_1bkd,
            key_exp,
            value_exp,
            cur_pos_tensor=pos_tensor_int32,
            scale=attn.scale,
        )
        check_pos(pos_tensor_int32, "After SDPA decode")

        print("\nDone with layer 23 attention")


if __name__ == "__main__":
    test_layer23()
