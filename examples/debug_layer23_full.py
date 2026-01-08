"""Debug layer 23 with full forward call."""

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


def test_layer23_full():
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
        check_pos(pos_tensor_int32, "Initial")

        layer = model.layers[23]
        past_kv = caches[23]

        residual = hidden_states
        check_pos(pos_tensor_int32, "After residual assignment")

        hidden_states = layer.input_layernorm(hidden_states)
        check_pos(pos_tensor_int32, "After input_layernorm")

        attn_out, updated_cache = layer.self_attn(
            hidden_states,
            attention_mask=None,
            position_ids=current_pos,
            past_key_value=past_kv,
            use_cache=True,
            mode="decode",
            rot_mats=None,
            transformation_mat=None,
            current_pos_tensor=pos_tensor_int32,
            pos_tensor=pos_tensor_uint32,
        )
        check_pos(pos_tensor_int32, "After self_attn")

        hidden_states = ttnn.add(residual, attn_out)
        check_pos(pos_tensor_int32, "After first residual add")

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        check_pos(pos_tensor_int32, "After post_attention_layernorm")

        hidden_states = layer.mlp(hidden_states, mode="decode")
        check_pos(pos_tensor_int32, "After MLP")

        hidden_states = ttnn.add(residual, hidden_states)
        check_pos(pos_tensor_int32, "After second residual add")

        print("\nDone")


if __name__ == "__main__":
    test_layer23_full()
