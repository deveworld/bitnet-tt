"""Debug memory aliasing issue."""

import torch
import ttnn
import numpy as np

from bitnet_tt.model.bitnet import create_model
from bitnet_tt.utils.device import device_context
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.layers.attention import KVCache
from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn


def test_memory_aliasing():
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

        print(f"\nOriginal pos_tensor_int32 buffer_address: {pos_tensor_int32.buffer_address()}")
        print(f"pos_tensor value: {ttnn.to_torch(pos_tensor_int32).numpy()}")

        pos_tensor_uint32 = ttnn.from_torch(
            torch.tensor([[current_pos]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        decode_token = np.array([[7]], dtype=np.int64)
        input_ids = numpy_int_to_ttnn(decode_token, device)
        hidden_states = model.embed_tokens(input_ids)

        for layer_idx in range(22):
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

        ttnn.synchronize_device(device)
        print(f"\nAfter 22 layers:")
        print(f"  pos_tensor buffer_address: {pos_tensor_int32.buffer_address()}")
        print(f"  pos_tensor value: {ttnn.to_torch(pos_tensor_int32).numpy()}")

        layer_22_output_addr = hidden_states.buffer_address()
        print(f"\n  hidden_states buffer_address: {layer_22_output_addr}")

        layer = model.layers[22]
        past_kv = caches[22]

        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        print(f"\nAfter layer 22 input_layernorm:")
        print(f"  pos_tensor value: {ttnn.to_torch(pos_tensor_int32).numpy()}")
        print(f"  hidden_states buffer: {hidden_states.buffer_address()}")

        attn = layer.self_attn
        query = attn.q_proj(hidden_states)
        print(f"  After q_proj, pos_tensor: {ttnn.to_torch(pos_tensor_int32).numpy()}")
        print(f"  query buffer: {query.buffer_address()}")

        key = attn.k_proj(hidden_states)
        print(f"  After k_proj, pos_tensor: {ttnn.to_torch(pos_tensor_int32).numpy()}")
        print(f"  key buffer: {key.buffer_address()}")

        value = attn.v_proj(hidden_states)
        print(f"  After v_proj, pos_tensor: {ttnn.to_torch(pos_tensor_int32).numpy()}")
        print(f"  value buffer: {value.buffer_address()}")

        ttnn.synchronize_device(device)
        print(f"\nAfter synchronize: pos_tensor = {ttnn.to_torch(pos_tensor_int32).numpy()}")
        print(f"  pos_tensor buffer still at: {pos_tensor_int32.buffer_address()}")

        print("\nDone")


if __name__ == "__main__":
    test_memory_aliasing()
