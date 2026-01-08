"""Debug which layer modifies the position tensor."""

import torch
import ttnn
import numpy as np

from bitnet_tt.model.bitnet import create_model
from bitnet_tt.utils.device import device_context
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.layers.attention import KVCache
from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn


def test_layer_by_layer():
    with device_context() as device:
        print("Loading BitNet model...")
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

        print(f"\nStarting decode, pos_tensor_int32 = {ttnn.to_torch(pos_tensor_int32).numpy()}")

        input_ids = numpy_int_to_ttnn(decode_token, device)
        hidden_states = model.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(model.layers):
            past_kv = caches[layer_idx]

            val_before = ttnn.to_torch(pos_tensor_int32).numpy()[0]

            hidden_states, updated_cache = layer(
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

            ttnn.synchronize_device(device)
            val_after = ttnn.to_torch(pos_tensor_int32).numpy()[0]

            if val_before != val_after:
                print(f"LAYER {layer_idx}: pos_tensor CHANGED from {val_before} to {val_after}")
                break
            else:
                if layer_idx % 10 == 0:
                    print(f"Layer {layer_idx}: OK (pos_tensor = {val_after})")

            caches[layer_idx] = updated_cache

        final_val = ttnn.to_torch(pos_tensor_int32).numpy()
        print(f"\nFinal pos_tensor_int32 = {final_val}")

        ttnn.deallocate(input_ids)
        ttnn.deallocate(hidden_states)


if __name__ == "__main__":
    test_layer_by_layer()
