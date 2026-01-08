"""Debug the update_decode_paged method directly."""

import torch
import ttnn
import numpy as np

from bitnet_tt.model.bitnet import create_model
from bitnet_tt.utils.device import device_context
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.layers.attention import KVCache
from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn


def test_update_method():
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

        print(f"\nAfter 23 layers, before calling update_decode_paged directly:")
        ttnn.synchronize_device(device)
        print(f"  pos_tensor = {ttnn.to_torch(pos_tensor_int32).numpy()[0]}")

        past_kv = caches[23]

        key = torch.randn(1, 5, 1, 128, dtype=torch.bfloat16)
        value = torch.randn(1, 5, 1, 128, dtype=torch.bfloat16)

        key_tt = ttnn.from_torch(key, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        value_tt = ttnn.from_torch(
            value, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

        print(f"\nCalling update_decode_paged on cache 23:")
        print(f"  Before: pos_tensor = {ttnn.to_torch(pos_tensor_int32).numpy()[0]}")

        k_exp, v_exp = past_kv.update_decode_paged(
            key_tt, value_tt, current_pos, pos_tensor_int32, num_kv_groups=4
        )

        ttnn.synchronize_device(device)
        print(f"  After: pos_tensor = {ttnn.to_torch(pos_tensor_int32).numpy()[0]}")

        print("\nDone")


if __name__ == "__main__":
    test_update_method()
