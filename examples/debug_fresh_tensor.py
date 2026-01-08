"""Debug using fresh pos tensor per layer."""

import torch
import ttnn
import numpy as np

from bitnet_tt.model.bitnet import create_model
from bitnet_tt.utils.device import device_context
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.layers.attention import KVCache
from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn


def test_fresh_tensor():
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

        decode_token = np.array([[7]], dtype=np.int64)
        input_ids = numpy_int_to_ttnn(decode_token, device)
        hidden_states = model.embed_tokens(input_ids)
        ttnn.synchronize_device(device)

        print(f"\nRunning decode with FRESH pos tensor each layer:")

        for layer_idx in range(30):
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

            val = ttnn.to_torch(pos_tensor_int32).numpy()[0]
            if val != 6:
                print(f"  Layer {layer_idx}: CHANGED to {val}")
            elif layer_idx % 10 == 0:
                print(f"  Layer {layer_idx}: OK ({val})")

            ttnn.deallocate(pos_tensor_int32)
            ttnn.deallocate(pos_tensor_uint32)

        print("\nNow running model norm + lm_head...")
        hidden_states = model.norm(hidden_states)
        logits = ttnn.matmul(hidden_states, model.lm_head_weight)

        ttnn.synchronize_device(device)
        logits_np = ttnn.to_torch(logits).float().numpy()
        print(f"  Logits shape: {logits_np.shape}")
        print(f"  Logits sample: {logits_np[0, 0, :5]}")

        next_token = np.argmax(logits_np[0, 0, :])
        print(f"  Next token: {next_token}")

        print("\nDone")


if __name__ == "__main__":
    test_fresh_tensor()
