#!/usr/bin/env python3
"""
Test batch-32 padding strategy for trace-compatible decode.

This script implements the tt-transformers pattern:
1. Pad batch dimension to 32 for decode
2. Use HEIGHT_SHARDED memory config throughout
3. Enable Metal Trace for 2-3x speedup

Key insight from tt-transformers (model_config.py:1506-1509):
    if batch < 32:
        zeros = torch.zeros(1, seq_len, 32, self.dim)
        zeros[:, :, :batch, :] = x
        x = zeros

This pads the batch dimension to 32, enabling HEIGHT_SHARDED operations
that work with trace without allocating new buffers.
"""

import time
import torch
import numpy as np
import ttnn

from bitnet_tt.config import BitNetConfig
from bitnet_tt.utils.device import device_context
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.model.bitnet import create_model


# Constants
TILE_SIZE = 32
PADDED_BATCH = 32  # Always pad to 32 for HEIGHT_SHARDED


def pad_batch_to_32(x: torch.Tensor, batch_dim: int = 0) -> torch.Tensor:
    """Pad tensor's batch dimension to 32."""
    shape = list(x.shape)
    batch = shape[batch_dim]
    if batch >= PADDED_BATCH:
        return x

    shape[batch_dim] = PADDED_BATCH
    padded = torch.zeros(shape, dtype=x.dtype)

    # Create slice to copy original data
    slices = [slice(None)] * len(shape)
    slices[batch_dim] = slice(0, batch)
    padded[tuple(slices)] = x
    return padded


def create_height_sharded_config(num_cores: int, shard_height: int, head_dim: int):
    """Create HEIGHT_SHARDED memory config for QKV heads."""
    # For batch=32, heads=20: total_rows = 32 * 20 = 640
    # With 32 cores (4x8 grid): each core gets 640/32 = 20 rows
    return ttnn.create_sharded_memory_config(
        shape=(shard_height, head_dim),
        core_grid=ttnn.CoreGrid(y=4, x=8),  # 32 cores
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def test_batch32_embedding_and_projection():
    """Test that batch-32 padded tensors work through embedding and projection."""
    print("\n=== Test: Batch-32 Padded Embedding and Projection ===\n")

    with device_context() as device:
        # Load model
        state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
        model = create_model(config, device)
        load_weights_to_model(model, state_dict)

        # Create single token (batch=1)
        token = torch.tensor([[1234]], dtype=torch.int64)  # [1, 1]

        # Pad to batch=32
        # Shape: [1, 1] -> [32, 1] (batch, seq)
        token_padded = pad_batch_to_32(token, batch_dim=0)
        print(f"Original token shape: {token.shape}")
        print(f"Padded token shape: {token_padded.shape}")

        # Convert to TT-NN
        token_tt = ttnn.from_torch(
            token_padded,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # Embedding
        embeds = model.embed_tokens(token_tt)
        print(f"Embeddings shape: {embeds.shape}")  # Should be [32, 1, 2560]

        layer0 = model.layers[0]
        q_proj = layer0.self_attn.q_proj

        hidden = layer0.input_layernorm(embeds)

        q = q_proj(hidden)
        print(f"Q projection shape: {q.shape}")  # Should be [32, 1, 2560]

        # Reshape for attention: [batch, seq, hidden] -> [batch, seq, heads, head_dim]
        # For batch=32: [32, 1, 2560] -> [32, 1, 20, 128]
        q_rm = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
        q_reshaped = ttnn.reshape(
            q_rm, (PADDED_BATCH, 1, config.num_attention_heads, config.head_dim)
        )
        print(f"Q reshaped: {q_reshaped.shape}")

        # Transpose to [batch, heads, seq, head_dim]: [32, 20, 1, 128]
        q_tile = ttnn.to_layout(q_reshaped, ttnn.TILE_LAYOUT)
        q_transposed = ttnn.transpose(q_tile, 1, 2)
        print(f"Q transposed: {q_transposed.shape}")

        # Now try the 1BKD format for decode: [seq, 1, batch, dim]
        # For batch=32: [1, 1, 32, 128*20] = [1, 1, 32, 2560]
        # After nlp_create_qkv_heads_decode: Q=[1, 32, 20, 128], K=[1, 32, 5, 128], V=[1, 32, 5, 128]
        print("\n--- Testing 1BKD format for decode ---")

        # Reshape hidden to [1, 1, 32, hidden_size] for nlp_create_qkv_heads_decode
        hidden_rm = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
        hidden_1bkd = ttnn.reshape(hidden_rm, (1, 1, PADDED_BATCH, config.hidden_size))
        hidden_1bkd = ttnn.to_layout(hidden_1bkd, ttnn.TILE_LAYOUT)
        print(f"Hidden 1BKD shape: {hidden_1bkd.shape}")

        attention = layer0.self_attn
        if hasattr(attention, "qkv_fused_weight") and attention.qkv_fused_weight is not None:
            print("Using fused QKV weight")
            qkv_fused = ttnn.matmul(hidden_1bkd, attention.qkv_fused_weight)
            print(f"Fused QKV shape: {qkv_fused.shape}")

            # Try nlp_create_qkv_heads_decode
            try:
                q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
                    qkv_fused,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                )
                print(f"Q heads shape: {q_heads.shape}")  # [1, 32, 20, 128]
                print(f"K heads shape: {k_heads.shape}")  # [1, 32, 5, 128]
                print(f"V heads shape: {v_heads.shape}")  # [1, 32, 5, 128]
                print("SUCCESS: nlp_create_qkv_heads_decode works with batch=32!")

                # Check if output is HEIGHT_SHARDED
                q_mem = q_heads.memory_config()
                print(f"Q memory config: {q_mem}")

                return True
            except Exception as e:
                print(f"ERROR: nlp_create_qkv_heads_decode failed: {e}")
                return False
        else:
            print("No fused QKV weight available")
            return False


def test_batch32_rope():
    """Test RoPE with batch-32 padded tensors using nlp_create_qkv_heads_decode output."""
    print("\n=== Test: Batch-32 RoPE (using nlp_create_qkv_heads_decode output) ===\n")

    with device_context() as device:
        state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
        model = create_model(config, device)
        load_weights_to_model(model, state_dict)

        head_dim = config.head_dim
        trans_mat = torch.zeros(1, 1, head_dim, head_dim, dtype=torch.bfloat16)
        for i in range(head_dim // 2):
            trans_mat[0, 0, i, i] = 1
            trans_mat[0, 0, head_dim // 2 + i, head_dim // 2 + i] = 1

        trans_mat_tt = ttnn.from_torch(
            trans_mat,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        theta = config.rope_theta
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        position = torch.tensor([0.0])
        freqs_pos = torch.outer(position, freqs)
        cos = torch.cos(freqs_pos).reshape(1, 1, 1, head_dim // 2).repeat(1, 1, 1, 2)
        sin = torch.sin(freqs_pos).reshape(1, 1, 1, head_dim // 2).repeat(1, 1, 1, 2)

        cos_tt = ttnn.from_torch(
            cos.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        sin_tt = ttnn.from_torch(
            sin.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        token_padded = torch.zeros((PADDED_BATCH, 1), dtype=torch.int64)
        token_padded[0, 0] = 1234

        token_tt = ttnn.from_torch(
            token_padded,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        embeds = model.embed_tokens(token_tt)
        layer0 = model.layers[0]
        hidden = layer0.input_layernorm(embeds)

        hidden_rm = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
        hidden_1bkd = ttnn.reshape(hidden_rm, (1, 1, PADDED_BATCH, config.hidden_size))
        hidden_1bkd = ttnn.to_layout(hidden_1bkd, ttnn.TILE_LAYOUT)

        attention = layer0.self_attn
        qkv_fused = ttnn.matmul(hidden_1bkd, attention.qkv_fused_weight)

        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv_fused,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )

        print(f"Q heads from nlp_create_qkv_heads_decode: {q_heads.shape}")
        print(f"Q memory config: {q_heads.memory_config()}")

        try:
            q_rotated = ttnn.experimental.rotary_embedding_llama(
                q_heads, cos_tt, sin_tt, trans_mat_tt, is_decode_mode=True
            )
            print(f"Q rotated shape: {q_rotated.shape}")
            print("SUCCESS: rotary_embedding_llama works with nlp_create_qkv_heads_decode output!")
            return True
        except Exception as e:
            print(f"ERROR: RoPE failed: {e}")

            print("\n--- Trying without HEIGHT_SHARDED requirement ---")
            q_interleaved = ttnn.to_memory_config(q_heads, ttnn.L1_MEMORY_CONFIG)
            print(f"Q interleaved: {q_interleaved.shape}, {q_interleaved.memory_config()}")

            try:
                q_rotated = ttnn.experimental.rotary_embedding_llama(
                    q_interleaved, cos_tt, sin_tt, trans_mat_tt, is_decode_mode=True
                )
                print(f"Q rotated (interleaved): {q_rotated.shape}")
                print("SUCCESS with L1_MEMORY_CONFIG!")
                return True
            except Exception as e2:
                print(f"ERROR: RoPE with L1 also failed: {e2}")
                return False


def test_batch32_full_decode_step():
    """Test full decode step with batch-32 padding."""
    print("\n=== Test: Full Batch-32 Decode Step ===\n")

    with device_context() as device:
        # Load model
        state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
        model = create_model(config, device)
        load_weights_to_model(model, state_dict)

        from bitnet_tt.inference.generator import TextGenerator
        from bitnet_tt.layers.attention import KVCache

        # Create generator with batch=32 (padded)
        generator = TextGenerator(model, batch_size=PADDED_BATCH, enable_trace=False)

        # Prefill with a prompt
        prompt = "Hello, world!"
        tokens = generator.tokenizer.encode(prompt)
        tokens_np = np.array([tokens], dtype=np.int64)

        # Pad batch to 32
        tokens_padded = np.zeros((PADDED_BATCH, len(tokens)), dtype=np.int64)
        tokens_padded[0, :] = tokens  # Only first batch has real data

        print(f"Prompt tokens shape: {tokens_np.shape}")
        print(f"Padded tokens shape: {tokens_padded.shape}")

        # Prefill
        kv_caches = generator._preallocate_kv_caches(use_paged=True)
        logits, kv_caches = generator.prefill_forward(tokens_padded, kv_cache=kv_caches)

        print(f"Prefill logits shape: {logits.shape}")

        # Decode step
        current_pos = len(tokens)
        next_token = np.argmax(ttnn.to_torch(logits).numpy()[0, -1, :])
        next_token_padded = np.zeros((PADDED_BATCH, 1), dtype=np.int64)
        next_token_padded[0, 0] = next_token

        print(f"\nDecode step at position {current_pos}")
        print(f"Next token (batch 0): {next_token}")

        # Time decode step
        start = time.perf_counter()
        for _ in range(10):
            logits, kv_caches = generator.decode_forward(
                next_token_padded, kv_cache=kv_caches, current_pos=current_pos
            )
            current_pos += 1
        end = time.perf_counter()

        elapsed = end - start
        tokens_generated = 10
        speed = tokens_generated / elapsed

        print(f"\n10 decode steps: {elapsed:.3f}s")
        print(f"Speed: {speed:.2f} t/s per batch element")
        print(f"Total throughput: {speed * PADDED_BATCH:.2f} t/s (32 batch)")

        # Extract real output (batch 0)
        final_logits = ttnn.to_torch(logits).numpy()[0, -1, :]
        final_token = np.argmax(final_logits)
        decoded_text = generator.tokenizer.decode([final_token])
        print(f"\nFinal token (batch 0): {final_token} = '{decoded_text}'")

        return True


def main():
    print("=" * 60)
    print("Batch-32 Padding Strategy Test")
    print("=" * 60)

    # Test 1: Basic embedding and projection
    test1_passed = test_batch32_embedding_and_projection()

    # Test 2: RoPE with HEIGHT_SHARDED
    test2_passed = test_batch32_rope()

    # Test 3: Full decode step (only if previous tests pass)
    if test1_passed and test2_passed:
        test3_passed = test_batch32_full_decode_step()
    else:
        print("\nSkipping full decode test due to earlier failures")
        test3_passed = False

    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Test 1 (Embedding/Projection): {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Test 2 (RoPE HEIGHT_SHARDED): {'PASS' if test2_passed else 'FAIL'}")
    print(f"  Test 3 (Full Decode Step): {'PASS' if test3_passed else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
