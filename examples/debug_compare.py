"""
Debug script to compare HuggingFace vs TT-NN implementation.

This script compares intermediate outputs to find where the divergence happens.
"""

import numpy as np
import torch
import ttnn

from transformers import AutoModelForCausalLM, AutoTokenizer


def compare_tensors(name: str, hf_tensor: torch.Tensor, ttnn_tensor: ttnn.Tensor) -> None:
    """Compare HuggingFace and TT-NN tensors."""
    # Convert TT-NN to numpy
    ttnn_np = ttnn.to_torch(ttnn_tensor).float().numpy()
    hf_np = hf_tensor.float().detach().numpy()

    # Compute statistics
    diff = np.abs(hf_np - ttnn_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Compute correlation
    hf_flat = hf_np.flatten()
    ttnn_flat = ttnn_np.flatten()
    if len(hf_flat) == len(ttnn_flat):
        corr = np.corrcoef(hf_flat, ttnn_flat)[0, 1]
    else:
        corr = float('nan')

    print(f"{name}:")
    print(f"  HF shape: {hf_np.shape}, TT-NN shape: {ttnn_np.shape}")
    print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
    print(f"  Correlation: {corr:.6f}")
    print(f"  HF range: [{hf_np.min():.4f}, {hf_np.max():.4f}]")
    print(f"  TT-NN range: [{ttnn_np.min():.4f}, {ttnn_np.max():.4f}]")
    print()


def main():
    print("=" * 60)
    print("Comparing HuggingFace vs TT-NN BitNet Implementation")
    print("=" * 60)

    # Load HuggingFace model
    print("\n[1] Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/bitnet-b1.58-2B-4T-bf16",
        torch_dtype=torch.float32,  # Use float32 for comparison
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")
    hf_model.eval()

    # Prepare input
    text = "Hello"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Input: '{text}'")
    print(f"Token IDs: {input_ids.tolist()}")

    # Open TT-NN device
    print("\n[2] Opening TT-NN device...")
    device = ttnn.open_device(device_id=0)

    try:
        # Get HuggingFace embeddings
        print("\n[3] Comparing embeddings...")
        with torch.no_grad():
            hf_embeddings = hf_model.model.embed_tokens(input_ids)
        print(f"HF embeddings shape: {hf_embeddings.shape}")

        # Get TT-NN embeddings
        from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn, numpy_to_ttnn
        from bitnet_tt.layers.embedding import Embedding

        # Load embedding weights
        embed_weight = hf_model.model.embed_tokens.weight.detach().numpy()
        ttnn_embed = Embedding(
            vocab_size=hf_model.config.vocab_size,
            embedding_dim=hf_model.config.hidden_size,
            device=device,
        )
        ttnn_embed.load_weights(embed_weight)

        # Get TT-NN embeddings
        ttnn_input_ids = numpy_int_to_ttnn(input_ids.numpy(), device)
        ttnn_embeddings = ttnn_embed(ttnn_input_ids)

        compare_tensors("Embeddings", hf_embeddings, ttnn_embeddings)

        # Compare first layer input_layernorm
        print("[4] Comparing first layer input_layernorm...")
        with torch.no_grad():
            hf_normed = hf_model.model.layers[0].input_layernorm(hf_embeddings)

        from bitnet_tt.layers.bitlinear import RMSNorm
        ttnn_norm = RMSNorm(
            hidden_size=hf_model.config.hidden_size,
            device=device,
            eps=hf_model.config.rms_norm_eps,
        )
        ttnn_norm.load_weights(
            hf_model.model.layers[0].input_layernorm.weight.detach().numpy()
        )
        ttnn_normed = ttnn_norm(ttnn_embeddings)

        compare_tensors("Layer 0 input_layernorm", hf_normed, ttnn_normed)

        # Compare Q projection with weight quantization (matching HF BitLinear)
        print("[5] Comparing Q projection (with BitLinear quantization)...")
        q_weight = hf_model.model.layers[0].self_attn.q_proj.weight.detach()

        # Manually apply HuggingFace BitLinear weight_quant formula:
        # s = 1.0 / weight.abs().mean()
        # result = (weight * s).round().clamp(-1, 1) / s
        def weight_quant_hf(weight: torch.Tensor) -> torch.Tensor:
            s = 1.0 / weight.abs().mean().clamp_(min=1e-5)
            return (weight * s).round().clamp_(-1, 1) / s

        with torch.no_grad():
            q_weight_quant = weight_quant_hf(q_weight)
            hf_q = torch.nn.functional.linear(hf_normed, q_weight_quant)

        from bitnet_tt.layers.bitlinear import Linear
        ttnn_q_proj = Linear(
            in_features=hf_model.config.hidden_size,
            out_features=hf_model.config.hidden_size,
            device=device,
        )
        # Load weights - our Linear applies the same quantization as HF BitLinear
        ttnn_q_proj.load_weights(q_weight.numpy())
        ttnn_q = ttnn_q_proj(ttnn_normed)

        compare_tensors("Layer 0 Q projection", hf_q, ttnn_q)

        # Compare full attention output (without RoPE for now)
        print("[6] Checking attention output shape and basic stats...")
        print(f"Q shape: HF={hf_q.shape}, TT-NN={ttnn_q.shape}")

        # Test reshape and permute with layout conversion
        batch_size = 1
        seq_len = input_ids.shape[1]
        num_heads = 20
        head_dim = 128

        # Convert to ROW_MAJOR for reshape/permute (matching the fixed attention.py)
        ttnn_q_row = ttnn.to_layout(ttnn_q, ttnn.ROW_MAJOR_LAYOUT)
        ttnn_q_reshaped = ttnn.reshape(ttnn_q_row, (batch_size, seq_len, num_heads, head_dim))
        ttnn_q_permuted = ttnn.permute(ttnn_q_reshaped, (0, 2, 1, 3))
        # Convert back to TILE_LAYOUT
        ttnn_q_permuted = ttnn.to_layout(ttnn_q_permuted, ttnn.TILE_LAYOUT)

        print(f"TT-NN Q after reshape: {ttnn_q_reshaped.shape}")
        print(f"TT-NN Q after permute: {ttnn_q_permuted.shape}")

        # Compare with PyTorch
        hf_q_reshaped = hf_q.view(batch_size, seq_len, num_heads, head_dim)
        hf_q_permuted = hf_q_reshaped.permute(0, 2, 1, 3)
        print(f"HF Q after reshape: {hf_q_reshaped.shape}")
        print(f"HF Q after permute: {hf_q_permuted.shape}")

        compare_tensors("Q after permute", hf_q_permuted, ttnn_q_permuted)

        # Test RoPE slicing
        print("\n[7] Testing RoPE slicing operations...")

        # Create test cos/sin tensors
        head_dim = 128
        max_seq = 10
        rope_theta = 500000.0

        # Compute frequencies like in attention.py
        freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        t = np.arange(max_seq, dtype=np.float32)
        freqs_outer = np.outer(t, freqs)
        emb = np.concatenate([freqs_outer, freqs_outer], axis=-1)
        cos_np = np.cos(emb).astype(np.float32).reshape(1, 1, max_seq, head_dim)
        sin_np = np.sin(emb).astype(np.float32).reshape(1, 1, max_seq, head_dim)

        print(f"cos_np shape: {cos_np.shape}")

        # Test slicing in TT-NN
        cos_ttnn = numpy_to_ttnn(cos_np, device)
        print(f"cos_ttnn shape: {cos_ttnn.shape}")

        # Slice to seq_len
        cos_sliced = cos_ttnn[:, :, :seq_len, :]
        print(f"cos_sliced shape: {cos_sliced.shape}")

        # Compare with numpy slicing
        cos_np_sliced = cos_np[:, :, :seq_len, :]
        print(f"cos_np_sliced shape: {cos_np_sliced.shape}")

        cos_sliced_np = ttnn.to_torch(cos_sliced).float().numpy()
        slice_diff = np.abs(cos_np_sliced - cos_sliced_np)
        print(f"Slicing max diff: {slice_diff.max():.6f}")
        print(f"Slicing mean diff: {slice_diff.mean():.6f}")

        # Test tensor splitting (for rotate_half)
        print("\n[8] Testing tensor splitting for rotate_half...")
        # Use Q tensor for testing
        half_dim = head_dim // 2

        # TT-NN slicing
        ttnn_x1 = ttnn_q_permuted[:, :, :, :half_dim]
        ttnn_x2 = ttnn_q_permuted[:, :, :, half_dim:]
        print(f"ttnn_x1 shape: {ttnn_x1.shape}")
        print(f"ttnn_x2 shape: {ttnn_x2.shape}")

        # PyTorch slicing
        hf_x1 = hf_q_permuted[:, :, :, :half_dim]
        hf_x2 = hf_q_permuted[:, :, :, half_dim:]
        print(f"hf_x1 shape: {hf_x1.shape}")
        print(f"hf_x2 shape: {hf_x2.shape}")

        # Compare
        compare_tensors("x1 (first half)", hf_x1, ttnn_x1)
        compare_tensors("x2 (second half)", hf_x2, ttnn_x2)

        # Test concat
        print("[9] Testing concat...")
        ttnn_x2_neg = ttnn.neg(ttnn_x2)
        ttnn_rotated = ttnn.concat([ttnn_x2_neg, ttnn_x1], dim=-1)
        print(f"ttnn_rotated shape: {ttnn_rotated.shape}")

        hf_rotated = torch.cat([-hf_x2, hf_x1], dim=-1)
        print(f"hf_rotated shape: {hf_rotated.shape}")

        compare_tensors("rotate_half result", hf_rotated, ttnn_rotated)

        # Test full RoPE application
        print("\n[10] Testing full RoPE application...")

        # Get cos/sin for current positions
        position_ids = np.arange(seq_len, dtype=np.int64)
        cos_pos = cos_np[:, :, position_ids, :]
        sin_pos = sin_np[:, :, position_ids, :]

        cos_ttnn_pos = numpy_to_ttnn(cos_pos.astype(np.float32), device)
        sin_ttnn_pos = numpy_to_ttnn(sin_pos.astype(np.float32), device)

        # Expand cos/sin to match Q shape
        cos_expanded = ttnn.repeat(cos_ttnn_pos, ttnn.Shape([1, num_heads, 1, 1]))
        sin_expanded = ttnn.repeat(sin_ttnn_pos, ttnn.Shape([1, num_heads, 1, 1]))

        # Apply RoPE in TT-NN
        ttnn_q_cos = ttnn.multiply(ttnn_q_permuted, cos_expanded)
        ttnn_q_rot_sin = ttnn.multiply(ttnn_rotated, sin_expanded)
        ttnn_q_rope = ttnn.add(ttnn_q_cos, ttnn_q_rot_sin)

        # Apply RoPE in PyTorch
        cos_torch = torch.from_numpy(cos_pos).expand(1, num_heads, seq_len, head_dim)
        sin_torch = torch.from_numpy(sin_pos).expand(1, num_heads, seq_len, head_dim)
        hf_q_cos = hf_q_permuted * cos_torch
        hf_q_rot_sin = hf_rotated * sin_torch
        hf_q_rope = hf_q_cos + hf_q_rot_sin

        compare_tensors("Q after RoPE", hf_q_rope, ttnn_q_rope)

        print("=" * 60)
        print("Debug comparison complete!")
        print("=" * 60)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
