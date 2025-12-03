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

        # Compare Q projection
        print("[5] Comparing Q projection...")
        with torch.no_grad():
            hf_q = hf_model.model.layers[0].self_attn.q_proj(hf_normed)

        from bitnet_tt.layers.bitlinear import Linear
        ttnn_q_proj = Linear(
            in_features=hf_model.config.hidden_size,
            out_features=hf_model.config.hidden_size,
            device=device,
        )
        ttnn_q_proj.load_weights(
            hf_model.model.layers[0].self_attn.q_proj.weight.detach().numpy()
        )
        ttnn_q = ttnn_q_proj(ttnn_normed)

        compare_tensors("Layer 0 Q projection", hf_q, ttnn_q)

        print("=" * 60)
        print("Debug comparison complete!")
        print("=" * 60)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
