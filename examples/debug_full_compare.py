"""
Full end-to-end comparison between HuggingFace and TT-NN BitNet implementation.

This script compares:
1. Final logits from both implementations
2. Top-k token predictions
3. Generated text sequences
"""

import numpy as np
import torch

# Disable torch dynamo/compile to avoid inductor issues
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import ttnn

from transformers import AutoModelForCausalLM, AutoTokenizer


def compare_logits(name: str, hf_logits: torch.Tensor, ttnn_logits: ttnn.Tensor) -> dict:
    """Compare logits and return statistics."""
    ttnn_np = ttnn.to_torch(ttnn_logits).float().numpy()
    hf_np = hf_logits.float().detach().numpy()

    # Compute statistics
    diff = np.abs(hf_np - ttnn_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Compute correlation
    hf_flat = hf_np.flatten()
    ttnn_flat = ttnn_np.flatten()
    corr = np.corrcoef(hf_flat, ttnn_flat)[0, 1]

    print(f"\n{name}:")
    print(f"  Shape: HF={hf_np.shape}, TT-NN={ttnn_np.shape}")
    print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
    print(f"  Correlation: {corr:.6f}")
    print(f"  HF range: [{hf_np.min():.4f}, {hf_np.max():.4f}]")
    print(f"  TT-NN range: [{ttnn_np.min():.4f}, {ttnn_np.max():.4f}]")

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "correlation": corr,
        "hf_logits": hf_np,
        "ttnn_logits": ttnn_np,
    }


def compare_top_k_predictions(hf_logits: np.ndarray, ttnn_logits: np.ndarray, tokenizer, k: int = 10):
    """Compare top-k token predictions."""
    # Get last token logits
    hf_last = hf_logits[0, -1, :]  # (vocab_size,)
    ttnn_last = ttnn_logits[0, -1, :]  # (vocab_size,)

    # Get top-k indices
    hf_top_k_idx = np.argsort(hf_last)[-k:][::-1]
    ttnn_top_k_idx = np.argsort(ttnn_last)[-k:][::-1]

    print(f"\n  Top-{k} predictions comparison:")
    print(f"  {'Rank':<6} {'HF Token':<20} {'HF Logit':<12} {'TT-NN Token':<20} {'TT-NN Logit':<12} {'Match'}")
    print("  " + "-" * 90)

    matches = 0
    for i in range(k):
        hf_idx = hf_top_k_idx[i]
        ttnn_idx = ttnn_top_k_idx[i]

        hf_token = tokenizer.decode([hf_idx])
        ttnn_token = tokenizer.decode([ttnn_idx])

        match = "Yes" if hf_idx == ttnn_idx else ""
        if hf_idx == ttnn_idx:
            matches += 1

        print(f"  {i+1:<6} {repr(hf_token):<20} {hf_last[hf_idx]:<12.4f} {repr(ttnn_token):<20} {ttnn_last[ttnn_idx]:<12.4f} {match}")

    print(f"\n  Top-{k} match rate: {matches}/{k} ({100*matches/k:.1f}%)")

    # Check if top-1 matches
    if hf_top_k_idx[0] == ttnn_top_k_idx[0]:
        print(f"  Top-1 prediction MATCHES: {repr(tokenizer.decode([hf_top_k_idx[0]]))}")
    else:
        print(f"  Top-1 prediction DIFFERS:")
        print(f"    HF:    {repr(tokenizer.decode([hf_top_k_idx[0]]))} (logit: {hf_last[hf_top_k_idx[0]]:.4f})")
        print(f"    TT-NN: {repr(tokenizer.decode([ttnn_top_k_idx[0]]))} (logit: {ttnn_last[ttnn_top_k_idx[0]]:.4f})")


def generate_and_compare(prompt: str, hf_model, ttnn_model, tokenizer, device, max_tokens: int = 20):
    """Generate tokens from both models and compare."""
    from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn, ttnn_to_numpy

    print(f"\n  Generating {max_tokens} tokens from prompt: {repr(prompt)}")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Generate with HuggingFace (greedy)
    with torch.no_grad():
        hf_generated = hf_model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    hf_text = tokenizer.decode(hf_generated[0], skip_special_tokens=True)

    # Generate with TT-NN (greedy)
    ttnn_input_ids = input_ids.numpy().copy()
    for _ in range(max_tokens):
        ttnn_input = numpy_int_to_ttnn(ttnn_input_ids, device)
        logits, _ = ttnn_model(ttnn_input, use_cache=False)
        logits_np = ttnn_to_numpy(logits)
        next_token = np.argmax(logits_np[0, -1, :])
        ttnn_input_ids = np.concatenate([ttnn_input_ids, [[next_token]]], axis=1)

        # Check for EOS
        if tokenizer.eos_token_id and next_token == tokenizer.eos_token_id:
            break

    ttnn_text = tokenizer.decode(ttnn_input_ids[0], skip_special_tokens=True)

    print(f"\n  HuggingFace output:")
    print(f"    {repr(hf_text)}")
    print(f"\n  TT-NN output:")
    print(f"    {repr(ttnn_text)}")

    # Compare token by token
    hf_tokens = hf_generated[0].tolist()
    ttnn_tokens = ttnn_input_ids[0].tolist()

    print(f"\n  Token comparison (first {min(len(hf_tokens), len(ttnn_tokens))} tokens):")
    print(f"  {'Pos':<5} {'HF Token':<10} {'HF Text':<15} {'TT-NN Token':<12} {'TT-NN Text':<15} {'Match'}")
    print("  " + "-" * 75)

    matches = 0
    for i in range(min(len(hf_tokens), len(ttnn_tokens))):
        hf_tok = hf_tokens[i]
        ttnn_tok = ttnn_tokens[i]
        match = "Yes" if hf_tok == ttnn_tok else ""
        if hf_tok == ttnn_tok:
            matches += 1

        hf_decoded = repr(tokenizer.decode([hf_tok]))
        ttnn_decoded = repr(tokenizer.decode([ttnn_tok]))

        print(f"  {i:<5} {hf_tok:<10} {hf_decoded:<15} {ttnn_tok:<12} {ttnn_decoded:<15} {match}")

    total = min(len(hf_tokens), len(ttnn_tokens))
    print(f"\n  Token match rate: {matches}/{total} ({100*matches/total:.1f}%)")

    return hf_text, ttnn_text


def main():
    print("=" * 70)
    print("Full End-to-End Comparison: HuggingFace vs TT-NN BitNet")
    print("=" * 70)

    # Load HuggingFace model
    print("\n[1] Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/bitnet-b1.58-2B-4T-bf16",
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")
    hf_model.eval()
    print(f"    Model loaded: {hf_model.config.hidden_size}d, {hf_model.config.num_hidden_layers}L")

    # Open TT-NN device and load model
    print("\n[2] Loading TT-NN model...")
    device = ttnn.open_device(device_id=0)

    try:
        from bitnet_tt.model.bitnet import create_model
        from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
        from bitnet_tt.layers.bitlinear import numpy_int_to_ttnn, ttnn_to_numpy

        state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
        ttnn_model = create_model(config, device)
        load_weights_to_model(ttnn_model, state_dict)
        print(f"    Model loaded: {config.hidden_size}d, {config.num_layers}L")

        # Test prompts
        test_prompts = [
            "Hello",
            "The capital of France is",
            "1 + 1 =",
        ]

        for prompt in test_prompts:
            print("\n" + "=" * 70)
            print(f"Testing prompt: {repr(prompt)}")
            print("=" * 70)

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            print(f"Token IDs: {input_ids.tolist()}")

            # Get HuggingFace logits
            print("\n[3] Computing logits...")
            with torch.no_grad():
                hf_outputs = hf_model(input_ids)
                hf_logits = hf_outputs.logits

            # Get TT-NN logits
            ttnn_input = numpy_int_to_ttnn(input_ids.numpy(), device)
            ttnn_logits, _ = ttnn_model(ttnn_input, use_cache=False)

            # Compare logits
            stats = compare_logits("Final Logits", hf_logits, ttnn_logits)

            # Compare top-k predictions
            compare_top_k_predictions(
                stats["hf_logits"],
                stats["ttnn_logits"],
                tokenizer,
                k=10
            )

        # Full generation comparison
        print("\n" + "=" * 70)
        print("Full Generation Comparison")
        print("=" * 70)

        generate_and_compare(
            "Hello, I am",
            hf_model,
            ttnn_model,
            tokenizer,
            device,
            max_tokens=15
        )

        print("\n" + "=" * 70)
        print("Comparison complete!")
        print("=" * 70)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
