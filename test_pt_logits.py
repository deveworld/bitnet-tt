#!/usr/bin/env python3
"""Compare prefill logits between bfp4 and packed_ternary."""
import numpy as np
import ttnn
from bitnet_tt.utils.device import get_device, close_device
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.model.bitnet import create_model
from bitnet_tt.layers.bitlinear import ttnn_to_numpy, numpy_int_to_ttnn
from transformers import AutoTokenizer

device = get_device()
state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")

prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
print(f"Input IDs: {input_ids}", flush=True)

for dtype in ["bfp4", "packed_ternary"]:
    print(f"\n=== {dtype} ===", flush=True)
    model = create_model(config, device, weight_dtype=dtype, use_fused_rope=False)
    load_weights_to_model(model, state_dict)

    ids_t = numpy_int_to_ttnn(input_ids, device)

    # Run prefill (all layers, get logits)
    output, _ = model(
        input_ids=ids_t,
        position_ids=np.arange(input_ids.shape[1]),
        mode="prefill",
    )

    logits = ttnn_to_numpy(output)
    print(f"Logits shape: {logits.shape}", flush=True)

    # Last token logits (prediction for next token)
    last_logits = logits[0, input_ids.shape[1]-1, :]  # [vocab_size]
    top5_idx = np.argsort(last_logits)[-5:][::-1]

    print("Top-5 predictions:", flush=True)
    for idx in top5_idx:
        token = tokenizer.decode([idx])
        print(f"  {token!r:15s} logit={last_logits[idx]:.2f}", flush=True)

    # Hidden state after first layer
    ids_t2 = numpy_int_to_ttnn(input_ids, device)
    emb = model.embed_tokens(ids_t2)
    x = emb
    layer0_out, _ = model.layers[0](
        x, position_ids=np.arange(input_ids.shape[1]), mode="prefill"
    )
    l0 = ttnn_to_numpy(layer0_out)
    print(f"Layer 0 output: shape={l0.shape} mean={l0.mean():.6f} abs_mean={np.abs(l0).mean():.6f}", flush=True)

    del model

close_device()
print("\nDONE", flush=True)
