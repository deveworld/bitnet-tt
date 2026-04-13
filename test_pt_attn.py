#!/usr/bin/env python3
"""Find where bfp4 and packed_ternary diverge in layer 0."""
import numpy as np
import ttnn
from bitnet_tt.utils.device import get_device, close_device
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.model.bitnet import create_model
from bitnet_tt.layers.bitlinear import ttnn_to_numpy, numpy_int_to_ttnn

device = get_device()
state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")

models = {}
for dtype in ["bfp4", "packed_ternary"]:
    m = create_model(config, device, weight_dtype=dtype, use_fused_rope=False)
    load_weights_to_model(m, state_dict)
    models[dtype] = m
print("Both models loaded", flush=True)

# Check attention layer structure
attn = models["packed_ternary"].layers[0].self_attn
print(f"\nAttention projections:", flush=True)
print(f"  qkv: dtype={attn.qkv_proj._weight_dtype_name} packed={attn.qkv_proj._use_packed_ternary} scale={attn.qkv_proj.weight_scale:.6f}", flush=True)
print(f"  o:   dtype={attn.o_proj._weight_dtype_name} packed={attn.o_proj._use_packed_ternary} scale={attn.o_proj.weight_scale:.6f}", flush=True)

# Compare scales
attn_bfp4 = models["bfp4"].layers[0].self_attn
print(f"\nbfp4 attention:", flush=True)
print(f"  qkv: dtype={attn_bfp4.qkv_proj._weight_dtype_name} scale={attn_bfp4.qkv_proj.weight_scale:.6f}", flush=True)
print(f"  o:   dtype={attn_bfp4.o_proj._weight_dtype_name} scale={attn_bfp4.o_proj.weight_scale:.6f}", flush=True)

# Run prefill through attention only
input_ids = np.array([[128000, 791, 6864, 315, 9822, 374]], dtype=np.int32)
ids_t = numpy_int_to_ttnn(input_ids, device)
pos = np.arange(input_ids.shape[1])

results = {}
for dtype, model in models.items():
    emb = model.embed_tokens(ids_t)
    x_norm = model.layers[0].input_layernorm(emb)
    x_norm_np = ttnn_to_numpy(x_norm)

    # Attention forward
    attn_layer = model.layers[0].self_attn
    attn_out, _ = attn_layer(x_norm, position_ids=pos, mode="prefill")
    attn_np = ttnn_to_numpy(attn_out)

    # Residual
    residual = ttnn_to_numpy(emb)
    post_attn = residual + attn_np

    # Post-attention layernorm
    post_attn_ttnn = ttnn.from_torch(
        __import__('torch').from_numpy(post_attn.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pan = model.layers[0].post_attention_layernorm(post_attn_ttnn)
    pan_np = ttnn_to_numpy(pan)

    # FFN
    ffn = model.layers[0].mlp
    ffn_out = ffn(pan)
    ffn_np = ttnn_to_numpy(ffn_out)

    results[dtype] = {
        "input_norm": x_norm_np,
        "attn_out": attn_np,
        "post_attn": post_attn,
        "pan_norm": pan_np,
        "ffn_out": ffn_np,
    }
    print(f"\n{dtype}:", flush=True)
    print(f"  input_norm: abs_mean={np.abs(x_norm_np).mean():.6f}", flush=True)
    print(f"  attn_out:   abs_mean={np.abs(attn_np).mean():.6f}", flush=True)
    print(f"  post_attn:  abs_mean={np.abs(post_attn).mean():.6f}", flush=True)
    print(f"  pan_norm:   abs_mean={np.abs(pan_np).mean():.6f}", flush=True)
    print(f"  ffn_out:    abs_mean={np.abs(ffn_np).mean():.6f}", flush=True)

# Compare
print("\n=== Comparison ===", flush=True)
for key in ["input_norm", "attn_out", "pan_norm", "ffn_out"]:
    a = results["bfp4"][key].flatten()
    b = results["packed_ternary"][key].flatten()
    sz = min(len(a), len(b))
    corr = np.corrcoef(a[:sz], b[:sz])[0, 1]
    ratio = np.abs(b).mean() / np.abs(a).mean() if np.abs(a).mean() > 0 else 0
    print(f"  {key:15s}: corr={corr:.6f} ratio={ratio:.4f}", flush=True)

close_device()
print("DONE", flush=True)
