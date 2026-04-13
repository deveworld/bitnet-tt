#!/usr/bin/env python3
"""Trace FFN stages to find divergence between bfp4 and packed_ternary."""
import numpy as np
import torch
import ttnn
from bitnet_tt.utils.device import get_device, close_device
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.model.bitnet import create_model
from bitnet_tt.layers.bitlinear import ttnn_to_numpy, numpy_int_to_ttnn


def comp(name, a, b):
    a_np = ttnn_to_numpy(a) if not isinstance(a, np.ndarray) else a
    b_np = ttnn_to_numpy(b) if not isinstance(b, np.ndarray) else b
    sz = min(a_np.size, b_np.size)
    af, bf = a_np.flatten()[:sz], b_np.flatten()[:sz]
    corr = np.corrcoef(af, bf)[0, 1] if sz > 1 else 0
    m1, m2 = np.abs(a_np).mean(), np.abs(b_np).mean()
    ratio = m2 / m1 if m1 > 0 else 0
    print(f"  {name:25s}: corr={corr:.6f} bfp4_mean={m1:.4f} pt_mean={m2:.4f} ratio={ratio:.4f}")


device = get_device()
state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")

m1 = create_model(config, device, weight_dtype="bfp4", use_fused_rope=False)
load_weights_to_model(m1, state_dict)
m2 = create_model(config, device, weight_dtype="packed_ternary", use_fused_rope=False)
load_weights_to_model(m2, state_dict)
print("Models loaded\n")

# Prepare input
input_ids = np.array([[128000, 791, 6864, 315, 9822, 374]], dtype=np.int32)
ids_t = numpy_int_to_ttnn(input_ids, device)
emb = m1.embed_tokens(ids_t)  # same for both

# Run through attention (same for both - all bfp4)
attn_norm = m1.layers[0].input_layernorm(emb)
attn_out1, _ = m1.layers[0].self_attn(attn_norm, position_ids=np.arange(6), mode="prefill")
# Use bfp4's attention output for BOTH to isolate FFN differences
residual_np = ttnn_to_numpy(emb)
attn_np = ttnn_to_numpy(attn_out1)
post_attn = residual_np + attn_np

# Make a fresh tensor for both FFNs
post_attn_t = ttnn.from_torch(
    torch.from_numpy(post_attn.astype(np.float32)),
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Post-attention layernorm (both models use same weights here)
pan1 = m1.layers[0].post_attention_layernorm(post_attn_t)
pan2 = m2.layers[0].post_attention_layernorm(post_attn_t)
comp("post_attn_norm", pan1, pan2)

# Now trace FFN step by step
ffn1 = m1.layers[0].mlp
ffn2 = m2.layers[0].mlp

# Use SAME input for both FFNs
x = pan1  # same normalized input

# Step 1: gate_up_proj
gu1 = ffn1.gate_up_proj(x)
gu2 = ffn2.gate_up_proj(x)
print(f"\ngate_up output shapes: bfp4={list(gu1.shape)} pt={list(gu2.shape)}")
# skip padded shape check
comp("gate_up_proj", gu1, gu2)

# Step 2: Split
int_sz = ffn1.intermediate_size  # 6912
s = [0] * len(gu1.shape)
e = list(gu1.shape)
e[-1] = int_sz
gate1 = ttnn.slice(gu1, s, e)
gate2 = ttnn.slice(gu2, s, e)

s2 = [0] * len(gu1.shape)
s2[-1] = int_sz
e2 = list(gu1.shape)
up1 = ttnn.slice(gu1, s2, e2)
up2 = ttnn.slice(gu2, s2, e2)
comp("gate (split)", gate1, gate2)
comp("up (split)", up1, up2)

# Step 3: Apply scales
gate1s = ttnn.multiply(gate1, ffn1._gate_scale)
gate2s = ttnn.multiply(gate2, ffn2._gate_scale)
up1s = ttnn.multiply(up1, ffn1._up_scale)
up2s = ttnn.multiply(up2, ffn2._up_scale)
print(f"\nScales: bfp4 gate={ffn1._gate_scale:.6f} up={ffn1._up_scale:.6f}")
print(f"Scales: pt   gate={ffn2._gate_scale:.6f} up={ffn2._up_scale:.6f}")
comp("gate_scaled", gate1s, gate2s)
comp("up_scaled", up1s, up2s)

# Step 4: Squared ReLU + multiply
gate1r = ttnn.relu(gate1s)
gate1r = ttnn.multiply(gate1r, gate1r)
gate2r = ttnn.relu(gate2s)
gate2r = ttnn.multiply(gate2r, gate2r)

hidden1 = ttnn.multiply(gate1r, up1s)
hidden2 = ttnn.multiply(gate2r, up2s)
comp("hidden (gate*up)", hidden1, hidden2)

# Step 5: Sub norm
hidden1n = ffn1.ffn_sub_norm(hidden1)
hidden2n = ffn2.ffn_sub_norm(hidden2)
comp("after_sub_norm", hidden1n, hidden2n)

# Step 6: down_proj
down1 = ffn1.down_proj(hidden1n)
down2 = ffn2.down_proj(hidden1n)  # Use SAME input for both!
print(f"\ndown_proj: bfp4 dtype={ffn1.down_proj._weight_dtype_name} pt dtype={ffn2.down_proj._weight_dtype_name}")
print(f"down_proj: bfp4 scale={ffn1.down_proj.weight_scale:.6f} pt scale={ffn2.down_proj.weight_scale:.6f}")
comp("down_proj(same_in)", down1, down2)

# Also test down_proj with its own input
down2_own = ffn2.down_proj(hidden2n)
comp("down_proj(own_in)", down1, down2_own)

close_device()
print("\nDONE")
