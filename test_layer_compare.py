#!/usr/bin/env python3
"""Compare bfp4 vs packed_ternary at each layer stage."""
import numpy as np
import ttnn
from bitnet_tt.utils.device import get_device, close_device
from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model
from bitnet_tt.model.bitnet import create_model
from bitnet_tt.layers.bitlinear import ttnn_to_numpy, numpy_int_to_ttnn


def compare(name, t1, t2):
    o1 = ttnn_to_numpy(t1)
    o2 = ttnn_to_numpy(t2)
    sz = min(o1.size, o2.size)
    a1 = o1.flatten()[:sz]
    a2 = o2.flatten()[:sz]
    corr = np.corrcoef(a1, a2)[0, 1] if sz > 1 else 0
    m1 = np.abs(o1).mean()
    m2 = np.abs(o2).mean()
    ratio = m2 / m1 if m1 > 0 else 0
    print(f"  {name:25s}: corr={corr:.6f}  abs_mean bfp4={m1:.4f} pt={m2:.4f}  ratio={ratio:.4f}")


def main():
    device = get_device()
    state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")

    m1 = create_model(config, device, weight_dtype="bfp4")
    load_weights_to_model(m1, state_dict)
    print("bfp4 model loaded")

    m2 = create_model(config, device, weight_dtype="packed_ternary")
    load_weights_to_model(m2, state_dict)
    print("packed_ternary model loaded")

    # Check Linear types
    gu1 = m1.layers[0].mlp.gate_up_proj
    gu2 = m2.layers[0].mlp.gate_up_proj
    dn2 = m2.layers[0].mlp.down_proj
    print(f"gate_up bfp4: dtype={gu1._weight_dtype_name} packed={gu1._use_packed_ternary} scale={gu1.weight_scale:.6f}")
    print(f"gate_up pt:   dtype={gu2._weight_dtype_name} packed={gu2._use_packed_ternary} scale={gu2.weight_scale:.6f}")
    print(f"down pt:      dtype={dn2._weight_dtype_name} packed={dn2._use_packed_ternary} scale={dn2.weight_scale:.6f}")

    # Create input
    ids = np.array([[1, 2, 3]], dtype=np.int32)
    ids_t = numpy_int_to_ttnn(ids, device)

    # Embedding (same for both)
    emb1 = m1.embed_tokens(ids_t)
    emb2 = m2.embed_tokens(ids_t)
    compare("embedding", emb1, emb2)

    # Layer 0: input layernorm
    x1 = m1.layers[0].input_layernorm(emb1)
    x2 = m2.layers[0].input_layernorm(emb2)
    compare("input_layernorm", x1, x2)

    # Layer 0: gate_up_proj
    gu_out1 = m1.layers[0].mlp.gate_up_proj(x1)
    gu_out2 = m2.layers[0].mlp.gate_up_proj(x2)
    compare("gate_up_proj", gu_out1, gu_out2)

    # Layer 0: down_proj (use bfp4's gate_up output for both to isolate down_proj)
    # First apply silu+mul like FFN does
    # Actually let's just test down_proj directly with same input
    dn1 = m1.layers[0].mlp.down_proj
    dn2 = m2.layers[0].mlp.down_proj
    print(f"down bfp4: dtype={dn1._weight_dtype_name} packed={dn1._use_packed_ternary} scale={dn1.weight_scale:.6f}")
    print(f"down pt:   dtype={dn2._weight_dtype_name} packed={dn2._use_packed_ternary} scale={dn2.weight_scale:.6f}")

    # Test down_proj with same input
    dn_out1 = dn1(gu_out1)
    dn_out2 = dn2(gu_out1)  # use bfp4's gate_up output for both
    compare("down_proj(same_in)", dn_out1, dn_out2)

    close_device()
    print("DONE")


if __name__ == "__main__":
    main()
