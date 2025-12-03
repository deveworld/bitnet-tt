"""
Weight loading utilities for BitNet models.

This module provides functions for loading pretrained BitNet weights
from HuggingFace and converting them to the format expected by our
TT-NN implementation.
"""

import json
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bitnet_tt.config import BitNet2B4TConfig, BitNetConfig


def load_bitnet_weights(
    model_id: str = "microsoft/bitnet-b1.58-2B-4T",
    cache_dir: str | None = None,
) -> tuple[dict[str, NDArray[np.floating]], BitNetConfig]:
    """
    Load BitNet weights from HuggingFace.

    Args:
        model_id: HuggingFace model ID
        cache_dir: Optional cache directory

    Returns:
        Tuple of (state_dict as numpy arrays, config)
    """
    try:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError(
            "Please install safetensors and huggingface_hub: "
            "pip install safetensors huggingface_hub"
        ) from e

    # Download model files
    model_path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "config.json"],
    )

    # Load config from JSON
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        hf_config = json.load(f)

    # Create our config from HuggingFace config
    config = BitNet2B4TConfig(
        hidden_size=hf_config.get("hidden_size", 2560),
        num_layers=hf_config.get("num_hidden_layers", 30),
        num_attention_heads=hf_config.get("num_attention_heads", 20),
        num_key_value_heads=hf_config.get("num_key_value_heads", 5),
        intermediate_size=hf_config.get("intermediate_size", 6912),
        vocab_size=hf_config.get("vocab_size", 128256),
        max_position_embeddings=hf_config.get("max_position_embeddings", 4096),
        rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
        rope_theta=hf_config.get("rope_theta", 500000.0),
    )

    # Load safetensors via torch (handles bfloat16), then convert to numpy
    import torch
    state_dict: dict[str, NDArray[np.floating]] = {}
    for filename in os.listdir(model_path):
        if filename.endswith(".safetensors"):
            filepath = os.path.join(model_path, filename)
            with safe_open(filepath, framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    # Convert bfloat16 to float32 for numpy compatibility
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.float()
                    state_dict[key] = tensor.numpy()

    # Convert HuggingFace keys to our keys
    converted_state_dict = convert_hf_state_dict(state_dict)

    return converted_state_dict, config


def convert_hf_state_dict(
    hf_state_dict: dict[str, NDArray[np.floating]],
) -> dict[str, NDArray[np.floating]]:
    """
    Convert HuggingFace state dict keys to our format.

    Args:
        hf_state_dict: State dict with HuggingFace keys

    Returns:
        State dict with our keys
    """
    # Key mapping from HuggingFace to our format
    key_mapping = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    # Layer-specific mappings
    layer_mapping = {
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "self_attn.q_proj.weight": "self_attn.q_proj.weight",
        "self_attn.k_proj.weight": "self_attn.k_proj.weight",
        "self_attn.v_proj.weight": "self_attn.v_proj.weight",
        "self_attn.o_proj.weight": "self_attn.o_proj.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
        # BitLinear norm weights
        "self_attn.q_proj.input_norm.weight": "self_attn.q_proj.input_norm.weight",
        "self_attn.k_proj.input_norm.weight": "self_attn.k_proj.input_norm.weight",
        "self_attn.v_proj.input_norm.weight": "self_attn.v_proj.input_norm.weight",
        "self_attn.o_proj.input_norm.weight": "self_attn.o_proj.input_norm.weight",
        "mlp.gate_proj.input_norm.weight": "mlp.gate_proj.input_norm.weight",
        "mlp.up_proj.input_norm.weight": "mlp.up_proj.input_norm.weight",
        "mlp.down_proj.input_norm.weight": "mlp.down_proj.input_norm.weight",
    }

    converted: dict[str, NDArray[np.floating]] = {}

    for hf_key, tensor in hf_state_dict.items():
        # Direct mapping
        if hf_key in key_mapping:
            converted[key_mapping[hf_key]] = tensor
            continue

        # Layer mapping
        if hf_key.startswith("model.layers."):
            parts = hf_key.split(".")
            layer_idx = parts[2]
            rest = ".".join(parts[3:])

            if rest in layer_mapping:
                new_key = f"layers.{layer_idx}.{layer_mapping[rest]}"
                converted[new_key] = tensor
            else:
                # Keep original if no mapping found
                new_key = f"layers.{layer_idx}.{rest}"
                converted[new_key] = tensor
        else:
            # Keep original key if no mapping
            converted[hf_key] = tensor

    return converted


def get_layer_weights(
    state_dict: dict[str, NDArray[np.floating]],
    layer_idx: int,
) -> dict[str, NDArray[np.floating]]:
    """
    Extract weights for a specific layer.

    Args:
        state_dict: Full state dict
        layer_idx: Layer index

    Returns:
        Dictionary of weights for the specified layer
    """
    prefix = f"layers.{layer_idx}."
    layer_weights: dict[str, NDArray[np.floating]] = {}

    for key, value in state_dict.items():
        if key.startswith(prefix):
            # Remove prefix
            new_key = key[len(prefix) :]
            layer_weights[new_key] = value

    return layer_weights


def save_numpy_weights(
    state_dict: dict[str, NDArray[np.floating]],
    output_path: str,
) -> None:
    """
    Save weights as numpy npz file.

    Args:
        state_dict: State dict with numpy arrays
        output_path: Output file path
    """
    np.savez_compressed(output_path, **state_dict)


def load_numpy_weights(
    input_path: str,
) -> dict[str, NDArray[np.floating]]:
    """
    Load weights from numpy npz file.

    Args:
        input_path: Input file path

    Returns:
        State dict with numpy arrays
    """
    data = np.load(input_path)
    return {key: data[key] for key in data.files}


def load_weights_to_model(
    model: Any,
    state_dict: dict[str, NDArray[np.floating]],
) -> None:
    """
    Load weights into a BitNetModel instance.

    Args:
        model: BitNetModel instance
        state_dict: State dict with numpy arrays
    """
    # Load embedding weights
    if "embed_tokens.weight" in state_dict:
        model.load_embedding_weights(state_dict["embed_tokens.weight"])

    # Load layer weights
    for layer_idx in range(len(model.layers)):
        layer_weights = get_layer_weights(state_dict, layer_idx)
        if layer_weights:
            model.load_layer_weights(layer_idx, layer_weights)

    # Load final weights
    if "norm.weight" in state_dict and "lm_head.weight" in state_dict:
        model.load_final_weights(
            state_dict["norm.weight"],
            state_dict["lm_head.weight"],
        )
