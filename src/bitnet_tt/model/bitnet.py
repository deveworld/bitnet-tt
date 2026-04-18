"""
Optimized BitNet model implementation using TT-NN.

This module provides the complete BitNet model for causal language modeling:
- Token embedding
- Stack of Transformer blocks with KV-Cache support
- Final normalization
- Language model head

Performance optimizations:
- Mode-aware forward (prefill vs decode)
- Pre-transposed LM head weight
- Memory config optimizations

Based on tt_transformers patterns.
"""

import os
from typing import Optional

import numpy as np
import torch
import ttnn
from numpy.typing import NDArray

from bitnet_tt.config import BitNetConfig
from bitnet_tt.layers.attention import KVCache
from bitnet_tt.layers.bitlinear import RMSNorm, numpy_to_ttnn
from bitnet_tt.layers.embedding import Embedding
from bitnet_tt.model.transformer import TransformerBlock


class BitNetModel:
    """
    TT-NN native BitNet Language Model.

    A decoder-only transformer model using BitLinear layers
    with ternary weights and 8-bit activations.
    """

    def __init__(
        self,
        config: BitNetConfig,
        device: ttnn.Device,
        use_lofi_mlp: bool = False,
        weight_dtype: str = "bfp4",
        use_fused_rope: bool = True,
    ) -> None:
        self.config = config
        self.device = device
        self._weight_dtype = weight_dtype

        # Token embedding (not quantized)
        self.embed_tokens = Embedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.hidden_size,
            device=device,
        )

        # Mixed-precision override: BITNET_BF16_LAYERS="0,29" keeps those
        # layers in bf16 while the rest use `weight_dtype`. Empty or
        # unset disables the override.
        self._bf16_layers: set[int] = set()
        for tok in os.environ.get("BITNET_BF16_LAYERS", "").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                self._bf16_layers.add(int(tok))
            except ValueError:
                pass

        # Transformer layers
        self.layers: list[TransformerBlock] = []
        for layer_idx in range(config.num_layers):
            layer_dtype = "bf16" if layer_idx in self._bf16_layers else weight_dtype
            self.layers.append(
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    intermediate_size=config.intermediate_size,
                    device=device,
                    max_position_embeddings=config.max_position_embeddings,
                    rope_theta=config.rope_theta,
                    rms_norm_eps=config.rms_norm_eps,
                    layer_idx=layer_idx,
                    use_lofi_mlp=use_lofi_mlp,
                    weight_dtype=layer_dtype,
                    use_fused_rope=use_fused_rope,
                )
            )

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, device, config.rms_norm_eps)

        # Language model head weight (will be loaded)
        self.lm_head_weight: ttnn.Tensor | None = None

    def load_embedding_weights(self, weight: NDArray[np.floating]) -> None:
        """
        Load embedding weights.

        Args:
            weight: Embedding weight array of shape (vocab_size, hidden_size)
        """
        self.embed_tokens.load_weights(weight)

    def load_layer_weights(self, layer_idx: int, weights: dict[str, NDArray[np.floating]]) -> None:
        """
        Load weights for a specific transformer layer.

        Args:
            layer_idx: Layer index
            weights: Dictionary of weight arrays
        """
        layer = self.layers[layer_idx]

        # BitNet b1.58 uses sub-norms:
        # - attn_sub_norm (hidden_size=2560): applied after attention output, before o_proj
        # - ffn_sub_norm (intermediate_size=6912): applied after gate*up, before down_proj
        attn_sub_norm = weights.get("self_attn.attn_sub_norm.weight")
        ffn_sub_norm = weights.get("mlp.ffn_sub_norm.weight")

        layer.load_weights(
            input_layernorm_weight=weights["input_layernorm.weight"],
            post_attention_layernorm_weight=weights["post_attention_layernorm.weight"],
            q_weight=weights["self_attn.q_proj.weight"],
            k_weight=weights["self_attn.k_proj.weight"],
            v_weight=weights["self_attn.v_proj.weight"],
            o_weight=weights["self_attn.o_proj.weight"],
            attn_sub_norm_weight=attn_sub_norm,
            gate_weight=weights["mlp.gate_proj.weight"],
            up_weight=weights["mlp.up_proj.weight"],
            down_weight=weights["mlp.down_proj.weight"],
            ffn_sub_norm_weight=ffn_sub_norm,
        )

    def load_final_weights(
        self,
        norm_weight: NDArray[np.floating],
        lm_head_weight: NDArray[np.floating],
    ) -> None:
        """
        Load final norm and LM head weights.

        LM head weight is pre-transposed to avoid transpose per forward.

        Args:
            norm_weight: Final norm weight
            lm_head_weight: LM head weight (vocab_size, hidden_size)
        """
        self.norm.load_weights(norm_weight)
        # Pre-transpose LM head: (vocab, hidden) -> (hidden, vocab)
        lm_head_t = lm_head_weight.T.astype(np.float32).copy()
        # lm_head dtype sweep (2026-04-18, 64-tok bench / 16-step accuracy,
        # packed_ternary weights, commit 5930113):
        #   bfp4:  p50=11.3ms  decode=70.88 t/s  PCC=0.9753  match=1/16
        #   bfp8:  p50=11.7ms  decode=68.86 t/s  PCC=0.9807  match=6/16  <- PICK
        #   bf16:  p50=12.4ms  decode=65.67 t/s  (slower, PCC not >>bfp8)
        # bfp8 is the Pareto-optimal point: +0.005 PCC and 6x greedy match
        # improvement for only -2 t/s vs bfp4. Argmax is still correct on
        # real hidden states (top1 match=True across all three).
        # BITNET_LM_HEAD_DTYPE env override ∈ {bf16, bfp8, bfp4} for sweep.
        _env_lh = os.environ.get("BITNET_LM_HEAD_DTYPE", "").lower()
        if _env_lh == "bf16":
            lm_head_dtype = ttnn.bfloat16
        elif _env_lh in ("bfp8", "bfp8_b", "bfloat8_b"):
            lm_head_dtype = ttnn.bfloat8_b
        elif _env_lh in ("bfp4", "bfp4_b", "bfloat4_b"):
            lm_head_dtype = ttnn.bfloat4_b
        else:
            lm_head_dtype = ttnn.bfloat8_b if self._weight_dtype != "bf16" else ttnn.bfloat16

        # Split lm_head weight along vocab dim. A single big ttnn.matmul for
        # [1, K] x [K, 128256] picks a per-core config sized for the whole
        # vocab and runs at ~2.2 ms in trace. Splitting into N smaller
        # matmuls lets each pick a tighter per-core config; the N outputs
        # concat back to the full logits in ~0.75 ms total (measured sweep:
        # split=3 757us, split=4 722us). We pick the largest split that keeps
        # each chunk tile-aligned (128256 % 32 == 0 so any divisor works).
        V = lm_head_t.shape[1]
        lm_head_split = 1
        for candidate in (4, 3, 6, 2):
            if V % candidate == 0 and (V // candidate) % 32 == 0:
                lm_head_split = candidate
                break
        self._lm_head_split = lm_head_split

        lm_head_tensor = torch.from_numpy(lm_head_t)
        if lm_head_split == 1:
            self.lm_head_weight = ttnn.from_torch(
                lm_head_tensor,
                dtype=lm_head_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.lm_head_weight_chunks: list[ttnn.Tensor] | None = None
        else:
            chunk_N = V // lm_head_split
            self.lm_head_weight_chunks = [
                ttnn.from_torch(
                    lm_head_tensor[:, i * chunk_N:(i + 1) * chunk_N].contiguous(),
                    dtype=lm_head_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for i in range(lm_head_split)
            ]
            # Keep a canonical reference too; some callers probe
            # self.lm_head_weight for shape. Use the first chunk by default
            # and rely on lm_head_weight_chunks being preferred.
            self.lm_head_weight = self.lm_head_weight_chunks[0]

    def __call__(
        self,
        input_ids: ttnn.Tensor | None = None,
        attention_mask: ttnn.Tensor | None = None,
        position_ids: NDArray[np.integer] | int | None = None,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
        mode: str = "prefill",
        current_pos: int | None = None,
        rot_mats: list | None = None,
        transformation_mat: ttnn.Tensor | None = None,
        current_pos_tensor: ttnn.Tensor | None = None,
        pos_tensor: ttnn.Tensor | None = None,  # For backward compatibility / alias
        cos_sin_tensors: tuple[ttnn.Tensor, ttnn.Tensor]
        | None = None,  # Pre-computed cos/sin for trace
        inputs_embeds: ttnn.Tensor | None = None,  # Pre-computed embeddings (for trace)
    ) -> tuple[ttnn.Tensor, Optional[list[KVCache]]]:
        """
        Forward pass with mode-aware optimization.

        Args:
            input_ids: Token indices tensor of shape (batch, seq_len). Can be None if inputs_embeds provided.
            attention_mask: Optional attention mask
            position_ids: Position indices for RoPE (int for decode mode)
            past_key_values: List of KV-Cache for each layer
            use_cache: Whether to return updated KV-Cache
            mode: "prefill" or "decode" - affects optimizations
            current_pos: Current position for decode mode (used if position_ids not set)
            rot_mats: [cos, sin] rotation matrices from RotarySetup (for optimized decode)
            transformation_mat: Transformation matrix for rotary_embedding_llama (for optimized decode)
            current_pos_tensor: Optional tensor containing current position (for trace)
            pos_tensor: Alias for current_pos_tensor
            inputs_embeds: Pre-computed embeddings (for trace mode - bypasses embedding layer)
        Returns:
            Tuple of (logits tensor, updated KV-Cache list if use_cache else None)
        """
        # Note: pos_tensor (uint32) is for RoPE embedding
        # current_pos_tensor (int32) is for KV cache paged_update_cache
        # They are NOT aliases - both may be used separately

        # Get embeddings - use pre-computed if provided (for trace mode)
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        elif input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Initialize cache list if using cache
        if use_cache and past_key_values is None:
            past_key_values = [None] * len(self.layers)

        updated_caches: list[KVCache] = [] if use_cache else None

        # Use current_pos as position_ids for decode mode if not explicitly set
        effective_position_ids = position_ids
        if mode == "decode" and position_ids is None and current_pos is not None:
            effective_position_ids = current_pos

        # Process through transformer layers
        for layer_idx, layer in enumerate(self.layers):
            past_kv = past_key_values[layer_idx] if past_key_values else None

            hidden_states, updated_cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=effective_position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
                mode=mode,
                rot_mats=rot_mats,
                transformation_mat=transformation_mat,
                current_pos_tensor=current_pos_tensor,  # int32 for KV cache
                pos_tensor=pos_tensor,  # uint32 for RoPE
                cos_sin_tensors=cos_sin_tensors,
            )

            if use_cache:
                updated_caches.append(updated_cache)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Language model head (weight is pre-transposed)
        if self.lm_head_weight is None:
            raise RuntimeError("LM head weights not loaded.")

        chunks = getattr(self, "lm_head_weight_chunks", None)
        if chunks is not None:
            chunk_outs = [ttnn.matmul(hidden_states, w) for w in chunks]
            logits = ttnn.concat(chunk_outs, dim=-1)
            for o in chunk_outs:
                ttnn.deallocate(o)
        else:
            logits = ttnn.matmul(hidden_states, self.lm_head_weight)

        return logits, updated_caches if use_cache else None

    def reset_cache(self, past_key_values: list[KVCache]) -> None:
        """
        Reset all KV-Caches.

        Args:
            past_key_values: List of KV-Cache to reset
        """
        for cache in past_key_values:
            if cache is not None:
                cache.reset()

    def preallocate_cache(
        self,
        batch_size: int = 1,
        max_seq_len: int | None = None,
    ) -> list[KVCache]:
        """
        Pre-allocate KV-Cache for all layers.

        This avoids memory reallocation during generation and improves performance.

        Args:
            batch_size: Batch size for generation
            max_seq_len: Maximum sequence length to cache (default: config.max_position_embeddings)

        Returns:
            List of pre-allocated KVCache objects for each layer
        """
        if max_seq_len is None:
            max_seq_len = self.config.max_position_embeddings

        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        caches: list[KVCache] = []
        for _ in range(len(self.layers)):
            cache = KVCache()
            cache.preallocate(
                batch_size=batch_size,
                num_kv_heads=num_kv_heads,
                max_seq_len=max_seq_len,
                head_dim=head_dim,
                device=self.device,
            )
            caches.append(cache)

        return caches


def create_model(
    config: BitNetConfig,
    device: ttnn.Device,
    use_lofi_mlp: bool = False,
    weight_dtype: str = "bfp4",
    use_fused_rope: bool = True,
) -> BitNetModel:
    return BitNetModel(config, device, use_lofi_mlp=use_lofi_mlp, weight_dtype=weight_dtype, use_fused_rope=use_fused_rope)
