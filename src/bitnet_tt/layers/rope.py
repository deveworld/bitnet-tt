"""
Rotary Position Embedding (RoPE) implementation using TT-NN.

This module provides optimized RoPE for Tenstorrent hardware with:
- Pre-computed cos/sin tables uploaded once to device
- Embedding-based lookup for decode mode (no host->device transfer per token)
- Transformation matrix for fused RoPE application

Based on tt_transformers/tt/rope.py patterns.
"""

from typing import Tuple

import torch
import ttnn


def compute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 500000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RoPE cos/sin tables in meta format (matching tt_transformers).

    Args:
        head_dim: Dimension of each attention head
        max_seq_len: Maximum sequence length
        theta: RoPE base frequency (500000 for BitNet)

    Returns:
        (cos, sin) tensors of shape [1, 1, max_seq_len, head_dim]
    """
    # Compute inverse frequencies for half the dimension
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # Compute position embeddings
    t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim/2)

    # Concatenate to get full dimension
    emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, head_dim)

    cos = emb.cos()
    sin = emb.sin()

    # Permute to meta format: interleave pairs
    # From: [cos(0), cos(1), ..., cos(d/2-1), cos(0), cos(1), ...]
    # To:   [cos(0), cos(0), cos(1), cos(1), ...]
    cos = cos[:, : head_dim // 2]
    cos = torch.stack((cos, cos), dim=-1).flatten(-2)

    sin = sin[:, : head_dim // 2]
    sin = torch.stack((sin, sin), dim=-1).flatten(-2)

    # Reshape to [1, 1, max_seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return cos, sin


def get_rot_transformation_mat(head_dim: int = 128) -> torch.Tensor:
    """
    Get rotation transformation matrix for fused RoPE.

    This matrix transforms x to rotate_half(x) in a single matmul.

    Args:
        head_dim: Head dimension (default 128 for BitNet)

    Returns:
        Transformation matrix of shape [1, 1, head_dim, head_dim]
    """
    # For each pair of elements (x_2i, x_2i+1), we want:
    # rotate_half: [-x_2i+1, x_2i]
    # This can be done with a block-diagonal matrix of 2x2 rotation blocks
    rot_emb_matrix = torch.zeros(head_dim, head_dim)

    for i in range(head_dim // 2):
        # For pair (2i, 2i+1): output is (-x_2i+1, x_2i)
        rot_emb_matrix[2 * i, 2 * i + 1] = -1.0
        rot_emb_matrix[2 * i + 1, 2 * i] = 1.0

    return rot_emb_matrix.unsqueeze(0).unsqueeze(0)


class RotarySetup:
    """
    Pre-computed rotary position embeddings for TT-NN.

    Stores cos/sin tables on device and provides embedding-based lookup
    for efficient decode mode (avoids host->device transfer per token).

    Usage:
        # Initialize once
        rope = RotarySetup(device, head_dim=128, max_seq_len=4096, rope_theta=500000)

        # During decode (no host->device transfer)
        cos, sin = rope.get_rot_mats(position_ids)  # position_ids: torch.Tensor

        # Apply RoPE using ttnn.experimental.rotary_embedding_llama
        q_rotated = ttnn.experimental.rotary_embedding_llama(
            q_heads, cos, sin, rope.transformation_mat, is_decode_mode=True
        )
    """

    def __init__(
        self,
        device: ttnn.Device,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float = 500000.0,
        batch_size: int = 1,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        """
        Initialize RoPE with pre-computed tables.

        Args:
            device: TT-NN device
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            rope_theta: Base frequency (500000 for BitNet)
            batch_size: Batch size for sharding
            dtype: Data type for tensors
        """
        self.device = device
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.batch_size = batch_size
        self.dtype = dtype

        # Compute cos/sin tables on CPU
        cos_cpu, sin_cpu = compute_rope_freqs(head_dim, max_seq_len, rope_theta)

        # Upload full tables to device (for embedding lookup)
        # Shape: [1, 1, max_seq_len, head_dim] -> [max_seq_len, head_dim] for embedding
        self.cos_matrix = ttnn.from_torch(
            cos_cpu.squeeze(0).squeeze(0),  # [max_seq_len, head_dim]
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
        )
        self.sin_matrix = ttnn.from_torch(
            sin_cpu.squeeze(0).squeeze(0),  # [max_seq_len, head_dim]
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
        )

        # Pre-compute transformation matrix
        trans_mat = get_rot_transformation_mat(head_dim)
        self.transformation_mat = ttnn.from_torch(
            trans_mat.repeat(1, 1, batch_size, 1),  # [1, 1, batch_size, head_dim]
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Prefill transformation matrix (different shape)
        self.transformation_mat_prefill = ttnn.from_torch(
            trans_mat,  # [1, 1, head_dim, head_dim]
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Store CPU tensors for fallback/prefill
        self._cos_cpu = cos_cpu
        self._sin_cpu = sin_cpu

    def get_rot_mats_decode(
        self,
        position_ids: torch.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Get rotation matrices for decode mode using embedding lookup.

        No host->device transfer needed - uses pre-uploaded tables.

        Args:
            position_ids: Position indices [batch] or [1, batch]

        Returns:
            (cos, sin) tensors ready for rotary_embedding_llama
        """
        # Ensure position_ids is [1, batch] shape
        if len(position_ids.shape) == 1:
            position_ids = position_ids.unsqueeze(0)

        batch = position_ids.shape[-1]

        # Pad to tile size if needed
        pad_size = (32 - batch % 32) % 32
        if pad_size > 0:
            position_ids = torch.nn.functional.pad(position_ids, (0, pad_size), value=0)

        # Convert to TTNN tensor
        rot_idxs = ttnn.from_torch(
            position_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Embedding lookup: [1, batch] -> [1, batch, head_dim]
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)

        # Reshape to [1, 1, batch, head_dim] -> [1, batch, 1, head_dim]
        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, batch, head_dim]
        sin = ttnn.unsqueeze_to_4D(sin)

        cos = ttnn.transpose(cos, 1, 2)  # [1, batch, 1, head_dim]
        sin = ttnn.transpose(sin, 1, 2)

        # Trim padding if added
        if pad_size > 0:
            cos = cos[:, :batch, :, :]
            sin = sin[:, :batch, :, :]

        return cos, sin

    def get_rot_mats_prefill(
        self,
        seq_len: int,
        start_pos: int = 0,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Get rotation matrices for prefill mode.

        Args:
            seq_len: Sequence length
            start_pos: Starting position (default 0)

        Returns:
            (cos, sin) tensors for prefill [1, 1, seq_len, head_dim]
        """
        # Slice from CPU and upload
        cos = self._cos_cpu[:, :, start_pos:start_pos + seq_len, :]
        sin = self._sin_cpu[:, :, start_pos:start_pos + seq_len, :]

        cos_ttnn = ttnn.from_torch(
            cos,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin_ttnn = ttnn.from_torch(
            sin,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return cos_ttnn, sin_ttnn

    def get_trans_mats(self) -> dict:
        """Get transformation matrices for decode and prefill modes."""
        return {
            "decode": self.transformation_mat,
            "prefill": self.transformation_mat_prefill,
        }


def apply_rope_ttnn(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    trans_mat: ttnn.Tensor,
    is_decode_mode: bool = True,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Apply rotary position embeddings using native TT-NN op.

    Falls back to manual implementation if ttnn.experimental.rotary_embedding_llama
    is not available.

    Args:
        q: Query tensor [batch, num_heads, seq, head_dim]
        k: Key tensor [batch, num_kv_heads, seq, head_dim]
        cos: Cosine embeddings
        sin: Sine embeddings
        trans_mat: Transformation matrix
        is_decode_mode: Whether in decode mode

    Returns:
        (q_rotated, k_rotated) tensors
    """
    try:
        # Try native fused op (faster)
        q_rotated = ttnn.experimental.rotary_embedding_llama(
            q, cos, sin, trans_mat, is_decode_mode=is_decode_mode
        )
        k_rotated = ttnn.experimental.rotary_embedding_llama(
            k, cos, sin, trans_mat, is_decode_mode=is_decode_mode
        )
        return q_rotated, k_rotated
    except (AttributeError, RuntimeError):
        # Fall back to manual implementation
        return _apply_rope_manual(q, k, cos, sin)


def _apply_rope_manual(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Manual RoPE implementation (fallback).

    Uses element-wise operations instead of fused kernel.
    """
    def rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
        """Rotate half the hidden dims."""
        head_dim = x.shape[-1]
        half = head_dim // 2
        x1 = x[:, :, :, :half]
        x2 = x[:, :, :, half:]
        return ttnn.concat([ttnn.neg(x2), x1], dim=-1)

    def apply_to_tensor(x: ttnn.Tensor) -> ttnn.Tensor:
        """Apply RoPE to a single tensor."""
        # Expand cos/sin to match tensor shape
        num_heads = x.shape[1]
        cos_exp = ttnn.repeat(cos, ttnn.Shape([1, num_heads, 1, 1]))
        sin_exp = ttnn.repeat(sin, ttnn.Shape([1, num_heads, 1, 1]))

        # x * cos + rotate_half(x) * sin
        x_cos = ttnn.multiply(x, cos_exp)
        x_rot_sin = ttnn.multiply(rotate_half(x), sin_exp)
        return ttnn.add(x_cos, x_rot_sin)

    return apply_to_tensor(q), apply_to_tensor(k)
