"""
Optimized Rotary Position Embedding (RoPE) for TT-NN.

Based on tt_transformers RotarySetup pattern:
- Pre-compute and upload cos/sin matrices ONCE at init
- Use ttnn.embedding for position lookup (instead of per-token numpy creation)
- Use ttnn.experimental.rotary_embedding_llama for fused RoPE application
"""

import math
from typing import List, Optional, Tuple

import torch
import ttnn


def get_rot_transformation_mat(dhead: int) -> torch.Tensor:
    """
    Create rotation transformation matrix for RoPE.

    This matrix is used by rotary_embedding_llama for the rotation operation.
    """
    rot_mat = torch.zeros(1, 1, dhead, dhead)
    half_dim = dhead // 2
    # Create rotation pattern: [x1, x2, ...] -> [-x_{half+1}, -x_{half+2}, ..., x1, x2, ...]
    for i in range(half_dim):
        rot_mat[0, 0, i, i + half_dim] = -1.0
        rot_mat[0, 0, i + half_dim, i] = 1.0
    return rot_mat


def compute_cos_sin_cache(
    head_dim: int,
    max_seq_len: int,
    rope_theta: float = 500000.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute cos/sin matrices for all positions.

    Returns:
        cos_cache: [1, 1, max_seq_len, head_dim]
        sin_cache: [1, 1, max_seq_len, head_dim]
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))

    # Compute position * frequency
    positions = torch.arange(max_seq_len, dtype=dtype)
    freqs = torch.outer(positions, inv_freq)  # [max_seq_len, head_dim/2]

    # Duplicate for full head_dim
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, head_dim]

    cos = emb.cos()
    sin = emb.sin()

    # Permute to meta format (for rotary_embedding_llama)
    # Stack interleaved: [cos0, cos0, cos1, cos1, ...]
    cos_half = cos[:, :head_dim // 2]
    sin_half = sin[:, :head_dim // 2]

    cos_interleaved = torch.stack([cos_half, cos_half], dim=-1).flatten(-2)
    sin_interleaved = torch.stack([sin_half, sin_half], dim=-1).flatten(-2)

    # Reshape to [1, 1, max_seq_len, head_dim]
    cos_cache = cos_interleaved.unsqueeze(0).unsqueeze(0)
    sin_cache = sin_interleaved.unsqueeze(0).unsqueeze(0)

    return cos_cache, sin_cache


class RotarySetup:
    """
    Optimized RoPE setup following tt_transformers pattern.

    Key optimizations:
    1. Pre-compute cos/sin for all positions at init
    2. Upload to device ONCE
    3. Use ttnn.embedding for position lookup (fast device-side indexing)
    4. Create transformation matrix for fused rotary_embedding_llama
    """

    def __init__(
        self,
        device: ttnn.Device,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float = 500000.0,
        batch_size: int = 1,
        datatype: ttnn.DataType = ttnn.bfloat16,
    ):
        """
        Initialize RoPE setup with pre-computed matrices.

        Args:
            device: TT-NN device
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length to support
            rope_theta: RoPE theta parameter
            batch_size: Maximum batch size
            datatype: Data type for tensors
        """
        self.device = device
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        # Compute cos/sin cache on CPU
        cos_cache, sin_cache = compute_cos_sin_cache(
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )

        # Upload cos/sin matrices to device ONCE
        # Shape for embedding lookup: [max_seq_len, head_dim] (2D for embedding)
        cos_for_embedding = cos_cache.squeeze(0).squeeze(0)  # [max_seq_len, head_dim]
        sin_for_embedding = sin_cache.squeeze(0).squeeze(0)  # [max_seq_len, head_dim]

        self.cos_matrix = ttnn.from_torch(
            cos_for_embedding,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
        )
        self.sin_matrix = ttnn.from_torch(
            sin_for_embedding,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
        )

        # Create transformation matrix for decode
        # Shape: [1, 1, batch_size, TILE_SIZE, TILE_SIZE] for sharded config
        # For single device, use simpler shape
        trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE)
        trans_mat_repeated = trans_mat.repeat(1, 1, batch_size, 1)

        self.transformation_mat_decode = ttnn.from_torch(
            trans_mat_repeated,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.L1_MEMORY_CONFIG,  # Keep in L1 for fast access
        )

        # Prefill transformation matrix (different size)
        trans_mat_prefill = get_rot_transformation_mat(dhead=head_dim)
        self.transformation_mat_prefill = ttnn.from_torch(
            trans_mat_prefill,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def get_rot_mats(
        self,
        position_ids: torch.Tensor | ttnn.Tensor,
    ) -> List[ttnn.Tensor]:
        """
        Get rotation matrices for given positions using embedding lookup.

        This is MUCH faster than creating tensors per-token because:
        1. No host-device transfer per token
        2. Uses device-side embedding lookup

        Args:
            position_ids: [batch] tensor of position indices (torch or ttnn)

        Returns:
            [cos, sin] tensors for rotary embedding
        """
        if isinstance(position_ids, ttnn.Tensor):
            # Already on device (for trace)
            rot_idxs = position_ids
            batch = position_ids.shape[1] # Assuming [1, batch] from trace setup
        else:
            batch = position_ids.shape[0]

            # Pad to tile size if needed
            if batch % 32 != 0:
                pad_size = ((batch + 31) // 32) * 32 - batch
                position_ids = torch.nn.functional.pad(position_ids, (0, pad_size), value=0)

            # Create position index tensor on device
            position_ids_2d = position_ids.reshape(1, -1)  # [1, batch_padded]
            rot_idxs = ttnn.from_torch(
                position_ids_2d,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Use embedding lookup for cos/sin (device-side operation!)
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)  # [1, batch, head_dim]
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)  # [1, batch, head_dim]

        # Reshape for rotary_embedding_llama: [1, batch, 1, head_dim]
        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, batch, head_dim]
        sin = ttnn.unsqueeze_to_4D(sin)  # [1, 1, batch, head_dim]

        cos = ttnn.transpose(cos, 1, 2)  # [1, batch, 1, head_dim]
        sin = ttnn.transpose(sin, 1, 2)  # [1, batch, 1, head_dim]

        # Slice to actual batch size if we padded
        if batch % 32 != 0:
            cos = cos[:, :batch, :, :]
            sin = sin[:, :batch, :, :]

        return [cos, sin]

    def get_rot_mats_prefill(
        self,
        seq_len: int,
        start_pos: int = 0,
    ) -> List[ttnn.Tensor]:
        """
        Get rotation matrices for prefill (full sequence).

        Args:
            seq_len: Sequence length
            start_pos: Starting position (for chunked prefill)

        Returns:
            [cos, sin] tensors for the full sequence
        """
        # Create position indices for the sequence
        position_ids = torch.arange(start_pos, start_pos + seq_len)
        position_ids_2d = position_ids.reshape(1, -1)  # [1, seq_len]

        rot_idxs = ttnn.from_torch(
            position_ids_2d,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Use embedding lookup
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)

        # Reshape for rotary_embedding_llama prefill: [1, 1, seq_len, head_dim]
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)

        return [cos, sin]

    def get_transformation_mats(self) -> dict:
        """Get transformation matrices for decode and prefill modes."""
        return {
            "decode": self.transformation_mat_decode,
            "prefill": self.transformation_mat_prefill,
        }
