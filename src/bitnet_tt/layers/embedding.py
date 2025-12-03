"""
Embedding layer implementation using TT-NN.

This module provides TT-NN native embedding lookup functionality.
"""

import numpy as np
import ttnn
from numpy.typing import NDArray

from bitnet_tt.layers.bitlinear import numpy_to_ttnn


class Embedding:
    """
    TT-NN native embedding layer.

    Standard embedding lookup without quantization, as embedding layers
    in BitNet are not quantized.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        device: ttnn.Device,
    ) -> None:
        """
        Initialize embedding layer.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            device: TT-NN device
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.weight: ttnn.Tensor | None = None

    def load_weights(self, weight: NDArray[np.floating]) -> None:
        """
        Load embedding weights to device.

        Args:
            weight: Weight array of shape (vocab_size, embedding_dim)
        """
        # Embedding uses ROW_MAJOR_LAYOUT
        self.weight = numpy_to_ttnn(
            weight.astype(np.float32),
            self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def __call__(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """
        Embedding lookup.

        Args:
            input_ids: Token indices tensor

        Returns:
            Embeddings tensor in TILE layout
        """
        if self.weight is None:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")

        # Embedding lookup returns ROW_MAJOR, convert to TILE for subsequent ops
        embeddings = ttnn.embedding(input_ids, self.weight)
        return ttnn.to_layout(embeddings, ttnn.TILE_LAYOUT)
