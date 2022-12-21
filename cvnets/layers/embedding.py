#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
from typing import Optional, Union, Tuple
import argparse

from utils import logger

from .base_layer import BaseLayer
from .normalization_layers import get_normalization_layer
from .non_linear_layers import get_activation_fn


class Embedding(nn.Embedding):
    """A lookup table that stores embeddings of a fixed dictionary and size.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     i.e. it remains as a fixed "pad". For a newly constructed Embedding,
                                     the embedding vector at :attr:`padding_idx` will default to all zeros,
                                     but can be updated to another value to be used as the padding vector.

    Shape:
        - Input: :math:`(*)`, IntTensor or LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`
    """

    def __init__(
        self,
        opts,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        *args,
        **kwargs
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim**-0.5)
        if self.padding_idx is not None:
            nn.init.constant_(self.weight[self.padding_idx], 0)

    def profile_module(self, input: Tensor, *args, **kwargs) -> (Tensor, float, float):
        return input, 0.0, 0.0
