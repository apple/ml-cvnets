#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
import math

from .base_layer import BaseLayer
from .dropout import Dropout


class PositionalEncoding(BaseLayer):
    """
    This layer adds sinusoidal positional embeddings to the input signal
        Adapted from Pytorch tutorial:
            https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2) # [B x E x Max_patches)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :, :x.size(-1)]
        return self.dropout(x)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        return input, 0.0, 0.0

    def __repr__(self):
        return "{}(dropout={})".format(self.__class__.__name__, self.dropout.p)
