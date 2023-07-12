#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from cvnets import parameter_list
from cvnets.layers import norm_layers_tuple
from cvnets.misc.init_utils import initialize_weights
from utils import logger
from utils.ddp_utils import is_master


class BaseTextEncoder(nn.Module):
    """Base class for text encoder"""

    def __init__(self, opts, projection_dim: int, *args, **kwargs) -> None:
        is_master_node = is_master(opts)
        vocab_size = getattr(opts, "dataset.text_vocab_size")
        if getattr(opts, "common.debug_mode", False):
            vocab_size = 100
        if vocab_size is None and is_master_node:
            logger.error(
                "Vocabulary size can't be None or -1 in {}. Got: {}".format(
                    self.__class__.__name__, vocab_size
                )
            )

        super(BaseTextEncoder, self).__init__()
        self.opts = opts
        self.projection_dim = projection_dim
        self.is_master_node = is_master_node
        self.vocab_size = vocab_size

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add model specific arguments"""
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--model.text.name",
            type=str,
            default=None,
            help="Name of the text encoder",
        )

        return parser

    def reset_parameters(self):
        """Initialize model weights"""
        initialize_weights(opts=self.opts, modules=self.modules())

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs
    ):

        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            *args,
            **kwargs
        )
        return param_list, [1.0] * len(param_list)

    def freeze_norm_layers(self) -> None:
        for m in self.modules():
            if isinstance(m, norm_layers_tuple):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.training = False

    def forward(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs
    ) -> Any:
        raise NotImplementedError

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        """Create dummy input and labels for CI/CD purposes. Child classes must override it
        if functionality is different.
        """
        seq_length = 77
        vocab_size = 10
        text_tensor = torch.randint(
            low=0, high=vocab_size, size=(batch_size, seq_length)
        ).long()
        return {"text": text_tensor}
