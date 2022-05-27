#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
import argparse
from typing import Any, Tuple


class BaseLayer(nn.Module):
    """
    Base class for neural network layers
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add layer specific arguments"""
        return parser

    def forward(self, *args, **kwargs) -> Any:
        pass

    def profile_module(self, *args, **kwargs) -> Tuple[Tensor, float, float]:
        raise NotImplementedError

    def __repr__(self):
        return "{}".format(self.__class__.__name__)
