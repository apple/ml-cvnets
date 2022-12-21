# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse

from .neural_aug import build_neural_augmentor, BaseNeuralAugmentor


def arguments_neural_augmentor(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    return BaseNeuralAugmentor.add_arguments(parser=parser)
