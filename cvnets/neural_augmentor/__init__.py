# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse

from cvnets.neural_augmentor.neural_aug import (
    BaseNeuralAugmentor,
    build_neural_augmentor,
)


def arguments_neural_augmentor(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    return BaseNeuralAugmentor.add_arguments(parser=parser)
