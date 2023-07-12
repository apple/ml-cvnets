#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse

from data.datasets.multi_modal_img_text.base_multi_modal_img_text import (
    BaseMultiModalImgText,
)
from data.datasets.multi_modal_img_text.zero_shot import arguments_zero_shot_dataset


def arguments_multi_modal_img_text(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:

    parser = arguments_zero_shot_dataset(parser)
    parser = BaseMultiModalImgText.add_arguments(parser)
    return parser
