#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse


def arguments_multi_modal_img_text(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        title="Multi-modal image-text arguments",
        description="Multi-modal image-text arguments",
    )

    group.add_argument(
        "--dataset.multi-modal-img-text.zero-shot-eval",
        action="store_true",
        help="Use zero shot evaluation",
    )

    group.add_argument(
        "--dataset.multi-modal-img-text.context-length",
        type=int,
        default=77,
        help="Context length for the text model",
    )

    group.add_argument(
        "--dataset.multi-modal-img-text.trunc-seq-len",
        action="store_true",
        help="Enable sequence length truncation",
    )

    from .zero_shot import arguments_zero_shot_dataset

    parser = arguments_zero_shot_dataset(parser)

    return parser
