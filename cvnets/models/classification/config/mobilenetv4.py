#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from typing import Dict


def get_configuration(opts) -> Dict:
    mv4_config = {
        "layer1": {
            "expansion_ratio": 1,
            "out_channels": 16,
            "num_blocks": 1,
            "stride": 1,
        },
        "layer2": {
            "expansion_ratio": 6,
            "out_channels": 24,
            "num_blocks": 2,
            "stride": 2,
        },
        "layer3": {
            "expansion_ratio": 6,
            "out_channels": 32,
            "num_blocks": 3,
            "stride": 2,
            "spatial_size": 32,
        },
        "layer4": {
            "expansion_ratio": 6,
            "out_channels": 64,
            "num_blocks": 4,
            "stride": 2,
            "spatial_size": 16,
        },
        "layer4_a": {
            "expansion_ratio": 6,
            "out_channels": 96,
            "num_blocks": 3,
            "stride": 1,
            "spatial_size": 16,
        },
        "layer5": {
            "expansion_ratio": 6,
            "out_channels": 160,
            "num_blocks": 3,
            "stride": 2,
            "spatial_size": 8,
        },
        "layer5_a": {
            "expansion_ratio": 6,
            "out_channels": 320,
            "num_blocks": 1,
            "stride": 1,
            "spatial_size": 8,
        },
    }
    return mv4_config
