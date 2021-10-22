#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from itertools import product
import torch
from math import sqrt
import numpy as np
from typing import Dict, Optional, List


class SSDAnchorGenerator(torch.nn.Module):
    """
        Generate anchors (or priors) for Single Shot object detector:
            https://arxiv.org/abs/1512.02325

        Anchor boxes can be generated for any image size
    """
    def __init__(self,
                 output_strides: List,
                 aspect_ratios: List,
                 min_ratio: Optional[float] = 0.1,
                 max_ratio: Optional[float] = 1.05,
                 no_clipping: Optional[bool] = False
                 ):
        super(SSDAnchorGenerator, self).__init__()
        output_strides_aspect_ratio = dict()
        for k, v in zip(output_strides, aspect_ratios):
            output_strides_aspect_ratio[k] = v

        self.anchors_dict = dict()
        scales = np.linspace(min_ratio, max_ratio, len(output_strides) + 1)
        self.sizes = dict()
        for i, s in enumerate(output_strides):
            self.sizes[s] = {
                "min": scales[i],
                "max": sqrt(scales[i] * scales[i+1])
            }
        self.output_strides_aspect_ratio = self.process_aspect_ratio(output_strides_aspect_ratio)

        self.clip = not no_clipping

    @staticmethod
    def process_aspect_ratio(output_strides_aspect_ratio: Dict) -> Dict:
        for os, curr_ar in output_strides_aspect_ratio.items():
            assert isinstance(curr_ar, list)
            new_ar = list(set(curr_ar)) # keep only unique values
            output_strides_aspect_ratio[os] = new_ar
        return output_strides_aspect_ratio

    def num_anchors_per_os(self):
        # Estimate num of anchors based on aspect ratios: 2 default boxes + 2 * aspect ratios in feature map.
        return [2 + 2 * len(ar) for os, ar in self.output_strides_aspect_ratio.items()]

    @torch.no_grad()
    def generate_anchors_center_form(self, height: int, width: int, output_stride: int, *args, **kwargs):
        min_size_h = self.sizes[output_stride]["min"]
        min_size_w = self.sizes[output_stride]["min"]

        max_size_h = self.sizes[output_stride]["max"]
        max_size_w = self.sizes[output_stride]["max"]
        aspect_ratio = self.output_strides_aspect_ratio[output_stride]

        default_anchors_ctr = []
        scale_x = (1.0 / width)
        scale_y = (1.0 / height)

        for y, x in product(range(height), range(width)):
            # [x, y, w, h] format
            cx = (x + 0.5) * scale_x
            cy = (y + 0.5) * scale_y

            # small size box
            default_anchors_ctr.append([cx, cy, min_size_w, min_size_h])

            # big size box
            default_anchors_ctr.append([cx, cy, max_size_w, max_size_h])

            # change h/w ratio of the small sized box based on aspect ratios
            for ratio in aspect_ratio:
                ratio = sqrt(ratio)
                default_anchors_ctr.append([cx, cy, min_size_w * ratio, min_size_h / ratio])
                default_anchors_ctr.append([cx, cy, min_size_w / ratio, min_size_h * ratio])
        default_anchors_ctr = torch.tensor(default_anchors_ctr, dtype=torch.float)
        if self.clip:
            default_anchors_ctr = torch.clamp(default_anchors_ctr, min=0.0, max=1.0)

        return default_anchors_ctr

    @torch.no_grad()
    def get_anchors(self, fm_height: int, fm_width: int, fm_output_stride: int) -> torch.Tensor:
        key = "h_{}_w_{}_os_{}".format(fm_height, fm_width, fm_output_stride)
        if key not in self.anchors_dict:
            default_anchors_ctr = self.generate_anchors_center_form(height=fm_height, width=fm_width, output_stride=fm_output_stride)
            self.anchors_dict[key] = default_anchors_ctr
            return default_anchors_ctr
        else:
            return self.anchors_dict[key]

    @torch.no_grad()
    def forward(self, fm_height: int, fm_width: int, fm_output_stride: int) -> torch.Tensor:
        return self.get_anchors(fm_height=fm_height, fm_width=fm_width, fm_output_stride=fm_output_stride)
