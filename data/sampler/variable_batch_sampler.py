#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import random
import argparse
from utils import logger
from typing import Optional
from common import DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT

from . import register_sampler, BaseSamplerDP, BaseSamplerDDP
from .utils import _image_batch_pairs


"""
    Varaible Batch Sampler for DP and DDP use cases. This sampler allows to sample different image sizes during training.
    For example, the first batch can contain 512 images of size 224x224, the second batch can contain 1024 imnages of 
    size 112x112, third batch can contain 384 images of size 224 x 112, and so on
    
    See paper for details: https://arxiv.org/abs/2110.02178?context=cs.LG
"""


@register_sampler(name="variable_batch_sampler")
class VariableBatchSampler(BaseSamplerDP):
    """
        Variable batch sampler for DataParallel
    """
    def __init__(self, opts, n_data_samples: int, is_training: Optional[bool] = False):
        """

        :param opts: arguments
        :param n_data_samples: number of data samples in the dataset
        :param is_training: Training or evaluation mode (eval mode includes validation mode)
        """
        super(VariableBatchSampler, self).__init__(opts=opts, n_data_samples=n_data_samples, is_training=is_training)

        crop_size_w: int = getattr(opts, "sampler.vbs.crop_size_width", DEFAULT_IMAGE_WIDTH)
        crop_size_h: int = getattr(opts, "sampler.vbs.crop_size_height", DEFAULT_IMAGE_HEIGHT)

        min_crop_size_w: int = getattr(opts, "sampler.vbs.min_crop_size_width", 160)
        max_crop_size_w: int = getattr(opts, "sampler.vbs.max_crop_size_width", 320)

        min_crop_size_h: int = getattr(opts, "sampler.vbs.min_crop_size_height", 160)
        max_crop_size_h: int = getattr(opts, "sampler.vbs.max_crop_size_height", 320)

        scale_inc: bool = getattr(opts, "sampler.vbs.scale_inc", False)
        scale_ep_intervals: list or int = getattr(opts, "sampler.vbs.ep_intervals", [40])
        scale_inc_factor: float = getattr(opts, "sampler.vbs.scale_inc_factor", 0.25)

        check_scale_div_factor: int = getattr(opts, "sampler.vbs.check_scale", 32)
        max_img_scales: int = getattr(opts, "sampler.vbs.max_n_scales", 10)

        if isinstance(scale_ep_intervals, int):
            scale_ep_intervals = [scale_ep_intervals]

        self.min_crop_size_w = min_crop_size_w
        self.max_crop_size_w = max_crop_size_w
        self.min_crop_size_h = min_crop_size_h
        self.max_crop_size_h = max_crop_size_h

        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h

        self.scale_inc_factor = scale_inc_factor
        self.scale_ep_intervals = scale_ep_intervals

        self.max_img_scales = max_img_scales
        self.check_scale_div_factor = check_scale_div_factor
        self.scale_inc = scale_inc

        if is_training:
            self.img_batch_tuples = _image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.n_gpus,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h
            )
        else:
            self.img_batch_tuples = [(crop_size_h, crop_size_w, self.batch_size_gpu0)]

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(self.img_indices)
            random.shuffle(self.img_batch_tuples)

        start_index = 0
        while start_index < self.n_samples:
            crop_h, crop_w, batch_size = random.choice(self.img_batch_tuples)

            end_index = min(start_index + batch_size, self.n_samples)
            batch_ids = self.img_indices[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if len(batch_ids) != batch_size:
                batch_ids += self.img_indices[:(batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [(crop_h, crop_w, b_id) for b_id in batch_ids]
                yield batch

    def update_scales(self, epoch, is_master_node=False, *args, **kwargs):
        if epoch in self.scale_ep_intervals and self.scale_inc:
            self.min_crop_size_w += int(self.min_crop_size_w * self.scale_inc_factor)
            self.max_crop_size_w += int(self.max_crop_size_w * self.scale_inc_factor)

            self.min_crop_size_h += int(self.min_crop_size_h * self.scale_inc_factor)
            self.max_crop_size_h += int(self.max_crop_size_h * self.scale_inc_factor)

            self.img_batch_tuples = _image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.n_gpus,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h
            )
            if is_master_node:
                logger.log('Scales updated in {}'.format(self.__class__.__name__))
                logger.log("New scales: {}".format(self.img_batch_tuples))

    def __repr__(self):
        repr_str = '{}('.format(self.__class__.__name__)
        repr_str += '\n\t base_im_size=(h={}, w={}), ' \
                    '\n\t base_batch_size={} ' \
                    '\n\t scales={} ' \
                    '\n\t scale_inc={} ' \
                    '\n\t scale_inc_factor={} ' \
                    '\n\t ep_intervals={}'.format(self.crop_size_h, self.crop_size_w,
                                                  self.batch_size_gpu0,
                                                  self.img_batch_tuples,
                                                  self.scale_inc,
                                                  self.scale_inc_factor,
                                                  self.scale_ep_intervals)
        repr_str += '\n)'
        return repr_str

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Variable batch sampler",
                                          description="Arguments related to variable batch sampler")
        group.add_argument('--sampler.vbs.crop-size-width', default=DEFAULT_IMAGE_WIDTH, type=int,
                           help='Base crop size (along width) during training')
        group.add_argument('--sampler.vbs.crop-size-height', default=DEFAULT_IMAGE_HEIGHT, type=int,
                           help='Base crop size (along height) during training')

        group.add_argument('--sampler.vbs.min-crop-size-width', default=160, type=int,
                           help='Min. crop size along width during training')
        group.add_argument('--sampler.vbs.max-crop-size-width', default=320, type=int,
                           help='Max. crop size along width during training')

        group.add_argument('--sampler.vbs.min-crop-size-height', default=160, type=int,
                           help='Min. crop size along height during training')
        group.add_argument('--sampler.vbs.max-crop-size-height', default=320, type=int,
                           help='Max. crop size along height during training')
        group.add_argument('--sampler.vbs.max-n-scales', default=5, type=int,
                           help='Max. scales in variable batch sampler. For example, [0.25, 0.5, 0.75, 1, 1.25] ')
        group.add_argument('--sampler.vbs.check-scale', default=32, type=int,
                           help='Image scales should be divisible by this factor')
        group.add_argument('--sampler.vbs.ep-intervals', default=[40], type=int,
                           help='Epoch intervals at which scales are adjusted')
        group.add_argument('--sampler.vbs.scale-inc-factor', default=0.25, type=float,
                           help='Factor by which we should increase the scale')
        group.add_argument('--sampler.vbs.scale-inc', action='store_true',
                           help='Increase image scales during training')

        return parser


@register_sampler(name="variable_batch_sampler_ddp")
class VariableBatchSamplerDDP(BaseSamplerDDP):
    """
        Variable batch sampler for DistributedDataParallel
    """
    def __init__(self, opts, n_data_samples: int, is_training: Optional[bool] = False):
        """

        :param opts: arguments
        :param n_data_samples: number of data samples in the dataset
        :param is_training: Training or evaluation mode (eval mode includes validation mode)
        """
        super(VariableBatchSamplerDDP, self).__init__(opts=opts, n_data_samples=n_data_samples, is_training=is_training)
        crop_size_w: int = getattr(opts, "sampler.vbs.crop_size_width", DEFAULT_IMAGE_WIDTH)
        crop_size_h: int = getattr(opts, "sampler.vbs.crop_size_height", DEFAULT_IMAGE_HEIGHT)

        min_crop_size_w: int = getattr(opts, "sampler.vbs.min_crop_size_width", 160)
        max_crop_size_w: int = getattr(opts, "sampler.vbs.max_crop_size_width", 320)

        min_crop_size_h: int = getattr(opts, "sampler.vbs.min_crop_size_height", 160)
        max_crop_size_h: int = getattr(opts, "sampler.vbs.max_crop_size_height", 320)

        scale_inc: bool = getattr(opts, "sampler.vbs.scale_inc", False)
        scale_ep_intervals: list or int = getattr(opts, "sampler.vbs.ep_intervals", [40])
        scale_inc_factor: float = getattr(opts, "sampler.vbs.scale_inc_factor", 0.25)
        check_scale_div_factor: int = getattr(opts, "sampler.vbs.check_scale", 32)

        max_img_scales: int = getattr(opts, "sampler.vbs.max_n_scales", 10)

        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        self.min_crop_size_h = min_crop_size_h
        self.max_crop_size_h = max_crop_size_h
        self.min_crop_size_w = min_crop_size_w
        self.max_crop_size_w = max_crop_size_w

        self.scale_inc_factor = scale_inc_factor
        self.scale_ep_intervals = scale_ep_intervals
        self.max_img_scales = max_img_scales
        self.check_scale_div_factor = check_scale_div_factor
        self.scale_inc = scale_inc

        if is_training:
            self.img_batch_tuples = _image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.num_replicas,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h
            )
        else:
            self.img_batch_tuples = [(self.crop_size_h, self.crop_size_w, self.batch_size_gpu0)]

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(self.img_indices)
            random.shuffle(self.img_batch_tuples)
            indices_rank_i = self.img_indices[self.rank:len(self.img_indices):self.num_replicas]
        else:
            indices_rank_i = self.img_indices[self.rank:len(self.img_indices):self.num_replicas]

        start_index = 0
        while start_index < self.n_samples_per_replica:
            crop_h, crop_w, batch_size = random.choice(self.img_batch_tuples)

            end_index = min(start_index + batch_size, self.n_samples_per_replica)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != batch_size:
                batch_ids += indices_rank_i[:(batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [(crop_h, crop_w, b_id) for b_id in batch_ids]
                yield batch

    def update_scales(self, epoch, is_master_node=False, *args, **kwargs):
        if (epoch in self.scale_ep_intervals) and self.scale_inc:  # Training mode
            self.min_crop_size_w += int(self.min_crop_size_w * self.scale_inc_factor)
            self.max_crop_size_w += int(self.max_crop_size_w * self.scale_inc_factor)

            self.min_crop_size_h += int(self.min_crop_size_h * self.scale_inc_factor)
            self.max_crop_size_h += int(self.max_crop_size_h * self.scale_inc_factor)

            self.img_batch_tuples = _image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.num_replicas,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h
            )
            if is_master_node:
                logger.log('Scales updated in {}'.format(self.__class__.__name__))
                logger.log('Min. scale: {}->{}, Max.scale: {}->{}'.format(self.min_scale - self.scale_inc_factor,
                                                                          self.min_scale,
                                                                          self.max_scale - self.scale_inc_factor,
                                                                          self.max_scale))
                logger.log("New scales: {}".format(self.img_batch_tuples))

    def __repr__(self):
        repr_str = '{}('.format(self.__class__.__name__)
        repr_str += '\n\t base_im_size=(h={}, w={}), ' \
                    '\n\t base_batch_size={} ' \
                    '\n\t scales={} ' \
                    '\n\t scale_inc={} ' \
                    '\n\t scale_inc_factor={} ' \
                    '\n\t ep_intervals={}'.format(self.crop_size_h, self.crop_size_w,
                                                  self.batch_size_gpu0,
                                                  self.img_batch_tuples,
                                                  self.scale_inc,
                                                  self.scale_inc_factor,
                                                  self.scale_ep_intervals)
        repr_str += '\n )'
        return repr_str
