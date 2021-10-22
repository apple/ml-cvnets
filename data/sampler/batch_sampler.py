#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import random
import argparse
from typing import Optional
from common import DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT

from . import register_sampler, BaseSamplerDDP, BaseSamplerDP


@register_sampler(name="batch_sampler")
class BatchSampler(BaseSamplerDP):
    """
        Standard Batch Sampler for DP
    """
    def __init__(self, opts, n_data_samples: int, is_training: Optional[bool] = False):
        """

        :param opts: arguments
        :param n_data_samples: number of data samples in the dataset
        :param is_training: Training or evaluation mode (eval mode includes validation mode)
        """
        super(BatchSampler, self).__init__(opts=opts, n_data_samples=n_data_samples, is_training=is_training)
        crop_size_w: int = getattr(opts, "sampler.bs.crop_size_width", DEFAULT_IMAGE_WIDTH)
        crop_size_h: int = getattr(opts, "sampler.bs.crop_size_height", DEFAULT_IMAGE_HEIGHT)

        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(self.img_indices)

        start_index = 0
        batch_size = self.batch_size_gpu0
        while start_index < self.n_samples:

            end_index = min(start_index + batch_size, self.n_samples)
            batch_ids = self.img_indices[start_index:end_index]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [(self.crop_size_h, self.crop_size_w, b_id) for b_id in batch_ids]
                yield batch

    def __repr__(self):
        repr_str = '{}('.format(self.__class__.__name__)
        repr_str += '\n \t base_im_size=(h={}, w={})\n \t base_batch_size={}\n\t'.format(self.crop_size_h,
                                                                                         self.crop_size_w,
                                                                                         self.batch_size_gpu0)
        repr_str += '\n)'
        return repr_str

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Batch sampler", description="Arguments related to Batch sampler")
        group.add_argument('--sampler.bs.crop-size-width', default=DEFAULT_IMAGE_WIDTH, type=int,
                           help='Base crop size (along width) during training')
        group.add_argument('--sampler.bs.crop-size-height', default=DEFAULT_IMAGE_HEIGHT, type=int,
                           help='Base crop size (along height) during training')
        return parser


@register_sampler(name="batch_sampler_ddp")
class BatchSamplerDDP(BaseSamplerDDP):
    """
        Standard Batch Sampler for DDP
    """
    def __init__(self, opts, n_data_samples: int, is_training: Optional[bool] = False):
        """

        :param opts: arguments
        :param n_data_samples: number of data samples in the dataset
        :param is_training: Training or evaluation mode (eval mode includes validation mode)
        """
        super(BatchSamplerDDP, self).__init__(opts=opts, n_data_samples=n_data_samples, is_training=is_training)
        crop_size_w: int = getattr(opts, "sampler.bs.crop_size_width", DEFAULT_IMAGE_WIDTH)
        crop_size_h: int = getattr(opts, "sampler.bs.crop_size_height", DEFAULT_IMAGE_HEIGHT)

        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(self.img_indices)
            indices_rank_i = self.img_indices[self.rank:len(self.img_indices):self.num_replicas]
        else:
            indices_rank_i = self.img_indices[self.rank:len(self.img_indices):self.num_replicas]

        start_index = 0
        batch_size = self.batch_size_gpu0
        while start_index < self.n_samples_per_replica:
            end_index = min(start_index + batch_size, self.n_samples_per_replica)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != batch_size:
                batch_ids += indices_rank_i[:(batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [(self.crop_size_h, self.crop_size_w, b_id) for b_id in batch_ids]
                yield batch

    def __repr__(self):
        repr_str = '{}('.format(self.__class__.__name__)
        repr_str += '\n \t base_im_size=(h={}, w={})\n \t base_batch_size={}\n\t'.format(self.crop_size_h,
                                                                                         self.crop_size_w,
                                                                                         self.batch_size_gpu0)
        repr_str += '\n)'
        return repr_str