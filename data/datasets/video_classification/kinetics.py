#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os.path
from typing import Optional, Tuple, List, Union
import torch
import pathlib
import glob
import argparse
import pickle

from utils import logger
from utils.download_utils import get_local_path
from utils.ddp_utils import is_master

from .. import register_dataset
from ..dataset_base import BaseImageDataset
from ...transforms import video as T
from ...video_reader import get_video_reader
from ...collate_fns import register_collate_fn


@register_dataset(name="kinetics", task="video_classification")
class KineticsDataset(BaseImageDataset):
    """
    Dataset class for the Kinetics dataset

    Args:
        opts: command-line arguments
        is_training (Optional[bool]): A flag used to indicate training or validation mode. Default: True
        is_evaluation (Optional[bool]): A flag used to indicate evaluation (or inference) mode. Default: False
    """

    def __init__(
        self,
        opts,
        is_training: Optional[bool] = True,
        is_evaluation: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> None:

        super(KineticsDataset, self).__init__(
            opts=opts, is_training=is_training, is_evaluation=is_evaluation
        )

        if not os.path.isdir(self.root):
            logger.error("Directory does not exist: {}".format(self.root))

        pyav_video_reader = get_video_reader(opts=opts, is_training=is_training)

        if is_training:
            metadata_file = getattr(opts, "dataset.kinetics.metadata_file_train", None)
        else:
            metadata_file = getattr(opts, "dataset.kinetics.metadata_file_val", None)

        if metadata_file is not None:
            # internally, we take care that master node only downloads the file
            metadata_file = get_local_path(opts=opts, path=metadata_file)
            with open(metadata_file, "rb") as f:
                self.samples = pickle.load(f)
            assert isinstance(self.samples, List)
        else:
            # each folder is a class
            class_names = sorted(
                (f.name for f in pathlib.Path(self.root).iterdir() if f.is_dir())
            )

            samples = []
            extensions = ["avi", "mp4"]
            for cls_idx in range(len(class_names)):
                cls_name = class_names[cls_idx]
                class_folder = os.path.join(self.root, cls_name)
                for video_path in glob.glob(f"{class_folder}/*"):
                    file_extn = video_path.split(".")[-1]
                    if (
                        (file_extn in extensions)
                        and os.path.isfile(video_path)
                        and pyav_video_reader.check_video(filename=video_path)
                    ):
                        samples.append({"label": cls_idx, "video_path": video_path})
            self.samples = samples
            results_loc = getattr(opts, "common.results_loc", None)
            if is_master(opts):
                stage = "train" if is_training else "val"
                metadata_file_loc = f"{results_loc}/kinetics_metadata_{stage}.pkl"

                with open(metadata_file_loc, "wb") as f:
                    pickle.dump(self.samples, f)
                logger.log("Metadata file saved at: {}".format(metadata_file_loc))

        self.pyav_video_reader = pyav_video_reader

    def __len__(self):
        return len(self.samples)

    def _training_transforms(self, size: tuple or int):
        """

        :param size: crop size (H, W)
        :return: list of augmentation methods
        """
        aug_list = [
            T.RandomResizedCrop(opts=self.opts, size=size),
            T.RandomHorizontalFlip(opts=self.opts),
        ]
        return T.Compose(opts=self.opts, video_transforms=aug_list)

    def _validation_transforms(self, size: Union[Tuple, List, int]):
        """

        :param size: crop size (H, W)
        :return: list of augmentation methods
        """
        aug_list = [
            T.Resize(opts=self.opts),
            T.CenterCrop(opts=self.opts, size=size),
        ]

        return T.Compose(opts=self.opts, video_transforms=aug_list)

    def _evaluation_transforms(self, size: tuple):
        """

        :param size: crop size (H, W)
        :return: list of augmentation methods
        """
        return self._validation_transforms(size=size)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--dataset.kinetics.metadata-file-train",
            type=str,
            default=None,
            help="Metadata file for kinetics train set",
        )
        group.add_argument(
            "--dataset.kinetics.metadata-file-val",
            type=str,
            default=None,
            help="Metadata file for kinetics validation set",
        )
        return parser

    def __getitem__(self, batch_indexes_tup):
        (
            crop_size_h,
            crop_size_w,
            index,
            n_frames_to_sample,
            clips_per_video,
        ) = batch_indexes_tup
        if self.is_training:
            transform_fn = self._training_transforms(size=(crop_size_h, crop_size_w))
        else:  # same for validation and evaluation
            transform_fn = self._validation_transforms(size=(crop_size_h, crop_size_w))

        try:
            info: dict = self.samples[index]
            target = info["label"]

            # Default is Tensor of size [K, N, C, H, W].
            # If --dataset.kinetics.frame-stack-format="channel_first", then clip is of size [K, C, N, H, W]
            # here, K --> no. of clips, C --> Image channels, N --> Number of frames per clip, H --> Height, W --> Width
            input_video = self.pyav_video_reader.process_video(
                vid_filename=info["video_path"],
                n_frames_per_clip=n_frames_to_sample,
                clips_per_video=clips_per_video,
                video_transform_fn=transform_fn,
                is_training=self.is_training,
            )

            if input_video is None:
                logger.log("Corrupted video file: {}".format(info["video_path"]))
                input_video = self.pyav_video_reader.dummy_video(
                    clips_per_video=clips_per_video,
                    n_frames_to_sample=n_frames_to_sample,
                    height=crop_size_h,
                    width=crop_size_w,
                )

                data = {"image": input_video}
                target = getattr(self.opts, "loss.ignore_idx", -1)
            else:
                data = {"image": input_video}

        except Exception as e:
            logger.log("Unable to load index: {}. Error: {}".format(index, str(e)))
            input_video = self.pyav_video_reader.dummy_video(
                clips_per_video=clips_per_video,
                n_frames_to_sample=n_frames_to_sample,
                height=crop_size_h,
                width=crop_size_w,
            )

            target = getattr(self.opts, "loss.ignore_idx", -1)
            data = {"image": input_video}

        # target is a 0-dimensional tensor
        data["label"] = torch.LongTensor(size=(input_video.shape[0],)).fill_(target)

        return data

    def __repr__(self):
        from utils.tensor_utils import video_size_from_opts

        im_h, im_w, n_frames = video_size_from_opts(opts=self.opts)

        if self.is_training:
            transforms_str = self._training_transforms(size=(im_h, im_w))
        else:
            transforms_str = self._validation_transforms(size=(im_h, im_w))

        if hasattr(self.pyav_video_reader, "frame_transforms_str"):
            frame_transforms_str = self.pyav_video_reader.frame_transforms_str
        else:
            frame_transforms_str = None

        return "{}(\n\troot={}\n\tis_training={}\n\tsamples={}\n\tvideo_transforms={}\n\tframe_transforms={}\n)".format(
            self.__class__.__name__,
            self.root,
            self.is_training,
            self.__len__(),
            transforms_str,
            frame_transforms_str,
        )


@register_collate_fn(name="kinetics_collate_fn")
def kinetics_collate_fn(batch: List, opts):
    batch_size = len(batch)

    images = []
    labels = []
    for b in range(batch_size):
        b_label = batch[b]["label"]
        images.append(batch[b]["image"])
        labels.append(b_label)

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)

    # check for contiguous
    if not images.is_contiguous():
        images = images.contiguous()

    if not labels.is_contiguous():
        labels = labels.contiguous()

    return {"image": images, "label": labels}


@register_collate_fn(name="kinetics_collate_fn_train")
def kinetics_collate_fn_train(batch: List, opts):
    batch_size = len(batch)
    ignore_label = getattr(opts, "loss.ignore_idx", -1)

    images = []
    labels = []
    for b in range(batch_size):
        b_label = batch[b]["label"]
        if ignore_label in b_label:
            continue
        images.append(batch[b]["image"])
        labels.append(b_label)

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)

    # check for contiguous
    if not images.is_contiguous():
        images = images.contiguous()

    if not labels.is_contiguous():
        labels = labels.contiguous()

    return {"image": images, "label": labels}
