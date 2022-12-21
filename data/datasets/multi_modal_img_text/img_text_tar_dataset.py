#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
from typing import Optional, Dict
import numpy as np
import torch
import argparse
from PIL import Image, ImageFile
import io
import pickle
import multiprocessing
from multiprocessing.pool import Pool
import tarfile
import glob

from utils import logger
from utils.download_utils import get_local_path
from utils.ddp_utils import dist_barrier

from .. import register_dataset
from .base_multi_modal_img_text import BaseMultiModalImgText

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def extract_content(tar_file, file_name):
    f = tar_file.extractfile(file_name)
    return f.read()


def decode_image(byte_data):
    return Image.open(io.BytesIO(byte_data)).convert("RGB")


def decode_text(byte_data):
    return byte_data.decode("utf-8")


def async_download_file_from_s3(
    opts, tar_file_name: str, cache_loc: str, *args, **kwargs
) -> None:
    # async download files form s3
    local_path = get_local_path(
        opts=opts,
        path=tar_file_name,
        cache_loc=cache_loc,
        quiet_download=True,
        force_delete=False,
        use_start_rank=False,
        sync_ranks=False,
    )

    # now extract the tar file and save the content as each separate file
    folder_name = local_path.replace(".tar.gz", "")
    with tarfile.open(local_path, "r:gz") as tar_file:
        tar_file.extractall(folder_name)

    # delete the tar file, to save space
    if os.path.isfile(local_path):
        os.remove(local_path)


@register_dataset(name="img_text_tar", task="multi_modal_img_text")
class ImgTextTarDataset(BaseMultiModalImgText):
    """
    ImgTextTarDataset class for datasets that store Image-Text pairs as tar files, each tar file with multiple pairs.

    Args:
        opts: command-line arguments
        is_training (Optional[bool]): A flag used to indicate training or validation mode. Default: True
        is_evaluation (Optional[bool]): A flag used to indicate evaluation (or inference) mode. Default: False

    """

    __separator = "-"
    __overlap_ratio = 10  # we have an overlap ratio of 10 files
    __file_extn = ".tar.gz"

    def __init__(
        self,
        opts,
        is_training: Optional[bool] = True,
        is_evaluation: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(
            opts=opts,
            is_training=is_training,
            is_evaluation=is_evaluation,
            *args,
            **kwargs,
        )
        self.zeros_shot_dataset = self.get_zero_shot_dataset()

        if is_training:
            dataset_metadata = self.get_dataset()

            total_files = 0
            if -1 in dataset_metadata.keys():
                # At key=-1, we store the information about total files.
                total_files = dataset_metadata.pop(-1)

            if total_files == 0:
                logger.error(
                    "Total files can't be 0. Please check if metadata has key -1, which stores the total number of files"
                )

            self.dataset: Dict = dataset_metadata
            self.total_pairs = total_files
            self.dataset_keys = list(self.dataset.keys())

            s3_bucket_path = getattr(
                self.opts,
                "dataset.multi_modal_img_text.img_text_tar.s3_bucket_path",
                None,
            )
            if s3_bucket_path is None:
                if self.is_master_node:
                    logger.log(
                        "{} needs the path of AWS bucket where data is stored.".format(
                            self.__class__.__name__
                        )
                    )

            self.s3_bucket_path = s3_bucket_path

            self._download_dataset()

    def get_dataset(self) -> Dict:
        if self.is_training:
            # read metadata file
            # metadata file is a dictionary storing the start image-text ids along with the tar file name.
            # Example {'0-18000': 'file_1.tar', 18000-29000', 'file_2.tar'}
            metadata_file_loc = getattr(
                self.opts,
                "dataset.multi_modal_img_text.img_text_tar.metadata_file",
                None,
            )
            if metadata_file_loc is None:
                if self.is_master_node:
                    logger.error(
                        "Please specify metadata file using "
                        "--dataset.multi-modal-img-text.img_text_tar.metadata-file for {}".format(
                            self.__class__.__name__
                        )
                    )

            metadata_file_local_path = get_local_path(
                self.opts, path=metadata_file_loc, force_delete=False
            )
            with open(metadata_file_local_path, "rb") as fp:
                metadata = pickle.load(fp)
                return metadata
        else:
            return {}

    def _download_dataset(self):
        if getattr(self.opts, "ddp.enable", False):
            if self.is_start_rank_node:
                logger.error(
                    "We need DDP for working with {} dataset".format(
                        self.__class__.__name__
                    )
                )

        # The total number of GPUs that a task is using is equal to the world size
        world_size = getattr(self.opts, "ddp.world_size", -1)
        if world_size is None or world_size == -1:
            if self.is_start_rank_node:
                logger.error("DDP world size should be greater than 1. Got: {}")

        # find the number of GPUs in each node
        n_gpus_per_node = torch.cuda.device_count()

        # Total number of GPUs = Total number of nodes * number of GPUs per Node
        n_nodes = max(1, world_size // n_gpus_per_node)

        # Find the node id based on current node rank
        # node_id = current_node_rank / n_gpus_per_node
        curr_node_rank = getattr(self.opts, "ddp.rank", None)
        if curr_node_rank is None:
            if self.is_start_rank_node:
                logger.error("Node rank can't be None.")
        node_id = curr_node_rank // n_gpus_per_node

        # Downloading the entire dataset on each node is not feasible. Instead, for each
        # node, we will download a subset of the dataset and learn from it.

        # Split the dataset almost equally among all nodes. The length of this split
        # is going to be the same as the number of nodes.
        node_wise_dataset_split = np.array_split(self.dataset_keys, n_nodes)

        # download files corresponding to ith node
        files_node_i = node_wise_dataset_split[node_id]

        # Dataset is organized as a dict where key corresponds to start_index of image-text pair and
        # value corresponds to the file name.

        # find the start and end image-text pair indexes for node_i.
        # Note that we overlap node_i and node_i+1 by at most 2 files
        start_idx_node_i = max(
            0, self.dataset_keys.index(files_node_i[0]) - self.__overlap_ratio
        )
        end_idx_node_i = min(
            len(self.dataset_keys),
            self.dataset_keys.index(files_node_i[-1]) + self.__overlap_ratio,
        )

        # Now, download the files concurrently using each rank on node i
        # Now, download the files concurrently using each rank on node i
        indexes_to_download_node_i = self.dataset_keys[start_idx_node_i:end_idx_node_i]
        indexes_to_download_node_i_rank_j = np.array_split(
            indexes_to_download_node_i, n_gpus_per_node
        )
        total_files_to_download = len(indexes_to_download_node_i)
        if self.is_start_rank_node:
            logger.log(f"Starting to downloading {total_files_to_download} files")

        current_device = torch.cuda.current_device()
        if getattr(
            self.opts,
            "dataset.multi_modal_img_text.img_text_tar.parallel_download",
            False,
        ):
            # download concurrently using many workers for each rank
            n_cpus = multiprocessing.cpu_count()
            n_process_per_gpu = max(
                1, n_cpus // torch.cuda.device_count()
            )  # max(1, min(4, n_cpus // torch.cuda.device_count()))
            with Pool(processes=n_process_per_gpu) as pool:
                pool.starmap(
                    async_download_file_from_s3,
                    [
                        (
                            self.opts,
                            os.path.join(
                                self.s3_bucket_path, self.dataset[img_text_idx]
                            ),
                            self.cache_loc,
                        )
                        for img_text_idx in indexes_to_download_node_i_rank_j[
                            current_device
                        ]
                    ],
                )
        else:
            # download sequentially (1 worker per rank)
            for count, img_text_idx in enumerate(
                indexes_to_download_node_i_rank_j[current_device]
            ):
                # Recall that dataset is organized as a dict where key corresponds to start_index of image-text pair
                # value corresponds to the tar file name.

                async_download_file_from_s3(
                    opts=self.opts,
                    tar_file_name=os.path.join(
                        self.s3_bucket_path, self.dataset[img_text_idx]
                    ),
                    cache_loc=self.cache_loc,
                )

                if count % 100 == 0 and self.is_start_rank_node:
                    n_files_downloaded = len(glob.glob(f"{self.cache_loc}/*"))
                    print(
                        f"Progress: {n_files_downloaded}/{total_files_to_download}",
                        end="\r",
                    )

        # synchronize between all DDP jobs
        if getattr(self.opts, "ddp.use_distributed", False):
            dist_barrier()

        if self.is_start_rank_node:
            n_files_downloaded = len(glob.glob(f"{self.cache_loc}/*"))
            logger.log(
                f"Download complete ({n_files_downloaded}/{total_files_to_download}). "
                f"Files are stored at: {self.cache_loc}"
            )

    def __len__(self):
        if self.zeros_shot_dataset is not None:
            return len(self.zeros_shot_dataset)
        return self.total_pairs

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add dataset-specific arguments to the parser."""
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--dataset.multi-modal-img-text.img-text-tar.metadata-file",
            type=str,
            default=None,
            help="Location of the metadata file",
        )

        group.add_argument(
            "--dataset.multi-modal-img-text.img-text-tar.s3-bucket-path",
            type=str,
            default=None,
            help="Path of the s3 bucket where data is stored",
        )

        group.add_argument(
            "--dataset.multi-modal-img-text.img-text-tar.parallel-download",
            action="store_true",
            help="Download the data in parallel on each rank of the DDP process",
        )

        return parser

    def get_dataset_pair(self, img_index):
        class_label = -1
        try:

            if img_index in self.dataset_keys:
                # file index is the same as the start index
                file_index = self.dataset_keys.index(img_index)
                # data index is 0 because file index is one of the start indices
                data_index = 0
                img_text_pair_id = self.dataset_keys[file_index]
            else:
                # find the index at which the element will be inserted.
                # Example: If we have an array of start indices as [0, 15, 35, 90] and
                # we want to find the position of image index 92, then the insertion index will be 4.
                insertion_idx = np.searchsorted(self.dataset_keys, img_index)

                # the image id corresponding to 92 is stored in file whose start index is 90.
                # So, the file index is one less than insertion index
                file_index = insertion_idx - 1

                img_text_pair_id = self.dataset_keys[file_index]

                # data index is delta between current value (92) and value at file index (90)
                data_index = img_index - img_text_pair_id

            # get the key corresponding to file index and retrieve the file name
            # concatenate the file name with cache location path
            tar_file_name_from_metadata = self.dataset[img_text_pair_id]

            tar_file_name = os.path.join(self.cache_loc, tar_file_name_from_metadata)
            # Tar file name is encoded as: <path>/<folder_start_end.tar.gz>
            # each file name in tar file is encoded as: <folder_dataIdx_image> and <folder_dataIdx_text>
            """ 
            Example.

            img_text_tar_dataset/00000000_0_1000.tar.gz
            |--- 00000000_0_image
            |--- 00000000_0_text
            |--- 00000000_1_image
            |--- 00000000_1_text
            |--- ...

            img_text_tar_dataset/00000000_1000_2000.tar.gz
            |--- 00000000_1000_image
            |--- 00000000_1000_text
            |--- 00000000_1001_image
            |--- 00000000_1001_text
            |--- ...
            """

            # remove the tar extension because we have extracted the data when downloaded
            tar_file_name = tar_file_name.replace(self.__file_extn, "")

            if not os.path.isdir(tar_file_name):
                async_download_file_from_s3(
                    opts=self.opts,
                    tar_file_name=os.path.join(
                        self.s3_bucket_path, tar_file_name_from_metadata
                    ),
                    cache_loc=self.cache_loc,
                )

            # Based on this, decode the folder information
            folder_name = tar_file_name.split(os.sep)[-1].split("_")[0]

            # adjust the data index with start_id offset
            start_id = tar_file_name.split(os.sep)[-1].split("_")[1]
            data_index = data_index + int(start_id)

            img_text_fname = f"{tar_file_name}/{folder_name}_{data_index}"
            with open(f"{img_text_fname}_image", "rb") as img_byte_data:
                input_img = decode_image(img_byte_data.read())

            with open(f"{img_text_fname}_text", "rb") as text_byte_data:
                captions_str = decode_text(text_byte_data.read())
        except Exception as e:
            logger.log("error loading {}. Error message: {}".format(img_index, str(e)))

            input_img = None
            captions_str = None
        return input_img, captions_str, class_label

    def __repr__(self):
        return "{}(\n\troot={}\n\t is_training={}\n\tzero_shot={}\n\tn_samples={}\n\t{}\n)".format(
            self.__class__.__name__,
            self.root,
            self.is_training,
            self.zeros_shot_dataset,
            self.__len__(),
            self.extra_transform_repr()
        )
