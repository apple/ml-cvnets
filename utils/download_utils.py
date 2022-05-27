#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import copy
import torch.distributed as dist
from urllib import request

from common import TMP_CACHE_LOC
from utils.ddp_utils import is_start_rank_node


try:
    from utils_internal.blobby_utils import get_local_path_blobby

    def get_local_path(opts, path, recursive=False, *args, **kwargs):
        """
        If File is in S3, download to /tmp and then return the local path. Otherwise, don't do anything
        """
        return get_local_path_blobby(opts=opts, path=path, recursive=recursive)


except ModuleNotFoundError as mnfe:

    def get_local_path(opts, path, *args, **kwargs):
        """
        If File name is a URL, download to TMP_CACHE_LOC and then return the local path. Otherwise, don't do anything
        """
        if (
            path.find("s3://") > -1
            or path.find("http://") > -1
            or path.find("https://") > -1
        ):
            url_path = copy.deepcopy(path)
            ckpt_name = path.split(os.sep)[-1]
            local_path = "{}/{}".format(TMP_CACHE_LOC, ckpt_name)
            local_path = str(local_path).strip()

            if os.path.isfile(local_path) and is_start_rank_node(opts):
                # If file exists, remove it and then download again
                # This is important because if we are downloading from bolt tasks, then checkpoint names are the same
                os.remove(local_path)

            if not os.path.isfile(local_path) and is_start_rank_node(opts):
                request.urlretrieve(url_path, local_path)

            if getattr(opts, "ddp.use_distributed", False):
                # syncronize between processes
                dist.barrier()
            return local_path
        return path
