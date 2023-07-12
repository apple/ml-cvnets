#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import argparse
import copy
import os
import time
from typing import Optional

import requests

from common import TMP_CACHE_LOC, if_test_env
from utils import logger
from utils.ddp_utils import dist_barrier, is_start_rank_node


def get_basic_local_path(
    opts: argparse.Namespace,
    path: str,
    cache_loc: str = TMP_CACHE_LOC,
    force_delete: Optional[bool] = None,
    use_start_rank: bool = True,
    sync_ranks: bool = True,
    *args,
    **kwargs
) -> str:
    """
    If File name is a URL, download to TMP_CACHE_LOC and then return the local path. Otherwise, don't do anything
    """
    if force_delete is None:
        # An alternative approach is to move this logic to the argument's default value in the function signature:
        #     def get_basic_local_path(..., force_delete = if_test_env(False, otherwise=True), ...):
        # But that won't always worker because pytest may set PYTEST_CURRENT_TEST env set loading this module.
        force_delete = if_test_env(False, otherwise=True)

    if (
        path.find("s3://") > -1
        or path.find("http://") > -1
        or path.find("https://") > -1
    ):
        url_path = copy.deepcopy(path)
        ckpt_name = path.split(os.sep)[-1]
        local_path = "{}/{}".format(cache_loc, ckpt_name)
        local_path = str(local_path).strip()

        if os.path.isfile(local_path) and force_delete:
            # If file exists, remove it and then download again
            # This is important because if we are downloading from bolt tasks, then checkpoint names are the same
            if use_start_rank:
                # remove files from start rank only
                if is_start_rank_node(opts):
                    os.remove(local_path)
                else:
                    while not os.path.isfile(local_path):
                        time.sleep(if_test_env(0, otherwise=1))
                        continue
            else:
                # All ranks in DDP can remove the files
                os.remove(local_path)

        if not os.path.isfile(local_path):
            if not use_start_rank or is_start_rank_node(opts):
                _download_file(url_path, local_path)
            else:
                while os.path.isfile(local_path):
                    # download file on start rank and let other ranks keep waiting till file is downloaded
                    # in DDP, download file in all ranks
                    time.sleep(if_test_env(0, otherwise=1))
                    continue

        if getattr(opts, "ddp.use_distributed", False) and sync_ranks:
            # synchronize between processes
            dist_barrier()
        return local_path
    return os.path.expanduser(path)


def _download_file(url_path: str, dest_loc: str) -> None:
    """
    Helper function to download a file with proxy (used when file fails)
    """
    response = requests.get(url_path, stream=True)
    if response.status_code == 403:
        # try with the HTTP/HTTPS proxy from ENV
        proxies = {
            "https": os.environ.get("HTTPS_PROXY", None),
            "http": os.environ.get("HTTP_PROXY", None),
        }
        response = requests.get(url_path, stream=True, proxies=proxies)

    if response.status_code == 200:
        with open(dest_loc, "wb") as f:
            f.write(response.raw.read())
    else:
        logger.error("Unable to download file {}".format(url_path))
