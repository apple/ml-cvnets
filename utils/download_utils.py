#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from utils.download_utils_base import get_basic_local_path

try:
    from internal.utils.blobby_utils import get_local_path_blobby

    get_local_path = get_local_path_blobby

except ModuleNotFoundError as mnfe:
    get_local_path = get_basic_local_path
