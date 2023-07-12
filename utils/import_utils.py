#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import importlib
import os
from typing import Sequence

from common import LIBRARY_ROOT
from utils import logger


def import_modules_from_folder(
    folder_name: str, extra_roots: Sequence[str] = ()
) -> None:
    """Automatically import all modules from public library root folder, in addition
    to the @extra_roots directories.

    The @folder_name directory must exist in LIBRARY_ROOT, but existence in @extra_roots
    is optional.

    Args:
        folder_name: Name of the folder to search for its internal and public modules.
        extra_roots: By default, this function only imports from
            `LIBRARY_ROOT/{folder_name}/**/*.py`. For any extra_root provided, it will
            also import `LIBRARY_ROOT/{extra_root}/{folder_name}/**/*.py` modules.
    """
    if not LIBRARY_ROOT.joinpath(folder_name).exists():
        logger.error(
            f"{folder_name} doesn't exist in the public library root directory."
        )

    for base_dir in [".", *extra_roots]:
        for path in LIBRARY_ROOT.glob(os.path.join(base_dir, folder_name, "**/*.py")):
            filename = path.name
            if filename[0] not in (".", "_"):
                module_name = str(
                    path.relative_to(LIBRARY_ROOT).with_suffix("")
                ).replace(os.sep, ".")
                importlib.import_module(module_name)
