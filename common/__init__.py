#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import os
from pathlib import Path
from typing import Any

LIBRARY_ROOT = Path(__file__).parent.parent

MIN_TORCH_VERSION = "1.11.0"

SUPPORTED_IMAGE_EXTNS = [".png", ".jpg", ".jpeg"]  # Add image formats here
SUPPORTED_MODALITIES = ["image", "video"]
SUPPORTED_VIDEO_CLIP_VOTING_FN = ["sum", "max"]
SUPPORTED_VIDEO_READER = ["pyav", "decord"]

DEFAULT_IMAGE_WIDTH = DEFAULT_IMAGE_HEIGHT = 256
DEFAULT_IMAGE_CHANNELS = 3
DEFAULT_VIDEO_FRAMES = 8
DEFAULT_LOG_FREQ = 500

DEFAULT_ITERATIONS = 300000
DEFAULT_EPOCHS = 300
DEFAULT_MAX_ITERATIONS = DEFAULT_MAX_EPOCHS = 10000000

TMP_RES_FOLDER = "results_tmp"

TMP_CACHE_LOC = "/tmp/cvnets"

Path(TMP_CACHE_LOC).mkdir(parents=True, exist_ok=True)


def is_test_env() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def if_test_env(then: Any, otherwise: Any) -> Any:
    return then if "PYTEST_CURRENT_TEST" in os.environ else otherwise
