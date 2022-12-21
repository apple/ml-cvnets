#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.append("..")

from data.sampler import build_sampler
from tests.configs import get_config


# We use a batch size of 1 to catch error that may arise due to reshaping operations inside the model
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_repeats", [12, 4])
@pytest.mark.parametrize("trunc_ra_sampler", [True, False])
def test_model(
    config_file: str, batch_size: int, num_repeats: int, trunc_ra_sampler: bool
):
    opts = get_config(config_file=config_file)
    setattr(opts, "common.debug_mode", True)
    setattr(opts, "dataset.train_batch_size0", batch_size)
    setattr(opts, "sampler.num_repeats", num_repeats)
    setattr(opts, "sampler.truncated_repeat_aug_sampler", trunc_ra_sampler)
    n_data_samples = 1000
    sampler = build_sampler(opts, n_data_samples=n_data_samples, is_training=True)

    np.testing.assert_equal(
        len(sampler), n_data_samples * (1 if trunc_ra_sampler else num_repeats)
    )


def pytest_generate_tests(metafunc):
    configs = [
        str(x) for x in Path("config").rglob("**/*.yaml") if "tune" not in str(x)
    ]
    metafunc.parametrize("config_file", configs)


if __name__ == "__main__":
    test_model()
