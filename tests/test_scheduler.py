#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import numpy as np
import random
from pprint import pprint
from tqdm import tqdm
from typing import Optional, Union, List
import sys

sys.path.append("..")

from options.opts import get_training_arguments
from optim.scheduler import build_scheduler
from utils import logger

LR_TOLERANCE = 1e-5

MAX_LRS = np.linspace(0.001, 0.1, 10)
WARMUP_ITERATIONS = [None, 100, 1000, 10000]

BATCH_SIZE = 100
DATASET_SIZE = 20000


def run_test(
    scheduler, num_epochs: int, num_batches: int, return_all_lrs: Optional[bool] = False
) -> Union[List, float]:
    end_lr = [] if return_all_lrs else 0.0
    curr_iter = 0
    for ep in range(num_epochs):
        for b in range(num_batches):
            lr = scheduler.get_lr(ep, curr_iter=curr_iter)
            curr_iter += 1

        # keep only epoch-wise LRs
        if return_all_lrs:
            end_lr.append(lr)
        else:
            end_lr = lr

    return end_lr


def test_polynomial_scheduler(*args, **kwargs):
    opts = get_training_arguments(parse_args=True)
    setattr(opts, "scheduler.max_iterations", 100000)
    setattr(opts, "scheduler.name", "polynomial")
    num_iterations = getattr(opts, "scheduler.max_iterations", 100000)
    num_batches = DATASET_SIZE // BATCH_SIZE
    num_epochs = num_iterations // num_batches

    # Test for iteration-based samplers
    setattr(opts, "scheduler.is_iteration_based", True)
    test_failed = 0
    test_passed = 0
    total_tests = 0
    failed_test_logs = []
    with tqdm(total=len(WARMUP_ITERATIONS) * len(MAX_LRS)) as pbar:
        for warmup_iteration in WARMUP_ITERATIONS:
            setattr(opts, "scheduler.warmup_iterations", warmup_iteration)
            for start_lr in MAX_LRS:
                end_lr = round(start_lr / random.randint(2, 10), 5)
                setattr(opts, "scheduler.polynomial.start_lr", start_lr)
                setattr(opts, "scheduler.polynomial.end_lr", end_lr)
                scheduler = build_scheduler(opts)
                lr = run_test(
                    scheduler=scheduler, num_epochs=num_epochs, num_batches=num_batches
                )
                diff = end_lr - lr
                if abs(diff) > LR_TOLERANCE:
                    failed_test_logs.append("Test failed for end_lr={}".format(end_lr))
                    test_failed += 1
                else:
                    test_passed += 1
                total_tests += 1
                # update the progress bar
                pbar.update(1)

    print("")
    if total_tests == test_passed:
        logger.log("All tests passed for iteration-based polynomial scheduler")
    else:
        logger.warning(
            "Tests passed={} and Tests failed={} for iteration-based polynomial scheduler".format(
                test_passed, test_failed
            )
        )
        pprint(failed_test_logs)

    # Test for epoch-based samplers
    setattr(opts, "scheduler.is_iteration_based", False)
    setattr(opts, "scheduler.max_epochs", num_epochs)
    setattr(opts, "scheduler.adjust_period_for_epochs", True)
    test_failed = 0
    test_passed = 0
    total_tests = 0
    failed_test_logs = []
    with tqdm(total=len(WARMUP_ITERATIONS) * len(MAX_LRS)) as pbar:
        for warmup_iteration in WARMUP_ITERATIONS:
            setattr(opts, "scheduler.warmup_iterations", warmup_iteration)
            for start_lr in MAX_LRS:
                end_lr = round(start_lr / random.randint(2, 10), 5)
                setattr(opts, "scheduler.polynomial.start_lr", start_lr)
                setattr(opts, "scheduler.polynomial.end_lr", end_lr)
                scheduler = build_scheduler(opts)
                lr = run_test(
                    scheduler=scheduler, num_epochs=num_epochs, num_batches=num_batches
                )
                diff = end_lr - lr
                if abs(diff) > 1e-3:
                    failed_test_logs.append(
                        "Test failed for end_lr={}. Got={}".format(end_lr, lr)
                    )
                    test_failed += 1
                else:
                    test_passed += 1
                total_tests += 1
                # update the progress bar
                pbar.update(1)

    print("")
    if total_tests == test_passed:
        logger.log("All tests passed for epoch-based polynomial scheduler")
    else:
        logger.warning(
            "Tests passed={} and Tests failed={} for epoch-based polynomial scheduler".format(
                test_passed, test_failed
            )
        )
        pprint(failed_test_logs)


def test_cosine_scheduler(*args, **kwargs):
    opts = get_training_arguments(parse_args=True)
    setattr(opts, "scheduler.max_iterations", 100000)
    setattr(opts, "scheduler.name", "cosine")

    num_iterations = getattr(opts, "scheduler.max_iterations", 100000)
    num_batches = DATASET_SIZE // BATCH_SIZE
    num_epochs = num_iterations // num_batches

    # first test for iteration-based samplers
    setattr(opts, "scheduler.is_iteration_based", True)
    test_failed = 0
    test_passed = 0
    total_tests = 0
    failed_test_logs = []
    with tqdm(total=len(WARMUP_ITERATIONS) * len(MAX_LRS)) as pbar:
        for warmup_iteration in WARMUP_ITERATIONS:
            setattr(opts, "scheduler.warmup_iterations", warmup_iteration)
            for start_lr in MAX_LRS:
                end_lr = round(start_lr / random.randint(2, 10), 5)
                setattr(opts, "scheduler.cosine.max_lr", start_lr)
                setattr(opts, "scheduler.cosine.min_lr", end_lr)
                scheduler = build_scheduler(opts)
                lr = run_test(
                    scheduler=scheduler, num_epochs=num_epochs, num_batches=num_batches
                )
                diff = end_lr - lr
                if abs(diff) > LR_TOLERANCE:
                    failed_test_logs.append("Test failed for end_lr={}".format(end_lr))
                    test_failed += 1
                else:
                    test_passed += 1
                total_tests += 1
                # update the progress bar
                pbar.update(1)

    print("")
    if total_tests == test_passed:
        logger.log("All tests passed for iteration-based cosine scheduler")
    else:
        logger.warning(
            "Tests passed={} and Tests failed={} for iteration-based cosine scheduler".format(
                test_passed, test_failed
            )
        )
        pprint(failed_test_logs)

    # Test for epoch-based samplers
    setattr(opts, "scheduler.is_iteration_based", False)
    setattr(opts, "scheduler.max_epochs", num_epochs)
    setattr(opts, "scheduler.adjust_period_for_epochs", True)
    test_failed = 0
    test_passed = 0
    total_tests = 0
    failed_test_logs = []
    with tqdm(total=len(WARMUP_ITERATIONS) * len(MAX_LRS)) as pbar:
        for warmup_iteration in WARMUP_ITERATIONS:
            setattr(opts, "scheduler.warmup_iterations", warmup_iteration)
            for start_lr in MAX_LRS:
                end_lr = round(start_lr / random.randint(2, 10), 5)
                setattr(opts, "scheduler.cosine.max_lr", start_lr)
                setattr(opts, "scheduler.cosine.min_lr", end_lr)
                scheduler = build_scheduler(opts)
                lr = run_test(
                    scheduler=scheduler, num_epochs=num_epochs, num_batches=num_batches
                )
                diff = end_lr - lr
                if abs(diff) > 1e-3:
                    failed_test_logs.append(
                        "Test failed for end_lr={}. Got={}".format(end_lr, lr)
                    )
                    test_failed += 1
                else:
                    test_passed += 1
                total_tests += 1
                # update the progress bar
                pbar.update(1)

    print("")
    if total_tests == test_passed:
        logger.log("All tests passed for epoch-based cosine scheduler")
    else:
        logger.warning(
            "Tests passed={} and Tests failed={} for epoch-based cosine scheduler".format(
                test_passed, test_failed
            )
        )
        pprint(failed_test_logs)


def test_fixed_scheduler(*args, **kwargs):
    opts = get_training_arguments(parse_args=True)
    setattr(opts, "scheduler.max_iterations", 100000)
    setattr(opts, "scheduler.name", "fixed")

    num_iterations = getattr(opts, "scheduler.max_iterations", 100000)
    num_batches = DATASET_SIZE // BATCH_SIZE
    num_epochs = num_iterations // num_batches

    # Test for iteration-based samplers
    setattr(opts, "scheduler.is_iteration_based", True)
    test_failed = 0
    test_passed = 0
    total_tests = 0
    failed_test_logs = []
    with tqdm(total=len(WARMUP_ITERATIONS) * len(MAX_LRS)) as pbar:
        for warmup_iteration in WARMUP_ITERATIONS:
            setattr(opts, "scheduler.warmup_iterations", warmup_iteration)
            for start_lr in MAX_LRS:
                setattr(opts, "scheduler.fixed.lr", start_lr)
                scheduler = build_scheduler(opts)
                lr = run_test(
                    scheduler=scheduler, num_epochs=num_epochs, num_batches=num_batches
                )
                diff = start_lr - lr
                if abs(diff) > LR_TOLERANCE:
                    failed_test_logs.append(
                        "Test failed for end_lr={}".format(start_lr)
                    )
                    test_failed += 1
                else:
                    test_passed += 1
                total_tests += 1
                # update the progress bar
                pbar.update(1)

    print("")
    if total_tests == test_passed:
        logger.log("All tests passed for iteration-based fixed scheduler")
    else:
        logger.warning(
            "Tests passed={} and Tests failed={} for iteration-based fixed scheduler".format(
                test_passed, test_failed
            )
        )
        pprint(failed_test_logs)

    # Test for epoch-based samplers
    setattr(opts, "scheduler.is_iteration_based", False)
    setattr(opts, "scheduler.max_epochs", num_epochs)
    setattr(opts, "scheduler.adjust_period_for_epochs", True)
    test_failed = 0
    test_passed = 0
    total_tests = 0
    failed_test_logs = []
    with tqdm(total=len(WARMUP_ITERATIONS) * len(MAX_LRS)) as pbar:
        for warmup_iteration in WARMUP_ITERATIONS:
            setattr(opts, "scheduler.warmup_iterations", warmup_iteration)
            for start_lr in MAX_LRS:
                setattr(opts, "scheduler.fixed.lr", start_lr)
                scheduler = build_scheduler(opts)
                lr = run_test(
                    scheduler=scheduler, num_epochs=num_epochs, num_batches=num_batches
                )
                diff = start_lr - lr
                if abs(diff) > 1e-3:
                    failed_test_logs.append(
                        "Test failed for lr={}. Got={}".format(start_lr, lr)
                    )
                    test_failed += 1
                else:
                    test_passed += 1
                total_tests += 1
                # update the progress bar
                pbar.update(1)

    print("")
    if total_tests == test_passed:
        logger.log("All tests passed for epoch-based fixed scheduler")
    else:
        logger.warning(
            "Tests passed={} and Tests failed={} for epoch-based fixed scheduler".format(
                test_passed, test_failed
            )
        )
        pprint(failed_test_logs)


def test_scheduler(*args, **kwargs):
    logger.info("Running tests with Polynomial schedule")
    test_polynomial_scheduler(*args, **kwargs)
    logger.double_dash_line()

    logger.info("Running tests with Cosine schedule")
    test_cosine_scheduler(*args, **kwargs)
    logger.double_dash_line()

    logger.info("Running tests with fixed schedule")
    test_fixed_scheduler(*args, **kwargs)
    logger.double_dash_line()


if __name__ == "__main__":
    test_scheduler()
