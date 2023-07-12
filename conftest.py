#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import signal
from types import FrameType
from typing import Optional

import pytest

session_timed_out = False


def handle_timeout(signum: int, frame: Optional[FrameType] = None) -> None:
    global session_timed_out
    session_timed_out = True
    # Call fail() to capture the output of the test.
    pytest.fail("timeout")


def pytest_sessionstart():
    timeout = os.environ.get("PYTEST_GLOBAL_TIMEOUT", "")
    if not timeout:
        return
    if timeout.endswith("s"):
        timeout = int(timeout[:-1])
    elif timeout.endswith("m"):
        timeout = int(timeout[:-1]) * 60
    else:
        raise ValueError(
            f"Timeout value {timeout} should either end with 'm' (minutes) or 's' (seconds)."
        )

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)


def pytest_runtest_logfinish(nodeid, location):
    if session_timed_out:
        pytest.exit("timeout")
