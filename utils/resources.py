#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

try:
    from internal.utils.resources import cpu_count
except ImportError:
    from multiprocessing import cpu_count

__all__ = ["cpu_count"]
