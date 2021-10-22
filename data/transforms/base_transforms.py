#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict


class BaseTransformation(object):
    """
        Base class for transformations
    """
    def __init__(self, opts):
        super(BaseTransformation, self).__init__()
        self.opts = opts

    def __call__(self, data: Dict):
        raise NotImplementedError

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser