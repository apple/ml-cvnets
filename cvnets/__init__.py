#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from cvnets.misc.common import parameter_list
from cvnets.layers import arguments_nn_layers
from cvnets.models import arguments_model, get_model
from cvnets.misc.averaging_utils import arguments_ema, EMA
from cvnets.misc.profiler import module_profile
from cvnets.models.detection.base_detection import DetectionPredTuple