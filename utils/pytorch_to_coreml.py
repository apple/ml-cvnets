#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import Tensor
import coremltools as ct
from typing import Optional, Dict, Tuple, Union
import numpy as np

from utils.tensor_utils import create_rand_tensor
from torch.utils.mobile_optimizer import optimize_for_mobile


def convert_pytorch_to_coreml(
        opts,
        pytorch_model: torch.nn.Module,
        input_tensor: Optional[torch.Tensor] = None,
        *args, **kwargs
) -> Dict:
    """
    Convert Pytorch model to CoreML

    :param opts: Arguments
    :param pytorch_model: Pytorch model that needs to be converted to JIT or CoreML
    :param input_tensor: Input tensor, usually a 4-dimensional tensor of shape Batch x 3 x Height x Width
    :return: CoreML model or package
    """
    if input_tensor is None:
        input_tensor = create_rand_tensor(opts=opts, device="cpu")

    if pytorch_model.training:
        pytorch_model.eval()

    with torch.no_grad():
        pytorch_out = pytorch_model(input_tensor)

        jit_model = torch.jit.trace(pytorch_model, input_tensor)
        jit_out = jit_model(input_tensor)

        jit_model_optimized = optimize_for_mobile(jit_model)
        jit_optimzied_out = jit_model_optimized(input_tensor)

        assertion_check(py_out=pytorch_out, jit_out=jit_out)
        assertion_check(py_out=pytorch_out, jit_out=jit_optimzied_out)

        coreml_model = ct.convert(
            model=jit_model,
            inputs=[ct.ImageType(name="input", shape=input_tensor.shape)],
            convert_to="neuralnetwork",
            #preprocessing_args={"scale": 1.0/255.0},
            #minimum_deployment_target=ct.target.iOS15,
            #compute_precision=ct.precision.FLOAT16
        )
        return {
            "coreml": coreml_model,
            "jit": jit_model,
            "jit_optimized": jit_model_optimized
        }


def assertion_check(py_out: Union[Tensor, Dict, Tuple], jit_out: Union[Tensor, Dict, Tuple]) -> None:
    if isinstance(py_out, Dict):
        assert isinstance(jit_out, Dict)
        keys = py_out.keys()
        for k in keys:
            np.testing.assert_almost_equal(py_out[k].cpu().numpy(), jit_out[k].cpu().numpy(),
                                           decimal=3,
                                           verbose=True)
    elif isinstance(py_out, Tensor):
        assert isinstance(jit_out, Tensor)
        np.testing.assert_almost_equal(py_out.cpu().numpy(), jit_out.cpu().numpy(),
                                       decimal=3,
                                       verbose=True)
    elif isinstance(py_out, Tuple):
        assert isinstance(jit_out, Tuple)
        for x, y in zip(py_out, jit_out):
            np.testing.assert_almost_equal(x.cpu().numpy(), y.cpu().numpy(), decimal=3, verbose=True)

    else:
        raise NotImplementedError("Only Dictionary[Tensors] or Tuple[Tensors] or Tensors are supported as outputs")
