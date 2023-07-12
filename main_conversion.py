#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
from typing import List, Optional

import torch

from common import TMP_RES_FOLDER
from cvnets import get_model
from options.opts import get_conversion_arguments
from utils import logger
from utils.checkpoint_utils import CHECKPOINT_EXTN
from utils.pytorch_to_coreml import convert_pytorch_to_coreml


def main_worker_conversion(args: Optional[List[str]] = None):
    opts = get_conversion_arguments(args=args)

    # set coreml conversion flag to true
    setattr(opts, "common.enable_coreml_compatible_module", True)

    norm_layer = getattr(opts, "model.normalization.name", "batch_norm")
    if norm_layer.find("sync") > -1:
        norm_layer = norm_layer.replace("sync_", "")
        setattr(opts, "model.normalization.name", norm_layer)

    model = get_model(opts)
    model_name = model.__class__.__name__

    model.eval()

    coreml_extn = getattr(opts, "conversion.coreml_extn", "mlmodel")

    results_folder = getattr(opts, "common.results_loc", TMP_RES_FOLDER)
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    model_dst_loc = "{}/{}.{}".format(results_folder, model_name, coreml_extn)

    if os.path.isfile(model_dst_loc):
        os.remove(model_dst_loc)

    conversion_success = False
    try:
        converted_models_dict = convert_pytorch_to_coreml(
            opts=opts, pytorch_model=model
        )
        coreml_model = converted_models_dict["coreml"]
        jit_model = converted_models_dict["jit"]
        jit_optimized = converted_models_dict["jit_optimized"]

        coreml_model.save(model_dst_loc)

        torch.jit.save(
            jit_model,
            model_dst_loc.replace(f".{coreml_extn}", f"_jit.{CHECKPOINT_EXTN}"),
        )
        jit_optimized._save_for_lite_interpreter(
            model_dst_loc.replace(f".{coreml_extn}", f"_jit_opt.{CHECKPOINT_EXTN}")
        )

        logger.log("PyTorch model converted to CoreML successfully.")
        logger.log("CoreML model location: {}".format(model_dst_loc))
        conversion_success = True
    except Exception as e:
        logger.error(
            "PyTorch to CoreML conversion failed. See below for error details:\n {}".format(
                e
            )
        )

    if conversion_success:
        try:
            from internal.utils.upload_utils import upload_to_server

            # upload function is only available for internal usage
            upload_to_server(
                opts=opts,
                coreml_extn=coreml_extn,
                file_loc=model_dst_loc,
                model_name=model_name,
            )
        except ModuleNotFoundError as mne:
            pass
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main_worker_conversion()
