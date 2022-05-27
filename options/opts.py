#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Optional

from data.sampler import arguments_sampler
from data.collate_fns import arguments_collate_fn
from options.utils import load_config_file
from data.datasets import arguments_dataset
from cvnets import arguments_model, arguments_nn_layers, arguments_ema
from cvnets.anchor_generator import arguments_anchor_gen
from loss_fn import arguments_loss_fn
from optim import arguments_optimizer
from optim.scheduler import arguments_scheduler
from common import SUPPORTED_MODALITIES
from data.transforms import arguments_augmentation
from metrics import arguments_stats
from data.video_reader import arguments_video_reader
from cvnets.matcher_det import arguments_box_matcher
from utils import logger


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace_dict = vars(namespace)

        if len(values) > 0:
            override_dict = {}
            # values are list of key-value pairs
            for value in values:
                key = None
                try:
                    key, value = value.split("=")
                except ValueError as e:
                    logger.error(
                        "For override arguments, a key-value pair of the form key=value is expected"
                    )

                if key in namespace_dict:
                    value_namespace = namespace_dict[key]
                    if value_namespace is None and value is None:
                        value = None
                    elif value_namespace is None and value is not None:
                        # possibly a string or list of strings or list of integers

                        # check if string is a list or not
                        value = value.split(",")
                        if len(value) == 1:
                            # its a string
                            value = str(value[0])

                            # check if its empty string or not
                            if value == "" or value.lower() == "none":
                                value = None
                        else:
                            # its a list of integers or strings
                            try:
                                # convert to int
                                value = [int(v) for v in value]
                            except:
                                # pass because its a string
                                pass
                    else:
                        try:
                            if value.lower() == "true":  # check for boolean
                                value = True
                            elif value.lower() == "false":
                                value = False
                            else:
                                desired_type = type(value_namespace)
                                value = desired_type(value)
                        except ValueError as e:
                            logger.warning(
                                "Type mismatch while over-riding. Skipping key: {}".format(
                                    key
                                )
                            )
                            continue

                    override_dict[key] = value
            setattr(namespace, "override_args", override_dict)
        else:
            setattr(namespace, "override_args", None)


def arguments_common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        title="Common arguments", description="Common arguments"
    )

    group.add_argument("--common.seed", type=int, default=0, help="Random seed")
    group.add_argument(
        "--common.config-file", type=str, default=None, help="Configuration file"
    )
    group.add_argument(
        "--common.results-loc",
        type=str,
        default="results",
        help="Directory where results will be stored",
    )
    group.add_argument(
        "--common.run-label",
        type=str,
        default="run_1",
        help="Label id for the current run",
    )

    group.add_argument(
        "--common.resume", type=str, default=None, help="Resume location"
    )
    group.add_argument(
        "--common.finetune_imagenet1k",
        type=str,
        default=None,
        help="Checkpoint location to be used for finetuning",
    )
    group.add_argument(
        "--common.finetune_imagenet1k-ema",
        type=str,
        default=None,
        help="EMA Checkpoint location to be used for finetuning",
    )

    group.add_argument(
        "--common.mixed-precision", action="store_true", help="Mixed precision training"
    )
    group.add_argument(
        "--common.accum-freq",
        type=int,
        default=1,
        help="Accumulate gradients for this number of iterations",
    )
    group.add_argument(
        "--common.accum-after-epoch",
        type=int,
        default=0,
        help="Start accumulation after this many epochs",
    )
    group.add_argument(
        "--common.log-freq",
        type=int,
        default=100,
        help="Display after these many iterations",
    )
    group.add_argument(
        "--common.auto-resume",
        action="store_true",
        help="Resume training from the last checkpoint",
    )
    group.add_argument(
        "--common.grad-clip", type=float, default=None, help="Gradient clipping value"
    )
    group.add_argument(
        "--common.k-best-checkpoints",
        type=int,
        default=5,
        help="Keep k-best checkpoints",
    )

    group.add_argument(
        "--common.inference-modality",
        type=str,
        default="image",
        choices=SUPPORTED_MODALITIES,
        help="Inference modality. Image or videos",
    )

    group.add_argument(
        "--common.channels-last",
        action="store_true",
        default=False,
        help="Use channel last format during training. "
        "Note 1: that some models may not support it, so we recommend to use it with caution"
        "Note 2: Channel last format does not work with 1-, 2-, and 3- tensors. "
        "Therefore, we support it via custom collate functions",
    )

    group.add_argument(
        "--common.tensorboard-logging",
        action="store_true",
        help="Enable tensorboard logging",
    )
    group.add_argument(
        "--common.bolt-logging", action="store_true", help="Enable bolt logging"
    )

    group.add_argument(
        "--common.override-kwargs",
        nargs="*",
        action=ParseKwargs,
        help="Override arguments. Example. To override the value of --sampler.vbs.crop-size-width, "
        "we can pass override argument as "
        "--common.override-kwargs sampler.vbs.crop_size_width=512 \n "
        "Note that keys in override arguments do not contain -- or -",
    )

    group.add_argument(
        "--common.enable-coreml-compatible-module",
        action="store_true",
        help="Use coreml compatible modules (if applicable) during inference",
    )

    group.add_argument(
        "--common.debug-mode",
        action="store_true",
        help="You can use this flag for debugging purposes.",
    )

    return parser


def arguments_ddp(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        title="DDP arguments", description="DDP arguments"
    )
    group.add_argument("--ddp.disable", action="store_true", help="Don't use DDP")
    group.add_argument(
        "--ddp.rank", type=int, default=0, help="Node rank for distributed training"
    )
    group.add_argument(
        "--ddp.world-size", type=int, default=-1, help="World size for DDP"
    )
    group.add_argument("--ddp.dist-url", type=str, default=None, help="DDP URL")
    group.add_argument(
        "--ddp.dist-port",
        type=int,
        default=30786,
        help="DDP Port. Only used when --ddp.dist-url is not specified",
    )
    group.add_argument("--ddp.device-id", type=int, default=None, help="Device ID")
    group.add_argument(
        "--ddp.no-spawn", action="store_true", help="Don't use DDP with spawn"
    )
    group.add_argument(
        "--ddp.backend", type=str, default="nccl", help="DDP backend. Default is nccl"
    )
    group.add_argument(
        "--ddp.find-unused-params",
        action="store_true",
        help="Find unused params in model. useful for debugging with DDP",
    )

    return parser


def get_training_arguments(parse_args: Optional[bool] = True):
    parser = argparse.ArgumentParser(description="Training arguments", add_help=True)

    # sampler related arguments
    parser = arguments_sampler(parser=parser)

    # dataset related arguments
    parser = arguments_dataset(parser=parser)

    # anchor generator arguments
    parser = arguments_anchor_gen(parser=parser)

    # arguments related to box matcher
    parser = arguments_box_matcher(parser=parser)

    # Video reader related arguments
    parser = arguments_video_reader(parser=parser)

    # collate fn  related arguments
    parser = arguments_collate_fn(parser=parser)

    # transform related arguments
    parser = arguments_augmentation(parser=parser)

    # model related arguments
    parser = arguments_nn_layers(parser=parser)
    parser = arguments_model(parser=parser)
    parser = arguments_ema(parser=parser)

    # loss function arguments
    parser = arguments_loss_fn(parser=parser)

    # optimizer arguments
    parser = arguments_optimizer(parser=parser)
    parser = arguments_scheduler(parser=parser)

    # DDP arguments
    parser = arguments_ddp(parser=parser)

    # stats arguments
    parser = arguments_stats(parser=parser)

    # common
    parser = arguments_common(parser=parser)

    if parse_args:
        # parse args
        opts = parser.parse_args()
        opts = load_config_file(opts)
        return opts
    else:
        return parser


def get_eval_arguments(parse_args=True):
    return get_training_arguments(parse_args=parse_args)


def get_conversion_arguments():
    parser = get_training_arguments(parse_args=False)

    # Arguments related to coreml conversion
    group = parser.add_argument_group("Conversion arguments")
    group.add_argument(
        "--conversion.coreml-extn",
        type=str,
        default="mlmodel",
        help="Extension for converted model. Default is mlmodel",
    )
    group.add_argument(
        "--conversion.input-image-path",
        type=str,
        default=None,
        help="Path of the image to be used for conversion",
    )

    # Arguments related to server.
    group.add_argument(
        "--conversion.bucket-name", type=str, help="Model job's bucket name"
    )
    group.add_argument("--conversion.task-id", type=str, help="Model job's id")
    group.add_argument(
        "--conversion.viewers",
        type=str,
        nargs="+",
        default=None,
        help="Users who can view your models on server",
    )

    # parse args
    opts = parser.parse_args()
    opts = load_config_file(opts)

    return opts


def get_bencmarking_arguments():
    parser = get_training_arguments(parse_args=False)

    #
    group = parser.add_argument_group("Benchmarking arguments")
    group.add_argument(
        "--benchmark.batch-size",
        type=int,
        default=1,
        help="Batch size for benchmarking",
    )
    group.add_argument(
        "--benchmark.warmup-iter", type=int, default=10, help="Warm-up iterations"
    )
    group.add_argument(
        "--benchmark.n-iter",
        type=int,
        default=100,
        help="Number of iterations for benchmarking",
    )
    group.add_argument(
        "--benchmark.use-jit-model",
        action="store_true",
        help="Convert the model to JIT and then benchmark it",
    )

    # parse args
    opts = parser.parse_args()
    opts = load_config_file(opts)

    return opts


def get_segmentation_eval_arguments():
    parser = get_training_arguments(parse_args=False)

    group = parser.add_argument_group("Segmentation evaluation related arguments")
    group.add_argument(
        "--evaluation.segmentation.apply-color-map",
        action="store_true",
        help="Apply color map to different classes in segmentation masks. Useful in visualization "
        "+ some competitions (e.g, PASCAL VOC) accept submissions with colored segmentation masks",
    )
    group.add_argument(
        "--evaluation.segmentation.save-overlay-rgb-pred",
        action="store_true",
        help="enable this flag to visualize predicted masks on top of input image",
    )
    group.add_argument(
        "--evaluation.segmentation.save-masks",
        action="store_true",
        help="save predicted masks without colormaps. Useful for submitting to "
        "competitions like Cityscapes",
    )
    group.add_argument(
        "--evaluation.segmentation.overlay-mask-weight",
        default=0.5,
        type=float,
        help="Contribution of mask when overlaying on top of RGB image. ",
    )
    group.add_argument(
        "--evaluation.segmentation.mode",
        type=str,
        default="validation_set",
        required=False,
        choices=["single_image", "image_folder", "validation_set"],
        help="Contribution of mask when overlaying on top of RGB image. ",
    )
    group.add_argument(
        "--evaluation.segmentation.path",
        type=str,
        default=None,
        help="Path of the image or image folder (only required for single_image and image_folder modes)",
    )
    group.add_argument(
        "--evaluation.segmentation.num-classes",
        type=str,
        default=None,
        help="Number of segmentation classes used during training",
    )
    group.add_argument(
        "--evaluation.segmentation.resize-input-images",
        action="store_true",
        help="Resize input images",
    )

    # parse args
    opts = parser.parse_args()
    opts = load_config_file(opts)

    return opts


def get_detection_eval_arguments():
    parser = get_training_arguments(parse_args=False)

    group = parser.add_argument_group("Detection evaluation related arguments")
    group.add_argument(
        "--evaluation.detection.save-overlay-boxes",
        action="store_true",
        help="enable this flag to visualize predicted masks on top of input image",
    )
    group.add_argument(
        "--evaluation.detection.mode",
        type=str,
        default="validation_set",
        required=False,
        choices=["single_image", "image_folder", "validation_set"],
        help="Contribution of mask when overlaying on top of RGB image. ",
    )
    group.add_argument(
        "--evaluation.detection.path",
        type=str,
        default=None,
        help="Path of the image or image folder (only required for single_image and image_folder modes)",
    )
    group.add_argument(
        "--evaluation.detection.num-classes",
        type=str,
        default=None,
        help="Number of segmentation classes used during training",
    )
    group.add_argument(
        "--evaluation.detection.resize-input-images",
        action="store_true",
        default=False,
        help="Resize the input images",
    )

    # parse args
    opts = parser.parse_args()
    opts = load_config_file(opts)

    return opts


def get_loss_landscape_args():
    parser = get_training_arguments(parse_args=False)

    group = parser.add_argument_group("Loss landscape related arguments")
    group.add_argument(
        "--loss-landscape.n-points",
        type=int,
        default=11,
        help="No. of grid points. Default is 11, so we have 11x11 grid",
    )
    group.add_argument(
        "--loss-landscape.min-x",
        type=float,
        default=-1.0,
        help="Min. value along x-axis",
    )
    group.add_argument(
        "--loss-landscape.max-x",
        type=float,
        default=1.0,
        help="Max. value along x-axis",
    )
    group.add_argument(
        "--loss-landscape.min-y",
        type=float,
        default=-1.0,
        help="Min. value along y-axis",
    )
    group.add_argument(
        "--loss-landscape.max-y",
        type=float,
        default=1.0,
        help="Max. value along y-axis",
    )

    # parse args
    opts = parser.parse_args()
    opts = load_config_file(opts)
    return opts
