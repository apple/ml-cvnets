#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint as gradient_checkpoint_fn
from typing import Optional, Dict, Tuple, Union, Any
import argparse

from utils import logger

from ... import parameter_list
from ...layers import norm_layers_tuple, LinearLayer
from ...misc.profiler import module_profile
from ...misc.init_utils import initialize_weights, initialize_fc_layer

from ...neural_augmentor import build_neural_augmentor


class BaseEncoder(nn.Module):
    """
    Base class for different classification models
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__()
        self.conv_1 = None
        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None
        self.layer_4 = None
        self.layer_5 = None
        self.conv_1x1_exp = None
        self.classifier = None
        self.round_nearest = 8

        # Segmentation architectures like Deeplab and PSPNet modifies the strides of the backbone
        # We allow that using output_stride and replace_stride_with_dilation arguments
        self.dilation = 1
        output_stride = kwargs.get("output_stride", None)
        self.dilate_l4 = False
        self.dilate_l5 = False
        if output_stride == 8:
            self.dilate_l4 = True
            self.dilate_l5 = True
        elif output_stride == 16:
            self.dilate_l5 = True

        self.model_conf_dict = dict()
        self.neural_augmentor = build_neural_augmentor(opts=opts, *args, **kwargs)
        self.gradient_checkpointing = getattr(
            opts, "model.classification.gradient_checkpointing", False
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model-specific arguments"""
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--model.classification.classifier-dropout",
            type=float,
            default=0.0,
            help="Dropout rate in classifier",
        )

        group.add_argument(
            "--model.classification.name", type=str, default=None, help="Model name"
        )
        group.add_argument(
            "--model.classification.n-classes",
            type=int,
            default=1000,
            help="Number of classes in the dataset",
        )
        group.add_argument(
            "--model.classification.pretrained",
            type=str,
            default=None,
            help="Path of the pretrained backbone",
        )
        group.add_argument(
            "--model.classification.freeze-batch-norm",
            action="store_true",
            help="Freeze batch norm layers",
        )
        group.add_argument(
            "--model.classification.activation.name",
            default=None,
            type=str,
            help="Non-linear function name (e.g., relu)",
        )
        group.add_argument(
            "--model.classification.activation.inplace",
            action="store_true",
            help="Inplace non-linear functions",
        )
        group.add_argument(
            "--model.classification.activation.neg-slope",
            default=0.1,
            type=float,
            help="Negative slope in leaky relu",
        )

        group.add_argument(
            "--model.classification.finetune-pretrained-model",
            action="store_true",
            help="Finetune a pretrained model",
        )
        group.add_argument(
            "--model.classification.n-pretrained-classes",
            type=int,
            default=None,
            help="Number of pre-trained classes",
        )

        group.add_argument(
            "--model.classification.gradient-checkpointing",
            action="store_true",
            help="Checkpoint output of each spatial level in the classification backbone. Note that"
            "we only take care of checkpointing in {}. If custom forward functions are used, please"
            "implement checkpointing accordingly",
        )

        return parser

    def check_model(self):
        assert (
            self.model_conf_dict
        ), "Model configuration dictionary should not be empty"
        assert self.conv_1 is not None, "Please implement self.conv_1"
        assert self.layer_1 is not None, "Please implement self.layer_1"
        assert self.layer_2 is not None, "Please implement self.layer_2"
        assert self.layer_3 is not None, "Please implement self.layer_3"
        assert self.layer_4 is not None, "Please implement self.layer_4"
        assert self.layer_5 is not None, "Please implement self.layer_5"
        assert self.conv_1x1_exp is not None, "Please implement self.conv_1x1_exp"
        assert self.classifier is not None, "Please implement self.classifier"

    def reset_parameters(self, opts):
        """Initialize model weights"""
        initialize_weights(opts=opts, modules=self.modules())

    def update_classifier(self, opts, n_classes: int) -> None:
        """
        This function updates the classification layer in a model. Useful for finetuning purposes.
        """
        linear_init_type = getattr(opts, "model.layer.linear_init", "normal")
        if isinstance(self.classifier, nn.Sequential):
            in_features = self.classifier[-1].in_features
            layer = LinearLayer(
                in_features=in_features, out_features=n_classes, bias=True
            )
            initialize_fc_layer(layer, init_method=linear_init_type)
            self.classifier[-1] = layer
        else:
            in_features = self.classifier.in_features
            layer = LinearLayer(
                in_features=in_features, out_features=n_classes, bias=True
            )
            initialize_fc_layer(layer, init_method=linear_init_type)

            # re-init head
            head_init_scale = 0.001
            layer.weight.data.mul_(head_init_scale)
            layer.bias.data.mul_(head_init_scale)

            self.classifier = layer

    def _forward_layer(self, layer: nn.Module, x: Tensor) -> Tensor:
        # Larger models with large input image size may not be able to fit into memory.
        # We can use gradient checkpointing to enable training with large models and large inputs
        return (
            gradient_checkpoint_fn(layer, x)
            if self.gradient_checkpointing
            else layer(x)
        )

    def extract_end_points_all(
        self,
        x: Tensor,
        use_l5: Optional[bool] = True,
        use_l5_exp: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Dict[str, Tensor]:
        out_dict = {}  # Use dictionary over NamedTuple so that JIT is happy

        if self.training and self.neural_augmentor is not None:
            x = self.neural_augmentor(x)
            out_dict["augmented_tensor"] = x

        x = self._forward_layer(self.conv_1, x)  # 112 x112
        x = self._forward_layer(self.layer_1, x)  # 112 x112
        out_dict["out_l1"] = x

        x = self._forward_layer(self.layer_2, x)  # 56 x 56
        out_dict["out_l2"] = x

        x = self._forward_layer(self.layer_3, x)  # 28 x 28
        out_dict["out_l3"] = x

        x = self._forward_layer(self.layer_4, x)  # 14 x 14
        out_dict["out_l4"] = x

        if use_l5:
            x = self._forward_layer(self.layer_5, x)  # 7 x 7
            out_dict["out_l5"] = x

            if use_l5_exp:
                x = self._forward_layer(self.conv_1x1_exp, x)
                out_dict["out_l5_exp"] = x
        return out_dict

    def extract_end_points_l4(self, x: Tensor, *args, **kwargs) -> Dict[str, Tensor]:
        return self.extract_end_points_all(x, use_l5=False)

    def _extract_features(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self._forward_layer(self.conv_1, x)
        x = self._forward_layer(self.layer_1, x)
        x = self._forward_layer(self.layer_2, x)
        x = self._forward_layer(self.layer_3, x)

        x = self._forward_layer(self.layer_4, x)
        x = self._forward_layer(self.layer_5, x)
        x = self._forward_layer(self.conv_1x1_exp, x)
        return x

    def _forward_classifier(self, x: Tensor, *args, **kwargs) -> Tensor:
        # We add another classifier function so that the classifiers
        # that do not adhere to the structure of BaseEncoder can still
        # use neural augmentor
        x = self._extract_features(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Any, *args, **kwargs) -> Any:
        if self.neural_augmentor is not None:
            if self.training:
                x_aug = self.neural_augmentor(x)
                prediction = self._forward_classifier(x_aug)  # .detach()
                out_dict = {"augmented_tensor": x_aug, "logits": prediction}
            else:
                out_dict = {
                    "augmented_tensor": None,
                    "logits": self._forward_classifier(x),
                }
            return out_dict
        else:
            x = self._forward_classifier(x, *args, **kwargs)
            return x

    def freeze_norm_layers(self) -> None:
        """Freeze normalization layers"""
        for m in self.modules():
            if isinstance(m, norm_layers_tuple):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.training = False

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs
    ):
        """Get trainable parameters"""
        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            *args,
            **kwargs
        )
        return param_list, [1.0] * len(param_list)

    @staticmethod
    def _profile_layers(
        layers, input, overall_params, overall_macs, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        if not isinstance(layers, list):
            layers = [layers]

        for layer in layers:
            if layer is None:
                continue
            input, layer_param, layer_macs = module_profile(module=layer, x=input)

            overall_params += layer_param
            overall_macs += layer_macs

            if isinstance(layer, nn.Sequential):
                module_name = "\n+".join([l.__class__.__name__ for l in layer])
            else:
                module_name = layer.__class__.__name__
            print(
                "{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
                    module_name,
                    "Params",
                    round(layer_param / 1e6, 3),
                    "MACs",
                    round(layer_macs / 1e6, 3),
                )
            )
            logger.singe_dash_line()
        return input, overall_params, overall_macs

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        """Create dummy input and labels for CI/CD purposes. Child classes must override it
        if functionality is different.
        """
        img_channels = 3
        height = 224
        width = 224
        n_labels = 10
        img_tensor = torch.randn(
            batch_size, img_channels, height, width, dtype=torch.float
        )
        label_tensor = torch.randint(low=0, high=n_labels, size=(batch_size,)).long()
        return {"samples": img_tensor, "targets": label_tensor}

    def profile_model(
        self, input: Tensor, is_classification: Optional[bool] = True, *args, **kwargs
    ) -> Tuple[Union[Tensor, Dict[str, Tensor]], float, float]:
        """
        Helper function to profile a model.

        .. note::
            Model profiling is for reference only and may contain errors as it solely relies on user implementation to
            compute theoretical FLOPs
        """
        overall_params, overall_macs = 0.0, 0.0

        input_fvcore = input.clone()

        if is_classification:
            logger.log("Model statistics for an input of size {}".format(input.size()))
            logger.double_dash_line(dashes=65)
            print("{:>35} Summary".format(self.__class__.__name__))
            logger.double_dash_line(dashes=65)

        out_dict = {}
        input, overall_params, overall_macs = self._profile_layers(
            [self.conv_1, self.layer_1],
            input=input,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )
        out_dict["out_l1"] = input

        input, overall_params, overall_macs = self._profile_layers(
            self.layer_2,
            input=input,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )
        out_dict["out_l2"] = input

        input, overall_params, overall_macs = self._profile_layers(
            self.layer_3,
            input=input,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )
        out_dict["out_l3"] = input

        input, overall_params, overall_macs = self._profile_layers(
            self.layer_4,
            input=input,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )
        out_dict["out_l4"] = input

        input, overall_params, overall_macs = self._profile_layers(
            self.layer_5,
            input=input,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )
        out_dict["out_l5"] = input

        if self.conv_1x1_exp is not None:
            input, overall_params, overall_macs = self._profile_layers(
                self.conv_1x1_exp,
                input=input,
                overall_params=overall_params,
                overall_macs=overall_macs,
            )
            out_dict["out_l5_exp"] = input

        if is_classification:
            classifier_params, classifier_macs = 0.0, 0.0
            if self.classifier is not None:
                input, classifier_params, classifier_macs = module_profile(
                    module=self.classifier, x=input
                )
                print(
                    "{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
                        "Classifier",
                        "Params",
                        round(classifier_params / 1e6, 3),
                        "MACs",
                        round(classifier_macs / 1e6, 3),
                    )
                )
            overall_params += classifier_params
            overall_macs += classifier_macs

            logger.double_dash_line(dashes=65)
            print(
                "{:<20} = {:>8.3f} M".format("Overall parameters", overall_params / 1e6)
            )
            overall_params_py = sum([p.numel() for p in self.parameters()])
            print(
                "{:<20} = {:>8.3f} M".format(
                    "Overall parameters (sanity check)", overall_params_py / 1e6
                )
            )

            # Counting Addition and Multiplication as 1 operation
            print(
                "{:<20} = {:>8.3f} M".format(
                    "Overall MACs (theoretical)", overall_macs / 1e6
                )
            )

            # compute flops using FVCore
            try:
                # compute flops using FVCore also
                from fvcore.nn import FlopCountAnalysis

                flop_analyzer = FlopCountAnalysis(self.eval(), input_fvcore)
                flop_analyzer.unsupported_ops_warnings(False)
                flop_analyzer.uncalled_modules_warnings(False)
                flops_fvcore = flop_analyzer.total()

                print(
                    "{:<20} = {:>8.3f} M".format(
                        "Overall MACs (FVCore)**", flops_fvcore / 1e6
                    )
                )
                print(
                    "\n** Theoretical and FVCore MACs may vary as theoretical MACs do not account "
                    "for certain operations which may or may not be accounted in FVCore"
                )
            except Exception:
                pass

            print("Note: Theoretical MACs depends on user-implementation. Be cautious")
            logger.double_dash_line(dashes=65)

        return out_dict, overall_params, overall_macs
