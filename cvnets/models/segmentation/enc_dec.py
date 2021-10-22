#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import Tensor
from utils import logger
from typing import Union, Dict, Tuple

from . import BaseSegmentation, register_segmentation_models
from ..classification import BaseEncoder
from .heads import build_segmentation_head


@register_segmentation_models(name="encoder_decoder")
class SegEncoderDecoder(BaseSegmentation):
    def __init__(self, opts, encoder: BaseEncoder) -> None:
        super(SegEncoderDecoder, self).__init__(opts=opts, encoder=encoder)

        # delete layers that are not required in segmentation network
        self.encoder.classifier = None
        use_l5_exp = getattr(opts, "model.segmentation.use_level5_exp", False)
        if not use_l5_exp:
            self.encoder.conv_1x1_exp = None

        self.seg_head = build_segmentation_head(opts=opts, enc_conf=self.encoder.model_conf_dict, use_l5_exp=use_l5_exp)
        self.use_l5_exp = use_l5_exp

    def get_trainable_parameters(self, weight_decay: float = 0.0, no_decay_bn_filter_bias: bool = False):
        encoder_params, enc_lr_mult = self.encoder.get_trainable_parameters(
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias
        )
        decoder_params, dec_lr_mult = self.seg_head.get_trainable_parameters(
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias
        )

        total_params = sum([p.numel() for p in self.parameters()])
        encoder_params_count = sum([p.numel() for p in self.encoder.parameters()])
        decoder_params_count = sum([p.numel() for p in self.seg_head.parameters()])

        assert total_params == encoder_params_count + decoder_params_count, "Total network parameters are not equal to " \
                                                                            "the sum of encoder and decoder. " \
                                                                            "{} != {} + {}".format(total_params,
                                                                                                   encoder_params_count,
                                                                                                   decoder_params_count
                                                                                                   )

        return encoder_params + decoder_params, enc_lr_mult + dec_lr_mult

    def forward(self, x: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:
        enc_end_points: Dict = self.encoder.extract_end_points_all(x, use_l5=True, use_l5_exp=self.use_l5_exp)
        return self.seg_head(enc_out=enc_end_points)

    def profile_model(self, input: Tensor):
        # Note: Model profiling is for reference only and may contain errors.
        # It relies heavily on the user to implement the underlying functions accurately.

        overall_params, overall_macs = 0.0, 0.0

        logger.log('Model statistics for an input of size {}'.format(input.size()))
        logger.double_dash_line(dashes=65)
        print('{:>35} Summary'.format(self.__class__.__name__))
        logger.double_dash_line(dashes=65)

        # profile encoder
        enc_str = logger.text_colors['logs'] + logger.text_colors['bold'] + 'Encoder  ' + logger.text_colors[
            'end_color']
        print('{:>45}'.format(enc_str))
        enc_end_points, encoder_params, encoder_macs = self.encoder.profile_model(input, is_classification=False)
        overall_params += encoder_params
        overall_macs += encoder_macs

        # profile decoder
        dec_str = logger.text_colors['logs'] + logger.text_colors['bold'] + 'Decoder  ' + logger.text_colors[
            'end_color']
        print('{:>45}'.format(dec_str))

        out, decoder_params, decoder_macs = self.seg_head.profile_module(enc_end_points)
        overall_params += decoder_params
        overall_macs += decoder_macs

        logger.double_dash_line(dashes=65)
        print('{:<20} = {:>8.3f} M'.format('Overall parameters', overall_params / 1e6))
        # Counting Addition and Multiplication as 1 operation
        print('{:<20} = {:>8.3f} M'.format('Overall MACs', overall_macs / 1e6))
        overall_params_py = sum([p.numel() for p in self.parameters()])
        print('{:<20} = {:>8.3f} M'.format('Overall parameters (sanity check)', overall_params_py / 1e6))
        logger.double_dash_line(dashes=65)
