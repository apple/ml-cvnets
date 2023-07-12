#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import torch

from cvnets.models.audio_classification import audio_byteformer
from cvnets.models.classification import byteformer as image_byteformer
from tests.models.classification import test_byteformer


def test_audio_byteformer() -> None:
    # Make sure it matches the image classification network.
    opts = test_byteformer.get_opts()

    byteformer1 = image_byteformer.ByteFormer(opts)
    byteformer2 = audio_byteformer.AudioByteFormer(opts)

    # Make their state_dicts match.
    byteformer2.load_state_dict(byteformer1.state_dict())

    batch_size, sequence_length = 2, 32

    x = torch.randint(0, 128, [batch_size, sequence_length])

    assert torch.all(byteformer1(x) == byteformer2({"audio": x}))
