#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import operator
import os
from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch

from data.transforms.video import (
    CropByBoundingBox,
    SaveInputs,
    ShuffleAudios,
    _resize_fn,
)
from data.video_reader.pyav_reader import PyAVReader
from tests.configs import get_config


def test_resize_fn() -> None:
    bs = 2
    sz = (10, 8)
    c = 3
    t = 4
    new_sz = (6, 4)
    data = {
        "samples": {
            "video": torch.randn(bs, t, c, *sz),
            "mask": torch.rand(bs, t, *sz),
        }
    }
    new_data = _resize_fn(data, size=new_sz)
    video = data["samples"]["video"]
    mask = data["samples"]["mask"]
    assert video.shape == (bs, t, c, *new_sz)
    assert mask.shape == (bs, t, *new_sz)


def test_crop_by_bounding_box() -> None:
    opts = get_config()
    image_size = (10, 10)
    setattr(opts, "video_augmentation.crop_by_bounding_box.image_size", image_size)
    setattr(opts, "video_augmentation.crop_by_bounding_box.multiplier", 1.14)
    transform = CropByBoundingBox(opts)

    N, T, C, H, W = 2, 3, 5, 81, 81
    video = torch.rand(N, T, C, H, W)
    box_coordinates = torch.concat(
        [
            torch.rand(N, T, 2) * 0.5,
            torch.rand(N, T, 2) * 0.5 + 0.5,
        ],
        dim=2,
    )

    x, y = torch.meshgrid(torch.linspace(10, 90, H), torch.linspace(1, 9, W))
    video[0, 0, 0, :, :] = (
        x + y
    )  # x values are in {10, 20, ..., 90} and y values are in {1, 2, ..., 9}
    # The bounding box is a small strip at the bottom of the image
    box_coordinates[0, 0, :] = torch.tensor([0, 0.8, 1, 0.9])

    data = {
        "samples": {
            "video": video,
        },
        "targets": {
            "traces": {
                "a_uuid": {
                    "box_coordinates": box_coordinates,
                }
            }
        },
    }
    result = transform(data)
    assert isinstance(result, dict), f"{type(result)} != dict"
    result_video = result["samples"]["video"]
    assert result_video.shape == (N, T, C, *image_size)
    result_box_coordinates = result["targets"]["traces"]["a_uuid"]["box_coordinates"]
    assert result_box_coordinates.shape == (N, T, 4)
    assert torch.all(0 < result_box_coordinates[:, :, :2])
    assert torch.all(result_box_coordinates[:, :, :2] < 0.07)  # 0.07 = (1.14 - 1) / 2
    assert torch.all(1 - 0.07 < result_box_coordinates[:, :, 2:])
    assert torch.all(result_box_coordinates[:, :, 2:] < 1)

    # Since the bounding box was aligned with the right and left edges of the image, we should observe 0 values in the
    # right and left edges of the cropped image, as a result of expansion
    assert torch.all(result_video[0, 0, 0, :, 0] == 0)
    assert torch.all(result_video[0, 0, 0, :, -1] == 0)
    # but not on the top and bottom edges
    assert torch.all(result_video[0, 0, 0, 0, 1:-1] > 0)
    assert torch.all(result_video[0, 0, 0, -1, 1:-1] > 0)

    # check the values of two cropped pixels created earlier by `video[0, 0, 0, :, :] = x + y`
    assert 70 <= result_video[0, 0, 0, 1, 1].item() <= 80
    assert 90 <= result_video[0, 0, 0, -2, -2].item() <= 100


@pytest.mark.parametrize("numel", [5, 6])
@pytest.mark.parametrize("is_training", [True, False])
def test_shuffle_audios_single_cycle_permutation(numel: int, is_training: bool) -> None:
    device = torch.device("cpu")
    prev_perm = ShuffleAudios._single_cycle_permutation(
        numel, device=device, is_training=is_training
    )
    identity = torch.arange(numel, device=device)
    is_random = False
    for _ in range(20):
        perm = ShuffleAudios._single_cycle_permutation(
            numel, device=device, is_training=is_training
        )
        if torch.any(perm != prev_perm):
            is_random = True
            prev_perm = perm

        assert torch.all(
            perm != identity
        ), f"Single cycle permutation should not have identity mapping: {perm}."

        sorted_perm, _ = perm.sort()
        assert torch.all(
            sorted_perm == identity
        ), f"Result is not a permutation: {perm}."

    assert is_random == is_training, "Outcomes should be random iff is_training."


@pytest.mark.parametrize(
    "N, mode, shuffle_ratio, generate_frame_level_targets, debug_mode",
    [
        (1000, "train", 0.2, True, False),
        (1000, "val", 0.3, False, False),
        (1000, "test", 0.7, True, True),
        (1, "train", 0.2, False, False),
        (1, "val", 0.2, False, False),
    ],
)
def test_shuffle_audios(
    N: int,
    mode: str,
    shuffle_ratio: float,
    generate_frame_level_targets: bool,
    debug_mode: bool,
) -> None:
    opts = get_config()
    setattr(
        opts,
        f"video_augmentation.shuffle_audios.shuffle_ratio_{mode}",
        shuffle_ratio,
    )
    setattr(
        opts,
        "video_augmentation.shuffle_audios.generate_frame_level_targets",
        generate_frame_level_targets,
    )
    setattr(
        opts,
        "video_augmentation.shuffle_audios.debug_mode",
        debug_mode,
    )

    C_v, C_a = 3, 2
    H, W = 8, 8
    num_video_frames = 3
    num_audio_frames = 5

    video = torch.rand(N, num_video_frames, C_v, H, W)
    # Generating unique audio elements (using torch.arange) so that we can compare them to check if they are shuffled.
    input_audio = torch.empty(N, num_audio_frames, C_a, dtype=torch.float)
    torch.arange(input_audio.numel(), dtype=torch.float, out=input_audio)

    data = {
        "samples": {
            "video": video,
            "audio": input_audio.clone(),
            "metadata": {},
        },
        "targets": {},
    }

    result = ShuffleAudios(
        opts=opts,
        is_training=mode == "train",
        is_evaluation=mode == "test",
        item_index=0,
    )(data)

    labels = data["targets"]["is_shuffled"]
    assert (
        labels.shape == (N, num_video_frames) if generate_frame_level_targets else (N,)
    )
    if generate_frame_level_targets:
        assert torch.all(
            labels == labels[:, :1].repeat(1, num_video_frames)
        ), "Labels should be identical among frames of the same clip."

    result_is_shuffled = (
        (~torch.isclose(input_audio, result["samples"]["audio"]))
        .float()  # shape: N x num_audio_frames x C_a
        .mean(axis=1)  # shape: N x C_a
        .mean(axis=1)  # shape: N
    )
    actual_participation_ratio = result_is_shuffled.float().mean()
    if N > 1:
        assert actual_participation_ratio == pytest.approx(shuffle_ratio, abs=0.05)
    else:
        assert actual_participation_ratio == pytest.approx(0.0, abs=0.05)

    assert torch.allclose(
        result_is_shuffled,
        (labels[:, 0] if generate_frame_level_targets else labels).float(),
    ), "Generated labels should match shuffled audios."

    if debug_mode:
        assert data["samples"]["metadata"]["shuffled_audio_permutation"].shape == (N,)
    else:
        assert "shuffled_audio_permutation" not in data["samples"]["metadata"]


@pytest.mark.parametrize(
    "t,expected",
    [
        (0, "00:00:0,000"),
        (0.5, "00:00:0,500"),
        (10.5, "00:00:10,500"),
        (70.5, "00:01:10,500"),
        (7270.5, "02:01:10,500"),
    ],
)
def test_save_inputs_srt_format_timestamp(t: float, expected: str) -> None:
    assert SaveInputs._srt_format_timestamp(t) == expected


@pytest.mark.parametrize(
    "params",
    [
        {
            "opts": {},
            "init_kwargs": {},
            "expected_output_count": 1,
            "video_only": False,
        },
        {
            "opts": {},
            "init_kwargs": {
                "get_frame_captions": (
                    lambda data: ["_"]
                    * operator.mul(*data["samples"]["video"].shape[:2])
                )
            },
            "expected_output_count": 1,
            "video_only": False,
        },
        {
            "opts": {"video_augmentation.save_inputs.symlink_to_original": True},
            "init_kwargs": {},
            "expected_output_count": 2,
            "video_only": True,
        },
    ],
)
def test_save_inputs(
    params: Tuple[Dict, Dict],
    tmp_path: Path,
) -> None:
    opts = get_config()
    setattr(opts, "video_augmentation.save_inputs.enable", True)
    setattr(opts, "video_augmentation.save_inputs.save_dir", str(tmp_path))
    for key, value in params["opts"].items():
        setattr(opts, key, value)

    data = {
        "samples": PyAVReader(opts).dummy_audio_video_clips(
            clips_per_video=2,
            num_frames_to_sample=3,
            height=24,
            width=24,
        )
    }
    if params["video_only"]:
        del data["samples"]["audio"]
    SaveInputs(opts, **params["init_kwargs"])(data)
    output_video_paths = list(tmp_path.glob("*/*.mp4"))
    assert (
        len(output_video_paths) == params["expected_output_count"]
    ), f"Expected {params['expected_output_count']} videos, but got: {output_video_paths}."
    for output_video_path in output_video_paths:
        if output_video_path.is_symlink():
            continue
        assert (
            os.stat(output_video_path).st_size > 1000
        ), f"The generated file ({output_video_path}) is too small ({os.stat(output_video_path).st_size})."
