#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
import math
import pathlib
import random
from typing import Callable, Dict, List, Optional, Union

import torch
import torchaudio

from data.transforms import TRANSFORMATIONS_REGISTRY, BaseTransformation
from data.transforms.audio_aux import mfccs


@TRANSFORMATIONS_REGISTRY.register(name="audio_gain", type="audio")
class Gain(BaseTransformation):
    """
    This class implements gain augmentation for audio.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts=opts)
        self.gain_levels = getattr(opts, "audio_augmentation.gain.levels")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--audio-augmentation.gain.levels",
            type=float,
            default=[0],
            nargs="+",
            help="Gain levels to use for augmentation (in dB).",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        """
        This function implements the gain transformation by scaling the
        input audio with a specific constant determined from the gain_levels

        Args:
            data: A dictionary containing {"samples": {"audio": @audio}}, where
                @audio is a tensor of shape [num_channels, sequence_length].

        Returns:
            The modified dictionary with the augmented audio.
        """
        audio = data["samples"]["audio"]
        gain_level = random.choice(self.gain_levels)
        augmented_audio = 10.0 ** (gain_level / 20.0) * audio
        data["samples"]["audio"] = augmented_audio
        return data

    def __repr__(self):
        return "{}(gain_levels={})".format(self.__class__.__name__, self.gain_levels)


@TRANSFORMATIONS_REGISTRY.register(name="audio_ambient_noise", type="audio")
class Noise(BaseTransformation):
    """
    This class implements ambient noise augmentation for audio.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        is_training: bool = True,
        noise_files_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(opts=opts)
        self.gain_levels = getattr(opts, "audio_augmentation.noise.levels")
        self.cache_size = getattr(opts, "audio_augmentation.noise.cache_size")
        self.refresh_freq = getattr(opts, "audio_augmentation.noise.refresh_freq")
        self.refresh_counter = self.refresh_freq

        self.noise_files_dir = noise_files_dir
        if self.noise_files_dir is None:
            self.noise_files_dir = getattr(opts, "audio_augmentation.noise.files_dir")

        self.noise_files = []
        self.pointer = 0
        if self.noise_files_dir is not None:
            self.noise_files = sorted(
                pathlib.Path(self.noise_files_dir).glob("**/*.wav")
            )
            if is_training:
                random.shuffle(self.noise_files)
            self.noise_waves = self.load_noise_files(cache_size=self.cache_size)
        if len(self.noise_files) == 0:
            raise ValueError(
                "--audio-augmentation.noise.files-dir must be provided for this augmentation"
            )

    def load_noise_files(self, cache_size: int) -> List[torch.TensorType]:
        """
        This method caches a list of noise files for on the fly augmentation.
        """
        noise_waves = []
        for i in range(cache_size):
            noise_wav_file = self.noise_files[self.pointer % len(self.noise_files)]
            self.pointer += 1
            noise, sample_rate = torchaudio.load(noise_wav_file)
            assert (
                noise.dtype == torch.float32
            ), f"Expected noise file {noise_wav_file} to decode to float32 audio, but got {noise.dtype}."
            noise_waves.append((noise, sample_rate))
        return noise_waves

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--audio-augmentation.noise.enable",
            action="store_true",
            help="Use {}. This flag is useful when you want to study the effect of"
            " different transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--audio-augmentation.noise.levels",
            type=float,
            default=[-100],
            nargs="+",
            help="Gain levels to use for noise augmentation (in dB).",
        )
        group.add_argument(
            "--audio-augmentation.noise.cache-size",
            type=int,
            default=10,
            help="Number of augmentation noises to cache.",
        )
        group.add_argument(
            "--audio-augmentation.noise.files-dir",
            type=str,
            default=None,
            help="Directory of noise files.",
        )
        group.add_argument(
            "--audio-augmentation.noise.refresh-freq",
            type=int,
            default=0,
            help="Frequency to refresh noise files (default 0 means never refresh).",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        """
        This function adds a random noise sample selected from the
        noise samples provided in the noise directory scaled by a random gain.
        The sample should contain floating-point values in [-1, 1].

        Args:
            data: A dictionary containing {"samples": {"audio": @audio}}, where
                @audio is a tensor of shape [num_channels, sequence_length].

        Returns:
            The modified dictionary with the augmented audio.
        """
        audio = data["samples"]["audio"]
        assert audio.shape[0] in [1, 2]
        gain_level = random.choice(self.gain_levels)
        noise_wave, noise_fps = random.choice(self.noise_waves)
        # @noise_wave is in [num_channels, sequence_length] format.
        assert math.isclose(data["metadata"]["audio_fps"], noise_fps, rel_tol=1e-6)
        if noise_wave.shape[-1] >= audio.shape[-1]:
            random_start_point = random.randint(
                0, noise_wave.shape[-1] - audio.shape[-1]
            )
            noise_wave = noise_wave[
                :, random_start_point : random_start_point + audio.shape[-1]
            ]
        else:
            noise_wave = torch.nn.functional.pad(
                noise_wave.unsqueeze(0),
                (0, audio.shape[-1] - noise_wave.shape[-1]),
                mode="circular",
            )
            # @noise_wave is in [1, num_channels, sequence_length] format.
            noise_wave = noise_wave[0]
        augmented_audio = audio + 10.0 ** (gain_level / 20.0) * noise_wave
        data["samples"]["audio"] = augmented_audio
        self.refresh_counter -= 1
        if (
            self.refresh_counter <= 0
            and self.refresh_freq > 0
            and self.noise_files_dir is not None
        ):
            # Refresh cache when met refresh criteria.
            self.noise_waves = self.load_noise_files(self.cache_size)
            self.refresh_counter = self.refresh_freq
        return data

    def __repr__(self):
        return "{}(gain_levels={}, noise_files_dir={})".format(
            self.__class__.__name__, self.gain_levels, self.noise_files_dir
        )


@TRANSFORMATIONS_REGISTRY.register(name="set_fixed_length", type="audio")
class SetFixedLength(BaseTransformation):
    """Set the audio buffer to a fixed length."""

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        self.length = getattr(opts, "audio_augmentation.set_fixed_length.length")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--audio-augmentation.set-fixed-length.enable",
            action="store_true",
            help="Use {}. This flag is useful when you want to study the effect of"
            " different transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--audio-augmentation.set-fixed-length.length",
            default=16000,
            type=int,
            help="Length to which to trim or pad the audio buffer.",
        )

        return parser

    def __call__(
        self,
        data: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor, int]],
        *args,
        **kwargs,
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor, int]]:
        """
        Apply the transformation to the input data.

        Input data must have {"samples": {"audio": torch.Tensor}}. The audio
        must be [C, N] in shape, where C is the number of channels, and N is the
        number of samples.

        Returns:
            The transformed batch.
        """
        audio = data["samples"]["audio"]
        if not audio.shape[0] in (1, 2):
            raise ValueError(f"Expected channels first. Got audio shape {audio.shape}")

        if audio.shape[1] < self.length:
            audio = torch.nn.functional.pad(audio, (0, self.length - audio.shape[1]))
        else:
            audio = audio[:, 0 : self.length]
        data["samples"]["audio"] = audio
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={self.length})"


@TRANSFORMATIONS_REGISTRY.register(name="roll", type="audio")
class Roll(BaseTransformation):
    """Perform a roll augmentation by shifting the window in a circular manner."""

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        self.window = getattr(opts, "audio_augmentation.roll.window")
        super().__init__(opts, *args, **kwargs)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--audio-augmentation.roll.enable",
            action="store_true",
            help="Use {}. This flag is useful when you want to study the effect of"
            " different transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--audio-augmentation.roll.window",
            default=0.1,
            type=float,
            help="Maximum fraction of the audio buffer to move.",
        )

        return parser

    def __call__(
        self,
        data: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor, int]],
        *args,
        **kwargs,
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor, int]]:
        """
        Apply the transformation to the input data.

        Input data must have {"samples": {"audio": torch.Tensor}}. The audio
        must be [C, N] in shape, where C is the number of channels, and N is the
        number of samples.

        Returns:
            The transformed batch.
        """
        audio = data["samples"]["audio"]
        C, N = audio.shape
        if not C in (1, 2):
            raise ValueError(f"Unexpected number of channels {C}")

        audio = torch.roll(
            audio,
            torch.randint(-int(N * self.window), int(N * self.window), [1]).item(),
            1,
        )
        data["samples"]["audio"] = audio
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(window={self.window})"


@TRANSFORMATIONS_REGISTRY.register(name="mfccs", type="audio")
class MFCCs(BaseTransformation):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        self.num_mfccs = getattr(opts, "audio_augmentation.mfccs.num_mfccs")
        self.window_length = getattr(opts, "audio_augmentation.mfccs.window_length")
        self.num_frames = getattr(opts, "audio_augmentation.mfccs.num_frames")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--audio-augmentation.mfccs.num-mfccs",
            default=20,
            type=int,
            help="Number of MFCC features.",
        )
        group.add_argument(
            "--audio-augmentation.mfccs.window-length",
            type=float,
            default=0.023,
            help="Window length (unit: seconds) for MFCC calculation.",
        )
        group.add_argument(
            "--audio-augmentation.mfccs.num-frames",
            type=int,
            default=8,
            help="Number of sub-time-slice temporal components. This argument is used"
            " for splitting the temporal dimension of the spectrogram into frames.",
        )

        return parser

    def __call__(self, data: Dict, *args, **kwargs) -> Dict:
        """
        Converts the audio signal of the samples to MFCC features. See the documentation
        of @cvnets.misc.get_mfcc_features for further details.

        Args: {
            "samples": {
                "audio": torch.FloatTensor[num_clips x temporal_size x num_channels]
                "metadata": {
                     "audio_fps": float
                }
            }
        },

        Returns: {
            "samples": {
                "audio": torch.FloatTensor[num_clips, C, num_mfccs, num_frames,
                ceil(spectrogram_length/num_frames)]
            }
        }
        """
        audio_fps = data["samples"]["metadata"]["audio_fps"]
        audio_image = mfccs.get_mfcc_features(
            data["samples"]["audio"],
            sampling_rate=audio_fps,
            num_mfccs=self.num_mfccs,
            window_length=self.window_length,
            num_frames=self.num_frames,
        ).detach()

        data["samples"]["audio"] = audio_image
        return data


class LambdaAudio(BaseTransformation):
    """
    Similar to @torchvision.transforms.Lambda, applies a user-defined lambda on the
    audio samples as a transform.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        func: Callable[[torch.Tensor], torch.Tensor],
        *args,
        **kwargs,
    ) -> None:
        self.func = func
        super(LambdaAudio, self).__init__(opts, *args, **kwargs)

    def __call__(self, data: Dict, *args, **kwargs) -> Dict:
        data["samples"]["audio"] = self.func(data["samples"]["audio"])
        return data


@TRANSFORMATIONS_REGISTRY.register(name="audio-resample", type="audio")
class AudioResample(BaseTransformation):
    """Resample audio to a specified framerate."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--audio-augmentation.audio-resample.audio-fps",
            default=16000,
            type=int,
            help="Frames per second in the incoming audio stream. Default to 16000.",
        )
        return parser

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        self.sample_rate = getattr(opts, "audio_augmentation.audio_resample.audio_fps")
        self.effects = [["rate", str(self.sample_rate)]]

    def __call__(self, data: Dict, *args, **kwargs) -> Dict:
        """Reample audio to the specified audio fps.

        Args:
            data: A dict of data input in the following format:
            {
                "samples": {
                    "audio": torch.FloatTensor[num_clips x temporal_size x num_channels]
                    "metadata": {
                        "audio_fps": float
                    }
                }
            },

        Returns: {
            "samples": {
                "audio": torch.FloatTensor[num_clips x temporal_size x num_channels]
                "metadata": {
                    "audio_fps": float
                }
            }
        }
        """
        audio = data["samples"]["audio"]

        audio_rate = data["samples"]["metadata"]["audio_fps"]
        resampled_audio = []
        for audio_tensor in audio:
            (
                resampled_audio_tensor,
                sample_rate,
            ) = torchaudio.sox_effects.apply_effects_tensor(
                audio_tensor, audio_rate, self.effects, channels_first=False
            )
            resampled_audio.append(resampled_audio_tensor)
        data["samples"]["audio"] = torch.stack(resampled_audio, dim=0)
        data["samples"]["metadata"]["audio_fps"] = self.sample_rate
        return data


@TRANSFORMATIONS_REGISTRY.register(name="standardize_channels", type="audio")
class StandardizeChannels(BaseTransformation):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        self.num_channels = getattr(
            opts, "audio_augmentation.standardize_channels.num_channels"
        )
        self.enable = getattr(opts, "audio_augmentation.standardize_channels.enable")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--audio-augmentation.standardize-channels.num-channels",
            default=2,
            type=int,
            help="Number of output audio channels. Defaults to 2.",
        )
        group.add_argument(
            "--audio-augmentation.standardize-channels.enable",
            default=False,
            action="store_true",
            help=f"Use {cls.__name__} transformation. Defaults to False.",
        )
        return parser

    def __call__(self, data: Dict, *args, **kwargs) -> Dict:
        """Ensures all audio samples have a specific number of channels.

        To reduce the number of audio channels from 2 to 1, the average values of the
        two channels is used.

        Args:
            data (Dict): {
                "samples": {
                    "audio": Tensor[N,T,C] where N is the number of audio clips, T is
                             the audio sequence length, and C is the number of channels.
                }
            }


        Returns:
            Dict: _description_
        """
        if not self.enable:
            return data

        audio = data["samples"]["audio"]  # N, T, C
        assert audio.ndim == 3, f"Invalid audio dimension {audio.ndim}. Expected 3."
        num_input_channels = audio.shape[2]

        if num_input_channels == self.num_channels:
            return data

        if (num_input_channels, self.num_channels) == (1, 2):
            audio = audio.repeat(1, 1, 2)  # N, T, 2
        elif (num_input_channels, self.num_channels) == (2, 1):
            audio = audio.mean(dim=2, keepdim=True)  # N, T, 1
        else:
            raise NotImplementedError(
                f"The logic for standardizing audio channels with input shape of"
                f" {audio.shape} to {self.num_channels} channels is not implemented."
            )

        data["samples"]["audio"] = audio
        return data
