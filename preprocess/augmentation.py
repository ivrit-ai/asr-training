import random
from typing import List, Optional, Tuple

import librosa
import numpy as np
from audiomentations import AddColorNoise, BandPassFilter, Compose, Gain, Resample
from audiomentations.core.transforms_interface import BaseWaveformTransform
from numpy.typing import NDArray

from .utils import get_crossfade_mask_pair

# This is the default in:
# https://iver56.github.io/audiomentations/waveform_transforms/shift/
# We assume it's a sane choice for this simple augmentation
fade_duration = 0.005


# Loosely based on:
# https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/shift.py
# With simpler interface and only fwd shift.
# also assumes extending the audio is required
def shift_audio_forward(audio: NDArray[np.float32], shift_sec: float, sample_rate: int) -> NDArray[np.float32]:
    assert shift_sec >= 0, "shift_sec must be non-negative"
    num_places_to_shift = int(round(shift_sec * sample_rate))

    # Assume audio length needs to be extended by the shift amount
    shifted_samples = np.zeros(audio.shape[-1] + num_places_to_shift, dtype=audio.dtype)

    # Copy the original audio to the shifted position
    shifted_samples[num_places_to_shift:] = audio
    # Apply a crossfade to smooth the transition
    fade_length = int(sample_rate * fade_duration)
    fade_in, fade_out = get_crossfade_mask_pair(fade_length)

    fade_in_start = num_places_to_shift
    fade_in_end = min(num_places_to_shift + fade_length, shifted_samples.shape[-1])
    fade_in_length = fade_in_end - fade_in_start

    shifted_samples[
        ...,
        fade_in_start:fade_in_end,
    ] *= fade_in[:fade_in_length]

    return shifted_samples


class SampleDownUpResample(BaseWaveformTransform):
    """
    Sample Down then back up to a original sample rate.

    """

    supports_multichannel = True

    def __init__(self, down_to_sample_rate: int = 8000, p: float = 0.5):
        """
        :param down_to_sample_rate: The sample rate to down sample to
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.down_to_sample_rate = down_to_sample_rate
        self.resample_down = Resample(p=1, min_sample_rate=down_to_sample_rate, max_sample_rate=down_to_sample_rate)

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        # From origin sample rate to down to sample rate
        samples = self.resample_down(samples=samples, sample_rate=sample_rate)

        samples = librosa.core.resample(
            samples,
            orig_sr=self.down_to_sample_rate,
            target_sr=sample_rate,
        )
        return samples


not_augmented_compose = None
noised_down_sampling_compose = Compose(
    [
        AddColorNoise(p=1),
        SampleDownUpResample(down_to_sample_rate=4000, p=1),
    ]
)
bandpass_down_sampling_compose = Compose(
    [
        BandPassFilter(
            p=1, min_center_freq=2500, max_center_freq=2500, min_bandwidth_fraction=0.8, max_bandwidth_fraction=0.8
        ),
        SampleDownUpResample(down_to_sample_rate=4000, p=1),
        Gain(p=1, min_gain_db=0),
    ]
)
noised_bandpass_compose = Compose(
    [
        AddColorNoise(p=1),
        BandPassFilter(
            p=1, min_center_freq=2500, max_center_freq=2500, min_bandwidth_fraction=0.8, max_bandwidth_fraction=0.8
        ),
        Gain(p=1, min_gain_db=0),
    ]
)


class AudioAugmenter:
    """Class to apply stochastic noise/compression/transforms to audio samples."""

    def __init__(
        self,
        augmentations: Optional[List[Tuple[Compose, float]]] = None,
    ):
        """Initialize the audio augmenter.
        Augmentation probabilities must sum to at most 1 (any left over mean no augmentation prob.)

        Args:
            augmentations: List of (augmentation, probability) tuples
        """
        self.augmentations = augmentations or []

        augmentation_total_prob = sum(prob for _, prob in self.augmentations)
        if augmentation_total_prob == 0:
            self.augmentations = [(not_augmented_compose, 1.0)]
        elif augmentation_total_prob > 1:
            raise ValueError("Augmentation probabilities must sum to at most 1")
        elif augmentation_total_prob < 1:
            no_augmentation_prob = 1 - augmentation_total_prob
            self.augmentations.append((not_augmented_compose, no_augmentation_prob))

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply a randomly selected augmentation to the audio.

        Args:
            audio: Audio signal as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Augmented audio signal
        """
        # Select an augmentation based on probabilities
        augmentations, probs = zip(*self.augmentations)
        selected_augmentation = random.choices(augmentations, weights=probs, k=1)[0]

        if selected_augmentation is None:
            return audio

        # Apply the selected augmentation
        return selected_augmentation(samples=audio, sample_rate=sample_rate)

    @classmethod
    def create_preset(cls, preset_name: str, prob: float = 1.0) -> "AudioAugmenter":
        """Create an AudioAugmenter with a predefined set of augmentations.

        Args:
            preset_name: Name of the preset
            **kwargs: Additional parameters for the preset

        Returns:
            AudioAugmenter instance with the preset augmentations
        """
        if preset_name == "degrade":
            operations = [noised_down_sampling_compose, noised_bandpass_compose, bandpass_down_sampling_compose]
            operations_probs = [0.4, 0.3, 0.3]
            scaled_compose_probs = [p * prob for p in operations_probs]
            return cls(list(zip(operations, scaled_compose_probs)))
        else:
            raise ValueError(f"Unknown preset: {preset_name}")
