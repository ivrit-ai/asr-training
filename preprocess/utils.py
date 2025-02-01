from functools import lru_cache
from typing import Tuple

import numpy as np


# Pulled off-of
# https://github.com/iver56/audiomentations/blob/main/audiomentations/core/utils.py
# Since no ned for the entire lib for the simple augmentation we perform.
# consider depending on the entire lib for more complex augmentations.
# (and remove this code!)
@lru_cache(maxsize=8)
def get_crossfade_mask_pair(length: int, equal_energy=True) -> Tuple[np.array, np.array]:
    """
    Equal-gain or equal-energy (within ~1%) cross-fade mask pair with
    smooth start and end.
    https://signalsmith-audio.co.uk/writing/2021/cheap-energy-crossfade/
    """
    x = np.linspace(0, 1, length, dtype=np.float32)
    x2 = 1 - x
    a = x * x2
    k = 1.4186 if equal_energy else -0.70912
    b = a * (1 + k * a)
    c = b + x
    d = b + x2
    fade_in = c * c
    fade_out = d * d
    return fade_in, fade_out
