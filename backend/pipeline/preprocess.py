"""Stage 1: Audio preprocessing — normalize, resample, trim silence."""
from __future__ import annotations

import numpy as np


def preprocess(audio_path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load audio, convert to mono, resample, trim leading/trailing silence.

    Returns (audio_array, sample_rate).
    """
    raise NotImplementedError
