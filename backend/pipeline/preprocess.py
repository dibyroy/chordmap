"""Stage 1: Audio preprocessing — normalize, resample, trim silence."""
from __future__ import annotations

import librosa
import numpy as np


def preprocess(audio_path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load audio, convert to mono, resample to target_sr, trim silence.

    Returns (audio_array, sample_rate).
    """
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = librosa.util.normalize(audio)
    return audio, sr
