"""Stage 3: Lyric alignment — forced alignment via WhisperX."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class WordTiming(BaseModel):
    word: str
    start: float
    end: float


def align_lyrics(audio: np.ndarray, sr: int, lyrics: str) -> list[WordTiming]:
    """Run WhisperX forced alignment and return word-level timestamps."""
    raise NotImplementedError
