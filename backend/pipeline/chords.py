"""Stage 2: Chord detection — chromagram → timestamped chord sequence."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class ChordEvent(BaseModel):
    start: float
    end: float
    chord: str
    confidence: float


def detect_chords(audio: np.ndarray, sr: int) -> list[ChordEvent]:
    """Run autochord on audio array and return timestamped chord events."""
    raise NotImplementedError
