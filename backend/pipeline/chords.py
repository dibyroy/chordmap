"""Stage 2: Chord detection — chromagram → timestamped chord sequence."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import soundfile as sf
from pydantic import BaseModel


class ChordEvent(BaseModel):
    start: float
    end: float
    chord: str
    confidence: float


def detect_chords(audio: np.ndarray, sr: int) -> list[ChordEvent]:
    """Write audio to a temp WAV, run autochord, return timestamped chord events.

    autochord returns (start, end, chord) tuples without confidence scores;
    confidence is set to 1.0 until we have a model that exposes it.
    """
    import autochord

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        tmp_path = tmp.name

    try:
        raw = autochord.recognize(tmp_path)
    finally:
        os.unlink(tmp_path)

    return [
        ChordEvent(start=float(start), end=float(end), chord=str(chord), confidence=1.0)
        for start, end, chord in raw
        if chord != "N"  # skip "no chord" segments
    ]
