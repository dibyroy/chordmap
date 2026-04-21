"""Stage 4: Merge chords + word timings onto a shared timeline."""
from __future__ import annotations

from pydantic import BaseModel

from pipeline.chords import ChordEvent
from pipeline.align import WordTiming

SNAP_THRESHOLD_MS = 200


class LyricLine(BaseModel):
    words: list[str]
    chord_markers: list[dict]  # {position: int, chord: str}


def merge(chords: list[ChordEvent], words: list[WordTiming]) -> list[LyricLine]:
    """Align chord changes to nearest word boundaries within SNAP_THRESHOLD_MS."""
    raise NotImplementedError
