"""Synthetic error injection for robustness evaluation (Suite C)."""
from __future__ import annotations

import random
from copy import deepcopy

from pipeline.chords import ChordEvent

RELATIVE_PAIRS = {
    "C": "Am", "G": "Em", "D": "Bm", "A": "F#m", "E": "C#m",
    "Am": "C", "Em": "G", "Bm": "D", "F#m": "A", "C#m": "E",
}


def substitute_relative(chords: list[ChordEvent], rate: float, rng: random.Random | None = None) -> list[ChordEvent]:
    """Replace a fraction of chords with their relative major/minor."""
    rng = rng or random.Random()
    result = deepcopy(chords)
    for event in result:
        if rng.random() < rate and event.chord in RELATIVE_PAIRS:
            event.chord = RELATIVE_PAIRS[event.chord]
    return result


def drop_sevenths(chords: list[ChordEvent], rate: float, rng: random.Random | None = None) -> list[ChordEvent]:
    """Strip seventh extensions (maj7, min7, 7) from a fraction of chords."""
    rng = rng or random.Random()
    result = deepcopy(chords)
    for event in result:
        if rng.random() < rate:
            event.chord = event.chord.replace("maj7", "").replace("min7", "").replace("m7", "m").replace("7", "")
    return result


def shift_timing(chords: list[ChordEvent], max_ms: float, rng: random.Random | None = None) -> list[ChordEvent]:
    """Randomly shift chord start/end times by ±max_ms milliseconds."""
    rng = rng or random.Random()
    result = deepcopy(chords)
    for event in result:
        offset = rng.uniform(-max_ms / 1000, max_ms / 1000)
        event.start = max(0.0, event.start + offset)
        event.end = max(event.start + 0.01, event.end + offset)
    return result


def swap_adjacent(chords: list[ChordEvent], rate: float, rng: random.Random | None = None) -> list[ChordEvent]:
    """Swap adjacent chord pairs at the given rate."""
    rng = rng or random.Random()
    result = deepcopy(chords)
    i = 0
    while i < len(result) - 1:
        if rng.random() < rate:
            result[i].chord, result[i + 1].chord = result[i + 1].chord, result[i].chord
            i += 2
        else:
            i += 1
    return result
