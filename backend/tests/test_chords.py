"""Tests for pipeline/chords.py."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.chords import ChordEvent


def make_sine(freq: float = 440.0, duration: float = 5.0, sr: int = 22050) -> tuple[np.ndarray, int]:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr


def test_detect_chords_returns_chord_events():
    from pipeline.chords import detect_chords

    audio, sr = make_sine(duration=5.0)
    events = detect_chords(audio, sr)

    assert isinstance(events, list)
    # A pure sine wave at 440 Hz may or may not produce chords; just verify structure
    for e in events:
        assert isinstance(e, ChordEvent)
        assert e.start >= 0
        assert e.end > e.start
        assert isinstance(e.chord, str)
        assert 0.0 <= e.confidence <= 1.0


def test_detect_chords_no_silence_only():
    """Detector shouldn't crash on near-silent input."""
    from pipeline.chords import detect_chords

    audio = np.zeros(22050 * 3, dtype=np.float32)
    events = detect_chords(audio, 22050)

    assert isinstance(events, list)
