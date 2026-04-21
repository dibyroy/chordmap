"""Tests for pipeline/align.py."""
from __future__ import annotations

import numpy as np
import pytest


LYRICS = "hello world this is a test"


def make_sine(duration: float = 3.0, sr: int = 22050) -> tuple[np.ndarray, int]:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr


def test_uniform_fallback():
    from pipeline.align import _uniform_fallback

    timings = _uniform_fallback(LYRICS, duration=6.0)
    words = LYRICS.split()

    assert len(timings) == len(words)
    for i, t in enumerate(timings):
        assert t.word == words[i]
        assert t.start >= 0
        assert t.end > t.start


def test_uniform_fallback_empty_lyrics():
    from pipeline.align import _uniform_fallback

    assert _uniform_fallback("", 5.0) == []


def test_uniform_fallback_covers_full_duration():
    from pipeline.align import _uniform_fallback

    timings = _uniform_fallback("one two three", 9.0)
    assert timings[0].start == pytest.approx(0.0)
    assert timings[-1].end == pytest.approx(9.0, abs=0.01)


# WhisperX integration test — only runs when model weights are available
@pytest.mark.skipif(
    True,  # flip to False to run manually after `pip install whisperx`
    reason="Requires WhisperX model weights; run manually",
)
def test_align_lyrics_returns_timings():
    from pipeline.align import align_lyrics

    audio, sr = make_sine(duration=5.0)
    timings = align_lyrics(audio, sr, LYRICS)

    assert len(timings) > 0
    for t in timings:
        assert t.start >= 0
        assert t.end >= t.start
        assert isinstance(t.word, str)
