"""Tests for pipeline/merge.py."""
from __future__ import annotations

import pytest

from pipeline.chords import ChordEvent
from pipeline.align import WordTiming
from pipeline.merge import merge, LyricLine


def words(*pairs: tuple[str, float, float]) -> list[WordTiming]:
    return [WordTiming(word=w, start=s, end=e) for w, s, e in pairs]


def chords(*triples: tuple[float, float, str]) -> list[ChordEvent]:
    return [ChordEvent(start=s, end=e, chord=c, confidence=1.0) for s, e, c in triples]


def test_empty_words():
    assert merge([], []) == []


def test_single_line_no_pauses():
    ws = words(("hello", 0.0, 0.5), ("world", 0.6, 1.0))
    cs = chords((0.0, 2.0, "C"))

    lines = merge(cs, ws)

    assert len(lines) == 1
    assert lines[0].words == ["hello", "world"]
    assert lines[0].chord_markers[0]["chord"] == "C"
    assert lines[0].chord_markers[0]["position"] == 0


def test_chord_snaps_to_nearest_word():
    # Chord starts at 0.55 — "world" starts at 0.5, closer than "hello" at 0.0
    ws = words(("hello", 0.0, 0.4), ("world", 0.5, 1.0))
    cs = chords((0.55, 1.5, "G"))

    lines = merge(cs, ws)

    assert lines[0].chord_markers[0]["position"] == 1  # snapped to "world"


def test_line_break_on_pause():
    # 2s gap between words triggers a line break
    ws = words(
        ("verse", 0.0, 0.5),
        ("one", 0.6, 1.0),
        ("chorus", 3.5, 4.0),
        ("now", 4.1, 4.5),
    )
    cs = []

    lines = merge(cs, ws)

    assert len(lines) == 2
    assert lines[0].words == ["verse", "one"]
    assert lines[1].words == ["chorus", "now"]


def test_chord_outside_line_not_attached():
    ws = words(("only", 1.0, 1.5))
    cs = chords((5.0, 6.0, "Dm"))  # chord is after the word

    lines = merge(cs, ws)

    assert lines[0].chord_markers == []


def test_duplicate_position_deduped():
    # Two chords very close together that would both snap to position 0
    ws = words(("word", 0.0, 0.5))
    cs = chords((0.0, 0.2, "C"), (0.05, 0.25, "G"))

    lines = merge(cs, ws)

    # Only one marker at position 0, not two
    positions = [m["position"] for m in lines[0].chord_markers]
    assert positions == list(set(positions))
