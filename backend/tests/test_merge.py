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


# ── gap-based fallback (no lyrics supplied) ───────────────────────────────────

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
    ws = words(("hello", 0.0, 0.4), ("world", 0.5, 1.0))
    cs = chords((0.55, 1.5, "G"))

    lines = merge(cs, ws)

    assert lines[0].chord_markers[0]["position"] == 1


def test_line_break_on_pause():
    ws = words(
        ("verse", 0.0, 0.5),
        ("one", 0.6, 1.0),
        ("chorus", 3.5, 4.0),
        ("now", 4.1, 4.5),
    )
    lines = merge([], ws)

    assert len(lines) == 2
    assert lines[0].words == ["verse", "one"]
    assert lines[1].words == ["chorus", "now"]


def test_chord_outside_line_not_attached():
    ws = words(("only", 1.0, 1.5))
    cs = chords((5.0, 6.0, "Dm"))

    lines = merge(cs, ws)

    assert lines[0].chord_markers == []


def test_duplicate_position_deduped():
    ws = words(("word", 0.0, 0.5))
    cs = chords((0.0, 0.2, "C"), (0.05, 0.25, "G"))

    lines = merge(cs, ws)

    positions = [m["position"] for m in lines[0].chord_markers]
    assert positions == list(set(positions))


# ── line-structure merge (lyrics supplied) ────────────────────────────────────

def test_lyrics_line_structure_used():
    # 4 words split across 2 lyric lines — should produce 2 output lines
    # regardless of timing gaps between words
    ws = words(
        ("hello", 0.0, 0.3),
        ("world", 0.35, 0.7),   # gap < 1s — gap-based would merge these
        ("foo", 0.8, 1.1),
        ("bar", 1.15, 1.5),
    )
    lyrics = "hello world\nfoo bar"

    lines = merge([], ws, lyrics)

    assert len(lines) == 2
    assert lines[0].words == ["hello", "world"]
    assert lines[1].words == ["foo", "bar"]


def test_lyrics_chord_placed_on_correct_line():
    ws = words(
        ("hello", 0.0, 0.5),
        ("world", 0.6, 1.0),
        ("foo", 1.1, 1.5),
        ("bar", 1.6, 2.0),
    )
    cs = chords((1.1, 2.0, "Am"))  # starts at "foo" — should be on line 2
    lyrics = "hello world\nfoo bar"

    lines = merge(cs, ws, lyrics)

    assert lines[0].chord_markers == []         # no chord on first line
    assert lines[1].chord_markers[0]["chord"] == "Am"


def test_lyrics_fallback_when_alignment_short():
    # alignment returned fewer words than lyrics contain
    ws = words(("hello", 0.0, 0.5))  # only 1 word
    lyrics = "hello world\nfoo bar"

    lines = merge([], ws, lyrics)

    # First line gets the one timed word; second line gets raw text, no markers
    assert lines[0].words == ["hello"]
    assert lines[1].words == ["foo", "bar"]
    assert lines[1].chord_markers == []
