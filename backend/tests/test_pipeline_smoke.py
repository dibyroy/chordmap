"""Smoke test: full pipeline on a real audio file.

Requires data/test_song.mp3 and ANTHROPIC_API_KEY to be set.
Run manually: pytest tests/test_pipeline_smoke.py -s

To use: drop an mp3 at data/test_song.mp3 and set TEST_LYRICS below.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent.parent / "data"
TEST_AUDIO = DATA_DIR / "test_song.mp3"

TEST_LYRICS = """\
This is a placeholder
Replace with the actual lyrics
for the song you put in data/test_song.mp3
"""

_skip = pytest.mark.skipif(
    not TEST_AUDIO.exists() or not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires data/test_song.mp3 and ANTHROPIC_API_KEY",
)


@_skip
def test_pipeline_smoke_chords():
    from pipeline.preprocess import preprocess
    from pipeline.chords import detect_chords

    audio, sr = preprocess(str(TEST_AUDIO))
    chords = detect_chords(audio, sr)

    assert len(chords) > 0, "chord detection produced no results"
    print(f"\nDetected {len(chords)} chord events")
    for c in chords[:5]:
        print(f"  {c.start:.2f}-{c.end:.2f}s  {c.chord}")


@_skip
def test_pipeline_smoke_alignment():
    from pipeline.preprocess import preprocess
    from pipeline.align import align_lyrics

    audio, sr = preprocess(str(TEST_AUDIO))
    words = align_lyrics(audio, sr, TEST_LYRICS)

    assert len(words) > 0, "lyric alignment produced no results"
    print(f"\nAligned {len(words)} words")
    for w in words[:5]:
        print(f"  {w.start:.2f}-{w.end:.2f}s  {w.word!r}")


@_skip
def test_pipeline_smoke_full():
    from pipeline.run import process_song, SongResult

    result = process_song(str(TEST_AUDIO), TEST_LYRICS)

    assert isinstance(result, SongResult)
    assert result.chords, "no chords detected"
    assert result.words, "no word timings"
    assert result.lines, "no lyric lines"
    assert result.analysis.key, "no key detected"
    assert result.analysis.roman_numerals, "no roman numerals"

    print("\n--- Full pipeline result ---")
    print(json.dumps(result.model_dump(), indent=2))
