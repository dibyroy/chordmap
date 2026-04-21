"""End-to-end pipeline runner.

CLI usage:
    python -m pipeline.run <audio_path> <lyrics_path>

Output: JSON to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

from pipeline.preprocess import preprocess
from pipeline.chords import ChordEvent, detect_chords
from pipeline.align import WordTiming, align_lyrics
from pipeline.merge import LyricLine, merge
from pipeline.analyze import AnalysisResult, analyze


class SongResult(BaseModel):
    chords: list[ChordEvent]
    words: list[WordTiming]
    lines: list[LyricLine]
    analysis: AnalysisResult


def process_song(audio_path: str, lyrics: str) -> SongResult:
    """Run all five pipeline stages and return a complete result."""
    audio, sr = preprocess(audio_path)
    chords = detect_chords(audio, sr)
    words = align_lyrics(audio, sr, lyrics)
    lines = merge(chords, words, lyrics)
    analysis = analyze([c.model_dump() for c in chords])
    return SongResult(chords=chords, words=words, lines=lines, analysis=analysis)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m pipeline.run <audio_path> <lyrics_path>", file=sys.stderr)
        sys.exit(1)

    audio_path = sys.argv[1]
    lyrics = Path(sys.argv[2]).read_text()
    result = process_song(audio_path, lyrics)
    print(json.dumps(result.model_dump(), indent=2))
