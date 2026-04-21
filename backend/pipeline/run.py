"""End-to-end pipeline runner. CLI: python -m pipeline.run <audio> <lyrics>."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

from pipeline.preprocess import preprocess
from pipeline.chords import detect_chords
from pipeline.align import align_lyrics
from pipeline.merge import merge
from pipeline.analyze import analyze


class SongResult(BaseModel):
    chords: list[dict]
    words: list[dict]
    lines: list[dict]
    analysis: dict


def process_song(audio_path: str, lyrics: str) -> SongResult:
    audio, sr = preprocess(audio_path)
    chords = detect_chords(audio, sr)
    words = align_lyrics(audio, sr, lyrics)
    lines = merge(chords, words)
    analysis = analyze([c.model_dump() for c in chords])
    return SongResult(
        chords=[c.model_dump() for c in chords],
        words=[w.model_dump() for w in words],
        lines=[l.model_dump() for l in lines],
        analysis=analysis.model_dump(),
    )


if __name__ == "__main__":
    audio_path = sys.argv[1]
    lyrics_path = sys.argv[2]
    lyrics = Path(lyrics_path).read_text()
    result = process_song(audio_path, lyrics)
    print(json.dumps(result.model_dump(), indent=2))
