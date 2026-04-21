"""Stage 3: Lyric alignment — forced alignment via WhisperX."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class WordTiming(BaseModel):
    word: str
    start: float
    end: float


def align_lyrics(audio: np.ndarray, sr: int, lyrics: str) -> list[WordTiming]:
    """Run WhisperX forced alignment against provided lyrics.

    Passes the full lyrics as a single segment; WhisperX wav2vec2 aligner
    then assigns word-level timestamps. Falls back to uniform spacing if
    any word has no timestamp (can happen with low-quality audio).
    """
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    duration = len(audio) / sr

    segments = [{"start": 0.0, "end": duration, "text": lyrics}]

    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)

    audio_f32 = audio.astype(np.float32)
    result = whisperx.align(
        segments,
        model_a,
        metadata,
        audio_f32,
        sr,
        device,
        return_char_alignments=False,
    )

    timings: list[WordTiming] = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            word = word_info.get("word", "").strip()
            if not word:
                continue
            timings.append(WordTiming(
                word=word,
                start=float(word_info.get("start", 0.0)),
                end=float(word_info.get("end", 0.0)),
            ))

    # If alignment produced no timestamps, fall back to uniform spacing
    if not timings:
        timings = _uniform_fallback(lyrics, duration)

    return timings


def _uniform_fallback(lyrics: str, duration: float) -> list[WordTiming]:
    """Distribute words evenly across the track duration."""
    words = [w for w in lyrics.split() if w]
    if not words:
        return []
    step = duration / len(words)
    return [
        WordTiming(word=w, start=i * step, end=(i + 1) * step)
        for i, w in enumerate(words)
    ]
