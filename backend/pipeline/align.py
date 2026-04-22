"""Stage 3: Lyric alignment — Whisper transcription + WhisperX forced alignment."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class WordTiming(BaseModel):
    word: str
    start: float
    end: float


def align_lyrics(
    audio: np.ndarray,
    sr: int,
    lyrics: str | None = None,
    whisper_model_size: str = "large-v3",
) -> tuple[list[WordTiming], str]:
    """Transcribe with Whisper then run WhisperX forced alignment.

    When `lyrics` is None, Whisper's transcription is used as-is.
    When `lyrics` is provided, it overrides the transcribed text but Whisper's
    segment timestamps anchor the aligner (fixes intro-music and mid-song drift).

    Returns (word_timings, effective_lyrics).
    """
    import whisperx
    import librosa

    device = "cpu"

    # Both Whisper and wav2vec2 expect 16 kHz mono.
    audio_16k = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)

    # Step 1: Whisper transcription to get segment-level timestamps.
    whisper_mdl = whisperx.load_model(whisper_model_size, device, compute_type="float32")
    transcription = whisper_mdl.transcribe(audio_16k, batch_size=8)
    whisper_segments = transcription.get("segments", [])

    if not whisper_segments:
        effective = lyrics or ""
        return _uniform_fallback(effective, len(audio) / sr), effective

    # Step 2: Build segments for forced alignment.
    if lyrics is None:
        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in whisper_segments
        ]
        effective_lyrics = " ".join(s["text"].strip() for s in whisper_segments)
    else:
        # Keep Whisper's timestamps but distribute user-provided lyrics across them.
        segments = _map_lyrics_to_segments(whisper_segments, lyrics)
        effective_lyrics = lyrics

    # Step 3: WhisperX forced alignment for word-level timestamps.
    model_a, metadata = whisperx.load_align_model(
        language_code="en",
        device=device,
        model_name="facebook/wav2vec2-base-960h",
    )

    result = whisperx.align(
        segments,
        model_a,
        metadata,
        audio_16k,
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

    if not timings:
        timings = _uniform_fallback(effective_lyrics, len(audio_16k) / 16000)

    return timings, effective_lyrics


def _map_lyrics_to_segments(
    whisper_segments: list[dict],
    lyrics: str,
) -> list[dict]:
    """Distribute user-provided lyrics across Whisper's timed segments.

    Whisper word count per segment is used as the weighting signal.
    """
    user_words = lyrics.split()
    total_user_words = len(user_words)
    whisper_word_counts = [len(s["text"].split()) for s in whisper_segments]
    total_whisper_words = sum(whisper_word_counts) or 1

    result: list[dict] = []
    cursor = 0

    for i, seg in enumerate(whisper_segments):
        if i == len(whisper_segments) - 1:
            chunk_words = user_words[cursor:]
        else:
            n = round(whisper_word_counts[i] / total_whisper_words * total_user_words)
            chunk_words = user_words[cursor: cursor + n]
            cursor += n

        result.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": " ".join(chunk_words),
        })

    return result


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
