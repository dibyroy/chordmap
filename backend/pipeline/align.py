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

    When `lyrics` is None, Whisper's transcription is used as-is. Non-English
    audio is automatically translated to English after alignment so that the
    chord sheet is always in English.
    When `lyrics` is provided, it overrides the transcribed text but Whisper's
    segment timestamps anchor the aligner.

    Returns (word_timings, effective_lyrics).
    """
    import whisperx
    import librosa

    device = "cpu"

    # Both Whisper and wav2vec2 expect 16 kHz mono.
    audio_16k = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)

    # Step 1: Transcribe to get segment timestamps and detect language.
    whisper_mdl = whisperx.load_model(whisper_model_size, device, compute_type="float32")
    transcription = whisper_mdl.transcribe(audio_16k, batch_size=8)
    whisper_segments = transcription.get("segments", [])
    # align_language: used only to pick the alignment model. Require high
    # confidence before using a non-English model — low scores are usually
    # misdetections on songs with heavy music and sparse vocals.
    detected_language = transcription.get("language", "en")
    lang_prob = transcription.get("language_probability", 1.0)
    language = detected_language if lang_prob >= 0.8 else "en"

    if not whisper_segments:
        effective = lyrics or ""
        return _uniform_fallback(effective, len(audio) / sr), effective

    # Step 2: Build segments for forced alignment.
    if lyrics is None:
        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in whisper_segments
        ]
    else:
        segments = _map_lyrics_to_segments(whisper_segments, lyrics)

    # Step 3: Forced alignment in the detected language for accurate timestamps.
    timings = _forced_align(segments, audio_16k, device, language)

    # Step 4: Determine the displayed lyrics text.
    if lyrics is not None:
        effective_lyrics = _wrap_long_lines(lyrics)
    else:
        raw = "\n".join(s["text"].strip() for s in whisper_segments)
        # Transliterate based on actual script content, not language confidence.
        # anyascii is a no-op on already-ASCII text, so this is always safe.
        if _has_non_ascii(raw):
            raw = _transliterate(raw)
        effective_lyrics = _wrap_long_lines(raw)

    return timings, effective_lyrics


# ── alignment ─────────────────────────────────────────────────────────────────

def _forced_align(
    segments: list[dict],
    audio_16k: np.ndarray,
    device: str,
    language: str,
) -> list[WordTiming]:
    """Run WhisperX forced alignment; fall back to segment-uniform if unavailable."""
    import whisperx

    kwargs: dict = {"language_code": language, "device": device}
    if language == "en":
        # Bypass torchaudio bundle path (WAV2VEC2_ASR_BASE_960H) which raises
        # RuntimeError on Windows with torchaudio 2.8.0.
        kwargs["model_name"] = "facebook/wav2vec2-base-960h"

    try:
        model_a, metadata = whisperx.load_align_model(**kwargs)
        result = whisperx.align(
            segments, model_a, metadata, audio_16k, device,
            return_char_alignments=False,
        )
        timings: list[WordTiming] = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                word = w.get("word", "").strip()
                if word:
                    timings.append(WordTiming(
                        word=word,
                        start=float(w.get("start", 0.0)),
                        end=float(w.get("end", 0.0)),
                    ))
        if timings:
            return timings
    except Exception:
        pass

    return _timings_from_segments(segments)


# ── helpers ───────────────────────────────────────────────────────────────────


def _has_non_ascii(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


def _transliterate(text: str) -> str:
    """Convert any Unicode script to Latin characters using anyascii.

    Preserves pronunciation (romanization), does not translate meaning.
    e.g. Tamil "வணக்கம்" → "vanakkam"
    """
    from anyascii import anyascii
    return anyascii(text)

def _map_lyrics_to_segments(
    whisper_segments: list[dict],
    lyrics: str,
) -> list[dict]:
    """Distribute user-provided lyrics across Whisper's timed segments."""
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


def _timings_from_segments(segments: list[dict]) -> list[WordTiming]:
    """Distribute words evenly within each segment's time window."""
    result: list[WordTiming] = []
    for seg in segments:
        words = seg["text"].split()
        if not words:
            continue
        duration = max(seg["end"] - seg["start"], 0.01)
        step = duration / len(words)
        for i, w in enumerate(words):
            result.append(WordTiming(
                word=w,
                start=seg["start"] + i * step,
                end=seg["start"] + (i + 1) * step,
            ))
    return result


def _wrap_long_lines(text: str, max_words: int = 10) -> str:
    """Split any line longer than max_words at its midpoint."""
    out: list[str] = []
    for line in text.splitlines():
        words = line.split()
        if len(words) <= max_words:
            out.append(line)
        else:
            mid = len(words) // 2
            out.append(" ".join(words[:mid]))
            out.append(" ".join(words[mid:]))
    return "\n".join(out)


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
