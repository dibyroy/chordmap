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

    Uses a tiny Whisper transcription to find when vocals actually start
    (skipping instrumental intros), then runs wav2vec2 forced alignment from
    that onset. Falls back to uniform spacing if alignment produces nothing.
    """
    import whisperx
    import librosa

    # Force CPU — wav2vec2 alignment is fast enough on CPU.
    # Explicitly use the HuggingFace model name to bypass the torchaudio bundle
    # path (WAV2VEC2_ASR_BASE_960H), which raises RuntimeError on Windows with
    # torchaudio 2.8.0 due to an accelerator-detection bug.
    device = "cpu"
    duration = len(audio) / sr

    # wav2vec2 expects 16 kHz mono audio
    audio_16k = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)

    # Find when vocals actually begin so instrumental intros don't mislead the
    # CTC forced aligner into anchoring words against instrument sounds.
    vocal_onset = _find_vocal_onset(audio_16k, device)

    segments = [{"start": vocal_onset, "end": duration, "text": lyrics}]

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

    # If alignment produced no timestamps, fall back to uniform spacing
    if not timings:
        timings = _uniform_fallback(lyrics, duration, onset=vocal_onset)

    return timings


def _find_vocal_onset(audio_16k: np.ndarray, device: str) -> float:
    """Use tiny Whisper to find when speech first appears in the audio.

    This handles instrumental intros: if the first transcribed segment starts
    at t=8s, we tell the forced aligner to start looking from t=8s rather than
    t=0, preventing it from anchoring lyrics to intro sounds.
    Returns 0.0 on any failure (graceful degradation).
    """
    try:
        import whisperx
        model = whisperx.load_model("tiny", device, compute_type="float32")
        result = model.transcribe(audio_16k, batch_size=1)
        del model
        segments = result.get("segments", [])
        if segments:
            return float(segments[0]["start"])
    except Exception:
        pass
    return 0.0


def _uniform_fallback(lyrics: str, duration: float, onset: float = 0.0) -> list[WordTiming]:
    """Distribute words evenly from onset to end of track."""
    words = [w for w in lyrics.split() if w]
    if not words:
        return []
    usable = max(duration - onset, 0.0)
    step = usable / len(words)
    return [
        WordTiming(word=w, start=onset + i * step, end=onset + (i + 1) * step)
        for i, w in enumerate(words)
    ]
