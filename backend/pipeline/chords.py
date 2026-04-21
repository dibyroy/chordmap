"""Stage 2: Chord detection — chromagram template matching via librosa.

autochord was dropped: its underlying `vamp` package requires native VAMP
SDK binaries that don't build reliably on Windows. This implementation
uses librosa's CQT chromagram with cosine-similarity template matching,
which is the same algorithmic approach and has no native dependencies.
"""
from __future__ import annotations

import librosa
import numpy as np
from pydantic import BaseModel

# Chromatic note names (C=0 … B=11)
_NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Semitone offsets for each chord quality (root = 0)
_MAJOR = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)  # 1-3-5
_MINOR = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)  # 1-b3-5


def _build_templates() -> tuple[np.ndarray, list[str]]:
    """Build 24 normalised chord templates (12 major + 12 minor)."""
    templates, names = [], []
    for i in range(12):
        for tmpl, suffix in ((_MAJOR, ""), (_MINOR, "m")):
            vec = np.roll(tmpl, i).astype(float)
            vec /= np.linalg.norm(vec)
            templates.append(vec)
            names.append(_NOTES[i] + suffix)
    return np.stack(templates), names  # (24, 12)


_TEMPLATES, _NAMES = _build_templates()

# Analysis window — larger hop = smoother but coarser chord boundaries
_HOP = 4096
_MIN_DURATION_S = 0.5  # merge segments shorter than this into neighbours


class ChordEvent(BaseModel):
    start: float
    end: float
    chord: str
    confidence: float


def detect_chords(audio: np.ndarray, sr: int) -> list[ChordEvent]:
    """Return timestamped chord events detected via chromagram template matching."""
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=_HOP, bins_per_octave=36)
    chroma = librosa.util.normalize(chroma, axis=0)         # (12, n_frames)
    scores = _TEMPLATES @ chroma                            # (24, n_frames)
    best_idx = np.argmax(scores, axis=0)                    # (n_frames,)
    best_conf = np.max(scores, axis=0)                      # (n_frames,)
    frame_times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=sr, hop_length=_HOP
    )
    return _group_frames(best_idx, best_conf, frame_times)


def _group_frames(
    best_idx: np.ndarray,
    best_conf: np.ndarray,
    times: np.ndarray,
) -> list[ChordEvent]:
    """Merge consecutive same-chord frames into ChordEvents."""
    if len(best_idx) == 0:
        return []

    events: list[ChordEvent] = []
    i = 0
    while i < len(best_idx):
        ci = best_idx[i]
        j = i + 1
        while j < len(best_idx) and best_idx[j] == ci:
            j += 1

        start = float(times[i])
        end = float(times[j]) if j < len(times) else float(times[-1]) + (
            float(times[-1]) - float(times[-2]) if len(times) > 1 else 0.1
        )
        duration = end - start
        confidence = float(np.mean(best_conf[i:j]))

        if duration >= _MIN_DURATION_S:
            events.append(ChordEvent(
                start=start,
                end=end,
                chord=_NAMES[ci],
                confidence=confidence,
            ))
        i = j

    return events
