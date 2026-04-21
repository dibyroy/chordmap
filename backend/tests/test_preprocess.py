"""Tests for pipeline/preprocess.py."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf


def make_tone(freq: float = 440.0, duration: float = 2.0, sr: int = 22050) -> tuple[np.ndarray, int]:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return audio, sr


def test_preprocess_returns_mono_at_target_sr(tmp_path):
    from pipeline.preprocess import preprocess

    audio, sr = make_tone()
    stereo = np.stack([audio, audio], axis=1)
    path = str(tmp_path / "test.wav")
    sf.write(path, stereo, sr)

    result, result_sr = preprocess(path, target_sr=22050)

    assert result.ndim == 1
    assert result_sr == 22050


def test_preprocess_normalizes(tmp_path):
    from pipeline.preprocess import preprocess

    audio, sr = make_tone()
    audio_quiet = audio * 0.01
    path = str(tmp_path / "quiet.wav")
    sf.write(path, audio_quiet, sr)

    result, _ = preprocess(path)

    assert np.max(np.abs(result)) == pytest.approx(1.0, abs=0.01)


def test_preprocess_trims_silence(tmp_path):
    from pipeline.preprocess import preprocess

    audio, sr = make_tone(duration=1.0)
    silence = np.zeros(int(sr * 1.0), dtype=np.float32)
    padded = np.concatenate([silence, audio, silence])
    path = str(tmp_path / "padded.wav")
    sf.write(path, padded, sr)

    result, _ = preprocess(path)

    # trimmed result should be shorter than original padded signal
    assert len(result) < len(padded)
