"""Tests for pipeline/analyze.py."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from pipeline.analyze import AnalysisResult, analyze

SAMPLE_CHORDS = [
    {"start": 0.0, "end": 2.0, "chord": "C", "confidence": 1.0},
    {"start": 2.0, "end": 4.0, "chord": "G", "confidence": 1.0},
    {"start": 4.0, "end": 6.0, "chord": "Am", "confidence": 1.0},
    {"start": 6.0, "end": 8.0, "chord": "F", "confidence": 1.0},
]

GOOD_RESPONSE = {
    "key": "C",
    "mode": "major",
    "roman_numerals": ["I", "V", "vi", "IV"],
    "progression_patterns": ["I-V-vi-IV"],
    "explanation": "This is C major using the classic pop progression.",
}


def _mock_message(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=content)]
    return msg


def test_analyze_parses_valid_response():
    with patch("pipeline.analyze.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _mock_message(
            json.dumps(GOOD_RESPONSE)
        )
        result = analyze(SAMPLE_CHORDS)

    assert isinstance(result, AnalysisResult)
    assert result.key == "C"
    assert result.mode == "major"
    assert result.roman_numerals == ["I", "V", "vi", "IV"]
    assert "I-V-vi-IV" in result.progression_patterns
    assert len(result.explanation) > 0


def test_analyze_raises_on_non_json():
    with patch("pipeline.analyze.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _mock_message(
            "sorry, I can't do that"
        )
        with pytest.raises(ValueError, match="non-JSON"):
            analyze(SAMPLE_CHORDS)


def test_analyze_raises_on_missing_fields():
    incomplete = {"key": "C"}  # missing mode, roman_numerals, etc.
    with patch("pipeline.analyze.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _mock_message(
            json.dumps(incomplete)
        )
        with pytest.raises(ValueError, match="missing required fields"):
            analyze(SAMPLE_CHORDS)


# Live API test — only runs when ANTHROPIC_API_KEY is set
@pytest.mark.skipif(
    True,  # flip to False to run against live API
    reason="Requires ANTHROPIC_API_KEY; run manually",
)
def test_analyze_live():
    result = analyze(SAMPLE_CHORDS)
    assert result.key in ("C", "A")  # lenient — model might disagree
    assert result.mode in ("major", "minor")
    assert len(result.roman_numerals) > 0
