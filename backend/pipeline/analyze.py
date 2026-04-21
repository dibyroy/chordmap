"""Stage 5: LLM analysis — chord sequence → key, roman numerals, progression."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

PROMPT_PATH = Path(__file__).parent / "prompts" / "analyze_v1.txt"
MODEL = "claude-sonnet-4-6"


class AnalysisResult(BaseModel):
    key: str
    mode: str
    roman_numerals: list[str]
    progression_patterns: list[str]
    explanation: str


def analyze(chords: list[dict]) -> AnalysisResult:
    """Call Claude with versioned prompt and return structured musical analysis."""
    raise NotImplementedError
