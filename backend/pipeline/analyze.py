"""Stage 5: LLM analysis — chord sequence → key, roman numerals, progression."""
from __future__ import annotations

import json
from pathlib import Path

import anthropic
from pydantic import BaseModel, ValidationError

PROMPT_PATH = Path(__file__).parent / "prompts" / "analyze_v1.txt"
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024


class AnalysisResult(BaseModel):
    key: str
    mode: str
    roman_numerals: list[str]
    progression_patterns: list[str]
    explanation: str


def analyze(chords: list[dict]) -> AnalysisResult:
    """Call Claude with the versioned system prompt and return structured analysis.

    The system prompt is sent with cache_control so repeated calls on the
    same session hit the prompt cache rather than re-tokenizing each time.
    """
    client = anthropic.Anthropic()
    system_prompt = PROMPT_PATH.read_text()

    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": json.dumps(chords, indent=2),
            }
        ],
    )

    raw = message.content[0].text
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned non-JSON: {raw!r}") from exc

    try:
        return AnalysisResult(**data)
    except ValidationError as exc:
        raise ValueError(f"Model response missing required fields: {exc}") from exc
