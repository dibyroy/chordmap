"""Eval data models: EvalExample, EvalResult, EvalRun."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EvalExample(BaseModel):
    id: str
    input: dict[str, Any]
    ground_truth: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    example_id: str
    prediction: dict[str, Any]
    scores: dict[str, float]
    passed: bool
    notes: str = ""


class EvalRun(BaseModel):
    run_id: str
    suite: str
    system_version: str
    dataset: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    results: list[EvalResult] = Field(default_factory=list)
    aggregate: dict[str, float] = Field(default_factory=dict)

    def compute_aggregate(self) -> None:
        if not self.results:
            return
        all_scores: dict[str, list[float]] = {}
        for r in self.results:
            for k, v in r.scores.items():
                all_scores.setdefault(k, []).append(v)
        self.aggregate = {k: sum(v) / len(v) for k, v in all_scores.items()}
