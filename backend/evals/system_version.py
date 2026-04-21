"""Stable version ID from prompt file hash + model name + pipeline config."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def compute_version(prompt_path: Path, model: str, pipeline_config: dict) -> str:
    prompt_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()[:8]
    config_hash = hashlib.sha256(
        json.dumps(pipeline_config, sort_keys=True).encode()
    ).hexdigest()[:8]
    return f"{model}_p{prompt_hash}_c{config_hash}"
