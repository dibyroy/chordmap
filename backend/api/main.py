"""FastAPI server — POST /analyze accepts audio + lyrics, returns pipeline result."""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline.run import process_song, SongResult

app = FastAPI(title="ChordMap API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=SongResult)
async def analyze(
    audio: UploadFile = File(...),
    lyrics: str | None = Form(None),
) -> SongResult:
    with tempfile.NamedTemporaryFile(suffix=Path(audio.filename).suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    return process_song(tmp_path, lyrics)
