"""Quick test UI for the pipeline (stages 1-4, no LLM).

Run from backend/:
    streamlit run test_app.py
"""
from __future__ import annotations

import os
import tempfile
import time

import streamlit as st

st.set_page_config(page_title="ChordMap — pipeline test", layout="wide")
st.title("ChordMap pipeline tester")
st.caption("Stages 1–4 only — no API key needed")

# ── inputs ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    audio_file = st.file_uploader("Audio file (mp3 / wav / flac)", type=["mp3", "wav", "flac", "m4a"])
    run_alignment = st.checkbox("Run lyric alignment (WhisperX — downloads ~360 MB on first run)", value=False)

with col_right:
    lyrics = st.text_area("Lyrics (optional — leave blank to auto-transcribe with Whisper large-v3)", height=200,
                          placeholder="Paste song lyrics here, or leave blank…")

run_btn = st.button("Run pipeline", type="primary", disabled=audio_file is None)

if not run_btn:
    st.stop()

# ── stage 1: preprocess ───────────────────────────────────────────────────────
with st.status("Stage 1 — preprocessing audio…", expanded=True) as status:
    from pipeline.preprocess import preprocess

    suffix = os.path.splitext(audio_file.name)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    t0 = time.perf_counter()
    audio, sr = preprocess(tmp_path)
    elapsed = time.perf_counter() - t0

    duration_s = len(audio) / sr
    st.write(f"✓ {duration_s:.1f}s of audio at {sr} Hz  ·  {elapsed:.2f}s")
    status.update(label="Stage 1 — done", state="complete")

# ── stage 2: chord detection ──────────────────────────────────────────────────
with st.status("Stage 2 — detecting chords…", expanded=True) as status:
    from pipeline.chords import detect_chords

    t0 = time.perf_counter()
    chords = detect_chords(audio, sr)
    elapsed = time.perf_counter() - t0

    st.write(f"✓ {len(chords)} chord events  ·  {elapsed:.2f}s")
    status.update(label="Stage 2 — done", state="complete")

# ── chord results ─────────────────────────────────────────────────────────────
st.subheader(f"Chords ({len(chords)} segments)")

if chords:
    import pandas as pd

    df = pd.DataFrame([c.model_dump() for c in chords])
    df["duration"] = (df["end"] - df["start"]).round(2)
    df["start"] = df["start"].round(2)
    df["end"] = df["end"].round(2)
    df["confidence"] = df["confidence"].round(3)
    st.dataframe(df[["start", "end", "duration", "chord", "confidence"]], width="stretch")

    unique_chords = [c.chord for c in chords]
    st.write("**Progression:** " + "  →  ".join(unique_chords))
else:
    st.warning("No chords detected — try a longer or clearer audio clip.")

# ── stage 3 + 4: alignment + merge (optional) ────────────────────────────────
if not run_alignment:
    st.info("Enable 'Run lyric alignment' above to see the chord sheet.")
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    st.stop()

with st.status("Stage 3 — aligning lyrics (WhisperX)…", expanded=True) as status:
    from pipeline.align import align_lyrics

    t0 = time.perf_counter()
    words, effective_lyrics = align_lyrics(audio, sr, lyrics.strip() or None)
    elapsed = time.perf_counter() - t0

    st.write(f"✓ {len(words)} word timestamps  ·  {elapsed:.2f}s")
    if not lyrics.strip():
        st.info("Auto-transcribed lyrics shown below.")

    # Word count sanity check
    expected_words = len(effective_lyrics.split())
    delta = len(words) - expected_words
    if abs(delta) > expected_words * 0.1:
        st.warning(f"⚠ Expected ~{expected_words} words, got {len(words)} "
                   f"({delta:+d}). Alignment may have drifted.")

    status.update(label="Stage 3 — done", state="complete")

# ── diagnostics expander ──────────────────────────────────────────────────────
with st.expander("🔍 Diagnostics — raw word timings"):
    wdf = pd.DataFrame([w.model_dump() for w in words])
    wdf["start"] = wdf["start"].round(3)
    wdf["end"] = wdf["end"].round(3)
    wdf["duration"] = (wdf["end"] - wdf["start"]).round(3)
    st.dataframe(wdf, width="stretch")

    st.caption(f"Expected {expected_words} words from lyrics · "
               f"WhisperX returned {len(words)} words · "
               f"Audio duration {duration_s:.1f}s")

with st.status("Stage 4 — merging chords + lyrics…", expanded=True) as status:
    from pipeline.merge import merge

    t0 = time.perf_counter()
    lines = merge(chords, words, effective_lyrics)
    elapsed = time.perf_counter() - t0

    st.write(f"✓ {len(lines)} lyric lines  ·  {elapsed:.2f}s")
    status.update(label="Stage 4 — done", state="complete")

# ── chord sheet ───────────────────────────────────────────────────────────────
st.subheader("Chord sheet")

def _render_ug_line(words: list[str], chord_markers: list[dict]) -> str:
    """Return a two-row string with chords above lyrics, UltimateGuitar style."""
    lyrics_line = " ".join(words)

    char_offsets: list[int] = []
    pos = 0
    for w in words:
        char_offsets.append(pos)
        pos += len(w) + 1

    chord_line = list(" " * len(lyrics_line))
    write_until = 0

    for marker in sorted(chord_markers, key=lambda m: m["position"]):
        idx = marker["position"]
        chord = marker["chord"]
        if idx >= len(char_offsets):
            continue
        start = max(char_offsets[idx], write_until)
        needed = start + len(chord)
        if needed > len(chord_line):
            chord_line.extend([" "] * (needed - len(chord_line)))
        for i, ch in enumerate(chord):
            chord_line[start + i] = ch
        write_until = start + len(chord) + 1

    chord_str = "".join(chord_line).rstrip()
    return f"{chord_str}\n{lyrics_line}" if chord_str.strip() else lyrics_line


def _render_instrumental_line(chord_markers: list[dict]) -> str:
    """Render an instrumental section: chords evenly spaced above the label."""
    chords = [m["chord"] for m in sorted(chord_markers, key=lambda m: m["position"])]
    if not chords:
        return "[Instrumental]"
    slot = max(max(len(c) for c in chords) + 2, 6)
    chord_line = "".join(f"{c:<{slot}}" for c in chords).rstrip()
    return f"{chord_line}\n[Instrumental]"


for line in lines:
    if line.is_instrumental:
        rendered = _render_instrumental_line(line.chord_markers)
    else:
        rendered = _render_ug_line(line.words, line.chord_markers)
    st.code(rendered, language=None)

try:
    os.unlink(tmp_path)
except OSError:
    pass
