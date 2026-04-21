"""Stage 4: Merge chords + word timings onto a shared timeline."""
from __future__ import annotations

import re

from pydantic import BaseModel

from pipeline.chords import ChordEvent
from pipeline.align import WordTiming

SNAP_THRESHOLD_MS = 200
# Chord changes often land just before the downbeat word (musician plays the
# chord slightly ahead of the beat). Allow this window so those chords aren't
# silently dropped.
PRE_ROLL_S = 0.1


class LyricLine(BaseModel):
    words: list[str]
    # Each marker: {"position": word_index, "chord": "Am"}
    chord_markers: list[dict]


def merge(
    chords: list[ChordEvent],
    words: list[WordTiming],
    lyrics: str = "",
) -> list[LyricLine]:
    """Group words into lines and attach chord change markers.

    When `lyrics` is supplied the original newline structure is used for line
    grouping (most accurate). Without it we fall back to inferring breaks from
    >1 s timing gaps, which is unreliable on long songs.
    """
    if not words:
        return []

    if lyrics.strip():
        lyric_lines = [ln.strip() for ln in lyrics.splitlines() if ln.strip()]
        if lyric_lines:
            return _merge_by_lines(chords, words, lyric_lines)

    return _merge_by_gaps(chords, words)


# ── line-structure merge (preferred) ─────────────────────────────────────────

def _merge_by_lines(
    chords: list[ChordEvent],
    words: list[WordTiming],
    lyric_lines: list[str],
) -> list[LyricLine]:
    """Assign word timings to their original lyrics lines by word count.

    Words are consumed sequentially. If alignment returns fewer words than the
    lyrics contain (can happen when WhisperX skips tokens), remaining lines get
    empty timing and no chord markers.
    """
    result: list[LyricLine] = []
    cursor = 0

    for line_text in lyric_lines:
        expected = _word_count(line_text)
        line_words = words[cursor: cursor + expected]
        cursor += expected

        if line_words:
            result.append(_build_line(line_words, chords))
        else:
            # alignment ran out of words — emit the text with no markers
            result.append(LyricLine(
                words=line_text.split(),
                chord_markers=[],
            ))

    return result


def _word_count(text: str) -> int:
    """Count words the same way WhisperX tokenises them (split on whitespace,
    strip punctuation-only tokens)."""
    return len(text.split())


# ── gap-based merge (fallback) ────────────────────────────────────────────────

def _merge_by_gaps(
    chords: list[ChordEvent],
    words: list[WordTiming],
) -> list[LyricLine]:
    lines: list[LyricLine] = []
    current: list[WordTiming] = []

    for i, word in enumerate(words):
        current.append(word)
        is_last = i == len(words) - 1
        next_gap = (words[i + 1].start - word.end) if not is_last else float("inf")

        if is_last or next_gap > 1.0:
            lines.append(_build_line(current, chords))
            current = []

    return lines


# ── shared helpers ────────────────────────────────────────────────────────────

def _build_line(words: list[WordTiming], chords: list[ChordEvent]) -> LyricLine:
    if not words:
        return LyricLine(words=[], chord_markers=[])

    line_start = words[0].start
    line_end = words[-1].end

    seen_positions: set[int] = set()
    markers: list[dict] = []

    for chord in chords:
        if not ((line_start - PRE_ROLL_S) <= chord.start <= line_end):
            continue

        nearest_pos, nearest_dist = _nearest_word(words, chord.start)

        if nearest_pos not in seen_positions:
            seen_positions.add(nearest_pos)
            markers.append({"position": nearest_pos, "chord": chord.chord})

    markers.sort(key=lambda m: m["position"])
    return LyricLine(words=[w.word for w in words], chord_markers=markers)


def _nearest_word(words: list[WordTiming], t: float) -> tuple[int, float]:
    best_i, best_dist = 0, abs(words[0].start - t)
    for i, w in enumerate(words[1:], 1):
        d = abs(w.start - t)
        if d < best_dist:
            best_dist, best_i = d, i
    return best_i, best_dist
