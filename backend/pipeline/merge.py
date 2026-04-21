"""Stage 4: Merge chords + word timings onto a shared timeline."""
from __future__ import annotations

from pydantic import BaseModel

from pipeline.chords import ChordEvent
from pipeline.align import WordTiming

# Snap a chord change to a word boundary if within this many milliseconds
SNAP_THRESHOLD_MS = 200


class LyricLine(BaseModel):
    words: list[str]
    # Each marker: {"position": word_index, "chord": "Am"}
    chord_markers: list[dict]


def merge(chords: list[ChordEvent], words: list[WordTiming]) -> list[LyricLine]:
    """Group words into lines and attach chord change markers.

    Lines break on pauses > 1 s between consecutive words.
    Chord changes snap to the nearest word start within SNAP_THRESHOLD_MS;
    changes further away are attached to the nearest word regardless.
    """
    if not words:
        return []

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


def _build_line(words: list[WordTiming], chords: list[ChordEvent]) -> LyricLine:
    if not words:
        return LyricLine(words=[], chord_markers=[])

    line_start = words[0].start
    line_end = words[-1].end
    snap_s = SNAP_THRESHOLD_MS / 1000.0

    seen_positions: set[int] = set()
    markers: list[dict] = []

    for chord in chords:
        # Only consider chord changes that start within this line's time span
        if not (line_start <= chord.start <= line_end):
            continue

        nearest_pos, nearest_dist = _nearest_word(words, chord.start)

        # Within threshold → snap; beyond threshold → still attach (best we can do)
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
