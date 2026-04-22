"""Stage 4: Merge chords + word timings onto a shared timeline."""
from __future__ import annotations

from pydantic import BaseModel

from pipeline.chords import ChordEvent
from pipeline.align import WordTiming

SNAP_THRESHOLD_MS = 200
MIN_INSTRUMENTAL_GAP = 2.0  # seconds — gaps shorter than this are skipped


class LyricLine(BaseModel):
    words: list[str]
    chord_markers: list[dict]
    start: float = 0.0
    end: float = 0.0
    is_instrumental: bool = False


def merge(
    chords: list[ChordEvent],
    words: list[WordTiming],
    lyrics: str = "",
) -> list[LyricLine]:
    """Group words into lines, attach chord markers, and insert Instrumental
    sections for any gap > 2s that contains chord changes."""
    if not words:
        return _all_instrumental(chords)

    if lyrics.strip():
        lyric_lines = [ln.strip() for ln in lyrics.splitlines() if ln.strip()]
        if lyric_lines:
            lines = _merge_by_lines(chords, words, lyric_lines)
        else:
            lines = _merge_by_gaps(chords, words)
    else:
        lines = _merge_by_gaps(chords, words)

    return _insert_instrumental(lines, chords)


# ── line-structure merge (preferred) ─────────────────────────────────────────

def _merge_by_lines(
    chords: list[ChordEvent],
    words: list[WordTiming],
    lyric_lines: list[str],
) -> list[LyricLine]:
    result: list[LyricLine] = []
    cursor = 0

    for line_text in lyric_lines:
        expected = len(line_text.split())
        line_words = words[cursor: cursor + expected]
        cursor += expected

        if line_words:
            result.append(_build_line(line_words, chords))
        else:
            result.append(LyricLine(words=line_text.split(), chord_markers=[]))

    return result


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


# ── instrumental section insertion ───────────────────────────────────────────

def _insert_instrumental(
    lines: list[LyricLine],
    chords: list[ChordEvent],
) -> list[LyricLine]:
    """Find gaps in the lyric timeline and insert Instrumental lines where
    chord changes occur."""
    if not chords or not lines:
        return lines

    timed = [l for l in lines if l.start or l.end]
    if not timed:
        return lines

    song_end = max(c.end for c in chords)

    # Collect gap windows: (gap_start, gap_end)
    gaps: list[tuple[float, float]] = []

    # Intro gap
    if timed[0].start > MIN_INSTRUMENTAL_GAP:
        gaps.append((0.0, timed[0].start))

    # Gaps between consecutive lines
    for i in range(len(timed) - 1):
        gap_start = timed[i].end
        gap_end = timed[i + 1].start
        if gap_end - gap_start > MIN_INSTRUMENTAL_GAP:
            gaps.append((gap_start, gap_end))

    # Outro gap
    if song_end - timed[-1].end > MIN_INSTRUMENTAL_GAP:
        gaps.append((timed[-1].end, song_end))

    instrumental: list[LyricLine] = []
    for gap_start, gap_end in gaps:
        gap_chords = [c for c in chords if gap_start <= c.start < gap_end]
        if not gap_chords:
            continue

        # Deduplicate consecutive identical chords
        unique: list[ChordEvent] = [gap_chords[0]]
        for c in gap_chords[1:]:
            if c.chord != unique[-1].chord:
                unique.append(c)

        markers = [{"position": i, "chord": c.chord} for i, c in enumerate(unique)]
        instrumental.append(LyricLine(
            words=[],
            chord_markers=markers,
            start=gap_start,
            end=gap_end,
            is_instrumental=True,
        ))

    if not instrumental:
        return lines

    combined = lines + instrumental
    combined.sort(key=lambda l: l.start)
    return combined


def _all_instrumental(chords: list[ChordEvent]) -> list[LyricLine]:
    """Whole song is instrumental — no vocals at all."""
    if not chords:
        return []
    unique: list[ChordEvent] = [chords[0]]
    for c in chords[1:]:
        if c.chord != unique[-1].chord:
            unique.append(c)
    markers = [{"position": i, "chord": c.chord} for i, c in enumerate(unique)]
    return [LyricLine(
        words=[],
        chord_markers=markers,
        start=chords[0].start,
        end=chords[-1].end,
        is_instrumental=True,
    )]


# ── shared helpers ────────────────────────────────────────────────────────────

def _build_line(words: list[WordTiming], chords: list[ChordEvent]) -> LyricLine:
    if not words:
        return LyricLine(words=[], chord_markers=[])

    line_start = words[0].start
    line_end = words[-1].end

    seen_positions: set[int] = set()
    markers: list[dict] = []

    for chord in chords:
        if not (line_start <= chord.start <= line_end):
            continue
        nearest_pos, _ = _nearest_word(words, chord.start)
        if nearest_pos not in seen_positions:
            seen_positions.add(nearest_pos)
            markers.append({"position": nearest_pos, "chord": chord.chord})

    markers.sort(key=lambda m: m["position"])
    return LyricLine(
        words=[w.word for w in words],
        chord_markers=markers,
        start=line_start,
        end=line_end,
    )


def _nearest_word(words: list[WordTiming], t: float) -> tuple[int, float]:
    best_i, best_dist = 0, abs(words[0].start - t)
    for i, w in enumerate(words[1:], 1):
        d = abs(w.start - t)
        if d < best_dist:
            best_dist, best_i = d, i
    return best_i, best_dist
