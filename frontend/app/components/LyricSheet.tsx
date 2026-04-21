"use client";

interface ChordMarker {
  position: number;
  chord: string;
}

interface LyricLine {
  words: string[];
  chord_markers: ChordMarker[];
}

interface LyricSheetProps {
  lines: LyricLine[];
  currentTime: number;
}

// Returns the character offset of a word at `position` within the joined line,
// accounting for actual word lengths and single spaces between words.
function charOffset(words: string[], position: number): number {
  return words.slice(0, position).reduce((acc, w) => acc + w.length + 1, 0);
}

export function LyricSheet({ lines, currentTime }: LyricSheetProps) {
  return (
    <div className="font-mono text-sm leading-8 space-y-6">
      {lines.map((line, i) => (
        <div key={i} className="relative">
          <div className="text-indigo-400 text-xs h-5">
            {line.chord_markers.map((m, j) => (
              <span key={j} style={{ marginLeft: `${charOffset(line.words, m.position)}ch` }}>
                {m.chord}
              </span>
            ))}
          </div>
          <div className="text-gray-100">{line.words.join(" ")}</div>
        </div>
      ))}
    </div>
  );
}
