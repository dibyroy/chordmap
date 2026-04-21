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

export function LyricSheet({ lines, currentTime }: LyricSheetProps) {
  return (
    <div className="font-mono text-sm leading-8 space-y-6">
      {lines.map((line, i) => (
        <div key={i} className="relative">
          <div className="text-indigo-400 text-xs h-5">
            {line.chord_markers.map((m, j) => (
              <span key={j} style={{ marginLeft: `${m.position * 0.6}ch` }}>
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
