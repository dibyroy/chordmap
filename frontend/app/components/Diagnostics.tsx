"use client";

import { useState } from "react";

interface ChordEvent {
  start: number;
  end: number;
  chord: string;
  confidence: number;
}

interface DiagnosticsProps {
  chords: ChordEvent[];
}

export function Diagnostics({ chords }: DiagnosticsProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="border border-gray-700 rounded text-sm">
      <button
        onClick={() => setOpen(!open)}
        className="w-full text-left px-4 py-2 text-gray-400 hover:text-gray-200"
      >
        {open ? "▾" : "▸"} Diagnostics
      </button>
      {open && (
        <div className="p-4 border-t border-gray-700 overflow-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-gray-400">
                <th className="text-left pr-4">Start</th>
                <th className="text-left pr-4">End</th>
                <th className="text-left pr-4">Chord</th>
                <th className="text-left">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {chords.map((c, i) => (
                <tr key={i} className="border-t border-gray-800">
                  <td className="pr-4">{c.start.toFixed(2)}s</td>
                  <td className="pr-4">{c.end.toFixed(2)}s</td>
                  <td className="pr-4 text-indigo-300">{c.chord}</td>
                  <td>
                    <span
                      className="inline-block h-2 bg-indigo-500 rounded"
                      style={{ width: `${c.confidence * 100}%`, maxWidth: "80px" }}
                    />
                    {" "}{(c.confidence * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
