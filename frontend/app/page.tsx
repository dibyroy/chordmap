"use client";

import { useState } from "react";
import { LyricSheet } from "./components/LyricSheet";
import { AudioPlayer } from "./components/AudioPlayer";
import { AnalysisSidebar } from "./components/AnalysisSidebar";
import { Diagnostics } from "./components/Diagnostics";

interface ChordEvent {
  start: number;
  end: number;
  chord: string;
  confidence: number;
}

interface SongResult {
  chords: ChordEvent[];
  words: { word: string; start: number; end: number }[];
  lines: { words: string[]; chord_markers: { position: number; chord: string }[] }[];
  analysis: {
    key: string;
    mode: string;
    roman_numerals: string[];
    progression_patterns: string[];
    explanation: string;
  };
}

export default function Home() {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [lyrics, setLyrics] = useState("");
  const [result, setResult] = useState<SongResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  function handleAudioChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0] ?? null;
    setAudioFile(file);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(file ? URL.createObjectURL(file) : null);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!audioFile || !lyrics) return;

    const form = new FormData();
    form.append("audio", audioFile);
    form.append("lyrics", lyrics);

    setLoading(true);
    try {
      const res = await fetch("/api/analyze", { method: "POST", body: form });
      setResult(await res.json());
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="max-w-5xl mx-auto p-8">
      <h1 className="text-2xl font-bold mb-8">ChordMap</h1>

      <form onSubmit={handleSubmit} className="space-y-4 mb-8">
        <div>
          <label className="block text-sm mb-1">Audio file</label>
          <input
            type="file"
            accept="audio/*"
            onChange={handleAudioChange}
            className="block w-full text-sm"
          />
        </div>
        <div>
          <label className="block text-sm mb-1">Lyrics</label>
          <textarea
            value={lyrics}
            onChange={(e) => setLyrics(e.target.value)}
            rows={10}
            placeholder="Paste lyrics here..."
            className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-sm"
          />
        </div>
        <button
          type="submit"
          disabled={loading || !audioFile || !lyrics}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 px-4 py-2 rounded text-sm"
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </form>

      {audioUrl && (
        <div className="mb-6">
          <AudioPlayer src={audioUrl} onTimeUpdate={setCurrentTime} />
        </div>
      )}

      {result && (
        <div className="space-y-6">
          <div className="flex gap-6 items-start">
            <div className="flex-1 min-w-0 bg-gray-900 rounded p-4">
              <LyricSheet lines={result.lines} currentTime={currentTime} />
            </div>
            <div className="w-72 shrink-0">
              <AnalysisSidebar analysis={result.analysis} />
            </div>
          </div>
          <Diagnostics chords={result.chords} />
        </div>
      )}
    </main>
  );
}
