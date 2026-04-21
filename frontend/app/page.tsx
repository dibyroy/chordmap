"use client";

import { useState } from "react";

export default function Home() {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [lyrics, setLyrics] = useState("");
  const [result, setResult] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);

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
    <main className="max-w-3xl mx-auto p-8">
      <h1 className="text-2xl font-bold mb-8">ChordMap</h1>

      <form onSubmit={handleSubmit} className="space-y-4 mb-8">
        <div>
          <label className="block text-sm mb-1">Audio file</label>
          <input
            type="file"
            accept="audio/*"
            onChange={(e) => setAudioFile(e.target.files?.[0] ?? null)}
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

      {result && (
        <pre className="bg-gray-900 p-4 rounded text-xs overflow-auto">
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </main>
  );
}
