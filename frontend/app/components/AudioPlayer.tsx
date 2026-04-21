"use client";

import { useRef, useState, useEffect } from "react";

interface AudioPlayerProps {
  src: string;
  onTimeUpdate?: (currentTime: number) => void;
}

export function AudioPlayer({ src, onTimeUpdate }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const rafRef = useRef<number>(0);

  function tick() {
    if (audioRef.current) {
      const t = audioRef.current.currentTime;
      setCurrentTime(t);
      onTimeUpdate?.(t);
    }
    rafRef.current = requestAnimationFrame(tick);
  }

  useEffect(() => {
    if (playing) {
      rafRef.current = requestAnimationFrame(tick);
    } else {
      cancelAnimationFrame(rafRef.current);
    }
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing]);

  function togglePlay() {
    if (!audioRef.current) return;
    if (playing) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setPlaying(!playing);
  }

  function seek(e: React.ChangeEvent<HTMLInputElement>) {
    if (!audioRef.current) return;
    audioRef.current.currentTime = Number(e.target.value);
  }

  return (
    <div className="flex items-center gap-4 bg-gray-900 rounded p-3">
      <audio
        ref={audioRef}
        src={src}
        onLoadedMetadata={(e) => setDuration((e.target as HTMLAudioElement).duration)}
        onEnded={() => setPlaying(false)}
      />
      <button onClick={togglePlay} className="text-sm px-3 py-1 bg-indigo-600 rounded">
        {playing ? "Pause" : "Play"}
      </button>
      <input
        type="range"
        min={0}
        max={duration}
        step={0.01}
        value={currentTime}
        onChange={seek}
        className="flex-1"
      />
      <span className="text-xs text-gray-400">
        {currentTime.toFixed(1)}s / {duration.toFixed(1)}s
      </span>
    </div>
  );
}
