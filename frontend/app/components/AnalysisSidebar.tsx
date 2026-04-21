interface Analysis {
  key: string;
  mode: string;
  roman_numerals: string[];
  progression_patterns: string[];
  explanation: string;
}

interface AnalysisSidebarProps {
  analysis: Analysis;
}

export function AnalysisSidebar({ analysis }: AnalysisSidebarProps) {
  return (
    <aside className="bg-gray-900 rounded p-4 space-y-4 text-sm">
      <div>
        <span className="text-gray-400">Key</span>
        <p className="text-lg font-bold">{analysis.key} {analysis.mode}</p>
      </div>
      <div>
        <span className="text-gray-400 block mb-1">Roman numerals</span>
        <p className="font-mono text-indigo-300">{analysis.roman_numerals.join(" – ")}</p>
      </div>
      {analysis.progression_patterns.length > 0 && (
        <div>
          <span className="text-gray-400 block mb-1">Patterns</span>
          <ul className="list-disc list-inside space-y-1">
            {analysis.progression_patterns.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </div>
      )}
      <div>
        <span className="text-gray-400 block mb-1">Explanation</span>
        <p className="text-gray-200 leading-relaxed">{analysis.explanation}</p>
      </div>
    </aside>
  );
}
