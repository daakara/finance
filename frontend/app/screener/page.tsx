"use client";

import { useState } from "react";
import { GemCandidate, runHiddenGemsScreener } from "@/lib/api";

export default function ScreenerPage() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<GemCandidate[]>([]);

  const handleRunScreener = async () => {
    setLoading(true);
    try {
      const candidateTickers = ["ENPH", "SEDG", "RUN", "FSLR", "PLTR", "SNOW", "CRWD"];
      const response = await runHiddenGemsScreener(candidateTickers);
      setResults(response.results);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Hidden Gems Screener</h1>
          <p className="text-sm text-gray-400">
            Multi-factor screening system identifying high-upside under-the-radar assets
          </p>
        </div>

        <button
          onClick={handleRunScreener}
          disabled={loading}
          className="bg-emerald-600 hover:bg-emerald-500 text-white px-6 py-2.5 rounded-lg font-semibold shadow-lg transition-colors disabled:opacity-50"
        >
          {loading ? "Screening Universe..." : "Run Screener"}
        </button>
      </div>

      {results.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {results.map((gem) => (
            <div key={gem.ticker} className="bg-[#161b22] border border-[#30363d] rounded-lg p-6 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold text-white">{gem.ticker}</span>
                <span className="bg-emerald-950 text-emerald-400 border border-emerald-800 text-xs px-3 py-1 rounded-full font-mono">
                  Score: {gem.composite_score.toFixed(1)}/100
                </span>
              </div>

              <div>
                <span className="text-xs text-gray-400 block">Risk Rating</span>
                <span className="text-sm font-medium text-white">{gem.risk_rating}</span>
              </div>

              <div>
                <span className="text-xs text-gray-400 block">Investment Thesis</span>
                <p className="text-sm text-gray-300 line-clamp-3">{gem.investment_thesis}</p>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-12 text-center text-gray-400">
          Click &quot;Run Screener&quot; to execute multi-factor screening across candidate tickers.
        </div>
      )}
    </div>
  );
}

