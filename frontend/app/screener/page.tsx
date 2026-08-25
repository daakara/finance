"use client";

import { useState } from "react";
import { GemCandidate, runHiddenGemsScreener } from "../../lib/api";

export default function ScreenerPage() {
  const [loading, setLoading] = useState(false);
  const [tickerInput, setTickerInput] = useState("ENPH, SEDG, RUN, FSLR, PLTR, SNOW, CRWD");
  const [results, setResults] = useState<GemCandidate[]>([]);

  const handleRunScreener = async () => {
    setLoading(true);
    try {
      const candidateTickers = tickerInput
        .split(",")
        .map((t) => t.trim().toUpperCase())
        .filter((t) => t.length > 0);

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
      {/* Screener Header Card */}
      <div className="bg-[#111722] border border-[#243044] rounded-xl p-6 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-slate-100 tracking-tight">
              Hidden Gems Screener
            </h1>
            <p className="text-xs sm:text-sm text-slate-400 mt-1">
              Multi-factor screening engine evaluating asset momentum, volatility regimes, and fundamental thesis
            </p>
          </div>

          <button
            onClick={handleRunScreener}
            disabled={loading}
            className="bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-slate-950 font-bold px-6 py-2.5 rounded-lg text-sm transition-all focus-ring disabled:opacity-50 shadow-lg shadow-emerald-950/40 flex items-center justify-center space-x-2"
          >
            {loading ? (
              <>
                <span className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin"></span>
                <span>Screening Universe...</span>
              </>
            ) : (
              <span>Run Screener Engine</span>
            )}
          </button>
        </div>

        {/* Ticker Input Bar */}
        <div className="space-y-1.5 pt-2 border-t border-[#1b2434]">
          <label className="text-xs font-medium text-slate-300 block">Candidate Tickers Universe (comma-separated):</label>
          <input
            type="text"
            value={tickerInput}
            onChange={(e) => setTickerInput(e.target.value)}
            placeholder="ENPH, SEDG, RUN..."
            className="w-full bg-[#090d14] border border-[#243044] focus:border-emerald-500 text-slate-100 text-sm rounded-lg px-3.5 py-2 font-mono focus-ring"
          />
        </div>
      </div>

      {/* Results Section */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-[#111722] border border-[#243044] rounded-xl p-6 space-y-4 animate-pulse">
              <div className="h-6 bg-[#1b2434] rounded w-1/3"></div>
              <div className="h-4 bg-[#1b2434] rounded w-1/2"></div>
              <div className="h-16 bg-[#1b2434] rounded w-full"></div>
            </div>
          ))}
        </div>
      ) : results.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {results.map((gem) => (
            <div key={gem.ticker} className="bg-[#111722] border border-[#243044] hover:border-[#364866] rounded-xl p-6 space-y-4 transition-colors shadow-xl">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold font-mono text-slate-100">{gem.ticker}</span>
                <span className="bg-emerald-950/80 text-emerald-400 border border-emerald-800/80 text-xs px-3 py-1 rounded-full font-mono font-semibold">
                  Score: {gem.composite_score.toFixed(1)}/100
                </span>
              </div>

              <div>
                <span className="text-xs text-slate-400 block font-medium">Risk Rating</span>
                <span className="text-sm font-semibold text-slate-200">{gem.risk_rating}</span>
              </div>

              <div>
                <span className="text-xs text-slate-400 block font-medium">Investment Thesis</span>
                <p className="text-sm text-slate-300 leading-relaxed mt-1">{gem.investment_thesis}</p>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-[#111722] border border-[#243044] rounded-xl p-12 text-center space-y-3">
          <div className="w-12 h-12 rounded-full bg-[#1b2434] text-emerald-400 flex items-center justify-center mx-auto text-xl font-mono">
            ??
          </div>
          <h3 className="text-base font-semibold text-slate-200">No Active Screening Results</h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Click &quot;Run Screener Engine&quot; above to execute multi-factor quantitative screening across candidate assets.
          </p>
        </div>
      )}
    </div>
  );
}

