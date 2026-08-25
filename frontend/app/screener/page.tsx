"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { GemCandidate, runHiddenGemsScreener } from "../../lib/api";

const DEFAULT_WATCHLIST = [
  "NVDA", "PLTR", "CRWD", "ENPH", "AAPL", "MSFT", "GOOGL", "TSLA", "AMD", "COIN", "AVGO", "SMCI"
];

export default function ScreenerPage() {
  const [candidates, setCandidates] = useState<GemCandidate[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterRisk, setFilterRisk] = useState<string>("ALL");

  useEffect(() => {
    async function loadScreener() {
      setLoading(true);
      try {
        const data = await runHiddenGemsScreener(DEFAULT_WATCHLIST);
        setCandidates(data.results || []);
      } catch (err) {
        console.error("Screener failed:", err);
      } finally {
        setLoading(false);
      }
    }
    loadScreener();
  }, []);

  const filteredCandidates = candidates.filter((c) => {
    if (filterRisk === "ALL") return true;
    return c.risk_rating.toLowerCase().includes(filterRisk.toLowerCase());
  });

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#111722] border border-[#243044] rounded-xl p-6 shadow-xl">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
            <h1 className="text-2xl sm:text-3xl font-bold text-slate-100 tracking-tight">
              Quantitative Multi-Factor Screener
            </h1>
          </div>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Algorithmic discovery scanning across growth rates, Piotroski fundamental health, and institutional smart-money alignment
          </p>
        </div>

        {/* Risk Rating Filter */}
        <div className="flex items-center space-x-2 bg-[#090d14] p-1.5 rounded-lg border border-[#243044]">
          {["ALL", "LOW", "MODERATE"].map((risk) => (
            <button
              key={risk}
              onClick={() => setFilterRisk(risk)}
              className={`px-3 py-1.5 rounded text-xs font-mono font-medium transition-colors ${
                filterRisk === risk
                  ? "bg-cyan-500 text-slate-950 font-bold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {risk === "ALL" ? "All Risk Tiers" : `${risk} Risk`}
            </button>
          ))}
        </div>
      </div>

      {/* Screener Results Grid */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="bg-[#111722] border border-[#243044] rounded-xl p-5 h-56 animate-pulse" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredCandidates.map((gem) => (
            <div
              key={gem.ticker}
              className="bg-[#111722] border border-[#243044] hover:border-cyan-500/60 rounded-xl p-5 shadow-xl space-y-4 transition-all group flex flex-col justify-between"
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2.5">
                    <span className="text-lg font-bold font-mono text-slate-100 group-hover:text-cyan-400 transition-colors">
                      {gem.ticker}
                    </span>
                    <span className="text-[10px] font-mono bg-[#1b2434] text-slate-300 border border-[#364866] px-2 py-0.5 rounded">
                      {gem.factor_verdict || "Strong Differential"}
                    </span>
                  </div>

                  <div className="text-right">
                    <span className="text-xs font-mono font-bold text-cyan-400 bg-cyan-950/80 border border-cyan-800/80 px-2.5 py-1 rounded-md">
                      Score: {gem.composite_score}
                    </span>
                  </div>
                </div>

                <div className="text-xs text-slate-300 font-mono space-y-2 leading-relaxed">
                  <p>
                    <span className="text-slate-400 font-semibold">Thesis: </span>
                    {gem.investment_thesis}
                  </p>
                  <p>
                    <span className="text-purple-400 font-semibold">Primary Catalyst: </span>
                    {gem.primary_catalyst}
                  </p>
                </div>
              </div>

              <div className="pt-3 border-t border-[#1b2434] flex items-center justify-between">
                <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/40 border border-emerald-800/40 px-2 py-0.5 rounded">
                  {gem.risk_rating}
                </span>

                <Link
                  href={`/?symbol=${gem.ticker}`}
                  className="text-xs font-mono text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-semibold group-hover:translate-x-0.5 transition-transform"
                >
                  Load in Terminal ?
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

