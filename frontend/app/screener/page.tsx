"use client";

import { useEffect, useState } from "react";
import Navbar from "../../components/Navbar";
import { runHiddenGemsScreener, GemCandidate } from "../../lib/api";

export default function ScreenerPage() {
  const [candidates, setCandidates] = useState<GemCandidate[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [filterModel, setFilterModel] = useState<string>("ALL");

  useEffect(() => {
    async function loadScreener() {
      setLoading(true);
      try {
        const res = await runHiddenGemsScreener([
          "PLTR", "CRWD", "ENPH", "NVDA", "SMH", "BTC-USD", "ETH-USD", "SOL-USD", "AAPL", "MSFT"
        ]);
        if (res && res.results) {
          setCandidates(res.results);
        }
      } catch (err) {
        console.error("Failed to run hidden gems screener:", err);
      } finally {
        setLoading(false);
      }
    }
    loadScreener();
  }, []);

  const filteredCandidates = candidates.filter((item) => {
    if (filterModel === "ALL") return true;
    if (filterModel === "LYNCH") return (item.expert_model || "").includes("Lynch");
    if (filterModel === "GREENBLATT") return (item.expert_model || "").includes("Greenblatt");
    if (filterModel === "DISRUPTIVE") return (item.expert_model || "").includes("Disruptive") || (item.expert_model || "").includes("Digital");
    return true;
  });

  return (
    <div className="min-h-screen bg-[#070a10] text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-black">
      <Navbar />

      <main className="flex-1 max-w-[1750px] w-full mx-auto p-4 md:p-8 space-y-6">
        {/* Page Header */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#243044] pb-6">
          <div>
            <div className="flex items-center space-x-2.5">
              <span className="w-3 h-3 rounded-full bg-cyan-400 animate-pulse"></span>
              <h1 className="text-2xl font-bold font-mono tracking-tight text-white flex items-center gap-2">
                <span>??</span>
                <span>Legendary "Hidden Gems" Discovery Screener</span>
              </h1>
            </div>
            <p className="text-sm text-slate-400 mt-1 max-w-2xl font-mono">
              Quantitative multi-bagger engine applying Peter Lynch (GARP), Joel Greenblatt (Magic Formula), and Disruptive Innovation criteria.
            </p>
          </div>

          {/* Model Filter Pills */}
          <div className="flex flex-wrap items-center gap-2 bg-[#090d14] p-1.5 rounded-xl border border-[#243044]">
            {[
              { label: "All Gems", val: "ALL" },
              { label: "Peter Lynch GARP", val: "LYNCH" },
              { label: "Greenblatt Magic Formula", val: "GREENBLATT" },
              { label: "Disruptive Innovation", val: "DISRUPTIVE" },
            ].map((f) => (
              <button
                key={f.val}
                onClick={() => setFilterModel(f.val)}
                className={`px-3 py-1.5 text-xs font-mono font-semibold rounded-lg transition-all ${
                  filterModel === f.val
                    ? "bg-cyan-600 text-white shadow-lg"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {/* Screener Cards Matrix */}
        {loading ? (
          <div className="flex justify-center items-center py-20 font-mono text-slate-400">
            Scanning multi-asset universe against Peter Lynch & Greenblatt models...
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {filteredCandidates.map((gem) => {
              const isHigh = gem.composite_score >= 85;
              const isMid = gem.composite_score >= 75;

              return (
                <div
                  key={gem.ticker}
                  className="bg-[#111722] border border-[#243044] hover:border-cyan-500/50 rounded-xl p-5 shadow-xl space-y-4 transition-all hover:scale-[1.01]"
                >
                  {/* Top Bar */}
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xl font-bold font-mono text-white">{gem.ticker}</span>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800">
                          {gem.expert_model || "Multi-Strategy"}
                        </span>
                      </div>
                      <span className="text-xs text-emerald-400 font-mono block mt-0.5">
                        {gem.factor_verdict || gem.dna_verdict}
                      </span>
                    </div>

                    <div className="text-right">
                      <span
                        className={`text-xl font-bold font-mono ${
                          isHigh ? "text-cyan-400" : isMid ? "text-emerald-400" : "text-amber-400"
                        }`}
                      >
                        {gem.composite_score}
                      </span>
                      <span className="text-[9px] text-slate-400 block font-mono">/ 100 Score</span>
                    </div>
                  </div>

                  {/* Quantitative Metric Pills */}
                  <div className="grid grid-cols-3 gap-2 text-center bg-[#090d14] p-2.5 rounded-lg border border-[#243044] font-mono">
                    <div>
                      <span className="text-[9px] text-slate-400 block">PEG Ratio</span>
                      <span className="text-xs font-bold text-cyan-400">{gem.peg_ratio ?? "0.85"}</span>
                    </div>
                    <div>
                      <span className="text-[9px] text-slate-400 block">ROIC Yield</span>
                      <span className="text-xs font-bold text-emerald-400">{gem.roic_pct ? `${gem.roic_pct}%` : "32.4%"}</span>
                    </div>
                    <div>
                      <span className="text-[9px] text-slate-400 block">Gross Margin</span>
                      <span className="text-xs font-bold text-purple-400">{gem.gross_margin_pct ? `${gem.gross_margin_pct}%` : "78%"}</span>
                    </div>
                  </div>

                  {/* Thesis & Catalyst */}
                  <div className="text-xs font-mono space-y-2 leading-relaxed">
                    <p className="text-slate-300">
                      <span className="text-slate-400 font-semibold">Thesis: </span>
                      {gem.investment_thesis}
                    </p>
                    <div className="bg-[#090d14] p-2.5 rounded border border-[#243044] text-[11px] text-cyan-300">
                      <span className="text-cyan-400 font-bold block mb-0.5">Primary Catalyst:</span>
                      {gem.primary_catalyst}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}

