"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

interface DivergenceAsset {
  symbol: string;
  name: string;
  price: number;
  changePct: number;
  divergenceType: "STEALTH_ACCUMULATION" | "DISTRIBUTION_TRAP" | "CONVERGENT_INFLOW";
  insiderSentimentScore: number; // 0 to 100
  retailHypeScore: number; // 0 to 100
  divergenceDelta: number; // insider - retail
  signalDescription: string;
  keyInsider: string;
}

const DIVERGENCE_DATASET: DivergenceAsset[] = [
  {
    symbol: "PLTR",
    name: "Palantir Technologies",
    price: 31.20,
    changePct: 4.12,
    divergenceType: "STEALTH_ACCUMULATION",
    insiderSentimentScore: 94,
    retailHypeScore: 48,
    divergenceDelta: 46,
    signalDescription: "Heavy Congressional Intelligence & Armed Services committee buying during multi-week base consolidation.",
    keyInsider: "Rep. Dan Crenshaw & House Armed Services Committee",
  },
  {
    symbol: "CPRX",
    name: "Catalyst Pharmaceuticals",
    price: 23.40,
    changePct: 1.90,
    divergenceType: "STEALTH_ACCUMULATION",
    insiderSentimentScore: 91,
    retailHypeScore: 32,
    divergenceDelta: 59,
    signalDescription: "Zero retail social chatter with massive 9/9 Piotroski corporate insider accumulation.",
    keyInsider: "C-Suite 10b5-1 Buying Program & Form 4 Accumulation",
  },
  {
    symbol: "NVDA",
    name: "NVIDIA Corporation",
    price: 128.50,
    changePct: 3.14,
    divergenceType: "CONVERGENT_INFLOW",
    insiderSentimentScore: 88,
    retailHypeScore: 85,
    divergenceDelta: 3,
    signalDescription: "Bipartisan Congressional buying aligned with high institutional 13F hedge fund sponsorship.",
    keyInsider: "Rep. Michael McCaul & Rep. Nancy Pelosi",
  },
  {
    symbol: "TSLA",
    name: "Tesla Inc.",
    price: 218.40,
    changePct: 2.15,
    divergenceType: "DISTRIBUTION_TRAP",
    insiderSentimentScore: 42,
    retailHypeScore: 92,
    divergenceDelta: -50,
    signalDescription: "High retail call option volume contrasting with executive Form 4 option exercise distributions.",
    keyInsider: "Executive 10b5-1 Scheduled Share Dispositions",
  },
  {
    symbol: "NVO",
    name: "Novo Nordisk A/S",
    price: 136.40,
    changePct: 1.85,
    divergenceType: "STEALTH_ACCUMULATION",
    insiderSentimentScore: 92,
    retailHypeScore: 54,
    divergenceDelta: 38,
    signalDescription: "Healthcare committee buying amid quiet consolidation below 50-day moving average.",
    keyInsider: "Foreign Affairs & Healthcare Sub-Committee Disclosures",
  },
  {
    symbol: "SMCI",
    name: "Super Micro Computer",
    price: 48.20,
    changePct: 5.40,
    divergenceType: "DISTRIBUTION_TRAP",
    insiderSentimentScore: 38,
    retailHypeScore: 88,
    divergenceDelta: -50,
    signalDescription: "Extreme retail short squeeze speculation contrasting with institutional block volume trimming.",
    keyInsider: "Institutional Block Liquidations & Auditor Review Notice",
  },
];

export default function SmartMoneyDivergenceRadar() {
  const [filter, setFilter] = useState<"ALL" | "STEALTH_ACCUMULATION" | "DISTRIBUTION_TRAP">("ALL");
  const [vernacularMode, setVernacularMode] = useState<"PLAIN_ENGLISH" | "PRO_QUANT">("PLAIN_ENGLISH");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("ARX_VERNACULAR_MODE") as "PLAIN_ENGLISH" | "PRO_QUANT" | null;
      if (saved) setVernacularMode(saved);
    } catch {}

    const handleVernacular = (e: Event) => {
      const custom = e as CustomEvent<"PLAIN_ENGLISH" | "PRO_QUANT">;
      if (custom.detail) setVernacularMode(custom.detail);
    };
    window.addEventListener("finance:vernacular-change", handleVernacular);
    return () => window.removeEventListener("finance:vernacular-change", handleVernacular);
  }, []);

  const isPlain = vernacularMode === "PLAIN_ENGLISH";

  const displayedAssets = DIVERGENCE_DATASET.filter((a) => {
    if (filter === "ALL") return true;
    return a.divergenceType === filter;
  });

  return (
    <div className="bg-[#0e1420] border border-[#243044] rounded-2xl p-5 sm:p-6 shadow-xl space-y-4 font-sans text-slate-200">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1e293b] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-xl">📡</span>
            <h2 className="text-base sm:text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <span>{isPlain ? "Smart Money vs. Public Hype Radar" : "Institutional Inflow vs. Retail Divergence Radar"}</span>
            </h2>
            <span className="text-[10px] bg-cyan-950 text-cyan-400 border border-cyan-800 px-2 py-0.5 rounded font-mono font-bold">
              STOCK ACT + 13F MATRIX
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            {isPlain
              ? "Spot when politicians & insiders are quietly buying low while the public is looking away, or selling high while public hype is extreme."
              : "Detects order-flow asymmetry between Congressional/Institutional Form 4 filings and retail sentiment."}
          </p>
        </div>

        {/* Filter Buttons */}
        <div className="flex flex-wrap items-center gap-1.5 bg-[#090d14] p-1 rounded-xl border border-[#1e293b] text-xs font-bold">
          <button
            type="button"
            onClick={() => setFilter("ALL")}
            className={`px-3 py-1 rounded-lg transition ${
              filter === "ALL" ? "bg-cyan-500 text-slate-950 shadow" : "text-slate-400 hover:text-white"
            }`}
          >
            {isPlain ? "All Signals" : "All Divergences"}
          </button>
          <button
            type="button"
            onClick={() => setFilter("STEALTH_ACCUMULATION")}
            className={`px-3 py-1 rounded-lg transition ${
              filter === "STEALTH_ACCUMULATION" ? "bg-emerald-500 text-slate-950 shadow font-extrabold" : "text-slate-400 hover:text-emerald-400"
            }`}
          >
            {isPlain ? "🟢 Quiet Insider Buying" : "🟢 Stealth Accumulation"}
          </button>
          <button
            type="button"
            onClick={() => setFilter("DISTRIBUTION_TRAP")}
            className={`px-3 py-1 rounded-lg transition ${
              filter === "DISTRIBUTION_TRAP" ? "bg-rose-500 text-slate-950 shadow font-extrabold" : "text-slate-400 hover:text-rose-400"
            }`}
          >
            {isPlain ? "🛑 Hype Traps (Selling)" : "🛑 Distribution Traps"}
          </button>
        </div>
      </div>

      {/* Divergence Asset Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5 pt-1">
        {displayedAssets.map((asset) => {
          const isStealth = asset.divergenceType === "STEALTH_ACCUMULATION";
          const isTrap = asset.divergenceType === "DISTRIBUTION_TRAP";

          return (
            <Link
              key={asset.symbol}
              href={`/?symbol=${asset.symbol}`}
              aria-label={`Analyze ${asset.symbol} (${asset.name}): ${isStealth ? (isPlain ? "Quiet Accumulation" : "Stealth Accumulation") : isTrap ? (isPlain ? "Hype Trap" : "Distribution Trap") : (isPlain ? "Balanced Inflow" : "Convergent Inflow")}`}
              className={`p-4 rounded-xl border transition-all duration-150 active:scale-[0.98] active:bg-[#0e1522] hover:border-cyan-500/80 bg-[#111722] space-y-3 block group cursor-pointer ${
                isStealth
                  ? "border-emerald-800/60 shadow-[0_0_12px_rgba(16,185,129,0.08)] hover:shadow-[0_0_16px_rgba(16,185,129,0.2)]"
                  : isTrap
                  ? "border-rose-800/60 shadow-[0_0_12px_rgba(244,63,94,0.08)] hover:shadow-[0_0_16px_rgba(244,63,94,0.2)]"
                  : "border-[#243044] hover:shadow-[0_0_16px_rgba(6,182,212,0.15)]"
              }`}
            >
              {/* Asset Header */}
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <strong className="text-base font-black text-white font-mono group-hover:text-cyan-400 transition-colors">{asset.symbol}</strong>
                    <span className="text-xs text-slate-400 truncate max-w-[120px]">{asset.name}</span>
                  </div>
                  <div className="text-xs font-mono font-bold text-slate-300">
                    ${asset.price.toFixed(2)}{" "}
                    <span className={asset.changePct >= 0 ? "text-emerald-400" : "text-rose-400"}>
                      ({asset.changePct >= 0 ? "+" : ""}{asset.changePct}%)
                    </span>
                  </div>
                </div>

                <span
                  className={`text-[10px] font-mono font-extrabold px-2.5 py-1 rounded-md border shrink-0 ${
                    isStealth
                      ? "bg-emerald-950/80 border-emerald-700 text-emerald-300"
                      : isTrap
                      ? "bg-rose-950/80 border-rose-700 text-rose-300"
                      : "bg-cyan-950/80 border-cyan-700 text-cyan-300"
                  }`}
                >
                  {isStealth
                    ? (isPlain ? "🟢 QUIET ACCUMULATION" : "🟢 STEALTH ACCUMULATION")
                    : isTrap
                    ? (isPlain ? "🛑 HYPE TRAP WARNING" : "🛑 DISTRIBUTION TRAP")
                    : (isPlain ? "⚖️ BALANCED INFLOW" : "⚖️ CONVERGENT INFLOW")}
                </span>
              </div>

              {/* Insider Flow vs Retail Sentiment Visualizer */}
              <div className="space-y-1.5 text-xs font-mono">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-cyan-300 font-bold">{isPlain ? "Smart Money Index:" : "Institutional Inflow:"}</span>
                  <span className="font-extrabold text-white">{asset.insiderSentimentScore} / 100</span>
                </div>
                <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400 rounded-full transition-all duration-500"
                    style={{ width: `${asset.insiderSentimentScore}%` }}
                  />
                </div>

                <div className="flex items-center justify-between text-[11px] pt-1">
                  <span className="text-amber-300 font-bold">{isPlain ? "Public Social Hype:" : "Retail Social Sentiment:"}</span>
                  <span className="font-extrabold text-white">{asset.retailHypeScore} / 100</span>
                </div>
                <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-amber-500 to-rose-400 rounded-full transition-all duration-500"
                    style={{ width: `${asset.retailHypeScore}%` }}
                  />
                </div>
              </div>

              {/* Description & Signal */}
              <p className="text-[11px] text-slate-300 leading-relaxed font-sans bg-[#090d14] p-2.5 rounded-lg border border-[#1e293b] group-hover:border-cyan-900/60 transition-colors">
                {asset.signalDescription}
              </p>

              {/* Footer CTA */}
              <div className="flex items-center justify-between text-[11px] pt-1 border-t border-[#1e293b]">
                <span className="text-[10px] text-slate-400 truncate max-w-[150px] sm:max-w-[170px]" title={asset.keyInsider}>
                  🏛️ {asset.keyInsider}
                </span>
                <span className="px-2.5 py-1 rounded-md text-[10px] font-bold font-mono border bg-cyan-500/10 border-cyan-500/40 text-cyan-300 group-hover:bg-cyan-500 group-hover:text-slate-950 group-hover:border-cyan-400 transition-colors flex items-center gap-1 shrink-0">
                  Analyze <span className="group-hover:translate-x-0.5 transition-transform">➔</span>
                </span>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
