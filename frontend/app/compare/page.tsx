"use client";

import { useState } from "react";
import Link from "next/link";
import Navbar from "../../components/Navbar";

interface CompetitorAsset {
  symbol: string;
  name: string;
  category: string;
  marketCap: string;
  peRatio: string;
  pegRatio: string;
  roic: string;
  grossMargin: string;
  piotroski: number;
  keyCatalyst: string;
  trialEfficacy: string;
  primaryRisk: string;
  verdict: string;
}

const COMPARISON_PAIRS: Record<string, [CompetitorAsset, CompetitorAsset]> = {
  "nvo-vs-lly": [
    {
      symbol: "NVO",
      name: "Novo Nordisk A/S",
      category: "Global Diabetes & Obesity Leader",
      marketCap: "$615 Billion",
      peRatio: "38.2x",
      pegRatio: "1.25",
      roic: "62.4%",
      grossMargin: "84.5%",
      piotroski: 9,
      keyCatalyst: "Amycretin Phase 2/3 (Oral GLP-1/Amylin Pill with 13.1% 12-wk weight loss)",
      trialEfficacy: "High convenience oral pill bypassing cold-chain supply chain bottlenecks",
      primaryRisk: "Capacity manufacturing constraints & compounded semaglutide litigation",
      verdict: "Elite ROIC & Margin Moat; Massive scaling if oral formulation succeeds",
    },
    {
      symbol: "LLY",
      name: "Eli Lilly & Company",
      category: "Global Oncology & Incretin Leader",
      marketCap: "$870 Billion",
      peRatio: "62.5x",
      pegRatio: "1.45",
      roic: "34.8%",
      grossMargin: "79.2%",
      piotroski: 8,
      keyCatalyst: "Retatrutide Phase 3 (Triple GGG Agonist with 24.2% 48-wk weight loss)",
      trialEfficacy: "Industry-leading clinical absolute weight reduction and liver fat clearance",
      primaryRisk: "Elevated valuation multiple (62x P/E) requiring flawless execution",
      verdict: "Clinical efficacy champion; Leading highest-tier weight reduction trials",
    },
  ],
  "spy-vs-qqq": [
    {
      symbol: "SPY",
      name: "SPDR S&P 500 ETF Trust",
      category: "Broad Market Benchmark",
      marketCap: "$580 Billion AUM",
      peRatio: "26.4x",
      pegRatio: "1.80",
      roic: "18.5%",
      grossMargin: "N/A (Index)",
      piotroski: 8,
      keyCatalyst: "US Economic Soft Landing & Federal Reserve Interest Rate Easing Cycles",
      trialEfficacy: "Broad diversification across 500 market leaders",
      primaryRisk: "Macro recession or systemic credit spread blowouts",
      verdict: "Core wealth compounding foundation with maximum market diversification",
    },
    {
      symbol: "QQQ",
      name: "Invesco QQQ Trust (Nasdaq-100)",
      category: "High-Growth Tech Benchmark",
      marketCap: "$290 Billion AUM",
      peRatio: "31.2x",
      pegRatio: "1.35",
      roic: "28.2%",
      grossMargin: "N/A (Index)",
      piotroski: 9,
      keyCatalyst: "Enterprise Generative AI Monetization & Hyperscaler Capex Expansion",
      trialEfficacy: "Concentrated secular tech dominance (Apple, Microsoft, Nvidia)",
      primaryRisk: "Multiple contraction during sudden interest rate surges",
      verdict: "High-beta growth engine capturing secular technology expansion",
    },
  ],
};

export default function ComparePage() {
  const [selectedPair, setSelectedPair] = useState<string>("nvo-vs-lly");
  const currentPair = COMPARISON_PAIRS[selectedPair];

  return (
    <main id="main-content" role="main" className="min-h-screen bg-[#070a11] text-slate-100 font-mono flex flex-col pb-20 sm:pb-8">
      <Navbar />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 w-full flex-1">
        {/* Header */}
        <div className="mb-6 border-b border-[#1b2434] pb-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-2xl">⚔️</span>
                <h1 className="text-xl sm:text-2xl font-black text-white tracking-tight">
                  Head-to-Head Asset & Pipeline Comparison
                </h1>
              </div>
              <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-3xl">
                Side-by-side institutional analysis comparing valuation, drug trials, operational efficiency, and future earnings potential.
              </p>
            </div>
            <Link
              href="/"
              className="px-3 py-1.5 bg-[#111722] hover:bg-[#182232] border border-[#243044] rounded-lg text-xs font-bold text-cyan-400 transition-colors flex items-center space-x-1 active:scale-[0.96]"
            >
              <span>← Return to Terminal</span>
            </Link>
          </div>
        </div>

        {/* Comparison Selector Tabs */}
        <div className="flex items-center space-x-2 mb-6">
          <button
            onClick={() => setSelectedPair("nvo-vs-lly")}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition-all border active:scale-[0.96] ${
              selectedPair === "nvo-vs-lly"
                ? "bg-cyan-500 text-slate-950 border-cyan-400 font-extrabold shadow-lg"
                : "bg-[#0d131f] text-slate-400 border-[#243044] hover:text-slate-200"
            }`}
          >
            💊 Novo Nordisk (NVO) vs. Eli Lilly (LLY)
          </button>
          <button
            onClick={() => setSelectedPair("spy-vs-qqq")}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition-all border active:scale-[0.96] ${
              selectedPair === "spy-vs-qqq"
                ? "bg-cyan-500 text-slate-950 border-cyan-400 font-extrabold shadow-lg"
                : "bg-[#0d131f] text-slate-400 border-[#243044] hover:text-slate-200"
            }`}
          >
            📊 S&P 500 (SPY) vs. Nasdaq-100 (QQQ)
          </button>
        </div>

        {/* Side-by-Side Comparison Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {currentPair.map((asset, idx) => (
            <div key={asset.symbol} className="bg-[#0e131d] border border-[#1b2434] rounded-2xl p-5 shadow-2xl space-y-4 flex flex-col justify-between">
              <div>
                {/* Header */}
                <div className="flex items-start justify-between border-b border-[#1b2434] pb-4">
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xl font-black text-white">{asset.symbol}</span>
                      <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-cyan-950 border border-cyan-800 text-cyan-400">
                        {asset.category}
                      </span>
                    </div>
                    <h2 className="text-sm text-slate-300 font-bold mt-1">{asset.name}</h2>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-slate-500 block">PIOTROSKI</span>
                    <span className="text-base font-black text-emerald-400">{asset.piotroski}/9</span>
                  </div>
                </div>

                {/* Metrics Table */}
                <div className="grid grid-cols-3 gap-2 my-4 bg-[#080c14] p-3 rounded-xl border border-[#182232] text-center text-xs tabular-nums">
                  <div>
                    <span className="text-[10px] text-slate-500 block">MARKET CAP</span>
                    <span className="font-bold text-slate-200">{asset.marketCap}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">P/E RATIO</span>
                    <span className="font-bold text-purple-300">{asset.peRatio}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">ROIC</span>
                    <span className="font-bold text-emerald-400">{asset.roic}</span>
                  </div>
                </div>

                {/* Pipeline & Catalyst Highlight */}
                <div className="space-y-3 text-xs">
                  <div className="p-3 bg-[#111722] rounded-xl border border-cyan-950">
                    <span className="text-[10px] text-cyan-400 font-black block uppercase tracking-wider">🔬 Primary Clinical Trial / Catalyst</span>
                    <p className="text-slate-200 mt-1 font-semibold text-[11px] leading-snug">{asset.keyCatalyst}</p>
                    <p className="text-slate-400 text-[10px] mt-1">{asset.trialEfficacy}</p>
                  </div>

                  <div>
                    <span className="text-[10px] text-slate-500 font-black block uppercase tracking-wider">⚠️ Key Operational Risk</span>
                    <p className="text-slate-400 text-[11px] mt-0.5">{asset.primaryRisk}</p>
                  </div>

                  <div>
                    <span className="text-[10px] text-emerald-500 font-black block uppercase tracking-wider">💡 Strategic Verdict</span>
                    <p className="text-slate-300 text-[11px] mt-0.5">{asset.verdict}</p>
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <div className="pt-4 border-t border-[#1b2434]">
                <Link
                  href={`/?symbol=${asset.symbol}`}
                  className="w-full py-2 bg-cyan-600/20 hover:bg-cyan-500 hover:text-slate-950 border border-cyan-500/50 rounded-xl text-xs font-bold text-cyan-300 transition-all flex items-center justify-center space-x-1 active:scale-[0.98]"
                >
                  <span>Load {asset.symbol} into Live Terminal →</span>
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
