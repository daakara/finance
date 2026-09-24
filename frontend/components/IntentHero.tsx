"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

interface IntentHeroProps {
  onSelectSymbol?: (sym: string) => void;
}

export default function IntentHero({ onSelectSymbol }: IntentHeroProps) {
  const router = useRouter();
  const [quickSymbol, setQuickSymbol] = useState("");
  const [isSearchOpen, setIsSearchOpen] = useState(false);

  const handleQuickSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const clean = quickSymbol.trim().toUpperCase();
    if (!clean) return;
    if (onSelectSymbol) {
      onSelectSymbol(clean);
    } else {
      router.push(`/?symbol=${clean}`);
    }
    setQuickSymbol("");
    setIsSearchOpen(false);
  };

  return (
    <div className="bg-[#0b101b] border border-[#1e2a3c] rounded-2xl p-4 sm:p-5 shadow-2xl space-y-4 font-sans text-slate-100">
      <div className="text-center max-w-2xl mx-auto space-y-1">
        <span className="text-[10px] font-mono uppercase tracking-widest text-cyan-400 font-bold">
          Your Market Workflow
        </span>
        <h2 className="text-xl sm:text-2xl font-black text-white tracking-tight">
          Four steps from opportunity to conviction
        </h2>
        <p className="text-xs text-slate-400 font-sans">
          Find → Analyze → Decide → Manage. Each step builds on the last.
        </p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {/* Step 1: Find opportunities */}
        <Link
          href="/radar"
          className="bg-[#070b13] hover:bg-[#0e1624] border border-[#1b2537] hover:border-cyan-500/60 rounded-xl p-3 sm:p-4 transition-all duration-200 group shadow-lg flex flex-col justify-between"
        >
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-lg sm:text-2xl">🔎</span>
              <span className="text-[9px] sm:text-[10px] font-mono font-bold text-cyan-400 bg-cyan-950/50 px-1.5 py-0.5 rounded">
                Step 1
              </span>
            </div>
            <div>
              <h3 className="text-xs sm:text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">
                Find opportunities
              </h3>
              <p className="text-[10px] sm:text-xs text-slate-400 leading-relaxed pt-0.5">
                Scan setups, compounders, and value candidates
              </p>
            </div>
          </div>
          <span className="text-[9px] sm:text-[10px] font-mono text-slate-500 mt-2 sm:mt-3 block border-t border-[#131d2b] pt-1.5 sm:pt-2">
            Radar
          </span>
        </Link>

        {/* Step 2: Analyze a stock */}
        <div
          onClick={() => setIsSearchOpen(true)}
          className="bg-[#070b13] hover:bg-[#0e1624] border border-[#1b2537] hover:border-cyan-500/60 rounded-xl p-3 sm:p-4 transition-all duration-200 group shadow-lg flex flex-col justify-between cursor-pointer"
        >
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-lg sm:text-2xl">📊</span>
              <span className="text-[9px] sm:text-[10px] font-mono font-bold text-cyan-400 bg-cyan-950/50 px-1.5 py-0.5 rounded">
                Step 2
              </span>
            </div>
            <div>
              <h3 className="text-xs sm:text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">
                Understand one
              </h3>
              <p className="text-[10px] sm:text-xs text-slate-400 leading-relaxed pt-0.5">
                Confluence, thesis health, and invalidation levels
              </p>
            </div>
          </div>
          <span className="text-[9px] sm:text-[10px] font-mono text-slate-500 mt-2 sm:mt-3 block border-t border-[#131d2b] pt-1.5 sm:pt-2">
            Analysis
          </span>
        </div>

        {/* Step 3: Decide whether/how */}
        <Link
          href="/setups"
          className="bg-[#070b13] hover:bg-[#0e1624] border border-[#1b2537] hover:border-cyan-500/60 rounded-xl p-3 sm:p-4 transition-all duration-200 group shadow-lg flex flex-col justify-between"
        >
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-lg sm:text-2xl">🎯</span>
              <span className="text-[9px] sm:text-[10px] font-mono font-bold text-cyan-400 bg-cyan-950/50 px-1.5 py-0.5 rounded">
                Step 3
              </span>
            </div>
            <div>
              <h3 className="text-xs sm:text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">
                Decide whether/how
              </h3>
              <p className="text-[10px] sm:text-xs text-slate-400 leading-relaxed pt-0.5">
                Position sizing, entry rules, and risk parameters
              </p>
            </div>
          </div>
          <span className="text-[9px] sm:text-[10px] font-mono text-slate-500 mt-2 sm:mt-3 block border-t border-[#131d2b] pt-1.5 sm:pt-2">
            Trade Plan
          </span>
        </Link>

        {/* Step 4: Manage what I own */}
        <Link
          href="/portfolio"
          className="bg-[#070b13] hover:bg-[#0e1624] border border-[#1b2537] hover:border-cyan-500/60 rounded-xl p-3 sm:p-4 transition-all duration-200 group shadow-lg flex flex-col justify-between"
        >
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-lg sm:text-2xl">🛡️</span>
              <span className="text-[9px] sm:text-[10px] font-mono font-bold text-cyan-400 bg-cyan-950/50 px-1.5 py-0.5 rounded">
                Step 4
              </span>
            </div>
            <div>
              <h3 className="text-xs sm:text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">
                Manage what I own
              </h3>
              <p className="text-[10px] sm:text-xs text-slate-400 leading-relaxed pt-0.5">
                Concentration, VaR, and cash reserve monitoring
              </p>
            </div>
          </div>
          <span className="text-[9px] sm:text-[10px] font-mono text-slate-500 mt-2 sm:mt-3 block border-t border-[#131d2b] pt-1.5 sm:pt-2">
            Portfolio
          </span>
        </Link>
      </div>

      {/* Quick Search Modal */}
      {isSearchOpen && (
        <form
          onSubmit={handleQuickSearch}
          className="bg-[#06090f] border border-cyan-800/60 p-3 rounded-xl flex items-center gap-2 animate-fade-in"
        >
          <span className="text-slate-400 text-xs font-mono font-bold">Ticker:</span>
          <input
            type="text"
            placeholder="e.g. NVDA, FIX, AAPL, MSFT, ANET..."
            value={quickSymbol}
            onChange={(e) => setQuickSymbol(e.target.value)}
            autoFocus
            className="flex-1 bg-transparent text-xs text-white placeholder-slate-500 font-mono font-bold focus:outline-none"
          />
          <button
            type="submit"
            className="px-3 py-1 bg-cyan-600 hover:bg-cyan-500 text-slate-950 rounded-lg text-xs font-mono font-bold transition-all cursor-pointer"
          >
            Launch Analysis
          </button>
          <button
            type="button"
            onClick={() => setIsSearchOpen(false)}
            className="text-slate-400 hover:text-white px-2 text-xs cursor-pointer"
          >
            ✕
          </button>
        </form>
      )}
    </div>
  );
}
