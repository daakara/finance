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
      <div className="text-center max-w-xl mx-auto space-y-1">
        <span className="text-[10px] font-mono uppercase tracking-widest text-cyan-400 font-bold">
          ARX Objective-Driven Workspace
        </span>
        <h2 className="text-xl sm:text-2xl font-black text-white tracking-tight">
          What are you looking to do today?
        </h2>
        <p className="text-xs text-slate-400 font-sans">
          Select an objective to launch a guided, goal-oriented market journey.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* Card 1: Find an investment */}
        <Link
          href="/screener"
          className="bg-[#070b13] hover:bg-[#0e1624] border border-[#1b2537] hover:border-cyan-500/60 rounded-xl p-4 transition-all duration-200 group shadow-lg flex flex-col justify-between"
        >
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-2xl">🔎</span>
              <span className="text-[10px] font-mono font-bold text-cyan-400 opacity-0 group-hover:opacity-100 transition-opacity">
                Explore →
              </span>
            </div>
            <div>
              <h3 className="text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">
                Find an investment
              </h3>
              <p className="text-[11px] text-cyan-400 font-mono mt-0.5">
                &ldquo;Show me stocks worth researching&rdquo;
              </p>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed pt-1">
              Scan high-conviction setups, growing compounders, and undervalued bargains.
            </p>
          </div>
          <span className="text-[10px] font-mono text-slate-500 mt-3 block border-t border-[#131d2b] pt-2">
            Goal-Driven Screener
          </span>
        </Link>

        {/* Card 2: Understand a stock */}
        <div
          onClick={() => setIsSearchOpen(true)}
          className="bg-[#070b13] hover:bg-[#0e1624] border border-[#1b2537] hover:border-cyan-500/60 rounded-xl p-4 transition-all duration-200 group shadow-lg flex flex-col justify-between cursor-pointer"
        >
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-2xl">📊</span>
              <span className="text-[10px] font-mono font-bold text-cyan-400 opacity-0 group-hover:opacity-100 transition-opacity">
                Analyze →
              </span>
            </div>
            <div>
              <h3 className="text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">
                Understand a stock
              </h3>
              <p className="text-[11px] text-cyan-400 font-mono mt-0.5">
                &ldquo;I have a ticker and want to evaluate it&rdquo;
              </p>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed pt-1">
              Check confluence evidence, thesis health, and downside invalidation levels.
            </p>
          </div>
          <span className="text-[10px] font-mono text-slate-500 mt-3 block border-t border-[#131d2b] pt-2">
            Adaptive Terminal Engine
          </span>
        </div>

        {/* Card 3: Check my portfolio */}
        <Link
          href="/portfolio"
          className="bg-[#070b13] hover:bg-[#0e1624] border border-[#1b2537] hover:border-cyan-500/60 rounded-xl p-4 transition-all duration-200 group shadow-lg flex flex-col justify-between"
        >
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-2xl">🛡️</span>
              <span className="text-[10px] font-mono font-bold text-cyan-400 opacity-0 group-hover:opacity-100 transition-opacity">
                Review →
              </span>
            </div>
            <div>
              <h3 className="text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">
                Check my portfolio
              </h3>
              <p className="text-[11px] text-cyan-400 font-mono mt-0.5">
                &ldquo;Show me where my biggest risks are&rdquo;
              </p>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed pt-1">
              Inspect position concentration, Cornish-Fisher VaR (95%), and cash reserve buffers.
            </p>
          </div>
          <span className="text-[10px] font-mono text-slate-500 mt-3 block border-t border-[#131d2b] pt-2">
            Zero-Login Private Storage
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
