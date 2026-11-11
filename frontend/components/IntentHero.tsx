"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Search, LineChart, Target, ShieldCheck } from "lucide-react";

interface IntentHeroProps {
  onSelectSymbol?: (sym: string) => void;
}

export default function IntentHero({ onSelectSymbol }: IntentHeroProps) {
  const router = useRouter();
  const [quickSymbol, setQuickSymbol] = useState("");
  const [isSearchOpen, setIsSearchOpen] = useState(false);

  const handleStep2Click = () => {
    // If landing search input exists on page, focus and scroll smoothly
    if (typeof document !== "undefined") {
      const searchInput = document.getElementById("landing-search-input") as HTMLInputElement | null;
      if (searchInput) {
        searchInput.focus();
        searchInput.scrollIntoView({ behavior: "smooth", block: "center" });
        return;
      }
    }
    // If not on landing page or search input not present, dispatch universal OmniSearch
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("arx:open-omni-search"));
      return;
    }
    // Fallback: toggle inline search
    setIsSearchOpen(true);
  };

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
    <div
      data-testid="decision-workflow-hero"
      className="bg-[var(--surface-1)] border border-[var(--border-subtle)] rounded-2xl p-4 sm:p-6 shadow-2xl space-y-4 font-sans text-slate-100"
    >
      <div className="text-center max-w-2xl mx-auto space-y-1.5">
        <span className="text-xs font-mono uppercase tracking-widest text-cyan-400 font-bold">
          Decision Workflow
        </span>
        <h2 className="text-xl sm:text-2xl font-black text-white tracking-tight">
          Discover → Analyze → Validate Risk → Plan
        </h2>
        <p className="text-xs text-slate-400 font-sans leading-relaxed">
          Four-tier institutional progression separating candidate screening, single-asset conviction, risk clearance, and capital allocation.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Step 1: Discover */}
        <Link
          href="/radar"
          data-testid="workflow-step-discover"
          className="bg-[var(--surface-2)] hover:bg-[#131d2c] border border-[var(--border-subtle)] hover:border-cyan-500/60 rounded-xl p-3.5 sm:p-4 transition-all duration-200 group shadow-lg flex flex-col justify-between min-h-[130px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500"
        >
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <Search className="w-5 h-5 sm:w-6 sm:h-6 text-cyan-400 group-hover:scale-110 transition-transform" aria-hidden="true" />
              <span className="text-xs font-mono font-bold text-cyan-300 bg-cyan-950/80 border border-cyan-800/80 px-2 py-0.5 rounded">
                Step 1 · Screen
              </span>
            </div>
            <div>
              <h3 className="text-xs sm:text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">
                Discover
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed pt-0.5 font-sans">
                Scan setups, compounders, and volume contraction anomalies
              </p>
            </div>
          </div>
          <span className="text-xs font-mono text-slate-400 mt-2 sm:mt-3 block border-t border-[var(--border-subtle)] pt-1.5 sm:pt-2">
            Radar Screener →
          </span>
        </Link>

        {/* Step 2: Analyze */}
        <button
          type="button"
          onClick={handleStep2Click}
          data-testid="workflow-step-analyze"
          className="bg-[var(--surface-2)] hover:bg-[#131d2c] border border-[var(--border-subtle)] hover:border-cyan-500/60 rounded-xl p-3.5 sm:p-4 transition-all duration-200 group shadow-lg flex flex-col justify-between cursor-pointer text-left min-h-[130px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500"
        >
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <LineChart className="w-5 h-5 sm:w-6 sm:h-6 text-cyan-400 group-hover:scale-110 transition-transform" aria-hidden="true" />
              <span className="text-xs font-mono font-bold text-cyan-300 bg-cyan-950/80 border border-cyan-800/80 px-2 py-0.5 rounded">
                Step 2 · Conviction
              </span>
            </div>
            <div>
              <h3 className="text-xs sm:text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">
                Analyze
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed pt-0.5 font-sans">
                Confluence attribution, 9-domain trace, and thesis health
              </p>
            </div>
          </div>
          <span className="text-xs font-mono text-slate-400 mt-2 sm:mt-3 block border-t border-[var(--border-subtle)] pt-1.5 sm:pt-2 w-full">
            Asset Cockpit →
          </span>
        </button>

        {/* Step 3: Validate Risk */}
        <Link
          href="/setups"
          data-testid="workflow-step-validate-risk"
          className="bg-[var(--surface-2)] hover:bg-[#131d2c] border border-[var(--border-subtle)] hover:border-cyan-500/60 rounded-xl p-3.5 sm:p-4 transition-all duration-200 group shadow-lg flex flex-col justify-between min-h-[130px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500"
        >
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <Target className="w-5 h-5 sm:w-6 sm:h-6 text-cyan-400 group-hover:scale-110 transition-transform" aria-hidden="true" />
              <span className="text-xs font-mono font-bold text-cyan-300 bg-cyan-950/80 border border-cyan-800/80 px-2 py-0.5 rounded">
                Step 3 · Clearance
              </span>
            </div>
            <div>
              <h3 className="text-xs sm:text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">
                Validate Risk
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed pt-0.5 font-sans">
                Pre-flight readiness, regime posture, and operational holds
              </p>
            </div>
          </div>
          <span className="text-xs font-mono text-slate-400 mt-2 sm:mt-3 block border-t border-[var(--border-subtle)] pt-1.5 sm:pt-2">
            Pre-Flight Gate →
          </span>
        </Link>

        {/* Step 4: Plan */}
        <Link
          href="/portfolio"
          data-testid="workflow-step-plan"
          className="bg-[var(--surface-2)] hover:bg-[#131d2c] border border-[var(--border-subtle)] hover:border-cyan-500/60 rounded-xl p-3.5 sm:p-4 transition-all duration-200 group shadow-lg flex flex-col justify-between min-h-[130px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500"
        >
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <ShieldCheck className="w-5 h-5 sm:w-6 sm:h-6 text-cyan-400 group-hover:scale-110 transition-transform" aria-hidden="true" />
              <span className="text-xs font-mono font-bold text-cyan-300 bg-cyan-950/80 border border-cyan-800/80 px-2 py-0.5 rounded">
                Step 4 · Allocation
              </span>
            </div>
            <div>
              <h3 className="text-xs sm:text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">
                Plan
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed pt-0.5 font-sans">
                Position sizing governor, asymmetric brackets, and cash limits
              </p>
            </div>
          </div>
          <span className="text-xs font-mono text-slate-400 mt-2 sm:mt-3 block border-t border-[var(--border-subtle)] pt-1.5 sm:pt-2">
            Portfolio & Sizing →
          </span>
        </Link>
      </div>

      {/* Quick Search Fallback (Only if explicitly open) */}
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
            className="flex-1 bg-transparent text-xs text-white placeholder-slate-500 font-mono font-bold focus:outline-none min-h-[36px]"
          />
          <button
            type="submit"
            className="px-3 py-1.5 bg-cyan-500 hover:bg-cyan-400 text-slate-950 rounded-lg text-xs font-mono font-bold transition-all cursor-pointer min-h-[36px]"
          >
            Launch Analysis
          </button>
          <button
            type="button"
            onClick={() => setIsSearchOpen(false)}
            className="text-slate-400 hover:text-white px-2.5 py-1 text-xs cursor-pointer min-h-[36px] flex items-center justify-center"
          >
            ✕
          </button>
        </form>
      )}
    </div>
  );
}
