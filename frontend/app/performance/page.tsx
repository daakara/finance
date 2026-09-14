"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import TerminalShell from "../../components/terminal/TerminalShell";
import PageIntro from "../../components/PageIntro";
import { fetchJournalTrades, JournalTradeRecord } from "../../lib/api";
import { loadPortfolioPositions, PortfolioPosition } from "../../lib/portfolio";
import {
  filterEligibleLiveTrades,
  computeRealizedMetrics,
} from "../../lib/performanceMetrics";

export default function PerformancePage() {
  const [, setLivePositions] = useState<PortfolioPosition[]>([]);
  const [liveTrades, setLiveTrades] = useState<JournalTradeRecord[]>([]);
  const [, setLiveLoading] = useState<boolean>(true);
  const [searchTicker, setSearchTicker] = useState("");

  useEffect(() => {
    if (typeof window !== "undefined") {
      const positions = loadPortfolioPositions();
      setLivePositions(positions);

      const params = new URLSearchParams(window.location.search);
      const sym = params.get("symbol") || params.get("ticker");
      if (sym) {
        setSearchTicker(sym.trim().toUpperCase());
      }
    }
  }, []);

  useEffect(() => {
    let isMounted = true;
    setLiveLoading(true);
    fetchJournalTrades(200)
      .then((trades) => {
        if (!isMounted) return;
        setLiveTrades(trades || []);
        setLiveLoading(false);
      })
      .catch((err) => {
        if (!isMounted) return;
        console.warn("Failed to fetch journal trades for performance:", err);
        setLiveLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const eligibleLiveTrades = filterEligibleLiveTrades(liveTrades);
  const liveSummary = computeRealizedMetrics(eligibleLiveTrades, liveTrades.length);

  return (
    <TerminalShell activeHub="performance" activeSymbol={searchTicker.trim() ? searchTicker.trim() : null}>
      <div className="space-y-6 pb-20">
        <PageIntro
          hubId="performance"
          title="Performance & Realized Attribution"
          purpose="Historical return metrics and counterfactual edge attribution (Planned for Post-R1)."
          badge="Deferred Scope"
          symbol={searchTicker || null}
          primaryAction={{
            label: "Manage Portfolio →",
            href: "/portfolio",
          }}
          secondaryAction={{
            label: "Explore Setups →",
            href: "/setups",
          }}
        />

        {/* Option A Deferred Scope Card */}
        <div className="p-6 md:p-8 rounded-2xl bg-slate-900/80 border border-amber-500/30 shadow-2xl backdrop-blur space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div className="flex items-center gap-2.5">
              <span className="px-2.5 py-1 rounded-md text-[11px] font-mono font-black uppercase tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/40">
                DEFERRED SCOPE · POST-R1
              </span>
              <span className="text-xs text-slate-400 font-mono">
                Canonical Journey: Radar → Analysis → Trade Plan → Portfolio
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-xs font-mono text-emerald-400">
                Analytics Primitives Preserved ({liveSummary.totalCompletedTrades} closed trades evaluated)
              </span>
            </div>
          </div>

          <div className="space-y-3">
            <h2 className="text-lg md:text-xl font-bold text-white">
              Dedicated Performance Analytics Surface Deferred to Post-R1
            </h2>
            <p className="text-sm text-slate-300 font-sans max-w-3xl leading-relaxed">
              The dedicated Performance and attribution analytics surface is deferred to a post-R1 release to focus Release 1 strictly on the core decision loop: <strong className="text-white">Find → Understand → Plan → Manage</strong>.
            </p>
            <p className="text-xs text-slate-400 font-sans max-w-3xl leading-relaxed">
              Trustworthy counterfactual edge and attribution analytics require an empirically mature trade history before public presentation. All underlying mathematical calculation primitives (<code className="text-cyan-300 font-mono text-xs">computeRealizedMetrics</code>, Sharpe ratio, win rates, expectancies) and trade records remain preserved and operational in code.
            </p>
          </div>

          {/* Core Release 1 Hub Return CTAs */}
          <div className="pt-2">
            <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3">
              Active Release 1 Destinations
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              <Link
                href="/portfolio"
                className="flex flex-col p-4 rounded-xl bg-cyan-950/40 hover:bg-cyan-900/50 border border-cyan-800 hover:border-cyan-500 transition group cursor-pointer"
              >
                <div className="flex items-center justify-between text-xs text-cyan-400 font-bold mb-1">
                  <span>💼 Manage Portfolio</span>
                  <span className="text-slate-400 group-hover:translate-x-0.5 transition-transform">→</span>
                </div>
                <span className="text-[11px] text-slate-300">
                  Track position risk, Cornish-Fisher VaR, and record exits.
                </span>
              </Link>

              <Link
                href="/setups"
                className="flex flex-col p-4 rounded-xl bg-slate-950/60 hover:bg-slate-900 border border-slate-800 hover:border-slate-700 transition group cursor-pointer"
              >
                <div className="flex items-center justify-between text-xs text-white font-bold mb-1">
                  <span>⚡ Plan Trade</span>
                  <span className="text-slate-400 group-hover:translate-x-0.5 transition-transform">→</span>
                </div>
                <span className="text-[11px] text-slate-300">
                  Calculate position sizing, stops, and target brackets.
                </span>
              </Link>

              <Link
                href="/"
                className="flex flex-col p-4 rounded-xl bg-slate-950/60 hover:bg-slate-900 border border-slate-800 hover:border-slate-700 transition group cursor-pointer"
              >
                <div className="flex items-center justify-between text-xs text-white font-bold mb-1">
                  <span>🔬 Deep Analysis</span>
                  <span className="text-slate-400 group-hover:translate-x-0.5 transition-transform">→</span>
                </div>
                <span className="text-[11px] text-slate-300">
                  Understand asset technicals, fundamentals, and regimes.
                </span>
              </Link>

              <Link
                href="/radar"
                className="flex flex-col p-4 rounded-xl bg-slate-950/60 hover:bg-slate-900 border border-slate-800 hover:border-slate-700 transition group cursor-pointer"
              >
                <div className="flex items-center justify-between text-xs text-white font-bold mb-1">
                  <span>📡 Market Radar</span>
                  <span className="text-slate-400 group-hover:translate-x-0.5 transition-transform">→</span>
                </div>
                <span className="text-[11px] text-slate-300">
                  Scan for Stage 2 breakouts, volume dry-up, and momentum.
                </span>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </TerminalShell>
  );
}
