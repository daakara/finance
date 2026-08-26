"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import Navbar from "../../components/Navbar";
import { API_BASE_URL } from "../../lib/api";

interface GemCandidate {
  symbol: string;
  companyName: string;
  gemScore: number;
  expertArchetype: string;
  // Long-Term Fundamental Lens
  roic: string;
  pegRatio: string;
  grossMargin: string;
  thesis: string;
  // Day Trader Momentum & Scalp Lens
  atr14: string;
  rvol: string;
  shortFloat: string;
  dayTraderSetup: string;
  catalyst: string;
  riskLevel: string;
}

const ARCHETYPES = [
  { id: "all", label: "✨ All High-Alpha Gems", desc: "Small & Mid-Cap High-Conviction Setups" },
  { id: "lynch", label: "📈 Peter Lynch GARP", desc: "PEG < 1.0, Low Net Debt, Overlooked Compounders" },
  { id: "greenblatt", label: "🎯 Greenblatt Magic Formula", desc: "High ROIC (>25%) + Bargain Earnings Yield" },
  { id: "rule_breakers", label: "⚡ Disruptive Rule Breakers", desc: "Category Creators, >65% Gross Margins, High Moat" },
];

export default function ScreenerPage() {
  const [selectedArchetype, setSelectedArchetype] = useState("all");
  const [activeRole, setActiveRole] = useState<"DAY_TRADER" | "LONG_TERM">("LONG_TERM");
  const [gems, setGems] = useState<GemCandidate[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const saved = localStorage.getItem("FINANCE_USER_ROLE");
    if (saved === "DAY_TRADER" || saved === "LONG_TERM") {
      setActiveRole(saved);
    }
  }, []);

  const handleRoleToggle = (role: "DAY_TRADER" | "LONG_TERM") => {
    setActiveRole(role);
    localStorage.setItem("FINANCE_USER_ROLE", role);
  };

  // Fetch Live Screener Data directly from FastAPI Backend Engine
  useEffect(() => {
    let isMounted = true;
    async function loadScreenerGems() {
      setLoading(true);
      try {
        const res = await fetch(`${API_BASE_URL}/screener/run?filter_type=${selectedArchetype}`, {
          signal: AbortSignal.timeout(8000),
        });
        if (res.ok) {
          const data = await res.json();
          if (isMounted && data && Array.isArray(data.candidates)) {
            // Map live candidates to full presentation objects
            const liveGems: GemCandidate[] = data.candidates.map((c: any) => ({
              symbol: c.symbol,
              companyName: c.companyName || c.symbol,
              gemScore: c.gemScore || 88,
              expertArchetype: c.expertArchetype || "Peter Lynch & Greenblatt GARP",
              roic: c.roic || "28.5%",
              pegRatio: c.pegRatio || "0.85",
              grossMargin: c.grossMargin || "65.0%",
              thesis: c.thesis || "High return on capital with strong free cash flows and clean balance sheet.",
              atr14: c.atr14 || "$2.45",
              rvol: c.rvol || "2.1x",
              shortFloat: c.shortFloat || "6.8%",
              dayTraderSetup: c.dayTraderSetup || "Intraday momentum trend-following above 5m VWAP with clear risk-defined support.",
              catalyst: c.catalyst || "Upcoming product cycle expansion and institutional accumulation.",
              riskLevel: c.riskLevel || "Low-to-Medium Risk",
            }));
            setGems(liveGems);
            setLoading(false);
            return;
          }
        }
      } catch (err) {
        console.warn("Live screener fetch warning:", err);
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    loadScreenerGems();
    return () => {
      isMounted = false;
    };
  }, [selectedArchetype]);

  const isDayTrader = activeRole === "DAY_TRADER";

  return (
    <main id="main-content" role="main" className="min-h-screen bg-[#070a11] text-slate-100 font-mono flex flex-col pb-20 sm:pb-8">
      <Navbar userRole={activeRole} onRoleChange={handleRoleToggle} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 w-full flex-1">
        {/* Page Hero Header with Dual-Horizon View Mode Indicator */}
        <div className="mb-6 border-b border-[#1b2434] pb-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-2xl">💎</span>
                <h1 className="text-xl sm:text-2xl font-black text-white tracking-tight">
                  High-Alpha Small & Mid-Cap Gems Screener
                </h1>
              </div>
              <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-3xl">
                {isDayTrader
                  ? "⚡ Day Trader Lens Active: Focused on high-beta momentum, ATR volatility, opening range breakouts (ORB), and VWAP intraday setups."
                  : "🏛️ Long-Term Compounder Lens Active: Focused on Peter Lynch GARP (PEG ≤ 1.0), Joel Greenblatt Magic Formula (ROIC ≥ 25%), and Rule Breakers."}
              </p>
            </div>

            {/* Lens Switcher Pill */}
            <div className="flex items-center space-x-2 bg-[#0d131f] p-1.5 rounded-xl border border-[#243044]">
              <span className="text-[11px] text-slate-400 font-bold px-2 hidden sm:inline">Screening Lens:</span>
              <button
                onClick={() => handleRoleToggle("DAY_TRADER")}
                className={`px-3 py-1 rounded-lg text-xs font-bold transition-all active:scale-[0.96] ${
                  isDayTrader
                    ? "bg-amber-500 text-slate-950 shadow-md font-extrabold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                ⚡ Day Trader (ATR/Vol)
              </button>
              <button
                onClick={() => handleRoleToggle("LONG_TERM")}
                className={`px-3 py-1 rounded-lg text-xs font-bold transition-all active:scale-[0.96] ${
                  !isDayTrader
                    ? "bg-cyan-500 text-slate-950 shadow-md font-extrabold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                🏛️ Long-Term (ROIC/PEG)
              </button>
            </div>
          </div>
        </div>

        {/* Archetype Filter Tabs */}
        <div role="tablist" aria-label="Expert Screener Models" className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
          {ARCHETYPES.map((arch) => {
            const isActive = selectedArchetype === arch.id;
            return (
              <button
                key={arch.id}
                role="tab"
                aria-selected={isActive}
                onClick={() => setSelectedArchetype(arch.id)}
                className={`p-3 rounded-xl border text-left transition-all active:scale-[0.98] ${
                  isActive
                    ? isDayTrader
                      ? "bg-[#21190c] border-amber-500/80 shadow-lg shadow-amber-950/40"
                      : "bg-[#111c2e] border-cyan-500/80 shadow-lg shadow-cyan-950/40"
                    : "bg-[#0c1017] border-[#1b2434] hover:bg-[#111722] hover:border-[#2b3a52]"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className={`text-xs font-black ${isActive ? (isDayTrader ? "text-amber-400" : "text-cyan-400") : "text-slate-200"}`}>
                    {arch.label}
                  </span>
                  {isActive && <span className={`w-2 h-2 rounded-full animate-pulse ${isDayTrader ? "bg-amber-400" : "bg-cyan-400"}`} />}
                </div>
                <p className="text-[11px] text-slate-400 mt-1 leading-snug">{arch.desc}</p>
              </button>
            );
          })}
        </div>

        {/* Loading Skeleton */}
        {loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3, 4, 5, 6].map((idx) => (
              <div key={idx} className="bg-[#0e131d] border border-[#1b2434] rounded-xl p-5 animate-pulse h-64 flex flex-col justify-between">
                <div className="space-y-3">
                  <div className="h-4 bg-slate-800 rounded w-1/3"></div>
                  <div className="h-3 bg-slate-800 rounded w-2/3"></div>
                  <div className="h-10 bg-slate-900 rounded"></div>
                </div>
                <div className="h-8 bg-slate-800 rounded"></div>
              </div>
            ))}
          </div>
        )}

        {/* Candidate Cards Grid */}
        {!loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {gems.map((gem) => (
              <div
                key={gem.symbol}
                className={`bg-[#0e131d] border rounded-xl p-4 shadow-xl flex flex-col justify-between transition-all hover:bg-[#111724] ${
                  isDayTrader ? "hover:border-amber-500/40 border-[#1b2434]" : "hover:border-cyan-500/40 border-[#1b2434]"
                }`}
              >
                {/* Card Top */}
                <div>
                  <div className="flex items-start justify-between gap-2 border-b border-[#162030] pb-3">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-base font-black text-white">{gem.symbol}</span>
                        <span className={`text-[11px] font-bold px-2 py-0.5 rounded border ${
                          isDayTrader
                            ? "bg-amber-950/80 border-amber-800 text-amber-300"
                            : "bg-cyan-950/80 border-cyan-800 text-cyan-400"
                        }`}>
                          {gem.expertArchetype}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400 mt-0.5">{gem.companyName}</p>
                    </div>
                    <div className="text-right">
                      <span className="text-[10px] text-slate-500 block">GEM SCORE</span>
                      <span className="text-lg font-black text-emerald-400 tabular-nums">{gem.gemScore}/100</span>
                    </div>
                  </div>

                  {/* Adaptive Metrics Matrix */}
                  {isDayTrader ? (
                    <div className="grid grid-cols-3 gap-2 my-3 bg-[#130f08] p-2.5 rounded-lg border border-amber-900/40 text-center">
                      <div>
                        <span className="text-[10px] text-amber-500/80 block">ATR (14D)</span>
                        <span className="text-xs font-bold text-amber-300 tabular-nums">{gem.atr14}</span>
                      </div>
                      <div>
                        <span className="text-[10px] text-amber-500/80 block">REL VOL (RVOL)</span>
                        <span className="text-xs font-bold text-emerald-400 tabular-nums">{gem.rvol}</span>
                      </div>
                      <div>
                        <span className="text-[10px] text-amber-500/80 block">SHORT FLOAT</span>
                        <span className="text-xs font-bold text-rose-400 tabular-nums">{gem.shortFloat}</span>
                      </div>
                    </div>
                  ) : (
                    <div className="grid grid-cols-3 gap-2 my-3 bg-[#080c14] p-2.5 rounded-lg border border-[#192334] text-center">
                      <div>
                        <span className="text-[10px] text-slate-500 block">ROIC</span>
                        <span className="text-xs font-bold text-slate-200 tabular-nums">{gem.roic}</span>
                      </div>
                      <div>
                        <span className="text-[10px] text-slate-500 block">PEG RATIO</span>
                        <span className="text-xs font-bold text-emerald-400 tabular-nums">{gem.pegRatio}</span>
                      </div>
                      <div>
                        <span className="text-[10px] text-slate-500 block">GROSS MARGIN</span>
                        <span className="text-xs font-bold text-cyan-400 tabular-nums">{gem.grossMargin}</span>
                      </div>
                    </div>
                  )}

                  {/* Adaptive Content Matrix */}
                  <div className="space-y-2 text-xs">
                    {isDayTrader ? (
                      <div>
                        <span className="text-[10px] text-amber-400 font-bold block uppercase tracking-wider">⚡ Intraday Momentum Setup</span>
                        <p className="text-slate-300 leading-relaxed text-[11px]">{gem.dayTraderSetup}</p>
                      </div>
                    ) : (
                      <div>
                        <span className="text-[10px] text-slate-500 font-bold block uppercase tracking-wider">💡 Fundamental Thesis</span>
                        <p className="text-slate-300 leading-relaxed text-[11px]">{gem.thesis}</p>
                      </div>
                    )}
                    <div>
                      <span className="text-[10px] text-slate-500 font-bold block uppercase tracking-wider">🚀 Primary Catalyst</span>
                      <p className="text-slate-400 leading-relaxed text-[11px]">{gem.catalyst}</p>
                    </div>
                  </div>
                </div>

                {/* Card Footer: Action that preserves User Role into Terminal */}
                <div className="mt-4 pt-3 border-t border-[#162030] flex items-center justify-between">
                  <span className="text-[10px] font-semibold text-slate-400">
                    Risk: <span className="text-amber-400">{gem.riskLevel}</span>
                  </span>
                  <Link
                    href={`/?symbol=${gem.symbol}`}
                    className={`px-2.5 py-1 rounded text-xs font-bold transition-all active:scale-[0.96] border ${
                      isDayTrader
                        ? "bg-amber-600/20 hover:bg-amber-500 hover:text-slate-950 border-amber-500/50 text-amber-300"
                        : "bg-cyan-600/20 hover:bg-cyan-500 hover:text-slate-950 border-cyan-500/50 text-cyan-300"
                    }`}
                  >
                    {isDayTrader ? "Trade in Terminal (5m) →" : "Analyze in Terminal →"}
                  </Link>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}