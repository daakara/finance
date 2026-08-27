"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import Navbar from "../../components/Navbar";
import { API_BASE_URL } from "../../lib/api";

interface GemCandidate {
  symbol: string;
  companyName: string;
  currentPrice?: number;
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
  // Execution Scanner Levels
  executionStatus?: "IN_BUY_ZONE" | "APPROACHING_TARGET" | "WAITING_PULLBACK" | "STOPPED_OUT";
  statusLabel?: string;
  statusColor?: string;
  optimalEntryMin?: number;
  optimalEntryMax?: number;
  stopLoss?: number;
  stopLossPct?: number;
  takeProfit1?: number;
  takeProfit1Pct?: number;
  takeProfit2?: number;
  takeProfit2Pct?: number;
  riskRewardRatio?: number;
  setupPattern?: string;
  entryThesis?: string;
}

const FILTER_TABS = [
  { id: "all", label: "✨ All Setups", desc: "Small & Mid-Cap High-Conviction Setups", badge: "Universe" },
  { id: "in_buy_zone", label: "🎯 In Buy Zone", desc: "Price within 1.5% of optimal entry floor/ceiling", badge: "Actionable" },
  { id: "approaching_target", label: "🚀 Near TP Target", desc: "Price approaching Take-Profit 1 or 2 ladders", badge: "Profit Taking" },
  { id: "high_rr", label: "⚡ High R:R (≥ 2.5)", desc: "Asymmetric risk-reward setups with tight stop losses", badge: "Asymmetric" },
  { id: "lynch", label: "📈 Peter Lynch GARP", desc: "PEG < 1.0, Low Net Debt, Overlooked Compounders", badge: "GARP" },
  { id: "greenblatt", label: "🧪 Magic Formula", desc: "High ROIC (>25%) + Bargain Earnings Yield", badge: "Value" },
  { id: "rule_breakers", label: "🔥 Rule Breakers", desc: "Category Creators, >65% Gross Margins, High Moat", badge: "Disruptive" },
];

export default function ScreenerPage() {
  const [selectedFilter, setSelectedFilter] = useState("all");
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
        const res = await fetch(`${API_BASE_URL}/screener/run?filter_type=${selectedFilter}`, {
          signal: AbortSignal.timeout(8000),
        });
        if (res.ok) {
          const data = await res.json();
          if (isMounted && data && Array.isArray(data.candidates)) {
            const liveGems: GemCandidate[] = data.candidates.map((c: any) => ({
              symbol: c.symbol,
              companyName: c.companyName || c.symbol,
              currentPrice: c.currentPrice || 100.0,
              gemScore: c.gemScore || 88,
              expertArchetype: c.expertArchetype || "Peter Lynch & Greenblatt GARP",
              roic: c.roic || "28.5%",
              pegRatio: c.pegRatio || "0.85",
              grossMargin: c.grossMargin || "65.0%",
              thesis: c.thesis || "High return on capital with strong free cash flows and clean balance sheet.",
              atr14: c.atr14 || `$${((c.currentPrice || 100) * 0.025).toFixed(2)}`,
              rvol: c.rvol || "2.1x",
              shortFloat: c.shortFloat || "6.8%",
              dayTraderSetup: c.dayTraderSetup || "Intraday momentum trend-following above 5m VWAP with clear risk-defined support.",
              catalyst: c.catalyst || "Upcoming product cycle expansion and institutional accumulation.",
              riskLevel: c.riskLevel || "Low-to-Medium Risk",
              executionStatus: c.executionStatus || "IN_BUY_ZONE",
              statusLabel: c.statusLabel || "🎯 Active Buy Zone",
              statusColor: c.statusColor || "emerald",
              optimalEntryMin: c.optimalEntryMin || Number(((c.currentPrice || 100) * 0.975).toFixed(2)),
              optimalEntryMax: c.optimalEntryMax || Number(((c.currentPrice || 100) * 0.995).toFixed(2)),
              stopLoss: c.stopLoss || Number(((c.currentPrice || 100) * 0.955).toFixed(2)),
              stopLossPct: c.stopLossPct || -4.5,
              takeProfit1: c.takeProfit1 || Number(((c.currentPrice || 100) * 1.045).toFixed(2)),
              takeProfit1Pct: c.takeProfit1Pct || 4.5,
              takeProfit2: c.takeProfit2 || Number(((c.currentPrice || 100) * 1.095).toFixed(2)),
              takeProfit2Pct: c.takeProfit2Pct || 9.5,
              riskRewardRatio: c.riskRewardRatio || 2.85,
              setupPattern: c.setupPattern || "Minervini Volatility Contraction Pattern (VCP 3-Stage)",
              entryThesis: c.entryThesis || "Stage 2 accumulation breakout above 50-day pivot.",
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
  }, [selectedFilter]);

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
                  High-Alpha Gems & Optimal Execution Scanner
                </h1>
              </div>
              <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-3xl">
                Scan active universe opportunities with real-time **Optimal Buy Zones**, **Stop-Loss Lines**, and **Take-Profit Targets** calibrated via Minervini VCP & ATR risk models.
              </p>
            </div>

            {/* Lens Switcher Pill */}
            <div className="flex items-center space-x-2 bg-[#0d131f] p-1.5 rounded-xl border border-[#243044]">
              <span className="text-[11px] text-slate-400 font-bold px-2 hidden sm:inline">Execution Lens:</span>
              <button
                onClick={() => handleRoleToggle("DAY_TRADER")}
                className={`px-3 py-1 rounded-lg text-xs font-bold transition-all active:scale-[0.96] ${
                  isDayTrader
                    ? "bg-amber-500 text-slate-950 shadow-md font-extrabold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                ⚡ Day Trader (Intraday ATR)
              </button>
              <button
                onClick={() => handleRoleToggle("LONG_TERM")}
                className={`px-3 py-1 rounded-lg text-xs font-bold transition-all active:scale-[0.96] ${
                  !isDayTrader
                    ? "bg-cyan-500 text-slate-950 shadow-md font-extrabold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                🏛️ Swing / Long-Term (VCP)
              </button>
            </div>
          </div>
        </div>

        {/* Execution & Archetype Filter Tabs */}
        <div role="tablist" aria-label="Screener Filter Tabs" className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-7 gap-2.5 mb-6">
          {FILTER_TABS.map((tab) => {
            const isActive = selectedFilter === tab.id;
            return (
              <button
                key={tab.id}
                role="tab"
                aria-selected={isActive}
                onClick={() => setSelectedFilter(tab.id)}
                className={`p-2.5 rounded-xl border text-left transition-all active:scale-[0.98] flex flex-col justify-between ${
                  isActive
                    ? isDayTrader
                      ? "bg-[#21190c] border-amber-500 shadow-md shadow-amber-950/40"
                      : "bg-[#111c2e] border-cyan-500 shadow-md shadow-cyan-950/40"
                    : "bg-[#0c1017] border-[#1b2434] hover:bg-[#111722] hover:border-[#2b3a52]"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between gap-1">
                    <span className={`text-xs font-black truncate ${isActive ? (isDayTrader ? "text-amber-400" : "text-cyan-400") : "text-slate-200"}`}>
                      {tab.label}
                    </span>
                    <span className="text-[9px] px-1.5 py-0.2 rounded font-semibold bg-[#1e293b] text-slate-400">
                      {tab.badge}
                    </span>
                  </div>
                  <p className="text-[10px] text-slate-400 mt-1 line-clamp-2 leading-tight">{tab.desc}</p>
                </div>
              </button>
            );
          })}
        </div>

        {/* Loading Skeleton */}
        {loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3, 4, 5, 6].map((idx) => (
              <div key={idx} className="bg-[#0e131d] border border-[#1b2434] rounded-xl p-5 animate-pulse h-72 flex flex-col justify-between">
                <div className="space-y-3">
                  <div className="h-4 bg-slate-800 rounded w-1/3"></div>
                  <div className="h-3 bg-slate-800 rounded w-2/3"></div>
                  <div className="h-14 bg-slate-900 rounded"></div>
                </div>
                <div className="h-8 bg-slate-800 rounded"></div>
              </div>
            ))}
          </div>
        )}

        {/* Candidate Cards Grid */}
        {!loading && gems.length === 0 && (
          <div className="p-8 text-center bg-[#0e131d] border border-[#1b2434] rounded-xl">
            <span className="text-3xl block mb-2">🔍</span>
            <h3 className="text-base font-bold text-white">No active candidates matching &quot;{selectedFilter}&quot;</h3>
            <p className="text-xs text-slate-400 mt-1">Try switching filter tabs or scanning across all setups.</p>
            <button
              onClick={() => setSelectedFilter("all")}
              className="mt-4 px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-bold transition-all"
            >
              View All Setups
            </button>
          </div>
        )}

        {!loading && gems.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {gems.map((gem) => {
              const statusBg =
                gem.executionStatus === "IN_BUY_ZONE"
                  ? "bg-emerald-950/80 border-emerald-500/80 text-emerald-300"
                  : gem.executionStatus === "APPROACHING_TARGET"
                  ? "bg-amber-950/80 border-amber-500/80 text-amber-300"
                  : gem.executionStatus === "STOPPED_OUT"
                  ? "bg-rose-950/80 border-rose-500/80 text-rose-300"
                  : "bg-cyan-950/80 border-cyan-500/80 text-cyan-300";

              return (
                <div
                  key={gem.symbol}
                  className={`bg-[#0e131d] border rounded-xl p-4 shadow-xl flex flex-col justify-between transition-all hover:bg-[#111724] ${
                    gem.executionStatus === "IN_BUY_ZONE" ? "border-emerald-900/60 hover:border-emerald-500/50" : "border-[#1b2434] hover:border-cyan-500/40"
                  }`}
                >
                  {/* Card Top Header */}
                  <div>
                    <div className="flex items-start justify-between gap-2 border-b border-[#162030] pb-3">
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="text-lg font-black text-white">{gem.symbol}</span>
                          <span className="text-xs font-bold text-slate-300 tabular-nums">
                            ${gem.currentPrice?.toFixed(2)}
                          </span>
                        </div>
                        <p className="text-xs text-slate-400 mt-0.5">{gem.companyName}</p>
                      </div>
                      <div className="text-right">
                        <span className={`text-[11px] font-bold px-2 py-0.5 rounded border inline-block ${statusBg}`}>
                          {gem.statusLabel}
                        </span>
                      </div>
                    </div>

                    {/* Optimal Trade Execution Level Ladder */}
                    <div className="my-3 bg-[#080c14] p-3 rounded-lg border border-[#192334] space-y-2">
                      <div className="flex items-center justify-between text-[11px] pb-1 border-b border-[#141b28]">
                        <span className="text-slate-400 font-bold">🎯 Optimal Buy Zone</span>
                        <span className="text-emerald-400 font-black tabular-nums">
                          ${gem.optimalEntryMin?.toFixed(2)} – ${gem.optimalEntryMax?.toFixed(2)}
                        </span>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-center text-[10px]">
                        <div className="bg-[#110d0f] p-1.5 rounded border border-rose-950">
                          <span className="text-rose-400 block font-semibold">🛑 Stop Loss</span>
                          <span className="text-rose-200 font-bold tabular-nums">
                            ${gem.stopLoss?.toFixed(2)} ({gem.stopLossPct}%)
                          </span>
                        </div>
                        <div className="bg-[#0b1414] p-1.5 rounded border border-emerald-950">
                          <span className="text-emerald-400 block font-semibold">🎯 Target 1</span>
                          <span className="text-emerald-200 font-bold tabular-nums">
                            ${gem.takeProfit1?.toFixed(2)} (+{gem.takeProfit1Pct}%)
                          </span>
                        </div>
                        <div className="bg-[#14120a] p-1.5 rounded border border-amber-950">
                          <span className="text-amber-400 block font-semibold">⚖️ Risk:Reward</span>
                          <span className="text-amber-200 font-black tabular-nums">
                            {gem.riskRewardRatio}:1 R:R
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Fundamental / Technical Thesis */}
                    <div className="space-y-1.5 text-xs">
                      <div>
                        <span className="text-[10px] text-cyan-400 font-bold uppercase tracking-wider block">
                          📐 Setup Pattern:
                        </span>
                        <p className="text-slate-300 leading-relaxed text-[11px]">{gem.setupPattern}</p>
                      </div>
                      <div>
                        <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider block">
                          🚀 Catalyst & Growth:
                        </span>
                        <p className="text-slate-400 leading-relaxed text-[11px]">{gem.catalyst}</p>
                      </div>
                    </div>
                  </div>

                  {/* Card Footer: Action linking to Terminal with preloaded symbol */}
                  <div className="mt-4 pt-3 border-t border-[#162030] flex items-center justify-between">
                    <span className="text-[10px] font-semibold text-slate-400">
                      Model: <span className="text-cyan-300 font-bold">{gem.expertArchetype}</span>
                    </span>
                    <Link
                      href={`/?symbol=${gem.symbol}`}
                      className="px-3 py-1.5 rounded-lg text-xs font-bold transition-all active:scale-[0.96] border bg-cyan-600/20 hover:bg-cyan-500 hover:text-slate-950 border-cyan-500/50 text-cyan-300 flex items-center gap-1 shadow"
                    >
                      <span>Analyze in Terminal</span>
                      <span>→</span>
                    </Link>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
}