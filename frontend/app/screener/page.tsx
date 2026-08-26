"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import Navbar from "../../components/Navbar";

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

const CURATED_SMALL_CAP_GEMS: GemCandidate[] = [
  {
    symbol: "ELF",
    companyName: "e.l.f. Beauty Inc.",
    gemScore: 94,
    expertArchetype: "Peter Lynch GARP Compounder",
    roic: "28.4%",
    pegRatio: "0.84",
    grossMargin: "71.2%",
    thesis: "Disruptive beauty brand taking rapid global market share with digitally-native marketing and negative working capital cycles.",
    atr14: "$3.45",
    rvol: "2.4x",
    shortFloat: "11.2%",
    dayTraderSetup: "High-beta intraday gap fills. Key support at 20 EMA with rapid momentum bounce on above-average morning volume.",
    catalyst: "International retail expansion (UK/Europe) and premium skincare segment acquisitions.",
    riskLevel: "Low-to-Medium",
  },
  {
    symbol: "MEDP",
    companyName: "Medpace Holdings Inc.",
    gemScore: 92,
    expertArchetype: "Greenblatt Magic Formula",
    roic: "38.6%",
    pegRatio: "1.10",
    grossMargin: "48.5%",
    thesis: "Elite clinical contract research organization with pure-play focus on small biopharma, high return on capital, and zero debt.",
    atr14: "$8.20",
    rvol: "1.8x",
    shortFloat: "5.4%",
    dayTraderSetup: "Clean institutional trend-following. Tight morning VWAP compressions leading to afternoon multi-point expansion breakouts.",
    catalyst: "Accelerating biopharma venture funding rounds and high RFP backlog conversion.",
    riskLevel: "Low",
  },
  {
    symbol: "DUOL",
    companyName: "Duolingo Inc.",
    gemScore: 91,
    expertArchetype: "Disruptive Rule Breaker",
    roic: "26.1%",
    pegRatio: "1.05",
    grossMargin: "73.4%",
    thesis: "Dominant organic mobile education platform with virality-driven user acquisition, GenAI learning tiers, and expanding operating leverage.",
    atr14: "$6.80",
    rvol: "3.1x",
    shortFloat: "8.6%",
    dayTraderSetup: "High-volume opening range breakout (ORB). Fast momentum swings off 5m VWAP with strong retail & institutional tape.",
    catalyst: "Duolingo Max GenAI monetization and enterprise English test global adoption.",
    riskLevel: "Medium",
  },
  {
    symbol: "POWI",
    companyName: "Power Integrations Inc.",
    gemScore: 89,
    expertArchetype: "Peter Lynch GARP Compounder",
    roic: "24.5%",
    pegRatio: "0.92",
    grossMargin: "54.8%",
    thesis: "Niche monopoly in energy-efficient GaN (Gallium Nitride) and high-voltage power conversion chips for EVs and data centers.",
    atr14: "$1.95",
    rvol: "1.5x",
    shortFloat: "4.2%",
    dayTraderSetup: "Range-bound mean reversion scalp between daily Bollinger Bands with minimal slippage and predictable order flow.",
    catalyst: "Server power supply efficiency mandates and GaN adoption in high-power appliances.",
    riskLevel: "Low-to-Medium",
  },
  {
    symbol: "CPRX",
    companyName: "Catalyst Pharmaceuticals",
    gemScore: 88,
    expertArchetype: "Greenblatt Magic Formula",
    roic: "34.2%",
    pegRatio: "0.78",
    grossMargin: "78.9%",
    thesis: "High-margin rare disease biotech with rock-solid free cash flow, massive operating margins (>45%), and pristine net cash position.",
    atr14: "$0.85",
    rvol: "1.9x",
    shortFloat: "7.1%",
    dayTraderSetup: "Low float accumulation scalps. High profit-to-loss ratio when buying intraday dips near VWAP with defined 1.5% stop loss.",
    catalyst: "Firdapse patent exclusivity defense and strategic orphan drug portfolio M&A.",
    riskLevel: "Medium",
  },
  {
    symbol: "IOT",
    companyName: "Samsara Inc.",
    gemScore: 87,
    expertArchetype: "Disruptive Rule Breaker",
    roic: "22.8%",
    pegRatio: "1.25",
    grossMargin: "75.1%",
    thesis: "Leader in physical operations cloud computing, connecting commercial vehicle fleets and industrial assets with recurring high-margin ARR.",
    atr14: "$1.65",
    rvol: "2.7x",
    shortFloat: "9.3%",
    dayTraderSetup: "Enterprise SaaS momentum runner. Strong multi-day continuation above intraday VWAP with tight 15m consolidation flags.",
    catalyst: "Connected asset safety mandates and multi-product customer expansion (>100k ARR tier).",
    riskLevel: "Medium",
  },
  {
    symbol: "AXON",
    companyName: "Axon Enterprise Inc.",
    gemScore: 93,
    expertArchetype: "Disruptive Rule Breaker",
    roic: "27.4%",
    pegRatio: "1.32",
    grossMargin: "62.3%",
    thesis: "Unassailable public safety software & hardware ecosystem (TASERS, Body Cams, Evidence.com cloud) with 120%+ net revenue retention.",
    atr14: "$9.40",
    rvol: "2.1x",
    shortFloat: "3.8%",
    dayTraderSetup: "Institutional trend continuation. Superb risk/reward on pullback touches to 20-period EMA during trending market sessions.",
    catalyst: "Draft One generative AI police report transcription software and international police force rollout.",
    riskLevel: "Low",
  },
  {
    symbol: "RKLB",
    companyName: "Rocket Lab USA Inc.",
    gemScore: 86,
    expertArchetype: "Disruptive Rule Breaker",
    roic: "19.5%",
    pegRatio: "1.40",
    grossMargin: "32.4%",
    thesis: "Leading commercial small-satellite launch provider and space systems manufacturer; the only viable Western competitor to SpaceX.",
    atr14: "$1.40",
    rvol: "4.5x",
    shortFloat: "14.8%",
    dayTraderSetup: "High-octane short-squeeze candidate. Massive intraday volume surges on DoD launch news; ideal for momentum breakout scalping.",
    catalyst: "Medium-lift Neutron rocket maiden flight and multi-billion-dollar DoD Space Development Agency contracts.",
    riskLevel: "High Growth",
  },
];

export default function ScreenerPage() {
  const [selectedArchetype, setSelectedArchetype] = useState("all");
  const [activeRole, setActiveRole] = useState<"DAY_TRADER" | "LONG_TERM">("LONG_TERM");
  const [gems, setGems] = useState<GemCandidate[]>(CURATED_SMALL_CAP_GEMS);

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

  useEffect(() => {
    if (selectedArchetype === "all") {
      setGems(CURATED_SMALL_CAP_GEMS);
    } else if (selectedArchetype === "lynch") {
      setGems(CURATED_SMALL_CAP_GEMS.filter((g) => g.expertArchetype.includes("Lynch")));
    } else if (selectedArchetype === "greenblatt") {
      setGems(CURATED_SMALL_CAP_GEMS.filter((g) => g.expertArchetype.includes("Greenblatt")));
    } else if (selectedArchetype === "rule_breakers") {
      setGems(CURATED_SMALL_CAP_GEMS.filter((g) => g.expertArchetype.includes("Rule Breaker")));
    }
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

        {/* Candidate Cards Grid */}
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

                {/* Adaptive Metrics Matrix (Swaps between Long-Term and Day Trader metrics) */}
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
      </div>
    </main>
  );
}

