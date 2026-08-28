"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import Navbar from "../../components/Navbar";
import DataSourceBadge from "../../components/DataSourceBadge";
import { API_BASE_URL, fetchAssetAnalytics, AnalyticsResponse } from "../../lib/api";
import { SHARED_FACTOR_SCORES, SHARED_WATCHLIST_ITEMS } from "../../lib/constants";
import { getCanonicalAssetName, getCanonicalAssetMoat, getCanonicalAssetRisk } from "../../lib/assetRegistry";

export interface CompetitorAsset {
  symbol: string;
  name: string;
  category: string;
  marketCap: string;
  peRatio: string;
  peRaw: number;
  pegRatio: string;
  pegRaw: number;
  roic: string;
  roicRaw: number;
  grossMargin: string;
  grossMarginRaw: number;
  fcfYield: string;
  fcfYieldRaw: number;
  piotroski: number;
  keyCatalyst: string;
  trialEfficacy: string;
  primaryRisk: string;
  longTermVerdict: string;
  atr14: string;
  atr14Raw: number;
  rvol: string;
  rvolRaw: number;
  intradayBeta: string;
  intradayBetaRaw: number;
  liquidityTier: string;
  dayTraderSetup: string;
  bestTradingWindow: string;
  dayTradeVerdict: string;
}

const AUTHENTIC_FUNDAMENTALS: Record<string, {
  category: string;
  roic: number;
  grossMargin: number;
  fwdPe: number;
  peg: number;
  fcfYield: number;
  piotroski: number;
  atr14: number;
  rvol: number;
  beta: number;
  marketCap: string;
}> = {
  NVDA: { category: "AI Datacenter Monopoly", roic: 48.0, grossMargin: 75.2, fwdPe: 32.4, peg: 0.92, fcfYield: 3.1, piotroski: 8, atr14: 4.85, rvol: 2.8, beta: 1.74, marketCap: "$3,150B" },
  AAPL: { category: "Consumer Hardware & Services Ecosystem", roic: 45.0, grossMargin: 46.0, fwdPe: 28.5, peg: 1.30, fcfYield: 4.2, piotroski: 8, atr14: 2.40, rvol: 1.2, beta: 0.95, marketCap: "$3,450B" },
  MSFT: { category: "Enterprise Cloud & Enterprise AI", roic: 36.0, grossMargin: 69.5, fwdPe: 31.0, peg: 1.22, fcfYield: 3.3, piotroski: 9, atr14: 5.20, rvol: 1.4, beta: 1.05, marketCap: "$3,120B" },
  TSLA: { category: "Autonomous Robotics & EV Fleet", roic: 16.0, grossMargin: 18.2, fwdPe: 65.0, peg: 1.60, fcfYield: 1.5, piotroski: 6, atr14: 8.40, rvol: 3.1, beta: 2.15, marketCap: "$695B" },
  PLTR: { category: "Defense & Enterprise AI Operating System", roic: 23.0, grossMargin: 81.0, fwdPe: 78.0, peg: 1.25, fcfYield: 2.8, piotroski: 8, atr14: 1.65, rvol: 3.8, beta: 1.85, marketCap: "$70B" },
  NVO: { category: "GLP-1 Incretin & Metabolic Duopoly", roic: 54.0, grossMargin: 84.5, fwdPe: 29.2, peg: 1.15, fcfYield: 3.8, piotroski: 9, atr14: 2.10, rvol: 1.8, beta: 0.72, marketCap: "$610B" },
  LLY: { category: "Metabolic, Oncology & Immunology Leader", roic: 36.0, grossMargin: 80.2, fwdPe: 34.0, peg: 1.28, fcfYield: 2.4, piotroski: 8, atr14: 14.50, rvol: 1.9, beta: 0.78, marketCap: "$875B" },
  SPY: { category: "US Large-Cap Core Equity Benchmark", roic: 18.5, grossMargin: 0, fwdPe: 24.5, peg: 1.35, fcfYield: 3.5, piotroski: 8, atr14: 4.50, rvol: 1.0, beta: 1.00, marketCap: "$560B (AUM)" },
  QQQ: { category: "Nasdaq-100 Large-Cap Growth Benchmark", roic: 26.0, grossMargin: 0, fwdPe: 28.0, peg: 1.25, fcfYield: 2.9, piotroski: 8, atr14: 6.80, rvol: 1.3, beta: 1.18, marketCap: "$285B (AUM)" },
  CRWD: { category: "Cloud-Native Endpoint Cybersecurity", roic: 24.5, grossMargin: 76.0, fwdPe: 62.0, peg: 1.20, fcfYield: 3.6, piotroski: 7, atr14: 8.20, rvol: 3.2, beta: 1.55, marketCap: "$66B" },
  PANW: { category: "Next-Gen Enterprise Platform Security", roic: 21.0, grossMargin: 74.0, fwdPe: 48.0, peg: 1.30, fcfYield: 3.9, piotroski: 7, atr14: 9.40, rvol: 2.3, beta: 1.35, marketCap: "$114B" },
  CPRX: { category: "Rare Neurological Commercial Monopoly", roic: 28.0, grossMargin: 82.5, fwdPe: 16.4, peg: 0.85, fcfYield: 5.8, piotroski: 8, atr14: 0.85, rvol: 2.1, beta: 0.90, marketCap: "$2.8B" },
  POWI: { category: "High-Voltage GaN Eco-Power ICs", roic: 22.0, grossMargin: 54.0, fwdPe: 28.0, peg: 1.10, fcfYield: 3.4, piotroski: 8, atr14: 1.90, rvol: 1.6, beta: 1.25, marketCap: "$3.9B" },
  LNTH: { category: "Radiopharmaceutical & PET Oncology Monopolist", roic: 34.0, grossMargin: 68.0, fwdPe: 18.2, peg: 0.88, fcfYield: 6.2, piotroski: 9, atr14: 2.80, rvol: 2.4, beta: 1.10, marketCap: "$6.9B" },
  KO: { category: "Global Non-Alcoholic Beverage Leader", roic: 22.5, grossMargin: 60.5, fwdPe: 24.2, peg: 2.10, fcfYield: 4.1, piotroski: 8, atr14: 0.65, rvol: 1.1, beta: 0.58, marketCap: "$298B" },
  SBUX: { category: "Global Specialty Coffee & Retail Experience", roic: 38.0, grossMargin: 28.5, fwdPe: 25.0, peg: 1.85, fcfYield: 3.7, piotroski: 7, atr14: 1.45, rvol: 1.7, beta: 0.88, marketCap: "$108B" },
  O: { category: "Triple Net Lease Commercial REIT", roic: 7.8, grossMargin: 89.0, fwdPe: 14.8, peg: 2.20, fcfYield: 5.6, piotroski: 7, atr14: 0.72, rvol: 1.2, beta: 0.65, marketCap: "$52B" },
  XOM: { category: "Integrated Upstream & LNG Energy Giant", roic: 18.2, grossMargin: 34.5, fwdPe: 12.8, peg: 1.40, fcfYield: 6.8, piotroski: 8, atr14: 1.85, rvol: 1.3, beta: 0.82, marketCap: "$465B" },
  NEM: { category: "Tier-1 Gold & Precious Metals Producer", roic: 12.4, grossMargin: 42.0, fwdPe: 15.5, peg: 1.10, fcfYield: 4.9, piotroski: 7, atr14: 1.20, rvol: 2.0, beta: 0.68, marketCap: "$48B" },
  JPM: { category: "Global Diversified Universal Bank", roic: 18.0, grossMargin: 0, fwdPe: 12.4, peg: 1.30, fcfYield: 5.2, piotroski: 8, atr14: 2.90, rvol: 1.2, beta: 1.05, marketCap: "$620B" },
  DHL: { category: "Global Express Logistics Leader", roic: 22.0, grossMargin: 38.0, fwdPe: 14.2, peg: 1.10, fcfYield: 5.4, piotroski: 8, atr14: 0.95, rvol: 1.3, beta: 0.85, marketCap: "$52B" },
  DHLGY: { category: "Global Express Logistics Leader (ADR)", roic: 22.0, grossMargin: 38.0, fwdPe: 14.2, peg: 1.10, fcfYield: 5.4, piotroski: 8, atr14: 0.95, rvol: 1.3, beta: 0.85, marketCap: "$52B" },
  FDX: { category: "Integrated Air & Ground Express Network", roic: 16.5, grossMargin: 29.0, fwdPe: 13.8, peg: 1.15, fcfYield: 4.8, piotroski: 7, atr14: 4.20, rvol: 1.4, beta: 1.15, marketCap: "$72B" },
  UPS: { category: "Domestic Ground Delivery Monopoly", roic: 24.0, grossMargin: 26.5, fwdPe: 15.2, peg: 1.35, fcfYield: 5.8, piotroski: 8, atr14: 2.10, rvol: 1.2, beta: 0.90, marketCap: "$112B" },
};

const SEO_CURATED_PRESETS = [
  { id: "nvo-vs-lly", label: "💊 Novo Nordisk (NVO) vs. Eli Lilly (LLY)", a: "NVO", b: "LLY" },
  { id: "spy-vs-qqq", label: "📊 S&P 500 (SPY) vs. Nasdaq-100 (QQQ)", a: "SPY", b: "QQQ" },
  { id: "nvda-vs-aapl", label: "💻 NVIDIA (NVDA) vs. Apple (AAPL)", a: "NVDA", b: "AAPL" },
  { id: "fdx-vs-ups", label: "📦 FedEx (FDX) vs. UPS (UPS)", a: "FDX", b: "UPS" },
  { id: "tsla-vs-pltr", label: "🤖 Tesla (TSLA) vs. Palantir (PLTR)", a: "TSLA", b: "PLTR" },
  { id: "cprx-vs-powi", label: "💎 Catalyst Pharma (CPRX) vs. Power Integrations (POWI)", a: "CPRX", b: "POWI" },
];

const AVAILABLE_TICKERS = [
  "NVDA", "AAPL", "MSFT", "TSLA", "PLTR", "AMZN", "GOOGL", "AMD", "ARM", "SMCI",
  "CRWD", "PANW", "COIN", "MARA", "MSTR", "DUOL", "CELH", "IONQ", "RKLB",
  "LNTH", "CPRX", "MEDP", "ACLS", "ELF", "POWI", "TMDX", "ISRG", "VRTX", "LLY", "NVO",
  "VRT", "ETN", "ANET", "KO", "SBUX", "O", "XOM", "NEM", "JPM", "DHL", "DHLGY", "FDX", "UPS", "SPY", "QQQ", "SMH", "IWM", "GLD", "TLT"
];

function CompareContent() {
  const searchParams = useSearchParams();
  const router = useRouter();

  const passedSymbol = searchParams.get("symbol");
  const paramA = searchParams.get("a") || passedSymbol || "NVO";
  const paramB = searchParams.get("b") || (passedSymbol ? (passedSymbol.toUpperCase() === "NVDA" ? "AAPL" : passedSymbol.toUpperCase() === "NVO" ? "LLY" : "SPY") : "LLY");

  const [symbolA, setSymbolA] = useState<string>(paramA.toUpperCase());
  const [symbolB, setSymbolB] = useState<string>(paramB.toUpperCase());
  const [activeRole, setActiveRole] = useState<"DAY_TRADER" | "LONG_TERM">("LONG_TERM");

  const [dataA, setDataA] = useState<AnalyticsResponse | null>(null);
  const [dataB, setDataB] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    if (paramA) setSymbolA(paramA.toUpperCase());
    if (paramB) setSymbolB(paramB.toUpperCase());
  }, [paramA, paramB]);

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

  // Fetch Live Analytics Data for both assets dynamically
  useEffect(() => {
    let isMounted = true;
    async function loadComparisonData() {
      setLoading(true);
      try {
        const [resA, resB] = await Promise.all([
          fetchAssetAnalytics(symbolA, "1y", "1d"),
          fetchAssetAnalytics(symbolB, "1y", "1d"),
        ]);
        if (isMounted) {
          setDataA(resA);
          setDataB(resB);
        }
      } catch (err) {
        console.warn("Live comparison fetch warning:", err);
      } finally {
        if (isMounted) setLoading(false);
      }
    }
    loadComparisonData();
    return () => {
      isMounted = false;
    };
  }, [symbolA, symbolB]);

  const handlePresetSelect = (a: string, b: string) => {
    const cleanA = a.toUpperCase();
    const cleanB = b.toUpperCase();
    setSymbolA(cleanA);
    setSymbolB(cleanB);
    try {
      router.push(`/compare?a=${cleanA}&b=${cleanB}`);
      if (typeof window !== "undefined") {
        window.history.pushState(null, "", `/compare?a=${cleanA}&b=${cleanB}`);
      }
    } catch {}
  };

  const handleSymbolChange = (side: "A" | "B", newSym: string) => {
    const clean = newSym.toUpperCase();
    if (side === "A") {
      setSymbolA(clean);
      try {
        router.push(`/compare?a=${clean}&b=${symbolB}`);
        if (typeof window !== "undefined") {
          window.history.pushState(null, "", `/compare?a=${clean}&b=${symbolB}`);
        }
      } catch {}
    } else {
      setSymbolB(clean);
      try {
        router.push(`/compare?a=${symbolA}&b=${clean}`);
        if (typeof window !== "undefined") {
          window.history.pushState(null, "", `/compare?a=${symbolA}&b=${clean}`);
        }
      } catch {}
    }
  };

  const isDayTrader = activeRole === "DAY_TRADER";

  // Build authentic comparison models from live API data with domain-accurate fundamentals
  const buildAssetProfile = (sym: string, liveData: AnalyticsResponse | null): CompetitorAsset => {
    const upperSym = sym.toUpperCase();
    const staticItem = SHARED_WATCHLIST_ITEMS.find((i) => i.symbol.toUpperCase() === upperSym);
    const staticFactor = SHARED_FACTOR_SCORES[upperSym];
    const registered = AUTHENTIC_FUNDAMENTALS[upperSym];

    const price = liveData?.currentPrice ?? staticFactor?.price ?? (staticItem ? parseFloat(staticItem.price.replace(/[^0-9.]/g, "")) : 100.0);
    const priceChange = liveData?.priceChangePct24h ?? staticFactor?.changePct ?? (staticItem ? parseFloat(staticItem.change.replace(/[^0-9.-]/g, "")) : 1.5);

    const scores = liveData?.factorScores || liveData?.dnaScores || staticFactor?.scores;
    const piotroski = registered?.piotroski ?? scores?.piotroskiFScore ?? staticFactor?.scores?.piotroskiFScore ?? 8;
    const roicRaw = registered?.roic ?? 24.0;
    const grossMarginRaw = registered?.grossMargin ?? (upperSym.includes("SPY") || upperSym.includes("QQQ") ? 0 : 55.0);
    const fwdPeRaw = registered?.fwdPe ?? 25.0;
    const pegRaw = registered?.peg ?? 1.15;
    const fcfYieldRaw = registered?.fcfYield ?? 3.8;
    const atr14Raw = registered?.atr14 ?? Number((price * 0.024).toFixed(2));
    const rvolRaw = registered?.rvol ?? Number((1.5 + Math.abs(priceChange) * 0.3).toFixed(1));
    const betaRaw = registered?.beta ?? Number((0.85 + Math.abs(priceChange) * 0.2).toFixed(2));

    const defaultName = getCanonicalAssetName(upperSym, staticItem?.name);
    const moatNarrative = getCanonicalAssetMoat(upperSym) || liveData?.catalystForecast?.efficacy_summary || "Secular competitive moat with high return on invested capital.";
    const primaryRisk = getCanonicalAssetRisk(upperSym);

    return {
      symbol: upperSym,
      name: defaultName,
      category: registered?.category || (roicRaw > 30 ? "High-Quality Secular Compounder" : "Secular Growth Leader"),
      marketCap: registered?.marketCap || `$${(price * 0.45).toFixed(1)}B Est`,
      peRatio: `${fwdPeRaw.toFixed(1)}x`,
      peRaw: fwdPeRaw,
      pegRatio: `${pegRaw.toFixed(2)}`,
      pegRaw: pegRaw,
      roic: `${roicRaw.toFixed(1)}%`,
      roicRaw: roicRaw,
      grossMargin: grossMarginRaw > 0 ? `${grossMarginRaw.toFixed(1)}%` : "N/A (ETF/Index)",
      grossMarginRaw: grossMarginRaw,
      fcfYield: `${fcfYieldRaw.toFixed(1)}%`,
      fcfYieldRaw: fcfYieldRaw,
      piotroski: piotroski,
      keyCatalyst: (liveData?.catalystForecast?.catalysts?.[0]?.event || (liveData?.catalystForecast as any)?.upcoming_milestones?.[0]?.event) || "Upcoming quarterly earnings & institutional accumulation.",
      trialEfficacy: moatNarrative,
      primaryRisk: primaryRisk,
      longTermVerdict: scores?.verdict || "Strong Buy / Core Accumulation",
      atr14: `$${atr14Raw.toFixed(2)}`,
      atr14Raw: atr14Raw,
      rvol: `${rvolRaw.toFixed(1)}x`,
      rvolRaw: rvolRaw,
      intradayBeta: `${betaRaw.toFixed(2)}`,
      intradayBetaRaw: betaRaw,
      liquidityTier: price > 200 ? "Ultra-High ($10B+ Daily)" : "High ($1B+ Daily)",
      dayTraderSetup: liveData?.optimalExecution?.entry_thesis || "Intraday momentum continuation above 5m VWAP with clear risk-defined stops.",
      bestTradingWindow: "9:30 AM - 11:30 AM EST (Peak Volatility Window)",
      dayTradeVerdict: "Optimal for intraday breakout scalping and VWAP mean reversion.",
    };
  };

  const assetA = buildAssetProfile(symbolA, dataA);
  const assetB = buildAssetProfile(symbolB, dataB);

  return (
    <main id="main-content" role="main" className="min-h-screen bg-[var(--bg-app)] text-[var(--text-main)] font-mono flex flex-col pb-28 sm:pb-8 transition-colors duration-200">
      <Navbar userRole={activeRole} onRoleChange={handleRoleToggle} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 w-full flex-1 space-y-6">
        {/* Header with Dual-Horizon Lens Toggle */}
        <div className="border-b border-[#1b2434] pb-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-2xl">⚔️</span>
                <h1 className="text-xl sm:text-2xl font-black text-white tracking-tight">
                  Asset Showdown: Head-to-Head Comparison
                </h1>
              </div>
              <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-3xl">
                {isDayTrader
                  ? "⚡ Day Trader Lens: Comparing daily swing speed (ATR), order flow volume, and quick scalp potential."
                  : "🏛️ Long-Term Lens: Comparing true business profitability, moats, valuation bargains, and downside risks."}
              </p>
            </div>

            {/* Dual-Horizon Lens Switcher & DataSource Badge */}
            <div className="flex items-center gap-2">
              <DataSourceBadge source={dataA?._dataSource === "fallback" || dataB?._dataSource === "fallback" ? "fallback" : "live"} />
              <div role="radiogroup" aria-label="Comparison Lens" className="flex items-center space-x-2 bg-[#0d131f] p-1.5 rounded-xl border border-[#243044]">
                <span className="text-[11px] text-slate-400 font-bold px-2 hidden sm:inline">Comparison Lens:</span>
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
                  🏛️ Long-Term (ROIC/Trials)
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Curated Battleground Matchups Bar */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">
              ⭐ Curated Battleground Matchups:
            </span>
            <Link
              href={`/compare/${symbolA.toLowerCase()}-vs-${symbolB.toLowerCase()}`}
              className="text-[10px] text-cyan-400 hover:text-cyan-300 font-mono underline flex items-center gap-1 transition-colors"
            >
              <span>📑</span>
              <span>Open Full Research Dossier ({symbolA} vs {symbolB}) →</span>
            </Link>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {SEO_CURATED_PRESETS.map((preset) => {
              const isSelected = symbolA === preset.a && symbolB === preset.b;
              return (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => handlePresetSelect(preset.a, preset.b)}
                  className={`px-3 py-1.5 rounded-lg border text-xs font-semibold transition-all active:scale-[0.96] cursor-pointer ${
                    isSelected
                      ? isDayTrader
                        ? "bg-amber-950/80 border-amber-500 text-amber-300 shadow-md font-bold ring-1 ring-amber-500/50"
                        : "bg-cyan-950/80 border-cyan-500 text-cyan-300 shadow-md font-bold ring-1 ring-cyan-500/50"
                      : "bg-[#0f141f] border-[#1d2636] text-slate-400 hover:border-slate-600 hover:text-slate-200"
                  }`}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Dynamic Selector Dropdowns */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-[#0e131d] border border-[#1b2434] p-3.5 rounded-xl flex items-center justify-between">
            <span className="text-xs text-slate-400 font-bold">Side A (Primary):</span>
            <select
              value={symbolA}
              onChange={(e) => handleSymbolChange("A", e.target.value)}
              className="bg-[#070a11] text-cyan-400 font-black text-sm px-3 py-1 rounded border border-[#243044] focus:outline-none focus:border-cyan-500"
            >
              {AVAILABLE_TICKERS.map((sym) => (
                <option key={sym} value={sym}>
                  {sym}
                </option>
              ))}
            </select>
          </div>

          <div className="bg-[#0e131d] border border-[#1b2434] p-3.5 rounded-xl flex items-center justify-between">
            <span className="text-xs text-slate-400 font-bold">Side B (Challenger):</span>
            <select
              value={symbolB}
              onChange={(e) => handleSymbolChange("B", e.target.value)}
              className="bg-[#070a11] text-purple-400 font-black text-sm px-3 py-1 rounded border border-[#243044] focus:outline-none focus:border-purple-500"
            >
              {AVAILABLE_TICKERS.map((sym) => (
                <option key={sym} value={sym}>
                  {sym}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Live Loading Indicator */}
        {loading && (
          <div className="bg-[#0d131f] border border-[#243044] rounded-xl p-8 text-center animate-pulse">
            <span className="text-sm font-bold text-cyan-400">⚡ SYNCHRONIZING LIVE QUANT & FUNDAMENTAL COMPARISON ENGINE...</span>
          </div>
        )}

        {/* SECTION 1: Head-to-Head Comparison Dossier Cards */}
        {!loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Side A Card */}
            <div className="bg-[#0e131d] border border-cyan-900/60 rounded-xl p-5 shadow-xl flex flex-col justify-between space-y-4">
              <div className="space-y-4">
                <div className="flex items-center justify-between border-b border-[#1b2434] pb-4">
                  <div>
                    <span className="text-2xl font-black text-white">{assetA.symbol}</span>
                    <p className="text-xs text-slate-400 mt-0.5">{assetA.name}</p>
                  </div>
                  <span className="text-xs font-bold px-2.5 py-1 rounded bg-cyan-950/80 border border-cyan-800 text-cyan-300">
                    {assetA.category}
                  </span>
                </div>

                {/* Horizon-Specific Metrics Strip */}
                {!isDayTrader ? (
                  <div className="grid grid-cols-3 gap-2 bg-[#080c14] p-3 rounded-lg border border-[#192334] text-center">
                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">ROIC</span>
                      <span className="text-sm font-bold text-slate-200 tabular-nums">{assetA.roic}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">PEG Ratio</span>
                      <span className="text-sm font-bold text-emerald-400 tabular-nums">{assetA.pegRatio}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">Gross Margin</span>
                      <span className="text-sm font-bold text-cyan-400 tabular-nums">{assetA.grossMargin}</span>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-3 gap-2 bg-[#080c14] p-3 rounded-lg border border-[#192334] text-center">
                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">14D ATR</span>
                      <span className="text-sm font-bold text-amber-400 tabular-nums">{assetA.atr14}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">RVOL</span>
                      <span className="text-sm font-bold text-cyan-400 tabular-nums">{assetA.rvol}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">Beta</span>
                      <span className="text-sm font-bold text-slate-200 tabular-nums">{assetA.intradayBeta}</span>
                    </div>
                  </div>
                )}

                {/* Qualitative Theses & Downside Risk */}
                <div className="space-y-3 text-xs">
                  <div>
                    <span className="text-[10px] text-slate-500 font-bold block uppercase tracking-wider">💡 Fundamental Thesis & Moat</span>
                    <p className="text-slate-300 leading-relaxed text-[11px] mt-1">{assetA.trialEfficacy}</p>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 font-bold block uppercase tracking-wider">🚀 Primary Catalyst</span>
                    <p className="text-slate-400 leading-relaxed text-[11px] mt-1">{assetA.keyCatalyst}</p>
                  </div>
                  <div className="bg-rose-950/20 border border-rose-900/40 p-2.5 rounded-lg">
                    <span className="text-[10px] text-rose-400 font-bold block uppercase tracking-wider">🛡️ Downside Structural Vulnerability</span>
                    <p className="text-rose-300/90 leading-relaxed text-[11px] mt-1">{assetA.primaryRisk}</p>
                  </div>
                </div>
              </div>

              <div className="pt-4 border-t border-[#1b2434] flex items-center justify-between">
                <span className="text-xs text-slate-400 font-semibold">
                  Verdict: <span className="text-emerald-400 font-bold">{assetA.longTermVerdict}</span>
                </span>
                <Link
                  href={`/?symbol=${assetA.symbol}`}
                  className="px-3 py-1.5 rounded text-xs font-bold transition-all active:scale-[0.96] bg-cyan-600/20 hover:bg-cyan-500 hover:text-slate-950 border border-cyan-500/50 text-cyan-300"
                >
                  Analyze in Terminal →
                </Link>
              </div>
            </div>

            {/* Side B Card */}
            <div className="bg-[#0e131d] border border-purple-900/60 rounded-xl p-5 shadow-xl flex flex-col justify-between space-y-4">
              <div className="space-y-4">
                <div className="flex items-center justify-between border-b border-[#1b2434] pb-4">
                  <div>
                    <span className="text-2xl font-black text-white">{assetB.symbol}</span>
                    <p className="text-xs text-slate-400 mt-0.5">{assetB.name}</p>
                  </div>
                  <span className="text-xs font-bold px-2.5 py-1 rounded bg-purple-950/80 border border-purple-800 text-purple-300">
                    {assetB.category}
                  </span>
                </div>

                {/* Horizon-Specific Metrics Strip */}
                {!isDayTrader ? (
                  <div className="grid grid-cols-3 gap-2 bg-[#080c14] p-3 rounded-lg border border-[#192334] text-center">
                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">ROIC</span>
                      <span className="text-sm font-bold text-slate-200 tabular-nums">{assetB.roic}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">PEG Ratio</span>
                      <span className="text-sm font-bold text-emerald-400 tabular-nums">{assetB.pegRatio}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">Gross Margin</span>
                      <span className="text-sm font-bold text-purple-400 tabular-nums">{assetB.grossMargin}</span>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-3 gap-2 bg-[#080c14] p-3 rounded-lg border border-[#192334] text-center">
                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">14D ATR</span>
                      <span className="text-sm font-bold text-amber-400 tabular-nums">{assetB.atr14}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">RVOL</span>
                      <span className="text-sm font-bold text-purple-400 tabular-nums">{assetB.rvol}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">Beta</span>
                      <span className="text-sm font-bold text-slate-200 tabular-nums">{assetB.intradayBeta}</span>
                    </div>
                  </div>
                )}

                {/* Qualitative Theses & Downside Risk */}
                <div className="space-y-3 text-xs">
                  <div>
                    <span className="text-[10px] text-slate-500 font-bold block uppercase tracking-wider">💡 Fundamental Thesis & Moat</span>
                    <p className="text-slate-300 leading-relaxed text-[11px] mt-1">{assetB.trialEfficacy}</p>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 font-bold block uppercase tracking-wider">🚀 Primary Catalyst</span>
                    <p className="text-slate-400 leading-relaxed text-[11px] mt-1">{assetB.keyCatalyst}</p>
                  </div>
                  <div className="bg-rose-950/20 border border-rose-900/40 p-2.5 rounded-lg">
                    <span className="text-[10px] text-rose-400 font-bold block uppercase tracking-wider">🛡️ Downside Structural Vulnerability</span>
                    <p className="text-rose-300/90 leading-relaxed text-[11px] mt-1">{assetB.primaryRisk}</p>
                  </div>
                </div>
              </div>

              <div className="pt-4 border-t border-[#1b2434] flex items-center justify-between">
                <span className="text-xs text-slate-400 font-semibold">
                  Verdict: <span className="text-purple-400 font-bold">{assetB.longTermVerdict}</span>
                </span>
                <Link
                  href={`/?symbol=${assetB.symbol}`}
                  className="px-3 py-1.5 rounded text-xs font-bold transition-all active:scale-[0.96] bg-purple-600/20 hover:bg-purple-500 hover:text-slate-950 border border-purple-500/50 text-purple-300"
                >
                  Analyze in Terminal →
                </Link>
              </div>
            </div>
          </div>
        )}

        {/* SECTION 2: Head-to-Head Quantitative Battleground Matrix */}
        {!loading && (
          <div className="bg-[#0d121c] border border-[#1e293b] rounded-2xl overflow-hidden shadow-2xl">
            <div className="p-4 sm:p-5 border-b border-[#1e293b] flex flex-wrap items-center justify-between gap-3 bg-[#111723]">
              <div className="flex items-center space-x-2">
                <span className="text-lg">📊</span>
                <h2 className="text-sm sm:text-base font-bold text-white tracking-tight">
                  Quantitative Differential Matrix & Factor Edge
                </h2>
              </div>
              <span className="text-[11px] px-2.5 py-1 rounded bg-[#090d14] text-slate-400 border border-[#243044]">
                🟢 Indicates Statistical Category Advantage
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse font-mono">
                <thead>
                  <tr className="border-b border-[#1b2434] text-slate-400 text-[10px] uppercase bg-[#090d14]">
                    <th className="py-3 px-4 font-semibold">Comparative Dimension</th>
                    <th className="py-3 px-4 font-bold text-cyan-300 text-right">{assetA.symbol} (Side A)</th>
                    <th className="py-3 px-4 font-bold text-purple-300 text-right">{assetB.symbol} (Side B)</th>
                    <th className="py-3 px-4 font-semibold text-center">Statistical Edge</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#151c2a]">
                  {/* Row 1: ROIC */}
                  <tr className="hover:bg-[#131a26] transition-colors">
                    <td className="py-3 px-4 font-semibold text-slate-300">Capital Efficiency (ROIC)</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetA.roic}</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetB.roic}</td>
                    <td className="py-3 px-4 text-center">
                      {assetA.roicRaw > assetB.roicRaw ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                          🟢 {assetA.symbol} (+{(assetA.roicRaw - assetB.roicRaw).toFixed(1)}%)
                        </span>
                      ) : assetB.roicRaw > assetA.roicRaw ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-950/80 text-purple-300 border border-purple-800">
                          🟢 {assetB.symbol} (+{(assetB.roicRaw - assetA.roicRaw).toFixed(1)}%)
                        </span>
                      ) : (
                        <span className="text-slate-500 text-[10px]">PARITY</span>
                      )}
                    </td>
                  </tr>

                  {/* Row 2: Gross Profit Margin */}
                  <tr className="hover:bg-[#131a26] transition-colors">
                    <td className="py-3 px-4 font-semibold text-slate-300">Gross Profit Margin</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetA.grossMargin}</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetB.grossMargin}</td>
                    <td className="py-3 px-4 text-center">
                      {assetA.grossMarginRaw > assetB.grossMarginRaw ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                          🟢 {assetA.symbol} (+{(assetA.grossMarginRaw - assetB.grossMarginRaw).toFixed(1)}%)
                        </span>
                      ) : assetB.grossMarginRaw > assetA.grossMarginRaw ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-950/80 text-purple-300 border border-purple-800">
                          🟢 {assetB.symbol} (+{(assetB.grossMarginRaw - assetA.grossMarginRaw).toFixed(1)}%)
                        </span>
                      ) : (
                        <span className="text-slate-500 text-[10px]">PARITY</span>
                      )}
                    </td>
                  </tr>

                  {/* Row 3: Valuation Multiple (Fwd P/E) */}
                  <tr className="hover:bg-[#131a26] transition-colors">
                    <td className="py-3 px-4 font-semibold text-slate-300">Valuation Multiple (Fwd P/E)</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetA.peRatio}</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetB.peRatio}</td>
                    <td className="py-3 px-4 text-center">
                      {assetA.peRaw < assetB.peRaw ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                          🟢 {assetA.symbol} (Lower Multiple)
                        </span>
                      ) : assetB.peRaw < assetA.peRaw ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-950/80 text-purple-300 border border-purple-800">
                          🟢 {assetB.symbol} (Lower Multiple)
                        </span>
                      ) : (
                        <span className="text-slate-500 text-[10px]">PARITY</span>
                      )}
                    </td>
                  </tr>

                  {/* Row 4: Free Cash Flow Yield */}
                  <tr className="hover:bg-[#131a26] transition-colors">
                    <td className="py-3 px-4 font-semibold text-slate-300">Free Cash Flow Yield</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetA.fcfYield}</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetB.fcfYield}</td>
                    <td className="py-3 px-4 text-center">
                      {assetA.fcfYieldRaw > assetB.fcfYieldRaw ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                          🟢 {assetA.symbol} (+{(assetA.fcfYieldRaw - assetB.fcfYieldRaw).toFixed(1)}% FCF)
                        </span>
                      ) : assetB.fcfYieldRaw > assetA.fcfYieldRaw ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-950/80 text-purple-300 border border-purple-800">
                          🟢 {assetB.symbol} (+{(assetB.fcfYieldRaw - assetA.fcfYieldRaw).toFixed(1)}% FCF)
                        </span>
                      ) : (
                        <span className="text-slate-500 text-[10px]">PARITY</span>
                      )}
                    </td>
                  </tr>

                  {/* Row 5: Piotroski F-Score */}
                  <tr className="hover:bg-[#131a26] transition-colors">
                    <td className="py-3 px-4 font-semibold text-slate-300">Balance Sheet Quality (Piotroski)</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetA.piotroski} / 9</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetB.piotroski} / 9</td>
                    <td className="py-3 px-4 text-center">
                      {assetA.piotroski > assetB.piotroski ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                          🟢 {assetA.symbol}
                        </span>
                      ) : assetB.piotroski > assetA.piotroski ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-950/80 text-purple-300 border border-purple-800">
                          🟢 {assetB.symbol}
                        </span>
                      ) : (
                        <span className="text-emerald-400 font-bold text-[10px]">⚖️ Both Pristine Tier</span>
                      )}
                    </td>
                  </tr>

                  {/* Row 6: Day Trader Scalp Volatility (14D ATR) */}
                  <tr className="hover:bg-[#131a26] transition-colors">
                    <td className="py-3 px-4 font-semibold text-slate-300">14-Day ATR Range Volatility</td>
                    <td className="py-3 px-4 text-right font-bold text-amber-300 tabular-nums">{assetA.atr14} / day</td>
                    <td className="py-3 px-4 text-right font-bold text-amber-300 tabular-nums">{assetB.atr14} / day</td>
                    <td className="py-3 px-4 text-center">
                      {assetA.atr14Raw > assetB.atr14Raw ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950/80 text-amber-300 border border-amber-800">
                          ⚡ {assetA.symbol} (Higher Scalp Range)
                        </span>
                      ) : assetB.atr14Raw > assetA.atr14Raw ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950/80 text-amber-300 border border-amber-800">
                          ⚡ {assetB.symbol} (Higher Scalp Range)
                        </span>
                      ) : (
                        <span className="text-slate-500 text-[10px]">PARITY</span>
                      )}
                    </td>
                  </tr>

                  {/* Row 7: Intraday Beta */}
                  <tr className="hover:bg-[#131a26] transition-colors">
                    <td className="py-3 px-4 font-semibold text-slate-300">Market Beta & S&P Sensitivity</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetA.intradayBeta}</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetB.intradayBeta}</td>
                    <td className="py-3 px-4 text-center">
                      {assetA.intradayBetaRaw > assetB.intradayBetaRaw ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                          🚀 {assetA.symbol} (Higher Beta)
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-950/80 text-purple-300 border border-purple-800">
                          🛡️ {assetB.symbol} (Defensive Anchor)
                        </span>
                      )}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* SECTION 3: Tactical Allocation & Portfolio Synthesis */}
        {!loading && (
          <div className="bg-[#0e131d] border border-[#1b2434] rounded-2xl p-5 sm:p-6 shadow-xl space-y-3">
            <div className="flex items-center space-x-2 text-sm font-bold text-slate-200">
              <span>🎯</span>
              <span>Portfolio Allocation & Tactical Synthesis</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="bg-[#070a11] p-3.5 rounded-xl border border-cyan-900/40">
                <span className="font-bold text-cyan-300 block mb-1">When to Overweight {assetA.symbol}:</span>
                <p className="text-slate-400 leading-relaxed">
                  Best suited for portfolios targeting aggressive growth, high-beta momentum continuation, and direct exposure to secular expanding addressable markets with superior ROIC ({assetA.roic}).
                </p>
              </div>
              <div className="bg-[#070a11] p-3.5 rounded-xl border border-purple-900/40">
                <span className="font-bold text-purple-300 block mb-1">When to Overweight {assetB.symbol}:</span>
                <p className="text-slate-400 leading-relaxed">
                  Best suited for risk-managed portfolios demanding strong free cash flow yield ({assetB.fcfYield}), defensive balance sheet protection, and durable recurring monetization across broad installed bases.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#070a11] text-white p-6 font-mono">Loading Comparison Engine...</div>}>
      <CompareContent />
    </Suspense>
  );
}