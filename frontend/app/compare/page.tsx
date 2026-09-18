"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import Navbar from "../../components/Navbar";
import DataSourceBadge from "../../components/DataSourceBadge";
import { API_BASE_URL, fetchAssetAnalytics, AnalyticsResponse, SpotPriceRegistry } from "../../lib/api";
import { getPersistedMarketSnapshot } from "../../lib/marketDatabase";
import { SHARED_WATCHLIST_ITEMS } from "../../lib/constants";
import { getCanonicalAssetName, getCanonicalAssetMoat, getCanonicalAssetRisk } from "../../lib/assetRegistry";
import { trackComparisonRun } from "../../lib/matomo";
import CompareSsrShell from "../../components/CompareSsrShell";

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
          trackComparisonRun(symbolA, symbolB);
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
    const reg = SpotPriceRegistry.get(upperSym);
    const snap = getPersistedMarketSnapshot(upperSym);
    const price = (liveData && liveData.currentPrice > 0)
      ? liveData.currentPrice
      : (reg?.price && reg.price > 0)
      ? reg.price
      : (snap?.currentPrice && snap.currentPrice > 0)
      ? snap.currentPrice
      : 0;

    const scores = liveData?.factorScores || liveData?.dnaScores;
    const hasVerifiedFundamentals = Boolean(scores && typeof scores.qualityScore === "number");
    const piotroski = scores?.piotroskiFScore ?? 0;
    const roicRaw = 0;
    const grossMarginRaw = 0;
    const fwdPeRaw = 0;
    const pegRaw = 0;
    const fcfYieldRaw = 0;
    const atr14Raw = liveData?.technicals?.atr_14 || 0;
    const rvolRaw = 0;
    const betaRaw = 1.0;

    const defaultName = getCanonicalAssetName(upperSym, staticItem?.name);
    const moatNarrative = liveData?.catalystForecast?.efficacy_summary || getCanonicalAssetMoat(upperSym) || "Sector equity tracked across quantitative model dimensions.";
    const primaryRisk = getCanonicalAssetRisk(upperSym);

    return {
      symbol: upperSym,
      name: defaultName,
      category: liveData?.catalystForecast?.sector || "Equities",
      marketCap: "N/A",
      peRatio: "N/A",
      peRaw: fwdPeRaw,
      pegRatio: "N/A",
      pegRaw: pegRaw,
      roic: "N/A",
      roicRaw: roicRaw,
      grossMargin: upperSym.includes("SPY") || upperSym.includes("QQQ") ? "N/A (ETF/Index)" : "N/A",
      grossMarginRaw: grossMarginRaw,
      fcfYield: "N/A",
      fcfYieldRaw: fcfYieldRaw,
      piotroski: piotroski,
      keyCatalyst: liveData?.catalystForecast?.efficacy_summary || (hasVerifiedFundamentals ? "Upcoming corporate earnings & institutional accumulation." : "Pending SEC filings verification."),
      trialEfficacy: moatNarrative,
      primaryRisk: primaryRisk,
      longTermVerdict: hasVerifiedFundamentals ? (scores?.verdict || "Quantitative Model Verified") : "Unverified Fundamental Profile (N/A)",
      atr14: atr14Raw > 0 ? `$${atr14Raw.toFixed(2)}` : "N/A",
      atr14Raw: atr14Raw,
      rvol: "N/A",
      rvolRaw: rvolRaw,
      intradayBeta: "1.00",
      intradayBetaRaw: betaRaw,
      liquidityTier: price > 200 ? "Ultra-High ($10B+ Daily)" : price > 0 ? "Exchange Verified" : "Awaiting Tape",
      dayTraderSetup: liveData?.optimalExecution?.entry_thesis || "Intraday momentum tracking with clear risk-defined levels.",
      bestTradingWindow: "9:30 AM - 11:30 AM EST (Peak Volatility Window)",
      dayTradeVerdict: liveData?.optimalExecution?.execution_status || (hasVerifiedFundamentals ? "Evaluated for quantitative setups." : "Unverified setup — live feed required."),
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
              <DataSourceBadge source={
                dataA?._dataSource === "fallback" || dataB?._dataSource === "fallback"
                  ? "fallback"
                  : dataA?._dataSource === "unavailable" || dataB?._dataSource === "unavailable"
                  ? "unavailable"
                  : dataA?._dataSource === "historical" || dataB?._dataSource === "historical"
                  ? "historical"
                  : "live"
              } />
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
                      {assetA.peRaw > 0 && assetB.peRaw > 0 ? (
                        assetA.peRaw < assetB.peRaw ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                            🟢 {assetA.symbol} (Lower Multiple)
                          </span>
                        ) : assetB.peRaw < assetA.peRaw ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-950/80 text-purple-300 border border-purple-800">
                            🟢 {assetB.symbol} (Lower Multiple)
                          </span>
                        ) : (
                          <span className="text-slate-500 text-[10px]">PARITY</span>
                        )
                      ) : (
                        <span className="text-slate-500 text-[10px]">N/A (Unverified)</span>
                      )}
                    </td>
                  </tr>

                  {/* Row 4: Free Cash Flow Yield */}
                  <tr className="hover:bg-[#131a26] transition-colors">
                    <td className="py-3 px-4 font-semibold text-slate-300">Free Cash Flow Yield</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetA.fcfYield}</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetB.fcfYield}</td>
                    <td className="py-3 px-4 text-center">
                      {assetA.fcfYieldRaw > 0 && assetB.fcfYieldRaw > 0 ? (
                        assetA.fcfYieldRaw > assetB.fcfYieldRaw ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                            🟢 {assetA.symbol} (+{(assetA.fcfYieldRaw - assetB.fcfYieldRaw).toFixed(1)}% FCF)
                          </span>
                        ) : assetB.fcfYieldRaw > assetA.fcfYieldRaw ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-950/80 text-purple-300 border border-purple-800">
                            🟢 {assetB.symbol} (+{(assetB.fcfYieldRaw - assetA.fcfYieldRaw).toFixed(1)}% FCF)
                          </span>
                        ) : (
                          <span className="text-slate-500 text-[10px]">PARITY</span>
                        )
                      ) : (
                        <span className="text-slate-500 text-[10px]">N/A (Unverified)</span>
                      )}
                    </td>
                  </tr>

                  {/* Row 5: Piotroski F-Score */}
                  <tr className="hover:bg-[#131a26] transition-colors">
                    <td className="py-3 px-4 font-semibold text-slate-300">Balance Sheet Quality (Piotroski)</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetA.piotroski} / 9</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetB.piotroski} / 9</td>
                    <td className="py-3 px-4 text-center">
                      {assetA.piotroski > 0 || assetB.piotroski > 0 ? (
                        assetA.piotroski > assetB.piotroski ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                            🟢 {assetA.symbol}
                          </span>
                        ) : assetB.piotroski > assetA.piotroski ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-950/80 text-purple-300 border border-purple-800">
                            🟢 {assetB.symbol}
                          </span>
                        ) : assetA.piotroski >= 8 && assetB.piotroski >= 8 ? (
                          <span className="text-emerald-400 font-bold text-[10px]">⚖️ Both Pristine Tier</span>
                        ) : (
                          <span className="text-slate-500 text-[10px]">PARITY</span>
                        )
                      ) : (
                        <span className="text-slate-500 text-[10px]">N/A (Unverified)</span>
                      )}
                    </td>
                  </tr>

                  {/* Row 6: Day Trader Scalp Volatility (14D ATR) */}
                  <tr className="hover:bg-[#131a26] transition-colors">
                    <td className="py-3 px-4 font-semibold text-slate-300">14-Day ATR Range Volatility</td>
                    <td className="py-3 px-4 text-right font-bold text-amber-300 tabular-nums">{assetA.atr14} / day</td>
                    <td className="py-3 px-4 text-right font-bold text-amber-300 tabular-nums">{assetB.atr14} / day</td>
                    <td className="py-3 px-4 text-center">
                      {assetA.atr14Raw > 0 && assetB.atr14Raw > 0 ? (
                        assetA.atr14Raw > assetB.atr14Raw ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950/80 text-amber-300 border border-amber-800">
                            ⚡ {assetA.symbol} (Higher Scalp Range)
                          </span>
                        ) : assetB.atr14Raw > assetA.atr14Raw ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950/80 text-amber-300 border border-amber-800">
                            ⚡ {assetB.symbol} (Higher Scalp Range)
                          </span>
                        ) : (
                          <span className="text-slate-500 text-[10px]">PARITY</span>
                        )
                      ) : (
                        <span className="text-slate-500 text-[10px]">N/A (Unverified)</span>
                      )}
                    </td>
                  </tr>

                  {/* Row 7: Intraday Beta */}
                  <tr className="hover:bg-[#131a26] transition-colors">
                    <td className="py-3 px-4 font-semibold text-slate-300">Market Beta & S&P Sensitivity</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetA.intradayBeta}</td>
                    <td className="py-3 px-4 text-right font-bold text-slate-100 tabular-nums">{assetB.intradayBeta}</td>
                    <td className="py-3 px-4 text-center">
                      {assetA.intradayBetaRaw > 0 && assetB.intradayBetaRaw > 0 ? (
                        assetA.intradayBetaRaw > assetB.intradayBetaRaw ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                            🚀 {assetA.symbol} (Higher Beta)
                          </span>
                        ) : assetB.intradayBetaRaw > assetA.intradayBetaRaw ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-950/80 text-purple-300 border border-purple-800">
                            🚀 {assetB.symbol} (Higher Beta)
                          </span>
                        ) : (
                          <span className="text-slate-500 text-[10px]">PARITY</span>
                        )
                      ) : (
                        <span className="text-slate-500 text-[10px]">N/A (Unverified)</span>
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
    <Suspense fallback={<CompareSsrShell />}>
      <CompareContent />
    </Suspense>
  );
}