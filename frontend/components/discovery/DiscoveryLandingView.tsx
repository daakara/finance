"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Search, Compass, TrendingUp, ShieldCheck, Activity, BarChart2, ArrowRight } from "lucide-react";
import { FredMacroData } from "../../lib/institutionalFeeds";
import { resolveFreshCachedVix } from "../../lib/preflightEvaluator";
import { useAssetDiscovery } from "../../hooks/useAssetDiscovery";
import { resolveDiscoveredSymbol, DISCOVERY_SYMBOL, type DiscoveredAsset } from "../../lib/assetDiscovery";
import { resolveAssetAlias } from "../../lib/assetRegistry";
import IntentHero from "../IntentHero";
import WeeklyConfluenceSpotlight from "../WeeklyConfluenceSpotlight";

export interface DiscoveryLandingViewProps {
  onSelectSymbol: (symbol: string) => void;
  macroData: FredMacroData | null;
  userRole?: "DAY_TRADER" | "LONG_TERM";
  onRoleChange?: (role: "DAY_TRADER" | "LONG_TERM") => void;
}

const POPULAR_ASSETS = [
  { symbol: "NVDA", name: "NVIDIA Corp", sector: "Semiconductors & AI", type: "EQUITY" },
  { symbol: "AAPL", name: "Apple Inc", sector: "Consumer Electronics", type: "EQUITY" },
  { symbol: "MSFT", name: "Microsoft Corp", sector: "Enterprise Cloud & AI", type: "EQUITY" },
  { symbol: "SPY", name: "SPDR S&P 500 ETF", sector: "Broad Market Benchmark", type: "ETF" },
  { symbol: "QQQ", name: "Invesco QQQ Trust", sector: "Nasdaq 100 Benchmark", type: "ETF" },
  { symbol: "TSLA", name: "Tesla Inc", sector: "EV & Autonomous Tech", type: "EQUITY" },
];

export default function DiscoveryLandingView({
  onSelectSymbol,
  macroData,
  userRole = "LONG_TERM",
  onRoleChange,
}: DiscoveryLandingViewProps) {
  const [searchInput, setSearchInput] = useState("");
  const freshVix = resolveFreshCachedVix();

  const cleanQ = searchInput.trim().toUpperCase();
  const aliasMatch = resolveAssetAlias(cleanQ);
  const discovery = useAssetDiscovery(searchInput, Boolean(searchInput.trim()));

  const handleSelectTicker = (sym: string) => {
    const cleanSym = sym.trim().toUpperCase();
    if (!cleanSym) return;
    onSelectSymbol(cleanSym);
    setSearchInput("");
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (discovery.status === "LOADING" && !aliasMatch) return;
    if (cleanQ) {
      if (aliasMatch) {
        handleSelectTicker(aliasMatch.canonicalTicker);
      } else if (resolveDiscoveredSymbol(cleanQ, discovery.results)) {
        handleSelectTicker(resolveDiscoveredSymbol(cleanQ, discovery.results)!);
      } else if (discovery.results.length > 0) {
        return; // Ambiguous company matches require explicit selection.
      } else if (DISCOVERY_SYMBOL.test(cleanQ)) {
        handleSelectTicker(cleanQ);
      }
    }
  };

  const handleOpenGlobalOmni = () => {
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("arx:open-omni-search"));
    }
  };

  return (
    <div data-testid="discovery-landing-view" className="space-y-6 sm:space-y-8 animate-fadeIn">
      {/* ── 1. Hero Search & Command Deck ────────────────────────────────────── */}
      <section
        aria-label="Market Search and Asset Discovery"
        className="bg-[var(--surface-1)] border border-[var(--border-subtle)] rounded-2xl p-5 sm:p-8 shadow-2xl space-y-5 text-slate-100 relative overflow-hidden"
      >
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-3xl space-y-2">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-cyan-950/80 border border-cyan-800/60 text-xs font-mono font-bold text-cyan-300">
            <Compass className="w-3.5 h-3.5" aria-hidden="true" />
            <span>DISCOVERY WORKSPACE · NO DEFAULT ASSET ASSUMED</span>
          </div>
          <h1 className="text-2xl sm:text-4xl font-black text-white tracking-tight leading-tight">
            ARX Quantitative Intelligence & Market Discovery
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 font-sans leading-relaxed">
            Real-time execution clearance, multi-model factor confluence, and institutional capital tracking. Search any equity, ETF, or asset to launch its quantitative cockpit.
          </p>
        </div>

        {/* Primary Search Input Form */}
        <form onSubmit={handleSearchSubmit} className="relative max-w-2xl">
          <div className="relative flex items-center">
            <Search className="absolute left-4 w-5 h-5 text-slate-400 pointer-events-none" aria-hidden="true" />
            <input
              type="text"
              id="landing-search-input"
              data-testid="landing-search-input"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder="Search ticker, company name, or ETF (e.g. NVDA, SPY, Apple)..."
              aria-label="Search ticker, company name, or ETF"
              className="w-full pl-12 pr-32 py-3.5 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)] focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/30 text-sm font-sans text-white placeholder-slate-400 transition-all shadow-inner outline-none min-h-[48px]"
            />
            <div className="absolute right-1.5 flex items-center gap-1.5">
              <button
                type="submit"
                disabled={!cleanQ || (discovery.status === "LOADING" && !aliasMatch)}
                aria-label={`Submit search for ${cleanQ || "asset"}`}
                className="px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:opacity-40 disabled:hover:bg-cyan-500 disabled:cursor-not-allowed text-slate-950 font-bold text-xs transition-all cursor-pointer min-h-[44px] flex items-center justify-center focus-visible:ring-2 focus-visible:ring-cyan-300 focus-visible:outline-none"
              >
                Analyze
              </button>
            </div>
          </div>

          {cleanQ.length >= 2 && (
            <div aria-live="polite" className="mt-2 px-1 text-xs font-mono text-slate-300">
              {discovery.status === "LOADING" && "Searching listed assets…"}
              {discovery.status === "UNAVAILABLE" && "Live search unavailable. Saved matches and direct ticker lookup remain available."}
              {discovery.status === "AVAILABLE" && discovery.results.length === 0 && "No provider matches. You can still look up a ticker directly."}
            </div>
          )}

          {aliasMatch && (
            <div className="mt-2 p-3 rounded-xl bg-cyan-950/70 border border-cyan-500/70 flex items-center justify-between">
              <div className="text-xs">
                <span className="text-cyan-300 font-bold">Did you mean: </span>
                <span className="text-white font-semibold">{aliasMatch.companyName} ({aliasMatch.canonicalTicker})</span>
              </div>
              <button
                type="button"
                onClick={() => handleSelectTicker(aliasMatch.canonicalTicker)}
                className="px-3 py-1 bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-bold rounded-lg transition-transform active:scale-95 cursor-pointer"
              >
                Analyze {aliasMatch.canonicalTicker} →
              </button>
            </div>
          )}

          {discovery.results.length > 0 && (
            <div className="mt-2 p-2 bg-[#0b1019] border border-cyan-800/80 rounded-xl space-y-1.5 shadow-xl">
              <div className="px-2 py-1 text-[11px] font-mono text-slate-400 border-b border-[#1b2537] flex items-center justify-between">
                <span>Live US-listed asset matches</span>
                <span className="text-cyan-400 text-[10px]">Analysis availability checked when opened</span>
              </div>
              <div className="space-y-1 max-h-56 overflow-y-auto pr-1" role="listbox">
                {discovery.results.map((item) => (
                  <button
                    key={`landing-discovery-${item.symbol}`}
                    type="button"
                    onClick={() => handleSelectTicker(item.symbol)}
                    className="w-full text-left p-2.5 rounded-lg bg-[#070a10] hover:bg-[#131b29] border border-[#1b2537] hover:border-cyan-500/80 transition-all flex items-center justify-between group cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none"
                  >
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono font-black text-cyan-300 group-hover:text-cyan-200">{item.symbol}</span>
                        <span className="text-xs text-white group-hover:text-cyan-100 font-medium">{item.name}</span>
                      </div>
                      <span className="text-[10px] text-slate-400 block mt-0.5">{item.exchange} · {item.type} · Provider match</span>
                    </div>
                    <span className="text-xs text-cyan-400 font-bold group-hover:translate-x-1 transition-transform">
                      Analyze →
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="flex items-center justify-between mt-2.5 text-xs font-mono text-slate-400 px-1">
            <span>Tip: Press <kbd className="px-1.5 py-0.5 rounded bg-slate-800 text-cyan-300 font-bold border border-slate-700">/</kbd> anywhere to open Universal OmniSearch</span>
            <button
              type="button"
              onClick={handleOpenGlobalOmni}
              className="text-cyan-400 hover:text-cyan-300 hover:underline cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none rounded px-1.5 py-1 min-h-[32px] inline-flex items-center"
            >
              Advanced Search (⌘K) →
            </button>
          </div>
        </form>

        {/* Fast Market Leader Quick Access */}
        <div className="space-y-2 pt-2 border-t border-[var(--border-subtle)]">
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-400 block">
            Popular Institutional Candidates & Benchmarks
          </span>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
            {POPULAR_ASSETS.map((asset) => (
              <button
                key={asset.symbol}
                type="button"
                onClick={() => onSelectSymbol(asset.symbol)}
                data-testid={`quick-pick-${asset.symbol}`}
                aria-label={`Analyze ${asset.symbol} - ${asset.name}`}
                className="p-2.5 rounded-xl bg-[var(--surface-2)] hover:bg-[#1a2436] border border-[var(--border-subtle)] hover:border-cyan-500/60 text-left transition-all group cursor-pointer flex flex-col justify-between min-h-[56px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono font-black text-xs text-white group-hover:text-cyan-300 transition-colors">
                    {asset.symbol}
                  </span>
                  <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700/60">
                    {asset.type}
                  </span>
                </div>
                <div className="text-[11px] text-slate-400 truncate mt-0.5 group-hover:text-slate-300 font-sans">
                  {asset.name}
                </div>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ── 2. Verified Macroeconomic & Yield Curve Overview ─────────────────── */}
      <section
        aria-label="Verified Macroeconomic Regime"
        className="bg-[var(--surface-1)] border border-[var(--border-subtle)] rounded-2xl p-4 sm:p-6 shadow-xl space-y-4"
      >
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border-subtle)] pb-3">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" aria-hidden="true" />
            <h2 className="text-sm sm:text-base font-bold text-white uppercase font-mono tracking-wide">
              Macroeconomic Regime & Capital Cost Horizon
            </h2>
          </div>
          <div className="flex items-center gap-1.5 text-xs font-mono">
            <span
              className={`px-2 py-0.5 rounded border font-semibold ${
                macroData?.isLive && typeof freshVix === "number" && freshVix > 0
                  ? "bg-cyan-950/80 text-cyan-300 border-cyan-800/80"
                  : macroData?.isLive
                  ? "bg-cyan-950/80 text-cyan-300 border-cyan-800/80"
                  : macroData?.status === "STALE"
                  ? "bg-amber-950/80 text-amber-300 border-amber-800/80"
                  : macroData?.status === "UNAVAILABLE"
                  ? "bg-slate-900/80 text-amber-400 border-amber-800/80"
                  : "bg-slate-900/80 text-slate-300 border-slate-700/80"
              }`}
            >
              {macroData?.isLive && typeof freshVix === "number" && freshVix > 0
                ? "FRED & CBOE Verified (Live)"
                : macroData?.isLive
                ? "FRED Verified (Live)"
                : typeof freshVix === "number" && freshVix > 0
                ? "CBOE Verified (Live)"
                : macroData?.status === "STALE"
                ? "FRED Telemetry Stale"
                : "FRED Telemetry Offline / Unavailable"}
            </span>
            {macroData?.lastUpdated && (
              <span className="text-slate-400 hidden sm:inline">
                Observed: {macroData.lastUpdated}
              </span>
            )}
          </div>
        </div>

        {macroData ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
            {/* Yield Curve 10Y-2Y */}
            <div className="p-3 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)] space-y-1">
              <span className="text-[10px] text-slate-400 uppercase font-bold block">10Y-2Y Yield Curve</span>
              {macroData.yieldCurve10Y2Y !== null ? (
                <>
                  <span
                    className={`text-base font-black block tabular-nums ${
                      macroData.yieldCurve10Y2Y >= 0 ? "text-emerald-400" : "text-rose-400"
                    }`}
                  >
                    {macroData.yieldCurve10Y2Y >= 0 ? "+" : ""}
                    {macroData.yieldCurve10Y2Y.toFixed(2)}%
                  </span>
                  <span className="text-[10px] text-slate-400 font-sans block">
                    {macroData.yieldCurve10Y2Y >= 0 ? "Normal Slope (Expansionary)" : "Inverted (Recession Hazard)"}
                  </span>
                  {(macroData.yieldCurveObservationDate || macroData.seriesObservations?.["T10Y2Y"]?.date) && (
                    <span className="text-[9px] text-slate-500 font-sans block">
                      FRED: T10Y2Y • {macroData.yieldCurveObservationDate || macroData.seriesObservations?.["T10Y2Y"]?.date}
                    </span>
                  )}
                </>
              ) : (
                <>
                  <span className="text-base font-black text-slate-400 block tabular-nums">Unavailable</span>
                  <span className="text-[10px] text-slate-500 font-sans block">FRED: T10Y2Y</span>
                </>
              )}
            </div>

            {/* 10Y Real Interest Rate */}
            <div className="p-3 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)] space-y-1">
              <span className="text-[10px] text-slate-400 uppercase font-bold block">10Y Real Rate (TIPS)</span>
              {macroData.realInterestRate10Y !== null ? (
                <>
                  <span className="text-base font-black text-cyan-300 block tabular-nums">
                    {macroData.realInterestRate10Y.toFixed(2)}%
                  </span>
                  <span className="text-[10px] text-slate-400 font-sans block">DFII10 Capital Hurdle</span>
                  {(macroData.realRateObservationDate || macroData.seriesObservations?.["DFII10"]?.date) && (
                    <span className="text-[9px] text-slate-500 font-sans block">
                      FRED: DFII10 • {macroData.realRateObservationDate || macroData.seriesObservations?.["DFII10"]?.date}
                    </span>
                  )}
                </>
              ) : (
                <>
                  <span className="text-base font-black text-slate-400 block tabular-nums">Unavailable</span>
                  <span className="text-[10px] text-slate-500 font-sans block">FRED: DFII10</span>
                </>
              )}
            </div>

            {/* High Yield Spread */}
            <div className="p-3 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)] space-y-1">
              <span className="text-[10px] text-slate-400 uppercase font-bold block">High Yield Credit Spread</span>
              {macroData.highYieldCreditSpread !== null ? (
                <>
                  <span
                    className={`text-base font-black block tabular-nums ${
                      macroData.highYieldCreditSpread < 4.0 ? "text-emerald-400" : "text-amber-400"
                    }`}
                  >
                    {macroData.highYieldCreditSpread.toFixed(2)}%
                  </span>
                  <span className="text-[10px] text-slate-400 font-sans block">
                    {macroData.highYieldCreditSpread < 4.0 ? "Tight / Risk Seeking" : "Wide / Risk Averse"}
                  </span>
                  {(macroData.creditSpreadObservationDate || macroData.seriesObservations?.["BAMLH0A0HYM2"]?.date) && (
                    <span className="text-[9px] text-slate-500 font-sans block">
                      FRED: BAMLH0A0HYM2 • {macroData.creditSpreadObservationDate || macroData.seriesObservations?.["BAMLH0A0HYM2"]?.date}
                    </span>
                  )}
                </>
              ) : (
                <>
                  <span className="text-base font-black text-slate-400 block tabular-nums">Unavailable</span>
                  <span className="text-[10px] text-slate-500 font-sans block">FRED: BAMLH0A0HYM2</span>
                </>
              )}
            </div>

            {/* Market Volatility Index (VIX) */}
            <div className="p-3 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)] space-y-1">
              <span className="text-[10px] text-slate-400 uppercase font-bold block">Macro Volatility (VIX)</span>
              <span className={`text-base font-black block tabular-nums ${(freshVix || 16.5) < 26.0 ? "text-emerald-400" : "text-rose-400"}`}>
                {(freshVix || 16.5).toFixed(1)}
              </span>
              <span className="text-[10px] text-slate-400 font-sans block">
                {(freshVix || 16.5) < 26.0 ? "Calm Regime (Execution Open)" : "Elevated Risk (Holds Active)"}
              </span>
              <span className="text-[9px] text-slate-500 font-sans block">
                CBOE: VIX Index • 26.0 Ceiling
              </span>
            </div>
          </div>
        ) : (
          <div className="p-4 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)] text-xs text-slate-400 font-mono flex items-center justify-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span>Synchronizing authenticated macroeconomic feeds...</span>
          </div>
        )}

        <div className="p-3.5 rounded-xl bg-[#090d16] border border-[#1b2537] flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div className="space-y-1">
            <span className="text-xs uppercase font-mono font-bold text-slate-400 block tracking-wide">
              Current Macro Posture Assessment:
            </span>
            <span className="font-bold text-white text-sm">
              {macroData?.regimeTitle || "Macroeconomic Baseline Active"}
            </span>
            <span className="text-slate-400 block text-xs font-sans leading-relaxed">
              {macroData?.regimeSubtitle || "Position sizing scales proportionally with broad market liquidity and credit spreads."}
            </span>
          </div>
          <Link
            href="/radar?filter=macro_aligned"
            className="text-xs font-mono font-bold text-cyan-400 hover:text-cyan-300 underline shrink-0 min-h-[44px] inline-flex items-center focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none rounded px-2"
          >
            Scan Macro-Aligned Candidates →
          </Link>
        </div>
      </section>

      {/* ── 3. Four-Step Market Workflow Navigator ───────────────────────────── */}
      <IntentHero onSelectSymbol={onSelectSymbol} />

      {/* ── 4. Governed Tactical Candidates Spotlight ────────────────────────── */}
      <WeeklyConfluenceSpotlight
        defaultCollapsed={false}
        onSelectSymbol={onSelectSymbol}
        selectedSymbol=""
      />
    </div>
  );
}
