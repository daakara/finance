"use client";

import { useEffect, useState, useRef, useCallback, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Navbar from "../components/Navbar";
import WatchlistSidebar from "../components/WatchlistSidebar";
import PriceChart from "../components/PriceChart";
import RiskMetricsCard from "../components/RiskMetricsCard";
import AssetFactorRadar from "../components/AssetFactorRadar";
import TraderArchetypesCard from "../components/TraderArchetypesCard";
import DayTraderPositionSizer from "../components/DayTraderPositionSizer";
import SelfHealingAccuracyCard from "../components/SelfHealingAccuracyCard";
import MarketGraphCard from "../components/MarketGraphCard";
import CatalystForecastCard from "../components/CatalystForecastCard";
import InstitutionalFeeds from "../components/InstitutionalFeeds";
import CompositeConvictionCard from "../components/CompositeConvictionCard";
import { FredMacroData, SecForm4Trade, fetchFredMacroRegime, fetchSecForm4Insiders } from "../lib/institutionalFeeds";
import CongressionalTradesCard from "../components/CongressionalTradesCard";
import OptimalEntryExitCard from "../components/OptimalEntryExitCard";
import DataSourceBadge from "../components/DataSourceBadge";
import WeeklyConfluenceSpotlight from "../components/WeeklyConfluenceSpotlight";
import IntentHero from "../components/IntentHero";
import AdaptiveTerminal from "../components/AdaptiveTerminal";
import { fetchAssetAnalytics, AnalyticsResponse, SpotPriceRegistry } from "../lib/api";
import { trackWorkspaceSwitch, trackRoleSwitch, trackSymbolSearch } from "../lib/matomo";
import { resolveAssetAlias } from "../lib/assetRegistry";
import TerminalSsrShell from "../components/TerminalSsrShell";

type WorkspaceTab = "EXECUTION" | "SMART_MONEY" | "FUNDAMENTALS" | "RISK_CONTAGION";

function TerminalContent() {
  const searchParams = useSearchParams();
  const urlSymbol = searchParams.get("symbol");
  const urlTab = searchParams.get("tab")?.toUpperCase();

  const [selectedSymbol, setSelectedSymbol] = useState<string>(urlSymbol ? urlSymbol.toUpperCase() : "AAPL");
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [interval, setInterval] = useState<string>("1y_hist");
  const [userRole, setUserRole] = useState<"DAY_TRADER" | "LONG_TERM">("LONG_TERM");
  const [activeTab, setActiveTab] = useState<WorkspaceTab>(
    (urlTab === "SMART_MONEY" || urlTab === "FUNDAMENTALS" || urlTab === "RISK_CONTAGION" || urlTab === "EXECUTION")
      ? urlTab as WorkspaceTab
      : "EXECUTION"
  );
  const [macroData, setMacroData] = useState<FredMacroData | null>(null);
  const [insiderTrades, setInsiderTrades] = useState<SecForm4Trade[]>([]);
  const [lastUpdatedTime, setLastUpdatedTime] = useState<string>("");
  const cacheRef = useRef<Map<string, AnalyticsResponse>>(new Map());

  useEffect(() => {
    setLastUpdatedTime(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
  }, [data]);

  useEffect(() => {
    fetchFredMacroRegime().then(setMacroData);
    fetchSecForm4Insiders(selectedSymbol).then(setInsiderTrades);
  }, [selectedSymbol]);

  // Sync URL search params when navigated from Screener, Compare, or Smart Money pages
  useEffect(() => {
    if (urlSymbol && urlSymbol.toUpperCase() !== selectedSymbol) {
      setSelectedSymbol(urlSymbol.toUpperCase());
    }
  }, [urlSymbol]);

  useEffect(() => {
    if (urlTab && (urlTab === "SMART_MONEY" || urlTab === "FUNDAMENTALS" || urlTab === "RISK_CONTAGION" || urlTab === "EXECUTION")) {
      if (urlTab !== activeTab) {
        setActiveTab(urlTab as WorkspaceTab);
      }
    }
  }, [urlTab]);

  const handleTabChange = useCallback((tab: WorkspaceTab, label: string) => {
    setActiveTab(tab);
    trackWorkspaceSwitch(label, selectedSymbol);
    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      url.searchParams.set("tab", tab.toLowerCase());
      if (selectedSymbol) url.searchParams.set("symbol", selectedSymbol);
      window.history.replaceState({}, "", url.toString());
    }
  }, [selectedSymbol]);

  const handleSelectSymbol = useCallback((newSym: string) => {
    const clean = newSym.trim().toUpperCase();
    if (!clean) return;
    setSelectedSymbol(clean);
    trackSymbolSearch(clean, "OmniSearch");
    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      url.searchParams.set("symbol", clean);
      window.history.replaceState({}, "", url.toString());
    }
  }, []);

  const aliasRecommendation = resolveAssetAlias(selectedSymbol);

  useEffect(() => {
    const saved = localStorage.getItem("FINANCE_USER_ROLE");
    if (saved === "DAY_TRADER" || saved === "LONG_TERM") {
      setUserRole(saved);
      if (saved === "DAY_TRADER") setInterval("5m");
      else setInterval("1y_hist");
    }
  }, []);

  const handleRoleChange = useCallback((role: "DAY_TRADER" | "LONG_TERM") => {
    trackRoleSwitch(role);
    setUserRole(role);
    if (role === "DAY_TRADER") {
      setInterval("5m");
    } else {
      setInterval("1y_hist");
    }
  }, []);

  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      const cacheKey = `${selectedSymbol}_${interval}`;
      const cached = cacheRef.current.get(cacheKey);

      // Instant optimistic display from memory cache (<10ms)
      if (cached) {
        setData(cached);
        setLoading(false);
        return;
      }

      setLoading(true);
      setError(null);
      try {
        let period = "1y";
        let apiInterval = "1d";

        // Map Day Trader intervals
        if (interval === "1m") {
          period = "1d";
          apiInterval = "1m";
        } else if (interval === "5m" || interval === "15m") {
          period = "5d";
          apiInterval = interval;
        } else if (interval === "1h") {
          period = "1mo";
          apiInterval = "1h";
        }
        // Map Long-Term Horizons up to 5 Years
        else if (interval === "1m_hist") {
          period = "1mo";
          apiInterval = "1d";
        } else if (interval === "6m_hist") {
          period = "6mo";
          apiInterval = "1d";
        } else if (interval === "1y_hist" || interval === "1d") {
          period = "1y";
          apiInterval = "1d";
        } else if (interval === "3y_hist" || interval === "1wk") {
          period = "3y";
          apiInterval = "1wk";
        } else if (interval === "5y_hist" || interval === "1mo") {
          period = "5y";
          apiInterval = "1mo";
        }

        const knownPrice = data?.symbol?.toUpperCase() === selectedSymbol.toUpperCase() ? data.currentPrice : undefined;
        const knownChange = data?.symbol?.toUpperCase() === selectedSymbol.toUpperCase() ? data.priceChangePct24h : undefined;

        const res = await fetchAssetAnalytics(selectedSymbol, period, apiInterval, knownPrice, knownChange);
        if (isMounted) {
          setData(res);
          setError(null);
          cacheRef.current.set(cacheKey, res);
        }
      } catch (err) {
        console.error("Failed to load asset analytics:", err);
        if (isMounted) {
          setError(err instanceof Error ? err.message : `Failed to load asset analytics for ${selectedSymbol}`);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    loadData();

    const handlePurge = () => {
      cacheRef.current.clear();
      loadData();
    };

    window.addEventListener("finance:cache-purge", handlePurge);

    return () => {
      isMounted = false;
      window.removeEventListener("finance:cache-purge", handlePurge);
    };
  }, [selectedSymbol, interval]);

  return (
    <div className="min-h-screen bg-[var(--bg-app)] text-[var(--text-main)] flex flex-col font-sans selection:bg-cyan-500 selection:text-black transition-colors duration-200">
      {/* Skip to Main Content Link for Keyboard Accessibility */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:px-4 focus:py-2 focus:bg-cyan-500 focus:text-black focus:font-bold focus:rounded-md focus:shadow-lg"
      >
        Skip to main content
      </a>

      <Navbar userRole={userRole} onRoleChange={handleRoleChange} />

      {/* Semantic Main Content Landmark */}
      <main id="main-content" role="main" className="flex-1 max-w-[1750px] w-full mx-auto p-2.5 sm:p-5 grid grid-cols-1 lg:grid-cols-4 gap-3 sm:gap-5 pb-28 sm:pb-5">
        {/* Main Terminal Workspace (Hero on mobile, Right column on desktop) */}
        <section aria-label="Market Workspace and Quantitative Analytics" className="lg:col-span-3 space-y-4 sm:space-y-5 order-1 lg:order-2 min-w-0">
          {/* Intent-First Home Hero: "What are you looking to do today?" */}
          <IntentHero onSelectSymbol={handleSelectSymbol} />

          {/* Ingestion Failure / Explicit Retry State OR Adaptive Multi-Tier Terminal Engine */}
          {error && !data ? (
            <div className="p-5 bg-rose-950/40 border border-rose-800/60 rounded-2xl text-rose-200 font-sans space-y-3 animate-fade-in shadow-xl">
              <div className="flex items-center gap-3">
                <span className="text-2xl">⚠️</span>
                <div>
                  <h2 className="text-sm sm:text-base font-bold text-white">Market Ingestion Unavailable for {selectedSymbol}</h2>
                  <p className="text-xs text-rose-300/90 leading-relaxed mt-0.5">
                    {error}. Live candle feeds or fundamental disclosures could not be verified.
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-rose-900/40">
                <button
                  type="button"
                  onClick={() => {
                    cacheRef.current.clear();
                    setError(null);
                    setLoading(true);
                    fetchAssetAnalytics(selectedSymbol, "1y", "1d")
                      .then((res) => {
                        setData(res);
                        setError(null);
                      })
                      .catch((e) => setError(e instanceof Error ? e.message : "Retry failed"))
                      .finally(() => setLoading(false));
                  }}
                  className="px-3.5 py-1.5 bg-rose-600 hover:bg-rose-500 text-white rounded-lg text-xs font-bold transition-all active:scale-95 cursor-pointer shadow"
                >
                  🔄 Retry Analysis
                </button>
                <button
                  type="button"
                  onClick={() => handleSelectSymbol("NVDA")}
                  className="px-3 py-1.5 bg-[#0e1420] hover:bg-[#182335] border border-[#24334a] text-slate-300 rounded-lg text-xs font-bold transition-all cursor-pointer"
                >
                  Analyze NVDA
                </button>
                <a
                  href="/screener"
                  className="px-3 py-1.5 bg-[#0e1420] hover:bg-[#182335] border border-[#24334a] text-cyan-400 rounded-lg text-xs font-bold transition-all hover:underline"
                >
                  Explore Screener Candidates →
                </a>
              </div>
            </div>
          ) : (
            /* 🎯 Adaptive Multi-Tier Terminal Engine (Guided · Standard · Advanced) */
            <AdaptiveTerminal
              symbol={selectedSymbol}
              companyName={aliasRecommendation?.companyName || selectedSymbol}
              currentPrice={data?.currentPrice ?? (SpotPriceRegistry.get(selectedSymbol)?.price || 0)}
              changePct={data?.priceChangePct24h || 0}
              setupScore={60}
              candles={data?.candles}
              dataSource={data?._dataSource}
            />
          )}

          {/* Main Candlestick Chart with Expanded 5-Year Horizons */}
          <div id="market-workspace-chart" className="min-h-[380px] sm:min-h-[420px]">
            {data && (
              <div className="flex flex-wrap items-center justify-between gap-2 mb-2 px-1 text-[10px] font-mono">
                <div className="flex items-center gap-2">
                  <DataSourceBadge source={data._dataSource} />
                  {lastUpdatedTime && (
                    <span className="text-slate-400 hidden sm:inline">
                      Updated: <span className="text-slate-300 font-semibold">{lastUpdatedTime}</span>
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  {selectedSymbol.toUpperCase().includes("-USD") || ["BTC", "ETH", "SOL"].includes(selectedSymbol.toUpperCase()) ? (
                    <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-950/80 border border-emerald-700/80 text-[10px] font-mono text-emerald-300 font-semibold">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                      <span>24/7 Digital Asset Market Active</span>
                    </span>
                  ) : (
                    <div className="text-slate-500 flex items-center gap-2">
                      <span className="hidden md:inline">NYSE/NASDAQ Session State</span>
                      <span className="px-1.5 py-0.5 rounded bg-[#162030] text-cyan-300 font-semibold">15m Delayed/EOD</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* 🎯 Weekly High-Confluence Alpha Spotlight (Auto-folded on dedicated ticker search) */}
            <WeeklyConfluenceSpotlight
              defaultCollapsed={selectedSymbol.toUpperCase() !== "SPY"}
              onSelectSymbol={handleSelectSymbol}
              selectedSymbol={selectedSymbol}
            />

            {/* 💡 Intelligent Did You Mean Ticker Recommendation Banner */}
            {aliasRecommendation && aliasRecommendation.canonicalTicker.toUpperCase() !== selectedSymbol.toUpperCase() && (
              <div className="mb-3 p-3 rounded-xl bg-[#141b29] border-2 border-cyan-500/80 text-xs font-mono text-cyan-200 shadow-xl flex flex-wrap items-center justify-between gap-3 animate-fadeIn">
                <div className="flex items-center gap-2.5">
                  <span className="text-xl">💡</span>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-white uppercase tracking-wide">Unlisted Search Query: &quot;{selectedSymbol}&quot;</span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-700/80">Company / Brand Match</span>
                    </div>
                    <p className="text-[11px] text-slate-300 mt-0.5">
                      &quot;{selectedSymbol}&quot; is a company brand, not an official exchange ticker. Did you mean to analyze:
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2 flex-wrap">
                  <button
                    type="button"
                    onClick={() => handleSelectSymbol(aliasRecommendation.canonicalTicker)}
                    className="px-3 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-black text-xs flex items-center gap-1.5 transition-transform active:scale-95 shadow-md cursor-pointer"
                  >
                    <span>🚀 Switch to {aliasRecommendation.canonicalTicker}</span>
                    <span className="hidden sm:inline font-semibold text-slate-900">({aliasRecommendation.companyName})</span>
                    <span>→</span>
                  </button>
                  {aliasRecommendation.peerSuggestion && (
                    <button
                      type="button"
                      onClick={() => handleSelectSymbol(aliasRecommendation.peerSuggestion!)}
                      className="px-2.5 py-1.5 rounded-lg bg-[#1a2538] hover:bg-[#23324d] text-slate-200 border border-[#2e405e] text-xs font-semibold transition-colors cursor-pointer"
                    >
                      Compare {aliasRecommendation.peerSuggestion}
                    </button>
                  )}
                </div>
              </div>
            )}

            {data?._dataSource === 'fallback' && (!aliasRecommendation || aliasRecommendation.canonicalTicker.toUpperCase() === selectedSymbol.toUpperCase()) && (
              <div className="mb-2 p-2 rounded-lg bg-amber-950/30 border border-amber-800/40 text-[11px] font-mono text-amber-300 flex items-center justify-between">
                <span>⚠️ Note: Operating in offline fallback simulation mode — live institutional feeds will auto-resume upon backend handshake.</span>
              </div>
            )}

            <PriceChart
              symbol={selectedSymbol}
              candles={data?.candles || []}
              currentPrice={data?.currentPrice}
              priceChangePct={data?.priceChangePct24h}
              interval={interval}
              userRole={userRole}
              onRoleChange={handleRoleChange}
              onIntervalChange={setInterval}
              tradeMarkers={
                [
                  ...(data?.smartMoney?.congressTrades?.map((ct) => ({
                    date: ct.transaction_date || (ct as any).date || ct.filing_date,
                    label: `🏛️ ${ct.politician?.split(" ")?.slice(-1)[0] || "Congress"} ${ct.transaction_type || (ct as any).type || "Buy"}`,
                    type: "CONGRESS" as const,
                    amount: ct.amount_range,
                  })) || []),
                  ...(((data?.smartMoney as any)?.secForm4Trades || (data?.smartMoney as any)?.secInsiderTrades || [])?.map((st: any) => ({
                    date: st.filingDate || st.date || st.filing_date,
                    label: `💼 ${st.insiderRole || "Insider"} Buy`,
                    type: "INSIDER" as const,
                    amount: st.transactionValue || st.shares,
                  })) || []),
                  ...(data?.smartMoney?.optionsFlow?.map((op) => ({
                    date: (op as any).date || (op as any).timestamp || "2026-04-01",
                    label: `⚡ ${op.type || "Call Sweep"}`,
                    type: "OPTIONS" as const,
                    amount: op.premium,
                  })) || []),
                ].filter((m) => m.date)
              }
              smartMoneyHeadline={
                data?.smartMoney?.congressTrades?.[0]
                  ? `${data.smartMoney.congressTrades[0].politician.split(" ")[0]} ${data.smartMoney.congressTrades[0].amount_range}`
                  : data?.smartMoney?.optionsFlow?.[0]
                  ? `${data.smartMoney.optionsFlow[0].type} (${data.smartMoney.optionsFlow[0].premium})`
                  : undefined
              }
              catalystHeadline={
                data?.catalystForecast?.primary_drug_trial
                  ? `${data.catalystForecast.primary_drug_trial} (${data.catalystForecast.trial_phase}) - ${data.catalystForecast.trial_readout_timeline}`
                  : data?.catalystForecast?.efficacy_summary
                  ? data.catalystForecast.efficacy_summary
                  : undefined
              }
              loading={loading}
              technicals={data?.technicals}
            />
          </div>

          {/* 💎 DECISION-SYNTHESIS COMPOSITE CONVICTION SCORECARD */}
          <CompositeConvictionCard
            symbol={selectedSymbol}
            data={data}
            macro={macroData}
            insiders={insiderTrades}
            userRole={userRole}
          />

          {/* 🗂️ MODULAR WORKSPACE TABS (Eliminates Cognitive Overload & Infinite Scroll) */}
          <div role="tablist" aria-label="Quantitative Domain Workspaces" className="bg-[#0c1017] p-1.5 rounded-2xl border border-[#243044] grid grid-cols-2 sm:grid-cols-4 gap-1.5 shadow-xl font-mono text-xs">
            <button
              role="tab"
              aria-selected={activeTab === "EXECUTION"}
              onClick={() => handleTabChange("EXECUTION", "Execution & Levels")}
              className={`flex items-center justify-center space-x-1.5 py-2 px-2 sm:py-2.5 sm:px-3 rounded-xl font-bold transition-all active:scale-[0.97] text-[11px] sm:text-xs cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                activeTab === "EXECUTION"
                  ? "bg-cyan-600 text-white shadow-sm font-extrabold"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
              }`}
            >
              <span>🎯</span>
              <span className="whitespace-nowrap"><span className="sm:hidden">Execution</span><span className="hidden sm:inline">Execution & Levels</span></span>
            </button>

            <button
              role="tab"
              aria-selected={activeTab === "SMART_MONEY"}
              onClick={() => handleTabChange("SMART_MONEY", "Smart Money")}
              className={`flex items-center justify-center space-x-1.5 py-2 px-2 sm:py-2.5 sm:px-3 rounded-xl font-bold transition-all active:scale-[0.97] text-[11px] sm:text-xs cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                activeTab === "SMART_MONEY"
                  ? "bg-cyan-600 text-white shadow-sm font-extrabold"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
              }`}
            >
              <span>🏛️</span>
              <span className="whitespace-nowrap">Smart Money</span>
            </button>

            <button
              role="tab"
              aria-selected={activeTab === "FUNDAMENTALS"}
              onClick={() => handleTabChange("FUNDAMENTALS", "Factors & Macro")}
              className={`flex items-center justify-center space-x-1.5 py-2 px-2 sm:py-2.5 sm:px-3 rounded-xl font-bold transition-all active:scale-[0.97] text-[11px] sm:text-xs cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                activeTab === "FUNDAMENTALS"
                  ? "bg-cyan-600 text-white shadow-sm font-extrabold"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
              }`}
            >
              <span>📊</span>
              <span className="whitespace-nowrap"><span className="sm:hidden">Factors</span><span className="hidden sm:inline">Factors & Macro</span></span>
            </button>

            <button
              role="tab"
              aria-selected={activeTab === "RISK_CONTAGION"}
              onClick={() => handleTabChange("RISK_CONTAGION", "Risk & Contagion")}
              className={`flex items-center justify-center space-x-1.5 py-2 px-2 sm:py-2.5 sm:px-3 rounded-xl font-bold transition-all active:scale-[0.97] text-[11px] sm:text-xs cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                activeTab === "RISK_CONTAGION"
                  ? "bg-cyan-600 text-white shadow-sm font-extrabold"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
              }`}
            >
              <span>🛡️</span>
              <span className="whitespace-nowrap"><span className="sm:hidden">Risk</span><span className="hidden sm:inline">Risk & Contagion</span></span>
            </button>
          </div>

          {/* TAB 1: EXECUTION & LEVELS */}
          {activeTab === "EXECUTION" && (
            <div className="space-y-4 sm:space-y-5 animate-fadeIn">
              {userRole === "DAY_TRADER" && data && (
                <DayTraderPositionSizer symbol={selectedSymbol} data={data} />
              )}
              <OptimalEntryExitCard
                symbol={selectedSymbol}
                executionPlan={data?.optimalExecution}
                userRole={userRole}
                smartMoney={data?.smartMoney}
                macroRegime={macroData}
              />
            </div>
          )}

          {/* TAB 2: SMART MONEY & INSIDER DISCLOSURES */}
          {activeTab === "SMART_MONEY" && (
            <div className="space-y-4 sm:space-y-5 animate-fadeIn">
              <CongressionalTradesCard
                symbol={selectedSymbol}
                congressTrades={data?.smartMoney?.congressTrades}
                optionsFlow={data?.smartMoney?.optionsFlow}
                userRole={userRole}
                onSelectSymbol={setSelectedSymbol}
              />
              <TraderArchetypesCard
                symbol={selectedSymbol}
                traderArchetypes={data?.traderArchetypes}
              />
            </div>
          )}

          {/* TAB 3: FUNDAMENTALS & MACRO REGIME */}
          {activeTab === "FUNDAMENTALS" && (
            <div className="space-y-4 sm:space-y-5 animate-fadeIn">
              <AssetFactorRadar
                symbol={selectedSymbol}
                factorScores={data?.factorScores}
                macroDifficulty={data?.macroDifficulty}
                expectedReturn={data?.expectedReturn}
              />
              <InstitutionalFeeds activeSymbol={selectedSymbol} />
              <CatalystForecastCard data={data?.catalystForecast} />
            </div>
          )}

          {/* TAB 4: RISK & CONTAGION */}
          {activeTab === "RISK_CONTAGION" && (
            <div className="space-y-4 sm:space-y-5 animate-fadeIn">
              <MarketGraphCard symbol={selectedSymbol} marketGraph={data?.marketGraph} />
              <SelfHealingAccuracyCard symbol={selectedSymbol} auditData={data?.selfHealingAudit} />
              <RiskMetricsCard analyticsData={data || undefined} userRole={userRole} />
            </div>
          )}
        </section>

        {/* Watchlist Sidebar (Left column on desktop, Below chart on mobile) */}
        <aside aria-label="Watchlist and Real-Time Feeds" className="lg:col-span-1 h-full order-2 lg:order-1 min-w-0">
          <WatchlistSidebar
            activeSymbol={selectedSymbol}
            onSelectSymbol={setSelectedSymbol}
            liveCurrentPrice={data?.currentPrice}
            livePriceChangePct={data?.priceChangePct24h}
          />
        </aside>
      </main>
    </div>
  );
}

export default function TerminalPage() {
  return (
    <Suspense fallback={<TerminalSsrShell />}>
      <TerminalContent />
    </Suspense>
  );
}