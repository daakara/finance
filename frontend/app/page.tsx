"use client";

import { useEffect, useState, useRef, Suspense } from "react";
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
import CongressionalTradesCard from "../components/CongressionalTradesCard";
import OptimalEntryExitCard from "../components/OptimalEntryExitCard";
import { fetchAssetAnalytics, AnalyticsResponse } from "../lib/api";

function TerminalContent() {
  const searchParams = useSearchParams();
  const urlSymbol = searchParams.get("symbol");

  const [selectedSymbol, setSelectedSymbol] = useState<string>(urlSymbol ? urlSymbol.toUpperCase() : "AAPL");
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [interval, setInterval] = useState<string>("1y_hist");
  const [userRole, setUserRole] = useState<"DAY_TRADER" | "LONG_TERM">("LONG_TERM");
  const cacheRef = useRef<Map<string, AnalyticsResponse>>(new Map());

  // Sync URL search params when navigated from Screener, Compare, or Smart Money pages
  useEffect(() => {
    if (urlSymbol && urlSymbol.toUpperCase() !== selectedSymbol) {
      setSelectedSymbol(urlSymbol.toUpperCase());
    }
  }, [urlSymbol]);

  useEffect(() => {
    const saved = localStorage.getItem("FINANCE_USER_ROLE");
    if (saved === "DAY_TRADER" || saved === "LONG_TERM") {
      setUserRole(saved);
      if (saved === "DAY_TRADER") setInterval("5m");
      else setInterval("1y_hist");
    }
  }, []);

  const handleRoleChange = (role: "DAY_TRADER" | "LONG_TERM") => {
    setUserRole(role);
    if (role === "DAY_TRADER") {
      setInterval("5m");
    } else {
      setInterval("1y_hist");
    }
  };

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

        const res = await fetchAssetAnalytics(selectedSymbol, period, apiInterval);
        if (isMounted) {
          setData(res);
          cacheRef.current.set(cacheKey, res);
        }
      } catch (err) {
        console.error("Failed to load asset analytics:", err);
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    loadData();

    return () => {
      isMounted = false;
    };
  }, [selectedSymbol, interval]);

  return (
    <div className="min-h-screen bg-[#070a10] text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-black">
      {/* Skip to Main Content Link for Keyboard Accessibility */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:px-4 focus:py-2 focus:bg-cyan-500 focus:text-black focus:font-bold focus:rounded-md focus:shadow-lg"
      >
        Skip to main content
      </a>

      <Navbar userRole={userRole} onRoleChange={handleRoleChange} />

      {/* Semantic Main Content Landmark */}
      <main id="main-content" role="main" className="flex-1 max-w-[1750px] w-full mx-auto p-3 sm:p-5 grid grid-cols-1 lg:grid-cols-4 gap-4 sm:gap-5 pb-20 sm:pb-5">
        {/* Left Column: Watchlist Sidebar */}
        <aside aria-label="Watchlist and Real-Time Feeds" className="lg:col-span-1 h-full">
          <WatchlistSidebar
            activeSymbol={selectedSymbol}
            onSelectSymbol={setSelectedSymbol}
            liveCurrentPrice={data?.currentPrice}
            livePriceChangePct={data?.priceChangePct24h}
          />
        </aside>

        {/* Right Column: Dynamic Terminal Workspace */}
        <section aria-label="Market Workspace and Quantitative Analytics" className="lg:col-span-3 space-y-4 sm:space-y-5">
          {/* Main Candlestick Chart with Expanded 5-Year Horizons */}
          <div className="min-h-[380px] sm:min-h-[420px]">
            <PriceChart
              symbol={selectedSymbol}
              candles={data?.candles || []}
              currentPrice={data?.currentPrice}
              priceChangePct={data?.priceChangePct24h}
              interval={interval}
              userRole={userRole}
              onIntervalChange={setInterval}
              smartMoneyHeadline={
                data?.smartMoney?.congressTrades?.[0]
                  ? `${data.smartMoney.congressTrades[0].politician.split(" ")[0]} ${data.smartMoney.congressTrades[0].amount_range}`
                  : data?.smartMoney?.optionsFlow?.[0]
                  ? `${data.smartMoney.optionsFlow[0].type} (${data.smartMoney.optionsFlow[0].premium})`
                  : undefined
              }
              loading={loading}
              technicals={data?.technicals}
            />
          </div>

          {/* DUAL-JOURNEY WORKSPACE ROUTING */}
          {userRole === "DAY_TRADER" ? (
            /* Day Trader Journey: Live Risk Sizer, Intraday Technicals, Continuous Self-Healing, Real-Time Flow Tape */
            <div className="space-y-4 sm:space-y-5">
              {data && <DayTraderPositionSizer symbol={selectedSymbol} data={data} />}
              <OptimalEntryExitCard symbol={selectedSymbol} executionPlan={data?.optimalExecution} userRole={userRole} />
              <CongressionalTradesCard
                symbol={selectedSymbol}
                congressTrades={data?.smartMoney?.congressTrades}
                optionsFlow={data?.smartMoney?.optionsFlow}
                userRole={userRole}
                onSelectSymbol={setSelectedSymbol}
              />
              <SelfHealingAccuracyCard symbol={selectedSymbol} auditData={data?.selfHealingAudit} />
              <RiskMetricsCard analyticsData={data || undefined} userRole={userRole} />
            </div>
          ) : (
            /* Long-Term Wealth Journey: Fundamental Factor Radar, Market Graph Contagion, Catalysts, Congressional Insider Trades */
            <div className="space-y-4 sm:space-y-5">
              <OptimalEntryExitCard symbol={selectedSymbol} executionPlan={data?.optimalExecution} userRole={userRole} />
              <AssetFactorRadar
                symbol={selectedSymbol}
                factorScores={data?.factorScores}
                macroDifficulty={data?.macroDifficulty}
                expectedReturn={data?.expectedReturn}
              />
              <MarketGraphCard symbol={selectedSymbol} marketGraph={data?.marketGraph} />
              <CongressionalTradesCard
                symbol={selectedSymbol}
                congressTrades={data?.smartMoney?.congressTrades}
                optionsFlow={data?.smartMoney?.optionsFlow}
                userRole={userRole}
                onSelectSymbol={setSelectedSymbol}
              />
              <CatalystForecastCard data={data?.catalystForecast} />
              <TraderArchetypesCard
                symbol={selectedSymbol}
                traderArchetypes={data?.traderArchetypes}
              />
              <RiskMetricsCard analyticsData={data || undefined} userRole={userRole} />
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default function TerminalPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#070a10] text-slate-100 flex items-center justify-center font-mono">Loading Terminal...</div>}>
      <TerminalContent />
    </Suspense>
  );
}