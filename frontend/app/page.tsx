"use client";

import { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import WatchlistSidebar from "../components/WatchlistSidebar";
import PriceChart from "../components/PriceChart";
import RiskMetricsCard from "../components/RiskMetricsCard";
import AssetFactorRadar from "../components/AssetFactorRadar";
import TraderArchetypesCard from "../components/TraderArchetypesCard";
import DayTraderPositionSizer from "../components/DayTraderPositionSizer";
import SelfHealingAccuracyCard from "../components/SelfHealingAccuracyCard";
import MarketGraphCard from "../components/MarketGraphCard";
import { fetchAssetAnalytics, AnalyticsResponse } from "../lib/api";

export default function TerminalPage() {
  const [selectedSymbol, setSelectedSymbol] = useState<string>("AAPL");
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [interval, setInterval] = useState<string>("1d");
  const [userRole, setUserRole] = useState<"DAY_TRADER" | "LONG_TERM">("LONG_TERM");

  useEffect(() => {
    const saved = localStorage.getItem("FINANCE_USER_ROLE");
    if (saved === "DAY_TRADER" || saved === "LONG_TERM") {
      setUserRole(saved);
      if (saved === "DAY_TRADER") setInterval("5m");
    }
  }, []);

  const handleRoleChange = (role: "DAY_TRADER" | "LONG_TERM") => {
    setUserRole(role);
    if (role === "DAY_TRADER") {
      setInterval("5m");
    } else {
      setInterval("1d");
    }
  };

  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      setLoading(true);
      try {
        const res = await fetchAssetAnalytics(selectedSymbol, "1y", interval);
        if (isMounted) setData(res);
      } catch (err) {
        console.error("Failed to load asset analytics:", err);
      } finally {
        if (isMounted) setLoading(false);
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
      <main id="main-content" role="main" className="flex-1 max-w-[1750px] w-full mx-auto p-3 sm:p-5 grid grid-cols-1 lg:grid-cols-4 gap-4 sm:gap-5">
        {/* Left Column: Watchlist Sidebar */}
        <aside aria-label="Watchlist and Real-Time Feeds" className="lg:col-span-1 h-full">
          <WatchlistSidebar activeSymbol={selectedSymbol} onSelectSymbol={setSelectedSymbol} />
        </aside>

        {/* Right Column: Dynamic Terminal Workspace */}
        <section aria-label="Market Workspace and Quantitative Analytics" className="lg:col-span-3 space-y-4 sm:space-y-5">
          {/* Main Candlestick Chart */}
          <div className="min-h-[380px] sm:min-h-[420px]">
            <PriceChart
              symbol={selectedSymbol}
              candles={data?.candles || []}
              currentPrice={data?.currentPrice}
              priceChangePct={data?.priceChangePct24h}
              interval={interval}
              onIntervalChange={setInterval}
              technicals={data?.technicals}
            />
          </div>

          {/* DUAL-JOURNEY WORKSPACE ROUTING */}
          {userRole === "DAY_TRADER" ? (
            /* Day Trader Journey: Live Risk Sizer, Intraday Technicals, Continuous Self-Healing */
            <div className="space-y-4 sm:space-y-5">
              {data && <DayTraderPositionSizer symbol={selectedSymbol} data={data} />}
              <SelfHealingAccuracyCard symbol={selectedSymbol} auditData={data?.selfHealingAudit} />
              <RiskMetricsCard analyticsData={data || undefined} />
            </div>
          ) : (
            /* Long-Term Wealth Journey: Fundamental Factor Radar, Market Graph Contagion, 5-Strategy Consensus */
            <div className="space-y-4 sm:space-y-5">
              <AssetFactorRadar
                symbol={selectedSymbol}
                factorScores={data?.factorScores}
                macroDifficulty={data?.macroDifficulty}
                expectedReturn={data?.expectedReturn}
              />
              <MarketGraphCard symbol={selectedSymbol} marketGraph={data?.marketGraph} />
              <TraderArchetypesCard
                symbol={selectedSymbol}
                traderArchetypes={data?.traderArchetypes}
              />
              <RiskMetricsCard analyticsData={data || undefined} />
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

