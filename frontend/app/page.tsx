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
      <Navbar userRole={userRole} onRoleChange={handleRoleChange} />

      <main className="flex-1 flex flex-col lg:flex-row p-3 md:p-6 gap-6 max-w-[1750px] w-full mx-auto">
        {/* Watchlist Sidebar */}
        <aside className="w-full lg:w-80 shrink-0">
          <WatchlistSidebar activeSymbol={selectedSymbol} onSelectSymbol={(sym) => setSelectedSymbol(sym)} />
        </aside>

        {/* Main Terminal Workspace (Dynamically Tailored by Journey) */}
        <section className="flex-1 flex flex-col space-y-6 min-w-0">
          {/* Top Interactive Candlestick Chart */}
          <div className="h-[440px] w-full">
            <PriceChart
              symbol={selectedSymbol}
              candles={data?.candles || []}
              currentPrice={data?.currentPrice}
              priceChangePct={data?.priceChangePct24h}
              interval={interval}
              onIntervalChange={(newInterval) => setInterval(newInterval)}
              technicals={data?.technicals}
            />
          </div>

          {/* DUAL-JOURNEY DYNAMIC VIEW */}
          {userRole === "DAY_TRADER" ? (
            <>
              {/* Day Trader Primary: Intraday Position Sizer & Execution Targets */}
              {data && (
                <DayTraderPositionSizer symbol={selectedSymbol} data={data} />
              )}

              {/* Day Trader Secondary: Self-Healing Accuracy & Walk-Forward Audit */}
              <SelfHealingAccuracyCard
                symbol={selectedSymbol}
                auditData={data?.selfHealingAudit}
              />

              {/* Day Trader Tertiary: Tail-Risk & Benchmark Ratios */}
              <RiskMetricsCard analyticsData={data || undefined} />

              {/* Day Trader Quaternary: Institutional Strategy Snapshot */}
              <TraderArchetypesCard
                symbol={selectedSymbol}
                traderArchetypes={data?.traderArchetypes}
              />
            </>
          ) : (
            <>
              {/* Long-Term Primary: 5-Factor Fundamental Scorecard & Macro Intelligence */}
              <AssetFactorRadar
                symbol={selectedSymbol}
                factorScores={data?.factorScores}
                macroDifficulty={data?.macroDifficulty}
                expectedReturn={data?.expectedReturn}
              />

              {/* Long-Term Secondary: Market Graph & Contagion Engine */}
              <MarketGraphCard
                symbol={selectedSymbol}
                marketGraph={data?.marketGraph}
              />

              {/* Long-Term Tertiary: Institutional Multi-Strategy Consensus */}
              <TraderArchetypesCard
                symbol={selectedSymbol}
                traderArchetypes={data?.traderArchetypes}
              />

              {/* Long-Term Quaternary: Self-Healing Forecast Auditor */}
              <SelfHealingAccuracyCard
                symbol={selectedSymbol}
                auditData={data?.selfHealingAudit}
              />

              {/* Long-Term Quinary: Risk & Distribution Analytics */}
              <RiskMetricsCard analyticsData={data || undefined} />
            </>
          )}
        </section>
      </main>
    </div>
  );
}

