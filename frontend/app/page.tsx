"use client";

import { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import WatchlistSidebar from "../components/WatchlistSidebar";
import PriceChart from "../components/PriceChart";
import RiskMetricsCard from "../components/RiskMetricsCard";
import AssetFactorRadar from "../components/AssetFactorRadar";
import TraderArchetypesCard from "../components/TraderArchetypesCard";
import DayTraderPositionSizer from "../components/DayTraderPositionSizer";
import { fetchAssetAnalytics, AnalyticsResponse } from "../lib/api";

export default function TerminalPage() {
  const [selectedSymbol, setSelectedSymbol] = useState<string>("AAPL");
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [interval, setInterval] = useState<string>("1d");

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
      <Navbar />

      <main className="flex-1 flex flex-col lg:flex-row p-3 md:p-6 gap-6 max-w-[1750px] w-full mx-auto">
        {/* Watchlist Sidebar */}
        <aside className="w-full lg:w-80 shrink-0">
          <WatchlistSidebar activeSymbol={selectedSymbol} onSelectSymbol={(sym) => setSelectedSymbol(sym)} />
        </aside>

        {/* Main Terminal Workspace */}
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

          {/* Day Trader Intraday Risk & Position Sizer */}
          {data && (
            <DayTraderPositionSizer symbol={selectedSymbol} data={data} />
          )}

          {/* 5-Factor Fundamental & Factor Profile Card */}
          <AssetFactorRadar
            symbol={selectedSymbol}
            factorScores={data?.factorScores}
            macroDifficulty={data?.macroDifficulty}
            expectedReturn={data?.expectedReturn}
          />

          {/* Institutional Strategy & Trader Archetypes Consensus Card */}
          <TraderArchetypesCard
            symbol={selectedSymbol}
            traderArchetypes={data?.traderArchetypes}
          />

          {/* Advanced Risk Management & Tail-Risk Grid */}
          <RiskMetricsCard analyticsData={data || undefined} />
        </section>
      </main>
    </div>
  );
}

