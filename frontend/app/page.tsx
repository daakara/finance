"use client";

import { useEffect, useState } from "react";
import TradingViewChart from "../components/TradingViewChart";
import RiskMetricsCard from "../components/RiskMetricsCard";
import WatchlistSidebar from "../components/WatchlistSidebar";
import AssetFactorRadar from "../components/AssetFactorRadar";
import TraderArchetypesCard from "../components/TraderArchetypesCard";
import { AnalyticsResponse, fetchAssetAnalytics } from "../lib/api";

const POPULAR_TICKERS = ["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA", "BTC-USD", "ETH-USD", "SOL-USD", "SPY", "QQQ"];

export default function DashboardPage() {
  const [symbol, setSymbol] = useState("AAPL");
  const [searchInput, setSearchInput] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [loading, setLoading] = useState(true);
  const [analyticsData, setAnalyticsData] = useState<AnalyticsResponse | null>(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const symParam = params.get("symbol");
      if (symParam) {
        setSymbol(symParam.toUpperCase());
      }
    }
  }, []);

  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      setLoading(true);
      try {
        const data = await fetchAssetAnalytics(symbol);
        if (isMounted) {
          setAnalyticsData(data);
        }
      } catch (err) {
        console.error("Error loading asset analytics:", err);
      } finally {
        if (isMounted) setLoading(false);
      }
    }
    loadData();
    return () => {
      isMounted = false;
    };
  }, [symbol]);

  const filteredTickers = POPULAR_TICKERS.filter((t) =>
    t.toLowerCase().includes(searchInput.toLowerCase())
  );

  const handleSelectSymbol = (newSymbol: string) => {
    setSymbol(newSymbol.toUpperCase());
    setSearchInput("");
    setIsSearching(false);
  };

  const handleCustomSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchInput.trim()) {
      handleSelectSymbol(searchInput.trim());
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 bg-[#111722] border border-[#243044] rounded-xl p-6 shadow-xl">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
            <h1 className="text-2xl sm:text-3xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
              Quantitative Market & Multi-Factor Terminal
            </h1>
          </div>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Live multi-asset data, 5-factor quantitative profiling, FRED macroeconomic indicators & iconic trader archetype consensus
          </p>
        </div>

        {/* Interactive Asset Search Combobox */}
        <div className="relative min-w-[260px]">
          <form onSubmit={handleCustomSubmit} className="relative">
            <input
              type="text"
              value={searchInput}
              onChange={(e) => {
                setSearchInput(e.target.value);
                setIsSearching(true);
              }}
              onFocus={() => setIsSearching(true)}
              placeholder="Search ticker (e.g. BTC, NVDA)..."
              className="w-full bg-[#090d14] border border-[#243044] focus:border-cyan-500 text-slate-100 text-sm rounded-lg px-3.5 py-2 pr-8 font-mono focus-ring placeholder-slate-500"
            />
            <button type="submit" className="absolute right-2.5 top-2.5 text-slate-400 hover:text-cyan-400 text-xs">
              ??
            </button>
          </form>

          {/* Autocomplete Suggestions Dropdown */}
          {isSearching && (
            <div className="absolute right-0 mt-1 w-full bg-[#162030] border border-[#364866] rounded-lg shadow-2xl z-50 overflow-hidden py-1">
              {filteredTickers.length > 0 ? (
                filteredTickers.map((t) => (
                  <button
                    key={t}
                    onClick={() => handleSelectSymbol(t)}
                    className="w-full text-left px-3.5 py-2 text-sm text-slate-200 hover:bg-[#1b2434] hover:text-cyan-400 font-mono transition-colors flex items-center justify-between"
                  >
                    <span>{t}</span>
                    <span className="text-[10px] text-slate-400">Select</span>
                  </button>
                ))
              ) : (
                <button
                  onClick={() => handleSelectSymbol(searchInput)}
                  className="w-full text-left px-3.5 py-2 text-sm text-cyan-400 hover:bg-[#1b2434] font-mono"
                >
                  Load &quot;{searchInput.toUpperCase()}&quot;
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Responsive 3-Pane Terminal Grid Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">
        {/* Left Watchlist Sidebar (1 Col) */}
        <div className="lg:col-span-1">
          <WatchlistSidebar activeSymbol={symbol} onSelectSymbol={handleSelectSymbol} />
        </div>

        {/* Center Main Interactive Chart, 5-Factor Profile, Trader Archetypes & Bottom Risk Grid (3 Cols) */}
        <div className="lg:col-span-3 space-y-6">
          {/* Main Chart */}
          <TradingViewChart symbol={symbol} data={analyticsData?.candles} />

          {/* 5-Factor Quantitative Asset Profile & FRED Macro Grid */}
          <AssetFactorRadar
            symbol={symbol}
            factorScores={analyticsData?.factorScores || analyticsData?.dnaScores}
            macroDifficulty={analyticsData?.macroDifficulty}
            expectedReturn={analyticsData?.expectedReturn}
          />

          {/* Iconic Trader Archetypes & Smart-Money Alignment */}
          <TraderArchetypesCard
            symbol={symbol}
            traderArchetypes={analyticsData?.traderArchetypes}
          />

          {/* Advanced Risk Metrics */}
          <RiskMetricsCard analyticsData={analyticsData || undefined} />

          {/* Legal Compliance & Financial Disclaimer */}
          <div className="bg-[#090d14] border border-[#243044] rounded-lg p-4 text-[11px] text-slate-500 font-mono leading-relaxed">
            <span className="text-slate-400 font-bold block mb-1">?? REGULATORY & COMPLIANCE DISCLAIMER:</span>
            Antigravity Quantitative Market Terminal is strictly an educational and quantitative research software tool. 
            All metrics, factor ratings, expected return estimates, volatility forecasts, and trader archetype alignment models represent algorithmic statistical heuristics and do not constitute financial, investment, tax, or legal advice. 
            Past statistical performance does not guarantee future results.
          </div>
        </div>
      </div>
    </div>
  );
}

