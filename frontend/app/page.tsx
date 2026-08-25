"use client";

import { useState } from "react";
import TradingViewChart from "../components/TradingViewChart";
import RiskMetricsCard from "../components/RiskMetricsCard";

const POPULAR_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "BTC-USD", "ETH-USD"];

export default function DashboardPage() {
  const [symbol, setSymbol] = useState("AAPL");
  const [searchInput, setSearchInput] = useState("");
  const [isSearching, setIsSearching] = useState(false);

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
          <h1 className="text-2xl sm:text-3xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            Quantitative Market Terminal
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Real-time multi-asset technical indicators, GARCH volatility forecasting & Cornish-Fisher VaR metrics
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
              placeholder="Search ticker (e.g. NVDA)..."
              className="w-full bg-[#090d14] border border-[#243044] focus:border-cyan-500 text-slate-100 text-sm rounded-lg px-3.5 py-2 pr-8 font-mono focus-ring placeholder-slate-500"
            />
            <button type="submit" className="absolute right-2.5 top-2.5 text-slate-400 hover:text-cyan-400 text-xs">
              ?
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

      {/* Quick Select Pill Buttons */}
      <div className="flex items-center space-x-2 overflow-x-auto pb-1">
        <span className="text-xs font-mono text-slate-400 uppercase tracking-wider pr-2">Favorites:</span>
        {POPULAR_TICKERS.map((sym) => (
          <button
            key={sym}
            onClick={() => handleSelectSymbol(sym)}
            className={`px-3 py-1 rounded-md text-xs font-mono font-medium transition-colors focus-ring ${
              symbol === sym
                ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/50"
                : "bg-[#111722] text-slate-400 border border-[#243044] hover:bg-[#162030] hover:text-slate-200"
            }`}
          >
            {sym}
          </button>
        ))}
      </div>

      {/* Main Chart & Risk Analytics Grid */}
      <TradingViewChart symbol={symbol} />
      <RiskMetricsCard />
    </div>
  );
}

