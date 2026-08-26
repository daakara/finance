"use client";

import { useEffect, useState } from "react";
import { fetchAssetAnalytics } from "../lib/api";

interface WatchlistSidebarProps {
  activeSymbol: string;
  onSelectSymbol: (symbol: string) => void;
}

interface WatchlistItem {
  symbol: string;
  name: string;
  price: string;
  change: string;
  isUp: boolean;
  type: "Stock" | "ETF" | "Crypto";
}

const INITIAL_WATCHLIST_ITEMS: WatchlistItem[] = [
  // Mega-Cap Tech & Global Pharma Equities
  { symbol: "NVO", name: "Novo Nordisk", price: "$138.50", change: "+1.85%", isUp: true, type: "Stock" },
  { symbol: "LLY", name: "Eli Lilly", price: "$920.40", change: "+2.10%", isUp: true, type: "Stock" },
  { symbol: "AAPL", name: "Apple Inc.", price: "$309.90", change: "-0.45%", isUp: false, type: "Stock" },
  { symbol: "NVDA", name: "NVIDIA Corp.", price: "$213.05", change: "+3.14%", isUp: true, type: "Stock" },
  { symbol: "MSFT", name: "Microsoft Corp.", price: "$491.71", change: "+0.85%", isUp: true, type: "Stock" },
  { symbol: "GOOGL", name: "Alphabet Inc.", price: "$346.96", change: "+1.40%", isUp: true, type: "Stock" },
  { symbol: "TSLA", name: "Tesla Inc.", price: "$350.25", change: "+2.15%", isUp: true, type: "Stock" },
  { symbol: "PLTR", name: "Palantir Tech", price: "$142.80", change: "+4.12%", isUp: true, type: "Stock" },

  // Institutional Index, Sector, Commodity & Bond ETFs
  { symbol: "SPY", name: "S&P 500 ETF", price: "$765.91", change: "+0.65%", isUp: true, type: "ETF" },
  { symbol: "QQQ", name: "Invesco QQQ", price: "$710.72", change: "+1.10%", isUp: true, type: "ETF" },
  { symbol: "SMH", name: "VanEck Semi ETF", price: "$288.40", change: "+2.45%", isUp: true, type: "ETF" },
  { symbol: "XLK", name: "Tech Select SPDR", price: "$246.15", change: "+1.30%", isUp: true, type: "ETF" },
  { symbol: "IWM", name: "Russell 2000 ETF", price: "$224.50", change: "+0.95%", isUp: true, type: "ETF" },
  { symbol: "GLD", name: "SPDR Gold Shares", price: "$264.20", change: "+0.40%", isUp: true, type: "ETF" },
  { symbol: "TLT", name: "20+ Yr Treasury", price: "$88.65", change: "-0.30%", isUp: false, type: "ETF" },
  { symbol: "XLE", name: "Energy Select ETF", price: "$86.10", change: "+1.05%", isUp: true, type: "ETF" },

  // Digital Assets / Crypto
  { symbol: "BTC-USD", name: "Bitcoin", price: "$78,213", change: "+2.80%", isUp: true, type: "Crypto" },
  { symbol: "ETH-USD", name: "Ethereum", price: "$2,438", change: "+1.65%", isUp: true, type: "Crypto" },
  { symbol: "SOL-USD", name: "Solana", price: "$96.73", change: "+0.24%", isUp: true, type: "Crypto" },
];

export default function WatchlistSidebar({ activeSymbol, onSelectSymbol }: WatchlistSidebarProps) {
  const [items, setItems] = useState<WatchlistItem[]>(INITIAL_WATCHLIST_ITEMS);
  const [isMobileExpanded, setIsMobileExpanded] = useState(false);
  const [activeCategory, setActiveCategory] = useState<"All" | "Stock" | "ETF" | "Crypto">("All");

  const filteredItems = items.filter((item) =>
    activeCategory === "All" ? true : item.type === activeCategory
  );

  const handleCategoryClick = (cat: "All" | "Stock" | "ETF" | "Crypto") => {
    setActiveCategory(cat);
    setIsMobileExpanded(true);

    if (cat !== "All") {
      const topItem = items.find((item) => item.type === cat);
      if (topItem && activeSymbol !== topItem.symbol) {
        onSelectSymbol(topItem.symbol);
      }
    }
  };

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-3.5 sm:p-4 shadow-xl space-y-3 h-full font-mono">
      {/* Header with Mobile Accordion Toggle */}
      <div className="flex items-center justify-between border-b border-[#1b2434] pb-3">
        <div className="flex items-center space-x-2">
          <h2 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
            Watchlist & Signals
          </h2>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#1e293b] text-cyan-400 font-semibold">
            LIVE
          </span>
        </div>

        {/* Mobile-Only Accordion Trigger */}
        <button
          onClick={() => setIsMobileExpanded(!isMobileExpanded)}
          aria-expanded={isMobileExpanded}
          aria-controls="watchlist-items-container"
          className="sm:hidden flex items-center space-x-1 text-xs text-cyan-400 font-bold px-2 py-1 bg-[#162030] rounded border border-[#243044] active:scale-[0.96] transition-transform"
        >
          <span>{isMobileExpanded ? "Hide Assets" : `Show (${filteredItems.length})`}</span>
          <svg
            className={`w-3.5 h-3.5 transform transition-transform ${isMobileExpanded ? "rotate-180" : ""}`}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </button>
      </div>

      {/* Asset Category Filters */}
      <div role="tablist" aria-label="Asset Class Filter" className="grid grid-cols-4 gap-1 p-1 bg-[#090d14] rounded-lg border border-[#1b2434] text-[11px]">
        {(["All", "Stock", "ETF", "Crypto"] as const).map((cat) => (
          <button
            key={cat}
            role="tab"
            aria-selected={activeCategory === cat}
            onClick={() => handleCategoryClick(cat)}
            className={`py-1 rounded font-bold transition-all active:scale-[0.96] ${
              activeCategory === cat
                ? "bg-cyan-500 text-slate-950 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Asset Items List */}
      <div
        id="watchlist-items-container"
        role="region"
        aria-label="Asset List"
        className={`space-y-1.5 overflow-y-auto max-h-[480px] pr-1 transition-all ${
          isMobileExpanded ? "block" : "hidden sm:block"
        }`}
      >
        {filteredItems.map((item) => {
          const itemClean = item.symbol.toUpperCase().replace("-USD", "");
          const activeClean = activeSymbol.toUpperCase().replace("-USD", "");
          const isSelected = activeClean === itemClean;
          return (
            <button
              key={item.symbol}
              onClick={() => onSelectSymbol(item.symbol)}
              aria-label={`Select ${item.name} (${item.symbol}), Price ${item.price}, Change ${item.change}`}
              className={`w-full flex items-center justify-between p-2.5 rounded-lg border text-left transition-all active:scale-[0.98] ${
                isSelected
                  ? "bg-[#162030] border-cyan-500 shadow-md shadow-cyan-950/40"
                  : "bg-[#0b1019] border-[#1b2434] hover:bg-[#131b28] hover:border-[#2b3a52]"
              }`}
            >
              <div>
                <div className="flex items-center space-x-1.5">
                  <span className="font-bold text-xs text-white">{item.symbol}</span>
                  <span className="text-[9px] px-1 py-0.2 rounded bg-[#1e293b] text-slate-400">
                    {item.type}
                  </span>
                </div>
                <div className="text-[11px] text-slate-400 truncate max-w-[120px]">
                  {item.name}
                </div>
              </div>

              <div className="text-right">
                <div className="text-xs font-bold text-slate-200 tabular-nums">{item.price}</div>
                <div
                  className={`text-[10px] font-semibold tabular-nums ${
                    item.isUp ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {item.change}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}