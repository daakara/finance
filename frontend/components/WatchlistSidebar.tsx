"use client";

import { useState } from "react";

interface WatchlistSidebarProps {
  activeSymbol: string;
  onSelectSymbol: (symbol: string) => void;
}

const WATCHLIST_ITEMS = [
  { symbol: "BTC-USD", name: "Bitcoin", price: "$78,213", change: "+2.80%", isUp: true, type: "Crypto" },
  { symbol: "ETH-USD", name: "Ethereum", price: "$2,438", change: "+1.65%", isUp: true, type: "Crypto" },
  { symbol: "AAPL", name: "Apple Inc.", price: "$309.90", change: "-0.45%", isUp: false, type: "Stock" },
  { symbol: "NVDA", name: "NVIDIA Corp.", price: "$213.05", change: "+3.14%", isUp: true, type: "Stock" },
  { symbol: "MSFT", name: "Microsoft", price: "$491.71", change: "+0.85%", isUp: true, type: "Stock" },
  { symbol: "GOOGL", name: "Alphabet Inc.", price: "$346.96", change: "+1.40%", isUp: true, type: "Stock" },
  { symbol: "TSLA", name: "Tesla Inc.", price: "$350.25", change: "+2.15%", isUp: true, type: "Stock" },
  { symbol: "SPY", name: "S&P 500 ETF", price: "$765.91", change: "+0.65%", isUp: true, type: "ETF" },
  { symbol: "QQQ", name: "Invesco QQQ", price: "$710.72", change: "+1.10%", isUp: true, type: "ETF" },
  { symbol: "SOL-USD", name: "Solana", price: "$96.73", change: "+0.24%", isUp: true, type: "Crypto" },
];

export default function WatchlistSidebar({ activeSymbol, onSelectSymbol }: WatchlistSidebarProps) {
  const [isMobileExpanded, setIsMobileExpanded] = useState(false);
  const [activeCategory, setActiveCategory] = useState<"All" | "Stock" | "ETF" | "Crypto">("All");

  const filteredItems = WATCHLIST_ITEMS.filter((item) =>
    activeCategory === "All" ? true : item.type === activeCategory
  );

  const handleCategoryClick = (cat: "All" | "Stock" | "ETF" | "Crypto") => {
    setActiveCategory(cat);
    setIsMobileExpanded(true);

    if (cat !== "All") {
      const topItem = WATCHLIST_ITEMS.find((item) => item.type === cat);
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
          <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
            Watchlist & Signals
          </span>
          <span className="text-[10px] text-cyan-400 bg-cyan-950/60 border border-cyan-800/80 px-2 py-0.5 rounded">
            Live Feeds
          </span>
        </div>

        {/* Accordion toggle button with tactile press feedback */}
        <button
          onClick={() => setIsMobileExpanded((prev) => !prev)}
          className="lg:hidden text-xs text-cyan-400 bg-[#090d14] border border-[#243044] px-2.5 py-1.5 min-h-[32px] rounded flex items-center gap-1.5 hover:bg-[#162030] active:scale-[0.96] transition-transform duration-100"
        >
          <span>{isMobileExpanded ? "Hide List" : "Show List"}</span>
          <span className="text-[10px]">{isMobileExpanded ? "▲" : "▼"}</span>
        </button>
      </div>

      {/* Asset Category Filters with 40px Touch Target on Mobile */}
      <div className="flex items-center space-x-1 bg-[#090d14] p-1 rounded-lg border border-[#243044]">
        {(["All", "Stock", "ETF", "Crypto"] as const).map((cat) => (
          <button
            key={cat}
            onClick={() => handleCategoryClick(cat)}
            className={`flex-1 py-1.5 sm:py-1 min-h-[36px] sm:min-h-[28px] rounded text-[11px] sm:text-[10px] font-bold transition-colors active:scale-[0.96] transition-transform duration-100 ${
              activeCategory === cat
                ? "bg-cyan-500 text-slate-950 shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Watchlist Stream Items with Tabular Numerals & Tactile Click */}
      <div
        className={`space-y-1.5 overflow-y-auto max-h-[440px] ${
          isMobileExpanded ? "block" : "hidden lg:block"
        }`}
      >
        {filteredItems.map((item) => (
          <button
            key={item.symbol}
            onClick={() => onSelectSymbol(item.symbol)}
            className={`w-full flex items-center justify-between p-2.5 sm:p-2 rounded-lg border text-left active:scale-[0.98] transition-transform transition-colors duration-150 ${
              activeSymbol === item.symbol
                ? "bg-[#1b2434] border-cyan-500 shadow-md"
                : "bg-[#090d14] border-[#1b2434] hover:border-[#364866]"
            }`}
          >
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xs font-bold text-slate-100">{item.symbol}</span>
                <span className="text-[9px] text-slate-400 px-1 py-0.2 rounded bg-[#162030]">
                  {item.type}
                </span>
              </div>
              <span className="text-[10px] text-slate-400 block truncate max-w-[120px]">{item.name}</span>
            </div>

            <div className="text-right">
              <span className="text-xs font-bold text-slate-200 block tabular-nums">{item.price}</span>
              <span
                className={`text-[10px] font-semibold tabular-nums ${
                  item.isUp ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {item.change}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

