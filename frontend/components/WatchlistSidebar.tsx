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

    // Auto-select the top asset in this category if current asset is not in it
    if (cat !== "All") {
      const topItem = WATCHLIST_ITEMS.find((item) => item.type === cat);
      if (topItem && activeSymbol !== topItem.symbol) {
        onSelectSymbol(topItem.symbol);
      }
    }
  };

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-4 shadow-xl space-y-3 h-full">
      {/* Header with Mobile Accordion Toggle */}
      <div className="flex items-center justify-between border-b border-[#1b2434] pb-3">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">
            Watchlist & Signals
          </span>
          <span className="text-[10px] text-cyan-400 bg-cyan-950/60 border border-cyan-800/80 px-2 py-0.5 rounded font-mono">
            Live Feeds
          </span>
        </div>

        {/* Accordion toggle button visible on mobile (< 1024px) */}
        <button
          onClick={() => setIsMobileExpanded((prev) => !prev)}
          className="lg:hidden text-xs text-cyan-400 bg-[#090d14] border border-[#243044] px-2.5 py-1 rounded font-mono flex items-center gap-1"
        >
          <span>{isMobileExpanded ? "Collapse ?" : "Expand ?"}</span>
        </button>
      </div>

      {/* Asset Category Filters */}
      <div className="flex items-center space-x-1 bg-[#090d14] p-1 rounded-lg border border-[#243044]">
        {(["All", "Stock", "ETF", "Crypto"] as const).map((cat) => (
          <button
            key={cat}
            onClick={() => handleCategoryClick(cat)}
            className={`flex-1 py-1 rounded text-[10px] font-mono font-medium transition-colors ${
              activeCategory === cat
                ? "bg-cyan-500 text-slate-950 font-bold shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Watchlist Stream Items */}
      <div
        className={`space-y-1.5 overflow-y-auto max-h-[440px] transition-all duration-200 ${
          isMobileExpanded ? "block" : "hidden lg:block"
        }`}
      >
        {filteredItems.map((item) => (
          <button
            key={item.symbol}
            onClick={() => {
              onSelectSymbol(item.symbol);
              setIsMobileExpanded(false);
            }}
            className={`w-full text-left p-2.5 rounded-lg transition-all flex items-center justify-between border min-h-[44px] ${
              activeSymbol === item.symbol
                ? "bg-[#1b2434] border-cyan-500/60 text-slate-100 shadow-md"
                : "bg-[#090d14] border-[#243044] text-slate-300 hover:border-[#364866]"
            }`}
          >
            <div>
              <div className="flex items-center space-x-1.5">
                <span className="font-bold font-mono text-sm block leading-tight text-slate-100">
                  {item.symbol}
                </span>
                <span className="text-[9px] px-1.5 py-0.2 rounded font-mono bg-[#1b2434] text-slate-400 border border-[#364866]">
                  {item.type}
                </span>
              </div>
              <span className="text-[10px] text-slate-400 block mt-0.5">{item.name}</span>
            </div>

            <div className="text-right font-mono">
              <span className="text-xs font-semibold block text-slate-200">{item.price}</span>
              <span className={`text-[10px] font-bold ${item.isUp ? "text-emerald-400" : "text-rose-400"}`}>
                {item.change}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

