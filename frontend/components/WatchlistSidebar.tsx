"use client";

import { useState } from "react";

interface WatchlistSidebarProps {
  activeSymbol: string;
  onSelectSymbol: (symbol: string) => void;
}

const WATCHLIST_ITEMS = [
  { symbol: "BTC-USD", name: "Bitcoin", price: "$64,250", change: "+4.50%", isUp: true, type: "Crypto" },
  { symbol: "ETH-USD", name: "Ethereum", price: "$3,420", change: "+3.20%", isUp: true, type: "Crypto" },
  { symbol: "NVDA", name: "NVIDIA Corp.", price: "$128.40", change: "+3.14%", isUp: true, type: "Stock" },
  { symbol: "AAPL", name: "Apple Inc.", price: "$226.50", change: "+1.25%", isUp: true, type: "Stock" },
  { symbol: "MSFT", name: "Microsoft", price: "$448.90", change: "-0.45%", isUp: false, type: "Stock" },
  { symbol: "SPY", name: "S&P 500 ETF", price: "$560.80", change: "+0.65%", isUp: true, type: "ETF" },
  { symbol: "QQQ", name: "Invesco QQQ", price: "$485.30", change: "+1.10%", isUp: true, type: "ETF" },
  { symbol: "SOL-USD", name: "Solana", price: "$145.80", change: "+6.80%", isUp: true, type: "Crypto" },
];

export default function WatchlistSidebar({ activeSymbol, onSelectSymbol }: WatchlistSidebarProps) {
  const [isMobileExpanded, setIsMobileExpanded] = useState(false);
  const [activeCategory, setActiveCategory] = useState<"All" | "Stock" | "ETF" | "Crypto">("All");

  const filteredItems = WATCHLIST_ITEMS.filter((item) =>
    activeCategory === "All" ? true : item.type === activeCategory
  );

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
            onClick={() => setActiveCategory(cat)}
            className={`flex-1 py-1 rounded text-[10px] font-mono font-medium transition-colors ${
              activeCategory === cat
                ? "bg-cyan-500 text-slate-950 font-bold"
                : "text-slate-400 hover:text-slate-200"
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

