"use client";

import { useState, useRef, useEffect } from "react";
import { SHARED_WATCHLIST_ITEMS, WatchlistDefinition } from "../lib/constants";

interface WatchlistSidebarProps {
  activeSymbol: string;
  onSelectSymbol: (symbol: string) => void;
}

export default function WatchlistSidebar({ activeSymbol, onSelectSymbol }: WatchlistSidebarProps) {
  const [items, setItems] = useState<WatchlistDefinition[]>(SHARED_WATCHLIST_ITEMS);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [isMobileExpanded, setIsMobileExpanded] = useState(false);
  const [activeCategory, setActiveCategory] = useState<"All" | "Stock" | "ETF" | "Crypto">("All");
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Global keyboard shortcut: Press "/" to focus search input
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.key === "/" &&
        document.activeElement?.tagName !== "INPUT" &&
        document.activeElement?.tagName !== "TEXTAREA"
      ) {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const cleanQuery = searchQuery.trim().toUpperCase();

  // Filter items by category AND search query (ticker or company name)
  const filteredItems = items.filter((item) => {
    const matchesCategory = activeCategory === "All" ? true : item.type === activeCategory;
    if (!matchesCategory) return false;

    if (!cleanQuery) return true;
    const matchSymbol = item.symbol.toUpperCase().includes(cleanQuery);
    const matchName = item.name.toUpperCase().includes(cleanQuery);
    return matchSymbol || matchName;
  });

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

  // Submit custom unlisted ticker
  const handleCustomTickerSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (cleanQuery) {
      onSelectSymbol(cleanQuery);
      setSearchQuery("");
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

      {/* 🔍 Search & Quick-Add Input Bar */}
      <form onSubmit={handleCustomTickerSubmit} className="relative">
        <div className="relative flex items-center">
          <svg
            className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 pointer-events-none"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            ref={searchInputRef}
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search ticker, name, or '/'..."
            className="w-full bg-[#090d14] border border-[#243044] rounded-lg pl-8 pr-7 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-colors"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => setSearchQuery("")}
              className="absolute right-2 text-slate-400 hover:text-white text-xs font-bold"
              aria-label="Clear search"
            >
              ✕
            </button>
          )}
        </div>
      </form>

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
        {filteredItems.length === 0 ? (
          <div className="p-3 text-center bg-[#090d14] rounded-lg border border-[#1e293b] space-y-2">
            <p className="text-xs text-slate-400">
              No preset asset matching &quot;{searchQuery}&quot;
            </p>
            {cleanQuery && (
              <button
                type="button"
                onClick={() => {
                  onSelectSymbol(cleanQuery);
                  setSearchQuery("");
                }}
                className="w-full bg-cyan-950 hover:bg-cyan-900 text-cyan-300 border border-cyan-700 py-1.5 px-2 rounded-md text-xs font-bold transition-transform active:scale-95 flex items-center justify-center gap-1.5 shadow-md shadow-cyan-950/40"
              >
                <span>➕</span>
                <span>Analyze &quot;{cleanQuery}&quot; in Terminal</span>
              </button>
            )}
          </div>
        ) : (
          filteredItems.map((item) => {
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
          })
        )}
      </div>
    </div>
  );
}