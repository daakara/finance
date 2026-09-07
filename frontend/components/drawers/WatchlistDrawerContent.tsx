"use client";

import React, { useState, useRef, useEffect } from "react";
import { SHARED_WATCHLIST_ITEMS, WatchlistDefinition } from "../../lib/constants";
import { fetchBatchQuotes } from "../../lib/api";
import { getAllPersistedMarketSnapshots } from "../../lib/marketDatabase";
import { getMasterBaselineQuote } from "../../lib/masterCatalog";
import MiniSparkline from "../MiniSparkline";

interface WatchlistItemDisplay extends WatchlistDefinition {
  price?: string;
  change?: string;
  isUp?: boolean;
}

export interface WatchlistDrawerContentProps {
  activeSymbol: string;
  onSelectSymbol: (symbol: string) => void;
  liveCurrentPrice?: number;
  livePriceChangePct?: number;
  onClose?: () => void;
}

export default function WatchlistDrawerContent({
  activeSymbol,
  onSelectSymbol,
  liveCurrentPrice,
  livePriceChangePct,
  onClose,
}: WatchlistDrawerContentProps) {
  // Initialize with persisted fresh database snapshots or SSOT master baseline quotes
  const [items, setItems] = useState<WatchlistItemDisplay[]>(() => {
    const snapshots = typeof window !== "undefined" ? getAllPersistedMarketSnapshots(true) : {};
    return SHARED_WATCHLIST_ITEMS.map((item) => {
      const symClean = item.symbol.toUpperCase().replace("-USD", "");
      const snap = snapshots[symClean];
      if (snap && snap.currentPrice > 0 && Math.abs(snap.currentPrice - 319.64) >= 0.01) {
        const isUp = snap.priceChangePct24h >= 0;
        return {
          ...item,
          price: `$${snap.currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
          change: `${isUp ? "+" : ""}${snap.priceChangePct24h.toFixed(2)}%`,
          isUp,
        };
      }
      const baseline = getMasterBaselineQuote(item.symbol);
      if (baseline.spot === undefined) {
        return {
          ...item,
          price: "Unavailable",
          change: "—",
          isUp: true,
        };
      }
      const isUp = (baseline.pctChange ?? 0) >= 0;
      return {
        ...item,
        price: `$${baseline.spot.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
        change: `${isUp ? "+" : ""}${(baseline.pctChange ?? 0).toFixed(2)}%`,
        isUp,
      };
    });
  });

  // Batch-fetch real-time live quotes on mount
  useEffect(() => {
    let isMounted = true;
    const symbols = SHARED_WATCHLIST_ITEMS.map((i) => i.symbol);

    fetchBatchQuotes(symbols).then((quotes) => {
      if (!isMounted || !quotes || Object.keys(quotes).length === 0) return;

      setItems((prevItems) =>
        prevItems.map((item) => {
          const symClean = item.symbol.toUpperCase().replace("-USD", "");
          const liveQuote = quotes[symClean];
          if (liveQuote && liveQuote.price > 0 && (symClean === "AAPL" || Math.abs(liveQuote.price - 319.64) >= 0.01)) {
            const isUp = liveQuote.changePct >= 0;
            return {
              ...item,
              price: `$${liveQuote.price.toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}`,
              change: `${isUp ? "+" : ""}${liveQuote.changePct.toFixed(2)}%`,
              isUp,
            };
          }
          return item;
        })
      );
    });

    return () => {
      isMounted = false;
    };
  }, []);

  // Synchronize incoming live current price and percent change with active symbol
  useEffect(() => {
    if (liveCurrentPrice !== undefined && activeSymbol) {
      const symClean = activeSymbol.toUpperCase().replace("-USD", "");
      const isUp = (livePriceChangePct ?? 0) >= 0;
      const changeStr = `${isUp ? "+" : ""}${(livePriceChangePct ?? 0).toFixed(2)}%`;
      const priceStr = `$${liveCurrentPrice.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`;

      setItems((prevItems) =>
        prevItems.map((item) => {
          const itemClean = item.symbol.toUpperCase().replace("-USD", "");
          if (itemClean === symClean) {
            return {
              ...item,
              price: priceStr,
              change: changeStr,
              isUp: isUp,
            };
          }
          return item;
        })
      );
    }
  }, [liveCurrentPrice, livePriceChangePct, activeSymbol]);

  const [pinnedSymbols, setPinnedSymbols] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [activeCategory, setActiveCategory] = useState<"All" | "Pinned" | "Stock" | "ETF" | "Crypto">("All");
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Load pinned symbols from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem("FINANCE_PINNED_SYMBOLS");
      if (saved) {
        setPinnedSymbols(JSON.parse(saved));
      } else {
        const defaultPins = ["AAPL", "NVDA", "BTC-USD"];
        setPinnedSymbols(defaultPins);
        localStorage.setItem("FINANCE_PINNED_SYMBOLS", JSON.stringify(defaultPins));
      }
    } catch {
      setPinnedSymbols(["AAPL", "NVDA"]);
    }
  }, []);

  const togglePin = (e: React.MouseEvent, sym: string) => {
    e.stopPropagation();
    const cleanSym = sym.toUpperCase();
    let updated: string[];
    if (pinnedSymbols.includes(cleanSym)) {
      updated = pinnedSymbols.filter((s) => s !== cleanSym);
    } else {
      updated = [...pinnedSymbols, cleanSym];
    }
    setPinnedSymbols(updated);
    try {
      localStorage.setItem("FINANCE_PINNED_SYMBOLS", JSON.stringify(updated));
    } catch (err) {
      console.warn("Could not save pinned symbols:", err);
    }
  };

  const cleanQuery = searchQuery.trim().toUpperCase();

  const filteredItems = items.filter((item) => {
    const cleanItemSym = item.symbol.toUpperCase();
    if (activeCategory === "Pinned") {
      if (!pinnedSymbols.includes(cleanItemSym)) return false;
    } else if (activeCategory !== "All") {
      if (item.type !== activeCategory) return false;
    }

    if (!cleanQuery) return true;
    const matchSymbol = item.symbol.toUpperCase().includes(cleanQuery);
    const matchName = item.name.toUpperCase().includes(cleanQuery);
    return matchSymbol || matchName;
  });

  return (
    <div className="flex flex-col h-full bg-[#0a0e17] text-slate-100 divide-y divide-[#1e293b]">
      {/* Search and Category Filter Header */}
      <div className="p-3 space-y-2.5 shrink-0 bg-[#0f1422]">
        {/* Search Input with Hotkey notation */}
        <div className="relative">
          <input
            ref={searchInputRef}
            type="text"
            placeholder="Search tickers (Press '/' to focus)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-[#161f33] text-xs font-mono text-slate-100 placeholder-slate-400 pl-8 pr-7 py-2 rounded-lg border border-slate-700/70 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-all"
            data-testid="watchlist-search-input"
          />
          <svg
            className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-400"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-2.5 top-2 text-slate-400 hover:text-slate-200 text-xs"
              aria-label="Clear search"
            >
              ✕
            </button>
          )}
        </div>

        {/* Category Pills */}
        <div
          role="tablist"
          aria-label="Watchlist categories"
          className="flex items-center gap-1 overflow-x-auto pb-0.5 scrollbar-none"
        >
          {(["All", "Pinned", "Stock", "ETF", "Crypto"] as const).map((cat) => {
            const isSelected = activeCategory === cat;
            return (
              <button
                key={cat}
                role="tab"
                aria-selected={isSelected}
                onClick={() => setActiveCategory(cat)}
                className={`px-2 py-0.5 rounded text-[11px] font-mono whitespace-nowrap transition-colors ${
                  isSelected
                    ? "bg-slate-700 text-slate-50 font-semibold"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                }`}
                data-testid={`watchlist-cat-${cat.toLowerCase()}`}
              >
                {cat === "Pinned" ? `★ ${cat}` : cat}
              </button>
            );
          })}
        </div>
      </div>

      {/* Watchlist Items Scrollable List */}
      <div
        role="listbox"
        aria-label="Watchlist assets"
        className="flex-1 overflow-y-auto p-2 space-y-1.5 focus:outline-none"
        data-testid="watchlist-items-container"
        tabIndex={0}
      >
        {filteredItems.length === 0 ? (
          <div className="py-12 text-center text-slate-400 text-xs">
            No assets match your filter.
          </div>
        ) : (
          filteredItems.map((item) => {
            const isSelected = activeSymbol === item.symbol;
            const isPinned = pinnedSymbols.includes(item.symbol.toUpperCase());
            const isUp = item.isUp ?? true;

            return (
              <div
                key={item.symbol}
                role="option"
                tabIndex={0}
                aria-selected={isSelected}
                onClick={() => {
                  onSelectSymbol(item.symbol);
                  if (typeof window !== "undefined" && window.innerWidth < 1024 && onClose) {
                    onClose();
                  }
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelectSymbol(item.symbol);
                  }
                }}
                data-testid={`watchlist-item-${item.symbol}`}
                className={`group flex items-center justify-between p-2 rounded-lg cursor-pointer transition-colors border ${
                  isSelected
                    ? "bg-[#161f33] border-cyan-400/40 text-slate-50"
                    : "bg-[#0f1422]/60 hover:bg-[#161f33]/80 border-slate-800/60 text-slate-200"
                }`}
              >
                {/* Left: Pin + Ticker + Name */}
                <div className="flex items-center gap-2 min-w-0">
                  <button
                    type="button"
                    onClick={(e) => togglePin(e, item.symbol)}
                    aria-label={isPinned ? `Unpin ${item.symbol}` : `Pin ${item.symbol}`}
                    className={`text-xs transition-colors shrink-0 p-1 ${
                      isPinned
                        ? "text-amber-400 hover:text-amber-300"
                        : "text-slate-600 hover:text-slate-400"
                    }`}
                  >
                    ★
                  </button>

                  <div className="flex flex-col min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono font-bold text-xs text-slate-100">
                        {item.symbol}
                      </span>
                      <span className="text-[9px] font-mono px-1 py-0.2 rounded bg-slate-800/80 text-slate-400 border border-slate-700/50">
                        {item.type}
                      </span>
                    </div>
                    <span className="text-[10px] text-slate-400 truncate max-w-[110px] sm:max-w-[130px]">
                      {item.name}
                    </span>
                  </div>
                </div>

                {/* Right: Sparkline + Spot Price + Delta */}
                <div className="flex items-center gap-2.5 shrink-0">
                  <div className="w-12 h-5 flex items-center">
                    <MiniSparkline
                      basePrice={parseFloat(item.price?.replace(/[^0-9.]/g, "") || "100") || 100}
                      changePct={parseFloat(item.change?.replace(/[^0-9.-]/g, "") || "0") || 0}
                      isPositive={isUp}
                      width={48}
                      height={20}
                    />
                  </div>

                  <div className="flex flex-col items-end">
                    <span className="font-mono font-bold text-xs text-slate-100 tabular-nums">
                      {item.price ?? "—"}
                    </span>
                    <span
                      className={`font-mono text-[10px] font-semibold tabular-nums ${
                        isUp ? "text-emerald-400" : "text-rose-400"
                      }`}
                    >
                      {item.change ?? "—"}
                    </span>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Footer Info & Shortcuts */}
      <div className="p-2.5 bg-[#0a0e17] text-[10px] font-mono text-slate-400 flex items-center justify-between shrink-0 border-t border-slate-800/60">
        <span>{filteredItems.length} Assets Loaded</span>
        <div className="flex items-center gap-2">
          <span>Toggle: <kbd className="px-1 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">[</kbd></span>
          <span>Close: <kbd className="px-1 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">Esc</kbd></span>
        </div>
      </div>
    </div>
  );
}
