"use client";

import { useState, useRef, useEffect } from "react";
import { SHARED_WATCHLIST_ITEMS, WatchlistDefinition } from "../lib/constants";
import { prefetchAssetAnalytics, fetchBatchQuotes } from "../lib/api";
import { getAllPersistedMarketSnapshots } from "../lib/marketDatabase";
import MiniSparkline from "./MiniSparkline";

interface WatchlistItemDisplay extends WatchlistDefinition {
  price?: string;
  change?: string;
  isUp?: boolean;
}

interface WatchlistSidebarProps {
  activeSymbol: string;
  onSelectSymbol: (symbol: string) => void;
  liveCurrentPrice?: number;
  livePriceChangePct?: number;
}

export default function WatchlistSidebar({ activeSymbol, onSelectSymbol, liveCurrentPrice, livePriceChangePct }: WatchlistSidebarProps) {
  // Initialize with persisted fresh database snapshots if available
  const [items, setItems] = useState<WatchlistItemDisplay[]>(() => {
    if (typeof window === "undefined") return SHARED_WATCHLIST_ITEMS;
    const snapshots = getAllPersistedMarketSnapshots(true);
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
      return item;
    });
  });

  // Batch-fetch real-time live quotes on mount to eliminate static price baselines
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

  // Synchronize incoming live current price and percent change with the active watchlist item
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
  const [isMobileExpanded, setIsMobileExpanded] = useState(false);
  const [activeCategory, setActiveCategory] = useState<"All" | "Pinned" | "Stock" | "ETF" | "Crypto">("All");
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Load pinned symbols from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem("FINANCE_PINNED_SYMBOLS");
      if (saved) {
        setPinnedSymbols(JSON.parse(saved));
      } else {
        // Default pinned items
        const defaultPins = ["AAPL", "NVDA", "BTC-USD"];
        setPinnedSymbols(defaultPins);
        localStorage.setItem("FINANCE_PINNED_SYMBOLS", JSON.stringify(defaultPins));
      }
    } catch {
      setPinnedSymbols(["AAPL", "NVDA"]);
    }
  }, []);

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

  // Filter items by category AND search query (ticker or company name)
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

  const handleCategoryClick = (cat: "All" | "Pinned" | "Stock" | "ETF" | "Crypto") => {
    setActiveCategory(cat);
    setIsMobileExpanded(true);

    if (cat !== "All" && cat !== "Pinned") {
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
      const alreadyInList = items.some((i) => i.symbol.toUpperCase() === cleanQuery);
      if (!alreadyInList) {
        const newItem: WatchlistItemDisplay = {
          symbol: cleanQuery,
          name: `${cleanQuery} Custom Asset`,
          type: "Stock",
        };
        const updatedItems = [newItem, ...items];
        setItems(updatedItems);
        const updatedPins = [cleanQuery, ...pinnedSymbols];
        setPinnedSymbols(updatedPins);
        localStorage.setItem("FINANCE_PINNED_SYMBOLS", JSON.stringify(updatedPins));
      }
      setIsMobileExpanded(false);
      setSearchQuery("");
    }
  };

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-3 sm:p-4 shadow-xl space-y-3 h-full font-mono">
      {/* Header with Mobile Accordion Toggle */}
      <div className="flex items-center justify-between border-b border-[#1b2434] pb-3 gap-2">
        <div className="flex items-center space-x-2 min-w-0">
          <h2 className="text-xs font-bold text-slate-200 uppercase tracking-wider truncate">
            Watchlist & Signals
          </h2>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#1e293b] text-cyan-400 font-semibold shrink-0">
            LIVE
          </span>
        </div>

        {/* Mobile-Only Accordion Trigger */}
        <button
          onClick={() => setIsMobileExpanded(!isMobileExpanded)}
          aria-expanded={isMobileExpanded}
          aria-controls="watchlist-items-container"
          className="sm:hidden flex items-center space-x-1 text-xs text-cyan-400 font-bold px-2.5 py-1 bg-[#162030] rounded-lg border border-[#243044] active:scale-[0.96] transition-transform shrink-0"
        >
          <span>{isMobileExpanded ? "Hide" : `Show (${filteredItems.length})`}</span>
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

      {/* Asset Category Filters with Pinned Filter (Horizontal Scroll on Mobile) */}
      <div role="tablist" aria-label="Asset Class Filter" className="flex items-center gap-1 overflow-x-auto no-scrollbar p-1 bg-[#090d14] rounded-lg border border-[#1b2434] text-[10px]">
        {(["All", "Pinned", "Stock", "ETF", "Crypto"] as const).map((cat) => (
          <button
            key={cat}
            role="tab"
            aria-selected={activeCategory === cat}
            onClick={() => handleCategoryClick(cat)}
            className={`flex-1 min-w-[54px] py-1 px-1.5 rounded font-bold transition-all active:scale-[0.96] flex items-center justify-center gap-0.5 shrink-0 sm:shrink ${
              activeCategory === cat
                ? "bg-cyan-500 text-slate-950 shadow-sm font-extrabold"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            {cat === "Pinned" && <span>⭐</span>}
            <span>{cat}</span>
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
                  setIsMobileExpanded(false);
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
            const isPinned = pinnedSymbols.includes(item.symbol.toUpperCase());
            return (
              <div
                key={item.symbol}
                onClick={() => {
                  onSelectSymbol(item.symbol);
                  setIsMobileExpanded(false);
                }}
                onMouseEnter={() => prefetchAssetAnalytics(item.symbol)}
                className={`w-full flex items-center justify-between p-2 rounded-lg border text-left cursor-pointer transition-all active:scale-[0.98] ${
                  isSelected
                    ? "bg-[#162030] border-cyan-500 shadow-md shadow-cyan-950/40"
                    : "bg-[#0b1019] border-[#1b2434] hover:bg-[#131b28] hover:border-[#2b3a52]"
                }`}
              >
                <div className="flex items-center space-x-2 min-w-0">
                  <button
                    type="button"
                    onClick={(e) => togglePin(e, item.symbol)}
                    aria-label={isPinned ? `Unpin ${item.symbol}` : `Pin ${item.symbol}`}
                    className="text-xs hover:scale-125 transition-transform shrink-0 p-0.5"
                  >
                    <span className={isPinned ? "text-amber-400" : "text-slate-600 hover:text-slate-400"}>
                      ★
                    </span>
                  </button>
                  <div className="min-w-0">
                    <div className="flex items-center space-x-1.5">
                      <span className="font-bold text-xs text-white">{item.symbol}</span>
                      <span className="text-[9px] px-1 py-0.2 rounded bg-[#1e293b] text-slate-400">
                        {item.type}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-400 truncate max-w-[90px] sm:max-w-[100px]">
                      {item.name}
                    </div>
                  </div>
                </div>

                {/* Inline Mini Sparkline */}
                <div className="hidden sm:block shrink-0 px-1">
                  {item.price ? (
                    <MiniSparkline
                      basePrice={parseFloat(item.price.replace(/[^0-9.]/g, "")) || 100}
                      changePct={parseFloat(item.change?.replace(/[^0-9.-]/g, "") || "0") || 0}
                      isPositive={item.isUp ?? true}
                      width={42}
                      height={18}
                    />
                  ) : (
                    <div className="w-[42px] h-[18px] bg-[#162030] rounded animate-pulse" />
                  )}
                </div>

                <div className="text-right shrink-0">
                  <div className="text-xs font-bold text-slate-200 tabular-nums">
                    {item.price || <span className="text-slate-500 font-mono text-[11px] animate-pulse">--.--</span>}
                  </div>
                  {item.change ? (
                    <div
                      title="24-Hour Daily Return relative to previous close"
                      aria-label={`24-hour change: ${item.change}`}
                      className={`text-[10px] font-semibold tabular-nums flex items-center justify-end gap-0.5 ${
                        item.isUp ? "text-emerald-400" : "text-rose-400"
                      }`}
                    >
                      <span>{item.change}</span>
                      <span className="text-[8px] opacity-70 font-normal">24H</span>
                    </div>
                  ) : null}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}