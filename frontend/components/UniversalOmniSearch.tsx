"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { SHARED_WATCHLIST_ITEMS } from "../lib/constants";
import { prefetchAssetAnalytics, SpotPriceRegistry } from "../lib/api";
import { resolveAssetAlias } from "../lib/assetRegistry";
import { getPersistedMarketSnapshot } from "../lib/marketDatabase";

export default function UniversalOmniSearch() {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Global hotkey: Press "/" or "Cmd+K" / "Ctrl+K" anywhere to open omni-search
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        (e.key === "/" || ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k")) &&
        document.activeElement?.tagName !== "INPUT" &&
        document.activeElement?.tagName !== "TEXTAREA"
      ) {
        e.preventDefault();
        setIsOpen(true);
      } else if (e.key === "Escape" && isOpen) {
        setIsOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen]);

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 50);
    } else {
      setQuery("");
    }
  }, [isOpen]);

  const cleanQ = query.trim().toUpperCase();
  const aliasMatch = resolveAssetAlias(cleanQ);

  // Search matching presets + allow arbitrary ticker execution
  const matchingPresets = SHARED_WATCHLIST_ITEMS.filter((item) => {
    if (!cleanQ) return true;
    return (
      item.symbol.toUpperCase().includes(cleanQ) ||
      item.name.toUpperCase().includes(cleanQ) ||
      item.type.toUpperCase().includes(cleanQ) ||
      (aliasMatch && item.symbol.toUpperCase() === aliasMatch.canonicalTicker.toUpperCase())
    );
  }).slice(0, 8);

  const handleSelectTicker = (sym: string) => {
    const cleanSym = sym.trim().toUpperCase();
    if (!cleanSym) return;
    setIsOpen(false);
    if (typeof window !== "undefined") {
      window.location.href = `/?symbol=${encodeURIComponent(cleanSym)}`;
    } else {
      router.push(`/?symbol=${encodeURIComponent(cleanSym)}`);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (cleanQ) {
      if (aliasMatch) {
        handleSelectTicker(aliasMatch.canonicalTicker);
      } else {
        handleSelectTicker(cleanQ);
      }
    }
  };

  return (
    <>
      {/* 🖥️ Desktop / Tablet: Full Wide Search Trigger */}
      <button
        onClick={() => setIsOpen(true)}
        type="button"
        aria-label="Search ticker, crypto, ETF or company across global markets (Press Slash or Command-K)"
        className="hidden md:flex w-full items-center justify-between bg-[#090d14] hover:bg-[#131b29] border border-[#2b3a52] hover:border-cyan-400 text-slate-200 hover:text-white px-3.5 py-1.5 sm:py-2 rounded-xl text-xs font-mono transition-all shadow-inner group cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none"
      >
        <div className="flex items-center space-x-2.5 min-w-0">
          <svg aria-hidden="true" className="w-4 h-4 text-cyan-400 shrink-0 group-hover:scale-110 transition-transform" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <span className="truncate text-slate-300 group-hover:text-white font-medium">
            Search any ticker, stock, ETF or crypto...
          </span>
        </div>
        <div className="flex items-center space-x-1 shrink-0 ml-2">
          <kbd className="inline-block bg-[#162030] text-cyan-300 font-bold text-[10px] px-1.5 py-0.5 rounded border border-[#2b394f]">
            ⌘K
          </kbd>
          <kbd className="inline-block bg-[#162030] text-slate-300 font-bold text-[10px] px-1.5 py-0.5 rounded border border-[#2b394f]">
            /
          </kbd>
        </div>
      </button>

      {/* 📱 Mobile: High-Visibility Accessible Search Button (Min 44x44px Touch Target) */}
      <button
        onClick={() => setIsOpen(true)}
        type="button"
        aria-label="Search any asset or ticker"
        className="md:hidden flex items-center justify-center min-w-[36px] min-h-[36px] p-2 rounded-lg bg-[#111722] hover:bg-[#1b2537] border border-cyan-400/70 text-cyan-300 hover:text-white active:scale-95 shadow-md shrink-0 cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none"
      >
        <svg aria-hidden="true" className="w-4 h-4 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
      </button>

      {/* Fullscreen Search Modal Backdrop */}
      {isOpen && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Universal Asset Search"
          onClick={() => setIsOpen(false)}
          className="fixed inset-0 z-[100] bg-black/85 backdrop-blur-md flex items-start justify-center pt-8 sm:pt-24 p-3 sm:p-4 font-mono animate-fadeIn"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="bg-[#0c1017] border border-[#2b3a52] rounded-2xl max-w-xl w-full p-4 sm:p-6 shadow-2xl space-y-4 relative max-h-[90vh] flex flex-col"
          >
            {/* Search Input Bar */}
            <div className="flex items-center justify-between pb-1">
              <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-1.5">
                <span>🔍</span>
                <span>Global Asset Terminal Search</span>
              </span>
              <button
                onClick={() => setIsOpen(false)}
                aria-label="Close search dialog"
                className="text-slate-300 hover:text-white text-xs px-2.5 py-1 bg-[#162030] hover:bg-[#223147] rounded-lg border border-[#2b394f] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none transition-colors"
              >
                ESC ✕
              </button>
            </div>

            <form onSubmit={handleFormSubmit} className="relative flex items-center">
              <span className="absolute left-3.5 text-cyan-400 text-base" aria-hidden="true">🔍</span>
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Type ANY ticker (e.g. NVDA, LLY, TSLA, BTC-USD, COIN, PLTR)..."
                aria-label="Type any ticker symbol or asset name"
                className="w-full bg-[#070a11] border-2 border-cyan-500 focus:border-cyan-400 text-white pl-10 pr-20 py-2.5 sm:py-3 rounded-xl text-sm outline-none placeholder:text-slate-400 font-mono tracking-wide shadow-lg focus-visible:ring-2 focus-visible:ring-cyan-400"
              />
              {cleanQ && (
                <button
                  type="submit"
                  aria-label={`Submit search for ${cleanQ}`}
                  className="absolute right-2 px-2.5 sm:px-3 py-1 sm:py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold rounded-lg transition-transform active:scale-95 shadow cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none"
                >
                  Go ↵
                </button>
              )}
            </form>

            {/* Live Matches / Direct Submission */}
            <div className="overflow-y-auto space-y-2 flex-1 pr-1" role="listbox">
              {/* 💡 Intelligent Did You Mean Recommendation Card */}
              {aliasMatch && (
                <button
                  type="button"
                  onClick={() => handleSelectTicker(aliasMatch.canonicalTicker)}
                  className="w-full text-left p-3 rounded-xl bg-cyan-950/60 border-2 border-cyan-400 hover:bg-cyan-900/60 flex items-center justify-between transition-all group cursor-pointer shadow-lg animate-fadeIn focus-visible:ring-2 focus-visible:ring-cyan-300"
                >
                  <div className="flex items-center space-x-3">
                    <div className="bg-cyan-400 text-slate-950 px-2.5 py-1 rounded text-xs font-black tracking-wider shadow-sm">
                      {aliasMatch.canonicalTicker}
                    </div>
                    <div>
                      <div className="text-xs font-bold text-cyan-200 flex items-center gap-1.5">
                        <span>💡 Did you mean:</span>
                        <span className="text-white underline decoration-cyan-400">{aliasMatch.companyName}</span>
                      </div>
                      <div className="text-[10px] text-cyan-300/80 font-mono mt-0.5">
                        {aliasMatch.description || `Official Exchange Ticker: ${aliasMatch.canonicalTicker}`}
                      </div>
                    </div>
                  </div>
                  <span className="text-xs font-bold text-cyan-300 group-hover:translate-x-1 transition-transform shrink-0 ml-2">
                    Select {aliasMatch.canonicalTicker} ↵
                  </span>
                </button>
              )}

              {cleanQ && !aliasMatch && (
                <button
                  type="button"
                  onClick={() => handleSelectTicker(cleanQ)}
                  className="w-full text-left p-3 rounded-xl bg-[#0b1019] border border-cyan-500 hover:border-cyan-400 flex items-center justify-between transition-colors group cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none"
                >
                  <div>
                    <span className="text-xs font-bold text-cyan-300">Run Live Analysis for Custom Asset:</span>
                    <div className="text-base sm:text-lg font-black text-white tracking-wider flex items-center gap-2">
                      <span>{cleanQ}</span>
                      <span className="text-[10px] bg-cyan-900 text-cyan-200 px-2 py-0.5 rounded font-mono font-semibold">Live Pipeline</span>
                    </div>
                  </div>
                  <span className="text-xs text-cyan-400 font-bold group-hover:translate-x-1 transition-transform">
                    Launch →
                  </span>
                </button>
              )}

              <div className="text-[10px] text-slate-300 font-bold uppercase tracking-wider pt-2 px-1 flex items-center justify-between">
                <span>{cleanQ ? "Matching Universe Assets" : "Popular Universe Assets"}</span>
                <span className="text-cyan-400">Instant Load</span>
              </div>

              {matchingPresets.map((item) => {
                const reg = SpotPriceRegistry.get(item.symbol);
                const snap = getPersistedMarketSnapshot(item.symbol);
                const price = reg?.price || snap?.currentPrice;
                const change = reg?.changePct ?? snap?.priceChangePct24h;
                const isUp = (change ?? 0) >= 0;

                return (
                  <button
                    key={item.symbol}
                    type="button"
                    onClick={() => handleSelectTicker(item.symbol)}
                    onMouseEnter={() => prefetchAssetAnalytics(item.symbol)}
                    className="w-full text-left p-2.5 rounded-xl bg-[#090d14] hover:bg-[#162030] border border-[#1e2a3c] hover:border-cyan-400 flex items-center justify-between transition-colors group cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="bg-[#111722] px-2.5 py-1 rounded text-xs font-black text-white group-hover:text-cyan-300 border border-[#243044]">
                        {item.symbol}
                      </div>
                      <div>
                        <div className="text-xs font-semibold text-slate-100">{item.name}</div>
                        <div className="text-[10px] text-slate-400">{item.type}</div>
                      </div>
                    </div>
                    {price ? (
                      <div className="text-right">
                        <div className="text-xs font-bold text-slate-200 tabular-nums">${price.toFixed(2)}</div>
                        {change !== undefined && (
                          <div className={`text-[10px] font-semibold tabular-nums ${isUp ? "text-emerald-400" : "text-rose-400"}`}>
                            {isUp ? "+" : ""}{change.toFixed(2)}%
                          </div>
                        )}
                      </div>
                    ) : null}
                  </button>
                );
              })}
            </div>

            {/* Keyboard Footer */}
            <div className="pt-2 sm:pt-3 border-t border-[#1e2a3c] flex items-center justify-between text-[11px] text-slate-400">
              <span>Press <kbd className="bg-[#162030] text-slate-200 px-1.5 py-0.5 rounded border border-[#2b394f]">ESC</kbd> to close</span>
              <span>Tap any symbol to analyze</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}