"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { SHARED_WATCHLIST_ITEMS } from "../lib/constants";

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

  // Search matching presets + allow arbitrary ticker execution
  const matchingPresets = SHARED_WATCHLIST_ITEMS.filter((item) => {
    if (!cleanQ) return true;
    return (
      item.symbol.toUpperCase().includes(cleanQ) ||
      item.name.toUpperCase().includes(cleanQ) ||
      item.type.toUpperCase().includes(cleanQ)
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
      handleSelectTicker(cleanQ);
    }
  };

  return (
    <>
      {/* 🖥️ Desktop / Tablet: Full Wide Search Trigger */}
      <button
        onClick={() => setIsOpen(true)}
        type="button"
        aria-label="Search ticker, crypto, ETF or company across global markets"
        className="hidden md:flex w-full items-center justify-between bg-[#070a11] hover:bg-[#111722] border border-[#243044] hover:border-cyan-500/80 text-slate-300 hover:text-white px-3.5 py-1.5 sm:py-2 rounded-xl text-xs font-mono transition-all shadow-inner group cursor-pointer"
      >
        <div className="flex items-center space-x-2.5 min-w-0">
          <svg aria-hidden="true" className="w-4 h-4 text-cyan-400 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <span className="truncate text-slate-400 group-hover:text-slate-200">
            Search any ticker, stock, ETF or crypto...
          </span>
        </div>
        <div className="flex items-center space-x-1 shrink-0 ml-2">
          <kbd className="inline-block bg-[#162030] text-cyan-400 font-bold text-[10px] px-1.5 py-0.5 rounded border border-[#2b394f]">
            ⌘K
          </kbd>
          <kbd className="inline-block bg-[#162030] text-slate-400 text-[10px] px-1.5 py-0.5 rounded border border-[#2b394f]">
            /
          </kbd>
        </div>
      </button>

      {/* 📱 Mobile: Dedicated High-Visibility Search Icon Button in Header */}
      <button
        onClick={() => setIsOpen(true)}
        type="button"
        aria-label="Search any asset"
        className="md:hidden flex items-center justify-center w-8 h-8 rounded-lg bg-[#111722] border border-cyan-500/50 text-cyan-400 hover:text-white active:scale-95 shadow-md shrink-0 cursor-pointer"
      >
        <svg aria-hidden="true" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
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
            className="bg-[#0c1017] border border-[#243044] rounded-2xl max-w-xl w-full p-4 sm:p-6 shadow-2xl space-y-4 relative max-h-[90vh] flex flex-col"
          >
            {/* Search Input Bar */}
            <div className="flex items-center justify-between pb-1">
              <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider">
                🔍 Global Asset Terminal Search
              </span>
              <button
                onClick={() => setIsOpen(false)}
                className="text-slate-400 hover:text-white text-xs px-2 py-1 bg-[#162030] rounded border border-[#243044]"
              >
                ESC ✕
              </button>
            </div>

            <form onSubmit={handleFormSubmit} className="relative flex items-center">
              <span className="absolute left-3.5 text-cyan-400 text-base">🔍</span>
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Type ANY ticker (e.g. NVDA, LLY, TSLA, BTC-USD, COIN, PLTR)..."
                className="w-full bg-[#070a11] border-2 border-cyan-500/70 focus:border-cyan-400 text-white pl-10 pr-20 py-2.5 sm:py-3 rounded-xl text-sm outline-none placeholder:text-slate-500 font-mono tracking-wide shadow-lg"
              />
              {cleanQ && (
                <button
                  type="submit"
                  className="absolute right-2 px-2.5 sm:px-3 py-1 sm:py-1.5 bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white text-xs font-bold rounded-lg transition-transform active:scale-95 shadow cursor-pointer"
                >
                  Go ↵
                </button>
              )}
            </form>

            {/* Live Matches / Direct Submission */}
            <div className="overflow-y-auto space-y-1.5 flex-1 pr-1">
              {cleanQ && (
                <button
                  type="button"
                  onClick={() => handleSelectTicker(cleanQ)}
                  className="w-full text-left p-3 rounded-xl bg-gradient-to-r from-cyan-950/60 to-indigo-950/60 border border-cyan-500/60 hover:border-cyan-400 flex items-center justify-between transition-colors group cursor-pointer"
                >
                  <div>
                    <span className="text-xs font-bold text-cyan-300">Run Live Analysis for Custom Asset:</span>
                    <div className="text-base sm:text-lg font-black text-white tracking-wider flex items-center gap-2">
                      <span>{cleanQ}</span>
                      <span className="text-[10px] bg-cyan-900/80 text-cyan-200 px-2 py-0.5 rounded font-mono font-semibold">Live Pipeline</span>
                    </div>
                  </div>
                  <span className="text-xs text-cyan-400 font-bold group-hover:translate-x-1 transition-transform">
                    Launch →
                  </span>
                </button>
              )}

              <div className="text-[10px] text-slate-400 font-bold uppercase tracking-wider pt-2 px-1 flex items-center justify-between">
                <span>{cleanQ ? "Matching Universe Assets" : "Popular Universe Assets"}</span>
                <span>Instant Load</span>
              </div>

              {matchingPresets.map((item) => (
                <button
                  key={item.symbol}
                  type="button"
                  onClick={() => handleSelectTicker(item.symbol)}
                  className="w-full text-left p-2.5 rounded-xl bg-[#090d14] hover:bg-[#162030] border border-[#1b2434] hover:border-cyan-500/60 flex items-center justify-between transition-colors group cursor-pointer"
                >
                  <div className="flex items-center space-x-3">
                    <div className="bg-[#111722] px-2.5 py-1 rounded text-xs font-black text-white group-hover:text-cyan-400 border border-[#243044]">
                      {item.symbol}
                    </div>
                    <div>
                      <div className="text-xs font-semibold text-slate-200">{item.name}</div>
                      <div className="text-[10px] text-slate-500">{item.type}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs font-bold text-slate-300">{item.price}</div>
                    <div className={`text-[10px] font-semibold ${item.isUp ? "text-emerald-400" : "text-rose-400"}`}>
                      {item.change}
                    </div>
                  </div>
                </button>
              ))}
            </div>

            {/* Keyboard Footer */}
            <div className="pt-2 sm:pt-3 border-t border-[#1b2434] flex items-center justify-between text-[11px] text-slate-500">
              <span>Press <kbd className="bg-[#162030] text-slate-300 px-1.5 py-0.5 rounded border border-[#2b394f]">ESC</kbd> to close</span>
              <span>Tap any symbol to analyze</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}