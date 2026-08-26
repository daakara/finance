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
    router.push(`/?symbol=${encodeURIComponent(cleanSym)}`);
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (cleanQ) {
      handleSelectTicker(cleanQ);
    }
  };

  return (
    <>
      {/* Navbar Trigger Button */}
      <button
        onClick={() => setIsOpen(true)}
        aria-label="Search ticker, crypto, ETF or company"
        className="flex items-center space-x-2 bg-[#090d14] hover:bg-[#111722] border border-[#243044] hover:border-cyan-500/60 text-slate-400 hover:text-slate-200 px-2.5 sm:px-3 py-1.5 rounded-xl text-xs font-mono transition-all group"
      >
        <svg aria-hidden="true" className="w-3.5 h-3.5 text-slate-400 group-hover:text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
        <span className="hidden sm:inline">Search any asset...</span>
        <span className="sm:hidden">Search</span>
        <kbd className="hidden md:inline-block bg-[#162030] text-slate-400 text-[10px] px-1.5 py-0.5 rounded border border-[#2b394f]">
          /
        </kbd>
      </button>

      {/* Fullscreen Modal Backdrop */}
      {isOpen && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Universal Asset Search"
          onClick={() => setIsOpen(false)}
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-start justify-center pt-16 sm:pt-24 p-3 sm:p-4 font-mono animate-fadeIn"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="bg-[#0c1017] border border-[#243044] rounded-2xl max-w-xl w-full p-4 sm:p-5 shadow-2xl space-y-4 relative max-h-[85vh] flex flex-col"
          >
            {/* Search Input Bar */}
            <form onSubmit={handleFormSubmit} className="relative flex items-center">
              <span className="absolute left-3.5 text-cyan-400 text-sm">🔍</span>
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search any ticker (e.g. NVDA, LLY, TSLA, BTC-USD, COIN)..."
                className="w-full bg-[#070a11] border border-[#243044] focus:border-cyan-500 text-white pl-10 pr-20 py-2.5 rounded-xl text-sm outline-none placeholder:text-slate-500 font-mono tracking-wide"
              />
              {cleanQ && (
                <button
                  type="submit"
                  className="absolute right-2 px-2.5 py-1 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold rounded-lg transition-transform active:scale-95"
                >
                  Analyze ↵
                </button>
              )}
            </form>

            {/* Live Matches / Direct Submission */}
            <div className="overflow-y-auto space-y-1.5 flex-1 pr-1">
              {cleanQ && (
                <button
                  onClick={() => handleSelectTicker(cleanQ)}
                  className="w-full text-left p-3 rounded-xl bg-cyan-950/40 border border-cyan-700/60 hover:bg-cyan-900/50 flex items-center justify-between transition-colors group"
                >
                  <div>
                    <span className="text-xs font-bold text-cyan-300">Analyze Custom Unlisted Ticker:</span>
                    <div className="text-base font-black text-white tracking-wider">{cleanQ}</div>
                  </div>
                  <span className="text-xs text-cyan-400 font-bold group-hover:translate-x-1 transition-transform">
                    Launch Live Pipeline →
                  </span>
                </button>
              )}

              <div className="text-[10px] text-slate-500 font-bold uppercase tracking-wider pt-2 px-1">
                {cleanQ ? "Matching Universe Assets" : "Popular Universe Assets"}
              </div>

              {matchingPresets.map((item) => (
                <button
                  key={item.symbol}
                  onClick={() => handleSelectTicker(item.symbol)}
                  className="w-full text-left p-2.5 rounded-xl bg-[#090d14] hover:bg-[#111722] border border-[#1b2434] hover:border-cyan-500/50 flex items-center justify-between transition-colors group"
                >
                  <div className="flex items-center space-x-3">
                    <div className="bg-[#111722] px-2 py-1 rounded text-xs font-black text-white group-hover:text-cyan-400 border border-[#243044]">
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
            <div className="pt-3 border-t border-[#1b2434] flex items-center justify-between text-[11px] text-slate-500">
              <span>Press <kbd className="bg-[#162030] text-slate-300 px-1 py-0.5 rounded border border-[#2b394f]">ESC</kbd> to close</span>
              <span>Type any symbol and press <kbd className="bg-[#162030] text-slate-300 px-1 py-0.5 rounded border border-[#2b394f]">ENTER</kbd></span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}