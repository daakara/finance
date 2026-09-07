"use client";

import React from "react";
import { useUIStore } from "../../state/ui-store";

export interface WatchlistDrawerTriggerProps {
  className?: string;
  variant?: "floating" | "navbar" | "inline";
}

export default function WatchlistDrawerTrigger({
  className = "",
  variant = "navbar",
}: WatchlistDrawerTriggerProps) {
  const watchlistOpen = useUIStore((state) => state.watchlistOpen);
  const toggleWatchlist = useUIStore((state) => state.toggleWatchlist);

  if (variant === "floating") {
    return (
      <button
        type="button"
        onClick={() => toggleWatchlist("mouse")}
        aria-expanded={watchlistOpen}
        aria-controls="watchlist-drawer"
        aria-label={watchlistOpen ? "Close Watchlist Drawer ([)" : "Open Watchlist Drawer ([)"}
        data-testid="watchlist-drawer-trigger"
        className={`fixed left-3 bottom-5 z-40 flex items-center gap-2 px-3 py-2 rounded-lg border bg-[#0f1422]/95 backdrop-blur text-slate-200 border-slate-700/80 shadow-xl hover:bg-[#161f33] hover:text-slate-100 hover:border-slate-600 transition-all ${className}`}
      >
        <svg
          className="w-4 h-4 text-emerald-400"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
        </svg>
        <span className="text-xs font-semibold">Watchlist</span>
        <kbd className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
          [
        </kbd>
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={() => toggleWatchlist("mouse")}
      aria-expanded={watchlistOpen}
      aria-controls="watchlist-drawer"
      aria-label={watchlistOpen ? "Close Watchlist Drawer ([)" : "Open Watchlist Drawer ([)"}
      data-testid="watchlist-drawer-trigger"
      className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border text-xs font-medium transition-colors ${
        watchlistOpen
          ? "bg-slate-800 text-slate-100 border-slate-600"
          : "bg-transparent text-slate-300 border-slate-800 hover:bg-slate-800/80 hover:text-slate-100 hover:border-slate-700"
      } ${className}`}
    >
      <svg
        className="w-3.5 h-3.5 text-emerald-400"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
      </svg>
      <span>Watchlist</span>
      <kbd className="hidden sm:inline text-[9px] font-mono px-1 py-0.2 rounded bg-slate-800 text-slate-400 border border-slate-700">
        [
      </kbd>
    </button>
  );
}
