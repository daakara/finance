"use client";

import { useEffect } from "react";
import Link from "next/link";

interface ErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function ErrorBoundary({ error, reset }: ErrorProps) {
  useEffect(() => {
    console.error("[ARX Error Boundary Captured]:", error);
  }, [error]);

  const handleClearCache = () => {
    try {
      if (typeof window !== "undefined") {
        const keys = Object.keys(localStorage);
        for (const k of keys) {
          if (k.startsWith("finance_market_db_")) {
            localStorage.removeItem(k);
          }
        }
        window.location.reload();
      }
    } catch {}
  };

  return (
    <div className="min-h-screen bg-[#070b12] text-slate-100 flex items-center justify-center p-4 font-mono">
      <div className="max-w-lg w-full bg-[#0d131f] border border-rose-900/60 rounded-2xl p-6 shadow-2xl space-y-5">
        <div className="flex items-center space-x-3 text-rose-400">
          <span className="text-2xl">⚠️</span>
          <div>
            <h1 className="text-lg font-bold text-white tracking-wide">
              Terminal Runtime Exception
            </h1>
            <p className="text-xs text-rose-400/80">
              Graceful Fault Recovery Active
            </p>
          </div>
        </div>

        <div className="p-3 bg-black/40 border border-slate-800 rounded-xl text-xs space-y-1">
          <span className="text-slate-400 block font-semibold">Error Diagnostic:</span>
          <p className="text-slate-300 break-words font-sans text-xs">
            {error?.message || "An unexpected client exception occurred during view rendering."}
          </p>
          {error?.digest && (
            <span className="text-[10px] text-slate-500 block">
              Digest: {error.digest}
            </span>
          )}
        </div>

        <p className="text-xs text-slate-400 leading-relaxed font-sans">
          The quantitative engine halted execution to protect capital state. You may re-initialize the terminal state machine, purge stale client cache, or navigate to the market overview.
        </p>

        <div className="flex flex-wrap gap-2.5 pt-2">
          <button
            onClick={() => reset()}
            type="button"
            className="flex-1 min-w-[120px] px-4 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold rounded-xl text-xs transition-colors shadow-lg cursor-pointer"
          >
            🔄 Retry View
          </button>
          <button
            onClick={handleClearCache}
            type="button"
            className="flex-1 min-w-[120px] px-4 py-2.5 bg-[#172235] hover:bg-[#202e47] text-slate-200 border border-slate-700 font-semibold rounded-xl text-xs transition-colors cursor-pointer"
          >
            🧹 Clear Cache & Reload
          </button>
          <Link
            href="/"
            className="w-full text-center px-4 py-2 text-slate-400 hover:text-white text-xs underline cursor-pointer"
          >
            ← Return to Live Overview
          </Link>
        </div>
      </div>
    </div>
  );
}
