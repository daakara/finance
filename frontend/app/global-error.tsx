"use client";

import { useEffect } from "react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[ARX Global Root Error Captured]:", error);
  }, [error]);

  return (
    <html lang="en">
      <body className="bg-[#070b12] text-slate-100 min-h-screen flex items-center justify-center p-4 font-mono">
        <div className="max-w-md w-full bg-[#0d131f] border border-rose-800 rounded-2xl p-6 shadow-2xl space-y-4">
          <div className="flex items-center space-x-3 text-rose-400">
            <span className="text-3xl">🛑</span>
            <div>
              <h1 className="text-base font-bold text-white">System Critical Error</h1>
              <p className="text-xs text-rose-400/80">Application Root Boundary Halted</p>
            </div>
          </div>
          <p className="text-xs text-slate-300 font-sans leading-relaxed">
            A fatal exception occurred in the application root. Execution was suspended to prevent state corruption.
          </p>
          <button
            onClick={() => reset()}
            type="button"
            className="w-full px-4 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold rounded-xl text-xs transition-colors cursor-pointer"
          >
            Re-initialize Application
          </button>
        </div>
      </body>
    </html>
  );
}
