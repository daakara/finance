"use client";

import React from "react";

export function PersistentBanner({ provenanceLabel }: { provenanceLabel?: string }) {
  return (
    <div
      role="banner"
      aria-label="Simulation Environment Disclosure"
      className="sticky top-0 z-50 bg-amber-950/80 border-b border-amber-500/40 text-amber-300 text-xs py-1.5 px-4 font-mono font-semibold tracking-wider text-center flex flex-wrap items-center justify-center gap-2 backdrop-blur-sm shadow-sm max-w-full overflow-hidden"
    >
      <span className="flex h-2 w-2 rounded-full bg-amber-400 animate-pulse shrink-0" aria-hidden="true" />
      <span className="text-center break-words max-w-full">⚠️ SIMULATED RESEARCH SANDBOX — OFFLINE / ZERO BROKER CONNECTIVITY — FINANCIAL DATA IS STATIC</span>
      {provenanceLabel && (
        <span className="bg-amber-900/60 border border-amber-600/40 px-2 py-0.5 rounded text-[10px] text-amber-200">
          PROVENANCE: {provenanceLabel}
        </span>
      )}
    </div>
  );
}
