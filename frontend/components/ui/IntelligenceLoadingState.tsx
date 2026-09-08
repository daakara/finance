"use client";

import React from "react";

export interface IntelligenceLoadingStateProps {
  message?: string;
  cardCount?: number;
}

export default function IntelligenceLoadingState({
  message = "Synthesizing executive telemetry across 11 intelligence centers...",
  cardCount = 4,
}: IntelligenceLoadingStateProps) {
  return (
    <div className="space-y-6 animate-pulse" aria-busy="true" aria-live="polite">
      <div className="rounded-xl border border-[#24324A] bg-[#121B2A] p-6 text-center space-y-3">
        <div className="inline-block w-8 h-8 rounded-full border-2 border-cyan-400 border-t-transparent animate-spin" />
        <p className="text-sm font-mono text-cyan-300">{message}</p>
        <div className="h-1.5 max-w-md mx-auto rounded-full bg-[#182336] overflow-hidden">
          <div className="h-full bg-cyan-400 animate-pulse w-2/3" />
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {Array.from({ length: cardCount }).map((_, i) => (
          <div key={i} className="h-28 rounded-xl border border-[#24324A] bg-[#121B2A] p-4 space-y-3">
            <div className="h-3 w-1/2 bg-[#182336] rounded" />
            <div className="h-6 w-3/4 bg-[#182336] rounded" />
            <div className="h-2 w-1/3 bg-[#182336] rounded" />
          </div>
        ))}
      </div>
    </div>
  );
}
