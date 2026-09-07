"use client";

import React from "react";
import { PredictionRecord } from "../../types/predictive-intelligence";

export interface WatchlistRiskRadarProps {
  predictions: PredictionRecord[];
  onSelectTicker?: (ticker: string) => void;
  className?: string;
}

export default function WatchlistRiskRadar({
  predictions,
  onSelectTicker,
  className = "",
}: WatchlistRiskRadarProps) {
  // Sort descending by probability (AC-PI-01, AC-PI-05)
  const sorted = [...predictions].sort((a, b) => b.probability - a.probability);

  return (
    <div
      role="region"
      aria-label="Watchlist Risk Radar"
      className={`rounded-lg border border-border-subtle bg-surface-subtle p-4 ${className}`}
    >
      <div className="flex items-center justify-between border-b border-border-subtle pb-3">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-accent-amber animate-pulse" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-text-primary">
            Watchlist Risk & Attention Radar
          </h3>
        </div>
        <span className="text-[11px] font-mono text-text-muted">
          {sorted.length} Assets Projected
        </span>
      </div>

      <div className="mt-3 space-y-2">
        {sorted.length === 0 ? (
          <p className="py-6 text-center text-xs text-text-muted">
            No upcoming high-probability attention shifts detected.
          </p>
        ) : (
          sorted.map((item) => {
            const probPct = Math.round(item.probability * 100);
            const isCritical = item.severity === "CRITICAL";

            return (
              <button
                key={item.predictionId}
                onClick={() => onSelectTicker?.(item.ticker)}
                className="w-full text-left rounded-md border border-border-subtle/40 bg-surface-card p-2.5 transition-all hover:border-border-strong hover:bg-surface-subtle/80 flex items-center justify-between group"
              >
                <div className="flex items-center gap-3">
                  <span className="font-mono font-bold text-sm text-text-primary group-hover:text-accent-blue transition-colors">
                    {item.ticker}
                  </span>
                  <span
                    className={`text-[10px] uppercase font-mono px-1.5 py-0.5 rounded ${
                      isCritical
                        ? "bg-accent-rose/10 text-accent-rose border border-accent-rose/20"
                        : "bg-accent-amber/10 text-accent-amber border border-accent-amber/20"
                    }`}
                  >
                    {item.predictionType.replace(/_/g, " ")}
                  </span>
                </div>

                <div className="flex items-center gap-3">
                  <div className="w-24 bg-surface-base h-2 rounded-full overflow-hidden border border-border-subtle/50">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        probPct >= 80 ? "bg-accent-rose" : "bg-accent-amber"
                      }`}
                      style={{ width: `${probPct}%` }}
                    />
                  </div>
                  <span className="font-mono font-bold text-xs text-text-primary w-9 text-right">
                    {probPct}%
                  </span>
                </div>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}
