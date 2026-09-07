"use client";

import React from "react";
import { AttentionSignal } from "../../types/change-intelligence";
import { trackTelemetryEvent } from "../../telemetry/tracker";

export interface AttentionFeedProps {
  signals: AttentionSignal[];
  onSelectTicker: (ticker: string) => void;
  className?: string;
}

export default function AttentionFeed({
  signals,
  onSelectTicker,
  className = "",
}: AttentionFeedProps) {
  const handleItemClick = (signal: AttentionSignal) => {
    trackTelemetryEvent(
      "DECISION",
      "attention_item_opened",
      {
        ticker: signal.ticker,
        severity: signal.severity,
        category: signal.category,
      },
      signal.ticker
    );
    onSelectTicker(signal.ticker);
  };

  return (
    <section
      data-testid="attention-feed"
      aria-labelledby="attention-feed-heading"
      className={`p-5 rounded-2xl bg-bg-surface border border-border-subtle shadow-xl space-y-4 font-sans ${className}`}
    >
      <div className="flex items-center justify-between pb-2 border-b border-border-subtle">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse" />
          <h3 id="attention-feed-heading" className="text-header-2 text-text-primary">
            Portfolio Attention Feed
          </h3>
          <span className="px-2 py-0.5 text-[10px] font-mono font-bold rounded-full bg-bg-surface-raised text-text-muted border border-border-subtle">
            {signals.length} Critical Events
          </span>
        </div>
        <span className="text-caption-mono text-text-muted text-xs hidden sm:inline">
          Material Changes Only (L3/L4)
        </span>
      </div>

      {signals.length === 0 ? (
        <div className="p-8 rounded-xl bg-bg-surface-raised border border-border-subtle text-center space-y-1">
          <span className="text-2xl">🛡️</span>
          <h4 className="text-body-ui font-bold text-text-primary">
            Zero Critical Thesis Changes
          </h4>
          <p className="text-caption-mono text-text-secondary text-xs">
            All tracked tickers remain within their acknowledged baseline tolerances.
          </p>
        </div>
      ) : (
        <div className="space-y-2.5">
          {signals.map((signal) => {
            const isCritical = signal.severity === "CRITICAL";

            return (
              <button
                key={signal.id}
                type="button"
                onClick={() => handleItemClick(signal)}
                className="w-full p-3.5 rounded-xl bg-bg-surface-raised hover:bg-bg-surface-elevated border border-border-subtle hover:border-accent-info/50 text-left transition-all cursor-pointer flex items-start justify-between gap-3 group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-info"
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-black text-sm text-text-primary group-hover:text-accent-info transition-colors">
                      {signal.ticker}
                    </span>
                    <span
                      className={`text-[10px] font-mono uppercase px-2 py-0.2 rounded border font-bold ${
                        isCritical
                          ? "bg-rose-950/60 text-rose-300 border-rose-800/80"
                          : "bg-amber-950/60 text-amber-300 border-amber-800/80"
                      }`}
                    >
                      {signal.category}
                    </span>
                  </div>

                  <p className="text-body-ui font-bold text-text-secondary group-hover:text-text-primary transition-colors text-xs sm:text-sm">
                    {signal.headline}
                  </p>

                  <p className="text-caption-mono text-text-muted text-[11px]">
                    {signal.rationale}
                  </p>
                </div>

                <span className="text-caption-mono text-accent-info font-bold text-xs shrink-0 self-center group-hover:translate-x-0.5 transition-transform">
                  Inspect →
                </span>
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}
