"use client";

import React, { useState, useEffect } from "react";
import { PortfolioAttentionEntry, PortfolioAttentionFeed as FeedType } from "../../types/portfolio-intelligence";
import { trackTelemetryEvent } from "../../telemetry/tracker";

export interface PortfolioAttentionFeedProps {
  feed: FeedType;
  onSelectTicker: (ticker: string, autoExpandDelta?: boolean) => void;
  className?: string;
}

export default function PortfolioAttentionFeed({
  feed,
  onSelectTicker,
  className = "",
}: PortfolioAttentionFeedProps) {
  const [expandedSummary, setExpandedSummary] = useState(false);

  useEffect(() => {
    trackTelemetryEvent(
      "DECISION",
      "portfolio_feed_loaded",
      {
        criticalCount: feed.criticalItems.length,
        materialCount: feed.materialItems.length,
        summaryCount: feed.summaryCount,
        totalAttention: feed.totalAttentionCount,
      }
    );
  }, [feed.criticalItems.length, feed.materialItems.length, feed.summaryCount, feed.totalAttentionCount]);

  const handleItemClick = (entry: PortfolioAttentionEntry) => {
    trackTelemetryEvent(
      "DECISION",
      "portfolio_item_opened",
      {
        ticker: entry.ticker,
        severity: entry.severity,
        category: entry.category,
      },
      entry.ticker
    );
    // Deep link directly to ticker workstation with delta banner expanded
    onSelectTicker(entry.ticker, true);
  };

  const hasItems = feed.criticalItems.length > 0 || feed.materialItems.length > 0;

  return (
    <section
      data-testid="portfolio-attention-feed"
      aria-label="Portfolio Attention Feed"
      className={`space-y-6 font-sans ${className}`}
    >
      {/* Header Bar */}
      <div className="flex items-center justify-between pb-3 border-b border-border-subtle">
        <div className="flex items-center gap-2.5">
          <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse" />
          <h3 className="text-header-2 text-text-primary">
            Things Requiring Attention Today
          </h3>
          <span className="px-2 py-0.5 text-xs font-mono font-bold rounded-full bg-bg-surface-raised text-text-secondary border border-border-subtle">
            {feed.totalAttentionCount} Actionable Assets
          </span>
        </div>

        <span className="text-caption-mono text-text-muted text-xs hidden sm:inline">
          Materiality Gate Active · Sub-Threshold Noise Suppressed
        </span>
      </div>

      {!hasItems ? (
        <div className="p-8 rounded-2xl bg-bg-surface border border-border-subtle text-center space-y-2">
          <span className="text-3xl">🛡️</span>
          <h4 className="text-body-ui font-bold text-text-primary">
            Zero Critical Portfolio Mutations
          </h4>
          <p className="text-caption-mono text-text-secondary text-xs max-w-md mx-auto">
            All tracked portfolio positions remain within their acknowledged thesis thresholds. Delta Trust Index: 100%.
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* SECTION 1: CRITICAL ACTIONS (L4) */}
          {feed.criticalItems.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-caption-mono text-rose-400 font-bold uppercase tracking-wider text-xs flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-rose-500" />
                  Critical Attention (Level 4)
                </span>
                <span className="text-caption-mono text-text-muted text-xs">
                  Execution State & Macro Rotations
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {feed.criticalItems.map((entry) => (
                  <div
                    key={entry.ticker}
                    onClick={() => handleItemClick(entry)}
                    className="p-4 rounded-xl bg-bg-surface hover:bg-bg-surface-elevated border border-rose-500/40 hover:border-rose-500 transition-all shadow-md cursor-pointer flex flex-col justify-between space-y-3 group"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-header-2 font-black text-text-primary font-mono group-hover:text-accent-info transition-colors">
                            ${entry.ticker}
                          </span>
                          <span className="px-2 py-0.5 text-[10px] font-mono font-bold rounded bg-rose-950 text-rose-300 border border-rose-800 uppercase">
                            {entry.category}
                          </span>
                        </div>
                        <p className="text-body-ui text-text-secondary text-xs mt-1 leading-snug">
                          {entry.headline}
                        </p>
                      </div>

                      <span className="px-2.5 py-1 text-xs font-mono font-bold rounded bg-rose-500 hover:bg-rose-400 text-slate-950 shrink-0 transition-colors">
                        Review →
                      </span>
                    </div>

                    <div className="flex items-center justify-between pt-2 border-t border-border-subtle/60 text-[11px] font-mono text-text-muted">
                      <span>{entry.itemCount} material change{entry.itemCount > 1 ? "s" : ""}</span>
                      <span className="text-accent-info group-hover:underline">Open Workspace & Delta Banner</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* SECTION 2: MATERIAL CONVICTION SHIFTS (L3) */}
          {feed.materialItems.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-caption-mono text-amber-400 font-bold uppercase tracking-wider text-xs flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-amber-500" />
                  Material Conviction Shifts (Level 3)
                </span>
                <span className="text-caption-mono text-text-muted text-xs">
                  Setup Score & Flow Surges
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {feed.materialItems.map((entry) => (
                  <div
                    key={entry.ticker}
                    onClick={() => handleItemClick(entry)}
                    className="p-4 rounded-xl bg-bg-surface hover:bg-bg-surface-elevated border border-amber-500/30 hover:border-amber-500/60 transition-all shadow cursor-pointer flex flex-col justify-between space-y-3 group"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-header-2 font-black text-text-primary font-mono group-hover:text-accent-info transition-colors">
                            ${entry.ticker}
                          </span>
                          <span className="px-2 py-0.5 text-[10px] font-mono font-bold rounded bg-amber-950 text-amber-300 border border-amber-800 uppercase">
                            {entry.category}
                          </span>
                        </div>
                        <p className="text-body-ui text-text-secondary text-xs mt-1 leading-snug">
                          {entry.headline}
                        </p>
                      </div>

                      <span className="px-2.5 py-1 text-xs font-mono font-bold rounded bg-bg-surface-raised hover:bg-bg-surface-elevated text-text-secondary border border-border-subtle shrink-0">
                        Inspect
                      </span>
                    </div>

                    <div className="flex items-center justify-between pt-2 border-t border-border-subtle/60 text-[11px] font-mono text-text-muted">
                      <span>{entry.itemCount} metric update{entry.itemCount > 1 ? "s" : ""}</span>
                      <span className="text-accent-info group-hover:underline">View Corridors</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* SECTION 3: COLLAPSED SUMMARY (L2 / OVERFLOW) */}
          {feed.summaryCount > 0 && (
            <div className="p-4 rounded-xl bg-bg-surface-raised border border-border-subtle flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-accent-info font-bold">ℹ️</span>
                <span className="text-body-ui text-text-secondary text-xs">
                  <strong>+{feed.summaryCount} Additional Assets</strong> experienced non-critical adjustments within established tolerance bands.
                </span>
              </div>

              <button
                type="button"
                onClick={() => setExpandedSummary(!expandedSummary)}
                className="text-xs font-mono text-accent-info hover:underline cursor-pointer"
              >
                {expandedSummary ? "Hide details ▲" : "View overview ▼"}
              </button>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
