"use client";

import React, { useEffect } from "react";
import { MorningBriefingSummary } from "../../types/portfolio-intelligence";
import { trackTelemetryEvent } from "../../telemetry/tracker";

export interface MorningBriefingCardProps {
  summary: MorningBriefingSummary;
  onViewAllAttention?: () => void;
  onSelectTicker?: (ticker: string) => void;
  className?: string;
}

export default function MorningBriefingCard({
  summary,
  onViewAllAttention,
  onSelectTicker,
  className = "",
}: MorningBriefingCardProps) {
  useEffect(() => {
    trackTelemetryEvent(
      "DECISION",
      "morning_brief_viewed",
      {
        criticalCount: summary.criticalCount,
        materialCount: summary.materialCount,
        totalReviewed: summary.totalAssetsReviewed,
      }
    );
  }, [summary.criticalCount, summary.materialCount, summary.totalAssetsReviewed]);

  const hasUrgent = summary.criticalCount > 0;

  return (
    <div
      data-testid="morning-briefing-card"
      className={`p-6 rounded-2xl border transition-all ${
        hasUrgent
          ? "bg-gradient-to-br from-bg-surface to-rose-950/20 border-rose-500/40 shadow-xl shadow-rose-950/10"
          : "bg-bg-surface border-border-subtle shadow-lg"
      } ${className}`}
    >
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-border-subtle/80">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[11px] font-mono uppercase tracking-widest text-text-muted font-bold">
              Institutional Morning Briefing
            </span>
            <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-bg-surface-raised text-text-secondary border border-border-subtle">
              Session Active
            </span>
          </div>
          <h2 className="text-display-2 text-text-primary font-bold tracking-tight">
            {summary.headline}
          </h2>
        </div>

        {onViewAllAttention && (
          <button
            type="button"
            onClick={onViewAllAttention}
            className="px-4 py-2 rounded-xl bg-accent-info/10 hover:bg-accent-info/20 text-accent-info border border-accent-info/30 text-xs font-mono font-bold transition-all shrink-0 cursor-pointer"
          >
            Review All Changes →
          </button>
        )}
      </div>

      {/* Glanceable Metrics Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4">
        <div className="p-3 rounded-xl bg-bg-surface-raised border border-border-subtle">
          <span className="text-[10px] uppercase font-mono text-text-muted font-bold block">
            Critical Actions (L4)
          </span>
          <span className={`text-xl font-mono font-black ${summary.criticalCount > 0 ? "text-rose-400" : "text-text-primary"}`}>
            {summary.criticalCount}
          </span>
        </div>

        <div className="p-3 rounded-xl bg-bg-surface-raised border border-border-subtle">
          <span className="text-[10px] uppercase font-mono text-text-muted font-bold block">
            Material Updates (L3)
          </span>
          <span className={`text-xl font-mono font-black ${summary.materialCount > 0 ? "text-amber-400" : "text-text-primary"}`}>
            {summary.materialCount}
          </span>
        </div>

        <div className="p-3 rounded-xl bg-bg-surface-raised border border-border-subtle">
          <span className="text-[10px] uppercase font-mono text-text-muted font-bold block">
            Assets Monitored
          </span>
          <span className="text-xl font-mono font-black text-text-primary">
            {summary.totalAssetsReviewed}
          </span>
        </div>

        <div className="p-3 rounded-xl bg-bg-surface-raised border border-border-subtle">
          <span className="text-[10px] uppercase font-mono text-text-muted font-bold block">
            Macro Regime
          </span>
          <span className="text-sm font-mono font-bold text-amber-300 flex items-center gap-1 mt-1">
            <span>🛡️</span> {summary.primaryRiskRegime}
          </span>
        </div>
      </div>

      {/* Top Action Tickers Row */}
      {summary.topActionTickers && summary.topActionTickers.length > 0 && (
        <div className="mt-4 pt-3 border-t border-border-subtle flex items-center gap-2 flex-wrap">
          <span className="text-[11px] font-mono text-text-secondary">
            Quick Jump:
          </span>
          {summary.topActionTickers.map((ticker) => (
            <button
              key={ticker}
              type="button"
              onClick={() => onSelectTicker && onSelectTicker(ticker)}
              className="px-2.5 py-1 rounded-lg bg-bg-surface-raised hover:bg-bg-surface-elevated text-xs font-mono font-bold text-text-primary border border-border-subtle transition-colors cursor-pointer"
            >
              ${ticker}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
