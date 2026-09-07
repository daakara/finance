"use client";

import React, { useState, useEffect, useRef } from "react";
import { DeltaReport } from "../../types/change-intelligence";
import { trackTelemetryEvent } from "../../telemetry/tracker";

export interface DeltaBannerProps {
  report: DeltaReport;
  onAcknowledge: () => void | Promise<void>;
  onViewDiagnostics?: () => void;
  className?: string;
}

export default function DeltaBanner({
  report,
  onAcknowledge,
  onViewDiagnostics,
  className = "",
}: DeltaBannerProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isAcknowledging, setIsAcknowledging] = useState(false);
  const openTimeRef = useRef<number>(0);

  const {
    ticker,
    daysSinceBaseline,
    baselineTimestamp,
    items,
    maxSeverity,
    isMaterial,
    headline,
  } = report;

  // Track telemetry on display
  useEffect(() => {
    openTimeRef.current = performance.now();
    trackTelemetryEvent(
      "DECISION",
      "delta_banner_displayed",
      {
        ticker,
        maxSeverity,
        itemCount: items.length,
        daysSinceBaseline,
      },
      ticker
    );
  }, [ticker, maxSeverity, items.length, daysSinceBaseline]);

  // If change is not material (L0 or L1), suppress the intrusive banner
  if (!isMaterial || items.length === 0) {
    return null;
  }

  const handleAcknowledgeClick = async () => {
    setIsAcknowledging(true);
    const timeToConfirmMs = Math.round(performance.now() - openTimeRef.current);

    trackTelemetryEvent(
      "DECISION",
      "delta_acknowledged",
      {
        ticker,
        maxSeverity,
        timeToConfirmMs,
      },
      ticker
    );

    try {
      await onAcknowledge();
    } finally {
      setIsAcknowledging(false);
    }
  };

  const handleToggleExpand = () => {
    setIsExpanded((prev) => {
      const next = !prev;
      if (next) {
        trackTelemetryEvent(
          "DECISION",
          "delta_banner_expanded",
          { ticker },
          ticker
        );
      }
      return next;
    });
  };

  // Anti-Cyan Color Scheme
  const isCritical = maxSeverity === "CRITICAL";
  const bannerColors = isCritical
    ? "bg-rose-950/40 border-rose-600/80 text-rose-100 shadow-rose-950/40"
    : "bg-emerald-950/40 border-emerald-600/80 text-emerald-100 shadow-emerald-950/40";

  const badgeColors = isCritical
    ? "bg-rose-900/60 border-rose-500 text-rose-200"
    : "bg-emerald-900/60 border-emerald-500 text-emerald-200";

  const formattedBaselineDate = baselineTimestamp ? baselineTimestamp.slice(0, 10) : "previous session";

  return (
    <aside
      data-testid="delta-banner"
      role="status"
      aria-live="polite"
      aria-label={`Thesis Change Intelligence Alert for ${ticker}`}
      className={`w-full p-4 rounded-2xl border shadow-xl flex flex-col gap-3 font-sans transition-all animate-fadeIn ${bannerColors} ${className}`}
    >
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
        {/* Left: Indicator & Headline */}
        <div className="flex items-start gap-3">
          <span className="text-xl shrink-0 mt-0.5" aria-hidden="true">
            {isCritical ? "🚨" : "⚡"}
          </span>

          <div className="space-y-1">
            <div className="flex items-center gap-2 flex-wrap">
              <span className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded border font-bold ${badgeColors}`}>
                Stage 6 · {maxSeverity} Delta
              </span>
              <span className="text-xs font-mono text-text-secondary">
                Since your last review {daysSinceBaseline === 0 ? "earlier today" : `${daysSinceBaseline} day${daysSinceBaseline > 1 ? "s" : ""} ago`}:
              </span>
            </div>

            <h4 className="text-body-ui font-bold text-text-primary">
              {headline}
            </h4>
          </div>
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-2.5 shrink-0 self-end md:self-center">
          <button
            type="button"
            onClick={handleToggleExpand}
            className="px-3 py-1.5 rounded-lg bg-bg-surface-raised hover:bg-bg-surface-elevated border border-border-subtle text-xs font-mono text-text-secondary hover:text-text-primary transition-colors cursor-pointer"
          >
            {isExpanded ? "Hide Details ▲" : `View ${items.length} Changes ▼`}
          </button>

          <button
            type="button"
            onClick={handleAcknowledgeClick}
            disabled={isAcknowledging}
            className="px-4 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-mono text-xs font-black shadow-md transition-all active:scale-95 cursor-pointer disabled:opacity-50 flex items-center gap-1.5"
          >
            <span>{isAcknowledging ? "Updating..." : "✓ Acknowledge & Update Baseline"}</span>
          </button>
        </div>
      </div>

      {/* Expanded Change Details Grid */}
      {isExpanded && (
        <div className="pt-3 border-t border-border-subtle/60 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5 animate-fadeIn">
          {items.map((item, idx) => (
            <div
              key={idx}
              className="p-2.5 rounded-xl bg-bg-surface/80 border border-border-subtle/80 flex flex-col justify-between space-y-1 text-xs font-mono"
            >
              <div className="flex items-center justify-between">
                <span className="text-[10px] uppercase text-text-muted font-bold">
                  {item.category} · {item.field}
                </span>
                <span
                  className={`text-[10px] px-1.5 py-0.2 rounded border font-bold ${
                    item.severity === "CRITICAL"
                      ? "bg-rose-950 text-rose-300 border-rose-800"
                      : "bg-emerald-950 text-emerald-300 border-emerald-800"
                  }`}
                >
                  {item.deltaDisplay}
                </span>
              </div>
              <p className="text-[11px] text-text-secondary leading-snug font-sans">
                {item.reason}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Footer Diagnostic Link */}
      {onViewDiagnostics && (
        <div className="pt-1 text-[10px] font-mono text-text-muted flex items-center justify-between">
          <span>Baseline established {formattedBaselineDate}</span>
          <button
            type="button"
            onClick={onViewDiagnostics}
            className="hover:text-accent-info hover:underline transition-colors cursor-pointer"
          >
            View 30-Day Rolling Diagnostics →
          </button>
        </div>
      )}
    </aside>
  );
}
