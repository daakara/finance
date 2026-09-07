"use client";

import React, { useState } from "react";

export interface FailureDriverItem {
  id: string;
  name: string;
  percentage: number;
  avgLoss: number;
  instancesCount: number;
  severity: "CRITICAL" | "HIGH" | "MEDIUM";
  description: string;
  mitigation: string;
}

export interface FailureDriversCardProps {
  drivers?: FailureDriverItem[];
  className?: string;
}

export default function FailureDriversCard({
  drivers = [
    {
      id: "regime-det",
      name: "Regime Deterioration",
      percentage: 42.0,
      avgLoss: -6.2,
      instancesCount: 146,
      severity: "CRITICAL",
      description: "Macro transition to DEFENSIVE regime; VIX volatility surge invalidating equity thesis",
      mitigation: "Enforce strict macro regime gating: suppress all breakout entries when VIX > 22 or inverted term structure.",
    },
    {
      id: "gap-fade",
      name: "Gap Fade Traps",
      percentage: 38.2,
      avgLoss: -4.8,
      instancesCount: 133,
      severity: "HIGH",
      description: "Entering extended opening gap-ups (>3.0%) into overhead structural resistance",
      mitigation: "Institute a mandatory 15-minute price discovery cooldown; require intraday consolidation base before execution.",
    },
    {
      id: "late-mom",
      name: "Late Momentum Chasing",
      percentage: 35.0,
      avgLoss: -5.1,
      instancesCount: 122,
      severity: "HIGH",
      description: "Entering on Day 4+ of vertical thrust without consolidation or volume support",
      mitigation: "Enforce Minervini Stage 2 base maturity criteria; reject setups more than 15% extended from 20-day EMA.",
    },
    {
      id: "flow-rev",
      name: "Flow Reversal Exhaustion",
      percentage: 24.1,
      avgLoss: -3.9,
      instancesCount: 84,
      severity: "MEDIUM",
      description: "Dark pool distribution disguised by retail aggressive market buying",
      mitigation: "Cross-reference block trade flow with tape speed; abort if institutional net flow turns negative.",
    },
  ],
  className = "",
}: FailureDriversCardProps) {
  const [selectedDriverId, setSelectedDriverId] = useState<string | null>(null);
  const [metricMode, setMetricMode] = useState<"percentage" | "avgLoss">("percentage");

  const sortedDrivers = [...drivers].sort((a, b) =>
    metricMode === "percentage" ? b.percentage - a.percentage : a.avgLoss - b.avgLoss
  );

  return (
    <div
      role="region"
      aria-label="Failure Drivers Card"
      className={`rounded-2xl border border-border-subtle bg-surface-card p-6 shadow-sm ${className}`}
    >
      {/* Header with Title & Sort Toggle */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border-subtle pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-rose-400" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-text-primary">
              What Doesn&apos;t Work · Root Cause Analysis
            </h3>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Systemic error modes, execution traps, and root causes of thesis invalidation.
          </p>
        </div>

        <div className="flex items-center bg-surface-base border border-border-subtle rounded-lg p-0.5 text-xs font-mono">
          <button
            type="button"
            onClick={() => setMetricMode("percentage")}
            className={`px-2.5 py-1 rounded-md transition-colors ${
              metricMode === "percentage"
                ? "bg-surface-raised text-rose-400 font-semibold"
                : "text-text-muted hover:text-text-secondary"
            }`}
          >
            Failure Share
          </button>
          <button
            type="button"
            onClick={() => setMetricMode("avgLoss")}
            className={`px-2.5 py-1 rounded-md transition-colors ${
              metricMode === "avgLoss"
                ? "bg-surface-raised text-rose-400 font-semibold"
                : "text-text-muted hover:text-text-secondary"
            }`}
          >
            Avg Loss
          </button>
        </div>
      </div>

      {/* Driver List */}
      <div className="mt-5 space-y-3">
        {sortedDrivers.map((driver) => {
          const isSelected = selectedDriverId === driver.id;

          return (
            <div
              key={driver.id}
              onClick={() => setSelectedDriverId(isSelected ? null : driver.id)}
              className={`group cursor-pointer rounded-xl border p-4 transition-all duration-200 ${
                isSelected
                  ? "border-rose-500/50 bg-rose-500/5 shadow-xs"
                  : "border-border-subtle/80 bg-surface-subtle/40 hover:border-border-subtle hover:bg-surface-subtle"
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-mono text-sm font-bold text-text-primary group-hover:text-rose-300 transition-colors">
                      {driver.name}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${
                        driver.severity === "CRITICAL"
                          ? "bg-rose-500/15 text-rose-400 border border-rose-500/30"
                          : driver.severity === "HIGH"
                          ? "bg-amber-500/15 text-amber-400 border border-amber-500/30"
                          : "bg-surface-subtle text-text-muted border border-border-subtle"
                      }`}
                    >
                      {driver.severity}
                    </span>
                  </div>
                  <p className="text-xs text-text-secondary">{driver.description}</p>
                </div>

                <div className="text-right shrink-0">
                  <div className="font-mono text-base font-bold text-rose-400">
                    {metricMode === "percentage" ? `${driver.percentage.toFixed(1)}%` : `${driver.avgLoss.toFixed(1)}%`}
                  </div>
                  <span className="font-mono text-[11px] text-text-muted">
                    {driver.instancesCount} instances
                  </span>
                </div>
              </div>

              {/* Visual Progress Bar */}
              <div className="mt-3 w-full bg-surface-base h-2 rounded-full overflow-hidden border border-border-subtle/40">
                <div
                  className="bg-gradient-to-r from-rose-700 to-rose-400 h-full rounded-full transition-all duration-700 ease-out"
                  style={{ width: `${Math.min(100, metricMode === "percentage" ? driver.percentage : Math.abs(driver.avgLoss) * 10)}%` }}
                />
              </div>

              {/* Expandable Mitigation Guidance */}
              {isSelected && (
                <div className="mt-3 pt-3 border-t border-rose-500/20 text-xs text-text-secondary space-y-1 animate-fadeIn">
                  <div className="flex items-center gap-1.5 font-mono text-[11px] text-rose-400 font-semibold">
                    <span>🛡️ Actionable Mitigation Rule:</span>
                  </div>
                  <p className="leading-relaxed">{driver.mitigation}</p>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer Insight */}
      <div className="mt-5 pt-4 border-t border-border-subtle/60 flex items-center justify-between text-xs font-mono">
        <span className="text-text-muted">Preservation Potential:</span>
        <span className="font-bold text-rose-400 bg-rose-500/10 px-2.5 py-1 rounded border border-rose-500/20">
          +7.6% Capital Saved via Top 2 Rules
        </span>
      </div>
    </div>
  );
}
