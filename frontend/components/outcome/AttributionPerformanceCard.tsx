"use client";

import React from "react";

export interface AttributionPerformanceCardProps {
  className?: string;
}

export default function AttributionPerformanceCard({
  className = "",
}: AttributionPerformanceCardProps) {
  const topDrivers = [
    { name: "Institutional Accumulation", winRate: 72, count: 684, desc: "Positive dark-pool & block flow velocity (+1.5σ)" },
    { name: "Regime Alignment", winRate: 69, count: 512, desc: "Macro tailwind supporting equity risk-on assets" },
    { name: "Validation Promotion", winRate: 67, count: 428, desc: "Quantitative confluence across VCP and Magic Formula" },
  ];

  const failureDrivers = [
    { name: "Regime Deterioration", pct: 42, count: 146, desc: "VIX spike / macro transition to DEFENSIVE regime" },
    { name: "Flow Reversal", pct: 24, count: 83, desc: "Institutional distribution during entry corridor" },
    { name: "Stop Triggered", pct: 21, count: 73, desc: "Excessive volatility breaching stop floor" },
  ];

  return (
    <div
      role="region"
      aria-label="Attribution Performance Card"
      className={`rounded-lg border border-border-subtle bg-surface-card p-5 ${className}`}
    >
      <h3 className="text-xs font-semibold uppercase tracking-wider text-text-primary border-b border-border-subtle pb-3">
        Attribution Performance &amp; Failure Analysis
      </h3>

      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Panel A: Top Drivers */}
        <div className="space-y-3">
          <span className="text-[11px] font-mono text-emerald-400 font-bold uppercase tracking-wider flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            Most Successful Prediction Drivers
          </span>
          <div className="space-y-2.5">
            {topDrivers.map((d, i) => (
              <div key={i} className="rounded border border-border-subtle/60 bg-surface-subtle p-2.5">
                <div className="flex justify-between items-center">
                  <span className="font-mono text-xs font-bold text-text-primary">{d.name}</span>
                  <span className="font-mono text-xs font-bold text-emerald-400">{d.winRate}% Win Rate</span>
                </div>
                <p className="text-[11px] text-text-muted mt-1">{d.desc}</p>
                <div className="mt-2 w-full bg-surface-base h-1.5 rounded-full overflow-hidden">
                  <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${d.winRate}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Panel B: Failure Drivers */}
        <div className="space-y-3">
          <span className="text-[11px] font-mono text-accent-rose font-bold uppercase tracking-wider flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-accent-rose" />
            Primary Failure Root Causes
          </span>
          <div className="space-y-2.5">
            {failureDrivers.map((f, i) => (
              <div key={i} className="rounded border border-border-subtle/60 bg-surface-subtle p-2.5">
                <div className="flex justify-between items-center">
                  <span className="font-mono text-xs font-bold text-text-primary">{f.name}</span>
                  <span className="font-mono text-xs font-bold text-accent-rose">{f.pct}% of Failures</span>
                </div>
                <p className="text-[11px] text-text-muted mt-1">{f.desc}</p>
                <div className="mt-2 w-full bg-surface-base h-1.5 rounded-full overflow-hidden">
                  <div className="bg-accent-rose h-full rounded-full" style={{ width: `${f.pct * 2}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
