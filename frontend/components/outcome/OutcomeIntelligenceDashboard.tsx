"use client";

import React from "react";
import { OutcomeSummary } from "../../types/outcome-intelligence";

export interface OutcomeIntelligenceDashboardProps {
  summary?: OutcomeSummary;
  className?: string;
}

export default function OutcomeIntelligenceDashboard({
  summary = {
    totalReviewed: 2184,
    successCount: 1472,
    partialSuccessCount: 323,
    failureCount: 347,
    expiredCount: 42,
    successRate: 67.4,
    partialSuccessRate: 14.8,
    failureRate: 15.9,
    expiredRate: 1.9,
    topDrivers: [
      { driver: "Institutional Accumulation", winRate: 0.72 },
      { driver: "Regime Alignment", winRate: 0.69 },
      { driver: "Validation Promotion", winRate: 0.67 },
    ],
    failureDrivers: [
      { driver: "Regime Deterioration", percentage: 0.42 },
      { driver: "Flow Reversal", percentage: 0.24 },
      { driver: "Stop Triggered", percentage: 0.21 },
    ],
    actionability: {
      reviewedPredictions: 2184,
      displayedPredictions: 3500,
      par: 0.624,
      target: 0.50,
      status: "PASS",
    },
  },
  className = "",
}: OutcomeIntelligenceDashboardProps) {
  const isParPassing = summary.actionability.par >= summary.actionability.target;

  return (
    <div
      role="region"
      aria-label="Outcome Intelligence Dashboard"
      className={`rounded-lg border border-border-subtle bg-surface-card p-5 ${className}`}
    >
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-border-subtle pb-4 gap-2">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-wider text-text-primary flex items-center gap-2">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-500" />
            Outcome Intelligence &amp; Prediction Learning Loop
          </h2>
          <p className="text-xs text-text-muted mt-1">
            Empirical validation of predictive hypotheses against resolved market outcomes.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-surface-subtle text-text-secondary border border-border-subtle">
            Total Resolved: <strong className="text-text-primary font-mono">{summary.totalReviewed.toLocaleString()}</strong>
          </span>
          <span className={`text-xs font-mono font-bold px-2.5 py-1 rounded ${
            isParPassing ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : "bg-accent-rose/10 text-accent-rose"
          }`}>
            PAR: {(summary.actionability.par * 100).toFixed(1)}% (Target ≥50%)
          </span>
        </div>
      </div>

      {/* Outcome Cards Grid */}
      <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="rounded-md border border-emerald-500/20 bg-emerald-500/5 p-3.5">
          <span className="text-[10px] font-mono uppercase text-emerald-400 font-semibold">Full Success</span>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold font-mono text-emerald-400">{summary.successRate}%</span>
            <span className="text-xs font-mono text-text-muted">{summary.successCount}</span>
          </div>
          <p className="text-[10px] text-text-muted mt-1">Target 1 corridor achieved</p>
        </div>

        <div className="rounded-md border border-accent-amber/20 bg-accent-amber/5 p-3.5">
          <span className="text-[10px] font-mono uppercase text-accent-amber font-semibold">Partial Success</span>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold font-mono text-accent-amber">{summary.partialSuccessRate}%</span>
            <span className="text-xs font-mono text-text-muted">{summary.partialSuccessCount}</span>
          </div>
          <p className="text-[10px] text-text-muted mt-1">&gt;50% favorable excursion</p>
        </div>

        <div className="rounded-md border border-accent-rose/20 bg-accent-rose/5 p-3.5">
          <span className="text-[10px] font-mono uppercase text-accent-rose font-semibold">Failure</span>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold font-mono text-accent-rose">{summary.failureRate}%</span>
            <span className="text-xs font-mono text-text-muted">{summary.failureCount}</span>
          </div>
          <p className="text-[10px] text-text-muted mt-1">Stop floor breached</p>
        </div>

        <div className="rounded-md border border-border-subtle bg-surface-subtle p-3.5">
          <span className="text-[10px] font-mono uppercase text-text-muted font-semibold">Expired / Invalidated</span>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold font-mono text-text-primary">{summary.expiredRate}%</span>
            <span className="text-xs font-mono text-text-muted">{summary.expiredCount}</span>
          </div>
          <p className="text-[10px] text-text-muted mt-1">Regime or time elapsed</p>
        </div>
      </div>

      {/* Aggregate Resolution Bar */}
      <div className="mt-4 pt-3 border-t border-border-subtle">
        <div className="flex justify-between text-[11px] font-mono text-text-muted mb-1.5">
          <span>Resolution Distribution</span>
          <span>100% Attributed &amp; Auditable</span>
        </div>
        <div className="h-3 w-full rounded-full bg-surface-base overflow-hidden flex border border-border-subtle/50">
          <div style={{ width: `${summary.successRate}%` }} className="h-full bg-emerald-500" title="Success" />
          <div style={{ width: `${summary.partialSuccessRate}%` }} className="h-full bg-accent-amber" title="Partial" />
          <div style={{ width: `${summary.failureRate}%` }} className="h-full bg-accent-rose" title="Failure" />
          <div style={{ width: `${summary.expiredRate}%` }} className="h-full bg-slate-600" title="Expired" />
        </div>
      </div>
    </div>
  );
}
