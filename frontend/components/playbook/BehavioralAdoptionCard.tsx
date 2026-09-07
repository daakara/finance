"use client";

import React from "react";
import { AdoptionMetrics } from "../../types/personal-intelligence";

export interface BehavioralAdoptionCardProps {
  metrics?: AdoptionMetrics;
  className?: string;
}

export default function BehavioralAdoptionCard({
  metrics = {
    recommendationsIssued: 112,
    recommendationsFollowed: 79,
    behavioralAdoptionRate: 70.5,
    repeatMistakeRate: -43.0,
    decisionDrift: 21.0,
    driftClassification: "LOW",
    ruleAdherence: {
      overall: 87.0,
      stopDiscipline: 91.0,
      macroRules: 72.0,
      positionSizing: 87.0,
      riskControls: 94.0,
    },
    complianceBreakdown: {
      doMore: 81.0,
      stopDoing: 74.0,
      calibrate: 63.0,
    },
  },
  className = "",
}: BehavioralAdoptionCardProps) {
  const getDriftBadge = (classification: "LOW" | "MEDIUM" | "HIGH") => {
    switch (classification) {
      case "LOW":
        return "bg-emerald-500/10 text-emerald-400 border-emerald-500/30";
      case "MEDIUM":
        return "bg-amber-500/10 text-amber-400 border-amber-500/30";
      case "HIGH":
        return "bg-rose-500/10 text-rose-400 border-rose-500/30";
    }
  };

  return (
    <div
      role="region"
      aria-label="Behavioral Adoption Card"
      className={`rounded-2xl border border-border-subtle bg-surface-card p-6 md:p-8 shadow-sm ${className}`}
    >
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border-subtle pb-5">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-text-primary">
              Behavioral Adoption &amp; Adherence Analytics
            </h3>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Tracking execution discipline, adherence to coach prescriptions, and playbook drift.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="text-text-muted">Issued Recommendations:</span>
          <span className="px-2.5 py-1 rounded bg-surface-base text-text-primary font-bold border border-border-subtle">
            {metrics.recommendationsIssued} Total
          </span>
        </div>
      </div>

      {/* Primary KPI Grid: BAR, Rule Adherence, Repeat Mistakes, Decision Drift */}
      <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Metric 1: Behavioral Adoption Rate (BAR) */}
        <div className="rounded-xl border border-border-subtle bg-surface-subtle/50 p-4 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono uppercase tracking-wider text-text-muted">
              Adoption Rate (BAR)
            </span>
            <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">
              ▲ +8.4%
            </span>
          </div>
          <div className="text-2xl font-bold font-mono text-text-primary mt-1">
            {metrics.behavioralAdoptionRate.toFixed(1)}%
          </div>
          <p className="text-[11px] text-text-muted font-mono">
            {metrics.recommendationsFollowed} of {metrics.recommendationsIssued} followed
          </p>
        </div>

        {/* Metric 2: Rule Adherence */}
        <div className="rounded-xl border border-border-subtle bg-surface-subtle/50 p-4 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono uppercase tracking-wider text-text-muted">
              Rule Adherence
            </span>
            <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">
              High
            </span>
          </div>
          <div className="text-2xl font-bold font-mono text-text-primary mt-1">
            {metrics.ruleAdherence.overall.toFixed(0)}%
          </div>
          <p className="text-[11px] text-text-muted font-mono">
            Across 4 core discipline categories
          </p>
        </div>

        {/* Metric 3: Repeat Mistakes */}
        <div className="rounded-xl border border-border-subtle bg-surface-subtle/50 p-4 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono uppercase tracking-wider text-text-muted">
              Repeat Mistakes
            </span>
            <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">
              {metrics.repeatMistakeRate.toFixed(0)}%
            </span>
          </div>
          <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">
            12 Instances
          </div>
          <p className="text-[11px] text-text-muted font-mono">
            Down from 21 prior quarter
          </p>
        </div>

        {/* Metric 4: Decision Drift */}
        <div className="rounded-xl border border-border-subtle bg-surface-subtle/50 p-4 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono uppercase tracking-wider text-text-muted">
              Decision Drift
            </span>
            <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded border ${getDriftBadge(metrics.driftClassification)}`}>
              {metrics.driftClassification} RISK
            </span>
          </div>
          <div className="text-2xl font-bold font-mono text-text-primary mt-1">
            {metrics.decisionDrift.toFixed(0)}%
          </div>
          <p className="text-[11px] text-text-muted font-mono">
            Playbook deviation metric
          </p>
        </div>
      </div>

      {/* Detailed Adherence Breakdown Rows */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6 pt-5 border-t border-border-subtle/60">
        {/* Panel A: Recommendation Compliance */}
        <div className="space-y-3">
          <span className="text-xs font-mono uppercase tracking-wider text-text-muted font-bold block">
            Recommendation Compliance by Category
          </span>
          <div className="space-y-2.5">
            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-text-secondary">DO MORE (Scale Alpha)</span>
                <span className="font-bold text-emerald-400">{metrics.complianceBreakdown.doMore}%</span>
              </div>
              <div className="w-full bg-surface-base h-2 rounded-full overflow-hidden border border-border-subtle/40">
                <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${metrics.complianceBreakdown.doMore}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-text-secondary">STOP DOING (Eliminate Traps)</span>
                <span className="font-bold text-rose-400">{metrics.complianceBreakdown.stopDoing}%</span>
              </div>
              <div className="w-full bg-surface-base h-2 rounded-full overflow-hidden border border-border-subtle/40">
                <div className="bg-rose-500 h-full rounded-full" style={{ width: `${metrics.complianceBreakdown.stopDoing}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-text-secondary">CALIBRATE (Sizing &amp; Stops)</span>
                <span className="font-bold text-purple-300">{metrics.complianceBreakdown.calibrate}%</span>
              </div>
              <div className="w-full bg-surface-base h-2 rounded-full overflow-hidden border border-border-subtle/40">
                <div className="bg-purple-500 h-full rounded-full" style={{ width: `${metrics.complianceBreakdown.calibrate}%` }} />
              </div>
            </div>
          </div>
        </div>

        {/* Panel B: Rule Adherence Breakdown */}
        <div className="space-y-3">
          <span className="text-xs font-mono uppercase tracking-wider text-text-muted font-bold block">
            Rule Adherence by Discipline
          </span>
          <div className="space-y-2.5">
            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-text-secondary">Risk Controls Floor</span>
                <span className="font-bold text-emerald-400">{metrics.ruleAdherence.riskControls}%</span>
              </div>
              <div className="w-full bg-surface-base h-2 rounded-full overflow-hidden border border-border-subtle/40">
                <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${metrics.ruleAdherence.riskControls}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-text-secondary">Stop Discipline</span>
                <span className="font-bold text-emerald-400">{metrics.ruleAdherence.stopDiscipline}%</span>
              </div>
              <div className="w-full bg-surface-base h-2 rounded-full overflow-hidden border border-border-subtle/40">
                <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${metrics.ruleAdherence.stopDiscipline}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-text-secondary">Position Sizing Rules</span>
                <span className="font-bold text-purple-300">{metrics.ruleAdherence.positionSizing}%</span>
              </div>
              <div className="w-full bg-surface-base h-2 rounded-full overflow-hidden border border-border-subtle/40">
                <div className="bg-purple-500 h-full rounded-full" style={{ width: `${metrics.ruleAdherence.positionSizing}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-text-secondary">Macro Gating</span>
                <span className="font-bold text-amber-400">{metrics.ruleAdherence.macroRules}%</span>
              </div>
              <div className="w-full bg-surface-base h-2 rounded-full overflow-hidden border border-border-subtle/40">
                <div className="bg-amber-500 h-full rounded-full" style={{ width: `${metrics.ruleAdherence.macroRules}%` }} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
