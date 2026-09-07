"use client";

import React, { useState } from "react";
import { JourneyMilestone } from "../../types/personal-intelligence";

export interface LearningJourneyTimelineProps {
  currentScore?: number;
  targetScore?: number;
  milestones?: JourneyMilestone[];
  className?: string;
}

export default function LearningJourneyTimeline({
  currentScore = 74,
  targetScore = 80,
  milestones = [
    {
      period: "2025 Q1",
      score: 62,
      delta: 0,
      milestone: "Baseline Architecture Established",
      keyImprovement: "Stop Discipline Framework",
      impactPts: 0,
      status: "COMPLETED",
    },
    {
      period: "2025 Q2",
      score: 64,
      delta: 2.0,
      milestone: "Stop Governance Framework",
      keyImprovement: "Hard Stop Governance (-3.5%)",
      impactPts: 2.0,
      status: "COMPLETED",
    },
    {
      period: "2025 Q3",
      score: 67,
      delta: 3.0,
      milestone: "Macro Regime Gating",
      keyImprovement: "Macro Regime Filter (VIX < 22)",
      impactPts: 3.0,
      status: "COMPLETED",
    },
    {
      period: "2025 Q4",
      score: 71,
      delta: 4.0,
      milestone: "Conviction Scaling & Flow",
      keyImprovement: "Institutional Flow Absorption (>2σ)",
      impactPts: 4.0,
      status: "COMPLETED",
    },
    {
      period: "Today",
      score: 74,
      delta: 3.0,
      milestone: "Institutional Playbook & Attribution",
      keyImprovement: "Personal Decision Playbook Loop",
      impactPts: 3.0,
      status: "ACTIVE",
    },
  ],
  className = "",
}: LearningJourneyTimelineProps) {
  const [expandedMilestone, setExpandedMilestone] = useState<string | null>(null);
  const pointsRemaining = Math.max(0, targetScore - currentScore);

  return (
    <div
      role="region"
      aria-label="Learning Journey Timeline"
      className={`rounded-2xl border border-border-subtle bg-surface-card p-6 md:p-8 shadow-sm max-w-[1440px] mx-auto ${className}`}
    >
      {/* Header: Title and Time Horizon */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border-subtle pb-5">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-text-primary">
              Your Decision Evolution · Last 12 Months
            </h3>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Chronological decision quality progression, milestone achievements, and cumulative impact.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="text-text-muted">Evolution:</span>
          <span className="px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 font-bold border border-emerald-500/20">
            62 → 74 (+19.3% Gain)
          </span>
        </div>
      </div>

      {/* Desktop Trajectory Curve (Hidden on Small Mobile) */}
      <div className="mt-8 hidden md:block">
        <div className="relative p-6 rounded-xl border border-border-subtle bg-surface-subtle/40">
          <div className="flex justify-between items-center text-xs font-mono text-text-muted mb-4 border-b border-border-subtle/60 pb-2">
            <span>Decision Quality Trajectory</span>
            <span className="text-purple-300 font-semibold">Target: {targetScore} Points</span>
          </div>

          {/* SVG Step Curve */}
          <div className="relative h-48 w-full flex items-end justify-between px-6 pt-6">
            {/* Background Grid Lines */}
            <div className="absolute inset-x-6 top-6 border-b border-dashed border-purple-500/20 flex justify-end">
              <span className="text-[10px] font-mono text-purple-400 pr-2">Target 80</span>
            </div>
            <div className="absolute inset-x-6 top-20 border-b border-dashed border-border-subtle/40 flex justify-end">
              <span className="text-[10px] font-mono text-text-muted pr-2">70</span>
            </div>
            <div className="absolute inset-x-6 top-32 border-b border-dashed border-border-subtle/40 flex justify-end">
              <span className="text-[10px] font-mono text-text-muted pr-2">60</span>
            </div>

            {/* Milestones Visual Points */}
            {milestones.map((ms, idx) => {
              const isCurrent = ms.status === "ACTIVE";
              // Calculate height percentage based on min 55 and max 85
              const heightPct = Math.max(10, Math.min(100, ((ms.score - 55) / 30) * 100));

              return (
                <div key={ms.period} className="relative z-10 flex flex-col items-center group">
                  <div
                    className={`flex items-center justify-center h-10 w-10 rounded-full font-mono text-xs font-bold border-2 transition-transform duration-200 group-hover:scale-110 ${
                      isCurrent
                        ? "bg-emerald-500/20 border-emerald-400 text-emerald-400 shadow-sm"
                        : "bg-surface-card border-border-subtle text-text-primary"
                    }`}
                  >
                    {ms.score}
                  </div>

                  {ms.delta > 0 && (
                    <span className="mt-1 text-[10px] font-mono text-emerald-400 font-semibold">
                      +{ms.delta}
                    </span>
                  )}

                  <span className="mt-2 text-xs font-mono text-text-muted font-medium">
                    {ms.period}
                  </span>

                  {isCurrent && (
                    <span className="mt-1 px-1.5 py-0.5 rounded text-[9px] font-mono font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                      CURRENT
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Milestone Cards Grid (Desktop & Tablet) */}
      <div className="mt-6 hidden sm:grid grid-cols-2 lg:grid-cols-5 gap-3.5">
        {milestones.map((ms) => (
          <div
            key={ms.period}
            className={`p-4 rounded-xl border transition-all ${
              ms.status === "ACTIVE"
                ? "border-emerald-500/40 bg-emerald-500/5 shadow-xs"
                : "border-border-subtle bg-surface-subtle/40"
            }`}
          >
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-text-muted">{ms.period}</span>
              <span className="font-bold text-text-primary">{ms.score}/100</span>
            </div>
            <h4 className="mt-2 text-xs font-mono font-bold text-text-primary truncate" title={ms.milestone}>
              {ms.milestone}
            </h4>
            <p className="mt-1 text-[11px] text-text-muted leading-relaxed line-clamp-2">
              {ms.keyImprovement}
            </p>
            {ms.impactPts > 0 && (
              <span className="mt-2 inline-block text-[10px] font-mono text-emerald-400 font-semibold">
                Impact: +{ms.impactPts} pts
              </span>
            )}
          </div>
        ))}
      </div>

      {/* Mobile Vertical Timeline (Visible on Small Viewports) */}
      <div className="mt-6 sm:hidden space-y-4">
        {/* Mobile Current Score Card */}
        <div className="p-4 rounded-xl border border-emerald-500/30 bg-emerald-500/5 text-center">
          <span className="text-xs font-mono text-text-muted uppercase tracking-wider block">Current Score</span>
          <div className="text-3xl font-bold font-mono text-emerald-400 mt-1">{currentScore}</div>
          <span className="text-xs font-mono text-emerald-400 font-semibold block mt-1">▲ Improving · Top 18% Decile</span>
        </div>

        {/* Vertical Step Nodes */}
        <div className="space-y-3 pl-2 border-l-2 border-border-subtle">
          {milestones.map((ms) => {
            const isExpanded = expandedMilestone === ms.period;

            return (
              <div
                key={ms.period}
                onClick={() => setExpandedMilestone(isExpanded ? null : ms.period)}
                className="cursor-pointer pl-4 relative"
              >
                <div className={`absolute -left-[9px] top-1.5 h-4 w-4 rounded-full border-2 ${
                  ms.status === "ACTIVE" ? "bg-emerald-400 border-emerald-200" : "bg-surface-card border-border-subtle"
                }`} />

                <div className="p-3 rounded-xl border border-border-subtle bg-surface-subtle/50">
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="font-bold text-text-primary">{ms.period}</span>
                    <span className="text-emerald-400 font-bold">{ms.score} Pts</span>
                  </div>
                  <h4 className="text-xs font-mono text-text-secondary mt-1">{ms.milestone}</h4>

                  {isExpanded && (
                    <div className="mt-2 pt-2 border-t border-border-subtle text-[11px] text-text-muted space-y-1 animate-fadeIn">
                      <p><strong>Improvement:</strong> {ms.keyImprovement}</p>
                      <p><strong>Impact:</strong> +{ms.impactPts} Quality Points</p>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Next Milestone Progress Bar Card */}
      <div className="mt-6 p-4 rounded-xl border border-purple-500/20 bg-purple-950/10 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="space-y-1">
          <span className="text-xs font-mono uppercase tracking-wider text-purple-300 font-bold">
            Next Milestone Target: Tier 80 Quality Score
          </span>
          <p className="text-xs text-text-secondary">
            {pointsRemaining} points remaining to achieve institutional Elite Decile status.
          </p>
        </div>

        <div className="w-full sm:w-64 space-y-1.5">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-text-muted">{currentScore}/100</span>
            <span className="text-purple-300 font-bold">{targetScore} Goal</span>
          </div>
          <div className="w-full bg-surface-base h-2.5 rounded-full overflow-hidden border border-purple-500/20">
            <div
              className="bg-gradient-to-r from-purple-600 to-purple-400 h-full rounded-full transition-all duration-700 ease-out"
              style={{ width: `${Math.min(100, (currentScore / targetScore) * 100)}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
