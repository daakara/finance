"use client";

import React, { useState } from "react";

export interface AIMentorCardProps {
  currentScore?: number;
  improvementPct?: number;
  largestContributorName?: string;
  largestContributorImpact?: number;
  nextOpportunityAction?: string;
  nextOpportunityGain?: number;
  confidenceScore?: number;
  className?: string;
}

export default function AIMentorCard({
  currentScore = 74,
  improvementPct = 19.3,
  largestContributorName = "Institutional Flow Filtering",
  largestContributorImpact = 6.2,
  nextOpportunityAction = "Reduce exposure during macro deterioration environments",
  nextOpportunityGain = 3.4,
  confidenceScore = 89,
  className = "",
}: AIMentorCardProps) {
  const [evidenceDrawerOpen, setEvidenceDrawerOpen] = useState<boolean>(false);

  return (
    <div
      role="region"
      aria-label="AI Decision Mentor"
      className={`rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-950/20 via-surface-card to-surface-card p-6 md:p-8 shadow-sm backdrop-blur-sm ${className}`}
    >
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-purple-500/20 pb-5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-600/20 border border-purple-500/40 text-purple-300">
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
              />
            </svg>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold uppercase tracking-wider text-text-primary">
                ARX AI Decision Mentor
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">
                ACTIVE
              </span>
            </div>
            <p className="text-xs text-text-muted mt-0.5">
              Personalized Decision Optimization &amp; Process Guidance
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 bg-surface-base/80 border border-purple-500/20 rounded-xl px-4 py-2">
          <div className="text-right">
            <span className="block text-[10px] font-mono uppercase text-text-muted">Mentor Confidence</span>
            <span className="text-sm font-bold font-mono text-purple-300">{confidenceScore}% Statistically Backed</span>
          </div>
          <div className="h-8 w-1.5 rounded-full bg-surface-raised overflow-hidden">
            <div className="h-full bg-gradient-to-t from-purple-600 to-purple-400 rounded-full" style={{ height: `${confidenceScore}%` }} />
          </div>
        </div>
      </div>

      {/* Core Executive Mentor Guidance */}
      <div className="mt-5 p-4 md:p-5 rounded-xl border border-purple-500/20 bg-purple-500/5 space-y-3">
        <p className="text-sm md:text-base text-text-primary leading-relaxed font-medium">
          &quot;You have improved your decision quality by <span className="text-emerald-400 font-bold font-mono">+{improvementPct.toFixed(1)}%</span> over the last 12 months. The largest contributor was <span className="text-purple-300 font-mono font-bold">{largestContributorName}</span>, generating an estimated <span className="text-emerald-400 font-mono font-bold">+{largestContributorImpact} pts</span> in quality score.&quot;
        </p>

        <div className="pt-3 border-t border-purple-500/15 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs font-mono">
          <div className="space-y-0.5">
            <span className="text-text-muted uppercase text-[10px]">Next Highest-Leverage Opportunity:</span>
            <p className="text-text-secondary font-medium">{nextOpportunityAction}</p>
          </div>
          <span className="shrink-0 text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded border border-emerald-500/20 font-bold">
            Projected Impact: +{nextOpportunityGain} pts
          </span>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="mt-5 flex items-center justify-between pt-2">
        <button
          type="button"
          onClick={() => setEvidenceDrawerOpen(!evidenceDrawerOpen)}
          className="px-4 py-2 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-mono text-xs font-semibold shadow-xs transition-colors flex items-center gap-2"
        >
          <span>{evidenceDrawerOpen ? "Hide Evidence" : "Show Evidence"}</span>
          <span className="text-[10px]">▼</span>
        </button>

        <span className="text-xs font-mono text-text-muted">
          Current Decision Quality: <strong className="text-text-primary">{currentScore}/100</strong>
        </span>
      </div>

      {/* Expandable Slide-Over / Evidence Drawer */}
      {evidenceDrawerOpen && (
        <div className="mt-5 pt-4 border-t border-purple-500/20 space-y-4 animate-fadeIn">
          <div className="p-4 rounded-xl border border-border-subtle bg-surface-subtle/60 space-y-3">
            <div className="flex justify-between items-center">
              <h4 className="text-xs font-mono uppercase tracking-wider text-purple-300 font-bold">
                Supporting Evidence &amp; Causal Attribution Records
              </h4>
              <button
                type="button"
                onClick={() => setEvidenceDrawerOpen(false)}
                className="text-xs font-mono text-text-muted hover:text-text-primary px-2 py-1 rounded bg-surface-base border border-border-subtle"
              >
                Close Drawer
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
              <div className="p-3 bg-surface-card rounded-lg border border-border-subtle">
                <span className="text-text-muted block text-[10px]">SUPPORTING DECISIONS</span>
                <span className="text-lg font-bold text-text-primary">842 Outcomes</span>
                <span className="text-[10px] text-emerald-400 block mt-0.5">72.0% Historical Win Rate</span>
              </div>
              <div className="p-3 bg-surface-card rounded-lg border border-border-subtle">
                <span className="text-text-muted block text-[10px]">ATTRIBUTION CORRELATION</span>
                <span className="text-lg font-bold text-purple-300">r = 0.84</span>
                <span className="text-[10px] text-text-muted block mt-0.5">p &lt; 0.001 Statistical Significance</span>
              </div>
              <div className="p-3 bg-surface-card rounded-lg border border-border-subtle">
                <span className="text-text-muted block text-[10px]">PREVENTED DRAWDOWN</span>
                <span className="text-lg font-bold text-emerald-400">-$184,000</span>
                <span className="text-[10px] text-emerald-400 block mt-0.5">Saved via Stop Floor &amp; Gating</span>
              </div>
            </div>

            <p className="text-xs text-text-muted leading-relaxed">
              Every recommendation is cross-verified against your immutable Decision Journal entries. The mentor identifies deviations from your historical winning playbook and flags recurring loss patterns before order execution.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
