"use client";

import React, { useState } from "react";

export interface TrendStep {
  period: string;
  score: number;
  delta: number;
  milestone: string;
  keyLearning: string;
  status: "ACTIVE" | "COMPLETED";
}

export interface DecisionQualityTrendProps {
  steps?: TrendStep[];
  className?: string;
}

export default function DecisionQualityTrend({
  steps = [
    {
      period: "Q1 2026",
      score: 62,
      delta: 0,
      milestone: "Baseline Architecture Established",
      keyLearning: "Initial deployment identified high discretionary slippage and lack of macro regime filters.",
      status: "COMPLETED",
    },
    {
      period: "Q2 2026",
      score: 64,
      delta: 2.0,
      milestone: "Introduced Automated Stop Floor",
      keyLearning: "Hard stop floor at -3.5% capped severe tail drawdowns and reduced maximum loss per trade.",
      status: "COMPLETED",
    },
    {
      period: "Q3 2026",
      score: 67,
      delta: 3.0,
      milestone: "Macro Regime Gating Integrated",
      keyLearning: "Filtering trades by macro regime (EXPANSION vs DEFENSIVE) eliminated 65% of chop whipsaws.",
      status: "COMPLETED",
    },
    {
      period: "Q4 2026",
      score: 71,
      delta: 4.0,
      milestone: "Dark Pool Accumulation Calibrated",
      keyLearning: "Requiring volume absorption >+2.0σ boosted primary setup win rate from 58% to 68%.",
      status: "COMPLETED",
    },
    {
      period: "Current",
      score: 74,
      delta: 3.0,
      milestone: "Full Learning Loop & Attribution Engine",
      keyLearning: "Closing the loop between predictions, outcomes, and AI coaching drove decision quality to top 18% decile.",
      status: "ACTIVE",
    },
  ],
  className = "",
}: DecisionQualityTrendProps) {
  const [selectedPeriod, setSelectedPeriod] = useState<string>("Current");

  const currentStep = steps.find((s) => s.period === selectedPeriod) || steps[steps.length - 1];

  return (
    <div
      role="region"
      aria-label="Decision Quality Trend"
      className={`rounded-2xl border border-border-subtle bg-surface-card p-6 shadow-sm ${className}`}
    >
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border-subtle pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-text-primary">
              Decision Quality Trajectory &amp; Learning Milestones
            </h3>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Empirical progression of institutional decision score across five consecutive review cycles.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="text-text-muted">Net Evolution:</span>
          <span className="px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 font-bold border border-emerald-500/20">
            62 → 74 (+12 pts)
          </span>
        </div>
      </div>

      {/* Trajectory Timeline Visualization */}
      <div className="mt-6 grid grid-cols-5 gap-2 sm:gap-4 relative">
        {steps.map((step) => {
          const isSelected = selectedPeriod === step.period;
          const isCurrent = step.period === "Current";

          return (
            <div
              key={step.period}
              onClick={() => setSelectedPeriod(step.period)}
              className={`cursor-pointer rounded-xl p-3 sm:p-4 text-center border transition-all duration-200 ${
                isSelected
                  ? "border-emerald-500/60 bg-emerald-500/10 shadow-xs ring-1 ring-emerald-500/30"
                  : "border-border-subtle bg-surface-subtle/40 hover:bg-surface-subtle"
              }`}
            >
              <span className="text-[10px] sm:text-xs font-mono text-text-muted block uppercase tracking-wider">
                {step.period}
              </span>
              <div className="mt-2 text-xl sm:text-2xl font-bold font-mono text-text-primary">
                {step.score}
              </div>
              <div className="mt-1 flex items-center justify-center gap-1 font-mono text-[10px] sm:text-xs">
                {step.delta > 0 ? (
                  <span className="text-emerald-400 font-semibold">+{step.delta}</span>
                ) : (
                  <span className="text-text-muted">Base</span>
                )}
              </div>
              {isCurrent && (
                <span className="mt-2 inline-block px-1.5 py-0.5 rounded text-[9px] font-mono font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                  LIVE
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Detail Callout for Selected Step */}
      <div className="mt-5 rounded-xl border border-border-subtle/80 bg-surface-subtle/60 p-4 animate-fadeIn">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border-subtle/60 pb-2">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs font-bold text-emerald-400 uppercase">
              {currentStep.period} Milestone:
            </span>
            <span className="font-mono text-xs font-semibold text-text-primary">
              {currentStep.milestone}
            </span>
          </div>
          <span className="text-xs font-mono text-text-muted">
            Decision Score: <strong className="text-text-primary">{currentStep.score}/100</strong>
          </span>
        </div>
        <p className="mt-2.5 text-xs text-text-secondary leading-relaxed">
          {currentStep.keyLearning}
        </p>
      </div>
    </div>
  );
}
