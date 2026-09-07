"use client";

import React, { useState } from "react";

export interface LearningSummaryHeroProps {
  score?: number;
  scoreDelta?: number;
  decileRank?: string;
  successRate?: number;
  successRateDelta?: number;
  parRate?: number;
  parTarget?: number;
  learningVelocity?: number;
  className?: string;
}

export default function LearningSummaryHero({
  score = 74,
  scoreDelta = 4.2,
  decileRank = "Top 18% Institutional Decile",
  successRate = 67.4,
  successRateDelta = 4.1,
  parRate = 62.4,
  parTarget = 50.0,
  learningVelocity = 8.2,
  className = "",
}: LearningSummaryHeroProps) {
  const [selectedTimeframe, setSelectedTimeframe] = useState<"30D" | "90D" | "YTD" | "ALL">("90D");

  return (
    <div
      role="region"
      aria-label="Learning Summary Hero"
      className={`rounded-2xl border border-border-subtle bg-gradient-to-b from-surface-card to-surface-subtle/80 p-6 md:p-8 shadow-sm backdrop-blur-sm ${className}`}
    >
      {/* Top Banner: Context and Timeframe Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border-subtle pb-5">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold uppercase tracking-wider">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>You Are Improving</span>
          </div>
          <span className="text-xs text-text-muted font-mono hidden md:inline">
            Rolling Evaluation · Institutional Learning Loop
          </span>
        </div>

        {/* Timeframe Toggle Buttons */}
        <div className="flex items-center bg-surface-base border border-border-subtle rounded-lg p-1 text-xs font-mono">
          {(["30D", "90D", "YTD", "ALL"] as const).map((tf) => (
            <button
              key={tf}
              type="button"
              onClick={() => setSelectedTimeframe(tf)}
              className={`px-3 py-1 rounded-md transition-all font-medium ${
                selectedTimeframe === tf
                  ? "bg-surface-raised text-text-primary shadow-xs font-semibold"
                  : "text-text-muted hover:text-text-secondary"
              }`}
            >
              {tf === "30D" ? "Last 30D" : tf === "90D" ? "Rolling 90D" : tf === "YTD" ? "YTD" : "All Time"}
            </button>
          ))}
        </div>
      </div>

      {/* Main Hero Metrics Grid */}
      <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        {/* Left: Big Decision Quality Score Ring & Badge */}
        <div className="lg:col-span-4 flex items-center gap-5 p-5 rounded-xl border border-border-subtle/80 bg-surface-card/60">
          <div className="relative flex items-center justify-center h-24 w-24 rounded-full border-4 border-emerald-500/20 bg-surface-base">
            <svg className="absolute inset-0 h-full w-full -rotate-90" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r="44"
                stroke="currentColor"
                strokeWidth="8"
                fill="transparent"
                className="text-emerald-500/10"
              />
              <circle
                cx="50"
                cy="50"
                r="44"
                stroke="currentColor"
                strokeWidth="8"
                fill="transparent"
                strokeDasharray={`${2 * Math.PI * 44}`}
                strokeDashoffset={`${2 * Math.PI * 44 * (1 - score / 100)}`}
                strokeLinecap="round"
                className="text-emerald-400 transition-all duration-1000 ease-out"
              />
            </svg>
            <div className="text-center">
              <span className="text-3xl font-bold font-mono text-text-primary tracking-tight">{score}</span>
              <span className="block text-[10px] font-mono text-text-muted">/100</span>
            </div>
          </div>

          <div className="space-y-1">
            <span className="text-xs font-mono uppercase tracking-wider text-text-muted">Decision Quality</span>
            <div className="flex items-center gap-1.5 text-emerald-400 font-mono text-sm font-bold">
              <span>▲ +{scoreDelta.toFixed(1)} pts</span>
              <span className="text-xs text-text-muted font-normal">vs prior 90d</span>
            </div>
            <p className="text-xs text-text-secondary font-medium">{decileRank}</p>
          </div>
        </div>

        {/* Right: Core KPI Trio */}
        <div className="lg:col-span-8 grid grid-cols-1 sm:grid-cols-3 gap-4">
          {/* KPI 1: Win / Success Rate */}
          <div className="p-4 rounded-xl border border-border-subtle bg-surface-card/40 hover:border-border-subtle/90 transition-colors">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase tracking-wider text-text-muted">Success Rate</span>
              <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">
                +{successRateDelta.toFixed(1)}%
              </span>
            </div>
            <div className="mt-2 text-2xl font-bold font-mono text-text-primary">
              {successRate.toFixed(1)}%
            </div>
            <p className="mt-1 text-[11px] text-text-muted">
              Target 1 reached or favorable excursion ≥50%
            </p>
          </div>

          {/* KPI 2: Actionability Rate (PAR) */}
          <div className="p-4 rounded-xl border border-border-subtle bg-surface-card/40 hover:border-border-subtle/90 transition-colors">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase tracking-wider text-text-muted">PAR Actionability</span>
              <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">
                Pass (≥{parTarget}%)
              </span>
            </div>
            <div className="mt-2 text-2xl font-bold font-mono text-text-primary">
              {parRate.toFixed(1)}%
            </div>
            <p className="mt-1 text-[11px] text-text-muted">
              Predictions formally audited & acted upon
            </p>
          </div>

          {/* KPI 3: Learning Velocity */}
          <div className="p-4 rounded-xl border border-border-subtle bg-surface-card/40 hover:border-border-subtle/90 transition-colors">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase tracking-wider text-text-muted">Learning Velocity</span>
              <span className="text-xs font-mono font-bold text-purple-400 bg-purple-500/10 px-2 py-0.5 rounded">
                Compounding
              </span>
            </div>
            <div className="mt-2 text-2xl font-bold font-mono text-purple-300">
              +{learningVelocity.toFixed(1)}%
            </div>
            <p className="mt-1 text-[11px] text-text-muted">
              Quarterly rate of error elimination
            </p>
          </div>
        </div>
      </div>

      {/* Bottom Summary Callout */}
      <div className="mt-6 pt-4 border-t border-border-subtle/60 flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs text-text-secondary">
        <p className="leading-relaxed">
          <strong className="text-text-primary">Executive Diagnosis:</strong> Your alpha edge in{" "}
          <span className="text-emerald-400 font-semibold font-mono">Institutional Accumulation</span> during expanding regimes is compounding.
          Stricter pre-trade validation eliminated <span className="text-emerald-400 font-semibold font-mono">65%</span> of chop traps.
        </p>
        <span className="font-mono text-[11px] text-text-muted whitespace-nowrap">
          2,184 Trades Analyzed · 0 Unresolved
        </span>
      </div>
    </div>
  );
}
