"use client";

import React, { useState } from "react";
import { PlaybookRule, PlaybookRuleCategory } from "../../types/personal-intelligence";

export interface PersonalPlaybookCardProps {
  confidenceScore?: number;
  excessReturn?: number;
  className?: string;
}

export default function PersonalPlaybookCard({
  confidenceScore = 91,
  excessReturn = 43.5,
  className = "",
}: PersonalPlaybookCardProps) {
  const [activeTab, setActiveTab] = useState<"EDGE" | "MISTAKES" | "RULES" | "BEST_SETUPS">("EDGE");

  const edgeRules: PlaybookRule[] = [
    {
      ruleId: "edge-1",
      category: "DO_MORE",
      title: "Institutional Flow Accumulation",
      explanation: "Dark-pool volume absorption >+2.0σ prior to price breakout provides structural support.",
      supportingOutcomes: 842,
      winRate: 72.0,
      averageReturn: 14.2,
      confidence: 0.91,
      status: "ACTIVE",
    },
    {
      ruleId: "edge-2",
      category: "DO_MORE",
      title: "Sector Rotation Alignment",
      explanation: "Cross-asset sector rotation confirming inflows into Semiconductor & Cloud leaders.",
      supportingOutcomes: 512,
      winRate: 69.4,
      averageReturn: 11.8,
      confidence: 0.88,
      status: "ACTIVE",
    },
    {
      ruleId: "edge-3",
      category: "DO_MORE",
      title: "Relative Strength Breakout",
      explanation: "Emerging from low-volatility bases at 52-week relative highs during market consolidation.",
      supportingOutcomes: 394,
      winRate: 66.1,
      averageReturn: 9.4,
      confidence: 0.84,
      status: "ACTIVE",
    },
  ];

  const mistakeRules: PlaybookRule[] = [
    {
      ruleId: "mistake-1",
      category: "STOP_DOING",
      title: "Gap-Fade Overextended Entries",
      explanation: "Buying opening gap-ups >3.0% into overhead resistance without a 15-minute consolidation base.",
      supportingOutcomes: 133,
      winRate: 28.5,
      averageReturn: -4.8,
      confidence: 0.89,
      status: "ACTIVE",
    },
    {
      ruleId: "mistake-2",
      category: "STOP_DOING",
      title: "Late Momentum Chasing",
      explanation: "Entering on Day 4+ of vertical thrust without volume confirmation or base formation.",
      supportingOutcomes: 122,
      winRate: 31.0,
      averageReturn: -5.1,
      confidence: 0.86,
      status: "ACTIVE",
    },
    {
      ruleId: "mistake-3",
      category: "STOP_DOING",
      title: "Regime Deterioration Blindness",
      explanation: "Holding or adding long equity exposure after VIX spikes >22 into DEFENSIVE regime.",
      supportingOutcomes: 146,
      winRate: 19.4,
      averageReturn: -6.2,
      confidence: 0.93,
      status: "ACTIVE",
    },
  ];

  const tradingRules = [
    { category: "DO_MORE", rule: "Scale size by +15% on Institutional Accumulation above +2.0σ", icon: "✅" },
    { category: "DO_MORE", rule: "Prioritize breakout setups with sector rotation confirmation", icon: "✅" },
    { category: "DO_MORE", rule: "Target relative strength percentile >85 in expanding regimes", icon: "✅" },
    { category: "STOP_DOING", rule: "Cease buying opening gap-ups >3.0% in first 15 minutes", icon: "❌" },
    { category: "STOP_DOING", rule: "Never average down after structural stop floor breach", icon: "❌" },
    { category: "STOP_DOING", rule: "Abort high-beta long additions when macro regime turns DEFENSIVE", icon: "❌" },
    { category: "CALIBRATE", rule: "Automate hard stop floor at -3.5% instead of discretionary close", icon: "⚠️" },
    { category: "CALIBRATE", rule: "Reduce position sizing by 15% when setup confidence <75%", icon: "⚠️" },
  ];

  return (
    <div
      role="region"
      aria-label="Personal Playbook Card"
      className={`rounded-2xl border border-border-subtle bg-surface-card p-6 md:p-8 shadow-sm ${className}`}
    >
      {/* Header with Title & Confidence Badge */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border-subtle pb-5">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="h-2.5 w-2.5 rounded-full bg-purple-400" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-text-primary">
              Personal Decision Playbook
            </h3>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-purple-500/15 text-purple-300 border border-purple-500/30">
              CONFIDENCE {confidenceScore}%
            </span>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Statistically validated strengths, repeatable rules, and systematic avoidance patterns.
          </p>
        </div>

        {/* Section Navigation Tabs */}
        <div className="flex items-center bg-surface-base border border-border-subtle rounded-lg p-1 text-xs font-mono overflow-x-auto">
          {(["EDGE", "MISTAKES", "RULES", "BEST_SETUPS"] as const).map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-1 rounded-md transition-all font-medium whitespace-nowrap ${
                activeTab === tab
                  ? "bg-surface-raised text-text-primary shadow-xs font-semibold"
                  : "text-text-muted hover:text-text-secondary"
              }`}
            >
              {tab === "EDGE" ? "My Edge" : tab === "MISTAKES" ? "Repeating Mistakes" : tab === "RULES" ? "Trading Rules" : "Best Setups"}
            </button>
          ))}
        </div>
      </div>

      {/* Tab 1: My Edge */}
      {activeTab === "EDGE" && (
        <div className="mt-6 space-y-4 animate-fadeIn">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {edgeRules.map((rule, idx) => (
              <div
                key={rule.ruleId}
                className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4 space-y-2 hover:border-emerald-500/40 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-emerald-400">
                    #{idx + 1} ALPHA SOURCE
                  </span>
                  <span className="font-mono text-xs font-bold text-emerald-300">
                    {rule.winRate.toFixed(1)}% Win Rate
                  </span>
                </div>
                <h4 className="text-sm font-mono font-bold text-text-primary">
                  {rule.title}
                </h4>
                <p className="text-xs text-text-secondary leading-relaxed">
                  {rule.explanation}
                </p>
                <div className="pt-2 border-t border-emerald-500/15 flex justify-between text-[11px] font-mono text-text-muted">
                  <span>{rule.supportingOutcomes} Trades</span>
                  <span className="text-emerald-400 font-semibold">+{rule.averageReturn.toFixed(1)}% Avg Gain</span>
                </div>
              </div>
            ))}
          </div>

          <div className="p-4 rounded-xl border border-border-subtle bg-surface-subtle/60 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs font-mono">
            <span className="text-text-muted">Total Cumulative Excess Return Generated:</span>
            <span className="text-sm font-bold text-emerald-400 bg-emerald-500/10 px-3 py-1 rounded border border-emerald-500/20">
              +{excessReturn.toFixed(1)}% vs Benchmark
            </span>
          </div>
        </div>
      )}

      {/* Tab 2: Repeating Mistakes */}
      {activeTab === "MISTAKES" && (
        <div className="mt-6 space-y-4 animate-fadeIn">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {mistakeRules.map((rule, idx) => (
              <div
                key={rule.ruleId}
                className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-4 space-y-2 hover:border-rose-500/40 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-rose-400">
                    #{idx + 1} LOSS TRAP
                  </span>
                  <span className="font-mono text-xs font-bold text-rose-300">
                    {(100 - rule.winRate).toFixed(1)}% Failure Rate
                  </span>
                </div>
                <h4 className="text-sm font-mono font-bold text-text-primary">
                  {rule.title}
                </h4>
                <p className="text-xs text-text-secondary leading-relaxed">
                  {rule.explanation}
                </p>
                <div className="pt-2 border-t border-rose-500/15 flex justify-between text-[11px] font-mono text-text-muted">
                  <span>{rule.supportingOutcomes} Stopouts</span>
                  <span className="text-rose-400 font-semibold">{rule.averageReturn.toFixed(1)}% Avg Loss</span>
                </div>
              </div>
            ))}
          </div>

          <div className="p-4 rounded-xl border border-border-subtle bg-surface-subtle/60 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs font-mono">
            <span className="text-text-muted">Preservation Opportunity:</span>
            <span className="text-sm font-bold text-rose-400 bg-rose-500/10 px-3 py-1 rounded border border-rose-500/20">
              Eliminating Top 2 Mistakes Preserves +7.6% Annual Capital
            </span>
          </div>
        </div>
      )}

      {/* Tab 3: Personal Trading Rules */}
      {activeTab === "RULES" && (
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4 animate-fadeIn">
          {/* DO MORE */}
          <div className="rounded-xl border border-emerald-500/20 bg-surface-card p-4 space-y-3">
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
              <span>✅ DO MORE</span>
            </span>
            <ul className="space-y-2.5 text-xs text-text-secondary">
              {tradingRules.filter(r => r.category === "DO_MORE").map((r, i) => (
                <li key={i} className="p-2.5 rounded bg-surface-subtle border border-border-subtle flex items-start gap-2">
                  <span className="shrink-0 text-emerald-400 font-bold">✓</span>
                  <span className="leading-relaxed">{r.rule}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* STOP DOING */}
          <div className="rounded-xl border border-rose-500/20 bg-surface-card p-4 space-y-3">
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-rose-400 flex items-center gap-1.5">
              <span>❌ STOP DOING</span>
            </span>
            <ul className="space-y-2.5 text-xs text-text-secondary">
              {tradingRules.filter(r => r.category === "STOP_DOING").map((r, i) => (
                <li key={i} className="p-2.5 rounded bg-surface-subtle border border-border-subtle flex items-start gap-2">
                  <span className="shrink-0 text-rose-400 font-bold">✗</span>
                  <span className="leading-relaxed">{r.rule}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* CALIBRATE */}
          <div className="rounded-xl border border-purple-500/20 bg-surface-card p-4 space-y-3">
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-purple-400 flex items-center gap-1.5">
              <span>⚠️ CALIBRATE</span>
            </span>
            <ul className="space-y-2.5 text-xs text-text-secondary">
              {tradingRules.filter(r => r.category === "CALIBRATE").map((r, i) => (
                <li key={i} className="p-2.5 rounded bg-surface-subtle border border-border-subtle flex items-start gap-2">
                  <span className="shrink-0 text-purple-300 font-bold">!</span>
                  <span className="leading-relaxed">{r.rule}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Tab 4: Best Historical Setups */}
      {activeTab === "BEST_SETUPS" && (
        <div className="mt-6 space-y-4 animate-fadeIn">
          <div className="p-5 rounded-xl border border-border-subtle bg-surface-subtle/40 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border-subtle/60 pb-3">
              <div>
                <span className="text-xs font-mono uppercase tracking-wider text-purple-300 font-bold">
                  Top Performing Institutional Setup Pattern
                </span>
                <h4 className="text-base font-mono font-bold text-text-primary mt-0.5">
                  Institutional Flow Absorption &amp; Expansion Breakout
                </h4>
              </div>
              <span className="font-mono text-xs text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded border border-emerald-500/20">
                Sharpe Ratio: 2.3
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
              <div className="p-3 bg-surface-card rounded-lg border border-border-subtle">
                <span className="text-text-muted block text-[10px]">HISTORICAL TRADES</span>
                <span className="text-lg font-bold text-text-primary">212 Trades</span>
              </div>
              <div className="p-3 bg-surface-card rounded-lg border border-border-subtle">
                <span className="text-text-muted block text-[10px]">SUCCESS RATE</span>
                <span className="text-lg font-bold text-emerald-400">72.0%</span>
              </div>
              <div className="p-3 bg-surface-card rounded-lg border border-border-subtle">
                <span className="text-text-muted block text-[10px]">AVERAGE GAIN</span>
                <span className="text-lg font-bold text-emerald-400">+14.2%</span>
              </div>
              <div className="p-3 bg-surface-card rounded-lg border border-border-subtle">
                <span className="text-text-muted block text-[10px]">MAX FAVORABLE EXCURSION</span>
                <span className="text-lg font-bold text-purple-300">+28.4%</span>
              </div>
            </div>

            <div className="pt-2 flex items-center gap-3 text-xs font-mono text-text-muted flex-wrap">
              <span>Most Similar Historical Outliers:</span>
              <span className="px-2 py-0.5 rounded bg-surface-raised text-text-primary border border-border-subtle font-bold">
                NVDA (+18.7%)
              </span>
              <span className="px-2 py-0.5 rounded bg-surface-raised text-text-primary border border-border-subtle font-bold">
                CPRX (+14.2%)
              </span>
              <span className="px-2 py-0.5 rounded bg-surface-raised text-text-primary border border-border-subtle font-bold">
                AMD (+12.1%)
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
