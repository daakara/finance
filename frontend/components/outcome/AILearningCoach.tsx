"use client";

import React, { useState } from "react";

export interface CoachingRule {
  id: string;
  category: "SCALE" | "STOP" | "CALIBRATE";
  title: string;
  prescription: string;
  expectedImpact: string;
  evidenceStat: string;
}

export interface AILearningCoachProps {
  confidenceScore?: number;
  tradesAnalyzed?: number;
  className?: string;
}

export default function AILearningCoach({
  confidenceScore = 91,
  tradesAnalyzed = 2184,
  className = "",
}: AILearningCoachProps) {
  const [activeTab, setActiveTab] = useState<"PRESCRIPTIONS" | "EVIDENCE" | "CHANGES" | "SIMULATION">("PRESCRIPTIONS");

  const rules: CoachingRule[] = [
    {
      id: "rule-1",
      category: "SCALE",
      title: "Scale Institutional Accumulation by +15%",
      prescription:
        "When market regime is EXPANSION and dark pool accumulation exceeds +2.0σ, scale position size from 1.0x to 1.15x. Your historical win rate is 72.0% with +14.2% average favorable excursion.",
      expectedImpact: "+3.2% Projected Annual Return",
      evidenceStat: "842 Trades · 72.0% Win Rate",
    },
    {
      id: "rule-2",
      category: "STOP",
      title: "Eliminate Opening Gap Chasing (>3.0%)",
      prescription:
        "Do not enter stocks gapping up >3.0% during the first 15 minutes of trading. Require a 15-minute consolidation base above the VWAP before triggering buy orders. Gap-chasing caused 38.2% of all stop-loss breaches.",
      expectedImpact: "-4.8% Drawdown Reduction",
      evidenceStat: "133 Trades · -4.8% Avg Loss",
    },
    {
      id: "rule-3",
      category: "CALIBRATE",
      title: "Automate Hard Stop at -3.5% on High-Beta",
      prescription:
        "Replace discretionary end-of-day stop checks with an automated bracket stop floor at -3.5% for high-beta setups. Discretionary exit delays accounted for $142,000 in excess slippage during volatile reversals.",
      expectedImpact: "+$142K Slippage Recovered",
      evidenceStat: "73 Stops · -6.2% Discretionary vs -3.5% Rule",
    },
  ];

  return (
    <div
      role="region"
      aria-label="AI Learning Coach"
      className={`rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-950/20 via-surface-card to-surface-card p-6 md:p-8 shadow-sm backdrop-blur-sm ${className}`}
    >
      {/* Header: Coach Identity & Confidence Meter */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-purple-500/20 pb-5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-600/20 border border-purple-500/40 text-purple-300">
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold uppercase tracking-wider text-text-primary">
                ARX AI Learning Coach
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">
                ACTIVE
              </span>
            </div>
            <p className="text-xs text-text-muted mt-0.5">
              Behavioral Decision Coach · Based on {tradesAnalyzed.toLocaleString()} resolved institutional outcomes
            </p>
          </div>
        </div>

        {/* Confidence Meter */}
        <div className="flex items-center gap-3 bg-surface-base/80 border border-purple-500/20 rounded-xl px-4 py-2">
          <div className="text-right">
            <span className="block text-[10px] font-mono uppercase text-text-muted">Coach Confidence</span>
            <span className="text-sm font-bold font-mono text-purple-300">{confidenceScore}% Highly Calibrated</span>
          </div>
          <div className="h-8 w-1.5 rounded-full bg-surface-raised overflow-hidden">
            <div className="h-full bg-gradient-to-t from-purple-600 to-purple-400 rounded-full" style={{ height: `${confidenceScore}%` }} />
          </div>
        </div>
      </div>

      {/* Primary Coach Diagnosis */}
      <div className="mt-5 rounded-xl border border-purple-500/20 bg-purple-500/5 p-4 md:p-5">
        <div className="flex items-center gap-2 text-xs font-mono font-semibold text-purple-400 uppercase tracking-wider">
          <span>🧠 Executive Behavioral Diagnosis</span>
        </div>
        <p className="mt-2 text-sm md:text-base text-text-primary leading-relaxed font-medium">
          &quot;Your decision quality improved by <span className="text-emerald-400 font-bold font-mono">+8.2%</span> this quarter. Macro regime discipline is now your primary edge. To reach the top decile, eliminate opening gap-chase traps and scale allocation into high-conviction accumulation.&quot;
        </p>
      </div>

      {/* Interactive Navigation Tabs */}
      <div className="mt-6 flex items-center gap-2 border-b border-border-subtle pb-2 overflow-x-auto text-xs font-mono">
        <button
          type="button"
          onClick={() => setActiveTab("PRESCRIPTIONS")}
          className={`px-3.5 py-1.5 rounded-lg font-medium transition-all ${
            activeTab === "PRESCRIPTIONS"
              ? "bg-purple-600 text-white font-semibold shadow-xs"
              : "text-text-muted hover:text-text-secondary hover:bg-surface-subtle"
          }`}
        >
          Actionable Prescriptions (3)
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("EVIDENCE")}
          className={`px-3.5 py-1.5 rounded-lg font-medium transition-all ${
            activeTab === "EVIDENCE"
              ? "bg-purple-600 text-white font-semibold shadow-xs"
              : "text-text-muted hover:text-text-secondary hover:bg-surface-subtle"
          }`}
        >
          Show Evidence
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("CHANGES")}
          className={`px-3.5 py-1.5 rounded-lg font-medium transition-all ${
            activeTab === "CHANGES"
              ? "bg-purple-600 text-white font-semibold shadow-xs"
              : "text-text-muted hover:text-text-secondary hover:bg-surface-subtle"
          }`}
        >
          What Changed?
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("SIMULATION")}
          className={`px-3.5 py-1.5 rounded-lg font-medium transition-all ${
            activeTab === "SIMULATION"
              ? "bg-purple-600 text-white font-semibold shadow-xs"
              : "text-text-muted hover:text-text-secondary hover:bg-surface-subtle"
          }`}
        >
          Simulate Impact
        </button>
      </div>

      {/* Tab 1: Prescriptions */}
      {activeTab === "PRESCRIPTIONS" && (
        <div className="mt-5 space-y-3.5">
          {rules.map((rule, idx) => (
            <div
              key={rule.id}
              className="rounded-xl border border-border-subtle/80 bg-surface-card/60 p-4 hover:border-purple-500/40 transition-colors"
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="flex items-center gap-2.5">
                  <span className="flex h-6 w-6 items-center justify-center rounded-full bg-purple-500/20 text-purple-300 font-mono text-xs font-bold">
                    {idx + 1}
                  </span>
                  <span className="font-mono text-sm font-bold text-text-primary">
                    {rule.title}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${
                      rule.category === "SCALE"
                        ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                        : rule.category === "STOP"
                        ? "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                        : "bg-purple-500/10 text-purple-300 border border-purple-500/20"
                    }`}
                  >
                    {rule.category === "SCALE" ? "DO MORE" : rule.category === "STOP" ? "STOP DOING" : "CALIBRATE"}
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-emerald-400 font-semibold bg-emerald-500/5 px-2 py-0.5 rounded border border-emerald-500/20">
                    {rule.expectedImpact}
                  </span>
                </div>
              </div>

              <p className="mt-2.5 text-xs text-text-secondary leading-relaxed pl-8">
                {rule.prescription}
              </p>

              <div className="mt-3 pl-8 flex items-center gap-4 text-[11px] font-mono text-text-muted">
                <span>Empirical Evidence: {rule.evidenceStat}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Tab 2: Show Evidence */}
      {activeTab === "EVIDENCE" && (
        <div className="mt-5 space-y-4 animate-fadeIn">
          <div className="p-4 rounded-xl border border-border-subtle bg-surface-subtle/50 space-y-3">
            <h4 className="text-xs font-mono uppercase tracking-wider text-text-primary font-bold">
              Empirical Backing & Attributed Trade Sample
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
              <div className="p-3 bg-surface-card rounded-lg border border-border-subtle">
                <span className="text-text-muted block text-[10px]">TOTAL OUTCOMES SAMPLED</span>
                <span className="text-lg font-bold text-text-primary">2,184 Trades</span>
                <span className="text-[10px] text-emerald-400 block mt-1">100% Immutable Provenance</span>
              </div>
              <div className="p-3 bg-surface-card rounded-lg border border-border-subtle">
                <span className="text-text-muted block text-[10px]">CAUSAL ATTRIBUTION CONFIDENCE</span>
                <span className="text-lg font-bold text-purple-300">94.6%</span>
                <span className="text-[10px] text-text-muted block mt-1">Deterministic Model Matching</span>
              </div>
              <div className="p-3 bg-surface-card rounded-lg border border-border-subtle">
                <span className="text-text-muted block text-[10px]">FALSE POSITIVE LEAKAGE</span>
                <span className="text-lg font-bold text-emerald-400">0.0%</span>
                <span className="text-[10px] text-emerald-400 block mt-1">Materiality Gated (L0–L4)</span>
              </div>
            </div>
            <p className="text-xs text-text-muted leading-relaxed">
              Every coaching recommendation is synthesized by cross-referencing resolved price excursions with the immutable Decision Journal ledger. Recommendations require at least 50 historical observations with statistical significance (p &lt; 0.01).
            </p>
          </div>
        </div>
      )}

      {/* Tab 3: What Changed? */}
      {activeTab === "CHANGES" && (
        <div className="mt-5 space-y-3 animate-fadeIn">
          <div className="p-4 rounded-xl border border-border-subtle bg-surface-subtle/50 space-y-2">
            <h4 className="text-xs font-mono uppercase tracking-wider text-text-primary font-bold">
              Quarter-Over-Quarter Behavioral Shifts
            </h4>
            <ul className="space-y-2 text-xs text-text-secondary">
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 font-bold">✓</span>
                <span><strong>Gap Chasing Frequency:</strong> Decreased by <span className="text-emerald-400 font-mono font-semibold">-18.4%</span> since implementing 15-minute wait recommendation.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 font-bold">✓</span>
                <span><strong>Accumulation Sizing:</strong> Increased allocation by <span className="text-emerald-400 font-mono font-semibold">+12.0%</span> on dark-pool confirmed setups, capturing +$84K in extra alpha.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-amber-400 font-bold">!</span>
                <span><strong>Discretionary Stops:</strong> Still lingering on 28% of trades. Transition to automated bracket stops is in progress.</span>
              </li>
            </ul>
          </div>
        </div>
      )}

      {/* Tab 4: Simulate Impact */}
      {activeTab === "SIMULATION" && (
        <div className="mt-5 space-y-4 animate-fadeIn">
          <div className="p-4 rounded-xl border border-purple-500/30 bg-purple-950/20 space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-mono uppercase tracking-wider text-purple-300 font-bold">
                Forward Impact Simulation (Next 90 Days)
              </h4>
              <span className="text-[10px] font-mono text-text-muted">Monte Carlo n=10,000</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
              <div className="p-3 bg-surface-card rounded-lg border border-border-subtle">
                <span className="text-text-muted block text-[10px]">PROJECTED WIN RATE</span>
                <div className="flex items-baseline gap-2 mt-1">
                  <span className="text-xl font-bold text-text-primary">67.4%</span>
                  <span className="text-text-muted">→</span>
                  <span className="text-xl font-bold text-emerald-400">72.1%</span>
                  <span className="text-xs text-emerald-400 font-semibold">(+4.7%)</span>
                </div>
              </div>
              <div className="p-3 bg-surface-card rounded-lg border border-border-subtle">
                <span className="text-text-muted block text-[10px]">MAX DRAWDOWN</span>
                <div className="flex items-baseline gap-2 mt-1">
                  <span className="text-xl font-bold text-text-primary">-11.2%</span>
                  <span className="text-text-muted">→</span>
                  <span className="text-xl font-bold text-emerald-400">-7.8%</span>
                  <span className="text-xs text-emerald-400 font-semibold">(-3.4% risk reduction)</span>
                </div>
              </div>
            </div>
            <p className="text-[11px] text-text-muted leading-relaxed">
              Simulation applies the 3 coaching rules over your trailing 90-day distribution of trades. Eliminating gap stops and adding accumulation weight compounds portfolio Sharpe from 1.84 to 2.31.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
