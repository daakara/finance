"use client";

import React, { useState } from "react";
import Link from "next/link";
import { QuantitativeInsight } from "../../types/insight";
import FinancialDisclaimer from "../FinancialDisclaimer";
import { evaluateLevelRelation } from "../../lib/reclaimSemantics";
import { deriveUnmetConditions } from "../../lib/decisionHierarchyUtils";

interface GuidedTerminalViewProps {
  insight: QuantitativeInsight;
  onOpenSizer: () => void;
  onOpenWhy: () => void;
  chartSlot?: React.ReactNode;
  planSlot?: React.ReactNode;
}

export default function GuidedTerminalView({
  insight,
  onOpenSizer,
  onOpenWhy,
  chartSlot,
  planSlot,
}: GuidedTerminalViewProps) {
  const [activeStep, setActiveStep] = useState<number | null>(null);
  const unmetConditions = deriveUnmetConditions(insight);
  const isActionable = Boolean(insight.terminalState.isActionable);

  const smaRel = evaluateLevelRelation(
    insight.price,
    insight.standard.keyLevels.sma50,
    "50D SMA",
    insight.symbol
  );

  const steps = [
    { title: "1. What's Happening?", text: `${insight.symbol} is trading at $${insight.price.toFixed(2)}, ${insight.changePct >= 0 ? "+" : ""}${insight.changePct.toFixed(2)}% today. ${insight.human.assessmentDescription}` },
    { title: "2. What's the Setup?", text: `ARX identifies the current structure as ${insight.standard.setupSummary}. ${insight.advanced.relativeStrengthScore !== undefined ? `Relative strength score is ${insight.advanced.relativeStrengthScore}/100.` : "Relative strength score is unverified for this security."}` },
    { title: "3. Why does ARX like/caution it?", text: insight.human.reclaimMilestone },
    { title: "4. What could go wrong?", text: `Every thesis has downside risk. If price breaks below $${insight.standard.keyLevels.stopLoss.toFixed(2)}, the setup is invalidated.` },
    { title: "5. How could I trade it?", text: `Plan: ${
      smaRel.status === "BELOW"
        ? `Watch for reclaim of $${(insight.standard.keyLevels.sma50 as number).toFixed(2)} (50D SMA).`
        : smaRel.status === "AT_LEVEL"
        ? `Testing 50D SMA at $${(insight.standard.keyLevels.sma50 as number).toFixed(2)}. Watch for decisive volume expansion.`
        : smaRel.status === "ABOVE"
        ? `Holding constructively above 50D SMA ($${(insight.standard.keyLevels.sma50 as number).toFixed(2)}). Watch for base confirmation.`
        : "Watch key technical levels."
    } Target 1 is ${insight.standard.keyLevels.target1 !== undefined ? `$${insight.standard.keyLevels.target1.toFixed(2)} (+${insight.standard.keyLevels.target1Pct}%)` : "N/A (< 50 sessions)"}.` },
    { title: "6. What should I monitor?", text: `Volume surges, 50-day moving average crossovers, and broader market regime stability.` },
  ];

  return (
    <div className="space-y-4 font-sans text-slate-100 animate-fade-in">
      {/* 1. DECISION VERDICT & 2. DECISION REASON */}
      <div
        data-testid="decision-verdict"
        className="bg-[#0b101b] border border-[#1d293d] rounded-2xl p-4 sm:p-5 shadow-xl space-y-3"
      >
        <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[#182335] pb-3">
          <div>
            <span className="text-xs text-slate-400 font-mono font-bold uppercase tracking-wider block mb-1">
              ARX Novice Guidance & Assessment
            </span>
            <h2 className="text-lg sm:text-2xl font-black text-white tracking-tight">
              {insight.human.assessmentHeadline}
            </h2>
            <div className="mt-1 flex items-center gap-2">
              <span className="text-xs font-mono text-cyan-300 font-bold">
                Canonical Verdict: {insight.verdictLabel}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`px-3 py-1 rounded-md text-xs font-mono font-bold border ${
                isActionable
                  ? "bg-emerald-950/80 text-emerald-300 border-emerald-700/80"
                  : "bg-amber-950/80 text-amber-300 border-amber-700/80"
              }`}
            >
              {isActionable ? "ACTIONABLE" : "WAIT FOR TRIGGER"}
            </span>
            <span className="px-3 py-1 rounded-md text-xs font-mono bg-[#162030] text-slate-300 border border-[#243044]">
              {insight.terminalState.uiStateLabel}
            </span>
          </div>
        </div>

        {/* 2. CANONICAL REASON */}
        <div
          data-testid="decision-reason"
          className="text-xs sm:text-sm text-slate-200 font-sans leading-relaxed pt-1"
        >
          <span className="text-xs text-slate-400 font-bold block uppercase mb-1">
            Plain English Explanation:
          </span>
          <p>{insight.human.assessmentDescription}</p>
        </div>
      </div>

      {/* 3. WHAT NEEDS TO CHANGE */}
      <div
        data-testid="unmet-condition"
        className="bg-[#080e18] border border-cyan-900/50 rounded-2xl p-4 sm:p-5 shadow-lg space-y-3"
      >
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#182335] pb-2">
          <h3 className="text-xs sm:text-sm font-bold text-cyan-300 uppercase font-mono tracking-wide flex items-center gap-2">
            <span>🎯</span>
            <span>What Needs to Change Before Buying</span>
          </h3>
          <span className="text-xs text-slate-400 font-mono">
            {isActionable ? "Requirements Satisfied" : "Awaiting Improvements"}
          </span>
        </div>

        <p className="text-xs text-slate-300 font-sans leading-relaxed">
          {insight.human.reclaimMilestone}
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 text-xs pt-1">
          {unmetConditions.map((cond) => (
            <div
              key={cond.id}
              className="p-3 rounded-xl bg-[#060b13] border border-[#1b2639] space-y-1"
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-200 text-xs">{cond.title}</span>
                <span
                  className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${
                    cond.status === "MET"
                      ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                      : cond.status === "UNMET"
                      ? "bg-rose-950 text-rose-400 border border-rose-800"
                      : "bg-amber-950 text-amber-400 border border-amber-800"
                  }`}
                >
                  {cond.status}
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed font-sans">{cond.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 4. PRICE / CHART CONTEXT */}
      {chartSlot && (
        <div data-testid="market-workspace-chart" className="space-y-2">
          {chartSlot}
        </div>
      )}

      {/* 5. CONDITIONAL TRADE PLAN */}
      {planSlot && (
        <div className="space-y-2">
          {planSlot}
        </div>
      )}

      {/* 6. SUPPORTING EVIDENCE */}
      <div
        data-testid="supporting-evidence"
        className="bg-[#0b101b] border border-[#1d293d] rounded-2xl p-4 sm:p-5 shadow-xl space-y-4"
      >
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#182335] pb-3">
          <div>
            <span className="text-xs text-slate-400 font-mono font-bold uppercase tracking-wider block">
              Supporting Evidence
            </span>
            <h3 className="text-sm sm:text-base font-bold text-white tracking-tight mt-0.5">
              Why ARX Thinks This & Level Analysis
            </h3>
          </div>

          {/* Subordinated Score Badge */}
          <div
            onClick={onOpenWhy}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onOpenWhy();
              }
            }}
            className={`flex items-center gap-3 px-3.5 py-2 bg-[#06090f] border rounded-xl cursor-pointer transition-all shrink-0 min-h-[36px] sm:min-h-[44px] text-xs ${
              insight.terminalState.overallEligibility !== "ELIGIBLE"
                ? "border-slate-700 hover:border-slate-500"
                : "border-[#24334b] hover:border-cyan-500/60"
            }`}
          >
            <div>
              <span className="text-xs text-slate-400 font-mono block">Setup Score</span>
              <span className="text-xs text-cyan-400 font-mono">Explain Score →</span>
            </div>
            <div className="text-right">
              <span
                className={`text-xl font-black font-mono ${
                  insight.terminalState.overallEligibility !== "ELIGIBLE"
                    ? "text-slate-500"
                    : insight.setupScore >= 75 ? "text-emerald-400" : insight.setupScore >= 55 ? "text-amber-400" : "text-rose-400"
                }`}
              >
                {insight.setupScore}
                {insight.terminalState.overallEligibility !== "ELIGIBLE" ? "*" : ""}
              </span>
              {insight.terminalState.overallEligibility !== "ELIGIBLE" && (
                <span className="text-xs text-amber-500/80 font-mono block">Partial</span>
              )}
            </div>
          </div>
        </div>

        {/* 4-Pills Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 text-xs">
          {(insight.human.whyPills || []).map((pill, idx) => (
            <div
              key={idx}
              className="bg-[#070b13] p-3 rounded-xl border border-[#1b2639] space-y-1.5"
            >
              <div className="flex items-center justify-between">
                <span className="text-slate-400 text-xs font-bold">{pill.category}</span>
                <span
                  className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${
                    pill.sentiment === "positive"
                      ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                      : pill.sentiment === "negative"
                      ? "bg-rose-950 text-rose-400 border border-rose-800"
                      : "bg-amber-950 text-amber-400 border border-amber-800"
                  }`}
                >
                  {pill.status}
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed font-sans">
                {pill.description}
              </p>
            </div>
          ))}
        </div>

        {/* Watch These Levels */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 font-mono text-xs">
          <div className="bg-[#080d16] p-3 rounded-xl border border-[#182335]">
            <span className="text-xs text-slate-500 uppercase block font-semibold">Watch Level</span>
            <span className="text-sm font-black text-amber-300 mt-0.5 block">{insight.human.watchLevels.watchZone}</span>
            <span className="text-xs text-slate-400 mt-0.5 block">Needs base rebound</span>
          </div>

          <div className="bg-[#080d16] p-3 rounded-xl border border-[#182335]">
            <span className="text-xs text-slate-500 uppercase block font-semibold">Key Level (50D SMA)</span>
            <span className="text-sm font-black text-cyan-300 mt-0.5 block">{insight.human.watchLevels.keyLevel}</span>
            <span className="text-xs text-slate-400 mt-0.5 block">{smaRel.uiBadgeLabel}</span>
          </div>

          <div className="bg-[#080d16] p-3 rounded-xl border border-rose-950/60">
            <span className="text-xs text-rose-400 uppercase block font-semibold">Risk Level (Stop)</span>
            <span className="text-sm font-black text-rose-300 mt-0.5 block">{insight.human.watchLevels.riskStop}</span>
            <span className="text-xs text-rose-500/80 mt-0.5 block">Protect if broken</span>
          </div>
        </div>

        {/* Action Guidance */}
        <div className="pt-2">
          {insight.terminalState.posture === "ACQUIRE" ? (
            <button
              onClick={onOpenSizer}
              className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 text-slate-950 rounded-xl text-xs font-mono font-black shadow-md transition-all active:scale-95 cursor-pointer min-h-[44px]"
            >
              ⚖️ Size Position (Buy Zone)
            </button>
          ) : insight.terminalState.posture === "EXIT_REVIEW" ? (
            <button
              onClick={onOpenWhy}
              className="w-full py-2.5 bg-rose-600 hover:bg-rose-500 text-white rounded-xl text-xs font-mono font-black shadow-md transition-all active:scale-95 cursor-pointer min-h-[44px]"
            >
              🚨 Review Invalidation
            </button>
          ) : insight.terminalState.posture === "RESEARCH" ? (
            <button
              onClick={onOpenWhy}
              className="w-full py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-xl text-xs font-mono font-black shadow-md transition-all active:scale-95 cursor-pointer min-h-[44px]"
            >
              📋 Open Research
            </button>
          ) : insight.terminalState.posture === "AVOID" ? (
            <Link
              href="/radar"
              className="w-full py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-mono font-black shadow-md transition-all active:scale-95 cursor-pointer min-h-[44px] flex items-center justify-center"
            >
              🔎 Find Setups
            </Link>
          ) : (
            <button
              onClick={onOpenWhy}
              className="w-full py-2.5 bg-cyan-600 hover:bg-cyan-500 text-slate-950 rounded-xl text-xs font-mono font-black shadow-md transition-all active:scale-95 cursor-pointer min-h-[44px]"
            >
              ⏳ Wait for Trigger
            </button>
          )}
        </div>
      </div>

      {/* 6-Step Novice Walkthrough */}
      <div className="bg-[#070b13] border border-[#1a2538] rounded-xl p-3.5 space-y-2.5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono font-bold text-slate-300 flex items-center gap-1.5">
            <span>🧭</span> Step-by-Step Stock Analysis Walkthrough
          </span>
          <span className="text-xs text-slate-500 font-mono">Click any step to inspect</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-1.5">
          {steps.map((step, idx) => (
            <button
              key={idx}
              id={`walkthrough-step-btn-${idx}`}
              onClick={() => setActiveStep(activeStep === idx ? null : idx)}
              aria-expanded={activeStep === idx}
              aria-controls={`walkthrough-step-panel-${idx}`}
              className={`p-2.5 rounded-lg text-left text-xs font-mono border transition-all cursor-pointer min-h-[36px] ${
                activeStep === idx
                  ? "bg-cyan-950/80 border-cyan-500 text-cyan-200"
                  : "bg-[#0b101b] border-[#182335] text-slate-400 hover:text-slate-200"
              }`}
            >
              <span className="font-bold block truncate">{step.title}</span>
            </button>
          ))}
        </div>

        {activeStep !== null && (
          <div
            id={`walkthrough-step-panel-${activeStep}`}
            role="region"
            aria-labelledby={`walkthrough-step-btn-${activeStep}`}
            className="p-3.5 bg-[#0d1422] border border-cyan-800/40 rounded-lg text-xs font-sans text-slate-200 animate-fade-in leading-relaxed"
          >
            <strong className="text-cyan-400 font-mono block mb-1 text-xs">{steps[activeStep].title}</strong>
            <span className="text-xs">{steps[activeStep].text}</span>
          </div>
        )}
      </div>

      <FinancialDisclaimer variant="compact" />
    </div>
  );
}
