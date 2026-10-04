"use client";

import React from "react";
import Link from "next/link";
import { QuantitativeInsight } from "../../types/insight";
import FinancialDisclaimer from "../FinancialDisclaimer";
import { deriveUnmetConditions } from "../../lib/decisionHierarchyUtils";

interface StandardTerminalViewProps {
  insight: QuantitativeInsight;
  onOpenSizer: () => void;
  onOpenWhy: () => void;
  chartSlot?: React.ReactNode;
  planSlot?: React.ReactNode;
}

export default function StandardTerminalView({
  insight,
  onOpenSizer,
  onOpenWhy,
  chartSlot,
  planSlot,
}: StandardTerminalViewProps) {
  const kl = insight.standard.keyLevels;
  const unmetConditions = deriveUnmetConditions(insight);
  const isActionable = Boolean(insight.terminalState.isActionable);

  return (
    <div className="space-y-4 font-sans text-slate-100 animate-fade-in">
      {/* 1. DECISION VERDICT (Dominant Visual Weight) & 2. DECISION REASON */}
      <div
        data-testid="decision-verdict"
        className="bg-[#0b101b] border border-[#1d293d] rounded-2xl p-4 sm:p-5 shadow-xl space-y-3"
      >
        <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[#182335] pb-3">
          <div>
            <span className="text-xs text-slate-400 font-mono font-bold uppercase tracking-wider block">
              ARX Analytical Verdict
            </span>
            <h2 className="text-lg sm:text-2xl font-black text-white tracking-tight mt-0.5">
              {insight.verdictLabel}
            </h2>
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

        {/* 2. CANONICAL REASON (Directly Below Verdict) */}
        <div
          data-testid="decision-reason"
          className="text-xs sm:text-sm text-slate-200 font-sans leading-relaxed pt-1"
        >
          <span className="text-xs text-slate-400 font-bold block uppercase mb-1">
            Reason / Bottom Line:
          </span>
          <p>{insight.standard.bottomLine}</p>
        </div>
      </div>

      {/* 3. WHAT NEEDS TO CHANGE (Unmet Conditions & Preconditions) */}
      <div
        data-testid="unmet-condition"
        className="bg-[#080e18] border border-cyan-900/50 rounded-2xl p-4 sm:p-5 shadow-lg space-y-3"
      >
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#182335] pb-2">
          <h3 className="text-xs sm:text-sm font-bold text-cyan-300 uppercase font-mono tracking-wide flex items-center gap-2">
            <span>🎯</span>
            <span>What Needs to Change (Execution Preconditions)</span>
          </h3>
          <span className="text-xs text-slate-400 font-mono">
            {isActionable ? "Preconditions Cleared" : "Awaiting Confirmation"}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 text-xs">
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

      {/* 6. SUPPORTING EVIDENCE (Score Subordinated & Confluence Breakdown) */}
      <div
        data-testid="supporting-evidence"
        className="bg-[#0b101b] border border-[#1d293d] rounded-2xl p-4 sm:p-5 shadow-xl space-y-4 font-sans text-xs"
      >
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#182335] pb-3">
          <div>
            <span className="text-xs text-slate-400 font-mono font-bold uppercase tracking-wider block">
              Supporting Evidence
            </span>
            <h3 className="text-sm sm:text-base font-bold text-white tracking-tight mt-0.5">
              Confluence Attribution & Multi-Model Breakdown
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
            className={`flex items-center gap-3 px-3.5 py-2 bg-[#06090f] border rounded-xl cursor-pointer transition-all shrink-0 min-h-[44px] ${
              insight.terminalState.overallEligibility !== "ELIGIBLE"
                ? "border-slate-700 hover:border-slate-500"
                : "border-[#24334b] hover:border-cyan-500"
            }`}
          >
            <div>
              <span className="text-xs text-slate-400 font-mono block">Setup Score</span>
              <span className="text-xs text-slate-500 font-mono">{insight.standard.signalsRatio}</span>
            </div>
            <div className="text-right">
              <span
                className={`text-xl font-black font-mono ${
                  insight.terminalState.overallEligibility !== "ELIGIBLE"
                    ? "text-slate-500"
                    : "text-cyan-400"
                }`}
              >
                {insight.setupScore}
                {insight.terminalState.overallEligibility !== "ELIGIBLE" ? "*" : ""}/100
              </span>
              {insight.terminalState.overallEligibility !== "ELIGIBLE" && (
                <span className="text-xs text-amber-500/80 font-mono block">Partial</span>
              )}
            </div>
          </div>
        </div>

        {/* Confluence Breakdown Bars */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-slate-300 font-bold">Confluence Breakdown</span>
            <button
              onClick={onOpenWhy}
              aria-label={`Why Score ${insight.setupScore}? Inspect Confluence Attribution`}
              className="text-xs text-cyan-400 hover:text-cyan-300 underline font-bold cursor-pointer py-1 px-1 min-h-[36px] inline-flex items-center focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none rounded"
            >
              Inspect Confluence Attribution (Why Score {insight.setupScore}?) →
            </button>
          </div>

          <div className="space-y-2">
            {(insight.standard.confluenceBreakdown || []).map((bar, idx) => (
              <div key={idx} className="space-y-1">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-400">{bar.dimension}</span>
                  <span className="text-white font-bold">{bar.score}/100</span>
                </div>
                <div className="w-full h-2 bg-[#070b13] rounded-full overflow-hidden border border-[#1b2639]">
                  <div
                    style={{ width: `${bar.score}%` }}
                    className={`h-full rounded-full transition-all duration-500 ${
                      bar.score >= 75
                        ? "bg-emerald-500"
                        : bar.score >= 50
                        ? "bg-cyan-500"
                        : "bg-rose-500"
                    }`}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Setup Summary & Key Levels Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
          <div className="p-3 bg-[#070c14] border border-[#182436] rounded-xl text-xs font-mono space-y-1">
            <span className="text-slate-500 uppercase font-bold text-xs block">Setup Structure</span>
            <span className="text-amber-300 font-bold block">{insight.standard.setupSummary}</span>
          </div>

          <div className="p-3 bg-[#070c14] border border-[#182436] rounded-xl text-xs font-mono space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-slate-400 font-bold">Profit / Risk Ratio:</span>
              <span className="text-emerald-400 font-bold text-sm">
                {kl.profitRiskRatio !== undefined ? `${kl.profitRiskRatio.toFixed(2)} : 1.0` : "N/A"}
              </span>
            </div>
            <div className="flex items-center justify-between text-slate-400">
              <span>Watch Zone:</span>
              <span className="text-amber-300 font-bold">{kl.watchZone}</span>
            </div>
          </div>
        </div>

        {/* Action Callout Button */}
        <div className="pt-2">
          {insight.terminalState.posture === "ACQUIRE" ? (
            <button
              onClick={onOpenSizer}
              className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-bold font-mono rounded-xl text-xs transition-all active:scale-95 cursor-pointer shadow-md min-h-[44px]"
            >
              ⚖️ Size & Execute Position (Buy Zone)
            </button>
          ) : insight.terminalState.posture === "EXIT_REVIEW" ? (
            <button
              onClick={onOpenWhy}
              className="w-full py-2.5 bg-rose-600 hover:bg-rose-500 text-white font-bold font-mono rounded-xl text-xs transition-all active:scale-95 cursor-pointer shadow-md min-h-[44px]"
            >
              🚨 Review Invalidation & Exit Triggers
            </button>
          ) : insight.terminalState.posture === "RESEARCH" ? (
            <button
              onClick={onOpenWhy}
              className="w-full py-2.5 bg-purple-600 hover:bg-purple-500 text-white font-bold font-mono rounded-xl text-xs transition-all active:scale-95 cursor-pointer shadow-md min-h-[44px]"
            >
              📋 Open Research & Evidence Ledger
            </button>
          ) : insight.terminalState.posture === "AVOID" ? (
            <Link
              href="/radar"
              className="w-full py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold font-mono rounded-xl text-xs transition-all active:scale-95 block text-center shadow-md cursor-pointer min-h-[44px] flex items-center justify-center"
            >
              🔎 Explore Screened Alternatives
            </Link>
          ) : (
            <button
              onClick={onOpenWhy}
              className="w-full py-2.5 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold font-mono rounded-xl text-xs transition-all active:scale-95 cursor-pointer shadow-md min-h-[44px]"
            >
              ⏳ Inspect Trigger & Milestone Criteria
            </button>
          )}
        </div>
      </div>

      <FinancialDisclaimer variant="compact" />
    </div>
  );
}
