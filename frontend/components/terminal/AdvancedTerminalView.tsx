"use client";

import React from "react";
import Link from "next/link";
import { QuantitativeInsight } from "../../types/insight";
import FinancialDisclaimer from "../FinancialDisclaimer";
import { deriveUnmetConditions } from "../../lib/decisionHierarchyUtils";

interface AdvancedTerminalViewProps {
  insight: QuantitativeInsight;
  onOpenSizer: () => void;
  onOpenWhy: () => void;
  chartSlot?: React.ReactNode;
  planSlot?: React.ReactNode;
}

export default function AdvancedTerminalView({
  insight,
  onOpenSizer,
  onOpenWhy,
  chartSlot,
  planSlot,
}: AdvancedTerminalViewProps) {
  const adv = insight.advanced;
  const unmetConditions = deriveUnmetConditions(insight);
  const isActionable = Boolean(insight.terminalState.isActionable);

  return (
    <div className="space-y-4 font-mono text-xs text-slate-100 animate-fade-in">
      {/* 🎯 FIRST-VIEWPORT COMPOSITION: QUANT VERDICT (LEFT 5-COL) + CHART (RIGHT 7-COL) */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 items-start">
        {/* Left Column (xl:col-span-5): Verdict Card & Preconditions */}
        <div className="xl:col-span-5 space-y-4 min-w-0">
          {/* 1. DECISION VERDICT & 2. DECISION REASON */}
          <div
            data-testid="decision-verdict"
            className="bg-[#0b101b] border border-[#1d293d] rounded-2xl p-4 sm:p-5 shadow-xl space-y-3 font-sans"
          >
            <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[#182335] pb-3">
              <div>
                <span className="text-xs text-slate-400 font-mono font-bold uppercase tracking-wider block">
                  ARX Quant Verdict & Posture
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

            {/* 2. CANONICAL REASON */}
            <div
              data-testid="decision-reason"
              className="text-xs sm:text-sm text-slate-200 font-sans leading-relaxed pt-1"
            >
              <span className="text-xs text-slate-400 font-bold block uppercase mb-1">
                Quant Thesis / Reason:
              </span>
              <p>{insight.terminalState.headlineExplanation || insight.standard.bottomLine}</p>
            </div>
          </div>

          {/* 3. WHAT NEEDS TO CHANGE */}
          <div
            data-testid="unmet-condition"
            className="bg-[#080e18] border border-cyan-900/50 rounded-2xl p-4 sm:p-5 shadow-lg space-y-3"
          >
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#182335] pb-2 font-mono">
              <h3 className="text-xs sm:text-sm font-bold text-cyan-300 uppercase tracking-wide flex items-center gap-2">
                <span>🎯</span>
                <span>What Needs to Change (Model Preconditions)</span>
              </h3>
              <span className="text-xs text-slate-400 font-mono">
                {isActionable ? "All Preconditions Cleared" : "Awaiting Trigger Confluence"}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-1 gap-2.5 text-xs">
              {unmetConditions.map((cond) => (
                <div
                  key={cond.id}
                  className="p-3 rounded-xl bg-[#060b13] border border-[#1b2639] space-y-1 font-mono"
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
        </div>

        {/* Right Column (xl:col-span-7): Price Chart */}
        <div className="xl:col-span-7 space-y-2 min-w-0">
          {/* 4. PRICE / CHART CONTEXT */}
          {chartSlot && (
            <div data-testid="market-workspace-chart" className="space-y-2">
              {chartSlot}
            </div>
          )}
        </div>
      </div>

      {/* 5. CONDITIONAL TRADE PLAN */}
      {planSlot && (
        <div className="space-y-2">
          {planSlot}
        </div>
      )}

      {/* 6. SUPPORTING EVIDENCE */}
      <div
        data-testid="supporting-evidence"
        className="space-y-4"
      >
        {/* Dense Quant Metrics Ribbon */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 bg-[#070b13] p-3 rounded-xl border border-[#1b2639]">
          <div className="bg-[#0b101b] p-2.5 rounded-lg border border-[#162132]">
            <span className="text-xs text-slate-500 block">RSI (14D)</span>
            <span className={`text-sm font-black ${adv.rsi !== undefined ? (adv.rsi < 30 ? "text-emerald-400" : adv.rsi > 70 ? "text-rose-400" : "text-slate-200") : "text-slate-500"}`}>
              {adv.rsi !== undefined ? adv.rsi.toFixed(1) : "N/A"}
            </span>
          </div>

          <div className="bg-[#0b101b] p-2.5 rounded-lg border border-[#162132]">
            <span className="text-xs text-slate-500 block">20 EMA</span>
            <span className="text-sm font-black text-cyan-300">
              {adv.ema20 !== undefined ? `$${adv.ema20.toFixed(2)}` : "N/A"}
            </span>
          </div>

          <div className="bg-[#0b101b] p-2.5 rounded-lg border border-[#162132]">
            <span className="text-xs text-slate-500 block">50 SMA</span>
            <span className="text-sm font-black text-indigo-300">
              {adv.sma50 !== undefined ? `$${adv.sma50.toFixed(2)}` : "N/A"}
            </span>
          </div>

          <div className="bg-[#0b101b] p-2.5 rounded-lg border border-[#162132]">
            <span className="text-xs text-slate-500 block">ATR (14D)</span>
            <span className="text-sm font-black text-amber-300">
              {adv.atr !== undefined ? `$${adv.atr.toFixed(2)}` : "N/A"}
            </span>
          </div>

          <div className="bg-[#0b101b] p-2.5 rounded-lg border border-[#162132]">
            <span className="text-xs text-slate-500 block">RVOL</span>
            <span className="text-sm font-black text-emerald-300">
              {adv.rvol !== undefined ? `${adv.rvol.toFixed(2)}×` : "N/A"}
            </span>
          </div>

          <div className="bg-[#0b101b] p-2.5 rounded-lg border border-[#162132]">
            <span className="text-xs text-slate-500 block">BETA (SPY)</span>
            <span className="text-sm font-black text-slate-200">
              {adv.beta !== undefined ? adv.beta.toFixed(2) : "N/A"}
            </span>
          </div>
        </div>

        {/* Quant Statistics & Fundamentals with Subordinated Score */}
        <div className="bg-[#0b101b] border border-[#1d293d] rounded-2xl p-4 sm:p-5 shadow-xl space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#182335] pb-3">
            <div>
              <span className="text-xs text-slate-400 font-mono font-bold uppercase tracking-wider block">
                Supporting Evidence
              </span>
              <h3 className="text-sm sm:text-base font-bold text-white tracking-tight mt-0.5">
                Quant Statistics & Factor Loadings
              </h3>
            </div>

            {/* Subordinated Score Badge */}
            <div
              onClick={onOpenWhy}
              role="button"
              aria-label="Decompose Score"
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
                  : "border-[#24334b] hover:border-cyan-500"
              }`}
            >
              <div>
                <span className="text-xs text-slate-400 font-mono block">Setup Score</span>
                <span className="text-xs text-purple-400 font-mono">Decompose Score Model →</span>
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

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
            <div className="bg-[#070b13] p-3 rounded-lg border border-[#182436]">
              <span className="text-xs text-slate-500 block">MARKET CAP</span>
              <span className="text-xs font-bold text-white mt-0.5 block">{adv.marketCap || "N/A"}</span>
            </div>

            <div className="bg-[#070b13] p-3 rounded-lg border border-[#182436]">
              <span className="text-xs text-slate-500 block">PE RATIO (TTM)</span>
              <span className="text-xs font-bold text-white mt-0.5 block">
                {adv.peRatio !== undefined ? `${adv.peRatio.toFixed(1)}×` : "N/A"}
              </span>
            </div>

            <div className="bg-[#070b13] p-3 rounded-lg border border-[#182436]">
              <span className="text-xs text-slate-500 block">ROIC (GREENBLATT)</span>
              <span className="text-xs font-bold text-emerald-400 mt-0.5 block">
                {adv.roic !== undefined ? `${adv.roic.toFixed(1)}%` : "N/A"}
              </span>
            </div>

            <div className="bg-[#070b13] p-3 rounded-lg border border-[#182436]">
              <span className="text-xs text-slate-500 block">DEBT / EQUITY</span>
              <span className="text-xs font-bold text-white mt-0.5 block">
                {adv.debtToEquity !== undefined ? adv.debtToEquity.toFixed(2) : "N/A"}
              </span>
            </div>

            <div className="bg-[#070b13] p-3 rounded-lg border border-[#182436]">
              <span className="text-xs text-slate-500 block">RELATIVE STRENGTH</span>
              <span className="text-xs font-bold text-cyan-400 mt-0.5 block">
                {adv.relativeStrengthScore !== undefined ? `${adv.relativeStrengthScore}/100` : "N/A"}
              </span>
            </div>

            <div className="bg-[#070b13] p-3 rounded-lg border border-[#182436]">
              <span className="text-xs text-slate-500 block">PARAMETRIC VAR</span>
              <span className="text-xs font-bold text-rose-400 mt-0.5 block">
                {adv.var95Pct !== undefined ? `-${adv.var95Pct}% (95% 1D)` : "N/A (< 20 sessions)"}
              </span>
            </div>
          </div>

          <div className="p-3 bg-[#06090f] border border-[#151f2e] rounded-xl text-slate-300 text-xs space-y-1 font-sans">
            <span className="text-slate-400 font-mono font-bold text-xs uppercase block">
              Multi-Factor Archetype Classification
            </span>
            <p className="text-slate-300 leading-relaxed font-mono text-xs">
              VCP Structure: {adv.vcpStage ? `Stage ${adv.vcpStage} Contraction` : "Stage 4 Correction"} | 50 SMA: {adv.sma50 !== undefined ? `$${adv.sma50.toFixed(2)}` : "N/A"}
            </p>
          </div>

          <div className="pt-1">
            {insight.terminalState.posture === "ACQUIRE" ? (
              <button
                onClick={onOpenSizer}
                className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-black rounded-xl transition-all active:scale-95 cursor-pointer shadow-md min-h-[44px]"
              >
                ⚖️ Open Institutional Position Sizer
              </button>
            ) : insight.terminalState.posture === "EXIT_REVIEW" ? (
              <button
                onClick={onOpenWhy}
                className="w-full py-2.5 bg-rose-600 hover:bg-rose-500 text-white font-black rounded-xl transition-all active:scale-95 cursor-pointer shadow-md min-h-[44px]"
              >
                🚨 Review Invalidation & Breaches
              </button>
            ) : insight.terminalState.posture === "RESEARCH" ? (
              <button
                onClick={onOpenWhy}
                className="w-full py-2.5 bg-purple-600 hover:bg-purple-500 text-white font-black rounded-xl transition-all active:scale-95 cursor-pointer shadow-md min-h-[44px]"
              >
                📋 Open Quantitative Evidence Ledger
              </button>
            ) : insight.terminalState.posture === "AVOID" ? (
              <Link
                href="/radar"
                className="w-full py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 font-black rounded-xl transition-all active:scale-95 cursor-pointer shadow-md min-h-[44px] flex items-center justify-center text-center"
              >
                🔎 Explore Screened Opportunities
              </Link>
            ) : (
              <button
                onClick={onOpenWhy}
                className="w-full py-2.5 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-black rounded-xl transition-all active:scale-95 cursor-pointer shadow-md min-h-[44px]"
              >
                ⏳ Monitor Technical Triggers
              </button>
            )}
          </div>
        </div>
      </div>

      <FinancialDisclaimer variant="compact" />
    </div>
  );
}
