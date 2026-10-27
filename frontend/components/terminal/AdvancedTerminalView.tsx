"use client";

import React from "react";
import Link from "next/link";
import { QuantitativeInsight } from "../../types/insight";
import FinancialDisclaimer from "../FinancialDisclaimer";
import { deriveUnmetConditions } from "../../lib/decisionHierarchyUtils";
import SharedAssetDeck from "./SharedAssetDeck";
import SharedExecutionLadder from "./SharedExecutionLadder";

interface AdvancedTerminalViewProps {
  insight: QuantitativeInsight;
  onOpenSizer: () => void;
  onOpenWhy: () => void;
  chartSlot?: React.ReactNode;
  planSlot?: React.ReactNode;
  preflightSlot?: React.ReactNode;
  evidenceSlot?: React.ReactNode;
  isDemo?: boolean;
}

export default function AdvancedTerminalView({
  insight,
  onOpenSizer,
  onOpenWhy,
  chartSlot,
  planSlot,
  preflightSlot,
  evidenceSlot,
  isDemo = false,
}: AdvancedTerminalViewProps) {
  const unmetConditions = deriveUnmetConditions(insight);

  const isStaleTape = Boolean(
    insight.terminalState?.decisionState === "STALE_DATA" ||
    insight.freshness?.status === "STALE" ||
    insight.freshness?.status === "STALE_HISTORICAL" ||
    (insight.freshness?.stalenessDays !== undefined && insight.freshness.stalenessDays > 4) ||
    insight.terminalState?.freshness?.status === "STALE" ||
    insight.terminalState?.freshness?.status === "STALE_HISTORICAL" ||
    (insight.terminalState?.freshness?.stalenessDays !== undefined && insight.terminalState.freshness.stalenessDays > 4) ||
    insight.terminalState?.freshnessStatus === "STALE_HISTORICAL" ||
    insight.terminalState?.freshnessStatus === "STALE" ||
    (insight.terminalState?.stalenessDays !== undefined && insight.terminalState.stalenessDays > 4)
  );

  const isActionable = isStaleTape ? false : Boolean(insight.terminalState?.isActionable ?? (insight as any).actionable);

  const kl = insight.standard?.keyLevels || (insight as any).keyLevels || {};
  const rawPrice = insight.price || kl.currentPrice;
  const hasValidPrice = typeof rawPrice === "number" && !isNaN(rawPrice) && rawPrice > 0;
  const isExtended = Boolean(kl.entryMax && hasValidPrice && (rawPrice as number) > kl.entryMax);
  const isWaitingPullback = Boolean(kl.entryMin && hasValidPrice && (rawPrice as number) < kl.entryMin);
  const isInBuyZone = Boolean(hasValidPrice && kl.entryMin && kl.entryMax && (rawPrice as number) >= kl.entryMin && (rawPrice as number) <= kl.entryMax);
  const pivotPrice = ((kl as any)?.breakoutPivot != null && !isNaN((kl as any)?.breakoutPivot))
    ? Number((kl as any).breakoutPivot)
    : (kl.entryMin != null ? Number(kl.entryMin) : null);

  const adv = insight.advanced || ({} as any);

  return (
    <div className="space-y-4 font-sans text-slate-100 animate-fade-in">
      {/* 🎯 FIRST-VIEWPORT COMPOSITION: QUANT VERDICT (LEFT 5-COL) + CHART (RIGHT 7-COL) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 xl:grid-cols-12 gap-4 items-start">
        {/* Left Column (lg:col-span-5 xl:col-span-5): Verdict Card & Preconditions */}
        <div className="lg:col-span-5 xl:col-span-5 space-y-2.5 sm:space-y-4 min-w-0 order-1 lg:order-1 xl:order-1">
          {/* 1. DECISION VERDICT & 2. DECISION REASON */}
          <article
            id="verdict-card"
            data-testid="decision-verdict"
            className="verdict-card bg-[#0b101b] border border-[#1d293d] rounded-2xl p-3 sm:p-5 shadow-xl space-y-3 font-sans text-slate-100"
            aria-label="ARX Quant Decision and Model Posture"
          >
            {/* Ticker & Price Deck */}
            <SharedAssetDeck
              insight={insight}
              isStaleTape={isStaleTape}
              isActionable={isActionable}
              isWaitingPullback={isWaitingPullback}
              isExtended={isExtended}
              isDemo={isDemo}
            />

            {/* Quant Verdict & Thesis Banner */}
            <div className="verdict-banner !flex !flex-col p-3 sm:p-4 rounded-xl bg-[#080e18] border border-[#182335] space-y-2">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div className="space-y-0.5">
                  <span className="text-[10px] text-slate-400 font-mono font-bold uppercase tracking-wider block">
                    ARX Quant Verdict & Model Posture
                  </span>
                  <h2 className="text-lg sm:text-2xl font-black text-white tracking-tight">
                    {insight.verdictLabel}
                  </h2>
                </div>
                {adv.relativeStrengthScore !== undefined && (
                  <div className="text-right">
                    <span className="text-[10px] text-slate-400 font-mono block">RS Score</span>
                    <span className="text-sm font-mono font-black text-purple-300">
                      {adv.relativeStrengthScore} / 100
                    </span>
                  </div>
                )}
              </div>

              {/* CANONICAL REASON / QUANT THESIS */}
              <div
                data-testid="decision-reason"
                className="text-xs sm:text-sm text-slate-200 font-sans leading-relaxed pt-1 border-t border-[#182335]/60"
              >
                <span className="text-[10px] text-slate-400 font-bold block uppercase mb-0.5">
                  Quant Thesis / Reason:
                </span>
                <p>{insight.terminalState?.headlineExplanation || insight.standard?.bottomLine}</p>
              </div>
            </div>

            {/* Quant Execution Ladder */}
            <SharedExecutionLadder
              insight={insight}
              mode="QUANT"
              isStaleTape={isStaleTape}
              isActionable={isActionable}
              isExtended={isExtended}
              isInBuyZone={isInBuyZone}
              pivotPrice={pivotPrice}
              onOpenSizer={onOpenSizer}
            />
          </article>

          {/* 3. WHAT NEEDS TO CHANGE */}
          <div
            data-testid="unmet-condition"
            className="bg-[#080e18] border border-cyan-900/50 rounded-2xl p-3 sm:p-5 shadow-lg space-y-2 sm:space-y-3"
          >
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#182335] pb-2 font-mono">
              <h3 className="text-xs sm:text-sm font-bold text-cyan-300 uppercase tracking-wide flex items-center gap-2">
                <span>🎯</span>
                <span>What Needs to Change (Model Criteria)</span>
              </h3>
              <span className="text-xs text-slate-400 font-mono">
                {isActionable ? "Setup Criteria Met" : "Awaiting Trigger Confluence"}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-1 xl:grid-cols-1 gap-2 text-xs">
              {unmetConditions.map((cond) => (
                <div
                  key={cond.id}
                  className="p-2.5 sm:p-3 rounded-xl bg-[#060b13] border border-[#1b2639] space-y-1 font-mono"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-slate-200 text-xs">{cond.title}</span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] sm:text-xs font-mono font-bold ${
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

        {/* Right Column (lg:col-span-7 xl:col-span-7): Price Chart */}
        <div className="lg:col-span-7 xl:col-span-7 space-y-2 min-w-0 order-2 lg:order-2 xl:order-2">
          {/* 4. PRICE / CHART CONTEXT */}
          {chartSlot && (
            <div data-testid="market-workspace-chart" className="space-y-2">
              {chartSlot}
            </div>
          )}
        </div>
      </div>

      {/* 🎯 SYNTHESIS E V3: INLINE PRE-FLIGHT CLEARANCE PANEL */}
      {preflightSlot !== undefined ? (
        preflightSlot
      ) : (
        <div id="preflight-readiness-bar" className="bg-[#0b101b] border border-[#1d293d] rounded-2xl p-4 text-xs font-mono text-slate-400 flex items-center justify-between">
          <span>Pre-Flight Readiness Bar</span>
          <span className="text-emerald-400 font-bold">CLEARANCE PASS</span>
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
            <span className="text-sm font-black text-purple-300">
              {adv.beta !== undefined ? adv.beta.toFixed(2) : "1.00"}
            </span>
          </div>
        </div>

        {/* Quant Confluence Breakdown & Evidence */}
        <div className="bg-[#0b101b] border border-[#1d293d] rounded-2xl p-4 sm:p-5 shadow-xl space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#182335] pb-3">
            <div>
              <span className="text-xs text-slate-400 font-mono font-bold uppercase tracking-wider block">
                Quant Confluence Factors
              </span>
              <span className="text-sm font-semibold text-white">
                Multi-Factor Statistical Weighting
              </span>
            </div>

            <button
              onClick={onOpenWhy}
              aria-label="Decompose Score"
              className="px-3 py-1.5 rounded-lg border border-purple-800/80 bg-purple-950/40 text-purple-300 hover:bg-purple-900/60 transition-colors flex items-center gap-1.5 min-h-[36px] text-xs font-mono cursor-pointer"
            >
              <span className="text-xs text-purple-400 font-mono">Decompose Score Model →</span>
            </button>
          </div>

          {evidenceSlot ? (
            evidenceSlot
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
              <div className="p-3 bg-[#080d16] rounded-xl border border-[#162132]">
                <span className="text-slate-400 block mb-1">VCP Compression Status</span>
                <span className="text-slate-200 font-bold">{adv.vcpStage || "Contraction Undetected"}</span>
              </div>
              <div className="p-3 bg-[#080d16] rounded-xl border border-[#162132]">
                <span className="text-slate-400 block mb-1">Volatility Regime</span>
                <span className="text-cyan-400 font-bold">{(adv as any)?.volatilityRegime || "NORMAL"}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <FinancialDisclaimer variant="compact" />
    </div>
  );
}
