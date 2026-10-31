"use client";

import React, { useState } from "react";
import Link from "next/link";
import { QuantitativeInsight } from "../../types/insight";
import FinancialDisclaimer from "../FinancialDisclaimer";
import { evaluateLevelRelation } from "../../lib/reclaimSemantics";
import { evaluatePreFlightChecks } from "../../lib/preflightEvaluator";
import SharedAssetDeck from "./SharedAssetDeck";
import SharedExecutionLadder from "./SharedExecutionLadder";
import PreFlightReadinessBar, { PreFlightReadinessBarProps } from "../PreFlightReadinessBar";

interface GuidedTerminalViewProps {
  insight: QuantitativeInsight;
  onOpenSizer: () => void;
  onOpenWhy: () => void;
  chartSlot?: React.ReactNode;
  planSlot?: React.ReactNode;
  preflightSlot?: React.ReactNode;
  evidenceSlot?: React.ReactNode;
  isDemo?: boolean;
  isCleared?: boolean;
  activeBlocker?: string | null;
  passedCount?: number;
}

export default function GuidedTerminalView({
  insight,
  onOpenSizer,
  onOpenWhy,
  chartSlot,
  planSlot,
  preflightSlot,
  evidenceSlot,
  isDemo = false,
  isCleared,
  activeBlocker,
  passedCount,
}: GuidedTerminalViewProps) {
  const [blockerTarget, setBlockerTarget] = React.useState<HTMLDivElement | null>(null);
  const [activeStep, setActiveStep] = useState<number | null>(null);

  const resolvedPreflight = React.useMemo(() => {
    if (isCleared !== undefined) {
      return { isCleared, activeBlocker: activeBlocker || null, passedCount: passedCount || 0 };
    }
    if (React.isValidElement(preflightSlot) && (preflightSlot.props as any)) {
      const pProps = preflightSlot.props as any;
      if (typeof pProps.isCleared === "boolean") {
        return { isCleared: pProps.isCleared, activeBlocker: pProps.activeBlocker || null, passedCount: pProps.passedCount || 0 };
      }
      try {
        const ev = evaluatePreFlightChecks(pProps);
        return { isCleared: ev.isCleared, activeBlocker: ev.activeBlocker, passedCount: ev.passedCount };
      } catch {}
    }
    return { isCleared: undefined, activeBlocker: null, passedCount: undefined };
  }, [isCleared, activeBlocker, passedCount, preflightSlot]);

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

  const smaRel = evaluateLevelRelation(
    insight.price,
    kl.sma50,
    "50D SMA",
    insight.symbol
  );

  const steps = [
    { title: "1. What's Happening?", text: `${insight.symbol} is trading at $${insight.price.toFixed(2)}, ${insight.changePct >= 0 ? "+" : ""}${insight.changePct.toFixed(2)}% today. ${insight.human?.assessmentDescription || insight.terminalState?.headlineExplanation}` },
    { title: "2. What's the Setup?", text: `ARX identifies the current structure as ${insight.standard?.setupSummary || "Evaluated Setup"}. ${insight.advanced?.relativeStrengthScore !== undefined ? `Relative strength score is ${insight.advanced.relativeStrengthScore}/100.` : "Relative strength score is unverified for this security."}` },
    { title: "3. Why does ARX like/caution it?", text: insight.human?.reclaimMilestone || insight.standard?.bottomLine || "Evaluating confluence factors." },
    { title: "4. What could go wrong?", text: `Every thesis has downside risk. If price breaks below $${(kl.stopLoss || 0).toFixed(2)}, the setup is invalidated.` },
    { title: "5. How could I trade it?", text: `Plan: ${
      smaRel.status === "BELOW"
        ? `Watch for reclaim of $${(kl.sma50 as number || 0).toFixed(2)} (50D SMA).`
        : smaRel.status === "AT_LEVEL"
        ? `Testing 50D SMA at $${(kl.sma50 as number || 0).toFixed(2)}. Watch for decisive volume expansion.`
        : smaRel.status === "ABOVE"
        ? `Holding constructively above 50D SMA ($${(kl.sma50 as number || 0).toFixed(2)}). Watch for base confirmation.`
        : "Watch key technical levels."
    } Target 1 is ${kl.target1 !== undefined ? `$${kl.target1.toFixed(2)} (+${kl.target1Pct || 0}%)` : "N/A (< 50 sessions)"}.` },
    { title: "6. What should I monitor?", text: "Volume surges, 50-day moving average crossovers, and broader market regime stability." },
  ];

  return (
    <div className="space-y-4 font-sans text-[var(--text-1)] animate-fade-in">

      {/* 🎯 FIRST-VIEWPORT COMPOSITION: VERDICT / GUIDANCE (LEFT 5-COL) + CHART (RIGHT 7-COL) */}
      <div data-testid="primary-workspace-grid" className="grid grid-cols-1 lg:grid-cols-12 xl:grid-cols-12 gap-4 items-start">
        {/* Left Column (lg:col-span-5 xl:col-span-5): Verdict Card & Preconditions */}
        <div className="lg:col-span-5 xl:col-span-5 space-y-2.5 sm:space-y-4 min-w-0 order-1 lg:order-1 xl:order-1">
          {/* 1. DECISION VERDICT & 2. DECISION REASON */}
          <article
            id="verdict-card"
            data-testid="decision-verdict"
            className="verdict-card bg-[var(--surface-1)] border border-[var(--border-subtle)] rounded-2xl p-3 sm:p-5 shadow-xl space-y-3 font-sans text-[var(--text-1)]"
            aria-label="ARX Novice Guidance and Execution Decision"
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

            {/* Guided Verdict & Novice Guidance Banner */}
            <div className="verdict-banner !flex !flex-col p-3 sm:p-4 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)] space-y-2">
              <div className="space-y-1">
                <span className="text-[10px] text-[var(--text-2)] font-mono font-bold uppercase tracking-wider block">
                  ARX Novice Guidance & Assessment
                </span>
                <h2 className="text-lg sm:text-2xl font-black text-white tracking-tight">
                  {insight.human?.assessmentHeadline || insight.verdictLabel}
                </h2>
                <div className="mt-1 flex items-center gap-2">
                  <span className="text-xs font-mono text-cyan-300 font-bold">
                    Canonical Verdict: {insight.verdictLabel}
                  </span>
                </div>
              </div>

              {/* 2. CANONICAL REASON */}
              <div
                data-testid="decision-reason"
                className="text-xs sm:text-sm text-[var(--text-1)] font-sans leading-relaxed pt-1 border-t border-[var(--border-subtle)]"
              >
                <span className="text-[10px] text-[var(--text-2)] font-bold block uppercase mb-0.5">
                  Plain English Explanation:
                </span>
                <p>{insight.human?.assessmentDescription || insight.terminalState?.headlineExplanation || insight.standard?.bottomLine}</p>
              </div>
            </div>

            {/* Guided Execution Ladder */}
            <div ref={setBlockerTarget} id="canonical-blocker-summary" data-testid="unmet-condition" aria-label="Canonical blocking reasons" />
            <SharedExecutionLadder
              insight={insight}
              mode="GUIDED"
              isStaleTape={isStaleTape}
              isActionable={isActionable}
              isExtended={isExtended}
              isInBuyZone={isInBuyZone}
              pivotPrice={pivotPrice}
              onOpenSizer={onOpenSizer}
              isCleared={resolvedPreflight.isCleared}
              activeBlocker={resolvedPreflight.activeBlocker}
              passedCount={resolvedPreflight.passedCount}
            />
          </article>

          {/* 3. WHAT NEEDS TO CHANGE */}

        </div>

        {/* Right Column (lg:col-span-7 xl:col-span-7): Price Chart */}
        <div className="lg:col-span-7 xl:col-span-7 space-y-2 min-w-0 order-2 lg:order-2 xl:order-2">
          {/* 4. PRICE / CHART CONTEXT */}
          {chartSlot && (
            <div data-testid="market-workspace-chart" className="space-y-2">
              {chartSlot}
            </div>
          )}
                    <section className="space-y-2" aria-label="Canonical preflight readiness">{preflightSlot !== undefined ? (
        React.isValidElement(preflightSlot) && preflightSlot.type === PreFlightReadinessBar
          ? React.cloneElement(preflightSlot as React.ReactElement<PreFlightReadinessBarProps>, { summaryTarget: blockerTarget })
          : preflightSlot
      ) : (
        <PreFlightReadinessBar summaryTarget={blockerTarget} symbol={insight.symbol} currentPrice={hasValidPrice ? rawPrice as number : 0}
          stopLoss={kl.stopLoss} takeProfit1={kl.target1} riskRewardRatio={kl.profitRiskRatio}
          optimalEntryMin={kl.entryMin} optimalEntryMax={kl.entryMax}
          isActionable={isActionable} decisionState={insight.terminalState?.decisionState}
          isSynthetic={isDemo} />
      )}</section>
        </div>

</div>

      {/* 🎯 SYNTHESIS E V3: INLINE PRE-FLIGHT CLEARANCE PANEL */}


      {/* 5. CONDITIONAL TRADE PLAN */}
      {planSlot && (
        <div className="space-y-2">
          {planSlot}
        </div>
      )}

      {/* 6. SUPPORTING EVIDENCE */}
      {evidenceSlot ? (
        evidenceSlot
      ) : (
        <div
          data-testid="supporting-evidence"
          className="bg-[var(--surface-1)] border border-[var(--border-subtle)] rounded-2xl p-4 sm:p-5 shadow-xl space-y-4"
        >
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border-subtle)] pb-3">
            <div>
              <span className="text-xs text-[var(--text-2)] font-mono font-bold uppercase tracking-wider block">
                Supporting Evidence
              </span>
              <span className="text-sm font-semibold text-white">
                Novice Plain-Language Confluence Factors
              </span>
            </div>

            <button
              onClick={onOpenWhy}
              className="px-3 py-1.5 rounded-lg border border-cyan-800/80 bg-cyan-950/40 text-cyan-300 hover:bg-cyan-900/60 transition-colors flex items-center gap-1.5 min-h-[36px] text-xs font-mono cursor-pointer"
            >
              <span className="text-xs text-cyan-400 font-mono">Explain Score →</span>
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            {insight.human?.whyPills?.map((pill, idx) => (
              <div
                key={idx}
                className={`p-3 rounded-xl border flex items-start gap-2.5 ${
                  pill.sentiment === "positive"
                    ? "bg-emerald-950/30 border-emerald-800/50 text-emerald-200"
                    : pill.sentiment === "negative"
                    ? "bg-rose-950/30 border-rose-800/50 text-rose-200"
                    : "bg-slate-900/50 border-slate-700/50 text-[var(--text-2)]"
                }`}
              >
                <span className="text-sm shrink-0">
                  {pill.sentiment === "positive" ? "✅" : pill.sentiment === "negative" ? "⚠️" : "ℹ️"}
                </span>
                <div>
                  <strong className="block font-mono text-[11px] uppercase tracking-wide opacity-80">
                    {pill.category}
                  </strong>
                  <p className="mt-0.5 leading-snug">{(pill as any).text || pill.description}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Posture-adaptive primary action CTA */}
          <div className="pt-2">
            {insight.terminalState.posture === "ACQUIRE" ? (
              <button
                onClick={onOpenSizer}
                className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 text-slate-950 rounded-xl text-xs font-mono font-black shadow-md transition-all active:scale-95 cursor-pointer min-h-[44px]"
              >
                🚀 Size Position (Buy Zone)
              </button>
            ) : insight.terminalState.posture === "EXIT_REVIEW" ? (
              <button
                onClick={onOpenWhy}
                className="w-full py-2.5 bg-rose-600 hover:bg-rose-500 text-white rounded-xl text-xs font-mono font-black shadow-md transition-all active:scale-95 cursor-pointer min-h-[44px]"
              >
                ⚠️ Review Invalidation
              </button>
            ) : insight.terminalState.posture === "RESEARCH" ? (
              <button
                onClick={onOpenWhy}
                className="w-full py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-xl text-xs font-mono font-black shadow-md transition-all active:scale-95 cursor-pointer min-h-[44px]"
              >
                📚 Open Research
              </button>
            ) : insight.terminalState.posture === "AVOID" ? (
              <Link
                href="/radar"
                className="w-full py-2.5 bg-slate-800 hover:bg-slate-700 text-[var(--text-1)] rounded-xl text-xs font-mono font-black shadow-md transition-all active:scale-95 cursor-pointer min-h-[44px] flex items-center justify-center"
              >
                🎯 Find Setups
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
      )}

      {/* 6-Step Novice Walkthrough */}
      <div className="bg-[var(--surface-2)] border border-[var(--border-subtle)] rounded-xl p-3.5 space-y-2.5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono font-bold text-[var(--text-2)] flex items-center gap-1.5">
            <span>🧭</span> Step-by-Step Stock Analysis Walkthrough
          </span>
          <span className="text-xs text-[var(--text-muted)] font-mono">Click any step to inspect</span>
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
                  : "bg-[var(--surface-1)] border-[var(--border-subtle)] text-[var(--text-2)] hover:text-slate-200"
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
            className="p-3.5 bg-[var(--surface-2)] border border-cyan-800/40 rounded-lg text-xs font-sans text-[var(--text-1)] animate-fade-in leading-relaxed"
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
