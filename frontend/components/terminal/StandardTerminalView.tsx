"use client";

import React from "react";
import Link from "next/link";
import { QuantitativeInsight } from "../../types/insight";
import FinancialDisclaimer from "../FinancialDisclaimer";
import { deriveUnmetConditions } from "../../lib/decisionHierarchyUtils";
import PreFlightReadinessBar from "../PreFlightReadinessBar";
import SharedAssetDeck from "./SharedAssetDeck";
import SharedExecutionLadder from "./SharedExecutionLadder";

interface StandardTerminalViewProps {
  insight: QuantitativeInsight;
  onOpenSizer: () => void;
  onOpenWhy: () => void;
  chartSlot?: React.ReactNode;
  planSlot?: React.ReactNode;
  preflightSlot?: React.ReactNode;
  evidenceSlot?: React.ReactNode;
  isDemo?: boolean;
}

export default function StandardTerminalView({
  insight,
  onOpenSizer,
  onOpenWhy,
  chartSlot,
  planSlot,
  preflightSlot,
  evidenceSlot,
  isDemo = false,
}: StandardTerminalViewProps) {
  const kl = insight.standard.keyLevels;
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

  const isActionable = isStaleTape ? false : Boolean(insight.terminalState.isActionable);
  const [isPreconditionsOpen, setIsPreconditionsOpen] = React.useState(false);
  const [isChartExpanded, setIsChartExpanded] = React.useState(false);

  const rawPrice = insight.price || kl.currentPrice;
  const hasValidPrice = typeof rawPrice === "number" && !isNaN(rawPrice) && rawPrice > 0;
  const isExtended = Boolean(kl.entryMax && hasValidPrice && (rawPrice as number) > kl.entryMax);
  const isWaitingPullback = Boolean(kl.entryMin && hasValidPrice && (rawPrice as number) < kl.entryMin);
  const isInBuyZone = Boolean(hasValidPrice && kl.entryMin && kl.entryMax && (rawPrice as number) >= kl.entryMin && (rawPrice as number) <= kl.entryMax);
  const pivotPrice = ((kl as any)?.breakoutPivot != null && !isNaN((kl as any)?.breakoutPivot))
    ? Number((kl as any).breakoutPivot)
    : (kl.entryMin != null ? Number(kl.entryMin) : null);
  const hasValidZone = Boolean(
    kl.watchZone &&
    kl.watchZone.trim().toLowerCase() !== "unavailable" &&
    (kl.entryMin != null || kl.entryMax != null || kl.watchZone.includes("$"))
  );

  return (
    <div className="space-y-4 font-sans text-[var(--text-1)] animate-fade-in">
      {/* 🎯 FIRST-VIEWPORT COMPOSITION: VERDICT (LEFT 5-COL) + CHART (RIGHT 7-COL) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 xl:grid-cols-12 gap-4 items-start">
        {/* Left Column (lg:col-span-5 xl:col-span-5): Verdict Card & Preconditions */}
        <div className="lg:col-span-5 xl:col-span-5 space-y-2.5 sm:space-y-4 min-w-0 order-1 lg:order-1 xl:order-1">
          {/* 1. DECISION VERDICT (Dominant Visual Weight) & 2. DECISION REASON */}
          <article
            id="verdict-card"
            data-testid="decision-verdict"
            className="verdict-card bg-[var(--surface-1)] border border-[var(--border-subtle)] rounded-2xl p-3 sm:p-5 shadow-xl space-y-3 font-sans text-[var(--text-1)]"
            aria-label="ARX Bottom Line and Execution Decision"
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

            {/* Verdict & Score Banner */}
            <div className="verdict-banner !flex !flex-col p-3 sm:p-4 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)] space-y-2.5">
              <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2.5 sm:gap-3 w-full">
                <div className="verdict-info space-y-1 min-w-0 flex-1">
                  <span id="card-verdict-label" className="verdict-label text-[10px] font-mono font-black text-cyan-400 uppercase tracking-widest block">
                    ARX BOTTOM LINE VERDICT
                  </span>
                  <h2 id="card-verdict-title" className="verdict-title text-base sm:text-lg font-black text-white tracking-tight leading-snug break-words">
                    {insight.verdictLabel}
                  </h2>
                </div>

                {/* Desktop Setup Score Pill & Why Score Button */}
                <div className="score-box hidden sm:flex flex-col items-center gap-1 shrink-0">
                  <div
                    id="card-score-pill"
                    onClick={onOpenWhy}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onOpenWhy();
                      }
                    }}
                    title="Setup Score: click to inspect confluence attribution"
                    className="score-pill px-3 py-1 rounded-xl bg-[var(--surface-2)] border border-cyan-500/40 text-center shadow-md cursor-pointer hover:border-cyan-400 transition-all"
                  >
                    <span id="card-score-num" className="score-num mono font-mono font-black text-base text-cyan-400 leading-none">
                      {insight.setupScore}
                    </span>
                    <span id="card-score-den" className="score-den font-mono text-xs text-[var(--text-2)]">
                      /100
                    </span>
                  </div>
                  <button
                    id="card-why-score-btn"
                    type="button"
                    onClick={onOpenWhy}
                    aria-label="Inspect Score Confluence"
                    className="why-score-btn text-[10px] font-mono text-cyan-400 hover:underline cursor-pointer flex items-center gap-1 min-h-[36px]"
                  >
                    <span id="why-score-text">
                      {isStaleTape ? "Historical Tape Score" : "Why Score?"}
                    </span>
                  </button>
                </div>
              </div>

              {/* 2. CANONICAL REASON (Directly Below Verdict) */}
              <div
                data-testid="decision-reason"
                className="text-xs text-[var(--text-2)] font-sans leading-relaxed border-t border-[var(--border-subtle)] pt-2 w-full"
              >
                <span className="text-[10px] text-[var(--text-2)] font-mono font-bold block uppercase mb-0.5 tracking-wider">
                  Reason / Bottom Line:
                </span>
                <p id="card-verdict-summary" className="verdict-summary leading-relaxed">
                  {insight.standard.bottomLine}
                </p>
              </div>

              {/* 3. MOBILE SCORE DECK (Positioned strictly AFTER Rationale on mobile) */}
              <div
                id="card-mobile-score-deck"
                data-testid="mobile-confluence-score"
                className="sm:hidden flex items-center justify-between pt-2 border-t border-[var(--border-subtle)] w-full"
              >
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-[var(--text-2)] uppercase tracking-wider">
                    Confluence Score:
                  </span>
                  <div
                    id="card-score-pill-mobile"
                    onClick={onOpenWhy}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onOpenWhy();
                      }
                    }}
                    title="Setup Score: click to inspect confluence attribution"
                    className="px-2.5 py-0.5 rounded-lg bg-[var(--surface-2)] border border-cyan-500/40 text-center cursor-pointer flex items-baseline gap-0.5"
                  >
                    <span className="score-num mono font-mono font-black text-sm text-cyan-400 leading-none">
                      {insight.setupScore}
                    </span>
                    <span className="score-den font-mono text-[10px] text-[var(--text-2)]">
                      /100
                    </span>
                  </div>
                </div>
                <button
                  id="card-why-score-btn-mobile"
                  type="button"
                  onClick={onOpenWhy}
                  aria-label="Inspect Score Confluence"
                  className="text-[10px] font-mono text-cyan-400 hover:underline min-h-[32px] flex items-center"
                >
                  {isStaleTape ? "Historical Tape Score" : "Why Score?"}
                </button>
              </div>
            </div>

            {/* Middle Panel: Risk & Execution Ladder */}
            <SharedExecutionLadder
              insight={insight}
              mode="STANDARD"
              isStaleTape={isStaleTape}
              isActionable={isActionable}
              isExtended={isExtended}
              isInBuyZone={isInBuyZone}
              pivotPrice={pivotPrice}
              onOpenSizer={onOpenSizer}
            />
          </article>

          {/* 3. WHAT NEEDS TO CHANGE (Unmet Conditions & Preconditions) */}
          <div
            data-testid="unmet-condition"
            className="bg-[var(--surface-2)] border border-cyan-900/50 rounded-2xl p-2.5 sm:p-5 shadow-lg space-y-2 sm:space-y-3"
          >
            <div
              className="flex flex-wrap items-center justify-between gap-1.5 sm:gap-2 border-b border-[var(--border-subtle)] pb-2 cursor-pointer sm:cursor-default"
              onClick={() => setIsPreconditionsOpen((prev) => !prev)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setIsPreconditionsOpen((prev) => !prev);
                }
              }}
              aria-expanded={isPreconditionsOpen}
            >
              <h3 className="text-xs sm:text-sm font-bold text-cyan-300 uppercase font-mono tracking-wide flex items-center gap-2">
                <span>🎯</span>
                <span>What Needs to Change (Setup Criteria)</span>
              </h3>
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] sm:text-xs text-[var(--text-2)] font-mono">
                  {isActionable ? "Setup Criteria Met" : "Awaiting Setup Triggers"}
                </span>
                <span className="text-[var(--text-2)] text-xs" aria-hidden="true">
                  {isPreconditionsOpen ? "▲" : "▼"}
                </span>
              </div>
            </div>

            <div className={`${isPreconditionsOpen ? "grid" : "hidden"} grid-cols-1 md:grid-cols-2 lg:grid-cols-1 xl:grid-cols-1 gap-1.5 sm:gap-2.5 text-xs`}>
              {unmetConditions.map((cond) => (
                <div
                  key={cond.id}
                  className="py-1.5 px-2.5 sm:p-3 rounded-lg sm:rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)] space-y-1"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-bold text-[var(--text-1)] text-xs truncate">{cond.title}</span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] sm:text-xs font-mono font-bold shrink-0 ${
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
                  <p className="text-xs text-[var(--text-2)] leading-relaxed font-sans hidden sm:block">
                    {cond.description}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column (lg:col-span-7 xl:col-span-7): Price Chart & Mobile Preview */}
        <div className="lg:col-span-7 xl:col-span-7 min-w-0 order-2 lg:order-2 xl:order-2">
          {/* Mobile Chart Preview Affordance (< md only, ~56-64px height) */}
          <div
            data-testid="mobile-chart-preview"
            className="md:hidden bg-[var(--surface-1)] border border-[var(--border-subtle)] rounded-xl px-3 py-2 flex items-center justify-between gap-2 shadow-md min-h-[56px] mb-2"
          >
            <div className="flex items-center gap-2 min-w-0">
              <div className="flex items-center gap-1.5 shrink-0">
                <span className="font-mono font-bold text-white text-xs sm:text-sm">
                  {insight.symbol || "UNAVAILABLE"}
                </span>
                {hasValidPrice ? (
                  <span
                    data-testid="preview-spot-price"
                    className="text-emerald-400 font-mono font-bold text-xs sm:text-sm tabular-nums"
                  >
                    ${(rawPrice as number).toFixed(2)}
                  </span>
                ) : (
                  <span
                    data-testid="preview-spot-price"
                    className="text-[var(--text-2)] font-mono text-xs font-semibold"
                  >
                    UNAVAILABLE
                  </span>
                )}
              </div>
              {hasValidZone ? (
                <span
                  data-testid="preview-watch-zone"
                  title={`Watch Zone: ${kl.watchZone}`}
                  className="text-[10px] font-mono text-amber-300 bg-amber-950/70 border border-amber-800/80 px-1.5 py-0.5 rounded truncate max-w-[120px]"
                >
                  {kl.watchZone}
                </span>
              ) : (
                <span
                  data-testid="preview-watch-zone"
                  className="text-[10px] font-mono text-[var(--text-2)] bg-slate-900 border border-slate-800 px-1.5 py-0.5 rounded"
                >
                  UNAVAILABLE
                </span>
              )}
            </div>

            <button
              type="button"
              onClick={() => setIsChartExpanded((prev) => !prev)}
              aria-expanded={isChartExpanded}
              aria-controls="market-workspace-chart"
              aria-label={isChartExpanded ? "Collapse candlestick price chart" : "Expand candlestick price chart"}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-cyan-950/90 hover:bg-cyan-900 text-cyan-300 border border-cyan-700/80 font-mono text-xs font-bold min-h-[44px] min-w-[44px] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none transition-colors touch-manipulation cursor-pointer shrink-0"
            >
              <span>📈</span>
              <span>{isChartExpanded ? "Hide Chart ▲" : "View Candlestick Chart ▾"}</span>
            </button>
          </div>

          {/* 4. PRICE / CHART CONTEXT (Full canonical chart) */}
          {chartSlot && (
            <div
              id="market-workspace-chart"
              data-testid="market-workspace-chart"
              className={`${isChartExpanded ? "block" : "hidden md:block"} space-y-2`}
            >
              {chartSlot}
            </div>
          )}

          {/* 🎯 SYNTHESIS E V3: INLINE PRE-FLIGHT CLEARANCE PANEL */}
          {preflightSlot !== undefined ? (
            preflightSlot
          ) : (
            <PreFlightReadinessBar
              symbol={insight.symbol}
              currentPrice={hasValidPrice ? (rawPrice as number) : 0}
              stopLoss={kl.stopLoss}
              takeProfit1={kl.target1}
              riskRewardRatio={kl.profitRiskRatio}
              setupPattern={insight.standard.setupSummary}
              isStage4={insight.standard.setupSummary?.toLowerCase().includes("stage 4") || (insight.advanced as any)?.is_stage4}
              optimalEntryMin={kl.entryMin}
              optimalEntryMax={kl.entryMax}
              isActionable={isActionable}
              decisionState={insight.decisionTrace?.decisionState}
              dataSource={(insight as any).dataSource || (insight as any)._dataSource}
              isHistorical={(insight as any).dataSource === 'historical' || (insight as any)._dataSource === 'historical'}
              isSynthetic={Boolean(isDemo || (insight as any).isSynthetic)}
            />
          )}
        </div>

        {/* 5. CONDITIONAL TRADE PLAN */}
        {planSlot && (
          <div className="order-3 lg:order-3 xl:order-3 lg:col-span-12 xl:col-span-12 space-y-2 w-full">
            {planSlot}
          </div>
        )}
      </div>

      {/* 6. SUPPORTING EVIDENCE (Score Subordinated & Confluence Breakdown) */}
      <div
        data-testid="supporting-evidence"
        className="w-full space-y-4"
      >
        {evidenceSlot ? (
          evidenceSlot
        ) : (
          <div className="bg-[var(--surface-1)] border border-[var(--border-subtle)] rounded-2xl p-4 sm:p-5 shadow-xl space-y-4 font-sans text-xs">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border-subtle)] pb-3">
              <div className="min-w-0 max-w-full">
                <span className="text-xs text-[var(--text-2)] font-mono font-bold uppercase tracking-wider block">
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
                className={`flex items-center gap-3 px-3.5 py-2 bg-[var(--surface-2)] border rounded-xl cursor-pointer transition-all shrink-0 max-w-full min-h-[44px] ${
                  insight.terminalState.overallEligibility !== "ELIGIBLE"
                    ? "border-slate-700 hover:border-slate-500"
                    : "border-[var(--border-subtle)] hover:border-cyan-500"
                }`}
              >
                <div>
                  <span className="text-xs text-[var(--text-2)] font-mono block">Setup Score</span>
                  <span className="text-xs text-[var(--text-muted)] font-mono">{insight.standard.signalsRatio}</span>
                </div>
                <div className="text-right">
                  <span
                    className={`text-xl font-black font-mono ${
                      insight.terminalState.overallEligibility !== "ELIGIBLE"
                        ? "text-[var(--text-muted)]"
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
                <span className="text-[var(--text-2)] font-bold">Confluence Breakdown</span>
                <button
                  id="why-score-btn"
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
                      <span className="text-[var(--text-2)]">{bar.dimension}</span>
                      <span className="text-white font-bold">{bar.score}/100</span>
                    </div>
                    <div className="w-full h-2 bg-[var(--surface-2)] rounded-full overflow-hidden border border-[var(--border-subtle)]">
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
              <div className="p-3 bg-[var(--surface-2)] border border-[var(--border-subtle)] rounded-xl text-xs font-mono space-y-1">
                <span className="text-[var(--text-muted)] uppercase font-bold text-xs block">Setup Structure</span>
                <span className="text-amber-300 font-bold block">{insight.standard.setupSummary}</span>
              </div>

              <div className="p-3 bg-[var(--surface-2)] border border-[var(--border-subtle)] rounded-xl text-xs font-mono space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[var(--text-2)] font-bold">Profit / Risk Ratio:</span>
                  <span className="text-emerald-400 font-bold text-sm">
                    {kl.profitRiskRatio !== undefined ? `${kl.profitRiskRatio.toFixed(2)} : 1.0` : "N/A"}
                  </span>
                </div>
                <div className="flex items-center justify-between text-[var(--text-2)]">
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
                  ⚖️ Size Position (Buy Zone)
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
                  className="w-full py-2.5 bg-slate-800 hover:bg-slate-700 text-[var(--text-1)] font-bold font-mono rounded-xl text-xs transition-all active:scale-95 block text-center shadow-md cursor-pointer min-h-[44px] flex items-center justify-center"
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
        )}
      </div>

      <FinancialDisclaimer variant="compact" />
    </div>
  );
}
