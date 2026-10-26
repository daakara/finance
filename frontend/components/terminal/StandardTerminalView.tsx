"use client";

import React from "react";
import Link from "next/link";
import { QuantitativeInsight } from "../../types/insight";
import FinancialDisclaimer from "../FinancialDisclaimer";
import { deriveUnmetConditions } from "../../lib/decisionHierarchyUtils";
import PreFlightReadinessBar from "../PreFlightReadinessBar";

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
    <div className="space-y-4 font-sans text-slate-100 animate-fade-in">
      {/* 🎯 FIRST-VIEWPORT COMPOSITION: VERDICT (LEFT 5-COL) + CHART (RIGHT 7-COL) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 xl:grid-cols-12 gap-4 items-start">
        {/* Left Column (lg:col-span-5 xl:col-span-5): Verdict Card & Preconditions */}
        <div className="lg:col-span-5 xl:col-span-5 space-y-2.5 sm:space-y-4 min-w-0 order-1 lg:order-1 xl:order-1">
          {/* 1. DECISION VERDICT (Dominant Visual Weight) & 2. DECISION REASON */}
          <article
            id="verdict-card"
            data-testid="decision-verdict"
            className="verdict-card bg-[#0b101b] border border-[#1d293d] rounded-2xl p-3 sm:p-5 shadow-xl space-y-3 font-sans text-slate-100"
            aria-label="ARX Bottom Line and Execution Decision"
          >
            {/* Ticker & Price Deck */}
            <div className="deck-top border-b border-[#182335] pb-3 space-y-2">
              <div className="symbol-title-group flex flex-wrap items-start justify-between gap-2">
                <div className="space-y-0.5">
                  <div className="symbol-row flex items-center gap-2 flex-wrap">
                    <span id="card-symbol" className="symbol-text mono font-mono font-black text-2xl sm:text-3xl text-white tracking-tight">
                      {insight.symbol || "UNAVAILABLE"}
                    </span>
                    <span id="card-exchange" className="macro-item px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[#141c2e] text-slate-400 border border-[#2b3a52]">
                      {(insight.advanced as any)?.exchange || (insight as any)?.exchange || "EXCHANGE UNAVAILABLE"}
                    </span>
                    <span
                      id="card-status-badge"
                      className={`status-badge px-2.5 py-0.5 rounded text-xs font-mono font-bold border ${
                        isStaleTape
                          ? "badge-neutral bg-slate-900/90 text-slate-300 border-slate-600"
                          : isActionable
                          ? "badge-positive bg-emerald-950/80 text-emerald-300 border-emerald-700/80"
                          : "badge-caution bg-amber-950/80 text-amber-300 border-amber-700/80"
                      }`}
                    >
                      {isStaleTape
                        ? "HISTORICAL DATA ONLY"
                        : isActionable
                        ? "IN BUY ZONE"
                        : isWaitingPullback
                        ? "WAITING PULLBACK"
                        : isExtended
                        ? "EXTENDED"
                        : "AWAITING SETUP"}
                    </span>
                    {isDemo && (
                      <span
                        id="demo-asset-tag"
                        className="px-2 py-0.5 rounded text-xs font-mono font-bold bg-amber-950/80 text-amber-300 border border-amber-800/80"
                      >
                        Demo Asset
                      </span>
                    )}
                  </div>
                  <span id="card-name" className="asset-name text-xs text-slate-400 font-sans block">
                    {insight.companyName || "Asset Overview"}
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <span
                    data-testid="actionability-badge"
                    className={`px-3 py-1 rounded-md text-xs font-mono font-bold border ${
                      isStaleTape
                        ? "bg-slate-900/90 text-slate-400 border-slate-700"
                        : isActionable
                        ? "bg-emerald-950/80 text-emerald-300 border-emerald-700/80"
                        : "bg-slate-900/80 text-slate-400 border-slate-700/80"
                    }`}
                  >
                    {isStaleTape
                      ? "NOT ACTIONABLE (STALE TAPE)"
                      : isActionable
                      ? "ACTIONABLE"
                      : "NOT ACTIONABLE"}
                  </span>
                </div>
              </div>

              {/* Price Line & Subdeck */}
              <div className="price-group space-y-1">
                <div className="price-line flex items-baseline gap-3 flex-wrap">
                  <span id="card-price" className="price-text mono font-mono font-black text-2xl sm:text-3xl text-white tabular-nums">
                    {hasValidPrice ? `$${(rawPrice as number).toFixed(2)}` : "UNAVAILABLE"}
                  </span>
                  {hasValidPrice && (
                    <span
                      id="card-delta"
                      className={`delta-text mono font-mono font-bold text-sm tabular-nums flex items-center gap-0.5 ${
                        (insight.changePct || 0) >= 0 ? "delta-up text-emerald-400" : "delta-down text-rose-400"
                      }`}
                    >
                      {(insight.changePct || 0) >= 0 ? "+" : ""}${Math.abs(((rawPrice as number) * (insight.changePct || 0)) / 100).toFixed(2)} ({(insight.changePct || 0) >= 0 ? "+" : ""}{(insight.changePct || 0).toFixed(2)}%)
                    </span>
                  )}
                </div>
                <div id="price-subdeck" className="price-subdeck flex items-center gap-2 text-[11px] font-mono text-slate-400 flex-wrap" aria-label="Price Authority and Session Metadata">
                  <span id="price-role" className={`price-tag px-1.5 py-0.5 rounded font-semibold border ${
                    isStaleTape
                      ? "bg-slate-800 text-slate-300 border-slate-600"
                      : "bg-[#141c2e] text-cyan-300 border border-[#2b3a52]"
                  }`}>
                    {isStaleTape ? "Historical EOD Reference (>4D Lag)" : "Setup Spot Reference"}
                  </span>
                  <span>·</span>
                  <span id="price-session" className="price-tag">
                    {isStaleTape ? "Tape Inactive" : ((insight.advanced as any)?.marketSession || "Market Closed")}
                  </span>
                  <span>·</span>
                  <span id="price-asof" className="price-tag mono">
                    {insight.terminalState.modelProvenance?.calculatedAt
                      ? `As of ${new Date(insight.terminalState.modelProvenance.calculatedAt).toISOString().replace("T", " ").slice(0, 16)} UTC`
                      : "Timestamp provenance unavailable"}
                  </span>
                </div>
              </div>
            </div>

            {/* Verdict & Score Banner */}
            <div className="verdict-banner !flex !flex-col p-3 sm:p-4 rounded-xl bg-[#080e18] border border-[#182335] space-y-2.5">
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
                    className="score-pill px-3 py-1 rounded-xl bg-[#141c2e] border border-cyan-500/40 text-center shadow-md cursor-pointer hover:border-cyan-400 transition-all"
                  >
                    <span id="card-score-num" className="score-num mono font-mono font-black text-base text-cyan-400 leading-none">
                      {insight.setupScore}
                    </span>
                    <span id="card-score-den" className="score-den font-mono text-xs text-slate-400">
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
                className="text-xs text-slate-300 font-sans leading-relaxed border-t border-[#182335]/60 pt-2 w-full"
              >
                <span className="text-[10px] text-slate-400 font-mono font-bold block uppercase mb-0.5 tracking-wider">
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
                className="sm:hidden flex items-center justify-between pt-2 border-t border-[#182335]/60 w-full"
              >
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
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
                    className="px-2.5 py-0.5 rounded-lg bg-[#141c2e] border border-cyan-500/40 text-center cursor-pointer flex items-baseline gap-0.5"
                  >
                    <span className="score-num mono font-mono font-black text-sm text-cyan-400 leading-none">
                      {insight.setupScore}
                    </span>
                    <span className="score-den font-mono text-[10px] text-slate-400">
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
            <div id="card-middle-panel" className="middle-panel space-y-2 pt-1">
              <div className="panel-head flex items-center justify-between text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wider">
                <span id="middle-panel-title">
                  {isStaleTape ? "Execution Ladder (Historical Tape)" : "Execution Ladder"}
                </span>
                <span id="middle-panel-sub" className="mono text-cyan-300">
                  R:R {kl.profitRiskRatio ? `${kl.profitRiskRatio.toFixed(2)} : 1.0` : "N/A"}
                </span>
              </div>
              <div id="middle-panel-content" data-testid="cockpit-execution-ladder" className="ladder-grid grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                <div className="ladder-cell min-w-0 p-2.5 rounded-xl bg-[#070b13] border border-[#1b2639]">
                  <span className="cell-lbl text-[10px] text-slate-400 block uppercase font-bold">
                    {isStaleTape ? "Historical Pivot" : "Pivot"}
                  </span>
                  <span id="cell-pivot" className="cell-val mono text-white font-bold block mt-0.5 whitespace-normal break-words text-xs leading-tight tabular-nums">
                    {pivotPrice != null ? `$${pivotPrice.toFixed(2)}` : "N/A"}
                  </span>
                </div>
                <div className="ladder-cell min-w-0 p-2.5 rounded-xl bg-[#070b13] border border-[#1b2639]">
                  <span className="cell-lbl text-[10px] text-slate-400 block uppercase font-bold">
                    {isStaleTape ? "Historical Zone" : "Buy Zone"}
                  </span>
                  <span id="cell-entry" className="cell-val mono text-amber-300 font-bold block mt-0.5 whitespace-normal break-words text-xs leading-tight">
                    {kl.entryMin != null && kl.entryMax != null ? (
                      <span className="flex flex-col sm:inline">
                        <span className="tabular-nums">${kl.entryMin.toFixed(2)}</span>
                        <span className="text-slate-400 font-normal text-[10px] sm:text-xs px-0.5">to</span>
                        <span className="tabular-nums">${kl.entryMax.toFixed(2)}</span>
                      </span>
                    ) : (
                      kl.watchZone || "Awaiting Setup"
                    )}
                  </span>
                </div>
                <div className="ladder-cell min-w-0 p-2.5 rounded-xl bg-[#070b13] border border-[#1b2639]">
                  <span className="cell-lbl text-[10px] text-slate-400 block uppercase font-bold">
                    {isStaleTape ? "Historical Stop" : "Stop Loss"}
                  </span>
                  <span id="cell-stop" className="cell-val negative mono text-rose-400 font-bold block mt-0.5 whitespace-normal break-words text-xs leading-tight">
                    {kl.stopLoss ? (
                      <span className="flex flex-col sm:inline">
                        <span className="tabular-nums">${kl.stopLoss.toFixed(2)}</span>
                        {pivotPrice != null && pivotPrice > 0 ? (
                          <span className="text-rose-400/80 font-normal text-[10px] px-0.5">
                            (-{Math.abs(((pivotPrice - kl.stopLoss) / pivotPrice) * 100).toFixed(1)}% base)
                          </span>
                        ) : null}
                      </span>
                    ) : "N/A"}
                  </span>
                </div>
                <div className="ladder-cell min-w-0 p-2.5 rounded-xl bg-[#070b13] border border-[#1b2639]">
                  <span className="cell-lbl text-[10px] text-slate-400 block uppercase font-bold">
                    {isStaleTape ? "Historical Target" : "Target 1"}
                  </span>
                  <span id="cell-target" className="cell-val positive mono text-emerald-400 font-bold block mt-0.5 whitespace-normal break-words text-xs leading-tight">
                    {kl.target1 ? (
                      <span className="flex flex-col sm:inline">
                        <span className="tabular-nums">${kl.target1.toFixed(2)}</span>
                        {pivotPrice != null && pivotPrice > 0 ? (
                          <span className="text-emerald-400/80 font-normal text-[10px] px-0.5">
                            (+{Math.abs(((kl.target1 - pivotPrice) / pivotPrice) * 100).toFixed(1)}%)
                          </span>
                        ) : null}
                      </span>
                    ) : "N/A"}
                  </span>
                </div>
              </div>
            </div>

            {/* Position Sizing & Risk Presentation */}
            <div id="card-sizing-bar" className="sizing-bar border-t border-[#182335] pt-2 space-y-1.5 text-xs font-mono">
              <div className="sizing-meta flex items-center justify-between text-slate-300">
                <span id="sizing-line-1">
                  Position Size:{" "}
                  {isStaleTape ? (
                    <span id="sizing-shares" className="text-slate-500 font-bold">
                      Execution Prohibited (Stale Tape)
                    </span>
                  ) : (
                    <button
                      type="button"
                      onClick={onOpenSizer}
                      id="sizing-shares"
                      className="sizing-bold mono font-bold text-cyan-400 hover:underline cursor-pointer inline-flex items-center gap-1"
                      title="Open Position Sizer to configure shares"
                    >
                      {isActionable ? "Configure Shares" : "0 Shares (Entry Unconfirmed)"}
                    </button>
                  )}
                </span>
                <span id="sizing-max-allowance" className="sizing-val text-slate-400 text-[11px]">
                  Risk Budget: User Configured
                </span>
              </div>
              <div className="sizing-matrix-grid grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px]">
                <div className="sizing-cell p-2.5 rounded-lg bg-[#070b13] border border-[#1b2639] space-y-1">
                  <div className="flex items-center justify-between text-[10px] text-slate-400 uppercase font-bold">
                    <span className="sizing-lbl">Stop Distance & Risk Basis</span>
                    <span className={`px-1 rounded text-[9px] font-mono ${
                      isStaleTape
                        ? "bg-slate-800 text-slate-400 border border-slate-700"
                        : isExtended
                        ? "bg-amber-950/80 text-amber-300 border border-amber-800/80"
                        : isInBuyZone
                        ? "bg-emerald-950/80 text-emerald-300 border border-emerald-800/80"
                        : "bg-[#141c2e] text-slate-400"
                    }`}>
                      {isStaleTape ? "Historical Tape" : isExtended ? "Extended (High Risk)" : isInBuyZone ? "In Buy Zone" : "Awaiting Setup"}
                    </span>
                  </div>
                  <div id="sizing-stop-loss" className="sizing-val negative mono font-bold text-rose-400 block text-[11px] leading-tight space-y-0.5">
                    {kl.stopLoss && hasValidPrice ? (
                      <>
                        <div className="flex items-baseline justify-between gap-1">
                          <span className="text-[10px] text-slate-400 font-normal">From Current Spot (${(rawPrice as number).toFixed(2)}):</span>
                          <span className="tabular-nums">
                            -${Math.abs((rawPrice as number) - kl.stopLoss).toFixed(2)} (-{Math.abs((((rawPrice as number) - kl.stopLoss) / (rawPrice as number)) * 100).toFixed(2)}%)
                          </span>
                        </div>
                        {pivotPrice != null && pivotPrice > 0 && Math.abs((rawPrice as number) - pivotPrice) > 0.05 && (
                          <div className="flex items-baseline justify-between gap-1 text-[10px] text-slate-400 font-normal border-t border-[#182335] pt-0.5">
                            <span>Planned from Pivot (${pivotPrice.toFixed(2)}):</span>
                            <span className="text-slate-300 font-mono tabular-nums">
                              -${Math.abs(pivotPrice - kl.stopLoss).toFixed(2)} (-{Math.abs(((pivotPrice - kl.stopLoss) / pivotPrice) * 100).toFixed(2)}%)
                            </span>
                          </div>
                        )}
                      </>
                    ) : "N/A"}
                  </div>
                </div>
                <div className="sizing-cell p-2.5 rounded-lg bg-[#070b13] border border-[#1b2639] space-y-1">
                  <div className="flex items-center justify-between text-[10px] text-slate-400 uppercase font-bold">
                    <span className="sizing-lbl">Budget Utilization</span>
                    <span className="text-[9px] font-mono text-cyan-400">
                      {isActionable ? "Authorized" : "Suppressed"}
                    </span>
                  </div>
                  <span id="sizing-utilization" className="sizing-val mono font-bold text-slate-300 block text-xs">
                    {isStaleTape
                      ? "Execution Prohibited (Stale Tape)"
                      : !isActionable
                      ? "0% (Awaiting Execution Clearance)"
                      : "100% of Risk Allowance"}
                  </span>
                </div>
              </div>
            </div>
          </article>

          {/* 3. WHAT NEEDS TO CHANGE (Unmet Conditions & Preconditions) */}
          <div
            data-testid="unmet-condition"
            className="bg-[#080e18] border border-cyan-900/50 rounded-2xl p-2.5 sm:p-5 shadow-lg space-y-2 sm:space-y-3"
          >
            <div
              className="flex flex-wrap items-center justify-between gap-1.5 sm:gap-2 border-b border-[#182335] pb-2 cursor-pointer sm:cursor-default"
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
                <span>What Needs to Change (Execution Preconditions)</span>
              </h3>
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] sm:text-xs text-slate-400 font-mono">
                  {isActionable ? "Preconditions Cleared" : "Awaiting Confirmation"}
                </span>
                <span className="text-slate-400 text-xs" aria-hidden="true">
                  {isPreconditionsOpen ? "▲" : "▼"}
                </span>
              </div>
            </div>

            <div className={`${isPreconditionsOpen ? "grid" : "hidden"} grid-cols-1 md:grid-cols-2 lg:grid-cols-1 xl:grid-cols-1 gap-1.5 sm:gap-2.5 text-xs`}>
              {unmetConditions.map((cond) => (
                <div
                  key={cond.id}
                  className="py-1.5 px-2.5 sm:p-3 rounded-lg sm:rounded-xl bg-[#060b13] border border-[#1b2639] space-y-1"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-bold text-slate-200 text-xs truncate">{cond.title}</span>
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
                  <p className="text-xs text-slate-400 leading-relaxed font-sans hidden sm:block">
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
            className="md:hidden bg-[#0e1420] border border-[#1e2a3c] rounded-xl px-3 py-2 flex items-center justify-between gap-2 shadow-md min-h-[56px] mb-2"
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
                    className="text-slate-400 font-mono text-xs font-semibold"
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
                  className="text-[10px] font-mono text-slate-400 bg-slate-900 border border-slate-800 px-1.5 py-0.5 rounded"
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
          <div className="bg-[#0b101b] border border-[#1d293d] rounded-2xl p-4 sm:p-5 shadow-xl space-y-4 font-sans text-xs">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#182335] pb-3">
              <div className="min-w-0 max-w-full">
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
                className={`flex items-center gap-3 px-3.5 py-2 bg-[#06090f] border rounded-xl cursor-pointer transition-all shrink-0 max-w-full min-h-[44px] ${
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
        )}
      </div>

      <FinancialDisclaimer variant="compact" />
    </div>
  );
}
