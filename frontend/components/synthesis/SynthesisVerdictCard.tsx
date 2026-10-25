"use client";

import React from "react";
import { ShieldAlert, CheckCircle2, AlertTriangle, ArrowUpRight, ArrowDownRight, Layers } from "lucide-react";

export interface SynthesisVerdictCardProps {
  symbol: string;
  companyName: string;
  exchange?: string;
  statusBadge?: string;
  statusBadgeType?: "positive" | "caution" | "negative" | "neutral";
  currentPrice: number;
  changePct: number;
  changeAmount?: number;
  priceRole?: string;
  marketSession?: string;
  asOfTimestamp?: string;
  verdictTitle: string;
  verdictSummary: string;
  setupScore: number;
  isActionable: boolean;
  buyZone: string;
  maxStop: number | null;
  profitTarget: number | null;
  profitRiskRatio: number | null;
  preflightCleared: boolean;
  preflightChecksPassed: number;
  preflightTotalChecks?: number;
  activeBlockerText?: string | null;
  volumeStatement?: string;
  dataFreshnessText?: string;
  onReviewBlockers?: () => void;
  onOpenTradePlan?: () => void;
}

export default function SynthesisVerdictCard({
  symbol,
  companyName,
  exchange = "NASDAQ",
  statusBadge = "IN BUY ZONE",
  statusBadgeType = "positive",
  currentPrice,
  changePct,
  changeAmount,
  priceRole = "Setup Spot Reference",
  marketSession = "Market Closed",
  asOfTimestamp,
  verdictTitle,
  verdictSummary,
  setupScore,
  isActionable,
  buyZone,
  maxStop,
  profitTarget,
  profitRiskRatio,
  preflightCleared,
  preflightChecksPassed,
  preflightTotalChecks = 5,
  activeBlockerText,
  volumeStatement,
  dataFreshnessText = "Verified Production Feed · Fail-Closed Authority",
  onReviewBlockers,
  onOpenTradePlan,
}: SynthesisVerdictCardProps) {
  const isUp = changePct >= 0;
  const computedDeltaAmt = changeAmount !== undefined ? changeAmount : (currentPrice * changePct) / 100;

  const badgeClass =
    statusBadgeType === "positive"
      ? "bg-emerald-950/80 text-emerald-300 border-emerald-700/80"
      : statusBadgeType === "caution"
      ? "bg-amber-950/80 text-amber-300 border-amber-700/80"
      : statusBadgeType === "negative"
      ? "bg-rose-950/80 text-rose-300 border-rose-700/80"
      : "bg-slate-900 text-slate-300 border-slate-700";

  return (
    <article
      id="verdict-card"
      className="w-full bg-[var(--surface-1)] border border-[var(--border-strong)] rounded-2xl p-4 sm:p-5 shadow-xl flex flex-col justify-between space-y-4 font-sans text-[var(--text-1)] transition-colors duration-200"
      aria-label="ARX Bottom Line and Execution Decision"
    >
      {/* 1. Asset Identity Header */}
      <div className="flex flex-wrap items-start justify-between gap-2 border-b border-[var(--border-subtle)] pb-3">
        <div className="flex flex-col">
          <div className="flex items-center gap-2 flex-wrap">
            <span id="card-symbol" className="font-mono font-black text-2xl sm:text-3xl text-white tracking-tight">
              {symbol}
            </span>
            <span id="card-exchange" className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[var(--surface-2)] text-[var(--text-muted)] border border-[var(--border-subtle)]">
              {exchange}
            </span>
            <span id="card-status-badge" className={`px-2.5 py-0.5 rounded text-xs font-mono font-bold border ${badgeClass}`}>
              {statusBadge}
            </span>
          </div>
          <span id="card-name" className="text-xs text-[var(--text-2)] font-sans mt-0.5 font-medium">
            {companyName}
          </span>
        </div>

        {/* Actionability Pill */}
        <div className="flex items-center gap-1.5">
          <span
            data-testid="actionability-badge"
            className={`px-3 py-1 rounded-md text-xs font-mono font-bold border ${
              isActionable
                ? "bg-emerald-950/80 text-emerald-300 border-emerald-700/80"
                : "bg-slate-900/80 text-slate-400 border-slate-700/80"
            }`}
          >
            {isActionable ? "ACTIONABLE" : "NOT ACTIONABLE"}
          </span>
        </div>
      </div>

      {/* 2. Price Line & Subdeck */}
      <div className="space-y-1">
        <div className="flex items-baseline gap-3 flex-wrap">
          <span id="card-price" className="font-mono font-black text-2xl sm:text-3xl tabular-nums text-white">
            ${currentPrice > 0 ? currentPrice.toFixed(2) : "0.00"}
          </span>
          <span
            id="card-delta"
            className={`font-mono font-bold text-sm sm:text-base tabular-nums flex items-center gap-0.5 ${
              isUp ? "text-emerald-400" : "text-rose-400"
            }`}
          >
            {isUp ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
            {isUp ? "+" : ""}${Math.abs(computedDeltaAmt).toFixed(2)} ({isUp ? "+" : ""}{changePct.toFixed(2)}%)
          </span>
        </div>

        <div id="price-subdeck" className="flex items-center gap-2 text-[11px] font-mono text-[var(--text-muted)] flex-wrap pt-0.5" aria-label="Price Authority and Session Metadata">
          <span id="price-role" className="px-1.5 py-0.5 rounded bg-[var(--surface-2)] text-cyan-300 font-semibold border border-[var(--border-subtle)]">
            {priceRole}
          </span>
          <span>·</span>
          <span id="price-session" className="text-[var(--text-2)]">
            {marketSession}
          </span>
          {asOfTimestamp && (
            <>
              <span>·</span>
              <span id="price-asof" className="text-[var(--text-muted)]">
                {asOfTimestamp}
              </span>
            </>
          )}
        </div>
      </div>

      {/* 3. Verdict Banner */}
      <div className="p-3 sm:p-4 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)] space-y-2">
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-0.5 min-w-0">
            <span id="card-verdict-label" className="text-[10px] font-mono font-black text-cyan-400 uppercase tracking-widest block">
              ARX BOTTOM LINE VERDICT
            </span>
            <h3 id="card-verdict-title" className="text-base sm:text-lg font-black text-white tracking-tight leading-snug">
              {verdictTitle}
            </h3>
          </div>

          {/* Setup Score Pill */}
          <div id="card-score-pill" className="shrink-0 px-3 py-1.5 rounded-xl bg-[var(--surface-elevated)] border border-cyan-500/40 text-center shadow-md">
            <span id="card-score-num" className="font-mono font-black text-lg text-cyan-400 leading-none">
              {setupScore}
            </span>
            <span id="card-score-den" className="font-mono text-xs text-[var(--text-muted)]">
              /100
            </span>
          </div>
        </div>

        <p id="card-verdict-summary" className="text-xs text-[var(--text-2)] font-sans leading-relaxed">
          {verdictSummary}
        </p>
      </div>

      {/* 4. Compact Execution Ladder */}
      <div id="card-middle-panel" className="space-y-2">
        <div className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider flex items-center justify-between">
          <span>Execution Boundaries & Risk Ratios</span>
          {onOpenTradePlan && (
            <button
              type="button"
              onClick={onOpenTradePlan}
              className="text-cyan-400 hover:underline cursor-pointer"
            >
              Open Trade Plan →
            </button>
          )}
        </div>

        <div id="middle-panel-content" className="grid grid-cols-2 gap-2 text-xs font-mono">
          <div className="p-2.5 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)]">
            <span className="text-[10px] text-[var(--text-muted)] block uppercase font-bold">Buy Zone</span>
            <span className="text-amber-300 font-bold truncate block mt-0.5">{buyZone || "Awaiting Setup"}</span>
          </div>
          <div className="p-2.5 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)]">
            <span className="text-[10px] text-[var(--text-muted)] block uppercase font-bold">Reward : Risk</span>
            <span className="text-emerald-400 font-bold block mt-0.5">
              {profitRiskRatio ? `${profitRiskRatio.toFixed(2)} : 1.0` : "N/A"}
            </span>
          </div>
          <div className="p-2.5 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)]">
            <span className="text-[10px] text-[var(--text-muted)] block uppercase font-bold">Profit Target</span>
            <span className="text-emerald-400 font-bold block mt-0.5">
              {profitTarget ? `$${profitTarget.toFixed(2)}` : "N/A"}
            </span>
          </div>
          <div className="p-2.5 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)]">
            <span className="text-[10px] text-[var(--text-muted)] block uppercase font-bold">Max Stop Loss</span>
            <span className="text-rose-400 font-bold block mt-0.5">
              {maxStop ? `$${maxStop.toFixed(2)}` : "N/A"}
            </span>
          </div>
        </div>
      </div>

      {/* 5. Left Pre-Flight Indicator */}
      <div
        id="left-preflight-indicator"
        className="p-3 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)] flex items-center justify-between gap-3 text-xs font-mono"
        role="region"
        aria-label="Pre-Flight Clearance Indicator"
      >
        <div className="flex items-center gap-2.5 min-w-0">
          {preflightCleared ? (
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
          ) : (
            <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0" />
          )}
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span
                id="compact-preflight-badge"
                className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  preflightCleared
                    ? "bg-emerald-950 text-emerald-300 border border-emerald-800"
                    : "bg-amber-950 text-amber-300 border border-amber-800"
                }`}
              >
                {preflightCleared ? `CLEARED (${preflightChecksPassed}/${preflightTotalChecks})` : `NOT CLEARED (${preflightChecksPassed}/${preflightTotalChecks})`}
              </span>
            </div>
            <p id="compact-preflight-blocker" className="text-[11px] text-[var(--text-2)] truncate mt-0.5">
              {activeBlockerText ? `Blocker: ${activeBlockerText}` : "All 5 gates verified"}
            </p>
          </div>
        </div>

        {onReviewBlockers && (
          <button
            id="review-preflight-btn"
            type="button"
            onClick={onReviewBlockers}
            className="px-2.5 py-1.5 rounded-lg bg-[var(--surface-elevated)] hover:bg-[var(--surface-hover)] border border-[var(--border-strong)] text-cyan-400 font-bold text-xs shrink-0 transition-colors cursor-pointer min-h-[44px] flex items-center justify-center"
          >
            Review →
          </button>
        )}
      </div>

      {/* 6. Footer Disclosures & Volume */}
      <div className="pt-2 border-t border-[var(--border-subtle)] flex items-center justify-between text-[10px] font-mono text-[var(--text-muted)] flex-wrap gap-2">
        <span id="card-data-freshness">{dataFreshnessText}</span>
        {volumeStatement && <span id="card-volume-statement">{volumeStatement}</span>}
      </div>
    </article>
  );
}
