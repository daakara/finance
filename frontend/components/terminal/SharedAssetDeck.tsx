"use client";

import React from "react";
import { QuantitativeInsight } from "../../types/insight";

export interface SharedAssetDeckProps {
  insight: QuantitativeInsight;
  isStaleTape: boolean;
  isActionable: boolean;
  isWaitingPullback: boolean;
  isExtended: boolean;
  isDemo?: boolean;
}

export function SharedAssetDeck({
  insight,
  isStaleTape,
  isActionable,
  isWaitingPullback,
  isExtended,
  isDemo = false,
}: SharedAssetDeckProps) {
  const kl = insight.standard?.keyLevels || (insight as any).keyLevels || {};
  const rawPrice = insight.price || kl.currentPrice;
  const hasValidPrice = typeof rawPrice === "number" && !isNaN(rawPrice) && rawPrice > 0;

  return (
    <div className="deck-top border-b border-[var(--border-subtle)] pb-3 space-y-2 font-sans">
      <div className="symbol-title-group flex flex-wrap items-start justify-between gap-2">
        <div className="space-y-0.5">
          <div className="symbol-row flex items-center gap-2 flex-wrap">
            <span
              id="card-symbol"
              data-testid="card-symbol"
              className="symbol-text mono font-mono font-black text-2xl sm:text-3xl text-white tracking-tight"
            >
              {insight.symbol || "UNAVAILABLE"}
            </span>
            <span
              id="card-exchange"
              data-testid="card-exchange"
              className="macro-item px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[var(--surface-2)] text-[var(--text-2)] border border-[var(--border-subtle)]"
            >
              {(() => {
                const ex = insight.advanced?.exchange || (insight as any)?.exchange;
                return ex && ex !== "UNKNOWN" ? ex : "UNAVAILABLE";
              })()}
            </span>
            <span
              id="card-status-badge"
              data-testid="card-status-badge"
              className={`status-badge px-2.5 py-0.5 rounded text-xs font-mono font-bold border ${
                isStaleTape
                  ? "badge-neutral bg-slate-900/90 text-[var(--text-2)] border-slate-600"
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
                data-testid="demo-asset-tag"
                className="px-2 py-0.5 rounded text-xs font-mono font-bold bg-amber-950/80 text-amber-300 border border-amber-800/80"
              >
                Demo Asset
              </span>
            )}
          </div>
          <span id="card-name" data-testid="card-name" className="asset-name text-xs text-[var(--text-2)] font-sans block">
            {insight.companyName || "Asset Overview"}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span
            data-testid="actionability-badge"
            className={`px-3 py-1 rounded-md text-xs font-mono font-bold border ${
              isStaleTape
                ? "bg-slate-900/90 text-[var(--text-2)] border-slate-700"
                : isActionable
                ? "bg-emerald-950/80 text-emerald-300 border-emerald-700/80"
                : "bg-slate-900/80 text-[var(--text-2)] border-slate-700/80"
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
          <span
            id="card-price"
            data-testid="card-price"
            className="price-text mono font-mono font-black text-2xl sm:text-3xl text-white tabular-nums"
          >
            {hasValidPrice ? `$${(rawPrice as number).toFixed(2)}` : "UNAVAILABLE"}
          </span>
          {hasValidPrice && (
            <span
              id="card-delta"
              data-testid="card-delta"
              className={`delta-text mono font-mono font-bold text-sm tabular-nums flex items-center gap-0.5 ${
                (insight.changePct || 0) >= 0 ? "delta-up text-emerald-400" : "delta-down text-rose-400"
              }`}
            >
              {(insight.changePct || 0) >= 0 ? "+" : ""}$
              {Math.abs(((rawPrice as number) * (insight.changePct || 0)) / 100).toFixed(2)} (
              {(insight.changePct || 0) >= 0 ? "+" : ""}
              {(insight.changePct || 0).toFixed(2)}%)
            </span>
          )}
        </div>
        <div
          id="price-subdeck"
          data-testid="price-subdeck"
          className="price-subdeck flex items-center gap-2 text-[11px] font-mono text-[var(--text-2)] flex-wrap"
          aria-label="Price Authority and Session Metadata"
        >
          <span
            id="price-role"
            data-testid="price-role"
            className={`price-tag px-1.5 py-0.5 rounded font-semibold border ${
              isStaleTape
                ? "bg-slate-800 text-[var(--text-2)] border-slate-600"
                : "bg-[var(--surface-2)] text-cyan-300 border border-[var(--border-subtle)]"
            }`}
          >
            {isStaleTape ? "Historical EOD Reference (>4D Lag)" : "Analysis Input Reference"}
          </span>
          <span>·</span>
          <span id="price-session" data-testid="price-session" className="price-tag">
            {isStaleTape ? "Tape Inactive" : (insight.advanced as any)?.marketSession || "Market Closed"}
          </span>
          <span>·</span>
          <span id="price-asof" data-testid="price-asof" className="price-tag mono" suppressHydrationWarning>
            {insight.terminalState?.modelProvenance?.calculatedAt
              ? `Context generated ${new Date(insight.terminalState.modelProvenance.calculatedAt)
                  .toISOString()
                  .replace("T", " ")
                  .slice(0, 16)} UTC`
              : "Timestamp provenance unavailable"}
          </span>
        </div>
      </div>
    </div>
  );
}

export default SharedAssetDeck;
