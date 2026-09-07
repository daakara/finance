"use client";

import React from "react";
import { TickerCommandStripProps } from "../../types/workstation";
import { useExperienceStore } from "../../state/experience-store";
import SetupScoreBadge from "./SetupScoreBadge";
import ExecutionStateBadge from "./ExecutionStateBadge";
import LiquidityBadge from "./LiquidityBadge";

export function TickerCommandStripSkeleton() {
  return (
    <header
      role="region"
      aria-label="Ticker Command Strip Loading"
      data-testid="ticker-command-strip-skeleton"
      className="min-h-[110px] lg:h-[110px] lg:max-h-[110px] w-full border-b border-[#243044] bg-[#0c1017]/95 backdrop-blur flex items-center overflow-hidden"
    >
      <div className="max-w-[1750px] mx-auto px-4 lg:px-6 w-full flex flex-col lg:flex-row lg:items-center justify-between gap-4 py-3">
        {/* Left identity skeleton */}
        <div className="flex items-center gap-6">
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-3">
              <div className="h-7 w-20 bg-slate-800/80 rounded animate-pulse" />
              <div className="h-5 w-44 bg-slate-800/80 rounded animate-pulse" />
            </div>
            <div className="h-4 w-32 bg-slate-800/80 rounded animate-pulse" />
          </div>
          <div className="h-10 w-px bg-slate-800 hidden sm:block" />
          <div className="flex flex-col gap-1.5">
            <div className="h-7 w-28 bg-slate-800/80 rounded animate-pulse" />
            <div className="h-4 w-20 bg-slate-800/80 rounded animate-pulse" />
          </div>
        </div>

        {/* Right metrics skeleton */}
        <div className="flex items-center gap-3 sm:gap-4">
          <div className="h-14 w-36 bg-slate-800/80 rounded-lg animate-pulse" />
          <div className="h-8 w-28 bg-slate-800/80 rounded animate-pulse" />
          <div className="h-8 w-24 bg-slate-800/80 rounded animate-pulse hidden sm:block" />
        </div>
      </div>
    </header>
  );
}

export default function TickerCommandStrip({
  ticker,
  companyName,
  spotPrice,
  priceChangePct,
  priceChange,
  setupScore,
  domainConfidence = "HIGH",
  executionState,
  liquidityTier = "HIGH",
  marketRegime = "RISK_ON",
  isSettlementPinned = false,
  marketSession = "CLOSED",
  sector,
  exchange,
  amihudScore,
  mode: propMode,
  className = "",
}: TickerCommandStripProps) {
  // Read store mode if prop not provided
  const storeMode = useExperienceStore((state) => state.mode);
  const activeMode = propMode || storeMode || "STANDARD";

  // Calculate price change if not explicitly provided
  const computedPriceChange =
    priceChange !== undefined
      ? priceChange
      : (spotPrice * priceChangePct) / 100;

  const isPositive = priceChangePct > 0;
  const isNegative = priceChangePct < 0;

  const priceColorClass = isPositive
    ? "text-emerald-400"
    : isNegative
    ? "text-rose-400"
    : "text-slate-300";

  const changePrefix = isPositive ? "+" : "";

  // Settlement banner condition: market closed or settlement explicitly pinned
  const showSettlementNotice = isSettlementPinned || marketSession === "CLOSED";

  return (
    <header
      role="region"
      aria-label={`Stage 1 Orientation: ${ticker} Command Strip`}
      data-testid="ticker-command-strip"
      className={`min-h-[110px] lg:h-[110px] lg:max-h-[110px] w-full border-b border-[#243044] bg-[#0c1017]/95 backdrop-blur flex items-center overflow-hidden relative z-10 ${className}`}
    >
      <div className="max-w-[1750px] mx-auto px-4 lg:px-6 w-full flex flex-col lg:flex-row lg:items-center justify-between gap-3 lg:gap-4 py-2.5">
        {/* LEFT COLUMN: Asset Identity & Spot Price Cluster */}
        <div className="flex flex-wrap items-center gap-4 sm:gap-6 min-w-0">
          {/* Identity Block */}
          <div className="flex flex-col min-w-0">
            <div className="flex items-center gap-2.5 flex-wrap">
              <span
                data-testid="ticker-symbol"
                className="text-2xl sm:text-3xl font-mono font-extrabold text-slate-50 tracking-tight"
              >
                {ticker}
              </span>
              <span
                data-testid="company-name"
                className="text-sm sm:text-base font-medium text-slate-300 truncate max-w-[200px] sm:max-w-[320px]"
                title={companyName}
              >
                {companyName}
              </span>
            </div>

            {/* Sub-identity Metadata (Exchange, Sector) */}
            <div className="flex items-center gap-2 text-xs text-slate-400 font-mono mt-0.5">
              {exchange && (
                <span
                  data-testid="ticker-exchange"
                  className="px-1.5 py-0.2 rounded bg-slate-800/80 text-slate-300 border border-slate-700/60 text-[10px]"
                >
                  {exchange}
                </span>
              )}
              {sector && (
                <span data-testid="ticker-sector" className="truncate text-slate-400">
                  {sector}
                </span>
              )}
            </div>
          </div>

          {/* Institutional Divider */}
          <div className="h-9 w-px bg-slate-800 hidden sm:block" />

          {/* Spot Price & 24h Delta Block */}
          <div className="flex flex-col">
            <div className="flex items-baseline gap-2">
              <span
                data-testid="spot-price"
                className="text-xl sm:text-2xl font-mono font-bold text-slate-100 tabular-nums"
              >
                ${spotPrice.toFixed(2)}
              </span>
              <span
                data-testid="price-delta"
                className={`text-xs sm:text-sm font-mono font-semibold tabular-nums ${priceColorClass}`}
              >
                {changePrefix}
                {computedPriceChange.toFixed(2)} ({changePrefix}
                {priceChangePct.toFixed(2)}%)
              </span>
            </div>

            {/* Pinned Settlement Banner / Market Session Notice */}
            {showSettlementNotice && (
              <div
                data-testid="settlement-pinned-notice"
                className="flex items-center gap-1.5 text-[10px] font-mono text-amber-400/90 mt-0.5"
              >
                <svg
                  className="w-3 h-3 shrink-0 text-amber-400"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                  <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                </svg>
                <span>[Session Closed / Friday Settlement Pinned]</span>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT COLUMN: Stage 1 Decision Orientation Metrics */}
        <div className="flex flex-wrap items-center gap-2.5 sm:gap-3 shrink-0">
          {/* Setup Score Gauge Badge */}
          <SetupScoreBadge
            score={setupScore}
            domainConfidence={domainConfidence}
            mode={activeMode}
          />

          {/* Execution State Badge */}
          <ExecutionStateBadge state={executionState} mode={activeMode} />

          {/* Liquidity Tier Badge */}
          <LiquidityBadge
            tier={liquidityTier}
            amihudScore={amihudScore}
            mode={activeMode}
          />

          {/* Quant Mode Only: Market Regime Indicator */}
          {activeMode === "QUANT" && (
            <div
              data-testid="quant-market-regime-badge"
              className={`hidden sm:inline-flex items-center gap-1 px-2 py-1 rounded text-[10px] font-mono border ${
                marketRegime === "RISK_ON"
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                  : marketRegime === "NEUTRAL"
                  ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                  : "bg-rose-500/10 text-rose-400 border-rose-500/30"
              }`}
            >
              <span>REGIME: {marketRegime}</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
