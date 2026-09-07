"use client";

import React, { useState, useEffect } from "react";
import { CandleData } from "../../lib/api";
import { ExperienceMode } from "../../types/insight";
import { trackTelemetryEvent } from "../../telemetry/tracker";
import PriceChart from "../PriceChart";

export interface PriceChartWorkspaceProps {
  ticker: string;
  spotPrice: number;
  priceChangePct: number;
  entryLow?: number;
  entryHigh?: number;
  stopLoss?: number;
  target1?: number;
  candles?: CandleData[];
  interval?: string;
  onIntervalChange?: (interval: string) => void;
  atr?: number;
  mode?: ExperienceMode;
  className?: string;
}

const DEFAULT_INTERVALS = ["1D", "1W", "1M", "1Y"];

export default function PriceChartWorkspace({
  ticker,
  spotPrice,
  priceChangePct,
  entryLow,
  entryHigh,
  stopLoss,
  target1,
  candles = [],
  interval = "1y_hist",
  onIntervalChange,
  atr = 0.74,
  mode = "STANDARD",
  className = "",
}: PriceChartWorkspaceProps) {
  const [activeInterval, setActiveInterval] = useState<string>(interval);

  const handleIntervalClick = (int: string) => {
    setActiveInterval(int);
    if (onIntervalChange) onIntervalChange(int);
    trackTelemetryEvent("ORIENTATION", "chart_viewed", { ticker, timeframe: int }, ticker);
  };

  useEffect(() => {
    trackTelemetryEvent("ORIENTATION", "chart_viewed", { ticker, timeframe: activeInterval }, ticker);
  }, [ticker, activeInterval]);

  return (
    <div
      data-testid="price-chart-workspace"
      className={`flex flex-col w-full h-full min-h-[420px] lg:min-h-[540px] bg-[#0c1017] border border-[#243044] rounded-xl overflow-hidden shadow-sm ${className}`}
    >
      {/* Chart Top Toolbar */}
      <div className="h-12 px-4 border-b border-slate-800 bg-[#0e1422] flex items-center justify-between shrink-0">
        {/* Left: Ticker & ATR Indicator */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="font-mono font-bold text-sm text-slate-100">{ticker}</span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
              Interactive Chart
            </span>
          </div>

          {atr > 0 && (
            <div className="hidden sm:flex items-center gap-1 text-[11px] font-mono text-slate-400">
              <span>ATR(14):</span>
              <span className="text-slate-200 font-semibold">${atr.toFixed(2)}</span>
            </div>
          )}
        </div>

        {/* Right: Timeframe Interval Switcher */}
        <div
          role="tablist"
          aria-label="Chart timeframes"
          className="flex items-center gap-1 bg-[#141b2d] p-0.5 rounded-lg border border-slate-800"
        >
          {DEFAULT_INTERVALS.map((int) => {
            const isSelected = activeInterval.toUpperCase().startsWith(int);
            return (
              <button
                key={int}
                type="button"
                role="tab"
                aria-selected={isSelected}
                onClick={() => handleIntervalClick(int.toLowerCase())}
                data-testid={`chart-interval-${int.toLowerCase()}`}
                className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                  isSelected
                    ? "bg-slate-700 text-slate-100 font-bold shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                }`}
              >
                {int}
              </button>
            );
          })}
        </div>
      </div>

      {/* Primary Chart Canvas Container */}
      <div className="flex-1 w-full relative min-h-[380px] bg-[#090d14]">
        {candles && candles.length > 0 ? (
          <PriceChart
            symbol={ticker}
            candles={candles}
            currentPrice={spotPrice}
            priceChangePct={priceChangePct}
            interval={activeInterval}
            onIntervalChange={handleIntervalClick}
            technicals={{ atr_14: atr }}
          />
        ) : (
          /* High-fidelity institutional fallback SVG canvas if raw historical candles are loading */
          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 space-y-3">
            <svg
              className="w-full h-48 text-emerald-500/20 max-w-lg"
              viewBox="0 0 400 120"
              fill="none"
              stroke="currentColor"
            >
              {/* Buy zone corridor shaded background */}
              {entryLow && entryHigh && (
                <rect x="50" y="45" width="300" height="25" fill="#10b981" fillOpacity="0.08" />
              )}
              {/* Trendline */}
              <path
                d="M 10 90 Q 90 85 160 55 T 260 48 T 390 30"
                stroke="#10b981"
                strokeWidth="2.5"
                fill="none"
              />
              {/* Invalidation dashed stop floor */}
              {stopLoss && (
                <line x1="10" y1="95" x2="390" y2="95" stroke="#f43f5e" strokeWidth="1.5" strokeDasharray="4 4" />
              )}
            </svg>
            <span className="text-xs font-mono text-slate-400">
              Interactive 65% TradingView Viewport Anchored ({ticker})
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
