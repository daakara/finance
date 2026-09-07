"use client";

import React, { useEffect } from "react";
import { ExecutionState } from "../../types/workstation";
import { ExperienceMode } from "../../types/insight";
import ExecutionStateBadge from "../command-strip/ExecutionStateBadge";
import { trackTelemetryEvent } from "../../telemetry/tracker";
import { useExperienceStore } from "../../state/experience-store";

export interface ExecutionCorridorProps {
  ticker: string;
  spotPrice: number;
  entryLow: number;
  entryHigh: number;
  stopLoss: number;
  target1: number;
  target2: number;
  riskRewardRatio: number;
  executionState: ExecutionState;
  advShareLimit: number;
  onOpenPositionSizer: () => void;
  mode?: ExperienceMode;
  className?: string;
}

export default function ExecutionCorridor({
  ticker,
  spotPrice,
  entryLow,
  entryHigh,
  stopLoss,
  target1,
  target2,
  riskRewardRatio,
  executionState,
  advShareLimit,
  onOpenPositionSizer,
  mode: propMode,
  className = "",
}: ExecutionCorridorProps) {
  const storeMode = useExperienceStore((state) => state.mode);
  const activeMode = propMode || storeMode || "STANDARD";

  // Calculate percentage gains/risks relative to current spot price or mid entry
  const entryMid = (entryLow + entryHigh) / 2;
  const target1Pct = ((target1 - spotPrice) / spotPrice) * 100;
  const target2Pct = ((target2 - spotPrice) / spotPrice) * 100;
  const stopLossPct = ((stopLoss - spotPrice) / spotPrice) * 100;

  const isInBuyZone = spotPrice >= entryLow && spotPrice <= entryHigh;

  // Emit telemetry event on mount
  useEffect(() => {
    trackTelemetryEvent(
      "DECISION",
      "execution_corridor_viewed",
      {
        ticker,
        state: executionState,
        spotPrice,
        entryLow,
        entryHigh,
        stopLoss,
        target1,
        riskRewardRatio,
      },
      ticker
    );
  }, [ticker, executionState, spotPrice, entryLow, entryHigh, stopLoss, target1, riskRewardRatio]);

  const handleSizerClick = () => {
    trackTelemetryEvent(
      "DECISION",
      "position_sizer_opened",
      { ticker, source: "EXECUTION_CORRIDOR_CTA" },
      ticker
    );
    trackTelemetryEvent("DECISION", "ttc_completed", { ticker }, ticker);
    onOpenPositionSizer();
  };

  return (
    <div
      data-testid="execution-corridor"
      className={`flex flex-col h-full justify-between bg-[#0e1422] border border-[#243044] rounded-xl p-4 sm:p-5 shadow-sm text-slate-100 font-sans ${className}`}
    >
      {/* Header with Title and Execution State Pill */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex flex-col">
          <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
            Stage 2 · 35% Execution Corridor
          </span>
          <h3 className="text-sm sm:text-base font-bold text-slate-50">
            {activeMode === "GUIDED" ? "Execution Decision Plan" : "Optimal Entry / Exit Ladder"}
          </h3>
        </div>

        <ExecutionStateBadge state={executionState} mode={activeMode} />
      </div>

      {/* Decision Ladder Levels */}
      <div className="py-4 space-y-3 flex-1 flex flex-col justify-center">
        {/* Level 1: Target 2 (Runner / Extended) */}
        <div
          data-testid="corridor-target-2"
          className="flex items-center justify-between p-2 rounded-lg bg-[#141b2d] border border-slate-800/80 text-xs"
        >
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 shrink-0" />
            <span className="text-slate-400 font-medium">Target 2 (Extended):</span>
          </div>
          <div className="flex items-center gap-2 font-mono">
            <span className="font-bold text-slate-50 tabular-nums">${target2.toFixed(2)}</span>
            <span className="text-emerald-400 font-semibold tabular-nums text-[11px]">
              +{target2Pct.toFixed(1)}%
            </span>
          </div>
        </div>

        {/* Level 2: Target 1 (Primary Profit Objective) */}
        <div
          data-testid="corridor-target-1"
          className="flex items-center justify-between p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/25 text-xs shadow-sm"
        >
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 shrink-0" />
            <div className="flex flex-col">
              <span className="text-slate-200 font-bold">Target 1 (Primary Exit):</span>
              <span className="text-[10px] font-mono text-emerald-400/90" data-testid="corridor-risk-reward">
                R/R Ratio: {riskRewardRatio.toFixed(2)}x
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2 font-mono">
            <span className="font-extrabold text-sm text-slate-50 tabular-nums">
              ${target1.toFixed(2)}
            </span>
            <span className="text-emerald-400 font-bold tabular-nums">
              +{target1Pct.toFixed(1)}%
            </span>
          </div>
        </div>

        {/* Level 3: Active Entry Corridor */}
        <div
          data-testid="corridor-entry"
          className={`p-3 rounded-lg border text-xs transition-all ${
            isInBuyZone
              ? "bg-[#162035] border-emerald-500/40 shadow-md ring-1 ring-emerald-500/20"
              : "bg-[#141b2d] border-slate-700/80"
          }`}
        >
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider font-semibold">
                Entry Corridor
              </span>
              {isInBuyZone && (
                <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300 font-bold">
                  ACTIVE
                </span>
              )}
            </div>
            <span className="text-[10px] font-mono text-slate-400">
              Spot: ${spotPrice.toFixed(2)}
            </span>
          </div>

          <div className="flex items-baseline justify-between">
            <span className="font-mono text-base sm:text-lg font-extrabold text-slate-50 tracking-tight">
              ${entryLow.toFixed(2)} – ${entryHigh.toFixed(2)}
            </span>
            <span className="text-[11px] font-mono text-slate-400">
              {isInBuyZone
                ? "In Buy Zone"
                : spotPrice > entryHigh
                ? `+${((spotPrice - entryHigh) / entryHigh * 100).toFixed(1)}% extended`
                : `${((spotPrice - entryLow) / entryLow * 100).toFixed(1)}% below pivot`}
            </span>
          </div>
        </div>

        {/* Level 4: Stop Loss Floor (Strict Invalidation Boundary) */}
        <div
          data-testid="corridor-stop"
          className="flex items-center justify-between p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/25 text-xs shadow-sm"
        >
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500 shrink-0" />
            <div className="flex flex-col">
              <span className="text-slate-200 font-bold">Stop Loss Floor:</span>
              <span className="text-[10px] font-mono text-rose-300/90">
                Key Invalidation Level
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2 font-mono">
            <span className="font-extrabold text-sm text-slate-50 tabular-nums">
              ${stopLoss.toFixed(2)}
            </span>
            <span className="text-rose-400 font-bold tabular-nums">
              {stopLossPct.toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* Bottom: ADV Limit & Sizer CTA */}
      <div className="pt-3 border-t border-slate-800 space-y-3">
        <div
          data-testid="corridor-adv-limit"
          className="flex items-center justify-between text-[11px] font-mono text-slate-400"
        >
          <span>{"Max Sizing (<1.0% ADV):"}</span>
          <span className="text-slate-200 font-semibold tabular-nums">
            {advShareLimit.toLocaleString()} shares
          </span>
        </div>

        {/* Primary Position Sizer Call to Action */}
        <button
          type="button"
          onClick={handleSizerClick}
          data-testid="corridor-sizer-cta"
          className="w-full py-2.5 px-4 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-bold text-xs sm:text-sm transition-all shadow-md active:scale-95 flex items-center justify-center gap-2 cursor-pointer"
        >
          <span>Size Position & Calculate Risk</span>
          <span aria-hidden="true">→</span>
        </button>
      </div>
    </div>
  );
}
