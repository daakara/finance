"use client";

import React from "react";
import { QuantitativeInsight } from "../../types/insight";

export interface SharedExecutionLadderProps {
  insight: QuantitativeInsight;
  mode?: "STANDARD" | "GUIDED" | "QUANT";
  isStaleTape: boolean;
  isActionable: boolean;
  isExtended: boolean;
  isInBuyZone: boolean;
  pivotPrice: number | null;
  onOpenSizer?: () => void;
}

export function SharedExecutionLadder({
  insight,
  mode = "STANDARD",
  isStaleTape,
  isActionable,
  isExtended,
  isInBuyZone,
  pivotPrice,
  onOpenSizer,
}: SharedExecutionLadderProps) {
  const kl = insight.standard?.keyLevels || (insight as any).keyLevels || {};
  const rawPrice = insight.price || kl.currentPrice;
  const hasValidPrice = typeof rawPrice === "number" && !isNaN(rawPrice) && rawPrice > 0;

  // Mode-adaptive label configuration (Guided plain-language, Standard institutional, Quant mathematical)
  const labels = {
    title:
      mode === "GUIDED"
        ? isStaleTape
          ? "Execution Plan (Historical)"
          : "Execution Plan"
        : mode === "QUANT"
        ? isStaleTape
          ? "Quantitative Execution Geometry (Historical)"
          : "Quantitative Execution Geometry"
        : isStaleTape
        ? "Execution Ladder (Historical Tape)"
        : "Execution Ladder",
    pivot:
      mode === "GUIDED"
        ? isStaleTape
          ? "Historical Trigger"
          : "Key Trigger"
        : mode === "QUANT"
        ? isStaleTape
          ? "Historical Pivot"
          : "Pivot / Baseline"
        : isStaleTape
        ? "Historical Pivot"
        : "Pivot",
    entry:
      mode === "GUIDED"
        ? isStaleTape
          ? "Historical Entry"
          : "Ideal Entry Zone"
        : mode === "QUANT"
        ? isStaleTape
          ? "Historical Corridor"
          : "Entry Corridor"
        : isStaleTape
        ? "Historical Zone"
        : "Buy Zone",
    stop:
      mode === "GUIDED"
        ? isStaleTape
          ? "Historical Stop"
          : "Safety Stop Level"
        : mode === "QUANT"
        ? isStaleTape
          ? "Historical Floor"
          : "Invalidation Floor"
        : isStaleTape
        ? "Historical Stop"
        : "Stop Loss",
    target:
      mode === "GUIDED"
        ? isStaleTape
          ? "Historical Target"
          : "First Profit Target"
        : mode === "QUANT"
        ? isStaleTape
          ? "Historical Target"
          : "Target 1 (Base)"
        : isStaleTape
        ? "Historical Target"
        : "Target 1",
    rr: mode === "GUIDED" ? "Risk/Reward" : mode === "QUANT" ? "Profit/Risk" : "R:R",
    sizingButton:
      mode === "GUIDED"
        ? isActionable
          ? "Position Sizing Calculator"
          : "0 Shares (Awaiting Trigger)"
        : mode === "QUANT"
        ? isActionable
          ? "Model Sizing Authority"
          : "0 Shares (Condition Unmet)"
        : isActionable
        ? "Configure Shares"
        : "0 Shares (Entry Unconfirmed)",
  };

  return (
    <div id="card-middle-panel" className="middle-panel space-y-2 pt-1 font-mono">
      <div className="panel-head flex items-center justify-between text-[10px] font-bold text-slate-400 uppercase tracking-wider">
        <span id="middle-panel-title">{labels.title}</span>
        <span id="middle-panel-sub" className="mono text-cyan-300">
          {labels.rr} {kl.profitRiskRatio ? `${kl.profitRiskRatio.toFixed(2)} : 1.0` : "N/A"}
        </span>
      </div>
      <div
        id="middle-panel-content"
        data-testid="cockpit-execution-ladder"
        className="ladder-grid grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 text-xs"
      >
        <div className="ladder-cell min-w-0 p-2.5 rounded-xl bg-[#070b13] border border-[#1b2639]">
          <span className="cell-lbl text-[10px] text-slate-400 block uppercase font-bold">
            {labels.pivot}
          </span>
          <span
            id="cell-pivot"
            data-testid="cell-pivot"
            className="cell-val mono text-white font-bold block mt-0.5 whitespace-nowrap text-xs leading-tight tabular-nums"
          >
            {pivotPrice != null ? `$${pivotPrice.toFixed(2)}` : "N/A"}
          </span>
        </div>
        <div className="ladder-cell min-w-0 p-2.5 rounded-xl bg-[#070b13] border border-[#1b2639]">
          <span className="cell-lbl text-[10px] text-slate-400 block uppercase font-bold">
            {labels.entry}
          </span>
          <span
            id="cell-entry"
            data-testid="cell-entry"
            className="cell-val mono text-amber-300 font-bold block mt-0.5 text-xs leading-tight"
          >
            {kl.entryMin != null && kl.entryMax != null && kl.entryMin > 0 && kl.entryMax >= kl.entryMin ? (
              <span className="flex flex-wrap items-baseline gap-1">
                <span className="tabular-nums whitespace-nowrap">${kl.entryMin.toFixed(2)}</span>
                <span className="text-slate-400 font-normal text-[10px] sm:text-xs">to</span>
                <span className="tabular-nums whitespace-nowrap">${kl.entryMax.toFixed(2)}</span>
              </span>
            ) : (
              <span className="whitespace-nowrap">{kl.watchZone || (kl.entryMin != null && kl.entryMin <= 0 ? "Suppressed" : "Awaiting Setup")}</span>
            )}
          </span>
        </div>
        <div className="ladder-cell min-w-0 p-2.5 rounded-xl bg-[#070b13] border border-[#1b2639]">
          <span className="cell-lbl text-[10px] text-slate-400 block uppercase font-bold">
            {labels.stop}
          </span>
          <span
            id="cell-stop"
            data-testid="cell-stop"
            className="cell-val negative mono text-rose-400 font-bold block mt-0.5 text-xs leading-tight"
          >
            {typeof kl.stopLoss === "number" && !isNaN(kl.stopLoss) && kl.stopLoss > 0 ? (
              <span className="flex flex-wrap items-baseline gap-1">
                <span className="tabular-nums whitespace-nowrap">${kl.stopLoss.toFixed(2)}</span>
                {pivotPrice != null && pivotPrice > 0 ? (
                  kl.stopLoss >= pivotPrice ? (
                    <span className="text-amber-400 font-normal text-[10px] whitespace-nowrap">
                      (Inverted Stop ≥ Pivot)
                    </span>
                  ) : (
                    <span className="text-rose-400/80 font-normal text-[10px] whitespace-nowrap">
                      (-{(((pivotPrice - kl.stopLoss) / pivotPrice) * 100).toFixed(1)}% base)
                    </span>
                  )
                ) : null}
              </span>
            ) : kl.stopLoss === 0 ? (
              "Suppressed"
            ) : (
              "N/A"
            )}
          </span>
        </div>
        <div className="ladder-cell min-w-0 p-2.5 rounded-xl bg-[#070b13] border border-[#1b2639]">
          <span className="cell-lbl text-[10px] text-slate-400 block uppercase font-bold">
            {labels.target}
          </span>
          <span
            id="cell-target"
            data-testid="cell-target"
            className="cell-val positive mono text-emerald-400 font-bold block mt-0.5 text-xs leading-tight"
          >
            {typeof kl.target1 === "number" && !isNaN(kl.target1) && kl.target1 > 0 ? (
              <span className="flex flex-wrap items-baseline gap-1">
                <span className="tabular-nums whitespace-nowrap">${kl.target1.toFixed(2)}</span>
                {pivotPrice != null && pivotPrice > 0 ? (
                  kl.target1 <= pivotPrice ? (
                    <span className="text-amber-400 font-normal text-[10px] whitespace-nowrap">
                      (Inverted Target ≤ Pivot)
                    </span>
                  ) : (
                    <span className="text-emerald-400/80 font-normal text-[10px] whitespace-nowrap">
                      (+{(((kl.target1 - pivotPrice) / pivotPrice) * 100).toFixed(1)}%)
                    </span>
                  )
                ) : null}
              </span>
            ) : kl.target1 === 0 ? (
              "Suppressed"
            ) : (
              "N/A"
            )}
          </span>
        </div>
      </div>

      {/* Position Sizing & Preflight Clearance Link */}
      <div id="card-sizing-bar" className="sizing-bar border-t border-[#182335] pt-2 space-y-1.5 text-xs font-mono">
        <div className="sizing-meta flex items-center justify-between text-slate-300">
          <span id="sizing-line-1">
            Position Size:{" "}
            {isStaleTape ? (
              <span id="sizing-shares" className="text-slate-500 font-bold">
                Execution Prohibited (Stale Tape)
              </span>
            ) : onOpenSizer ? (
              <button
                type="button"
                onClick={onOpenSizer}
                id="sizing-shares"
                className="sizing-bold mono font-bold text-cyan-400 hover:underline cursor-pointer inline-flex items-center gap-1"
                title="Open Position Sizer to configure shares"
              >
                {labels.sizingButton}
              </button>
            ) : (
              <span id="sizing-shares" className="text-slate-400 font-bold">
                {isActionable ? "Configured" : "0 Shares (Pending)"}
              </span>
            )}
          </span>
          <a
            href="#preflight-readiness-bar"
            id="preflight-link"
            className="text-[11px] text-cyan-400 hover:text-cyan-300 hover:underline flex items-center gap-1"
            title="Jump to Pre-Flight Clearance Checklist"
          >
            <span>🛡️ Pre-Flight Clearance ↓</span>
          </a>
        </div>
        <div className="sizing-matrix-grid grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px]">
          <div className="sizing-cell p-2.5 rounded-lg bg-[#070b13] border border-[#1b2639] space-y-1">
            <div className="flex items-center justify-between text-[10px] text-slate-400 uppercase font-bold">
              <span className="sizing-lbl">Stop Distance & Risk Basis</span>
              <span
                className={`px-1 rounded text-[9px] font-mono ${
                  isStaleTape
                    ? "bg-slate-800 text-slate-400 border border-slate-700"
                    : isExtended
                    ? "bg-amber-950/80 text-amber-300 border border-amber-800/80"
                    : isInBuyZone
                    ? "bg-emerald-950/80 text-emerald-300 border border-emerald-800/80"
                    : "bg-[#141c2e] text-slate-400"
                }`}
              >
                {isStaleTape
                  ? "Historical Tape"
                  : isExtended
                  ? "Extended (High Risk)"
                  : isInBuyZone
                  ? "In Buy Zone"
                  : "Awaiting Setup"}
              </span>
            </div>
            <div
              id="sizing-stop-loss"
              className="sizing-val negative mono font-bold text-rose-400 block text-[11px] leading-tight space-y-0.5"
            >
              {kl.stopLoss && hasValidPrice ? (
                <>
                  <div className="flex items-baseline justify-between gap-1">
                    <span className="text-[10px] text-slate-400 font-normal">
                      From Current Spot (${(rawPrice as number).toFixed(2)}):
                    </span>
                    <span className="tabular-nums">
                      -${Math.abs((rawPrice as number) - kl.stopLoss).toFixed(2)} (-
                      {Math.abs((((rawPrice as number) - kl.stopLoss) / (rawPrice as number)) * 100).toFixed(2)}%)
                    </span>
                  </div>
                  {pivotPrice != null &&
                    pivotPrice > 0 &&
                    Math.abs((rawPrice as number) - pivotPrice) > 0.05 && (
                      <div className="flex items-baseline justify-between gap-1 text-[10px] text-slate-400 font-normal border-t border-[#182335] pt-0.5">
                        <span>Planned from Pivot (${pivotPrice.toFixed(2)}):</span>
                        <span className="text-slate-300 font-mono tabular-nums">
                          -${Math.abs(pivotPrice - kl.stopLoss).toFixed(2)} (-
                          {Math.abs(((pivotPrice - kl.stopLoss) / pivotPrice) * 100).toFixed(2)}%)
                        </span>
                      </div>
                    )}
                </>
              ) : (
                "N/A"
              )}
            </div>
          </div>
          <div className="sizing-cell p-2.5 rounded-lg bg-[#070b13] border border-[#1b2639] space-y-1">
            <div className="flex items-center justify-between text-[10px] text-slate-400 uppercase font-bold">
              <span className="sizing-lbl">Position Sizing Basis</span>
              <span className="text-[9px] font-mono text-cyan-400">
                {isStaleTape ? "Prohibited" : isActionable ? "Model Active" : "Suppressed"}
              </span>
            </div>
            <span id="sizing-utilization" className="sizing-val mono font-bold text-slate-300 block text-xs">
              {isStaleTape
                ? "Execution Prohibited (Stale Tape)"
                : !isActionable
                ? "0% (Awaiting Setup Trigger)"
                : "Informational Sizing (1% Risk Model)"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SharedExecutionLadder;
