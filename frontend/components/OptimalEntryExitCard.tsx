"use client";

import { useState } from "react";
import { OptimalExecutionPlan } from "../lib/api";
import InsightProvenanceModal from "./InsightProvenanceModal";
import PositionSizerModal from "./PositionSizerModal";
import AlertTriggerModal from "./AlertTriggerModal";

interface OptimalEntryExitCardProps {
  symbol: string;
  executionPlan?: OptimalExecutionPlan;
  userRole?: "DAY_TRADER" | "LONG_TERM";
  smartMoney?: any;
  macroRegime?: any;
}

export default function OptimalEntryExitCard({
  symbol,
  executionPlan,
  userRole = "LONG_TERM",
  smartMoney,
  macroRegime,
}: OptimalEntryExitCardProps) {
  const [isSizerOpen, setIsSizerOpen] = useState<boolean>(false);
  const [isAlertOpen, setIsAlertOpen] = useState<boolean>(false);

  if (!executionPlan) return null;

  const isDayTrader = userRole === "DAY_TRADER";
  const {
    current_price,
    optimal_entry_min,
    optimal_entry_max,
    stop_loss,
    stop_loss_pct,
    take_profit_1,
    take_profit_1_pct,
    take_profit_2,
    take_profit_2_pct,
    risk_reward_ratio,
    setup_pattern,
    entry_thesis,
    invalidation_condition,
    stage_phase,
    vcp_contraction_status,
  } = executionPlan;

  const isStage4 = Boolean(
    setup_pattern?.includes("Stage 4") ||
    stage_phase?.includes("Stage 4") ||
    setup_pattern?.includes("Correction")
  );

  const entryMin = Math.min(optimal_entry_min, optimal_entry_max);
  const entryMax = Math.max(optimal_entry_min, optimal_entry_max);
  const inZone = current_price >= entryMin && current_price <= entryMax;
  const zoneWidth = Math.max(0.01, entryMax - entryMin);
  const zonePositionPct = inZone ? ((current_price - entryMin) / zoneWidth) * 100 : 50;

  // Tactical Execution Hint
  let zoneTacticalHint: { label: string; advice: string; color: string } | null = null;
  if (!isStage4 && inZone) {
    if (zonePositionPct > 65) {
      zoneTacticalHint = {
        label: "⚠️ Near Zone Ceiling",
        advice: `Spot ($${current_price.toFixed(2)}) is near the upper bound of the buy zone. Scale in with 30% initial size or place limit orders near $${entryMin.toFixed(2)} to maximize asymmetric R:R.`,
        color: "text-amber-300 border-amber-900/60 bg-amber-950/40",
      };
    } else if (zonePositionPct < 35) {
      zoneTacticalHint = {
        label: "🎯 Near Support Floor",
        advice: `Spot ($${current_price.toFixed(2)}) is at the bottom of the accumulation corridor. Favorable asymmetric entry with tight invalidation floor.`,
        color: "text-emerald-300 border-emerald-900/60 bg-emerald-950/40",
      };
    } else {
      zoneTacticalHint = {
        label: "✅ Mid-Zone Value Area",
        advice: `Spot is comfortably centered within the accumulation corridor with favorable 2.5:1+ R:R structure.`,
        color: "text-cyan-300 border-cyan-900/60 bg-cyan-950/40",
      };
    }
  }

  const hasSmartMoneyConfluence = Boolean(
    (smartMoney?.optionsFlow && smartMoney.optionsFlow.some((f: any) => f.sentiment === "Bullish" || f.type?.includes("CALL"))) ||
    (smartMoney?.congressTrades && smartMoney.congressTrades.some((t: any) => t.tx_type === "Purchase"))
  );
  const isAdverseMacro = Boolean(
    macroRegime?.vix && Number(macroRegime.vix) > 20
  );

  return (
    <div
      className={`bg-[#111722] border rounded-xl p-4 sm:p-5 shadow-xl space-y-4 font-sans transition-colors ${
        isDayTrader ? "border-amber-900/40" : "border-[#243044]"
      }`}
    >
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-3">
        <div>
          <div className="flex items-center space-x-2">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                isDayTrader ? "bg-amber-400" : "bg-emerald-400"
              } animate-pulse`}
            ></span>
            <h3 className="text-sm sm:text-base font-bold text-slate-100 tracking-tight flex items-center gap-2">
              <span>🎯 {symbol} Optimal Execution Ladder</span>
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5 font-normal">
            {isDayTrader
              ? "Trend Momentum Pullback & Volatility-Protected Stop Ladder"
              : setup_pattern?.includes("Stage 4")
              ? "Stage 4 Correction & Volatility-Constrained Risk Boundaries"
              : "Institutional Accumulation Breakout & Precision Entry Ladder"}
          </p>
          <div className="flex flex-wrap items-center gap-2 mt-1.5 text-[11px] font-medium text-slate-400">
            <span className="px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800/80 inline-flex items-center gap-1 font-mono text-[10px]">
              <span>📡</span> Minervini VCP + 14-ATR
            </span>
            {hasSmartMoneyConfluence && (
              <span className="px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-700/80 inline-flex items-center gap-1 font-mono text-[10px]" title="Smart Money Confluence: Institutional call sweeps or congressional purchases detected on this asset.">
                <span>🏛️</span> Smart Money Inflow (+15% Buffer)
              </span>
            )}
            {isAdverseMacro && (
              <span className="px-2 py-0.5 rounded bg-amber-950/80 text-amber-300 border border-amber-700/80 inline-flex items-center gap-1 font-mono text-[10px]" title="Elevated macro market volatility (VIX > 20). Defensive sizing active.">
                <span>⚠️</span> Macro Buffer Active
              </span>
            )}
            <span className="hidden sm:inline text-slate-500">•</span>
            <span className="text-slate-400 text-[11px]">Volatility-anchored risk limits</span>
          </div>
        </div>

        {/* Reward-to-Risk Pill */}
        <div className="flex items-center space-x-2">
          <div className="bg-[#090d14] px-3 py-1 rounded-lg border border-[#243044] text-right" title={isStage4 ? "Post-Breakout Pivot Expected Reward-to-Risk Ratio" : "Reward-to-Risk ratio: Potential gain to TP1 relative to maximum risk at Stop Loss"}>
            <span className="text-[9px] text-slate-500 block uppercase font-bold font-mono">
              {isStage4 ? "Post-Pivot R:R" : "Reward : Risk"}
            </span>
            <span className={`text-sm font-extrabold font-mono tabular-nums ${isStage4 ? "text-amber-400" : "text-emerald-400"}`}>
              {risk_reward_ratio} : 1.0
            </span>
          </div>
          <button
            type="button"
            onClick={() => {
              const nextRole = isDayTrader ? "LONG_TERM" : "DAY_TRADER";
              try { localStorage.setItem("FINANCE_USER_ROLE", nextRole); } catch {}
              window.dispatchEvent(new CustomEvent("finance:role-change", { detail: nextRole }));
            }}
            aria-label={`Current mode: ${isDayTrader ? "Intraday Playbook" : "Swing/Growth Playbook"}. Click to switch mode.`}
            className={`text-xs px-2.5 py-1 rounded-md font-semibold border cursor-pointer active:scale-95 transition-transform focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
              isDayTrader
                ? "text-amber-400 bg-amber-950/60 border-amber-800 hover:bg-amber-900/80"
                : "text-emerald-400 bg-emerald-950/60 border-emerald-800 hover:bg-emerald-900/80"
            }`}
          >
            <span>{isDayTrader ? "⚡ Intraday ⇄" : "🏛️ Swing/Growth ⇄"}</span>
          </button>
        </div>
      </div>

      {/* Interactive Execution Price Ladder */}
      <div className="space-y-2 bg-[#090d14] p-3.5 rounded-xl border border-[#1e293b]">
        <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider flex items-center justify-between">
          <span>Mathematical Execution Ladder</span>
          <span className="text-slate-400">Current Spot: ${current_price.toFixed(2)}</span>
        </div>

        {/* Take Profit 2 */}
        <div className="flex items-center justify-between p-2 rounded-lg bg-emerald-950/30 border border-emerald-800/40 text-xs">
          <div className="flex items-center space-x-2">
            <span className="text-emerald-400 font-bold">🟢 TARGET 2 (Extended Runner)</span>
            <span className="text-[10px] text-slate-400 hidden sm:inline">• Major Resistance / +3.5x ATR</span>
          </div>
          <div className="text-right">
            <strong className="text-emerald-400 text-sm font-bold tabular-nums">
              ${take_profit_2.toFixed(2)}
            </strong>
            <span className="text-[10px] text-emerald-500 ml-1.5 tabular-nums">
              (+{take_profit_2_pct}%)
            </span>
          </div>
        </div>

        {/* Take Profit 1 */}
        <div className="flex items-center justify-between p-2 rounded-lg bg-emerald-950/20 border border-emerald-800/30 text-xs">
          <div className="flex items-center space-x-2">
            <span className="text-emerald-400 font-bold">🟢 TARGET 1 (Take Profit / Scale)</span>
            <span className="text-[10px] text-slate-400 hidden sm:inline">• Prior Swing High / +2.0x ATR</span>
          </div>
          <div className="text-right">
            <strong className="text-emerald-400 text-sm font-bold tabular-nums">
              ${take_profit_1.toFixed(2)}
            </strong>
            <span className="text-[10px] text-emerald-500 ml-1.5 tabular-nums">
              (+{take_profit_1_pct}%)
            </span>
          </div>
        </div>

        {/* CURRENT SPOT BENCHMARK */}
        <div className="flex items-center justify-between p-2.5 rounded-lg bg-[#162030] border border-cyan-500/50 text-xs shadow-inner">
          <div className="flex items-center space-x-2">
            <span className="text-cyan-300 font-extrabold flex items-center gap-1">
              <span>⚡ CURRENT MARKET PRICE</span>
            </span>
            <span className="text-[10px] text-slate-400 hidden sm:inline">• Live Spot</span>
          </div>
          <strong className="text-white text-base font-bold tabular-nums">
            ${current_price.toFixed(2)}
          </strong>
        </div>

        {/* Optimal Entry / Prospective Base Range */}
        <div className={`flex items-center justify-between p-2 rounded-lg text-xs transition-colors ${
          isStage4
            ? "bg-[#151922] border border-amber-800/60"
            : "bg-cyan-950/30 border border-cyan-800/40"
        }`}>
          <div className="flex items-center space-x-2">
            <span className={isStage4 ? "text-amber-400 font-bold" : "text-cyan-400 font-bold"}>
              {isStage4 ? "⏳ PROSPECTIVE BASE CORRIDOR (AWAITING PIVOT)" : "🔵 OPTIMAL ENTRY ACCUMULATION ZONE"}
            </span>
            <span className="text-[10px] text-slate-400 hidden sm:inline">
              {isStage4 ? "• 50-Day SMA Reclaim Required" : "• 20 EMA & Value Area Pullback"}
            </span>
          </div>
          <strong className={`text-sm font-bold font-mono tabular-nums ${isStage4 ? "text-amber-300" : "text-cyan-300"}`}>
            ${Math.min(optimal_entry_min, optimal_entry_max).toFixed(2)} – ${Math.max(optimal_entry_min, optimal_entry_max).toFixed(2)}
          </strong>
        </div>

        {/* Tactical Relative Position Execution Badge */}
        {zoneTacticalHint && (
          <div className={`p-2 rounded-lg border text-xs flex items-start gap-2 ${zoneTacticalHint.color}`}>
            <span className="font-bold shrink-0">{zoneTacticalHint.label}:</span>
            <span className="text-[11px] leading-relaxed text-slate-200 font-sans">{zoneTacticalHint.advice}</span>
          </div>
        )}

        {/* Stop Loss / Invalidation */}
        <div className="flex items-center justify-between p-2 rounded-lg bg-rose-950/30 border border-rose-800/40 text-xs">
          <div className="flex items-center space-x-2">
            <span className="text-rose-400 font-bold">🛑 HARD STOP-LOSS / INVALIDATION</span>
            <span className="text-[10px] text-slate-400 hidden sm:inline">• -1.5x ATR Volatility Cut Floor</span>
          </div>
          <div className="text-right">
            <strong className="text-rose-400 text-sm font-bold font-mono tabular-nums">
              ${stop_loss.toFixed(2)}
            </strong>
            <span className="text-[10px] text-rose-500 ml-1.5 font-mono tabular-nums">
              ({stop_loss_pct}%)
            </span>
          </div>
        </div>
      </div>

      {/* Quantitative Context Details */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-sans">
        <div className="bg-[#090d14] p-3 rounded-lg border border-[#1e293b] space-y-1">
          <span className="text-[10px] text-cyan-400 block uppercase font-bold font-mono">
            Setup Pattern & Stage Analysis
          </span>
          <div className="font-bold text-slate-200">{setup_pattern}</div>
          <div className="text-[11px] text-slate-400">{entry_thesis}</div>
          {isStage4 && (
            <div className="mt-2 pt-1.5 border-t border-[#1e293b] flex items-center justify-between text-[11px] text-amber-300">
              <span>🎯 Key Breakout Pivot (50-Day SMA):</span>
              <strong className="font-mono text-amber-400 font-bold">${(executionPlan.breakout_pivot || current_price * 1.072).toFixed(2)}</strong>
            </div>
          )}
        </div>

        <div className="bg-[#090d14] p-3 rounded-lg border border-[#1e293b] space-y-1">
          <span className="text-[10px] text-rose-400 block uppercase font-bold font-mono">
            Strict Invalidation & Exit Condition
          </span>
          <div className="font-bold text-slate-200">{stage_phase}</div>
          <div className="text-[11px] text-slate-400">{invalidation_condition}</div>
        </div>
      </div>

      {/* 📜 Deep Dive Provenance, Position Sizer & Alerts Triggers */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-[#1b2434]">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setIsSizerOpen(true)}
            className="px-3 py-1.5 rounded-lg text-xs font-bold transition-all active:scale-[0.96] border bg-cyan-600/20 hover:bg-cyan-500 hover:text-slate-950 border-cyan-500/50 text-cyan-300 flex items-center gap-1.5 shadow"
          >
            <span>⚖️</span>
            <span>Size Position</span>
          </button>

          <button
            type="button"
            onClick={() => setIsAlertOpen(true)}
            className="px-3 py-1.5 rounded-lg text-xs font-bold transition-all active:scale-[0.96] border bg-amber-600/20 hover:bg-amber-500 hover:text-slate-950 border-amber-500/50 text-amber-300 flex items-center gap-1.5 shadow"
          >
            <span>🔔</span>
            <span>Set Trigger Alert</span>
          </button>
        </div>

        <InsightProvenanceModal
          symbol={symbol}
          executionPlan={executionPlan}
          userRole={userRole}
        />
      </div>

      <PositionSizerModal
        isOpen={isSizerOpen}
        onClose={() => setIsSizerOpen(false)}
        symbol={symbol}
        entryPrice={current_price}
        stopLoss={stop_loss}
        takeProfit1={take_profit_1}
        riskRewardRatio={risk_reward_ratio}
        isStage4={isStage4}
      />

      <AlertTriggerModal
        isOpen={isAlertOpen}
        onClose={() => setIsAlertOpen(false)}
        symbol={symbol}
        currentPrice={current_price}
        optimalEntryMin={optimal_entry_min}
        optimalEntryMax={optimal_entry_max}
        stopLoss={stop_loss}
        takeProfit1={take_profit_1}
        isStage4={isStage4}
        breakoutPivot={executionPlan.breakout_pivot || Number((current_price * 1.072).toFixed(2))}
      />
    </div>
  );
}