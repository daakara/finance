"use client";

import { OptimalExecutionPlan } from "../lib/api";
import InsightProvenanceModal from "./InsightProvenanceModal";

interface OptimalEntryExitCardProps {
  symbol: string;
  executionPlan?: OptimalExecutionPlan;
  userRole?: "DAY_TRADER" | "LONG_TERM";
}

export default function OptimalEntryExitCard({
  symbol,
  executionPlan,
  userRole = "LONG_TERM",
}: OptimalEntryExitCardProps) {
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

  return (
    <div
      className={`bg-[#111722] border rounded-xl p-4 sm:p-5 shadow-xl space-y-4 font-mono transition-colors ${
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
              <span>🎯 {symbol} Optimal Entry, Stop-Loss & Target Ladder</span>
            </h3>
          </div>
          <p className="text-[11px] sm:text-xs text-slate-400 mt-0.5">
            {isDayTrader
              ? "Linda Raschke 20 EMA Momentum Pullback & Turtle ATR Sizing"
              : "Mark Minervini Volatility Contraction Pattern (VCP) & Stage 2 Pivot"}
          </p>
        </div>

        {/* Reward-to-Risk Pill */}
        <div className="flex items-center space-x-2">
          <div className="bg-[#090d14] px-3 py-1 rounded-lg border border-[#243044] text-right">
            <span className="text-[9px] text-slate-500 block uppercase font-bold">Reward : Risk</span>
            <span className="text-sm font-extrabold text-emerald-400 tabular-nums">
              {risk_reward_ratio} : 1.0
            </span>
          </div>
          <span
            className={`text-[10px] sm:text-[11px] px-2.5 py-1 rounded-md font-semibold border ${
              isDayTrader
                ? "text-amber-400 bg-amber-950/60 border-amber-800/80"
                : "text-emerald-400 bg-emerald-950/60 border-emerald-800/80"
            }`}
          >
            {isDayTrader ? "⚡ Intraday Playbook" : "🏛️ Swing/Growth Playbook"}
          </span>
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

        {/* Optimal Entry Range */}
        <div className="flex items-center justify-between p-2 rounded-lg bg-cyan-950/30 border border-cyan-800/40 text-xs">
          <div className="flex items-center space-x-2">
            <span className="text-cyan-400 font-bold">🔵 OPTIMAL ENTRY ACCUMULATION ZONE</span>
            <span className="text-[10px] text-slate-400 hidden sm:inline">• 20 EMA & Value Area Pullback</span>
          </div>
          <strong className="text-cyan-300 text-sm font-bold tabular-nums">
            ${optimal_entry_min.toFixed(2)} – ${optimal_entry_max.toFixed(2)}
          </strong>
        </div>

        {/* Stop Loss / Invalidation */}
        <div className="flex items-center justify-between p-2 rounded-lg bg-rose-950/30 border border-rose-800/40 text-xs">
          <div className="flex items-center space-x-2">
            <span className="text-rose-400 font-bold">🛑 HARD STOP-LOSS / INVALIDATION</span>
            <span className="text-[10px] text-slate-400 hidden sm:inline">• -1.5x ATR Volatility Cut Floor</span>
          </div>
          <div className="text-right">
            <strong className="text-rose-400 text-sm font-bold tabular-nums">
              ${stop_loss.toFixed(2)}
            </strong>
            <span className="text-[10px] text-rose-500 ml-1.5 tabular-nums">
              ({stop_loss_pct}%)
            </span>
          </div>
        </div>
      </div>

      {/* Quantitative Context Details */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
        <div className="bg-[#090d14] p-3 rounded-lg border border-[#1e293b] space-y-1">
          <span className="text-[10px] text-cyan-400 block uppercase font-bold">
            Setup Pattern & Stage Analysis
          </span>
          <div className="font-bold text-slate-200">{setup_pattern}</div>
          <div className="text-[11px] text-slate-400">{entry_thesis}</div>
        </div>

        <div className="bg-[#090d14] p-3 rounded-lg border border-[#1e293b] space-y-1">
          <span className="text-[10px] text-rose-400 block uppercase font-bold">
            Strict Invalidation & Exit Condition
          </span>
          <div className="font-bold text-slate-200">{stage_phase}</div>
          <div className="text-[11px] text-slate-400">{invalidation_condition}</div>
        </div>
      </div>

      {/* 📜 Deep Dive & Verified Sources Provenance Trigger */}
      <InsightProvenanceModal
        symbol={symbol}
        executionPlan={executionPlan}
        userRole={userRole}
      />
    </div>
  );
}