"use client";

import React from "react";
import { QuantitativeInsight } from "../../types/insight";
import FinancialDisclaimer from "../FinancialDisclaimer";

interface AdvancedTerminalViewProps {
  insight: QuantitativeInsight;
  onOpenSizer: () => void;
  onOpenWhy: () => void;
}

export default function AdvancedTerminalView({
  insight,
  onOpenSizer,
  onOpenWhy,
}: AdvancedTerminalViewProps) {
  const adv = insight.advanced;
  const kl = insight.standard.keyLevels;

  return (
    <div className="space-y-4 font-mono text-xs text-slate-100 animate-fade-in">
      {/* Advanced Badge Banner */}
      <div className="flex items-center justify-between bg-purple-950/40 border border-purple-800/50 p-2.5 rounded-xl text-xs">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-purple-400" />
          <span className="text-purple-300 font-bold">🟣 ADVANCED WORKSTATION</span>
          <span className="text-slate-400 hidden sm:inline">• Maximum quantitative density, raw models & execution ladder</span>
        </div>
        <button
          onClick={onOpenWhy}
          className="text-[11px] text-purple-400 hover:text-purple-300 underline font-bold cursor-pointer"
        >
          Decompose Score →
        </button>
      </div>

      {/* Dense Quant Metrics Ribbon */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 bg-[#070b13] p-3 rounded-xl border border-[#1b2639]">
        <div className="bg-[#0b101b] p-2 rounded-lg border border-[#162132]">
          <span className="text-[10px] text-slate-500 block">RSI (14D)</span>
          <span className={`text-sm font-black ${adv.rsi < 30 ? "text-emerald-400" : adv.rsi > 70 ? "text-rose-400" : "text-slate-200"}`}>
            {adv.rsi.toFixed(1)}
          </span>
        </div>

        <div className="bg-[#0b101b] p-2 rounded-lg border border-[#162132]">
          <span className="text-[10px] text-slate-500 block">20 EMA</span>
          <span className="text-sm font-black text-cyan-300">
            {adv.ema20 !== undefined ? `$${adv.ema20.toFixed(2)}` : "N/A"}
          </span>
        </div>

        <div className="bg-[#0b101b] p-2 rounded-lg border border-[#162132]">
          <span className="text-[10px] text-slate-500 block">50 SMA</span>
          <span className="text-sm font-black text-indigo-300">
            {adv.sma50 !== undefined ? `$${adv.sma50.toFixed(2)}` : "N/A"}
          </span>
        </div>

        <div className="bg-[#0b101b] p-2 rounded-lg border border-[#162132]">
          <span className="text-[10px] text-slate-500 block">ATR (14D)</span>
          <span className="text-sm font-black text-amber-300">
            {adv.atr !== undefined ? `$${adv.atr.toFixed(2)}` : "N/A"}
          </span>
        </div>

        <div className="bg-[#0b101b] p-2 rounded-lg border border-[#162132]">
          <span className="text-[10px] text-slate-500 block">RVOL</span>
          <span className="text-sm font-black text-emerald-300">
            {adv.rvol !== undefined ? `${adv.rvol.toFixed(2)}×` : "N/A"}
          </span>
        </div>

        <div className="bg-[#0b101b] p-2 rounded-lg border border-[#162132]">
          <span className="text-[10px] text-slate-500 block">BETA (SPY)</span>
          <span className="text-sm font-black text-slate-200">
            {adv.beta !== undefined ? adv.beta.toFixed(2) : "N/A"}
          </span>
        </div>
      </div>

      {/* Main Advanced Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Execution Ladder Column */}
        <div className="bg-[#0b101b] border border-[#1d293d] rounded-2xl p-4 shadow-xl space-y-3">
          <div className="flex items-center justify-between border-b border-[#182335] pb-2">
            <span className="text-slate-300 font-bold">Execution Ladder</span>
            <span className="text-emerald-400 font-bold">
              {kl.profitRiskRatio !== undefined ? `${kl.profitRiskRatio.toFixed(2)}:1 R:R` : "N/A (< 50 sessions)"}
            </span>
          </div>

          <div className="space-y-1.5 font-mono text-xs">
            <div className="flex items-center justify-between p-2 rounded bg-[#07130e] border border-emerald-950/60">
              <span className="text-emerald-400 font-bold">TARGET 2 (R2)</span>
              <span className="text-emerald-300 font-black">
                {kl.target2 !== undefined ? `$${kl.target2.toFixed(2)} (+${kl.target2Pct}%)` : "N/A"}
              </span>
            </div>

            <div className="flex items-center justify-between p-2 rounded bg-[#07130e] border border-emerald-950/60">
              <span className="text-emerald-400 font-bold">TARGET 1 (R1)</span>
              <span className="text-emerald-300 font-black">
                {kl.target1 !== undefined ? `$${kl.target1.toFixed(2)} (+${kl.target1Pct}%)` : "N/A"}
              </span>
            </div>

            <div className="flex items-center justify-between p-2 rounded bg-[#130f07] border border-amber-950/60">
              <span className="text-amber-400 font-bold">WATCH ZONE</span>
              <span className="text-amber-300 font-black">{kl.watchZone}</span>
            </div>

            <div className="flex items-center justify-between p-2 rounded bg-[#09101c] border border-cyan-900/50">
              <span className="text-cyan-400 font-bold">SPOT PRICE</span>
              <span className="text-white font-black">${insight.price.toFixed(2)}</span>
            </div>

            <div className="flex items-center justify-between p-2 rounded bg-[#150a0d] border border-rose-950/60">
              <span className="text-rose-400 font-bold">STOP LOSS (S1)</span>
              <span className="text-rose-300 font-black">${kl.stopLoss.toFixed(2)} ({kl.stopLossPct}%)</span>
            </div>
          </div>

          {insight.terminalState.posture === "ACQUIRE" ? (
            <button
              onClick={onOpenSizer}
              className="w-full py-2 bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-black rounded-xl transition-all active:scale-95 cursor-pointer shadow-md mt-2"
            >
              ⚖️ Open Institutional Position Sizer
            </button>
          ) : insight.terminalState.posture === "EXIT_REVIEW" ? (
            <button
              onClick={onOpenWhy}
              className="w-full py-2 bg-rose-600 hover:bg-rose-500 text-white font-black rounded-xl transition-all active:scale-95 cursor-pointer shadow-md mt-2"
            >
              🚨 Review Invalidation & Breaches
            </button>
          ) : insight.terminalState.posture === "RESEARCH" ? (
            <button
              onClick={onOpenWhy}
              className="w-full py-2 bg-purple-600 hover:bg-purple-500 text-white font-black rounded-xl transition-all active:scale-95 cursor-pointer shadow-md mt-2"
            >
              📋 Open Quantitative Evidence Ledger
            </button>
          ) : insight.terminalState.posture === "AVOID" ? (
            <a
              href="/screener"
              className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-black rounded-xl transition-all active:scale-95 cursor-pointer shadow-md mt-2 block text-center"
            >
              🔎 Explore Screened Opportunities
            </a>
          ) : (
            <button
              onClick={onOpenWhy}
              className="w-full py-2 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-black rounded-xl transition-all active:scale-95 cursor-pointer shadow-md mt-2"
            >
              ⏳ Monitor Technical Triggers
            </button>
          )}
        </div>

        {/* Quant Statistics & Fundamentals */}
        <div className="lg:col-span-2 bg-[#0b101b] border border-[#1d293d] rounded-2xl p-4 shadow-xl space-y-3">
          <div className="flex items-center justify-between border-b border-[#182335] pb-2">
            <span className="text-slate-300 font-bold">Quant Statistics & Factor Loadings</span>
            <button
              onClick={onOpenWhy}
              className="text-[11px] text-cyan-400 hover:text-cyan-300 underline font-bold cursor-pointer"
            >
              Inspect Attribution Model
            </button>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
            <div className="bg-[#070b13] p-2.5 rounded-lg border border-[#182436]">
              <span className="text-[10px] text-slate-500 block">MARKET CAP</span>
              <span className="text-xs font-bold text-white mt-0.5 block">{adv.marketCap || "N/A"}</span>
            </div>

            <div className="bg-[#070b13] p-2.5 rounded-lg border border-[#182436]">
              <span className="text-[10px] text-slate-500 block">PE RATIO (TTM)</span>
              <span className="text-xs font-bold text-white mt-0.5 block">
                {adv.peRatio !== undefined ? `${adv.peRatio.toFixed(1)}×` : "N/A"}
              </span>
            </div>

            <div className="bg-[#070b13] p-2.5 rounded-lg border border-[#182436]">
              <span className="text-[10px] text-slate-500 block">ROIC (GREENBLATT)</span>
              <span className="text-xs font-bold text-emerald-400 mt-0.5 block">
                {adv.roic !== undefined ? `${adv.roic.toFixed(1)}%` : "N/A"}
              </span>
            </div>

            <div className="bg-[#070b13] p-2.5 rounded-lg border border-[#182436]">
              <span className="text-[10px] text-slate-500 block">DEBT / EQUITY</span>
              <span className="text-xs font-bold text-white mt-0.5 block">
                {adv.debtToEquity !== undefined ? adv.debtToEquity.toFixed(2) : "N/A"}
              </span>
            </div>

            <div className="bg-[#070b13] p-2.5 rounded-lg border border-[#182436]">
              <span className="text-[10px] text-slate-500 block">RELATIVE STRENGTH</span>
              <span className="text-xs font-bold text-cyan-400 mt-0.5 block">{adv.relativeStrengthScore || 65}/100</span>
            </div>

            <div className="bg-[#070b13] p-2.5 rounded-lg border border-[#182436]">
              <span className="text-[10px] text-slate-500 block">CORNISH-FISHER VAR</span>
              <span className="text-xs font-bold text-rose-400 mt-0.5 block">-{adv.var95Pct || 3.2}% (95% 1D)</span>
            </div>
          </div>

          <div className="p-3 bg-[#06090f] border border-[#151f2e] rounded-xl text-slate-300 text-xs space-y-1 font-sans">
            <span className="text-slate-400 font-mono font-bold text-[10px] uppercase block">
              Multi-Factor Archetype Classification
            </span>
            <p className="text-slate-300 leading-relaxed font-mono text-[11px]">
              VCP Structure: {adv.vcpStage ? `Stage ${adv.vcpStage} Contraction` : "Stage 4 Correction"} | 50 SMA: {adv.sma50 !== undefined ? `$${adv.sma50.toFixed(2)}` : "N/A"} | Pivot Proximity: -10.3%
            </p>
          </div>
        </div>
      </div>

      <FinancialDisclaimer variant="compact" />
    </div>
  );
}
