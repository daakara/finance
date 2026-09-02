"use client";

import React from "react";
import { QuantitativeInsight } from "../../types/insight";

interface StandardTerminalViewProps {
  insight: QuantitativeInsight;
  onOpenSizer: () => void;
  onOpenWhy: () => void;
}

export default function StandardTerminalView({
  insight,
  onOpenSizer,
  onOpenWhy,
}: StandardTerminalViewProps) {
  const kl = insight.standard.keyLevels;

  return (
    <div className="space-y-4 font-sans text-slate-100 animate-fade-in">
      {/* Standard Badge Banner */}
      <div className="flex items-center justify-between bg-cyan-950/40 border border-cyan-800/50 p-2.5 rounded-xl text-xs font-mono">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400" />
          <span className="text-cyan-300 font-bold">🔵 STANDARD EXPERIENCE</span>
          <span className="text-slate-400 hidden sm:inline">• Confluence signals, key levels & decision triggers</span>
        </div>
        <button
          onClick={onOpenWhy}
          className="text-[11px] text-cyan-400 hover:text-cyan-300 underline font-bold cursor-pointer"
        >
          Why Score {insight.setupScore}? →
        </button>
      </div>

      {/* Main Standard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left Column: Bottom Line & Confluence Bars */}
        <div className="lg:col-span-2 bg-[#0b101b] border border-[#1d293d] rounded-2xl p-4 sm:p-5 shadow-xl space-y-4">
          <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[#182335] pb-3">
            <div>
              <span className="text-[10px] text-slate-400 font-mono font-bold uppercase tracking-wider block">
                ARX Bottom Line
              </span>
              <h2 className="text-base sm:text-xl font-black text-white tracking-tight mt-0.5">
                {insight.verdictLabel}
              </h2>
              <p className="text-xs text-slate-300 mt-1 font-sans">
                {insight.standard.bottomLine}
              </p>
            </div>

            <div
              onClick={onOpenWhy}
              className="flex flex-col items-center justify-center p-2.5 bg-[#06090f] border border-[#24334b] hover:border-cyan-500 rounded-xl cursor-pointer transition-all shrink-0"
            >
              <span className="text-[10px] text-slate-400 font-mono">Setup Score</span>
              <span className="text-xl font-black text-cyan-400 font-mono">{insight.setupScore}/100</span>
              <span className="text-[9px] text-slate-500 font-mono">{insight.standard.signalsRatio}</span>
            </div>
          </div>

          {/* Confluence Breakdown */}
          <div className="space-y-2.5">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-300 font-bold">Confluence Breakdown</span>
              <span className="text-slate-500 text-[10px]">Independent Quant Models</span>
            </div>

            <div className="space-y-2">
              {insight.standard.confluenceBreakdown.map((bar, idx) => (
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

          {/* Setup Summary */}
          <div className="p-3 bg-[#070c14] border border-[#182436] rounded-xl text-xs font-mono space-y-1">
            <span className="text-slate-500 uppercase font-bold text-[10px] block">Setup Structure</span>
            <span className="text-amber-300 font-bold block">{insight.standard.setupSummary}</span>
          </div>
        </div>

        {/* Right Column: Key Levels & Profit/Risk */}
        <div className="bg-[#0b101b] border border-[#1d293d] rounded-2xl p-4 sm:p-5 shadow-xl space-y-3 font-mono text-xs flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center justify-between border-b border-[#182335] pb-2">
              <span className="text-slate-400 font-bold">Key Levels</span>
              <span className="text-cyan-400 text-[11px] font-bold">Execution Map</span>
            </div>

            <div className="space-y-1.5 divide-y divide-[#151f2e]">
              <div className="flex items-center justify-between pt-1">
                <span className="text-slate-400">Current Price:</span>
                <span className="text-white font-bold">${kl.currentPrice.toFixed(2)}</span>
              </div>
              <div className="flex items-center justify-between pt-1.5">
                <span className="text-slate-400">Watch Zone:</span>
                <span className="text-amber-300 font-bold">{kl.watchZone}</span>
              </div>
              <div className="flex items-center justify-between pt-1.5">
                <span className="text-slate-400">50D SMA Floor:</span>
                <span className="text-cyan-300 font-bold">${kl.sma50.toFixed(2)}</span>
              </div>
              <div className="flex items-center justify-between pt-1.5">
                <span className="text-rose-400">Stop Loss:</span>
                <span className="text-rose-300 font-bold">${kl.stopLoss.toFixed(2)} ({kl.stopLossPct}%)</span>
              </div>
              <div className="flex items-center justify-between pt-1.5">
                <span className="text-emerald-400">Target 1:</span>
                <span className="text-emerald-300 font-bold">${kl.target1.toFixed(2)} (+{kl.target1Pct}%)</span>
              </div>
              <div className="flex items-center justify-between pt-1.5">
                <span className="text-emerald-400">Target 2:</span>
                <span className="text-emerald-300 font-bold">${kl.target2.toFixed(2)} (+{kl.target2Pct}%)</span>
              </div>
            </div>

            <div className="bg-[#070b13] p-2.5 rounded-xl border border-[#1b2639] flex items-center justify-between">
              <span className="text-slate-400 font-bold">Profit / Risk:</span>
              <span className="text-emerald-400 font-bold text-sm">{kl.profitRiskRatio.toFixed(2)} : 1.0</span>
            </div>
          </div>

          <div className="pt-2 space-y-2">
            <button
              onClick={onOpenSizer}
              className="w-full py-2 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold font-mono rounded-xl text-xs transition-all active:scale-95 cursor-pointer shadow-md"
            >
              ⚖️ Institutional Position Sizer
            </button>
            <button
              onClick={onOpenWhy}
              className="w-full py-1.5 bg-[#090e18] hover:bg-[#141e2e] border border-[#1c283c] text-slate-300 font-bold font-mono rounded-xl text-xs transition-all cursor-pointer"
            >
              🔍 Inspect Why Score {insight.setupScore}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
