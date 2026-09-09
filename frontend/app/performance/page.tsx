"use client";

import React from "react";
import Link from "next/link";
import TerminalShell from "../../components/terminal/TerminalShell";

export default function PerformancePage() {
  const proofMetrics = {
    capitalPreserved: 6140,
    drawdownWithARX: -8.4,
    drawdownUnconstrained: -19.2,
    interventionsTotal: 31,
    winRateWithARX: 58.3,
    winRateUnconstrained: 51.2,
    profitFactor: 2.14,
  };

  const equityCurvePoints = [
    { trade: 'T0', actual: 50000, unconstrained: 50000 },
    { trade: 'T10', actual: 54200, unconstrained: 52100 },
    { trade: 'T20', actual: 52800, unconstrained: 48900 },
    { trade: 'T30', actual: 59100, unconstrained: 53400 },
    { trade: 'T40', actual: 64800, unconstrained: 58200 },
  ];

  return (
    <TerminalShell activeHub="performance">
      <div className="space-y-6">
        {/* Proof of Edge Headline Card */}
        <div className="p-6 rounded-2xl bg-gradient-to-r from-emerald-950/40 via-slate-900 to-slate-900 border border-emerald-800/50 flex flex-col md:flex-row md:items-center justify-between gap-6 shadow-xl">
          <div className="space-y-1">
            <span className="text-xs font-mono uppercase font-bold text-emerald-400 bg-emerald-950 px-2.5 py-0.5 rounded border border-emerald-800">
              Counterfactual Proof of Edge
            </span>
            <h2 className="text-3xl font-extrabold text-white tracking-tight">
              +${proofMetrics.capitalPreserved.toLocaleString()} Capital Preserved
            </h2>
            <p className="text-xs text-slate-300">
              Direct dollar outperformance achieved by the Governor dynamically clamping sizing during losing streaks and late-day sessions.
            </p>
          </div>

          <div className="flex items-center space-x-6 bg-slate-950/80 border border-slate-800 p-4 rounded-xl font-mono text-center">
            <div>
              <span className="text-[10px] text-slate-400 uppercase block">Max Drawdown</span>
              <span className="text-lg font-bold text-emerald-400">{proofMetrics.drawdownWithARX}%</span>
              <span className="text-[10px] text-slate-500 block">vs. {proofMetrics.drawdownUnconstrained}% unclamped</span>
            </div>
            <div className="h-8 w-px bg-slate-800" />
            <div>
              <span className="text-[10px] text-slate-400 uppercase block">Win Rate</span>
              <span className="text-lg font-bold text-cyan-400">{proofMetrics.winRateWithARX}%</span>
              <span className="text-[10px] text-slate-500 block">vs. {proofMetrics.winRateUnconstrained}% unclamped</span>
            </div>
          </div>
        </div>

        {/* Counterfactual Equity Comparison */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 p-6 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-white">Counterfactual Equity Trajectory</h3>
              <span className="text-xs font-mono text-emerald-400 font-semibold">+11.3% Alpha Spread</span>
            </div>

            <div className="h-48 rounded-xl bg-slate-950 border border-slate-800/80 flex items-center justify-center p-4">
              <div className="text-center space-y-2 font-mono">
                <span className="text-2xl">📈</span>
                <p className="text-sm text-slate-200">Actual Equity: $64,800 · Unconstrained: $58,200</p>
                <p className="text-xs text-slate-500">Drawdown reduced by 10.8 percentage points across 40 audited trades.</p>
              </div>
            </div>
          </div>

          <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-4">
            <h3 className="text-base font-bold text-white">Governor Interventions</h3>
            <div className="space-y-3 text-xs font-mono">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex justify-between">
                <span className="text-slate-400">Total Clamps:</span>
                <span className="text-white font-bold">{proofMetrics.interventionsTotal} Setups</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex justify-between">
                <span className="text-slate-400">Adherence Rate:</span>
                <span className="text-emerald-400 font-bold">100%</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex justify-between">
                <span className="text-slate-400">Profit Factor:</span>
                <span className="text-cyan-400 font-bold">{proofMetrics.profitFactor}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </TerminalShell>
  );
}
