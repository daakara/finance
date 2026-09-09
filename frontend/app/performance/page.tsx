"use client";

import React from 'react';
import Link from 'next/link';

export default function PerformancePage() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 font-sans">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
          <div>
            <div className="text-xs font-mono font-semibold text-emerald-400 mb-1">
              ARX Terminal / Proof of Edge
            </div>
            <h1 className="text-2xl font-black tracking-tight text-white flex items-center gap-2">
              <span>Performance Attribution &amp; Value Engine</span>
              <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/80">
                Empirical Proof
              </span>
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Answering the core question: <em>How much has ARX improved your investing outcomes?</em>
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Link
              href="/journal"
              className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-xs font-mono text-slate-300 border border-slate-800 transition-colors"
            >
              ← Trade Journal
            </Link>
            <Link
              href="/setups"
              className="px-3 py-1.5 rounded-lg bg-cyan-950/80 hover:bg-cyan-900/80 text-xs font-mono text-cyan-300 border border-cyan-700/80 transition-colors"
            >
              Open Setups →
            </Link>
          </div>
        </div>

        {/* TOP 4 VALUE ATTRIBUTION CARDS */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 font-mono">
          <div className="p-4 rounded-xl border border-emerald-900/60 bg-gradient-to-br from-emerald-950/30 to-slate-900 shadow-sm">
            <div className="text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">
              Capital Preserved
            </div>
            <div className="text-3xl font-extrabold text-white">+$6,140.00</div>
            <p className="text-[11px] text-slate-400 mt-1">
              Dollars saved via Governor clamps during loss streaks and tilt windows.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 shadow-sm">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Drawdown Reduction
            </div>
            <div className="text-3xl font-extrabold text-cyan-400">-19.2% → -8.4%</div>
            <p className="text-[11px] text-slate-400 mt-1">
              +56.2% portfolio resilience improvement over past 120 days.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 shadow-sm">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Governor Interventions
            </div>
            <div className="text-3xl font-extrabold text-amber-400">31 Active Clamps</div>
            <p className="text-[11px] text-slate-400 mt-1">
              19 impulsive or high-drawdown entries dynamically neutralized.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 shadow-sm">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Profit Factor Lift
            </div>
            <div className="text-3xl font-extrabold text-indigo-400">1.41 → 2.34</div>
            <p className="text-[11px] text-slate-400 mt-1">
              Statistical edge compounding through strict stop adherence.
            </p>
          </div>
        </div>

        {/* COUNTERFACTUAL EQUITY CURVE: GOVERNED VS UNCLAMPED */}
        <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h2 className="text-base font-bold text-white font-mono">
                Counterfactual Equity Curve (Governed vs. Unclamped Retail)
              </h2>
              <p className="text-xs text-slate-400">
                Comparing actual returns under ARX Behavioral Governance vs. theoretical PnL without risk clamping.
              </p>
            </div>
            <div className="flex items-center gap-4 text-xs font-mono">
              <div className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
                <span className="text-slate-200">ARX Governed (+18.4% / Max DD -8.4%)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-slate-500" />
                <span className="text-slate-400">Unclamped (+4.1% / Max DD -19.2%)</span>
              </div>
            </div>
          </div>

          {/* Simple ASCII / Visual Representation of Curve */}
          <div className="p-6 bg-slate-950 rounded-xl border border-slate-800/80 font-mono text-xs space-y-3">
            <div className="flex justify-between text-slate-400 text-[11px] border-b border-slate-800 pb-2">
              <span>$125,000</span>
              <span className="text-emerald-400 font-bold">ARX Trajectory: Steady Compounding, Zero Drawdown Breach</span>
            </div>
            <div className="py-4 space-y-2 text-slate-300">
              <div className="flex items-center gap-2">
                <span className="w-16 text-slate-500 text-[10px]">Month 1:</span>
                <div className="w-1/3 bg-emerald-500/80 h-3 rounded" />
                <span className="text-[11px] text-emerald-400 font-bold">+$3,200</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16 text-slate-500 text-[10px]">Month 2:</span>
                <div className="w-1/2 bg-emerald-500/80 h-3 rounded" />
                <span className="text-[11px] text-emerald-400 font-bold">+$5,800</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16 text-slate-500 text-[10px]">Month 3:</span>
                <div className="w-3/4 bg-emerald-500/80 h-3 rounded" />
                <span className="text-[11px] text-emerald-400 font-bold">+$12,400</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16 text-slate-500 text-[10px]">Month 4:</span>
                <div className="w-full bg-emerald-400 h-3 rounded shadow-sm" />
                <span className="text-[11px] text-emerald-300 font-bold">+$18,400 (Net Capital Compounded)</span>
              </div>
            </div>
          </div>
        </div>

        {/* EDGE BY SETUP & EXECUTION WINDOW */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* SETUP EDGE */}
          <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
            <h3 className="text-sm font-bold text-white font-mono uppercase tracking-wider">
              Edge By Setup Archetype
            </h3>
            <div className="space-y-2 font-mono text-xs">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
                <div>
                  <div className="font-bold text-white">Minervini VCP Breakouts</div>
                  <div className="text-[10px] text-slate-400">42 Trades · Avg Skew: +2.8R</div>
                </div>
                <div className="text-right">
                  <div className="text-emerald-400 font-bold">68.4% Win Rate</div>
                  <div className="text-[10px] text-slate-500">Profit Factor: 3.41</div>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
                <div>
                  <div className="font-bold text-white">High-RS Gap Continuations</div>
                  <div className="text-[10px] text-slate-400">24 Trades · Avg Skew: +2.1R</div>
                </div>
                <div className="text-right">
                  <div className="text-cyan-400 font-bold">54.2% Win Rate</div>
                  <div className="text-[10px] text-slate-500">Profit Factor: 2.10</div>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
                <div>
                  <div className="font-bold text-slate-300">Mean-Reversion Dip Buys</div>
                  <div className="text-[10px] text-slate-500">18 Trades · Negative Skew</div>
                </div>
                <div className="text-right">
                  <div className="text-rose-400 font-bold">27.7% Win Rate</div>
                  <div className="text-[10px] text-rose-300/80">Auto-Clamped by Governor</div>
                </div>
              </div>
            </div>
          </div>

          {/* TIME OF DAY WINDOWS */}
          <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
            <h3 className="text-sm font-bold text-white font-mono uppercase tracking-wider">
              Trader Execution Windows
            </h3>
            <div className="space-y-2 font-mono text-xs">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
                <div>
                  <div className="font-bold text-white">09:30 – 11:30 (Morning Peak)</div>
                  <div className="text-[10px] text-slate-400">High Poise · 94.1% Rule Adherence</div>
                </div>
                <div className="text-right">
                  <div className="text-emerald-400 font-bold">72.5% Win Rate</div>
                  <div className="text-[10px] text-emerald-300">Full Sizing Allowed</div>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
                <div>
                  <div className="font-bold text-white">11:30 – 14:00 (Mid-Day Consolidation)</div>
                  <div className="text-[10px] text-slate-400">Moderate Volatility · 83.3% Adherence</div>
                </div>
                <div className="text-right">
                  <div className="text-cyan-400 font-bold">50.0% Win Rate</div>
                  <div className="text-[10px] text-slate-400">Standard Sizing</div>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
                <div>
                  <div className="font-bold text-slate-300">14:00 – 16:00 (Afternoon Fatigue)</div>
                  <div className="text-[10px] text-slate-500">Degraded Discipline · Elevated Chasing</div>
                </div>
                <div className="text-right">
                  <div className="text-rose-400 font-bold">28.5% Win Rate</div>
                  <div className="text-[10px] text-amber-400">Sizing Clamped (-50%)</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
