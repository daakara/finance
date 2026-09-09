"use client";

import React from "react";
import Link from "next/link";
import TerminalShell from "../../components/terminal/TerminalShell";

export default function JournalPage() {
  const disciplineSummary = {
    adherenceRatePct: 94.2,
    tradesLogged: 48,
    brierCalibrationScore: 0.18,
    lossStreakState: 'NORMAL (0 Active Losses)',
    revengeTradingAlert: 'NONE (Nominal State)',
  };

  const tradeLogs = [
    { id: 'TR-108', ticker: 'GOOGL', date: '2026-09-08', setup: 'VCP 4T', rAchieved: 2.1, followedRules: true, pnl: '+$1,050' },
    { id: 'TR-107', ticker: 'NVDA', date: '2026-09-05', setup: 'RS Breakout', rAchieved: -1.0, followedRules: true, pnl: '-$375' },
    { id: 'TR-106', ticker: 'ANET', date: '2026-09-02', setup: '20-EMA Bounce', rAchieved: 2.4, followedRules: true, pnl: '+$1,200' },
    { id: 'TR-105', ticker: 'MSFT', date: '2026-08-28', setup: 'Base Consolidation', rAchieved: 1.8, followedRules: true, pnl: '+$900' },
    { id: 'TR-104', ticker: 'PLTR', date: '2026-08-22', setup: 'Smart Money Breakout', rAchieved: 2.8, followedRules: true, pnl: '+$1,450' },
    { id: 'TR-103', ticker: 'AMD', date: '2026-08-18', setup: 'VCP 3T Pivot', rAchieved: -0.8, followedRules: true, pnl: '-$320' },
  ];

  // Brier Calibration Buckets (Predicted vs Observed)
  const calibrationBuckets = [
    { conviction: '50-60%', predicted: 55, observed: 58, count: 12 },
    { conviction: '60-70%', predicted: 65, observed: 67, count: 18 },
    { conviction: '70-80%', predicted: 75, observed: 74, count: 14 },
    { conviction: '80-90%', predicted: 85, observed: 82, count: 4 },
  ];

  return (
    <TerminalShell activeHub="journal">
      <div className="space-y-6">
        {/* Level 0: Asymmetric Discipline Status Hero */}
        <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 p-5 md:p-6 shadow-2xl space-y-4">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider font-bold bg-emerald-950/80 text-emerald-400 border border-emerald-800/80">
                  Level 0 · Operational Discipline
                </span>
                <span className="text-xs text-slate-400 font-sans">
                  Did I execute according to my verified statistical edge?
                </span>
              </div>
              <div className="flex items-baseline gap-3">
                <span className="text-3xl sm:text-4xl font-black font-mono text-emerald-400 tabular-nums">
                  {disciplineSummary.adherenceRatePct}%
                </span>
                <span className="text-sm font-mono text-emerald-300/80 font-bold">
                  Rule Adherence Score (Grade A)
                </span>
              </div>
              <p className="text-xs text-slate-300 font-sans max-w-2xl leading-relaxed">
                Execution discipline intact across {disciplineSummary.tradesLogged} logged trades. Zero stop loss violations detected, with strict &le; 1.0R loss containment and calibrated probability assessments.
              </p>
            </div>

            {/* Behavioral State Cluster */}
            <div className="grid grid-cols-2 gap-3 shrink-0 font-mono text-xs">
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase block">Behavioral State</span>
                <span className="text-base font-bold text-emerald-400">CALM</span>
                <span className="text-[10px] text-slate-500 block mt-0.5">Zero Tilt Detected</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase block">Brier Calibration</span>
                <span className="text-base font-bold text-cyan-400 tabular-nums">{disciplineSummary.brierCalibrationScore}</span>
                <span className="text-[10px] text-slate-500 block mt-0.5">&le; 0.25 (Calibrated)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Level 1: Brier Calibration Curve & 4-Quadrant Anti-Tilt Matrix */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 font-mono text-xs">
          {/* Brier Probabilistic Calibration Curve */}
          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-4 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <span className="text-xs font-bold text-white uppercase">Probabilistic Calibration Curve</span>
                <p className="text-[11px] text-slate-400 font-sans mt-0.5">Comparing subjective trader conviction vs realized win rate</p>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950 text-cyan-400 border border-cyan-800">
                Brier: {disciplineSummary.brierCalibrationScore}
              </span>
            </div>

            <div className="space-y-3 pt-1">
              {calibrationBuckets.map((bucket) => (
                <div key={bucket.conviction} className="space-y-1">
                  <div className="flex justify-between text-[11px]">
                    <span className="text-slate-400">{bucket.conviction} Conviction ({bucket.count} trades):</span>
                    <span className="text-white font-bold">Predicted {bucket.predicted}% &rarr; Observed {bucket.observed}%</span>
                  </div>
                  <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden flex">
                    <div
                      className="h-full bg-cyan-500 rounded-full transition-all"
                      style={{ width: `${bucket.observed}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <p className="text-[10px] text-slate-500 font-sans pt-1">
              Target Brier Score &le; 0.25 indicates well-calibrated odds where stated confidence accurately matches empirical win rates.
            </p>
          </div>

          {/* 4-Quadrant Anti-Tilt Behavioral Matrix */}
          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-4 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <span className="text-xs font-bold text-white uppercase">4-Quadrant Anti-Tilt Monitor</span>
                <p className="text-[11px] text-slate-400 font-sans mt-0.5">Live telemetry on psychological biases and tilt drivers</p>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-800">
                Status: Nominal
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
                <span className="text-[10px] text-slate-400 uppercase block">Active Loss Streak</span>
                <span className="text-base font-bold text-emerald-400">0 Losses</span>
                <span className="text-[10px] text-slate-500 block">Clamp triggers at 2 losses</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
                <span className="text-[10px] text-slate-400 uppercase block">Execution Window</span>
                <span className="text-base font-bold text-cyan-400">100% Adherence</span>
                <span className="text-[10px] text-slate-500 block">Morning Prime strictly followed</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
                <span className="text-[10px] text-slate-400 uppercase block">Loss Containment</span>
                <span className="text-base font-bold text-purple-400">&le; 1.0R</span>
                <span className="text-[10px] text-slate-500 block">Zero stop losses blown past plan</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
                <span className="text-[10px] text-slate-400 uppercase block">Revenge Trading</span>
                <span className="text-base font-bold text-emerald-400">Nominal</span>
                <span className="text-[10px] text-slate-500 block">Zero emotional re-entries</span>
              </div>
            </div>

            <p className="text-[10px] text-slate-500 font-sans pt-1">
              Behavioral telemetry is synchronized with the Behavioral Governor to enforce pre-trade sizing clamps automatically.
            </p>
          </div>
        </div>

        {/* Level 2: Execution Discipline Ledger */}
        <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-4 shadow-xl">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-bold text-white">Execution Discipline Ledger</h3>
              <p className="text-xs text-slate-400 font-sans mt-0.5">
                Audited chronological log of recent setup executions and rule verification stamps
              </p>
            </div>
            <span className="text-xs font-mono text-slate-400">{tradeLogs.length} Recent Trades Audited</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 text-[10px] uppercase tracking-wider">
                  <th className="pb-3">Trade ID</th>
                  <th className="pb-3">Date</th>
                  <th className="pb-3">Ticker</th>
                  <th className="pb-3">Setup Archetype</th>
                  <th className="pb-3 text-center">R-Multiple</th>
                  <th className="pb-3 text-center">Rule Verification</th>
                  <th className="pb-3 text-right">Realized P&amp;L</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {tradeLogs.map((log) => (
                  <tr key={log.id} className="text-slate-300 hover:bg-slate-900/60 transition-colors">
                    <td className="py-3 font-semibold text-white">{log.id}</td>
                    <td className="py-3 text-slate-400">{log.date}</td>
                    <td className="py-3 font-bold text-white">{log.ticker}</td>
                    <td className="py-3 text-slate-300">{log.setup}</td>
                    <td className={`py-3 text-center font-bold ${log.rAchieved >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {log.rAchieved > 0 ? `+${log.rAchieved}R` : `${log.rAchieved}R`}
                    </td>
                    <td className="py-3 text-center">
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
                        VERIFIED
                      </span>
                    </td>
                    <td className={`py-3 text-right font-bold ${log.pnl.startsWith('+') ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {log.pnl}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </TerminalShell>
  );
}
