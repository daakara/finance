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
  ];

  return (
    <TerminalShell activeHub="journal">
      <div className="space-y-6">
        {/* Top Discipline Dashboard */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-[10px] uppercase font-mono text-slate-400">Rule Adherence</span>
            <p className="text-2xl font-mono font-bold text-emerald-400">{disciplineSummary.adherenceRatePct}%</p>
            <p className="text-xs text-slate-500">45 of 48 trades strictly followed rules.</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-[10px] uppercase font-mono text-slate-400">Brier Calibration</span>
            <p className="text-2xl font-mono font-bold text-cyan-400">{disciplineSummary.brierCalibrationScore}</p>
            <p className="text-xs text-slate-500">Target $\le 0.25$ indicates well-calibrated odds.</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-[10px] uppercase font-mono text-slate-400">Anti-Tilt Monitor</span>
            <p className="text-2xl font-mono font-bold text-white">CALM</p>
            <p className="text-xs text-slate-500">{disciplineSummary.revengeTradingAlert}</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-[10px] uppercase font-mono text-slate-400">Loss Containment</span>
            <p className="text-2xl font-mono font-bold text-purple-400">&le; 1.0R</p>
            <p className="text-xs text-slate-500">Zero stop losses blown past plan.</p>
          </div>
        </div>

        {/* Execution Log Table */}
        <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-white">Execution Discipline Ledger</h3>
            <span className="text-xs font-mono text-slate-400">{tradeLogs.length} Recent Trades Audited</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="pb-2">Trade ID</th>
                  <th className="pb-2">Date</th>
                  <th className="pb-2">Ticker</th>
                  <th className="pb-2">Setup</th>
                  <th className="pb-2">R-Multiple</th>
                  <th className="pb-2">Rule Followed</th>
                  <th className="pb-2">Actual PnL</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {tradeLogs.map((log) => (
                  <tr key={log.id} className="text-slate-300">
                    <td className="py-3 font-semibold text-white">{log.id}</td>
                    <td className="py-3 text-slate-400">{log.date}</td>
                    <td className="py-3 font-bold text-cyan-400">{log.ticker}</td>
                    <td className="py-3">{log.setup}</td>
                    <td className={`py-3 font-bold ${log.rAchieved >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {log.rAchieved > 0 ? `+${log.rAchieved}R` : `${log.rAchieved}R`}
                    </td>
                    <td className="py-3">
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-800">
                        VERIFIED
                      </span>
                    </td>
                    <td className={`py-3 font-bold ${log.pnl.startsWith('+') ? 'text-emerald-400' : 'text-rose-400'}`}>
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
