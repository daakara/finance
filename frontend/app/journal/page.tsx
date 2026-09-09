"use client";

import React from 'react';
import Link from 'next/link';

interface LoggedTrade {
  id: string;
  ticker: string;
  date: string;
  entry: number;
  exit: number;
  pnlR: number;
  ruleAdhered: boolean;
  governorClamped: boolean;
  notes: string;
}

const CANONICAL_LOGGED_TRADES: LoggedTrade[] = [
  {
    id: 'tr-101',
    ticker: 'GOOGL',
    date: '2026-09-08',
    entry: 182.40,
    exit: 195.10,
    pnlR: +2.02,
    ruleAdhered: true,
    governorClamped: false,
    notes: 'Entered on 4T breakout with volume surge. Exited Target 1 seamlessly.',
  },
  {
    id: 'tr-102',
    ticker: 'NVDA',
    date: '2026-09-07',
    entry: 128.50,
    exit: 123.80,
    pnlR: -1.00,
    ruleAdhered: true,
    governorClamped: true,
    notes: 'Stopped out cleanly at 123.80. Governor clamp preserved $235 of capital.',
  },
  {
    id: 'tr-103',
    ticker: 'ANET',
    date: '2026-09-05',
    entry: 312.10,
    exit: 333.50,
    pnlR: +2.02,
    ruleAdhered: true,
    governorClamped: false,
    notes: '20-EMA institutional bounce executed inside morning peak window.',
  },
];

export default function JournalPage() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 font-sans">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
          <div>
            <div className="text-xs font-mono font-semibold text-cyan-400 mb-1">
              ARX Terminal / Trade Execution Journal
            </div>
            <h1 className="text-2xl font-black tracking-tight text-white flex items-center gap-2">
              <span>Execution Discipline &amp; Calibration Ledger</span>
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Audit rule compliance, Brier confidence scoring, and anti-tilt defense metrics.
            </p>
          </div>

          <div className="flex items-center gap-2 font-mono text-xs">
            <Link
              href="/performance"
              className="px-3 py-1.5 rounded-lg bg-emerald-950/80 hover:bg-emerald-900/80 text-emerald-300 border border-emerald-700/80 transition-colors font-bold"
            >
              View Attribution Proof →
            </Link>
          </div>
        </div>

        {/* METRICS ROW */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono">
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
            <div className="text-xs text-slate-400">Rule Compliance Index</div>
            <div className="text-3xl font-extrabold text-emerald-400 mt-1">94.2%</div>
            <div className="text-[10px] text-slate-500 mt-1">Stop discipline maintained on 24 of 25 trades</div>
          </div>

          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
            <div className="text-xs text-slate-400">Brier Calibration Fit</div>
            <div className="text-3xl font-extrabold text-cyan-400 mt-1">0.14</div>
            <div className="text-[10px] text-slate-500 mt-1">Institutional threshold: &lt; 0.20 (Well-Calibrated)</div>
          </div>

          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
            <div className="text-xs text-slate-400">Anti-Tilt Shield Status</div>
            <div className="text-3xl font-extrabold text-white mt-1">0 Violations</div>
            <div className="text-[10px] text-slate-500 mt-1">Zero revenge trades or widened stops in 30 days</div>
          </div>
        </div>

        {/* LOGGED TRADES TABLE */}
        <div className="border border-slate-800 rounded-xl bg-slate-900/40 overflow-hidden font-mono text-xs">
          <table className="w-full text-left">
            <thead className="bg-slate-950/80 border-b border-slate-800 text-slate-400 uppercase text-[11px]">
              <tr>
                <th className="p-3.5">Date &amp; Asset</th>
                <th className="p-3.5">Entry / Exit</th>
                <th className="p-3.5">R-Multiple</th>
                <th className="p-3.5">Rule Compliance</th>
                <th className="p-3.5">Governor Action</th>
                <th className="p-3.5">Audit Note</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {CANONICAL_LOGGED_TRADES.map((trade) => (
                <tr key={trade.id} className="hover:bg-slate-800/30 transition-colors">
                  <td className="p-3.5 font-bold text-white">
                    {trade.ticker}
                    <div className="text-[10px] text-slate-500 font-normal">{trade.date}</div>
                  </td>
                  <td className="p-3.5 text-slate-300">
                    ${trade.entry.toFixed(2)} → ${trade.exit.toFixed(2)}
                  </td>
                  <td className="p-3.5 font-bold">
                    <span className={trade.pnlR > 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      {trade.pnlR > 0 ? `+${trade.pnlR}R` : `${trade.pnlR}R`}
                    </span>
                  </td>
                  <td className="p-3.5">
                    <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 text-[10px] border border-emerald-800">
                      Rule Followed
                    </span>
                  </td>
                  <td className="p-3.5">
                    {trade.governorClamped ? (
                      <span className="px-2 py-0.5 rounded bg-amber-950 text-amber-300 text-[10px] border border-amber-800">
                        Clamped -47%
                      </span>
                    ) : (
                      <span className="text-slate-500 text-[10px]">Unconstrained</span>
                    )}
                  </td>
                  <td className="p-3.5 text-slate-300 text-[11px] max-w-sm truncate">{trade.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
