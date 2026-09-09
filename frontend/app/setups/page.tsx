"use client";

import React, { useState } from 'react';
import Link from 'next/link';
import {
  CANONICAL_TACTICAL_SETUPS,
  calculateGovernedPositionSize,
  TraderContext,
} from '../../lib/simulation/governorSizingEngine';

export default function SetupsPage() {
  const [traderContext] = useState<TraderContext>({
    accountEquity: 100000,
    standardRiskBudgetPct: 0.005, // $500 standard risk
    consecutiveLossStreak: 2,     // Triggers -25% loss-streak clamp
    tradingHour: 15,              // 3:00 PM (Afternoon window: additional -20% clamp)
    liquidRunwayMonths: 14.2,     // Unencumbered cash safe
    dailyDrawdownPct: 0.008,
  });

  const [committedList, setCommittedList] = useState<string[]>([]);

  const handleCommit = (ticker: string) => {
    setCommittedList((prev) => [...prev, ticker]);
    alert(`Trade setup for ${ticker} committed! Order forwarded to /portfolio with trailing stop activated.`);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 font-sans">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
          <div>
            <div className="text-xs font-mono font-semibold text-cyan-400 mb-1">
              ARX Terminal / Tactical Entries
            </div>
            <h1 className="text-2xl font-black tracking-tight text-white flex items-center gap-2">
              <span>Actionable Setups</span>
              <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/80">
                Live Pivot Engine
              </span>
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Exact pivot triggers, defined stop-loss levels, and dynamic position sizing with Clean Room Behavioral Risk Governance.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Link
              href="/radar"
              className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-xs font-mono text-slate-300 border border-slate-800 transition-colors"
            >
              ← Back to Radar
            </Link>
            <Link
              href="/portfolio"
              className="px-3 py-1.5 rounded-lg bg-cyan-950/80 hover:bg-cyan-900/80 text-xs font-mono text-cyan-300 border border-cyan-700/80 transition-colors"
            >
              Open Portfolio →
            </Link>
          </div>
        </div>

        {/* Setups List */}
        <div className="space-y-6">
          {CANONICAL_TACTICAL_SETUPS.map((setup) => {
            const sizing = calculateGovernedPositionSize(setup, traderContext);
            const isCommitted = committedList.includes(setup.ticker);

            return (
              <div
                key={setup.ticker}
                className="p-6 rounded-xl border border-slate-800 bg-gradient-to-br from-slate-900 to-slate-950 shadow-md space-y-5"
              >
                {/* Top Setup Row */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800/80 pb-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xl font-black text-white font-mono">{setup.ticker}</span>
                      <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                        {setup.setupName}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 font-mono text-xs">
                    <span className="text-slate-400">Confluence:</span>
                    <span className="px-2.5 py-0.5 rounded-full bg-cyan-950 text-cyan-300 font-bold border border-cyan-700/60">
                      {setup.confluenceScore} / 100
                    </span>
                  </div>
                </div>

                {/* Price, Stop & Target Metrics */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-center">
                  <div className="p-3 bg-slate-950/80 rounded-lg border border-slate-800">
                    <div className="text-[10px] uppercase text-slate-400">Entry Pivot</div>
                    <div className="text-lg font-bold text-white">${setup.entryPivot.toFixed(2)}</div>
                    <div className="text-[10px] text-cyan-400">Crossing Resistance</div>
                  </div>

                  <div className="p-3 bg-slate-950/80 rounded-lg border border-slate-800">
                    <div className="text-[10px] uppercase text-slate-400">Hard Stop-Loss</div>
                    <div className="text-lg font-bold text-rose-400">${setup.stopLoss.toFixed(2)}</div>
                    <div className="text-[10px] text-rose-300">-{sizing.stopDistancePct}% Risk</div>
                  </div>

                  <div className="p-3 bg-slate-950/80 rounded-lg border border-slate-800">
                    <div className="text-[10px] uppercase text-slate-400">Target 1 (2.0R)</div>
                    <div className="text-lg font-bold text-emerald-400">${setup.target1.toFixed(2)}</div>
                    <div className="text-[10px] text-emerald-300">+{sizing.rMultipleTarget1}R Skew</div>
                  </div>

                  <div className="p-3 bg-slate-950/80 rounded-lg border border-slate-800">
                    <div className="text-[10px] uppercase text-slate-400">Target 2 (3.8R)</div>
                    <div className="text-lg font-bold text-emerald-400">${setup.target2.toFixed(2)}</div>
                    <div className="text-[10px] text-emerald-300">+{sizing.rMultipleTarget2}R Skew</div>
                  </div>
                </div>

                {/* Behavioral Sizing Governor Ticket */}
                <div className="p-4 rounded-lg bg-slate-950 border border-cyan-900/40 space-y-3 font-mono">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
                    <div className="text-slate-300 font-bold uppercase tracking-wider flex items-center gap-1.5">
                      <span className="h-2 w-2 rounded-full bg-cyan-400" />
                      <span>Behavioral Sizing Governor Allocation</span>
                    </div>
                    <div className="text-cyan-400 font-bold">
                      Clamp: {sizing.clampFactorPct}% Risk
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs pt-1">
                    <div>
                      <div className="text-slate-400">Recommended Shares:</div>
                      <div className="text-base font-black text-white">{sizing.recommendedShares} Shares</div>
                      <div className="text-[10px] text-slate-500">Unclamped: {sizing.unclampedShares} shares</div>
                    </div>
                    <div>
                      <div className="text-slate-400">Max Dollar Risk:</div>
                      <div className="text-base font-black text-emerald-400">${sizing.recommendedDollarRisk}</div>
                      <div className="text-[10px] text-slate-500">Standard Budget: ${sizing.unclampedDollarRisk}</div>
                    </div>
                    <div>
                      <div className="text-slate-400">Capital Allocated:</div>
                      <div className="text-base font-black text-slate-200">${sizing.estimatedCapitalAllocated.toLocaleString()}</div>
                      <div className="text-[10px] text-slate-500">Account: ${(traderContext.accountEquity).toLocaleString()}</div>
                    </div>
                  </div>

                  {/* Clean Room Rationale Banner (INV-OI112-P Compliant) */}
                  <div className="p-2.5 rounded bg-slate-900/90 border border-slate-800 text-[11px] text-slate-300 leading-relaxed">
                    <span className="text-amber-400 font-bold uppercase">Governor Defense Notice: </span>
                    {sizing.cleanRoomRationale}
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex items-center justify-between pt-2">
                  <div className="text-[11px] text-slate-400 font-mono">
                    Underlying stop distance: ${sizing.stopDistanceDollar} per share
                  </div>
                  <button
                    disabled={isCommitted}
                    onClick={() => handleCommit(setup.ticker)}
                    className={`px-5 py-2 rounded-lg font-mono font-bold text-xs transition-colors shadow-sm ${
                      isCommitted
                        ? 'bg-emerald-950 text-emerald-400 border border-emerald-800 cursor-default'
                        : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950'
                    }`}
                  >
                    {isCommitted ? '✓ Trade Committed' : `Commit to ${setup.ticker} Setup`}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
