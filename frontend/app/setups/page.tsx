"use client";

import React, { useState } from 'react';
import Link from 'next/link';
import TerminalShell from '../../components/terminal/TerminalShell';
import {
  CANONICAL_TACTICAL_SETUPS,
  calculateGovernedPositionSize,
  getTraderContextFromUnifiedCockpit,
  TradeSetupSpec,
} from '../../lib/simulation/governorSizingEngine';

export default function SetupsPage() {
  const [selectedSetup, setSelectedSetup] = useState<TradeSetupSpec>(CANONICAL_TACTICAL_SETUPS[0]);
  const context = getTraderContextFromUnifiedCockpit();
  const sizing = calculateGovernedPositionSize(selectedSetup, context);

  return (
    <TerminalShell activeHub="setups">
      <div className="space-y-6">
        {/* Top Control & Guidance Banner */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 backdrop-blur-md flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="text-xs font-mono font-bold uppercase tracking-wider text-cyan-400">
              Tactical Execution Ticket
            </div>
            <div className="text-sm font-semibold text-slate-200 mt-0.5">
              Strict Minervini pivot entries, maximum 8% stop losses, and dynamic Behavioral Governor sizing.
            </div>
          </div>
          <div className="text-xs font-mono text-slate-400 bg-slate-950/80 px-3 py-1.5 rounded-lg border border-slate-800">
            Account Equity: <strong className="text-white">${context.accountEquity.toLocaleString()}</strong> · Risk: <strong>{(context.standardRiskBudgetPct * 100).toFixed(1)}%</strong>
          </div>
        </div>

        {/* Tactical Setup Selector */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {CANONICAL_TACTICAL_SETUPS.map((setup) => {
            const isSelected = setup.ticker === selectedSetup.ticker;
            return (
              <button
                key={setup.ticker}
                onClick={() => setSelectedSetup(setup)}
                className={`p-4 rounded-xl text-left transition-all border ${
                  isSelected
                    ? 'border-cyan-500/80 bg-cyan-950/20 shadow-lg shadow-cyan-950/40'
                    : 'border-slate-800 bg-slate-900/40 hover:bg-slate-900/80 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-base font-bold font-mono text-white">{setup.ticker}</span>
                  <span className="text-xs font-mono font-bold text-cyan-400">{setup.confluenceScore}/100</span>
                </div>
                <div className="text-xs text-slate-400 mt-1 font-medium">{setup.setupName}</div>
                <div className="text-xs font-mono text-slate-400 mt-2">
                  Pivot: <strong className="text-white">${setup.entryPivot.toFixed(2)}</strong> · Stop: <strong className="text-rose-400">${setup.stopLoss.toFixed(2)}</strong>
                </div>
              </button>
            );
          })}
        </div>

        {/* Selected Setup Execution Ticket & Governor Sizing */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 p-6 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div>
                <h2 className="text-xl font-bold text-white tracking-tight">
                  {selectedSetup.ticker} — {selectedSetup.setupName}
                </h2>
                <p className="text-xs font-mono text-slate-400 mt-1">
                  Validated against Stage 2 Uptrend & Volatility Contraction Pattern
                </p>
              </div>
              <span className="text-xs font-mono font-bold px-3 py-1 rounded bg-emerald-950/60 border border-emerald-800 text-emerald-400">
                Actionable Breakout
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                <span className="text-slate-400 uppercase text-[10px] block">Entry Pivot</span>
                <span className="text-base font-bold text-white">${sizing.entryPivot.toFixed(2)}</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                <span className="text-slate-400 uppercase text-[10px] block">Stop Loss</span>
                <span className="text-base font-bold text-rose-400">${sizing.stopLoss.toFixed(2)}</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                <span className="text-slate-400 uppercase text-[10px] block">Target 1 (2.0R)</span>
                <span className="text-base font-bold text-emerald-400">${selectedSetup.target1.toFixed(2)}</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                <span className="text-slate-400 uppercase text-[10px] block">Target 2 (3.8R)</span>
                <span className="text-base font-bold text-purple-400">${selectedSetup.target2.toFixed(2)}</span>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 space-y-2 text-xs font-mono">
              <div className="flex justify-between border-b border-slate-900 pb-2">
                <span className="text-slate-400">Stop Distance:</span>
                <span className="text-white">${sizing.stopDistanceDollar.toFixed(2)} ({sizing.stopDistancePct.toFixed(2)}%)</span>
              </div>
              <div className="flex justify-between border-b border-slate-900 pb-2">
                <span className="text-slate-400">Reward/Risk (T1):</span>
                <span className="text-emerald-400 font-bold">{sizing.rMultipleTarget1}R</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Reward/Risk (T2):</span>
                <span className="text-purple-400 font-bold">{sizing.rMultipleTarget2}R</span>
              </div>
            </div>
          </div>

          {/* Clean Room Governor Sizing Box */}
          <div className="p-6 rounded-2xl border border-cyan-800/60 bg-gradient-to-b from-slate-900 to-cyan-950/20 space-y-5">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider">
                  Behavioral Governor
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 border border-cyan-800 text-cyan-300">
                  {sizing.primaryGovernorCategory}
                </span>
              </div>
              <h3 className="text-lg font-bold text-white mt-1">
                Dynamic Risk Allocation
              </h3>
            </div>

            <div className="space-y-3 text-xs font-mono">
              <div className="flex justify-between p-2.5 rounded-lg bg-slate-950/60 border border-slate-800">
                <span className="text-slate-400">Standard Risk:</span>
                <span className="text-slate-300 font-bold">${sizing.unclampedDollarRisk} ({sizing.unclampedShares} Shs)</span>
              </div>
              <div className="flex justify-between p-2.5 rounded-lg bg-cyan-950/60 border border-cyan-800">
                <span className="text-cyan-300 font-semibold">Governed Risk:</span>
                <span className="text-emerald-400 font-bold">${sizing.recommendedDollarRisk} ({sizing.recommendedShares} Shs)</span>
              </div>
              <div className="flex justify-between p-2.5 rounded-lg bg-slate-950/60 border border-slate-800">
                <span className="text-slate-400">Governor Clamp:</span>
                <span className={`font-bold ${sizing.clampFactorPct < 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
                  {sizing.clampFactorPct}%
                </span>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] text-slate-300">
              {sizing.cleanRoomRationale}
            </div>

            <button className="w-full py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-mono font-bold text-xs tracking-tight transition-colors shadow-md">
              Authorize Order: {sizing.recommendedShares} Shares (${sizing.estimatedCapitalAllocated.toLocaleString()})
            </button>
          </div>
        </div>
      </div>
    </TerminalShell>
  );
}
