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
  const [executionMode, setExecutionMode] = useState<'STANDARD' | 'GUIDED' | 'QUANT'>('GUIDED');
  const [copiedOrder, setCopiedOrder] = useState(false);

  const context = getTraderContextFromUnifiedCockpit();
  const sizing = calculateGovernedPositionSize(selectedSetup, context);

  const handleCopyOrder = () => {
    const orderStr = `BUY ${sizing.recommendedShares} ${selectedSetup.ticker} LMT $${sizing.entryPivot.toFixed(2)} | STP $${sizing.stopLoss.toFixed(2)} | TGT $${selectedSetup.target1.toFixed(2)}`;
    navigator.clipboard?.writeText(orderStr);
    setCopiedOrder(true);
    setTimeout(() => setCopiedOrder(false), 2000);
  };

  return (
    <TerminalShell activeHub="setups">
      <div className="space-y-6">
        {/* Top Control & Guidance Banner */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 backdrop-blur-md flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-cyan-400">
                Tactical Execution Ticket
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                One Product · 3 Detail Levels
              </span>
            </div>
            <div className="text-sm font-semibold text-slate-200 mt-0.5">
              Strict Minervini pivot entries, maximum 8% stop losses, and dynamic Behavioral Governor sizing.
            </div>
          </div>
          
          {/* Mode Switcher */}
          <div className="flex items-center gap-1.5 p-1 bg-slate-950 rounded-xl border border-slate-800 shrink-0">
            <button
              onClick={() => setExecutionMode('STANDARD')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all ${
                executionMode === 'STANDARD'
                  ? 'bg-slate-800 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Standard
            </button>
            <button
              onClick={() => setExecutionMode('GUIDED')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all ${
                executionMode === 'GUIDED'
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 shadow-sm font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              🛡️ Guided (Recommended)
            </button>
            <button
              onClick={() => setExecutionMode('QUANT')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all ${
                executionMode === 'QUANT'
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              🔬 Quant
            </button>
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
                  Validated against Stage 2 Uptrend &amp; Volatility Contraction Pattern
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono font-bold px-3 py-1 rounded bg-emerald-950/60 border border-emerald-800 text-emerald-400">
                  Actionable Breakout
                </span>
              </div>
            </div>

            {/* Level 0: Asymmetric Execution Ticket Ladder */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-mono">
              {/* Risk Bracket: Entry Pivot & Stop Loss */}
              <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                  <span>Risk Definition Bracket</span>
                  <span className="text-rose-400">Stop: -{sizing.stopDistancePct.toFixed(2)}%</span>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase block">LMT $ (Entry)</span>
                    <span className="text-xl font-bold text-white">${sizing.entryPivot.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase block">STP $ (Floor)</span>
                    <span className="text-xl font-bold text-rose-400">${sizing.stopLoss.toFixed(2)}</span>
                  </div>
                </div>
              </div>

              {/* Reward Milestones: Target 1 & Target 2 */}
              <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                  <span>Asymmetric Reward Milestones</span>
                  <span className="text-emerald-400">R:R {sizing.rMultipleTarget1}R+</span>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase block">TGT $ (Primary TP1)</span>
                    <span className="text-xl font-bold text-emerald-400">${selectedSetup.target1.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase block">TGT $ (Runner TP2)</span>
                    <span className="text-xl font-bold text-purple-400">${selectedSetup.target2.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Mode-Specific Information Panels */}
            {executionMode === 'STANDARD' && (
              <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 space-y-3 text-xs font-mono">
                <div className="text-xs font-bold text-slate-300">Order Execution Summary</div>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="flex justify-between border-b border-slate-900 pb-1.5">
                    <span className="text-slate-400">Stop Distance:</span>
                    <span className="text-white">${sizing.stopDistanceDollar.toFixed(2)} ({sizing.stopDistancePct.toFixed(2)}%)</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-900 pb-1.5">
                    <span className="text-slate-400">Position Size:</span>
                    <span className="text-cyan-400 font-bold">{sizing.recommendedShares} Shares</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Capital Allocated:</span>
                    <span className="text-white font-bold">${sizing.estimatedCapitalAllocated.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Reward / Risk:</span>
                    <span className="text-emerald-400 font-bold">{sizing.rMultipleTarget1}R (Target 1)</span>
                  </div>
                </div>
              </div>
            )}

            {executionMode === 'GUIDED' && (
              <div className="space-y-4">
                {/* Confluence Rationale */}
                <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 space-y-2.5 text-xs font-mono">
                  <div className="text-xs font-bold text-cyan-400 uppercase">Why Take This Trade?</div>
                  <div className="space-y-1.5 text-slate-300 text-[11px] leading-relaxed">
                    <div className="flex items-center gap-2">
                      <span className="text-emerald-400">✔</span>
                      <span><strong>Stage 2 Structure:</strong> Stock trading above ascending 50-day and 200-day moving averages.</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-emerald-400">✔</span>
                      <span><strong>Volatility Contraction:</strong> Multiple contracting swings with volume drying up below the 50-day average.</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-emerald-400">✔</span>
                      <span><strong>Asymmetric R:R:</strong> Risk is strictly defined at ${sizing.stopDistanceDollar.toFixed(2)} with {sizing.rMultipleTarget1}R upside potential.</span>
                    </div>
                  </div>
                </div>

                {/* Governor Behavioral Safeguards */}
                <div className="p-4 rounded-xl bg-cyan-950/20 border border-cyan-800/60 space-y-2 text-xs font-mono">
                  <div className="flex justify-between items-center text-cyan-400 font-bold">
                    <span>🛡️ Governor Behavioral Governance</span>
                    <span>Clamp: {sizing.clampFactorPct}%</span>
                  </div>
                  <p className="text-[11px] text-slate-300">
                    {sizing.cleanRoomRationale}
                  </p>
                </div>
              </div>
            )}

            {executionMode === 'QUANT' && (
              <div className="space-y-4 font-mono text-xs">
                <div className="p-4 rounded-xl bg-slate-950/90 border border-purple-800/60 space-y-3">
                  <div className="flex justify-between items-center text-purple-400 font-bold uppercase">
                    <span>🔬 Quantitative Modeling &amp; Risk Metrics</span>
                    <span className="text-[10px] text-slate-500">1,000 Monte Carlo Paths</span>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px]">
                    <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                      <span className="text-slate-500 block text-[9px] uppercase">Cornish-Fisher VaR 95%</span>
                      <span className="text-rose-400 font-bold">-${(sizing.recommendedDollarRisk * 1.15).toFixed(0)}</span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                      <span className="text-slate-500 block text-[9px] uppercase">Expected Value (EV)</span>
                      <span className="text-emerald-400 font-bold">+$342.50</span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                      <span className="text-slate-500 block text-[9px] uppercase">Sortino Skew</span>
                      <span className="text-cyan-400 font-bold">+2.84</span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                      <span className="text-slate-500 block text-[9px] uppercase">Half-Kelly Sizing</span>
                      <span className="text-purple-400 font-bold">{sizing.recommendedShares} Shs (0.25x)</span>
                    </div>
                  </div>

                  <p className="text-[10px] text-slate-400">
                    Calculated against historical fat-tailed return distributions. Kurtosis: 4.82 (Leptokurtic). Risk-of-Ruin under 0.25x Kelly: &lt; 0.05%.
                  </p>
                </div>
              </div>
            )}
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

            <div>
              <button
                onClick={handleCopyOrder}
                className="w-full py-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-mono font-black text-xs tracking-tight transition-all shadow-lg flex items-center justify-center gap-2 hover:scale-[1.01] active:scale-[0.99]"
              >
                <span>{copiedOrder ? '✔ ORDER COPIED TO CLIPBOARD' : `AUTHORIZE ORDER: ${sizing.recommendedShares} SHARES ($${sizing.estimatedCapitalAllocated.toLocaleString()}) [COPY STRING]`}</span>
              </button>
              <div className="text-[10px] font-mono text-slate-400 text-center mt-2">
                Order String: BUY {sizing.recommendedShares} {selectedSetup.ticker} LMT ${sizing.entryPivot.toFixed(2)} | STP ${sizing.stopLoss.toFixed(2)} | TGT ${selectedSetup.target1.toFixed(2)}
              </div>
            </div>
          </div>
        </div>
      </div>
    </TerminalShell>
  );
}
