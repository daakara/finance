"use client";

import React from "react";
import Link from "next/link";
import { getUnifiedCockpitState } from "../../../lib/simulation/unifiedCockpitStore";

export default function SimulationWorkbench() {
  const state = getUnifiedCockpitState();
  const { triad, runway, futurePaths, outcomeForecasts } = state;

  return (
    <main className="min-h-screen bg-[#070b12] text-gray-100 p-4 md:p-8 space-y-8 max-w-7xl mx-auto">
      <header className="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-gray-800 gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <Link href="/future" className="text-xs font-mono text-gray-400 hover:text-white">
              ← Return to /future
            </Link>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-blue-950 text-blue-300 border border-blue-800">
              Specialist Workbench
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white mt-1">
            Simulation & Trajectories Workbench
          </h1>
          <p className="text-xs text-gray-400 mt-0.5">
            Multi-year Monte Carlo trajectories, macroeconomic stress-testing, and future states.
          </p>
        </div>

        <div className="flex items-center space-x-3 bg-gray-900 border border-gray-800 p-3 rounded-xl">
          <div className="text-center px-3 border-r border-gray-800">
            <span className="text-[10px] uppercase font-mono text-gray-400 block">LHI</span>
            <span className="text-xl font-mono font-bold text-emerald-400">{triad.lhi}</span>
          </div>
          <div className="text-center px-3 border-r border-gray-800">
            <span className="text-[10px] uppercase font-mono text-gray-400 block">HHI</span>
            <span className="text-xl font-mono font-bold text-blue-400">{triad.hhi}</span>
          </div>
          <div className="text-center px-3">
            <span className="text-[10px] uppercase font-mono text-gray-400 block">IAI</span>
            <span className="text-xl font-mono font-bold text-purple-400">{triad.iai}</span>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-6 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase font-mono">Runway Buffer Shield</h3>
          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between border-b border-gray-800 pb-2">
              <span className="text-gray-400">Months Unencumbered</span>
              <span className="text-emerald-400 font-bold">{runway.monthsUnencumbered} Months</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 pb-2">
              <span className="text-gray-400">Liquid Cash Reserves</span>
              <span className="text-white">${runway.liquidReserves.toLocaleString()}</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 pb-2">
              <span className="text-gray-400">Monthly Burn Rate</span>
              <span className="text-white">${runway.burnRateMonthly.toLocaleString()}/mo</span>
            </div>
            <p className="text-gray-400 pt-2 font-sans">{runway.capitalFloorRule}</p>
          </div>
        </div>

        <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-6 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase font-mono">3-Year Trajectory Paths</h3>
          <div className="space-y-3">
            {futurePaths.map((p) => (
              <div key={p.id} className="p-3 rounded-lg bg-gray-950 border border-gray-800 text-xs space-y-1">
                <div className="flex justify-between font-mono">
                  <span className="text-white font-bold">{p.name}</span>
                  <span className="text-cyan-400">{Math.round(p.probability * 100)}% Prob</span>
                </div>
                <p className="text-gray-400">Expected Net Worth: <strong className="text-emerald-400">{p.expectedNetWorth3Yr}</strong> • Downside buffer: {p.downsideBufferMonths} mo</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}
