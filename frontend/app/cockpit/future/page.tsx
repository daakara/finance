"use client";

import CockpitShell from "../../../components/cockpit/CockpitShell";

import React from "react";
import Link from "next/link";
import { getUnifiedCockpitState } from "../../../lib/simulation/unifiedCockpitStore";
import SemanticZoom from "../../../components/cockpit/SemanticZoom";

export default function FutureHubPage() {
  const state = getUnifiedCockpitState();
  const { triad, runway, primaryForecast, futurePaths, outcomeForecasts } = state;

  return (
    <CockpitShell activeHub="future"><div className="space-y-8">
      {/* Top Header */}
      <header className="flex flex-col md:flex-row md:items-center md:justify-between pb-6 border-b border-gray-800/80 gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <span className="text-xs font-mono font-bold tracking-wider uppercase text-cyan-400 bg-cyan-950/60 border border-cyan-800/60 px-2.5 py-1 rounded">
              Core Hub 2
            </span>
            <span className="text-xs font-mono text-gray-400">Horizon 9 & 10 Future States</span>
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white mt-1">
            Future & Scenarios
          </h1>
          <p className="text-sm text-gray-400 mt-0.5">
            Multi-year trajectory sequencing, runway resilience, and outcome forecasting.
          </p>
        </div>

        {/* Global Triad Banner */}
        <div className="flex items-center space-x-3 bg-gray-900/90 border border-gray-800 rounded-xl p-3 shadow-inner">
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

      {/* Runway Shield Banner */}
      <section className="p-6 rounded-xl bg-gradient-to-r from-blue-950/40 via-gray-900 to-gray-900 border border-blue-800/50 flex flex-col md:flex-row md:items-center justify-between gap-6 shadow-xl">
        <div className="space-y-1 max-w-xl">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-mono uppercase font-bold text-cyan-400 bg-cyan-950 px-2 py-0.5 rounded border border-cyan-800">
              Runway Shield Active
            </span>
            <span className="text-xs font-mono text-emerald-400 font-semibold">{runway.runwayShieldStatus}</span>
          </div>
          <h2 className="text-2xl font-bold text-white tracking-tight">
            {runway.monthsUnencumbered} Months Unencumbered Liquidity
          </h2>
          <p className="text-xs text-gray-400">
            {runway.capitalFloorRule}
          </p>
        </div>

        <div className="flex items-center space-x-4 bg-gray-950/70 border border-gray-800 p-4 rounded-xl">
          <div>
            <span className="text-[10px] font-mono uppercase text-gray-400 block">Liquid Reserves</span>
            <span className="text-lg font-mono font-bold text-white">${runway.liquidReserves.toLocaleString()}</span>
          </div>
          <div className="h-8 w-px bg-gray-800" />
          <div>
            <span className="text-[10px] font-mono uppercase text-gray-400 block">Monthly Burn</span>
            <span className="text-lg font-mono font-bold text-gray-300">${runway.burnRateMonthly.toLocaleString()}/mo</span>
          </div>
        </div>
      </section>

      {/* Semantic Zoom Container for Future Pathways */}
      <SemanticZoom
        hubTitle="Trajectory Sequencing & Scenario Comparison"
        workbenchRoute="/workbench/simulation"
        workbenchName="Simulation & Trajectories Workbench"
        level0Content={
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {futurePaths.map((p, idx) => (
                <div
                  key={p.id}
                  className={`p-5 rounded-xl border flex flex-col justify-between space-y-4 ${
                    idx === 0
                      ? 'bg-gradient-to-b from-gray-900 to-gray-950 border-cyan-500/60 shadow-lg'
                      : 'bg-gray-900/60 border-gray-800'
                  }`}
                >
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className={idx === 0 ? 'text-cyan-400 font-bold' : 'text-gray-400'}>
                        {Math.round(p.probability * 100)}% Probability
                      </span>
                      {idx === 0 && (
                        <span className="text-[10px] bg-cyan-950 text-cyan-300 border border-cyan-800 px-2 py-0.5 rounded uppercase font-semibold">
                          Recommended
                        </span>
                      )}
                    </div>
                    <h4 className="text-base font-bold text-white">{p.name}</h4>
                    <p className="text-xs text-gray-400">{p.tradeoffs}</p>
                  </div>

                  <div className="pt-3 border-t border-gray-800/80 space-y-1">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-gray-400">3-Yr Net Worth:</span>
                      <span className="text-emerald-400 font-bold">{p.expectedNetWorth3Yr}</span>
                    </div>
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-gray-400">Identity Fulfillment:</span>
                      <span className="text-purple-400 font-semibold">{p.identityFulfillmentPct}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Primary Forecast Summary */}
            <div className="p-5 rounded-xl bg-gray-900/60 border border-gray-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono uppercase text-gray-400">Primary Forecast</span>
                <span className="text-xs font-mono text-cyan-400 font-semibold">{primaryForecast.confidencePct}% Confidence</span>
              </div>
              <h4 className="text-lg font-bold text-white">{primaryForecast.title}</h4>
              <p className="text-sm text-gray-300">
                Current: <strong className="text-white">{primaryForecast.currentValue}</strong> → Projected 3-Year: <strong className="text-emerald-400">{primaryForecast.projectedValue3Yr}</strong>. Primary driver: {primaryForecast.primaryDriver}.
              </p>
            </div>
          </div>
        }
        level1Content={
          <div className="space-y-6">
            <div className="p-5 rounded-xl bg-gray-950 border border-gray-800 space-y-4">
              <h4 className="text-sm font-bold text-white">Multi-Scenario Comparative Ledger</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-gray-800 text-gray-400">
                      <th className="pb-2">Scenario Pathway</th>
                      <th className="pb-2">Probability</th>
                      <th className="pb-2">3-Yr Expected Wealth</th>
                      <th className="pb-2">Identity Fulfillment</th>
                      <th className="pb-2">Downside Runway Buffer</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800/60">
                    {futurePaths.map((path) => (
                      <tr key={path.id} className="text-gray-300">
                        <td className="py-2.5 font-sans font-medium text-white">{path.name}</td>
                        <td className="py-2.5 font-bold text-cyan-400">{Math.round(path.probability * 100)}%</td>
                        <td className="py-2.5 text-emerald-400 font-bold">{path.expectedNetWorth3Yr}</td>
                        <td className="py-2.5 text-purple-400">{path.identityFulfillmentPct}%</td>
                        <td className="py-2.5">{path.downsideBufferMonths} Months</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {outcomeForecasts.map((f) => (
                <div key={f.id} className="p-4 rounded-xl bg-gray-900/60 border border-gray-800 space-y-2">
                  <span className="text-xs font-mono text-cyan-400 font-semibold">{f.metric} ({f.confidencePct}% Confidence)</span>
                  <h5 className="text-sm font-bold text-white">{f.title}</h5>
                  <p className="text-xs text-gray-300">{f.currentValue} → {f.projectedValue3Yr}</p>
                  <p className="text-[11px] text-amber-300/80">Risk factor: {f.riskFactors.join('; ')}</p>
                </div>
              ))}
            </div>
          </div>
        }
      />
    </div></CockpitShell>
  );
}
