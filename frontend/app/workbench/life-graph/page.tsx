"use client";

import React from "react";
import Link from "next/link";
import { useUnifiedCockpit, refreshUnifiedCockpit } from "../../../lib/simulation/unifiedCockpitStore";

export default function LifeGraphWorkbench() {
  const state = useUnifiedCockpit();
  const { status, errorMessage, triad, householdHealth, sharedResources } = state;

  return (
    <main className="min-h-screen bg-[#070b12] text-gray-100 p-4 md:p-8 space-y-8 max-w-7xl mx-auto">
      {status === 'ERROR' && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-xs font-mono text-rose-300 flex items-center justify-between">
          <span>Failed to load household life-graph telemetry: {errorMessage || "Network error"}</span>
          <button
            onClick={() => refreshUnifiedCockpit()}
            className="px-3 py-1 bg-rose-800 hover:bg-rose-700 text-white rounded font-bold transition-all"
          >
            Retry Connection
          </button>
        </div>
      )}
      {status === 'LOADING' && (
        <div className="p-3 rounded-lg bg-purple-950/30 border border-purple-800/40 text-xs font-mono text-purple-300 animate-pulse">
          Synchronizing causal life-graph nodes and shared resources...
        </div>
      )}

      <header className="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-gray-800 gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <Link href="/household" className="text-xs font-mono text-gray-400 hover:text-white">
              ← Return to /household
            </Link>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800">
              Specialist Workbench
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white mt-1">
            Life Graph & Causal Ripple Workbench
          </h1>
          <p className="text-xs text-gray-400 mt-0.5">
            Node topology, multi-domain causal chains, and systemic friction mapping.
          </p>
        </div>

        <div className="flex items-center space-x-3 bg-gray-900 border border-gray-800 p-3 rounded-xl">
          <div className="text-center px-3 border-r border-gray-800">
            <span className="text-[10px] uppercase font-mono text-gray-400 block">LHI</span>
            <span className="text-xl font-mono font-bold text-emerald-400">{triad?.lhi ?? "--"}</span>
          </div>
          <div className="text-center px-3 border-r border-gray-800">
            <span className="text-[10px] uppercase font-mono text-gray-400 block">HHI</span>
            <span className="text-xl font-mono font-bold text-blue-400">{triad?.hhi ?? "--"}</span>
          </div>
          <div className="text-center px-3">
            <span className="text-[10px] uppercase font-mono text-gray-400 block">IAI</span>
            <span className="text-xl font-mono font-bold text-purple-400">{triad?.iai ?? "--"}</span>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2 bg-gray-900/60 border border-gray-800 rounded-xl p-6 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase font-mono">Causal Node Topology</h3>
          <div className="h-64 rounded-lg bg-gray-950 border border-gray-800/80 flex items-center justify-center p-4">
            <div className="text-center space-y-2">
              <span className="text-2xl">🕸️</span>
              <p className="text-sm font-mono text-gray-300">48 Interconnected Causal Nodes Active</p>
              <p className="text-xs text-gray-500">Domains: Career (14) • Capital (12) • Health (10) • Relational (12)</p>
            </div>
          </div>
          <div className="text-xs text-gray-400">
            Cross-domain feedback loop: Execution of AI Systems Architecture directly relieves Career stagnation, while Zone 2 runs preserve autonomic recovery for trading windows.
          </div>
        </div>

        <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-6 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase font-mono">Relational Nodes</h3>
          <div className="space-y-3">
            <div className="p-3 rounded-lg bg-gray-950 border border-gray-800 text-xs space-y-1">
              <span className="text-blue-400 font-bold">Partner Alignment: {householdHealth?.partnerAlignment ?? "--"}%</span>
              <p className="text-gray-300">Next sync: {householdHealth?.keySyncItem ?? "Not Scheduled"}</p>
            </div>
            {sharedResources && sharedResources.length > 0 ? (
              sharedResources.map((r) => (
                <div key={r.name} className="p-3 rounded-lg bg-gray-950 border border-gray-800 text-xs space-y-1">
                  <span className="text-gray-300 font-semibold">{r.name}</span>
                  <p className="text-gray-400">{r.capacityAllocatedPct}% allocated • {r.conflictStatus}</p>
                </div>
              ))
            ) : (
              <div className="p-3 rounded-lg bg-gray-950 border border-gray-800 text-xs text-gray-500 font-mono">
                Zero shared resources registered.
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
