"use client";

import React from "react";
import Link from "next/link";
import { useUnifiedCockpit, refreshUnifiedCockpit } from "../../../lib/simulation/unifiedCockpitStore";

export default function SignalsWorkbench() {
  const state = useUnifiedCockpit();
  const { status, errorMessage, triad, signalQuality, recoveryIndicator } = state;

  return (
    <main className="min-h-screen bg-[#070b12] text-gray-100 p-4 md:p-8 space-y-8 max-w-7xl mx-auto">
      {status === 'ERROR' && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-xs font-mono text-rose-300 flex items-center justify-between">
          <span>Failed to load personal signals telemetry: {errorMessage || "Network error"}</span>
          <button
            onClick={() => refreshUnifiedCockpit()}
            className="px-3 py-1 bg-rose-800 hover:bg-rose-700 text-white rounded font-bold transition-all"
          >
            Retry Connection
          </button>
        </div>
      )}
      {status === 'LOADING' && (
        <div className="p-3 rounded-lg bg-cyan-950/30 border border-cyan-800/40 text-xs font-mono text-cyan-400 animate-pulse">
          Synchronizing telemetry signals stream...
        </div>
      )}

      <header className="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-gray-800 gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <Link href="/today" className="text-xs font-mono text-gray-400 hover:text-white">
              ← Return to /today
            </Link>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
              Specialist Workbench
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white mt-1">
            Personal Signals & Telemetry Workbench
          </h1>
          <p className="text-xs text-gray-400 mt-0.5">
            High-frequency telemetry streams, biometric recovery, and conviction indicators.
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
        <div className="p-6 rounded-xl bg-gray-900/60 border border-gray-800 space-y-3">
          <span className="text-xs font-mono uppercase text-gray-400">Telemetry Freshness</span>
          <p className="text-2xl font-mono font-bold text-emerald-400">{signalQuality?.freshness ?? "UNAVAILABLE"}</p>
          <p className="text-xs text-gray-400">{signalQuality?.activeSignalsCount ?? 0} active telemetry streams ({signalQuality?.confidence ?? 0}% confidence).</p>
        </div>

        <div className="p-6 rounded-xl bg-gray-900/60 border border-gray-800 space-y-3">
          <span className="text-xs font-mono uppercase text-gray-400">Autonomic Recovery</span>
          <p className="text-2xl font-mono font-bold text-cyan-400">{recoveryIndicator?.hrvTrend ?? "UNAVAILABLE"}</p>
          <p className="text-xs text-gray-400">Sleep score: {recoveryIndicator?.sleepScore ?? "--"} • Capacity: {recoveryIndicator?.energyCapacity ?? "--"}%.</p>
        </div>

        <div className="p-6 rounded-xl bg-gray-900/60 border border-gray-800 space-y-3">
          <span className="text-xs font-mono uppercase text-gray-400">Circadian Optimization</span>
          <p className="text-2xl font-mono font-bold text-purple-400">{recoveryIndicator?.circadianPhase ?? "--"}</p>
          <p className="text-xs text-gray-400">Peak cognitive window: {recoveryIndicator?.primeWindow ?? "--"}.</p>
        </div>
      </div>
    </main>
  );
}
