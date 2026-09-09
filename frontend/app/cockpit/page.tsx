"use client";

import React from "react";
import Link from "next/link";
import { getUnifiedCockpitState } from "../../lib/simulation/unifiedCockpitStore";

export default function GovernorPortalPage() {
  const state = getUnifiedCockpitState();
  const { triad, runway, recoveryIndicator, householdHealth, workbenches } = state;

  return (
    <main className="min-h-screen bg-[#070b12] text-gray-100 p-4 md:p-8 space-y-8 max-w-7xl mx-auto">
      {/* Return to Terminal Bar */}
      <div className="flex items-center justify-between pb-4 border-b border-gray-800">
        <Link
          href="/radar"
          className="inline-flex items-center space-x-2 text-xs font-mono text-cyan-400 hover:text-cyan-300 font-semibold"
        >
          <span>← Return to ARX Terminal</span>
        </Link>
        <span className="text-xs font-mono text-emerald-400 bg-emerald-950/80 border border-emerald-800 px-2.5 py-1 rounded-full flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          Behavioral Governor Engine: Active
        </span>
      </div>

      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-mono uppercase tracking-wider text-purple-400 font-bold bg-purple-950/60 border border-purple-800/60 px-2.5 py-1 rounded">
            Underlying Infrastructure
          </span>
          <h1 className="text-3xl font-extrabold text-white tracking-tight mt-2">
            Behavioral Governor & Personal Intelligence
          </h1>
          <p className="text-sm text-gray-400 mt-1 max-w-2xl">
            This internal engine room monitors sleep debt, drawdown streaks, liquid runway floors, and cognitive capacity to dynamically clamp trading risk and prevent emotional mistakes.
          </p>
        </div>

        {/* Global Triad Summary */}
        <div className="flex items-center space-x-3 bg-gray-900/90 border border-gray-800 rounded-xl p-3 shadow-inner shrink-0">
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

      {/* 4 Core Intelligence Hubs Grid */}
      <section className="space-y-3">
        <h2 className="text-xs font-mono uppercase tracking-wider text-gray-400">
          Four Core Intelligence Hubs
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Link
            href="/today"
            className="p-5 rounded-xl bg-gray-900/70 border border-gray-800 hover:border-emerald-500/60 transition-all space-y-3 group"
          >
            <div className="flex items-center justify-between">
              <span className="text-2xl">⚡</span>
              <span className="text-xs font-mono text-emerald-400 font-bold group-hover:underline">Open Hub →</span>
            </div>
            <div>
              <h3 className="text-base font-bold text-white group-hover:text-emerald-300">Today & Capacity</h3>
              <p className="text-xs text-gray-400 mt-1">Next best action, recovery indicator, and active capacity constraints.</p>
            </div>
            <div className="pt-2 border-t border-gray-800/80 text-[11px] font-mono text-gray-400">
              Recovery: <strong className="text-white">{recoveryIndicator.hrvTrend}</strong> (Sleep {recoveryIndicator.sleepScore})
            </div>
          </Link>

          <Link
            href="/future"
            className="p-5 rounded-xl bg-gray-900/70 border border-gray-800 hover:border-cyan-500/60 transition-all space-y-3 group"
          >
            <div className="flex items-center justify-between">
              <span className="text-2xl">🔮</span>
              <span className="text-xs font-mono text-cyan-400 font-bold group-hover:underline">Open Hub →</span>
            </div>
            <div>
              <h3 className="text-base font-bold text-white group-hover:text-cyan-300">Future & Scenarios</h3>
              <p className="text-xs text-gray-400 mt-1">Runway shield, 3-year trajectories, and downside capital preservation floors.</p>
            </div>
            <div className="pt-2 border-t border-gray-800/80 text-[11px] font-mono text-gray-400">
              Runway: <strong className="text-white">{runway.monthsUnencumbered} Mo</strong> ({runway.runwayShieldStatus})
            </div>
          </Link>

          <Link
            href="/progress"
            className="p-5 rounded-xl bg-gray-900/70 border border-gray-800 hover:border-purple-500/60 transition-all space-y-3 group"
          >
            <div className="flex items-center justify-between">
              <span className="text-2xl">🎯</span>
              <span className="text-xs font-mono text-purple-400 font-bold group-hover:underline">Open Hub →</span>
            </div>
            <div>
              <h3 className="text-base font-bold text-white group-hover:text-purple-300">Progress & Calibration</h3>
              <p className="text-xs text-gray-400 mt-1">Identity twin evolution, behavioral drift alerts, and Brier judgment scores.</p>
            </div>
            <div className="pt-2 border-t border-gray-800/80 text-[11px] font-mono text-gray-400">
              Brier Score: <strong className="text-white">{state.calibrationScore.brierScore}</strong> (Calibrated)
            </div>
          </Link>

          <Link
            href="/household"
            className="p-5 rounded-xl bg-gray-900/70 border border-gray-800 hover:border-blue-500/60 transition-all space-y-3 group"
          >
            <div className="flex items-center justify-between">
              <span className="text-2xl">🏡</span>
              <span className="text-xs font-mono text-blue-400 font-bold group-hover:underline">Open Hub →</span>
            </div>
            <div>
              <h3 className="text-base font-bold text-white group-hover:text-blue-300">Household & Relational</h3>
              <p className="text-xs text-gray-400 mt-1">Household Health Index (HHI 89), shared resources, and relational conflict radar.</p>
            </div>
            <div className="pt-2 border-t border-gray-800/80 text-[11px] font-mono text-gray-400">
              Alignment: <strong className="text-white">{householdHealth.partnerAlignment}%</strong> (Zero Collisions)
            </div>
          </Link>
        </div>
      </section>

      {/* Specialist Workbenches */}
      <section className="space-y-3">
        <h2 className="text-xs font-mono uppercase tracking-wider text-gray-400">
          Specialist Workbenches
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {workbenches.map((wb) => (
            <Link
              key={wb.id}
              href={wb.route}
              className="p-4 rounded-xl bg-gray-900/50 border border-gray-800 hover:border-gray-700 transition-colors space-y-2 block"
            >
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-purple-400 font-bold">{wb.category}</span>
                <span className="text-gray-500">{wb.activeMetricsCount} Metrics</span>
              </div>
              <h4 className="text-sm font-semibold text-white">{wb.name}</h4>
              <p className="text-xs text-gray-400">{wb.description}</p>
            </Link>
          ))}
        </div>
      </section>
    </main>
  );
}
