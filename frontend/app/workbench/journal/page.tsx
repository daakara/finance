"use client";

import React from "react";
import Link from "next/link";
import { getUnifiedCockpitState } from "../../../lib/simulation/unifiedCockpitStore";

export default function JournalWorkbench() {
  const state = getUnifiedCockpitState();
  const { triad, calibrationScore, identityDrift } = state;

  return (
    <main className="min-h-screen bg-[#070b12] text-gray-100 p-4 md:p-8 space-y-8 max-w-7xl mx-auto">
      <header className="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-gray-800 gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <Link href="/progress" className="text-xs font-mono text-gray-400 hover:text-white">
              ← Return to /progress
            </Link>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">
              Specialist Workbench
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white mt-1">
            Decision Journal & Calibration Workbench
          </h1>
          <p className="text-xs text-gray-400 mt-0.5">
            Probabilistic judgment scoring, Brier score calibration, and execution post-mortems.
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

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="p-6 rounded-xl bg-gray-900/60 border border-gray-800 space-y-2">
          <span className="text-xs font-mono uppercase text-gray-400">Brier Score</span>
          <p className="text-3xl font-mono font-bold text-emerald-400">{calibrationScore.brierScore}</p>
          <p className="text-xs text-gray-400">Target $\le 0.25$ indicates well-calibrated probabilistic judgment.</p>
        </div>

        <div className="p-6 rounded-xl bg-gray-900/60 border border-gray-800 space-y-2">
          <span className="text-xs font-mono uppercase text-gray-400">Accuracy & Overconfidence</span>
          <p className="text-3xl font-mono font-bold text-cyan-400">{calibrationScore.accuracyPct}%</p>
          <p className="text-xs text-gray-400">Bias status: {calibrationScore.overconfidenceBias} ({calibrationScore.trend}).</p>
        </div>

        <div className="p-6 rounded-xl bg-gray-900/60 border border-gray-800 space-y-2">
          <span className="text-xs font-mono uppercase text-gray-400">Active Drift Alert</span>
          <p className="text-lg font-mono font-bold text-amber-400">{identityDrift.domain}</p>
          <p className="text-xs text-gray-400">{identityDrift.inactiveDays}d inactive • {identityDrift.remedyAction}.</p>
        </div>
      </div>
    </main>
  );
}
