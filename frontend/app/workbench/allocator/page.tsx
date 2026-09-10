"use client";

import React from "react";
import Link from "next/link";
import { useUnifiedCockpit, refreshUnifiedCockpit } from "../../../lib/simulation/unifiedCockpitStore";

export default function AllocatorWorkbench() {
  const state = useUnifiedCockpit();
  const { status, errorMessage, triad, activeConstraints, nextBestAction, secondaryActions } = state;

  return (
    <main className="min-h-screen bg-[#070b12] text-gray-100 p-4 md:p-8 space-y-8 max-w-7xl mx-auto">
      {status === 'ERROR' && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-xs font-mono text-rose-300 flex items-center justify-between">
          <span>Failed to load allocation constraints telemetry: {errorMessage || "Network error"}</span>
          <button
            onClick={() => refreshUnifiedCockpit()}
            className="px-3 py-1 bg-rose-800 hover:bg-rose-700 text-white rounded font-bold transition-all"
          >
            Retry Connection
          </button>
        </div>
      )}
      {status === 'LOADING' && (
        <div className="p-3 rounded-lg bg-emerald-950/30 border border-emerald-800/40 text-xs font-mono text-emerald-400 animate-pulse">
          Synchronizing 168-hour allocation envelope and constraints...
        </div>
      )}

      <header className="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-gray-800 gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <Link href="/today" className="text-xs font-mono text-gray-400 hover:text-white">
              ← Return to /today
            </Link>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
              Specialist Workbench
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white mt-1">
            168-Hour Allocator Workbench
          </h1>
          <p className="text-xs text-gray-400 mt-0.5">
            Time, energy, and capital envelope budgeting with calendar collision resolution.
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

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-6 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase font-mono">168-Hour Weekly Allocation</h3>
          <div className="space-y-3 text-xs">
            <div className="flex justify-between border-b border-gray-800 pb-2">
              <span className="text-gray-400">Sleep & Biological Rest</span>
              <span className="font-mono text-white">56.0 Hours (33.3%)</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 pb-2">
              <span className="text-gray-400">Deep Work & Strategic Career</span>
              <span className="font-mono text-cyan-400">42.0 Hours (25.0%)</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 pb-2">
              <span className="text-gray-400">Household & Relational Presence</span>
              <span className="font-mono text-blue-400">28.0 Hours (16.7%)</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 pb-2">
              <span className="text-gray-400">Fitness & Active Recovery</span>
              <span className="font-mono text-emerald-400">16.0 Hours (9.5%)</span>
            </div>
            <div className="flex justify-between pt-1">
              <span className="text-gray-300 font-semibold">Uncommitted Buffer</span>
              <span className="font-mono text-purple-400 font-bold">26.0 Hours (15.5%)</span>
            </div>
          </div>
        </div>

        <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-6 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase font-mono">Active Constraints & Sizing Rules</h3>
          <div className="space-y-3">
            {activeConstraints && activeConstraints.length > 0 ? (
              activeConstraints.map((c) => (
                <div key={c.id} className="p-3 rounded-lg bg-gray-950 border border-gray-800 text-xs space-y-1">
                  <span className="text-amber-400 font-bold">{c.type}: {c.currentUtilization}</span>
                  <p className="text-gray-300">{c.message}</p>
                  <p className="text-gray-500">{c.enforcementRule}</p>
                </div>
              ))
            ) : (
              <div className="p-3 rounded-lg bg-gray-950 border border-gray-800 text-xs text-gray-500 font-mono">
                Zero active constraints registered.
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
