"use client";

import CockpitShell from "../../../components/cockpit/CockpitShell";

import React from "react";
import Link from "next/link";
import { getUnifiedCockpitState } from "../../../lib/simulation/unifiedCockpitStore";
import SemanticZoom from "../../../components/cockpit/SemanticZoom";

export default function TodayHubPage() {
  const state = getUnifiedCockpitState();
  const { triad, nextBestAction, secondaryActions, recoveryIndicator, activeConstraints } = state;

  return (
    <CockpitShell activeHub="today"><div className="space-y-8">
      {/* Top Header Bar */}
      <header className="flex flex-col md:flex-row md:items-center md:justify-between pb-6 border-b border-gray-800/80 gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <span className="text-xs font-mono font-bold tracking-wider uppercase text-emerald-400 bg-emerald-950/60 border border-emerald-800/60 px-2.5 py-1 rounded">
              Core Hub 1
            </span>
            <span className="text-xs font-mono text-gray-400">Single Source of Truth (INV-OI110-P)</span>
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white mt-1">
            Today & Execution
          </h1>
          <p className="text-sm text-gray-400 mt-0.5">
            What deserves your cognitive attention, energy, and execution today?
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

      {/* Recovery & Circadian Window */}
      <section className="bg-gray-900/60 border border-gray-800/70 rounded-xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center space-x-4">
          <div className="w-12 h-12 rounded-xl bg-emerald-950/80 border border-emerald-800/60 flex items-center justify-center text-emerald-300 font-mono text-lg font-bold">
            {recoveryIndicator.sleepScore}
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-sm font-semibold text-white">Autonomic Recovery: {recoveryIndicator.hrvTrend}</span>
              <span className="text-xs font-mono text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-900">
                {recoveryIndicator.circadianPhase}
              </span>
            </div>
            <p className="text-xs text-gray-400 mt-0.5">
              Prime cognitive bandwidth window: <strong className="text-gray-200">{recoveryIndicator.primeWindow}</strong> • Energy capacity: {recoveryIndicator.energyCapacity}%
            </p>
          </div>
        </div>
        <div className="text-xs font-mono text-gray-400 bg-gray-950/80 border border-gray-800 px-3 py-2 rounded-lg">
          State Sync: Realtime Telemetry Connected
        </div>
      </section>

      {/* Constraint Alerts */}
      {activeConstraints.length > 0 && (
        <section className="space-y-2">
          {activeConstraints.map((c) => (
            <div
              key={c.id}
              className={`p-4 rounded-xl border flex items-start justify-between text-xs ${
                c.severity === 'WARNING'
                  ? 'bg-amber-950/20 border-amber-800/60 text-amber-200'
                  : 'bg-cyan-950/20 border-cyan-800/60 text-cyan-200'
              }`}
            >
              <div className="space-y-1">
                <span className="font-mono font-bold tracking-wide uppercase px-2 py-0.5 rounded text-[10px] bg-black/40 border border-current">
                  {c.type} CONSTRAINT
                </span>
                <p className="font-medium text-sm text-white">{c.message}</p>
                <p className="text-gray-400">{c.enforcementRule}</p>
              </div>
              <span className="font-mono text-xs px-2.5 py-1 rounded bg-black/30 border border-gray-800">
                {c.currentUtilization}
              </span>
            </div>
          ))}
        </section>
      )}

      {/* Semantic Zoom Experience for Today's Actions */}
      <SemanticZoom
        hubTitle="Execution & Capacity Governance"
        workbenchRoute="/workbench/allocator"
        workbenchName="168-Hour Allocator Workbench"
        level0Content={
          <div className="space-y-6">
            {/* Primary Action Card (INV-OI101-P) */}
            <div className="p-6 rounded-xl bg-gradient-to-br from-gray-900 to-gray-950 border-2 border-emerald-500/60 shadow-lg relative overflow-hidden">
              <div className="absolute top-0 right-0 bg-emerald-500 text-black text-[11px] font-mono font-bold px-3 py-1 rounded-bl-lg uppercase tracking-wider">
                Next Best Action
              </div>
              <div className="space-y-3 max-w-2xl">
                <div className="flex items-center space-x-2 text-xs font-mono text-emerald-400">
                  <span>{nextBestAction.domain}</span>
                  <span>•</span>
                  <span>{nextBestAction.durationMinutes} Minutes</span>
                  <span>•</span>
                  <span>Window: {nextBestAction.scheduledTimeWindow}</span>
                </div>
                <h3 className="text-xl font-bold text-white tracking-tight">
                  {nextBestAction.title}
                </h3>
                <p className="text-sm text-gray-300">
                  {nextBestAction.rationale}
                </p>
                <div className="pt-2 flex items-center space-x-4">
                  <button className="px-5 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-black font-semibold text-sm transition-all shadow-md">
                    Start Execution
                  </button>
                  <span className="text-xs font-mono text-gray-400">
                    +{nextBestAction.identityContribution} Identity Momentum Pts
                  </span>
                </div>
              </div>
            </div>

            {/* Secondary Actions (≤ 2 allowed) */}
            <div className="space-y-3">
              <h4 className="text-xs font-mono uppercase tracking-wider text-gray-400">
                Secondary Actions ({secondaryActions.length} / 2 Max)
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {secondaryActions.map((action) => (
                  <div
                    key={action.id}
                    className="p-4 rounded-xl bg-gray-900/60 border border-gray-800 hover:border-gray-700 transition-colors space-y-2"
                  >
                    <div className="flex items-center justify-between text-xs font-mono text-gray-400">
                      <span className="text-cyan-400">{action.domain}</span>
                      <span>{action.durationMinutes}m • {action.scheduledTimeWindow}</span>
                    </div>
                    <h5 className="text-sm font-semibold text-white">{action.title}</h5>
                    <p className="text-xs text-gray-400 line-clamp-2">{action.rationale}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        }
        level1Content={
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 rounded-xl bg-gray-900/80 border border-gray-800 space-y-2">
                <span className="text-xs font-mono text-gray-400 uppercase">Chronotype Alignment</span>
                <p className="text-sm text-white font-medium">Morning Peak Window Active</p>
                <p className="text-xs text-gray-400">High-cognitive architectural work matches prefrontal metabolic efficiency.</p>
              </div>
              <div className="p-4 rounded-xl bg-gray-900/80 border border-gray-800 space-y-2">
                <span className="text-xs font-mono text-gray-400 uppercase">Friction Minimization</span>
                <p className="text-sm text-emerald-400 font-medium">Zero Calendar Conflicts</p>
                <p className="text-xs text-gray-400">Time-window protected from meeting encroachments and multi-tasking.</p>
              </div>
              <div className="p-4 rounded-xl bg-gray-900/80 border border-gray-800 space-y-2">
                <span className="text-xs font-mono text-gray-400 uppercase">Energy Demand</span>
                <p className="text-sm text-purple-400 font-medium">{nextBestAction.energyRequired}</p>
                <p className="text-xs text-gray-400">Requires 45m uninterrupted flow block. Restorative cooldown scheduled at 17:00.</p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-gray-950 border border-gray-800/80 space-y-3">
              <h5 className="text-xs font-mono uppercase text-gray-300 font-semibold">Causal Lineage & Multi-Domain Impact</h5>
              <div className="space-y-2 text-xs text-gray-400">
                <div className="flex items-center justify-between border-b border-gray-900 pb-2">
                  <span>Target Identity Trajectory</span>
                  <span className="font-mono text-white">AI Strategy Leader (+24 pts gap closure)</span>
                </div>
                <div className="flex items-center justify-between border-b border-gray-900 pb-2">
                  <span>Weekly 168-Hour Envelope</span>
                  <span className="font-mono text-white">45m allocated from 26h restorative buffer</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Autonomic Resilience Impact</span>
                  <span className="font-mono text-white">Baseline HRV protected via evening Zone 2 run</span>
                </div>
              </div>
            </div>
          </div>
        }
      />
    </div></CockpitShell>
  );
}
