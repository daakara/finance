"use client";

import CockpitShell from "../../../components/cockpit/CockpitShell";

import React, { useState } from "react";
import Link from "next/link";
import { useUnifiedCockpit } from "../../../lib/simulation/unifiedCockpitStore";
import SemanticZoom from "../../../components/cockpit/SemanticZoom";

export default function TodayHubPage() {
  const state = useUnifiedCockpit();
  const { triad, nextBestAction, secondaryActions, recoveryIndicator, activeConstraints } = state;
  const [zoomLevel, setZoomLevel] = useState<number>(1);

  return (
    <CockpitShell activeHub="today">
      <div className="space-y-6">
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
              Today &amp; Execution
            </h1>
            <p className="text-sm text-gray-400 mt-0.5">
              What deserves your cognitive attention, energy, and execution today?
            </p>
          </div>

          {/* Global Triad Banner & Quick Zoom Switcher */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center space-x-3 bg-gray-900/90 border border-gray-800 rounded-xl p-3 shadow-inner">
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

            {/* Quick Semantic Zoom Mode Selector */}
            <div className="flex items-center bg-gray-950 border border-gray-800 rounded-xl p-1 font-mono text-xs shadow-inner">
              <button
                type="button"
                onClick={() => setZoomLevel(0)}
                className={`px-3 py-1.5 rounded-lg transition-all ${
                  zoomLevel === 0
                    ? "bg-emerald-600 text-white font-bold shadow"
                    : "text-gray-400 hover:text-white"
                }`}
                title="Level 0: Glanceable Execution Strip"
              >
                L0 Strip
              </button>
              <button
                type="button"
                onClick={() => setZoomLevel(1)}
                className={`px-3 py-1.5 rounded-lg transition-all ${
                  zoomLevel === 1
                    ? "bg-cyan-600 text-white font-bold shadow"
                    : "text-gray-400 hover:text-white"
                }`}
                title="Level 1: Active Operating Horizon"
              >
                L1 Horizon
              </button>
              <button
                type="button"
                onClick={() => setZoomLevel(2)}
                className={`px-3 py-1.5 rounded-lg transition-all ${
                  zoomLevel === 2
                    ? "bg-purple-600 text-white font-bold shadow"
                    : "text-gray-400 hover:text-white"
                }`}
                title="Level 2: Full Causal Diagnostics & Allocator Handoff"
              >
                L2 Diagnostics
              </button>
            </div>
          </div>
        </header>

        {/* =================================================================== */}
        {/* LEVEL 0: Glanceable Execution Strip (Triad + Next Best Action)     */}
        {/* =================================================================== */}
        {zoomLevel === 0 && (
          <div className="space-y-4 animate-fadeIn">
            <div className="flex items-center justify-between text-xs font-mono text-gray-400 border-b border-gray-800/80 pb-2">
              <span className="text-emerald-400 font-semibold uppercase tracking-wider">
                ⚡ Level 0: Execution Strip (Glanceable Focus)
              </span>
              <span>Prime Window: <strong className="text-white">{recoveryIndicator?.primeWindow || "09:30 - 11:30 ET"}</strong></span>
            </div>

            {nextBestAction ? (
              <section className="p-6 rounded-2xl bg-gradient-to-br from-emerald-950/50 via-gray-900 to-gray-900 border border-emerald-700/80 space-y-4 shadow-2xl">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-mono font-bold text-emerald-400 uppercase tracking-wider bg-emerald-950/80 px-2.5 py-1 rounded border border-emerald-800/60">
                      Next Best Action
                    </span>
                    <span className="text-xs font-mono text-gray-400">Priority: {nextBestAction.priorityScore}/100</span>
                  </div>
                  <span className="text-xs font-mono text-purple-300 bg-purple-950/60 border border-purple-800/60 px-2 py-0.5 rounded">
                    +{nextBestAction.identityContribution} IAI Impact
                  </span>
                </div>

                <div>
                  <h2 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
                    {nextBestAction.title}
                  </h2>
                  <p className="text-sm text-gray-300 mt-1.5 max-w-2xl leading-relaxed">
                    {nextBestAction.rationale}
                  </p>
                </div>

                <div className="flex flex-wrap items-center justify-between gap-4 text-xs font-mono text-gray-400 pt-3 border-t border-gray-800">
                  <div className="flex items-center gap-4">
                    <span>Window: <strong className="text-white">{nextBestAction.scheduledTimeWindow || "Immediate"}</strong></span>
                    <span>Duration: <strong className="text-white">{nextBestAction.durationMinutes}m</strong></span>
                    <span>Energy: <strong className="text-emerald-400">{nextBestAction.energyRequired}</strong></span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setZoomLevel(1)}
                    className="text-cyan-400 hover:text-cyan-300 hover:underline text-xs flex items-center gap-1"
                  >
                    <span>Expand Active Horizon (L1)</span>
                    <span>→</span>
                  </button>
                </div>
              </section>
            ) : (
              <section className="p-8 rounded-2xl bg-gray-900/40 border border-gray-800/60 text-center space-y-2">
                <h3 className="text-sm font-mono font-bold text-gray-300">Zero Execution Actions Queued</h3>
                <p className="text-xs text-gray-400">No active friction detected. All horizons calibrated.</p>
              </section>
            )}
          </div>
        )}

        {/* =================================================================== */}
        {/* LEVEL 1 & 2: Active Operating Horizon (Recovery + NBA + Secondary)  */}
        {/* =================================================================== */}
        {zoomLevel >= 1 && (
          <div className="space-y-6 animate-fadeIn">
            {/* Recovery & Circadian Window */}
            {recoveryIndicator ? (
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
                      Prime Cognitive Execution Window: <strong className="text-white font-mono">{recoveryIndicator.primeWindow}</strong> • Energy Capacity: <strong className="text-emerald-400">{recoveryIndicator.energyCapacity}%</strong>
                    </p>
                  </div>
                </div>
                <Link
                  href="/workbench/signals"
                  className="text-xs font-mono text-cyan-400 hover:text-cyan-300 border border-cyan-800/60 bg-cyan-950/40 px-3 py-1.5 rounded-lg transition-colors shrink-0"
                >
                  Open Signals Workbench →
                </Link>
              </section>
            ) : (
              <section className="bg-gray-900/40 border border-gray-800/60 rounded-xl p-4 flex items-center justify-between text-xs text-gray-400 font-mono">
                <span>Autonomic Recovery: Telemetry uninitialized.</span>
                <Link href="/workbench/signals" className="text-cyan-400 hover:underline">Open Signals Workbench →</Link>
              </section>
            )}

            {/* Primary Next Best Action Card (One Thing Hero) */}
            {nextBestAction ? (
              <section className="p-6 rounded-2xl bg-gradient-to-br from-emerald-950/40 via-gray-900 to-gray-900 border border-emerald-800/60 space-y-4 shadow-xl">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-mono font-bold text-emerald-400 uppercase tracking-wider bg-emerald-950/80 px-2.5 py-1 rounded border border-emerald-800/60">
                      Next Best Action
                    </span>
                    <span className="text-xs font-mono text-gray-400">Score: {nextBestAction.priorityScore}/100</span>
                  </div>
                  <span className="text-xs font-mono text-purple-300 bg-purple-950/60 border border-purple-800/60 px-2 py-0.5 rounded">
                    +{nextBestAction.identityContribution} IAI Impact
                  </span>
                </div>

                <div>
                  <h2 className="text-2xl font-black text-white tracking-tight">
                    {nextBestAction.title}
                  </h2>
                  <p className="text-sm text-gray-300 mt-1 max-w-2xl">
                    {nextBestAction.rationale}
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-4 text-xs font-mono text-gray-400 pt-2 border-t border-gray-800">
                  <span>Window: <strong className="text-white">{nextBestAction.scheduledTimeWindow || "Immediate"}</strong></span>
                  <span>Duration: <strong className="text-white">{nextBestAction.durationMinutes}m</strong></span>
                  <span>Energy: <strong className="text-emerald-400">{nextBestAction.energyRequired}</strong></span>
                  <span className="ml-auto">
                    <Link href="/workbench/allocator" className="text-cyan-400 hover:text-cyan-300 underline">
                      Adjust in 168h Allocator →
                    </Link>
                  </span>
                </div>
              </section>
            ) : (
              <section className="p-6 rounded-2xl bg-gray-900/40 border border-gray-800/60 text-center space-y-2">
                <h3 className="text-sm font-mono font-bold text-gray-300">Zero Execution Actions Queued</h3>
                <p className="text-xs text-gray-400">No active high-priority friction detected. All horizons calibrated.</p>
              </section>
            )}

            {/* Secondary Actions Deck */}
            <section className="space-y-3">
              <h3 className="text-sm font-mono font-bold text-gray-400 uppercase tracking-wider">
                Secondary Action Queue (Max 2)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {secondaryActions && secondaryActions.length > 0 ? (
                  secondaryActions.map((action) => (
                    <div
                      key={action.id}
                      className="p-4 rounded-xl bg-gray-900/70 border border-gray-800 flex flex-col justify-between space-y-3 hover:border-gray-700 transition-colors"
                    >
                      <div>
                        <div className="flex items-center justify-between text-xs font-mono">
                          <span className="text-gray-400 font-semibold">{action.domain}</span>
                          <span className="text-purple-400">+{action.identityContribution} IAI</span>
                        </div>
                        <h4 className="text-base font-bold text-white mt-1">{action.title}</h4>
                        <p className="text-xs text-gray-400 mt-1">{action.rationale}</p>
                      </div>
                      <div className="flex items-center justify-between text-xs font-mono text-gray-500 pt-2 border-t border-gray-800/60">
                        <span>{action.durationMinutes}m • {action.scheduledTimeWindow || "Flexible"}</span>
                        <span className="text-emerald-400">{action.energyRequired}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="col-span-2 p-4 rounded-xl bg-gray-900/30 border border-gray-800 text-xs text-gray-400 font-mono">
                    No secondary actions queued.
                  </div>
                )}
              </div>
            </section>

            {/* Active Behavioral & Risk Constraints */}
            <section className="space-y-3">
              <h3 className="text-sm font-mono font-bold text-gray-400 uppercase tracking-wider">
                Active Behavioral Clamps &amp; Capacity Floors
              </h3>
              <div className="space-y-2">
                {activeConstraints && activeConstraints.length > 0 ? (
                  activeConstraints.map((c) => (
                    <div
                      key={c.id}
                      className={`p-3.5 rounded-xl border flex items-center justify-between text-xs font-mono ${
                        c.severity === 'WARNING'
                          ? 'bg-amber-950/30 border-amber-800/60 text-amber-200'
                          : 'bg-blue-950/30 border-blue-800/60 text-blue-200'
                      }`}
                    >
                      <div>
                        <strong className="text-white block">{c.message}</strong>
                        <span className="text-gray-400 text-[11px]">{c.enforcementRule}</span>
                      </div>
                      <span className="px-2 py-1 rounded bg-black/40 border border-white/10 text-[10px] shrink-0 font-bold">
                        {c.currentUtilization}
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="p-3.5 rounded-xl bg-gray-900/30 border border-gray-800 text-xs text-gray-400 font-mono">
                    Zero active behavioral clamps or capacity constraints.
                  </div>
                )}
              </div>
            </section>
          </div>
        )}

        {/* =================================================================== */}
        {/* LEVEL 2 ONLY: Full Causal Diagnostics & 168h Allocator Handoff     */}
        {/* =================================================================== */}
        {zoomLevel === 2 && (
          <div className="space-y-6 pt-4 border-t border-purple-900/60 animate-fadeIn">
            {/* Causal Lineage & Multi-Horizon Impact */}
            <section className="p-6 rounded-2xl bg-purple-950/20 border border-purple-800/60 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-xs font-mono font-bold uppercase tracking-wider text-purple-400 bg-purple-950/80 border border-purple-800/60 px-2.5 py-1 rounded">
                    Level 2 · Causal Lineage Diagnostics
                  </span>
                  <h3 className="text-lg font-bold text-white mt-2">
                    Multi-Horizon Causal Impact Trace
                  </h3>
                </div>
                <Link
                  href="/workbench/life-graph"
                  className="text-xs font-mono text-purple-300 hover:text-purple-200 border border-purple-700/60 bg-purple-900/40 px-3 py-1.5 rounded-lg transition-colors"
                >
                  View Life Graph Nodes →
                </Link>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs">
                <div className="p-4 rounded-xl bg-slate-950/70 border border-purple-900/40 space-y-2">
                  <span className="text-[10px] text-gray-400 uppercase block">LHI Causal Impact</span>
                  <div className="text-xl font-bold text-emerald-400">+2.4 pts</div>
                  <p className="text-[11px] text-gray-300">
                    Executing Next Best Action unlocks primary flow state and mitigates cognitive drag.
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-950/70 border border-purple-900/40 space-y-2">
                  <span className="text-[10px] text-gray-400 uppercase block">HHI Capital Protection</span>
                  <div className="text-xl font-bold text-cyan-400">Zero Drawdown</div>
                  <p className="text-[11px] text-gray-300">
                    Sizing governed under strict capital floor preservation rules.
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-950/70 border border-purple-900/40 space-y-2">
                  <span className="text-[10px] text-gray-400 uppercase block">IAI Identity Contribution</span>
                  <div className="text-xl font-bold text-purple-400">+{nextBestAction?.identityContribution || 4} Impact</div>
                  <p className="text-[11px] text-gray-300">
                    Direct reinforcement of master identity vector through deliberate execution.
                  </p>
                </div>
              </div>
            </section>

            {/* 168-Hour Time & Capital Allocator Deep-Link Handoff */}
            <section className="p-6 rounded-2xl bg-gradient-to-r from-[#0b1220] via-purple-950/30 to-[#0b1220] border border-purple-700/60 space-y-4 shadow-xl">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <span className="text-xs font-mono font-bold uppercase tracking-wider text-cyan-400 bg-cyan-950/80 border border-cyan-800/60 px-2.5 py-1 rounded">
                    168-Hour Allocator Bridge
                  </span>
                  <h3 className="text-xl font-bold text-white mt-2">
                    Allocate Time, Capital &amp; Energy in Specialist Workbench
                  </h3>
                  <p className="text-xs text-gray-300 mt-1 max-w-xl">
                    Calibrate your 168-hour weekly budget across sleep, deep execution, administrative overhead, and margin buffer to eliminate cognitive overcommitment.
                  </p>
                </div>
                <Link
                  href="/workbench/allocator"
                  className="inline-flex items-center justify-center space-x-2 px-5 py-3 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-mono font-bold text-xs tracking-tight transition-all shadow-lg shadow-purple-900/50 shrink-0"
                >
                  <span>Launch 168h Allocator Workbench</span>
                  <span>→</span>
                </Link>
              </div>
            </section>
          </div>
        )}

        {/* Semantic Zoom Footer for Progressive Disclosure */}
        <footer className="pt-6 border-t border-gray-800/80">
          <SemanticZoom
            activeLevel={zoomLevel}
            onLevelChange={setZoomLevel}
            workbenchRoute="/workbench/allocator"
            workbenchName="168-Hour Allocator Workbench"
            levels={[
              {
                level: 0,
                label: "0: Execution Strip",
                detail: "High-density single metric strip with zero commentary. Just Next Best Action + prime window.",
              },
              {
                level: 1,
                label: "1: Active Horizon",
                detail: "Standard operating view: Single NBA card, 2 secondary actions, autonomic recovery, and active capacity clamps.",
              },
              {
                level: 2,
                label: "2: Full Diagnostics",
                detail: "Expanded deep-dive: Causal life graph links, telemetry signal traces, and 168-hour allocation ledger.",
              },
            ]}
          />
        </footer>
      </div>
    </CockpitShell>
  );
}
