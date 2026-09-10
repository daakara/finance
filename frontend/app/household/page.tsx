"use client";

import CockpitShell from "../../components/cockpit/CockpitShell";

import React from "react";
import Link from "next/link";
import { useUnifiedCockpit } from "../../lib/simulation/unifiedCockpitStore";
import SemanticZoom from "../../components/cockpit/SemanticZoom";

export default function HouseholdHubPage() {
  const state = useUnifiedCockpit();
  const { triad, householdHealth, sharedResources } = state;

  return (
    <CockpitShell activeHub="household"><div className="space-y-8">
      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-center md:justify-between pb-6 border-b border-gray-800/80 gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <span className="text-xs font-mono font-bold tracking-wider uppercase text-blue-400 bg-blue-950/60 border border-blue-800/60 px-2.5 py-1 rounded">
              Core Hub 4
            </span>
            <span className="text-xs font-mono text-gray-400">Horizon 8 Relational Intelligence</span>
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white mt-1">
            Household &amp; Relational
          </h1>
          <p className="text-sm text-gray-400 mt-0.5">
            Household health, shared resource capacity, and proactive conflict mitigation.
          </p>
        </div>

        {/* Global Triad Banner */}
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
      </header>

      {/* Household Status Header */}
      {householdHealth ? (
        <section className="p-6 rounded-xl bg-gradient-to-r from-blue-950/40 to-gray-900 border border-blue-800/50 flex flex-col md:flex-row md:items-center justify-between gap-6 shadow-xl">
          <div className="space-y-1">
            <span className="text-xs font-mono uppercase font-bold text-blue-400 bg-blue-950 px-2 py-0.5 rounded border border-blue-800">
              Household Health Index (HHI)
            </span>
            <h2 className="text-3xl font-extrabold text-white tracking-tight">
              {householdHealth.hhi} / 100 Cohesion
            </h2>
            <p className="text-xs text-gray-400">
              Partner Alignment: <strong className="text-white">{householdHealth.partnerAlignment}%</strong> • Shared Resource Load: <strong className="text-emerald-400">{Math.round(householdHealth.sharedResourceLoad * 100)}%</strong>
            </p>
          </div>

          <div className="p-4 rounded-xl bg-gray-950/70 border border-gray-800 max-w-sm space-y-1">
            <span className="text-[10px] font-mono uppercase text-gray-400 block">Key Weekly Sync Agenda</span>
            <p className="text-xs font-mono text-cyan-300 font-semibold">{householdHealth.keySyncItem}</p>
          </div>
        </section>
      ) : (
        <section className="p-4 rounded-xl bg-gray-900/40 border border-gray-800 text-xs font-mono text-gray-400 flex items-center justify-between">
          <span>Household Cohesion Unconfigured: Record shared commitments to track relational balance.</span>
          <span className="text-blue-400 font-bold">UNRECORDED</span>
        </section>
      )}

      {/* Semantic Zoom Container for Household */}
      <SemanticZoom
        hubTitle="Household Resource Allocation & Coordination"
        workbenchRoute="/workbench/life-graph"
        workbenchName="Life Graph & Causal Ripple Workbench"
        level0Content={
          <div className="space-y-4">
            <h3 className="text-sm font-mono font-bold text-gray-400 uppercase tracking-wider">Shared Capacity Nodes</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {sharedResources && sharedResources.length > 0 ? (
                sharedResources.map((res) => (
                  <div key={res.name} className="p-4 rounded-xl bg-gray-900/60 border border-gray-800 space-y-2">
                    <span className="text-xs font-mono font-bold text-white">{res.name}</span>
                    <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${res.capacityAllocatedPct > 80 ? 'bg-amber-400' : 'bg-blue-500'}`}
                        style={{ width: `${res.capacityAllocatedPct}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-[10px] font-mono text-gray-400">
                      <span>{res.capacityAllocatedPct}% Allocated</span>
                      <span className="text-emerald-400">{res.conflictStatus}</span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="col-span-3 p-4 rounded-xl bg-gray-900/30 border border-gray-800 text-xs font-mono text-gray-500 text-center">
                  Zero shared resources registered.
                </div>
              )}
            </div>
          </div>
        }
        level1Content={
          <div className="p-5 rounded-xl bg-gray-950 border border-gray-800 text-xs font-mono space-y-2">
            <h4 className="text-sm font-bold text-white">Relational Coordination &amp; Friction Prevention</h4>
            <p className="text-gray-300">Active Stakeholders: {householdHealth?.stakeholderCount ?? 0}</p>
            <p className="text-gray-300">Conflict Risk Probability: <strong className="text-emerald-400">{householdHealth?.conflictRisk ?? "LOW"}</strong></p>
          </div>
        }
      />
    </div></CockpitShell>
  );
}
