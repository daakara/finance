"use client";

import React from "react";
import Link from "next/link";
import { getUnifiedCockpitState } from "../../lib/simulation/unifiedCockpitStore";
import SemanticZoom from "../../components/cockpit/SemanticZoom";

export default function HouseholdHubPage() {
  const state = getUnifiedCockpitState();
  const { triad, householdHealth, sharedResources } = state;

  return (
    <main className="min-h-screen bg-[#070b12] text-gray-100 p-4 md:p-8 space-y-8 max-w-7xl mx-auto">
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
            Household & Relational
          </h1>
          <p className="text-sm text-gray-400 mt-0.5">
            Household health, shared resource capacity, and proactive conflict mitigation.
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

      {/* Household Status Header */}
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

        <div className="flex items-center space-x-4 bg-gray-950/80 border border-gray-800 p-4 rounded-xl">
          <div>
            <span className="text-[10px] font-mono uppercase text-gray-400 block">Conflict Risk</span>
            <span className="text-base font-mono font-bold text-emerald-400 uppercase">{householdHealth.conflictRisk}</span>
          </div>
          <div className="h-8 w-px bg-gray-800" />
          <div>
            <span className="text-[10px] font-mono uppercase text-gray-400 block">Stakeholders</span>
            <span className="text-base font-mono font-bold text-white">{householdHealth.stakeholderCount} Relational Twins</span>
          </div>
        </div>
      </section>

      {/* Semantic Zoom for Household */}
      <SemanticZoom
        hubTitle="Shared Resource Allocation & Conflict Radar"
        workbenchRoute="/workbench/life-graph"
        workbenchName="Life Graph Workbench"
        level0Content={
          <div className="space-y-6">
            {/* Key Alignment Item */}
            <div className="p-4 rounded-xl bg-gray-900/80 border border-blue-800/60 flex items-center justify-between">
              <div>
                <span className="text-xs font-mono text-blue-400 uppercase block">Next Alignment Sync</span>
                <p className="text-sm font-semibold text-white">{householdHealth.keySyncItem}</p>
              </div>
              <span className="text-xs font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-800/60 px-2.5 py-1 rounded">
                Conflict Free
              </span>
            </div>

            {/* Shared Resources Grid */}
            <div className="space-y-3">
              <h4 className="text-xs font-mono uppercase tracking-wider text-gray-400">
                Shared Resource Utilization
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {sharedResources.map((res) => (
                  <div key={res.name} className="p-4 rounded-xl bg-gray-900/60 border border-gray-800 space-y-2">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-gray-400">{res.conflictStatus}</span>
                      <span className="text-cyan-400 font-bold">{res.capacityAllocatedPct}%</span>
                    </div>
                    <h5 className="text-sm font-bold text-white">{res.name}</h5>
                    <p className="text-xs text-gray-400">Users: {res.primaryUsers.join(', ')}</p>
                    <div className="h-1.5 w-full bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-cyan-500 rounded-full"
                        style={{ width: `${res.capacityAllocatedPct}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        }
        level1Content={
          <div className="space-y-6">
            <div className="p-5 rounded-xl bg-gray-950 border border-gray-800 space-y-4">
              <h4 className="text-sm font-bold text-white">Relational Impact Visibility (INV-OI91-P)</h4>
              <p className="text-xs text-gray-300">
                Every career and financial action carries evaluated relational consequences. Zero double-booking detected across shared vehicles or evening childcare coverage.
              </p>
            </div>
          </div>
        }
      />
    </main>
  );
}
