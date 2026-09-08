"use client";

import ExecutiveIntelligenceNav from "../../components/committee/ExecutiveIntelligenceNav";
import DissentImpactCard from "../../components/committee/DissentImpactCard";
import DissentExplorer from "../../components/committee/DissentExplorer";
import {
  CANONICAL_DISSENTS,
  computeDissentUtilizationRate,
} from "../../lib/telemetry/committeeIntelligenceEngine";

export default function DissentExplorerPage() {
  const utilizationRate = computeDissentUtilizationRate(CANONICAL_DISSENTS);

  return (
    <div className="min-h-screen bg-[#0c1017] text-slate-100 font-mono">
      <ExecutiveIntelligenceNav badgeText="INV-OI14 PRESERVED" />

      <main className="max-w-[1750px] mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Page Title & Context Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-[#202d44] pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-purple-400" />
              <h1 className="text-xl font-bold tracking-tight text-slate-100">
                Dissent Explorer &amp; Counter-Thesis Registry
              </h1>
              <span className="px-2 py-0.5 rounded bg-purple-950/60 border border-purple-500/40 text-purple-300 text-[10px] font-bold">
                DI-001 &ndash; DI-010
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-3xl">
              INV-OI14 Invariant: Zero lost dissents. Minority objections, alternative strategies, and downside risk assessments are permanently recorded and incorporated into decision reviews.
            </p>
          </div>

          <div className="flex items-center space-x-2 text-xs">
            <div className="px-2.5 py-1 rounded bg-[#111724] border border-[#202d44] text-slate-300">
              Active Dissents: <strong className="text-purple-400">{CANONICAL_DISSENTS.length}</strong>
            </div>
            <div className="px-2.5 py-1 rounded bg-emerald-950/50 border border-emerald-500/40 text-emerald-400 font-semibold">
              100% Coverage Certified
            </div>
          </div>
        </div>

        {/* Dissent Impact Summary Metrics */}
        <DissentImpactCard
          totalDissents={CANONICAL_DISSENTS.length}
          utilizationRate={utilizationRate}
          coveragePct={100.0}
        />

        {/* Full Dissent Explorer Component */}
        <DissentExplorer dissents={CANONICAL_DISSENTS} />
      </main>
    </div>
  );
}
