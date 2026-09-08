"use client";

import { Suspense } from "react";
import ExecutiveIntelligenceNav from "../../components/committee/ExecutiveIntelligenceNav";
import DecisionExplorer from "../../components/committee/DecisionExplorer";
import { CANONICAL_COMMITTEE_DECISIONS } from "../../lib/telemetry/committeeIntelligenceEngine";

export default function DecisionExplorerPage() {
  return (
    <div className="min-h-screen bg-[#0c1017] text-slate-100 font-mono">
      <ExecutiveIntelligenceNav badgeText="INV-OI13 VERIFIED" />

      <main className="max-w-[1750px] mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Page Title & Context Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-[#202d44] pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
              <h1 className="text-xl font-bold tracking-tight text-slate-100">
                Decision Explorer &amp; Traceability Timeline
              </h1>
              <span className="px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 text-[10px] font-bold">
                DE-001 &ndash; DE-010
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-3xl">
              Inspect end-to-end decision lineage from proposal and verified evidence vault, through voting quorum and preserved dissent, to measured outcome and exact 100% attribution.
            </p>
          </div>

          <div className="flex items-center space-x-2 text-xs">
            <div className="px-2.5 py-1 rounded bg-[#111724] border border-[#202d44] text-slate-300">
              Traceable Decisions: <strong className="text-cyan-400">{CANONICAL_COMMITTEE_DECISIONS.length}</strong>
            </div>
            <div className="px-2.5 py-1 rounded bg-emerald-950/50 border border-emerald-500/40 text-emerald-400 text-xs font-semibold">
              100% Attribution Sum
            </div>
          </div>
        </div>

        {/* Decision Explorer Component wrapped in Suspense for useSearchParams */}
        <Suspense fallback={<div className="p-8 text-center text-slate-400 text-xs">Loading Decision Explorer...</div>}>
          <DecisionExplorer decisions={CANONICAL_COMMITTEE_DECISIONS} />
        </Suspense>
      </main>
    </div>
  );
}
