"use client";

import ExecutiveIntelligenceNav from "../../components/committee/ExecutiveIntelligenceNav";
import CommitteeNetworkGraph from "../../components/committee/CommitteeNetworkGraph";
import InfluenceHeatmap from "../../components/committee/InfluenceHeatmap";
import {
  CANONICAL_NETWORK_NODES,
  CANONICAL_NETWORK_EDGES,
} from "../../lib/telemetry/committeeIntelligenceEngine";

export default function CommitteeNetworkPage() {
  return (
    <div className="min-h-screen bg-[#0c1017] text-slate-100 font-mono">
      <ExecutiveIntelligenceNav badgeText="INV-OI15 &amp; INV-OI16 VERIFIED" />

      <main className="max-w-[1750px] mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Page Title & Context Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-[#202d44] pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
              <h1 className="text-xl font-bold tracking-tight text-slate-100">
                Decision Network Topology &amp; Cross-Committee Influence
              </h1>
              <span className="px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 text-[10px] font-bold">
                CN-001 &ndash; CN-010
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-3xl">
              INV-OI15 &amp; INV-OI16 Invariants: Directed influence flows between governing bodies are strictly bounded in [0, 100], 100% connected, explainable, and continuously audited for circular influence cycles.
            </p>
          </div>

          <div className="flex items-center space-x-2 text-xs">
            <div className="px-2.5 py-1 rounded bg-[#111724] border border-[#202d44] text-slate-300">
              Nodes: <strong className="text-emerald-400">{CANONICAL_NETWORK_NODES.length}</strong>
            </div>
            <div className="px-2.5 py-1 rounded bg-[#111724] border border-[#202d44] text-slate-300">
              Directed Flows: <strong className="text-cyan-400">{CANONICAL_NETWORK_EDGES.length}</strong>
            </div>
          </div>
        </div>

        {/* Interactive SVG Network Graph */}
        <CommitteeNetworkGraph
          nodes={CANONICAL_NETWORK_NODES}
          edges={CANONICAL_NETWORK_EDGES}
        />

        {/* Pairwise Influence Matrix & Heatmap */}
        <InfluenceHeatmap />
      </main>
    </div>
  );
}
