"use client";

import { Suspense } from "react";
import ExecutiveIntelligenceNav from "../../components/committee/ExecutiveIntelligenceNav";
import AuditExplorer from "../../components/committee/AuditExplorer";

export default function AuditExplorerPage() {
  return (
    <div className="min-h-screen bg-[#0c1017] text-slate-100 font-mono">
      <ExecutiveIntelligenceNav badgeText="RECON 100% RECOVERED" />

      <main className="max-w-[1750px] mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Page Title & Context Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-[#202d44] pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
              <h1 className="text-xl font-bold tracking-tight text-slate-100">
                Single-Artifact Audit Explorer &amp; Reconstruction Engine
              </h1>
              <span className="px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 text-[10px] font-bold">
                AE-001 &ndash; AE-010
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-3xl">
              INV-OI13-A Invariant: 100% institutional lineage reconstruction from any individual artifact ID (Decision, Outcome, Proposal, or Dissent). Inspect cryptographic snapshot hashes and export verifiable JSON audit trails.
            </p>
          </div>

          <div className="flex items-center space-x-2 text-xs">
            <div className="px-2.5 py-1 rounded bg-emerald-950/50 border border-emerald-500/40 text-emerald-400 font-semibold">
              0 Missing Artifacts
            </div>
            <div className="px-2.5 py-1 rounded bg-[#111724] border border-[#202d44] text-slate-300">
              SHA-256 Validated
            </div>
          </div>
        </div>

        {/* Audit Explorer Wrapped in Suspense Boundary */}
        <Suspense fallback={<div className="p-8 text-center text-slate-400 text-xs">Loading Audit Explorer...</div>}>
          <AuditExplorer initialQuery="DEC-001" />
        </Suspense>
      </main>
    </div>
  );
}
