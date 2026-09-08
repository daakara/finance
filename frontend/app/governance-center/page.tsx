"use client";

import ExecutiveIntelligenceNav from "../../components/committee/ExecutiveIntelligenceNav";
import CertificationStatusBanner from "../../components/committee/CertificationStatusBanner";
import GovernanceAlertFeed from "../../components/committee/GovernanceAlertFeed";

export default function GovernanceCenterPage() {
  return (
    <div className="min-h-screen bg-[#0c1017] text-slate-100 font-mono">
      <ExecutiveIntelligenceNav badgeText="13/13 GATES CERTIFIED" />

      <main className="max-w-[1750px] mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Page Title & Context Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-[#202d44] pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
              <h1 className="text-xl font-bold tracking-tight text-slate-100">
                Institutional Governance Center &amp; Threat Defense
              </h1>
              <span className="px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-500/40 text-emerald-300 text-[10px] font-bold">
                GC-001 &ndash; GC-010
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-3xl">
              Centralized mission control for all 13 Adaptive Intelligence certification gates, deterministic replay verification, and active monitoring of Byzantine attack vectors.
            </p>
          </div>

          <div className="flex items-center space-x-2 text-xs">
            <div className="px-2.5 py-1 rounded bg-emerald-950/50 border border-emerald-500/40 text-emerald-400 font-semibold">
              Replay: 100/100 (0 Drift)
            </div>
            <div className="px-2.5 py-1 rounded bg-[#111724] border border-[#202d44] text-slate-300">
              FDS: <strong className="text-cyan-400">86.8 / 100</strong>
            </div>
          </div>
        </div>

        {/* 13 CII-Gates Status Banner */}
        <CertificationStatusBanner />

        {/* Byzantine Threat Defense & Log Feed */}
        <GovernanceAlertFeed />
      </main>
    </div>
  );
}
