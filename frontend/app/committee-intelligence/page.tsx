"use client";

import ExecutiveIntelligenceNav from "../../components/committee/ExecutiveIntelligenceNav";
import CommitteeMetricsPanel from "../../components/committee/CommitteeMetricsPanel";
import CommitteeHealthGrid from "../../components/committee/CommitteeHealthGrid";
import HistoricalTrendsPanel from "../../components/committee/HistoricalTrendsPanel";
import {
  getCommitteeIntelligenceDashboard,
  CANONICAL_COMMITTEES,
} from "../../lib/telemetry/committeeIntelligenceEngine";

export default function CommitteeIntelligencePage() {
  const dashboard = getCommitteeIntelligenceDashboard();

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
                Committee Health &amp; Scorecards
              </h1>
              <span className="px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 text-[10px] font-bold">
                CI-001 &ndash; CI-010
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-3xl">
              Systematic tracking of collective decision quality, Organizational Decision Effectiveness Index (ODEI &ge; 80.0), Decision-to-Intent impact ratios, and dissent retention.
            </p>
          </div>

          <div className="flex items-center space-x-2 text-xs">
            <div className="px-2.5 py-1 rounded bg-[#111724] border border-[#202d44] text-slate-300">
              Active Bodies: <strong className="text-emerald-400">{CANONICAL_COMMITTEES.length}</strong>
            </div>
            <div className="px-2.5 py-1 rounded bg-[#111724] border border-[#202d44] text-slate-300">
              Avg ODEI: <strong className="text-cyan-400">{dashboard.committeeODEI.toFixed(1)}</strong>
            </div>
          </div>
        </div>

        {/* Aggregate Institutional Metrics Panel */}
        <CommitteeMetricsPanel dashboard={dashboard} />

        {/* Historical Trends Engine (HT-01 to HT-06) */}
        <HistoricalTrendsPanel initialCommitteeId="COM-001" initialMetric="ODEI" initialTimeframe="90D" />

        {/* Detailed Committee Health Grid with Sorting & Filtering */}
        <div className="space-y-3">
          <div className="text-xs font-bold text-slate-300 uppercase tracking-wider">
            Registered Committee Bodies &amp; Invariant Status
          </div>
          <CommitteeHealthGrid committees={CANONICAL_COMMITTEES} />
        </div>
      </main>
    </div>
  );
}
