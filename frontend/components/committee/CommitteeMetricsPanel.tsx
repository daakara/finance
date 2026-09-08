"use client";

import { CommitteeIntelligenceDashboard } from "../../types/committee-intelligence";

export interface CommitteeMetricsPanelProps {
  dashboard: CommitteeIntelligenceDashboard;
}

export default function CommitteeMetricsPanel({ dashboard }: CommitteeMetricsPanelProps) {
  return (
    <section aria-label="Institutional Decision Intelligence Metrics" className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 font-mono">
      <div className="bg-[#121824] border border-[#202d44] p-3 rounded-xl">
        <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
          Average ODEI
        </span>
        <div className="flex items-baseline space-x-1.5 mt-1">
          <span className="text-2xl font-bold text-emerald-400">
            {dashboard.committeeODEI.toFixed(1)}
          </span>
          <span className="text-[10px] text-slate-500">/ 100</span>
        </div>
        <span className="text-[10px] text-emerald-500/80 block mt-1">
          &ge;80.0 Institutional Floor
        </span>
      </div>

      <div className="bg-[#121824] border border-[#202d44] p-3 rounded-xl">
        <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
          DIR Impact Ratio
        </span>
        <div className="flex items-baseline space-x-1 mt-1">
          <span className="text-2xl font-bold text-cyan-400">
            +{dashboard.committeeDIRatio.toFixed(1)}%
          </span>
        </div>
        <span className="text-[10px] text-cyan-500/80 block mt-1">
          &ge;20.0% Spread Certified
        </span>
      </div>

      <div className="bg-[#121824] border border-[#202d44] p-3 rounded-xl">
        <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
          Dissent Utilization
        </span>
        <div className="flex items-baseline space-x-1 mt-1">
          <span className="text-2xl font-bold text-purple-400">
            {dashboard.dissentUtilizationRate.toFixed(1)}%
          </span>
        </div>
        <span className="text-[10px] text-purple-400/80 block mt-1">
          &gt;25.0% Floor Target
        </span>
      </div>

      <div className="bg-[#121824] border border-[#202d44] p-3 rounded-xl">
        <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
          Active Committees
        </span>
        <div className="flex items-baseline space-x-1 mt-1">
          <span className="text-2xl font-bold text-slate-100">
            {dashboard.committeeCount}
          </span>
          <span className="text-[10px] text-slate-500">Bodies</span>
        </div>
        <span className="text-[10px] text-slate-400 block mt-1">
          100% Registered
        </span>
      </div>

      <div className="bg-[#121824] border border-[#202d44] p-3 rounded-xl">
        <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
          Governance Score
        </span>
        <div className="flex items-baseline space-x-1 mt-1">
          <span className="text-2xl font-bold text-slate-100">
            {dashboard.governanceCompliancePct.toFixed(1)}%
          </span>
        </div>
        <span className="text-[10px] text-emerald-400 block mt-1">
          INV-OI13/14 Audited
        </span>
      </div>

      <div className="bg-[#121824] border border-[#202d44] p-3 rounded-xl flex flex-col justify-between">
        <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
          CII Certification
        </span>
        <div className="flex items-center space-x-2 my-1">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-sm font-bold text-emerald-400">PASS</span>
        </div>
        <span className="text-[10px] text-slate-400 block">
          13/13 Gates Active
        </span>
      </div>
    </section>
  );
}
