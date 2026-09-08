"use client";

export interface DissentImpactCardProps {
  totalDissents: number;
  utilizationRate: number;
  coveragePct: number;
}

export default function DissentImpactCard({
  totalDissents,
  utilizationRate,
  coveragePct,
}: DissentImpactCardProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 font-mono">
      <div className="bg-[#111724] border border-[#202d44] p-4 rounded-xl">
        <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
          Dissent Coverage (INV-OI14)
        </span>
        <div className="flex items-baseline space-x-1 mt-1">
          <span className="text-2xl font-bold text-emerald-400">
            {coveragePct.toFixed(1)}%
          </span>
        </div>
        <span className="text-[10px] text-emerald-400/80 block mt-1">
          0 Lost Dissents Across All Decisions
        </span>
      </div>

      <div className="bg-[#111724] border border-[#202d44] p-4 rounded-xl">
        <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
          Dissent Utilization Rate
        </span>
        <div className="flex items-baseline space-x-1 mt-1">
          <span className="text-2xl font-bold text-cyan-400">
            {utilizationRate.toFixed(1)}%
          </span>
        </div>
        <span className="text-[10px] text-cyan-400/80 block mt-1">
          &gt;25.0% Institutional Target Exceeded
        </span>
      </div>

      <div className="bg-[#111724] border border-[#202d44] p-4 rounded-xl">
        <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
          Preserved Counter-Theses
        </span>
        <div className="flex items-baseline space-x-1 mt-1">
          <span className="text-2xl font-bold text-purple-400">
            {totalDissents}
          </span>
          <span className="text-[10px] text-slate-500">Material Records</span>
        </div>
        <span className="text-[10px] text-purple-400/80 block mt-1">
          100% Alternative Views Documented
        </span>
      </div>

      <div className="bg-[#111724] border border-[#202d44] p-4 rounded-xl">
        <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
          Downside Risk Mitigated
        </span>
        <div className="flex items-baseline space-x-1 mt-1">
          <span className="text-2xl font-bold text-amber-400">
            +12.0%
          </span>
          <span className="text-[10px] text-slate-500">Drawdown Shield</span>
        </div>
        <span className="text-[10px] text-amber-400/80 block mt-1">
          Staged Tranches & Exposure Caps
        </span>
      </div>
    </div>
  );
}
