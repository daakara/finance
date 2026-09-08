"use client";

import Link from "next/link";
import { CommitteeHealth } from "../../types/committee-intelligence";

export interface CommitteeScorecardProps {
  committee: CommitteeHealth & { committeeId: string; name: string; odei: number };
  rank?: number;
}

export default function CommitteeScorecard({ committee, rank }: CommitteeScorecardProps) {
  const isCertified = committee.odei >= 80.0 && committee.cdqi >= 80.0;
  const status = isCertified ? "PASS" : "WARNING";

  return (
    <article
      data-testid={`committee-scorecard-${committee.committeeId}`}
      className="bg-[#121927] border border-[#223048] rounded-xl p-5 hover:border-cyan-500/40 transition-all duration-200 flex flex-col justify-between shadow-lg shadow-black/30 group"
    >
      <div>
        {/* Header */}
        <div className="flex items-start justify-between gap-2 mb-3">
          <div className="flex items-center space-x-2">
            {rank !== undefined && (
              <span className="w-5 h-5 rounded-full bg-[#1b263b] border border-[#2b3d5b] text-[10px] font-mono text-slate-300 flex items-center justify-center font-bold">
                {rank}
              </span>
            )}
            <div>
              <h2 className="text-base font-semibold text-slate-100 group-hover:text-cyan-300 transition-colors tracking-tight">
                {committee.name}
              </h2>
              <span className="text-[10px] font-mono text-slate-400">
                {committee.committeeId}
              </span>
            </div>
          </div>

          {/* Certification Badge */}
          <div
            className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold tracking-wider uppercase border flex items-center space-x-1 ${
              status === "PASS"
                ? "bg-emerald-950/60 border-emerald-500/40 text-emerald-400"
                : "bg-amber-950/60 border-amber-500/40 text-amber-400"
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                status === "PASS" ? "bg-emerald-400" : "bg-amber-400"
              }`}
            />
            <span>{status}</span>
          </div>
        </div>

        {/* Primary Metrics Grid */}
        <div className="grid grid-cols-2 gap-2 my-4">
          <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1b263b]">
            <span className="text-[10px] font-mono uppercase text-slate-400 block">
              CDQI Index
            </span>
            <div className="flex items-baseline space-x-1.5 mt-0.5">
              <span className="text-xl font-bold font-mono text-slate-100">
                {committee.cdqi.toFixed(1)}
              </span>
              <span className="text-[10px] font-mono text-slate-400">/ 100</span>
            </div>
            <div className="w-full bg-[#1b263b] h-1 rounded-full mt-2 overflow-hidden">
              <div
                className="bg-cyan-400 h-full rounded-full transition-all"
                style={{ width: `${Math.min(100, committee.cdqi)}%` }}
              />
            </div>
          </div>

          <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1b263b]">
            <span className="text-[10px] font-mono uppercase text-slate-400 block">
              Committee ODEI
            </span>
            <div className="flex items-baseline space-x-1.5 mt-0.5">
              <span className="text-xl font-bold font-mono text-emerald-400">
                {committee.odei.toFixed(1)}
              </span>
              <span className="text-[10px] font-mono text-slate-400">/ 100</span>
            </div>
            <div className="w-full bg-[#1b263b] h-1 rounded-full mt-2 overflow-hidden">
              <div
                className="bg-emerald-400 h-full rounded-full transition-all"
                style={{ width: `${Math.min(100, committee.odei)}%` }}
              />
            </div>
          </div>
        </div>

        {/* Sub-Telemetry Key-Values */}
        <div className="space-y-1.5 font-mono text-xs text-slate-300 border-t border-[#1d293d] pt-3">
          <div className="flex justify-between items-center text-[11px]">
            <span className="text-slate-400">Impact Ratio (DIR):</span>
            <span className="font-semibold text-emerald-400">
              +{committee.committeeDIRatio.toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between items-center text-[11px]">
            <span className="text-slate-400">Dissent Coverage:</span>
            <span className="font-semibold text-slate-200">
              {committee.dissentCoveragePct.toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between items-center text-[11px]">
            <span className="text-slate-400">Transparency (OI13):</span>
            <span className="font-semibold text-slate-200">
              {committee.transparencyCoveragePct.toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between items-center text-[11px]">
            <span className="text-slate-400">Learning Velocity:</span>
            <span className="font-semibold text-cyan-400">
              +{committee.learningVelocityPct.toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between items-center text-[11px]">
            <span className="text-slate-400">Governance Compliance:</span>
            <span className="font-semibold text-slate-200">
              {committee.governanceCompliancePct.toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* Action Footers */}
      <div className="mt-4 pt-3 border-t border-[#1d293d] flex items-center justify-between">
        <span className="text-[10px] font-mono text-slate-400">
          Quorum & Chair Verified
        </span>
        <Link
          href={`/decision-explorer?committeeId=${committee.committeeId}`}
          className="text-xs font-mono text-cyan-400 hover:text-cyan-300 flex items-center space-x-1 group-hover:translate-x-0.5 transition-transform"
        >
          <span>View Decisions</span>
          <span>&rarr;</span>
        </Link>
      </div>
    </article>
  );
}
