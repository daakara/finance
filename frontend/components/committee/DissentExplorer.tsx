"use client";

import { useState } from "react";
import Link from "next/link";
import { CommitteeDissent } from "../../types/committee-intelligence";
import { CANONICAL_DISSENTS } from "../../lib/telemetry/committeeIntelligenceEngine";

export interface DissentExplorerProps {
  dissents?: CommitteeDissent[];
}

export default function DissentExplorer({ dissents = CANONICAL_DISSENTS }: DissentExplorerProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [severityFilter, setSeverityFilter] = useState<string>("ALL");

  const filteredDissents = dissents.filter((d) => {
    const matchesSearch =
      d.dissentId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      d.decisionId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      d.authorId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      d.alternativeRecommendation.toLowerCase().includes(searchTerm.toLowerCase()) ||
      d.riskAssessment.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesSeverity = severityFilter === "ALL" || d.severity === severityFilter;

    return matchesSearch && matchesSeverity;
  });

  return (
    <div className="space-y-4 font-mono">
      {/* Header & Filter Bar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 bg-[#111724] border border-[#202d44] p-3 rounded-xl">
        <div className="flex items-center space-x-2">
          <span className="w-2 h-2 rounded-full bg-purple-400" />
          <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wide">
            Preserved Dissents Registry (INV-OI14)
          </h2>
          <span className="text-xs text-slate-400">({filteredDissents.length} records)</span>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="text"
            placeholder="Search dissents, authors, or decisions..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="px-3 py-1.5 bg-[#0c1017] border border-[#243044] rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 w-full sm:w-64"
          />

          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="px-2.5 py-1.5 bg-[#0c1017] border border-[#243044] rounded-lg text-xs text-slate-300 focus:outline-none focus:border-cyan-500"
          >
            <option value="ALL">All Severities</option>
            <option value="MATERIAL">MATERIAL</option>
            <option value="HIGH">HIGH</option>
          </select>
        </div>
      </div>

      {/* Dissents Grid / List */}
      <div className="grid grid-cols-1 gap-3">
        {filteredDissents.map((dissent) => (
          <div
            key={dissent.dissentId}
            className="bg-[#111724] border border-[#202d44] hover:border-purple-500/40 transition-colors p-4 rounded-xl space-y-3"
          >
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#202d44] pb-2.5">
              <div className="flex items-center space-x-2">
                <span className="px-2 py-0.5 rounded bg-purple-950/60 border border-purple-500/40 text-purple-400 text-xs font-bold">
                  {dissent.dissentId}
                </span>
                <span className="text-slate-400 text-xs">on decision</span>
                <Link
                  href={`/decision-explorer?decisionId=${dissent.decisionId}`}
                  className="px-2 py-0.5 rounded bg-cyan-950/50 border border-cyan-500/30 text-cyan-300 text-xs font-semibold hover:bg-cyan-900/50 transition-colors flex items-center space-x-1"
                >
                  <span>{dissent.decisionId}</span>
                  <span className="text-[10px]">&rarr;</span>
                </Link>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    dissent.severity === "HIGH"
                      ? "bg-rose-950/60 border border-rose-500/40 text-rose-400"
                      : "bg-amber-950/60 border border-amber-500/40 text-amber-400"
                  }`}
                >
                  {dissent.severity} SEVERITY
                </span>
              </div>

              <div className="flex items-center space-x-2 text-xs text-slate-400">
                <span className="text-slate-300">Author: <strong className="text-slate-100">{dissent.authorId}</strong></span>
                <span>&bull;</span>
                <span>{new Date(dissent.timestampUtc).toLocaleDateString()}</span>
                {dissent.acceptedForReview && (
                  <span className="px-2 py-0.5 rounded bg-emerald-950/50 border border-emerald-500/30 text-emerald-400 text-[10px] font-semibold">
                    &check; Reviewed
                  </span>
                )}
              </div>
            </div>

            {/* Alternative Recommendation */}
            <div className="bg-[#0c1017] border border-[#202d44] p-3 rounded-lg">
              <div className="text-[10px] text-purple-400 font-bold uppercase tracking-wider mb-1 flex items-center space-x-1">
                <span>&bull;</span>
                <span>Alternative Recommendation</span>
              </div>
              <p className="text-xs text-slate-200 leading-relaxed">
                {dissent.alternativeRecommendation}
              </p>
            </div>

            {/* Risk Assessment */}
            <div className="bg-[#0c1017] border border-amber-900/30 p-3 rounded-lg">
              <div className="text-[10px] text-amber-400 font-bold uppercase tracking-wider mb-1 flex items-center space-x-1">
                <span>&#9888;</span>
                <span>Downside Risk Assessment</span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                {dissent.riskAssessment}
              </p>
            </div>

            {/* Evidence & Compliance Tags */}
            <div className="flex flex-wrap items-center justify-between gap-2 pt-1 text-xs">
              <div className="flex items-center space-x-1.5">
                <span className="text-[10px] text-slate-500">Supporting Evidence:</span>
                {dissent.evidenceIds.map((evd) => (
                  <span
                    key={evd}
                    className="px-1.5 py-0.5 rounded bg-[#162032] border border-[#243044] text-[10px] text-slate-300"
                  >
                    {evd}
                  </span>
                ))}
              </div>

              <div className="flex items-center space-x-2">
                <span className="px-2 py-0.5 rounded bg-emerald-950/40 border border-emerald-500/30 text-emerald-400 text-[10px] font-medium">
                  INV-OI14 Certified
                </span>
                <span className="text-[10px] text-slate-500">0% Omission Risk</span>
              </div>
            </div>
          </div>
        ))}

        {filteredDissents.length === 0 && (
          <div className="bg-[#111724] border border-[#202d44] p-8 rounded-xl text-center text-slate-400 text-xs">
            No dissents found matching &quot;{searchTerm}&quot;.
          </div>
        )}
      </div>
    </div>
  );
}
