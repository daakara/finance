"use client";

import React from "react";
import Link from "next/link";

export interface RelatedArtifactLink {
  id: string;
  type: "DECISION" | "RISK" | "LEARNING" | "SIMULATION" | "RECOMMENDATION" | "RUNBOOK" | "COMMITTEE" | "AUDIT" | "OUTCOME" | "INCIDENT" | "SCENARIO";
  title: string;
  href: string;
  badge?: string;
  summary?: string;
}

export interface RelatedArtifactsPanelProps {
  title?: string;
  artifacts: RelatedArtifactLink[];
  className?: string;
}

const TYPE_COLORS: Record<string, string> = {
  DECISION: 'text-cyan-300 border-cyan-500/30 bg-cyan-950/20',
  RISK: 'text-amber-300 border-amber-500/30 bg-amber-950/20',
  LEARNING: 'text-purple-300 border-purple-500/30 bg-purple-950/20',
  SIMULATION: 'text-blue-300 border-blue-500/30 bg-blue-950/20',
  RECOMMENDATION: 'text-emerald-300 border-emerald-500/30 bg-emerald-950/20',
  RUNBOOK: 'text-rose-300 border-rose-500/30 bg-rose-950/20',
  COMMITTEE: 'text-indigo-300 border-indigo-500/30 bg-indigo-950/20',
  AUDIT: 'text-slate-300 border-slate-500/30 bg-slate-900/30',
  OUTCOME: 'text-emerald-400 border-emerald-500/30 bg-emerald-950/20',
  INCIDENT: 'text-orange-400 border-orange-500/30 bg-orange-950/20',
  SCENARIO: 'text-blue-400 border-blue-500/30 bg-blue-950/20',
};

export default function RelatedArtifactsPanel({
  title = "Related Institutional Artifacts & Lineage Links",
  artifacts,
  className = "",
}: RelatedArtifactsPanelProps) {
  if (!artifacts || artifacts.length === 0) return null;

  return (
    <div className={`p-4 rounded-xl bg-[#121B2A] border border-[#24324A] space-y-3 ${className}`}>
      <h3 className="text-xs font-mono uppercase tracking-widest text-[#94A3B8] font-semibold">
        {title} ({artifacts.length})
      </h3>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
        {artifacts.map((art) => {
          const typeStyle = TYPE_COLORS[art.type] || TYPE_COLORS.AUDIT;
          return (
            <Link
              key={art.id}
              href={art.href}
              className="p-3 rounded-lg bg-[#0B1220] border border-[#24324A] hover:border-cyan-500/40 hover:bg-[#162236] transition-all flex flex-col justify-between group focus:outline-none focus:ring-2 focus:ring-cyan-400"
            >
              <div className="flex items-center justify-between gap-1 mb-1.5">
                <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded border ${typeStyle}`}>
                  {art.type}
                </span>
                <span className="text-[10px] font-mono text-cyan-400 group-hover:translate-x-0.5 transition-transform">
                  &rarr;
                </span>
              </div>
              <div className="text-xs font-semibold text-slate-200 group-hover:text-cyan-300 truncate">
                {art.title}
              </div>
              <div className="flex items-center justify-between mt-2 pt-1 border-t border-[#24324A]/40 text-[10px] font-mono text-slate-400">
                <span>{art.id}</span>
                {art.badge && <span className="text-emerald-400">{art.badge}</span>}
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
