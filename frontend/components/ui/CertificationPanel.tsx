"use client";

import React from "react";
import SeverityBadge from "./SeverityBadge";
import { HorizonStatus } from "../../lib/ui/horizonTokens";

export interface CertificationGateItem {
  id: string;
  name: string;
  status: HorizonStatus | "PASS" | "FAIL" | "WARN";
  rule: string;
  evidence?: string;
}

export interface CertificationPanelProps {
  title?: string;
  subtitle?: string;
  overallStatus: HorizonStatus | string;
  auditHash?: string;
  timestamp?: string;
  gates: CertificationGateItem[];
  onReverify?: () => void;
  className?: string;
}

export default function CertificationPanel({
  title = "Autonomous Safety & Governance Certification",
  subtitle = "Fail-Close Boundary Verification across Active Release Gates",
  overallStatus,
  auditHash,
  timestamp,
  gates,
  onReverify,
  className = "",
}: CertificationPanelProps) {
  return (
    <section aria-label={title} className={`p-4 rounded-xl bg-[#121B2A] border border-[#24324A] space-y-4 ${className}`}>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold text-white tracking-tight flex items-center gap-2">
            <span>{title}</span>
            <SeverityBadge status={overallStatus} size="sm" />
          </h2>
          <p className="text-xs text-[#94A3B8] font-mono mt-0.5">
            {subtitle}
          </p>
        </div>
        {onReverify && (
          <button
            onClick={onReverify}
            className="px-3 py-1 rounded bg-[#182336] hover:bg-[#202e47] text-cyan-300 border border-cyan-500/30 text-xs font-mono transition-colors self-start sm:self-center focus:outline-none focus:ring-2 focus:ring-cyan-400"
          >
            Re-certify Gates
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
        {gates.map((g) => (
          <div
            key={g.id}
            className="p-2.5 rounded-lg bg-[#0B1220] border border-[#24324A] flex items-center justify-between text-xs font-mono"
          >
            <div className="space-y-0.5">
              <div className="font-semibold text-slate-200 flex items-center gap-1.5">
                <span className="text-cyan-400">{g.id}:</span>
                <span>{g.name}</span>
              </div>
              <div className="text-[11px] text-slate-400">{g.rule}</div>
              {g.evidence && (
                <div className="text-[10px] text-emerald-400/80">Evidence: {g.evidence}</div>
              )}
            </div>
            <SeverityBadge status={g.status} size="sm" />
          </div>
        ))}
      </div>

      {(auditHash || timestamp) && (
        <div className="pt-2 border-t border-[#24324A]/60 flex flex-wrap items-center justify-between text-[11px] font-mono text-[#94A3B8]">
          {auditHash && (
            <span title={auditHash}>
              Audit Hash: <span className="text-cyan-400">{auditHash.substring(0, 16)}...</span>
            </span>
          )}
          {timestamp && <span>Certified: {timestamp}</span>}
        </div>
      )}
    </section>
  );
}
