"use client";

import React, { useState } from "react";
import { EvidenceDomainItem, DomainStatus, DomainApplicability } from "../../lib/evidenceDomains";

interface EvidenceDomainStatusCardProps {
  symbol: string;
  domains: EvidenceDomainItem[];
  defaultExpanded?: boolean;
}

export default function EvidenceDomainStatusCard({
  symbol,
  domains,
  defaultExpanded = false,
}: EvidenceDomainStatusCardProps) {
  const [isExpanded, setIsExpanded] = useState<boolean>(defaultExpanded);

  if (!domains || domains.length === 0) {
    return null;
  }

  const verifiedCount = domains.filter((d) => d.status === "VERIFIED").length;
  const optionalCount = domains.filter((d) => d.applicability === "OPTIONAL" || d.applicability === "INAPPLICABLE").length;
  const blockingIssues = domains.filter((d) => d.impact === "BLOCKING" && d.status !== "VERIFIED");

  const getStatusBadge = (status: DomainStatus, label: string) => {
    switch (status) {
      case "VERIFIED":
        return "bg-emerald-950/80 text-emerald-300 border-emerald-700/80";
      case "CAUTION":
        return "bg-amber-950/80 text-amber-300 border-amber-700/80";
      case "NO_OBSERVATIONS":
        return "bg-slate-900/90 text-slate-300 border-slate-700/80";
      case "INAPPLICABLE":
        return "bg-slate-900/60 text-slate-500 border-slate-800";
      case "AWAITING_TRIGGER":
        return "bg-amber-950/80 text-amber-300 border-amber-600/80";
      case "BLOCKED":
        return "bg-rose-950/80 text-rose-300 border-rose-700/80";
      case "INSUFFICIENT_HISTORY":
      case "UNAVAILABLE":
      default:
        return "bg-slate-900/90 text-slate-400 border-slate-700";
    }
  };

  const getApplicabilityBadge = (app: DomainApplicability) => {
    switch (app) {
      case "REQUIRED":
        return "bg-cyan-950/60 text-cyan-300 border-cyan-800/60";
      case "OPTIONAL":
        return "bg-slate-900/80 text-slate-400 border-slate-700/60";
      case "INAPPLICABLE":
        return "bg-slate-900/40 text-slate-500 border-slate-800/50";
    }
  };

  return (
    <div
      data-testid="evidence-domain-status-card"
      className="bg-[var(--surface-1)] border border-[var(--border-subtle)] rounded-xl overflow-hidden shadow-sm font-sans transition-colors"
    >
      {/* 🛡️ Header Summary Row (Always Visible) */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setIsExpanded(!isExpanded);
          }
        }}
        aria-expanded={isExpanded}
        aria-label="Toggle Evidence Domain Status"
        className="w-full px-3 sm:px-4 py-2.5 flex flex-wrap items-center justify-between gap-2 bg-[var(--surface-2)]/60 hover:bg-[var(--surface-2)] cursor-pointer transition-colors select-none"
      >
        <div className="flex items-center gap-2 sm:gap-2.5">
          <span className="text-sm shrink-0" aria-hidden="true">🛡️</span>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-white tracking-wide">
                Evidence Domain Status
              </span>
              <span className="text-[10px] font-mono text-slate-400 hidden sm:inline">
                ({domains.length} Domains Audited)
              </span>
            </div>
            <p className="text-[11px] text-[var(--text-2)] hidden md:block">
              Authoritative evidence verification across technical, fundamental, regulatory, and execution gates.
            </p>
          </div>
        </div>

        {/* Quick Summary Badges & Toggle */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 text-[11px] font-mono">
            <span
              data-testid="evidence-summary-verified"
              className="px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-700/80 font-semibold"
            >
              {verifiedCount} Verified
            </span>
            {blockingIssues.length > 0 ? (
              <span
                data-testid="evidence-summary-blocking"
                className="px-2 py-0.5 rounded bg-amber-950/80 text-amber-300 border border-amber-700/80 font-semibold"
              >
                {blockingIssues.length} Missing Core
              </span>
            ) : (
              <span
                data-testid="evidence-summary-optional"
                className="px-2 py-0.5 rounded bg-slate-900/80 text-slate-400 border border-slate-700 font-normal hidden xs:inline"
              >
                {optionalCount} Optional/Exempt
              </span>
            )}
          </div>

          <button
            type="button"
            className="text-xs font-mono font-semibold text-cyan-400 hover:text-cyan-300 px-2 py-1 rounded bg-[var(--surface-1)] border border-cyan-800/40 hover:border-cyan-600/60 transition-colors flex items-center gap-1"
          >
            <span>{isExpanded ? "Hide" : "Inspect"}</span>
            <span className="text-[10px]">{isExpanded ? "▲" : "▼"}</span>
          </button>
        </div>
      </div>

      {/* 📋 Expanded 9-Domain Details Table */}
      {isExpanded && (
        <div className="p-3 sm:p-4 border-t border-[var(--border-subtle)] space-y-3 bg-[var(--surface-1)] animate-fadeIn">
          <div className="overflow-x-auto -mx-3 sm:mx-0">
            <table className="w-full text-left text-xs font-sans min-w-[640px]">
              <thead>
                <tr className="border-b border-slate-800 text-[10px] font-mono uppercase tracking-wider text-slate-400">
                  <th className="py-1.5 px-2">Domain & Authority</th>
                  <th className="py-1.5 px-2">Category</th>
                  <th className="py-1.5 px-2">Role</th>
                  <th className="py-1.5 px-2">Status</th>
                  <th className="py-1.5 px-2">Verdict Impact</th>
                  <th className="py-1.5 px-2">Details & Provenance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {domains.map((d) => (
                  <tr key={d.id} data-testid={`domain-row-${d.id}`} className="hover:bg-slate-900/40 transition-colors">
                    {/* Domain & Authority */}
                    <td className="py-2 px-2 align-top">
                      <div className="font-semibold text-slate-200">{d.name}</div>
                      <div className="text-[10px] font-mono text-slate-400">{d.authority}</div>
                    </td>

                    {/* Category */}
                    <td className="py-2 px-2 align-top text-[11px] text-slate-300">
                      {d.category}
                    </td>

                    {/* Role / Applicability */}
                    <td className="py-2 px-2 align-top">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold border ${getApplicabilityBadge(d.applicability)}`}>
                        {d.applicability}
                      </span>
                    </td>

                    {/* Status */}
                    <td className="py-2 px-2 align-top">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border inline-block whitespace-nowrap ${getStatusBadge(d.status, d.statusLabel)}`}>
                        {d.statusLabel}
                      </span>
                    </td>

                    {/* Verdict Impact */}
                    <td className="py-2 px-2 align-top">
                      <span className={`text-[11px] font-mono font-semibold ${
                        d.impact === "BLOCKING"
                          ? (d.status === "VERIFIED" ? "text-slate-400" : "text-amber-400 font-bold")
                          : "text-slate-500"
                      }`}>
                        {d.impact === "BLOCKING" ? "Blocking" : "Advisory (Non-blocking)"}
                      </span>
                    </td>

                    {/* Details & Provenance */}
                    <td className="py-2 px-2 align-top space-y-0.5 max-w-[320px]">
                      <p className="text-[11px] text-slate-300 leading-snug">
                        {d.explanation}
                      </p>
                      <div className="text-[10px] font-mono text-slate-500">
                        Source: {d.provenance}
                        {d.observationTime ? ` · ${d.observationTime}` : ""}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="pt-2 border-t border-slate-800/80 flex flex-wrap items-center justify-between text-[11px] text-slate-400 font-mono">
            <span>Asset Scope: <strong className="text-slate-200">{symbol}</strong></span>
            <span>Invariant: Missing advisory enrichment never demotes actionable setups.</span>
          </div>
        </div>
      )}
    </div>
  );
}
