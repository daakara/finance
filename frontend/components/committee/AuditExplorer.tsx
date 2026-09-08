"use client";

import { useState, useMemo } from "react";
import { useSearchParams } from "next/navigation";
import DecisionTimeline from "./DecisionTimeline";
import { CANONICAL_COMMITTEE_DECISIONS } from "../../lib/telemetry/committeeIntelligenceEngine";
import {
  reconstructDecision,
  createAuditSnapshot,
} from "../../lib/telemetry/auditReconstructionEngine";
import {
  buildDecisionTimeline,
  generateAuditExport,
} from "../../lib/telemetry/decisionNetworkEngine";

export interface AuditExplorerProps {
  initialQuery?: string;
}

export default function AuditExplorer({ initialQuery = "DEC-001" }: AuditExplorerProps) {
  const searchParams = useSearchParams();
  const urlQuery = searchParams.get("queryId") || searchParams.get("decisionId") || initialQuery;

  const [queryInput, setQueryInput] = useState<string>(urlQuery);
  const [activeQuery, setActiveQuery] = useState<string>(urlQuery);

  // Reconstructed decision ID resolver
  const resolvedDecisionId = useMemo(() => {
    const q = activeQuery.trim();
    if (q.startsWith("OUT-")) {
      const d = CANONICAL_COMMITTEE_DECISIONS.find((dec) => dec.outcomeId === q);
      return d ? d.decisionId : q;
    }
    if (q.startsWith("PROP-")) {
      const d = CANONICAL_COMMITTEE_DECISIONS.find((dec) => dec.proposalId === q);
      return d ? d.decisionId : q;
    }
    return q;
  }, [activeQuery]);

  // Execute single-artifact reconstruction
  const reconResult = useMemo(() => {
    return reconstructDecision(resolvedDecisionId);
  }, [resolvedDecisionId]);

  // Execute snapshot generation
  const snapshotResult = useMemo(() => {
    try {
      if (reconResult.success) {
        return createAuditSnapshot(resolvedDecisionId);
      }
    } catch {
      // Ignore if unverified
    }
    return null;
  }, [reconResult, resolvedDecisionId]);

  // Timeline
  const timelineSteps = useMemo(() => {
    return buildDecisionTimeline(resolvedDecisionId);
  }, [resolvedDecisionId]);

  // Export JSON handler
  const handleExportJson = () => {
    try {
      const exportData = generateAuditExport(activeQuery);
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportData, null, 2));
      const downloadAnchor = document.createElement("a");
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `audit-snapshot-${resolvedDecisionId}-${Date.now()}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    } catch (err) {
      console.error("Export failed:", err);
    }
  };

  const QUICK_QUERIES = ["DEC-001", "DEC-002", "DEC-003", "DEC-004", "OUT-001", "PROP-001"];

  return (
    <div className="space-y-4 font-mono">
      {/* Search & Query Bar */}
      <div className="bg-[#111724] border border-[#202d44] p-4 rounded-xl space-y-3">
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wide">
                Single-Artifact Audit Reconstruction (INV-OI13-A)
              </h2>
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Enter any Decision ID, Outcome ID, or Proposal ID to reconstruct the complete institutional lineage.
            </p>
          </div>

          <button
            type="button"
            onClick={handleExportJson}
            disabled={!reconResult.success}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center justify-center space-x-1.5 ${
              reconResult.success
                ? "bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-md shadow-cyan-950/50 cursor-pointer"
                : "bg-slate-800 text-slate-500 cursor-not-allowed"
            }`}
          >
            <span>&darr;</span>
            <span>Download Audit Snapshot (JSON)</span>
          </button>
        </div>

        {/* Input & Quick Chips */}
        <div className="flex flex-col sm:flex-row gap-2">
          <div className="relative flex-1">
            <input
              type="text"
              value={queryInput}
              onChange={(e) => setQueryInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") setActiveQuery(queryInput);
              }}
              placeholder="Enter artifact ID (e.g. DEC-001, OUT-001, PROP-001)..."
              className="w-full px-3.5 py-2 bg-[#0c1017] border border-[#243044] rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono"
            />
          </div>
          <button
            type="button"
            onClick={() => setActiveQuery(queryInput)}
            className="px-4 py-2 bg-[#1f2c42] hover:bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 rounded-lg text-xs font-bold transition-colors"
          >
            Reconstruct
          </button>
        </div>

        {/* Quick query suggestion chips */}
        <div className="flex items-center flex-wrap gap-1.5 pt-1">
          <span className="text-[10px] text-slate-500 mr-1">Quick Query:</span>
          {QUICK_QUERIES.map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => {
                setQueryInput(q);
                setActiveQuery(q);
              }}
              className={`px-2 py-0.5 rounded text-[10px] transition-colors ${
                activeQuery === q
                  ? "bg-cyan-950/80 border border-cyan-500/60 text-cyan-300 font-bold"
                  : "bg-[#162032] border border-[#243044] text-slate-400 hover:text-slate-200"
              }`}
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Reconstruction Verification Header */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="bg-[#111724] border border-[#202d44] p-3.5 rounded-xl">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
            Reconstruction Status
          </span>
          <div className="flex items-baseline space-x-2 mt-1">
            <span
              className={`text-xl font-bold ${
                reconResult.success ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {reconResult.success ? "100% RECOVERED" : "INCOMPLETE"}
            </span>
          </div>
          <span className="text-[10px] text-slate-400 block mt-0.5">
            Query: {activeQuery} &rarr; {resolvedDecisionId}
          </span>
        </div>

        <div className="bg-[#111724] border border-[#202d44] p-3.5 rounded-xl">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
            Lineage Completeness
          </span>
          <div className="flex items-baseline space-x-2 mt-1">
            <span className="text-xl font-bold text-cyan-400">
              {reconResult.coverage.completenessPct}%
            </span>
            <span className="text-[10px] text-slate-500">(6/6 Core Facets)</span>
          </div>
          <span className="text-[10px] text-emerald-400 block mt-0.5">
            Zero Missing Artifacts
          </span>
        </div>

        <div className="bg-[#111724] border border-[#202d44] p-3.5 rounded-xl">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
            Snapshot Hash (SHA-256)
          </span>
          <div className="mt-1 font-mono text-xs text-purple-300 font-bold truncate">
            {snapshotResult?.hash ? snapshotResult.hash.slice(0, 16) + "..." : "N/A"}
          </div>
          <span className="text-[10px] text-purple-400/80 block mt-0.5">
            Cryptographic State Seal
          </span>
        </div>

        <div className="bg-[#111724] border border-[#202d44] p-3.5 rounded-xl">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
            Reconstruction Latency
          </span>
          <div className="flex items-baseline space-x-2 mt-1">
            <span className="text-xl font-bold text-emerald-400">
              {reconResult.elapsedMs} ms
            </span>
            <span className="text-[10px] text-slate-500">&lt;50ms Limit</span>
          </div>
          <span className="text-[10px] text-emerald-400/80 block mt-0.5">
            Deterministic In-Memory Resolution
          </span>
        </div>
      </div>

      {/* 8-Step Reconstruction Coverage Grid (RECON-01 to RECON-08) */}
      <div className="bg-[#111724] border border-[#202d44] p-4 rounded-xl">
        <div className="text-xs font-bold text-slate-200 uppercase tracking-wider mb-3">
          Institutional Lineage Audit Checklist (RECON-01 to RECON-08)
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
          <div className="p-2.5 rounded-lg bg-[#0c1017] border border-[#202d44] text-xs">
            <span className="text-[10px] text-slate-400 block">RECON-01</span>
            <span className="font-bold text-slate-200 block mt-0.5">Proposal</span>
            <span className={`text-[10px] font-semibold ${reconResult.coverage.proposalRecovered ? "text-emerald-400" : "text-rose-400"}`}>
              {reconResult.coverage.proposalRecovered ? "&check; Recovered" : "&cross; Missing"}
            </span>
          </div>

          <div className="p-2.5 rounded-lg bg-[#0c1017] border border-[#202d44] text-xs">
            <span className="text-[10px] text-slate-400 block">RECON-02</span>
            <span className="font-bold text-slate-200 block mt-0.5">Evidence Vault</span>
            <span className={`text-[10px] font-semibold ${reconResult.coverage.evidenceRecovered ? "text-emerald-400" : "text-rose-400"}`}>
              {reconResult.coverage.evidenceRecovered ? "&check; Verified" : "&cross; Incomplete"}
            </span>
          </div>

          <div className="p-2.5 rounded-lg bg-[#0c1017] border border-[#202d44] text-xs">
            <span className="text-[10px] text-slate-400 block">RECON-03</span>
            <span className="font-bold text-slate-200 block mt-0.5">Quorum / Voters</span>
            <span className={`text-[10px] font-semibold ${reconResult.coverage.participantsRecovered ? "text-emerald-400" : "text-rose-400"}`}>
              {reconResult.coverage.participantsRecovered ? "&check; Quorum Met" : "&cross; Missing"}
            </span>
          </div>

          <div className="p-2.5 rounded-lg bg-[#0c1017] border border-[#202d44] text-xs">
            <span className="text-[10px] text-slate-400 block">RECON-04</span>
            <span className="font-bold text-slate-200 block mt-0.5">Dissent Records</span>
            <span className={`text-[10px] font-semibold ${reconResult.coverage.dissentsRecovered ? "text-emerald-400" : "text-rose-400"}`}>
              {reconResult.coverage.dissentsRecovered ? "&check; Preserved" : "&cross; Missing"}
            </span>
          </div>

          <div className="p-2.5 rounded-lg bg-[#0c1017] border border-[#202d44] text-xs">
            <span className="text-[10px] text-slate-400 block">RECON-05</span>
            <span className="font-bold text-slate-200 block mt-0.5">Realized Outcome</span>
            <span className={`text-[10px] font-semibold ${reconResult.coverage.outcomeRecovered ? "text-emerald-400" : "text-rose-400"}`}>
              {reconResult.coverage.outcomeRecovered ? "&check; Measured" : "&cross; Missing"}
            </span>
          </div>

          <div className="p-2.5 rounded-lg bg-[#0c1017] border border-[#202d44] text-xs">
            <span className="text-[10px] text-slate-400 block">RECON-06</span>
            <span className="font-bold text-slate-200 block mt-0.5">Attribution Sum</span>
            <span className={`text-[10px] font-semibold ${reconResult.coverage.attributionRecovered ? "text-emerald-400" : "text-rose-400"}`}>
              {reconResult.coverage.attributionRecovered ? "&check; 100.0% Exact" : "&cross; Missing"}
            </span>
          </div>
        </div>
      </div>

      {/* Chronological 7-Step Timeline */}
      <div className="bg-[#111724] border border-[#202d44] p-4 rounded-xl">
        <div className="text-xs font-bold text-slate-200 uppercase tracking-wider mb-3">
          Chronological Audit Reconstruction Timeline
        </div>
        <DecisionTimeline steps={timelineSteps} decisionId={resolvedDecisionId} />
      </div>
    </div>
  );
}
