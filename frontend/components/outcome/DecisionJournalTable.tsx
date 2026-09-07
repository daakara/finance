"use client";

import React from "react";
import { OutcomeRecord } from "../../types/outcome-intelligence";

export interface DecisionJournalTableProps {
  outcomes?: OutcomeRecord[];
  className?: string;
}

export default function DecisionJournalTable({
  outcomes = [
    {
      outcomeId: "out-cprx-01",
      predictionId: "pred-cprx-orig",
      ticker: "CPRX",
      predictedAt: "2026-09-01T08:00:00Z",
      resolvedAt: "2026-09-05T16:00:00Z",
      outcomeClass: "SUCCESS",
      attributionCategory: "TARGET_REACHED",
      explanation: "Institutional accumulation velocity (+2.1σ) absorbed supply and achieved Target 1",
      outcomeReturnPct: 7.4,
      outcomeConfidence: 0.89,
      snapshotHash: "sha256:cprxbaselinehash1",
    },
    {
      outcomeId: "out-nvda-01",
      predictionId: "pred-nvda-orig",
      ticker: "NVDA",
      predictedAt: "2026-09-02T08:00:00Z",
      resolvedAt: "2026-09-06T16:00:00Z",
      outcomeClass: "SUCCESS",
      attributionCategory: "TARGET_REACHED",
      explanation: "Execution buy zone entry triggered with strong semiconductor momentum",
      outcomeReturnPct: 8.2,
      outcomeConfidence: 0.86,
      snapshotHash: "sha256:nvdabaselinehash1",
    },
    {
      outcomeId: "out-intc-01",
      predictionId: "pred-intc-orig",
      ticker: "INTC",
      predictedAt: "2026-09-01T08:00:00Z",
      resolvedAt: "2026-09-04T16:00:00Z",
      outcomeClass: "FAILURE",
      attributionCategory: "STOP_TRIGGERED",
      explanation: "Adverse price movement breached stop floor at $19.20",
      outcomeReturnPct: -4.1,
      outcomeConfidence: 0.82,
      snapshotHash: "sha256:intcbaselinehash1",
    },
    {
      outcomeId: "out-spy-01",
      predictionId: "pred-spy-orig",
      ticker: "SPY",
      predictedAt: "2026-09-03T08:00:00Z",
      resolvedAt: "2026-09-06T16:00:00Z",
      outcomeClass: "INVALIDATED",
      attributionCategory: "REGIME_CHANGE",
      explanation: "VIX surge to 24.5 triggered macro transition to DEFENSIVE regime",
      outcomeReturnPct: -0.8,
      outcomeConfidence: 0.78,
      snapshotHash: "sha256:spybaselinehash1",
    },
  ],
  className = "",
}: DecisionJournalTableProps) {
  return (
    <div
      role="region"
      aria-label="Decision Journal Table"
      className={`rounded-lg border border-border-subtle bg-surface-card p-5 ${className}`}
    >
      <div className="flex items-center justify-between border-b border-border-subtle pb-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-text-primary">
          Institutional Decision Journal &amp; Outcome Ledger
        </h3>
        <span className="text-[11px] font-mono text-text-muted">
          {outcomes.length} Audit Entries
        </span>
      </div>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-left text-xs font-mono">
          <thead>
            <tr className="text-text-muted border-b border-border-subtle/60 text-[11px]">
              <th className="pb-2">Ticker</th>
              <th className="pb-2">Resolved Date</th>
              <th className="pb-2">Outcome</th>
              <th className="pb-2">Attribution</th>
              <th className="pb-2 text-right">Realized Return</th>
              <th className="pb-2 pl-4">Causal Explanation</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border-subtle/30 text-text-secondary">
            {outcomes.map((out) => {
              const isSuccess = out.outcomeClass === "SUCCESS";
              const isFailure = out.outcomeClass === "FAILURE";
              const isInvalidated = out.outcomeClass === "INVALIDATED";

              return (
                <tr key={out.outcomeId} className="hover:bg-surface-subtle/50 transition-colors">
                  <td className="py-2.5 font-bold text-text-primary">{out.ticker}</td>
                  <td className="py-2.5 text-text-muted">
                    {new Date(out.resolvedAt).toLocaleDateString()}
                  </td>
                  <td className="py-2.5">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        isSuccess
                          ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                          : isFailure
                          ? "bg-accent-rose/10 text-accent-rose border border-accent-rose/20"
                          : isInvalidated
                          ? "bg-slate-700/40 text-slate-300 border border-slate-600/30"
                          : "bg-accent-amber/10 text-accent-amber border border-accent-amber/20"
                      }`}
                    >
                      {out.outcomeClass}
                    </span>
                  </td>
                  <td className="py-2.5 text-text-secondary">
                    {out.attributionCategory.replace(/_/g, " ")}
                  </td>
                  <td className="py-2.5 text-right font-bold">
                    <span
                      className={
                        (out.outcomeReturnPct ?? 0) >= 0 ? "text-emerald-400" : "text-accent-rose"
                      }
                    >
                      {(out.outcomeReturnPct ?? 0) >= 0 ? "+" : ""}
                      {(out.outcomeReturnPct ?? 0).toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-2.5 pl-4 text-text-muted text-[11px] max-w-xs truncate" title={out.explanation}>
                    {out.explanation}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
