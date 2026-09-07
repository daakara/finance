"use client";

import React, { useState } from "react";
import { AuditEvent } from "../../types/committee-intelligence";

export interface AuditTrailExplorerProps {
  events: AuditEvent[];
  isChainValid?: boolean;
  className?: string;
}

export default function AuditTrailExplorer({
  events,
  isChainValid = true,
  className = "",
}: AuditTrailExplorerProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  return (
    <div
      data-testid="audit-trail-explorer"
      className={`p-6 rounded-2xl bg-bg-surface border border-border-subtle shadow-xl space-y-6 font-sans ${className}`}
    >
      {/* Header & Verification Badge */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-border-subtle">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl">📜</span>
            <h3 className="text-header-2 text-text-primary">
              Immutable Governance Audit Trail
            </h3>
          </div>
          <p className="text-body-ui text-text-secondary text-xs mt-0.5">
            Cryptographically chained regulatory record (SHA-256). Strictly append-only.
          </p>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <span
            className={`px-3 py-1 text-xs font-mono font-bold rounded-lg border flex items-center gap-1.5 ${
              isChainValid
                ? "bg-emerald-950 text-emerald-300 border-emerald-800"
                : "bg-rose-950 text-rose-300 border-rose-800 animate-pulse"
            }`}
          >
            <span>{isChainValid ? "🛡️" : "⚠️"}</span>
            {isChainValid ? "Chain Verified (100% Tamper-Evident)" : "Tampering Detected"}
          </span>
        </div>
      </div>

      {/* Chronological Timeline */}
      <div className="space-y-3">
        {events.length === 0 ? (
          <p className="text-xs font-mono text-text-muted text-center py-6">
            No audit records found for this scope.
          </p>
        ) : (
          events.map((ev, idx) => (
            <div
              key={ev.eventId}
              className="p-3.5 rounded-xl bg-bg-surface-raised border border-border-subtle hover:border-accent-info/40 transition-colors space-y-2"
            >
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-accent-info" />
                  <span className="text-xs font-mono font-bold text-text-primary">
                    {ev.action}
                  </span>
                  <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-bg-app text-text-secondary border border-border-subtle">
                    {ev.actorRole} ({ev.actorId})
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-caption-mono text-text-muted text-xs">
                    {new Date(ev.timestamp).toLocaleString()}
                  </span>
                  <button
                    type="button"
                    onClick={() => setExpandedIndex(expandedIndex === idx ? null : idx)}
                    className="text-caption-mono text-accent-info hover:underline text-xs cursor-pointer ml-1"
                  >
                    {expandedIndex === idx ? "Hide Hash ▲" : "Inspect Hash ▼"}
                  </button>
                </div>
              </div>

              {/* Cryptographic Hash Inspector */}
              {expandedIndex === idx && (
                <div className="p-3 rounded-lg bg-bg-app border border-border-subtle font-mono text-[11px] text-text-secondary space-y-1.5 mt-2 animate-fadeIn">
                  <div className="flex items-center justify-between">
                    <span className="text-text-muted">Event ID:</span>
                    <span className="text-text-primary">{ev.eventId}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-text-muted">Previous Hash:</span>
                    <span className="text-accent-info truncate max-w-xs">{ev.previousHash}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-text-muted">Event Hash:</span>
                    <span className="text-emerald-400 font-bold truncate max-w-xs">{ev.eventHash}</span>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
