"use client";

import React, { useState } from "react";
import { OutcomeClass } from "../../types/outcome-intelligence";

export interface OutcomeCardItem {
  outcomeId: string;
  predictionId: string;
  ticker: string;
  resolvedAt: string;
  outcomeClass: OutcomeClass;
  primaryDriver: string;
  returnPct: number;
  explanation: string;
  snapshotHash: string;
  confidence: number;
}

export interface RecentOutcomesScorecardProps {
  outcomes?: OutcomeCardItem[];
  className?: string;
}

export default function RecentOutcomesScorecard({
  outcomes = [
    {
      outcomeId: "out-cprx-01",
      predictionId: "pred-cprx-orig",
      ticker: "CPRX",
      resolvedAt: "2026-09-05T16:00:00Z",
      outcomeClass: "SUCCESS",
      primaryDriver: "Institutional Accumulation",
      returnPct: 14.2,
      explanation: "Institutional accumulation velocity (+2.1σ) absorbed supply and achieved Target 1 corridor at $19.80.",
      snapshotHash: "sha256:cprxbaselinehash1",
      confidence: 0.89,
    },
    {
      outcomeId: "out-nvda-01",
      predictionId: "pred-nvda-orig",
      ticker: "NVDA",
      resolvedAt: "2026-09-06T16:00:00Z",
      outcomeClass: "SUCCESS",
      primaryDriver: "Relative Strength Breakout",
      returnPct: 18.7,
      explanation: "Semiconductor breadth leadership propelled price past Target 2 resistance with accelerating volume.",
      snapshotHash: "sha256:nvdabaselinehash1",
      confidence: 0.86,
    },
    {
      outcomeId: "out-aapl-01",
      predictionId: "pred-aapl-orig",
      ticker: "AAPL",
      resolvedAt: "2026-09-04T16:00:00Z",
      outcomeClass: "FAILURE",
      primaryDriver: "Regime Deterioration",
      returnPct: -4.3,
      explanation: "Macro transition to DEFENSIVE regime sparked aggressive tech selloff, triggering hard stop floor at $19.20.",
      snapshotHash: "sha256:aaplbaselinehash1",
      confidence: 0.82,
    },
    {
      outcomeId: "out-amd-01",
      predictionId: "pred-amd-orig",
      ticker: "AMD",
      resolvedAt: "2026-09-05T18:00:00Z",
      outcomeClass: "SUCCESS",
      primaryDriver: "Sector Rotation Alignment",
      returnPct: 12.1,
      explanation: "Capital rotation into datacenter hardware drove clean continuation to Target 1.",
      snapshotHash: "sha256:amdbaselinehash1",
      confidence: 0.85,
    },
    {
      outcomeId: "out-msft-01",
      predictionId: "pred-msft-orig",
      ticker: "MSFT",
      resolvedAt: "2026-09-03T16:00:00Z",
      outcomeClass: "PARTIAL_SUCCESS",
      primaryDriver: "Macro Tailwinds",
      returnPct: 6.4,
      explanation: "Achieved >50% favorable excursion toward Target 1 before encountering pre-earnings consolidation.",
      snapshotHash: "sha256:msftbaselinehash1",
      confidence: 0.81,
    },
    {
      outcomeId: "out-tsla-01",
      predictionId: "pred-tsla-orig",
      ticker: "TSLA",
      resolvedAt: "2026-09-02T16:00:00Z",
      outcomeClass: "EXPIRED",
      primaryDriver: "Thesis Duration Elapsed",
      returnPct: 0.0,
      explanation: "Observation window closed with price remaining within neutral range (-0.4% to +1.2%).",
      snapshotHash: "sha256:tslabaselinehash1",
      confidence: 0.74,
    },
  ],
  className = "",
}: RecentOutcomesScorecardProps) {
  const [filterClass, setFilterClass] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const filtered = outcomes.filter((item) => {
    if (filterClass !== "ALL" && item.outcomeClass !== filterClass) return false;
    if (searchQuery.trim() && !item.ticker.toLowerCase().includes(searchQuery.toLowerCase().trim())) {
      return false;
    }
    return true;
  });

  const getStatusBadge = (status: OutcomeClass) => {
    switch (status) {
      case "SUCCESS":
        return "bg-emerald-500/10 text-emerald-400 border-emerald-500/30";
      case "PARTIAL_SUCCESS":
        return "bg-amber-500/10 text-amber-400 border-amber-500/30";
      case "FAILURE":
        return "bg-rose-500/10 text-rose-400 border-rose-500/30";
      case "EXPIRED":
        return "bg-surface-raised text-text-muted border-border-subtle";
      case "INVALIDATED":
        return "bg-purple-500/10 text-purple-300 border-purple-500/30";
      default:
        return "bg-surface-raised text-text-muted border-border-subtle";
    }
  };

  return (
    <div
      role="region"
      aria-label="Recent Outcomes Scorecard"
      className={`rounded-2xl border border-border-subtle bg-surface-card p-6 shadow-sm ${className}`}
    >
      {/* Header with Search and Filter Pills */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border-subtle pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-text-primary">
              Recent Decision Resolutions · Empirical Scorecard
            </h3>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Auditable trail of recently resolved trade theses with deterministic attribution.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <input
            type="text"
            placeholder="Search ticker..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="px-3 py-1 bg-surface-base border border-border-subtle rounded-lg text-xs font-mono text-text-primary placeholder:text-text-muted focus:outline-none focus:border-emerald-500/50 w-28 sm:w-36"
          />

          <div className="flex items-center bg-surface-base border border-border-subtle rounded-lg p-0.5 text-xs font-mono">
            {["ALL", "SUCCESS", "FAILURE"].map((cls) => (
              <button
                key={cls}
                type="button"
                onClick={() => setFilterClass(cls)}
                className={`px-2.5 py-1 rounded-md transition-colors ${
                  filterClass === cls
                    ? "bg-surface-raised text-text-primary font-semibold"
                    : "text-text-muted hover:text-text-secondary"
                }`}
              >
                {cls}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Grid of Outcome Cards */}
      <div className="mt-5 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map((item) => {
          const isExpanded = expandedId === item.outcomeId;
          const isSuccess = item.outcomeClass === "SUCCESS";
          const isFailure = item.outcomeClass === "FAILURE";

          return (
            <div
              key={item.outcomeId}
              onClick={() => setExpandedId(isExpanded ? null : item.outcomeId)}
              className={`cursor-pointer rounded-xl border p-4 transition-all duration-200 ${
                isExpanded
                  ? "border-emerald-500/50 bg-surface-subtle shadow-xs"
                  : "border-border-subtle bg-surface-subtle/40 hover:border-border-subtle hover:bg-surface-subtle"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-base font-bold text-text-primary">
                    {item.ticker}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${getStatusBadge(
                      item.outcomeClass
                    )}`}
                  >
                    {item.outcomeClass}
                  </span>
                </div>

                <div
                  className={`font-mono text-sm font-bold ${
                    isSuccess
                      ? "text-emerald-400"
                      : isFailure
                      ? "text-rose-400"
                      : "text-text-secondary"
                  }`}
                >
                  {item.returnPct > 0 ? `+${item.returnPct.toFixed(1)}%` : `${item.returnPct.toFixed(1)}%`}
                </div>
              </div>

              <div className="mt-2.5 flex items-center justify-between text-xs text-text-muted font-mono">
                <span>Driver: <strong className="text-text-secondary">{item.primaryDriver}</strong></span>
                <span>{(item.confidence * 100).toFixed(0)}% Conf</span>
              </div>

              <p className="mt-2 text-xs text-text-secondary line-clamp-2 leading-relaxed">
                {item.explanation}
              </p>

              {/* Expandable Technical Ledger Metadata */}
              {isExpanded && (
                <div className="mt-3 pt-3 border-t border-border-subtle text-[11px] font-mono space-y-1.5 animate-fadeIn">
                  <div className="flex justify-between text-text-muted">
                    <span>Outcome ID:</span>
                    <span className="text-text-primary">{item.outcomeId}</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>Prediction ID:</span>
                    <span className="text-text-primary">{item.predictionId}</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>Resolved At:</span>
                    <span className="text-text-primary">{new Date(item.resolvedAt).toLocaleDateString()}</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>Snapshot Hash:</span>
                    <span className="text-text-muted truncate max-w-[150px]">{item.snapshotHash}</span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {filtered.length === 0 && (
        <div className="mt-5 p-8 text-center text-xs font-mono text-text-muted rounded-xl border border-dashed border-border-subtle">
          No resolved outcomes matching filter criteria. Adjust your search or filter tags.
        </div>
      )}
    </div>
  );
}
