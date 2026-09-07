"use client";

import React, { useState } from "react";
import {
  AcknowledgementStatus,
  CommitteeBaseline,
  CommitteeConsensus,
  CommitteeRole,
} from "../../types/committee-intelligence";

export interface CommitteeBaselineBannerProps {
  baseline: CommitteeBaseline;
  consensus?: CommitteeConsensus;
  userRole?: CommitteeRole;
  onAcknowledge?: (status: AcknowledgementStatus, rationale?: string) => void;
  onOpenAuditTrail?: () => void;
  className?: string;
}

export default function CommitteeBaselineBanner({
  baseline,
  consensus = {
    ticker: baseline.ticker,
    totalReviewers: 10,
    acknowledged: 8,
    disagreed: 2,
    pending: 0,
    escalated: 0,
    consensusPercent: 80,
  },
  userRole = CommitteeRole.PORTFOLIO_MANAGER,
  onAcknowledge,
  onOpenAuditTrail,
  className = "",
}: CommitteeBaselineBannerProps) {
  const [showRationaleModal, setShowRationaleModal] = useState(false);
  const [rationale, setRationale] = useState("");
  const [userStatus, setUserStatus] = useState<AcknowledgementStatus | null>(null);

  const handleAgree = () => {
    setUserStatus(AcknowledgementStatus.ACKNOWLEDGED);
    if (onAcknowledge) onAcknowledge(AcknowledgementStatus.ACKNOWLEDGED);
  };

  const handleDisagreeSubmit = () => {
    if (!rationale.trim()) return;
    setUserStatus(AcknowledgementStatus.DISAGREED);
    setShowRationaleModal(false);
    if (onAcknowledge) onAcknowledge(AcknowledgementStatus.DISAGREED, rationale);
  };

  return (
    <div
      data-testid="committee-baseline-banner"
      className={`p-4 rounded-2xl bg-bg-surface border border-accent-info/30 shadow-lg space-y-3 font-sans ${className}`}
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border-subtle/80">
        <div className="flex items-center gap-2.5 flex-wrap">
          <span className="px-2.5 py-0.5 text-caption-mono font-bold uppercase rounded bg-accent-info/15 text-accent-info border border-accent-info/30">
            Committee Scope
          </span>
          <span className="text-xs font-mono font-bold text-text-primary">
            {baseline.committeeId.toUpperCase()} Baseline: ${baseline.ticker}
          </span>
          <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-bg-surface-raised text-text-secondary border border-border-subtle">
            Approved by {baseline.approvedBy || "CIO Office"}
          </span>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <span className="text-caption-mono text-xs text-text-secondary">
            Consensus:
          </span>
          <span className="px-2 py-0.5 text-xs font-mono font-bold rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
            {consensus.consensusPercent}% Aligned
          </span>
          {onOpenAuditTrail && (
            <button
              type="button"
              onClick={onOpenAuditTrail}
              className="text-xs font-mono text-accent-info hover:underline ml-2 cursor-pointer"
            >
              Audit Trail →
            </button>
          )}
        </div>
      </div>

      {/* Consensus Progress & Voting Action Row */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pt-1">
        <div className="space-y-1 flex-1 max-w-md">
          <div className="flex justify-between text-[11px] font-mono text-text-muted">
            <span>{consensus.acknowledged} Acknowledged</span>
            <span>{consensus.disagreed} Disagreed</span>
          </div>
          <div className="h-1.5 w-full bg-bg-surface-raised rounded-full overflow-hidden flex">
            <div
              className="bg-emerald-500 h-full"
              style={{ width: `${consensus.consensusPercent}%` }}
            />
            <div
              className="bg-rose-500 h-full"
              style={{ width: `${100 - consensus.consensusPercent}%` }}
            />
          </div>
        </div>

        {/* Action CTAs */}
        <div className="flex items-center gap-2 shrink-0">
          {userStatus === AcknowledgementStatus.ACKNOWLEDGED && (
            <span className="text-xs font-mono text-emerald-400 font-bold flex items-center gap-1">
              ✓ Signed & Aligned
            </span>
          )}
          {userStatus === AcknowledgementStatus.DISAGREED && (
            <span className="text-xs font-mono text-rose-400 font-bold flex items-center gap-1">
              ✗ Disagreement Recorded
            </span>
          )}

          {!userStatus && (
            <>
              <button
                type="button"
                onClick={handleAgree}
                className="px-3 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-xs font-mono font-bold shadow transition-all active:scale-95 cursor-pointer"
              >
                Agree & Sign
              </button>
              <button
                type="button"
                onClick={() => setShowRationaleModal(true)}
                className="px-3 py-1.5 rounded-lg bg-bg-surface-raised hover:bg-bg-surface-elevated text-text-secondary hover:text-rose-400 border border-border-subtle text-xs font-mono transition-all cursor-pointer"
              >
                Disagree
              </button>
            </>
          )}
        </div>
      </div>

      {/* Disagreement Rationale Modal */}
      {showRationaleModal && (
        <div className="p-4 rounded-xl bg-bg-surface-raised border border-rose-500/40 space-y-3 mt-3 animate-fadeIn">
          <h4 className="text-xs font-mono font-bold text-rose-400">
            Record Model Disagreement & Thesis Counterargument
          </h4>
          <textarea
            value={rationale}
            onChange={(e) => setRationale(e.target.value)}
            placeholder="Explain why you disagree with this baseline model output (e.g. macro liquidity headwind, delayed filing)..."
            className="w-full p-2.5 rounded-lg bg-bg-app border border-border-subtle text-xs font-sans text-text-primary focus:outline-none focus:border-accent-info"
            rows={3}
          />
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setShowRationaleModal(false)}
              className="px-3 py-1 text-xs font-mono text-text-secondary hover:text-text-primary cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleDisagreeSubmit}
              disabled={!rationale.trim()}
              className="px-3 py-1 rounded bg-rose-500 hover:bg-rose-400 text-slate-950 text-xs font-mono font-bold transition-colors cursor-pointer disabled:opacity-50"
            >
              Submit Disagreement
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
