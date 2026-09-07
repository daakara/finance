"use client";

import React, { useState } from "react";
import { PredictionRecord } from "../../types/predictive-intelligence";

export interface PredictedAttentionFeedProps {
  predictions: PredictionRecord[];
  onAcknowledge?: (predictionId: string) => void;
  className?: string;
}

export default function PredictedAttentionFeed({
  predictions,
  onAcknowledge,
  className = "",
}: PredictedAttentionFeedProps) {
  const [acknowledgedIds, setAcknowledgedIds] = useState<Set<string>>(new Set());

  // Filter out sub-threshold or expired items
  const now = new Date().toISOString();
  const validPredictions = predictions.filter(
    (p) => p.status === "ACTIVE" && p.expirationAt > now && p.probability >= 0.60
  );

  const handleAck = (id: string) => {
    setAcknowledgedIds((prev) => new Set(prev).add(id));
    onAcknowledge?.(id);
  };

  return (
    <div
      role="region"
      aria-label="Predicted Attention Feed"
      className={`rounded-lg border border-border-subtle bg-surface-card p-5 ${className}`}
    >
      <div className="flex items-center justify-between border-b border-border-subtle pb-4">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-wider text-text-primary flex items-center gap-2">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-accent-blue" />
            Predicted Attention Feed (Stage 6 Anticipatory Engine)
          </h2>
          <p className="text-xs text-text-muted mt-1">
            Hypotheses grounded in empirical flow momentum, corridor proximity, and volatility drift.
          </p>
        </div>
        <span className="text-xs font-mono px-2.5 py-1 rounded bg-surface-subtle text-text-secondary border border-border-subtle">
          {validPredictions.length} Active Forecasts
        </span>
      </div>

      <div className="mt-4 space-y-3">
        {validPredictions.length === 0 ? (
          <div className="py-8 text-center text-xs text-text-muted">
            No pending attention forecasts require review. All active assets remain in equilibrium.
          </div>
        ) : (
          validPredictions.map((pred) => {
            const isAck = acknowledgedIds.has(pred.predictionId);
            const isCritical = pred.severity === "CRITICAL";
            const probPct = Math.round(pred.probability * 100);

            return (
              <div
                key={pred.predictionId}
                className={`rounded-md border p-4 transition-colors ${
                  isAck
                    ? "border-border-subtle/50 bg-surface-subtle/30 opacity-70"
                    : isCritical
                    ? "border-accent-rose/30 bg-accent-rose/5"
                    : "border-border-subtle bg-surface-subtle"
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2.5">
                    <span className="font-mono font-bold text-base text-text-primary">
                      {pred.ticker}
                    </span>
                    <span
                      className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded font-semibold ${
                        isCritical
                          ? "bg-accent-rose/15 text-accent-rose border border-accent-rose/30"
                          : "bg-accent-amber/15 text-accent-amber border border-accent-amber/30"
                      }`}
                    >
                      {pred.predictionType.replace(/_/g, " ")}
                    </span>
                    <span className="text-xs font-mono text-text-muted">
                      Confidence: <strong className="text-text-primary">{pred.confidence} ({probPct}%)</strong>
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="text-[11px] font-mono text-text-muted">
                      Expires: {new Date(pred.expirationAt).toLocaleDateString()}
                    </span>
                    {!isAck && (
                      <button
                        onClick={() => handleAck(pred.predictionId)}
                        className="rounded border border-border-subtle bg-surface-card px-2.5 py-1 text-xs font-medium text-text-primary hover:bg-surface-base transition-colors"
                      >
                        Acknowledge
                      </button>
                    )}
                  </div>
                </div>

                {/* Explanations First (AC-PI-07) */}
                <div className="mt-3 pt-3 border-t border-border-subtle/50">
                  <p className="text-[11px] font-semibold text-text-muted uppercase tracking-wider mb-1">
                    Contributing Drivers &amp; Rationale:
                  </p>
                  <ul className="space-y-1">
                    {pred.rationale.map((reason, idx) => (
                      <li key={idx} className="text-xs text-text-secondary flex items-start gap-1.5">
                        <span className="text-accent-blue font-bold">•</span>
                        <span>{reason}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="mt-3 flex items-center justify-between text-[11px] font-mono text-text-muted">
                  <span>Model: {pred.modelVersion}</span>
                  <span>Anchor Hash: {pred.currentStateHash.substring(0, 16)}...</span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
