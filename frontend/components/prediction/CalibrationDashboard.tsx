"use client";

import React, { useState } from "react";
import { CalibrationReport, DriftReport, ModelMetadata } from "../../types/predictive-intelligence";

export interface CalibrationDashboardProps {
  activeModel: ModelMetadata;
  calibrationReport?: CalibrationReport;
  driftReport?: DriftReport;
  onRollback?: (type: "SOFT" | "HARD") => void;
  className?: string;
}

export default function CalibrationDashboard({
  activeModel,
  calibrationReport = {
    ece: 0.034,
    brierScore: 0.12,
    samples: 1482,
    buckets: [
      { rangeMin: 0.0, rangeMax: 0.1, predictionCount: 120, actualSuccessRate: 0.05 },
      { rangeMin: 0.1, rangeMax: 0.2, predictionCount: 140, actualSuccessRate: 0.14 },
      { rangeMin: 0.2, rangeMax: 0.3, predictionCount: 160, actualSuccessRate: 0.22 },
      { rangeMin: 0.3, rangeMax: 0.4, predictionCount: 180, actualSuccessRate: 0.35 },
      { rangeMin: 0.4, rangeMax: 0.5, predictionCount: 150, actualSuccessRate: 0.44 },
      { rangeMin: 0.5, rangeMax: 0.6, predictionCount: 170, actualSuccessRate: 0.53 },
      { rangeMin: 0.6, rangeMax: 0.7, predictionCount: 190, actualSuccessRate: 0.64 },
      { rangeMin: 0.7, rangeMax: 0.8, predictionCount: 140, actualSuccessRate: 0.73 },
      { rangeMin: 0.8, rangeMax: 0.9, predictionCount: 130, actualSuccessRate: 0.84 },
      { rangeMin: 0.9, rangeMax: 1.0, predictionCount: 102, actualSuccessRate: 0.92 },
    ],
    status: "PASS",
  },
  driftReport = {
    generatedAt: new Date().toISOString(),
    overallStatus: "STABLE",
    metrics: [
      {
        metricName: "Feature Distribution Shift",
        warningThreshold: 10.0,
        criticalThreshold: 20.0,
        currentShiftPct: 4.2,
        status: "STABLE",
      },
      {
        metricName: "Macro Regime Divergence",
        warningThreshold: 10.0,
        criticalThreshold: 20.0,
        currentShiftPct: 6.8,
        status: "STABLE",
      },
      {
        metricName: "Outcome Error Shift",
        warningThreshold: 10.0,
        criticalThreshold: 20.0,
        currentShiftPct: 3.1,
        status: "STABLE",
      },
    ],
  },
  onRollback,
  className = "",
}: CalibrationDashboardProps) {
  const [rollbackStatus, setRollbackStatus] = useState<string | null>(null);

  const handleRollbackClick = (type: "SOFT" | "HARD") => {
    onRollback?.(type);
    setRollbackStatus(type === "SOFT" ? "Soft Rollback Active (UI Hidden)" : "Hard Rollback Executed (Restored Previous Model)");
  };

  const isEcePassing = calibrationReport.ece <= 0.05;
  const isBrierPassing = calibrationReport.brierScore <= 0.15;

  return (
    <div
      role="region"
      aria-label="Calibration Dashboard"
      className={`rounded-lg border border-border-subtle bg-surface-card p-5 ${className}`}
    >
      <div className="flex items-center justify-between border-b border-border-subtle pb-4">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-wider text-text-primary flex items-center gap-2">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-500" />
            Institutional Model Calibration &amp; Drift Monitor
          </h2>
          <p className="text-xs text-text-muted mt-1">
            Active Model: <span className="font-mono text-text-primary font-semibold">{activeModel.modelName} ({activeModel.version})</span>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] text-text-muted">
            Checksum: {activeModel.checksum.substring(0, 16)}...
          </span>
          <span className="px-2 py-0.5 rounded text-xs font-mono font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            ACTIVE
          </span>
        </div>
      </div>

      {rollbackStatus && (
        <div className="mt-3 rounded border border-accent-amber/30 bg-accent-amber/10 p-2.5 text-xs text-accent-amber font-mono">
          {rollbackStatus}
        </div>
      )}

      {/* Metrics Row */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="rounded-md border border-border-subtle bg-surface-subtle p-3">
          <span className="text-[10px] font-mono text-text-muted uppercase">Expected Calibration Error (ECE)</span>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="font-mono text-xl font-bold text-text-primary">
              {(calibrationReport.ece * 100).toFixed(1)}%
            </span>
            <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
              isEcePassing ? "bg-emerald-500/10 text-emerald-400" : "bg-accent-rose/10 text-accent-rose"
            }`}>
              {isEcePassing ? "PASS (≤5%)" : "FAIL (>5%)"}
            </span>
          </div>
        </div>

        <div className="rounded-md border border-border-subtle bg-surface-subtle p-3">
          <span className="text-[10px] font-mono text-text-muted uppercase">Brier Score (Mean Squared Error)</span>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="font-mono text-xl font-bold text-text-primary">
              {calibrationReport.brierScore.toFixed(3)}
            </span>
            <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
              isBrierPassing ? "bg-emerald-500/10 text-emerald-400" : "bg-accent-rose/10 text-accent-rose"
            }`}>
              {isBrierPassing ? "PASS (≤0.15)" : "FAIL (>0.15)"}
            </span>
          </div>
        </div>

        <div className="rounded-md border border-border-subtle bg-surface-subtle p-3">
          <span className="text-[10px] font-mono text-text-muted uppercase">Drift Status</span>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="font-mono text-xl font-bold text-text-primary">
              {driftReport.overallStatus}
            </span>
            <span className="text-[10px] font-mono text-text-muted">
              {calibrationReport.samples} Samples
            </span>
          </div>
        </div>
      </div>

      {/* Reliability Curve Mini Table */}
      <div className="mt-4 pt-3 border-t border-border-subtle">
        <span className="text-xs font-semibold text-text-primary uppercase tracking-wider">
          10-Bin Reliability Curve Breakdown
        </span>
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-[11px] font-mono text-left">
            <thead>
              <tr className="text-text-muted border-b border-border-subtle/60">
                <th className="pb-1">Bucket</th>
                <th className="pb-1">Samples</th>
                <th className="pb-1">Observed Success</th>
                <th className="pb-1">Alignment</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-subtle/30 text-text-secondary">
              {calibrationReport.buckets.map((b, idx) => {
                const mid = (b.rangeMin + b.rangeMax) / 2;
                const err = Math.abs(b.actualSuccessRate - mid);
                return (
                  <tr key={idx}>
                    <td className="py-1">[{b.rangeMin.toFixed(1)} - {b.rangeMax.toFixed(1)}]</td>
                    <td className="py-1">{b.predictionCount}</td>
                    <td className="py-1">{(b.actualSuccessRate * 100).toFixed(1)}%</td>
                    <td className="py-1">
                      <span className={err <= 0.05 ? "text-emerald-400" : "text-accent-amber"}>
                        Δ {(err * 100).toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Rollback Safety Controls */}
      <div className="mt-4 pt-4 border-t border-border-subtle flex items-center justify-between">
        <span className="text-xs text-text-muted">
          Governance Safety: Automated rollback on ECE &gt; 15% or Precision &lt; 50%.
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => handleRollbackClick("SOFT")}
            className="px-3 py-1.5 rounded border border-accent-amber/40 bg-accent-amber/10 text-accent-amber text-xs font-medium hover:bg-accent-amber/20 transition-colors"
          >
            Trigger Soft Rollback
          </button>
          <button
            onClick={() => handleRollbackClick("HARD")}
            className="px-3 py-1.5 rounded border border-accent-rose/40 bg-accent-rose/10 text-accent-rose text-xs font-medium hover:bg-accent-rose/20 transition-colors"
          >
            Trigger Hard Rollback
          </button>
        </div>
      </div>
    </div>
  );
}
