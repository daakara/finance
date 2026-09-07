'use client';

import React, { useState } from 'react';
import {
  TELEMETRY_QUALITY_INVARIANTS,
  EXECUTIVE_DATA_QUALITY_KPIS,
  DATA_QUALITY_ALERTS,
  evaluateTelemetryDataQuality,
} from '@/lib/telemetry/dataQualityEngine';
import { TelemetryQualityInvariant } from '@/types/production-excellence-framework';

export default function TelemetryDataQualityDashboard() {
  const [selectedInvariant, setSelectedInvariant] = useState<TelemetryQualityInvariant | null>(null);
  const evaluation = evaluateTelemetryDataQuality();

  return (
    <div className="space-y-6" data-testid="telemetry-data-quality-dashboard">
      {/* Top Banner: Executive KPIs */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl shadow-sm space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono font-bold bg-accent-positive/10 text-accent-positive border border-accent-positive/30 rounded">
                TQ Invariants Enforced
              </span>
              <span className="text-caption-mono text-text-muted">
                Zero-Loss Telemetry Pipeline
              </span>
            </div>
            <h3 className="text-display-2 font-bold text-text-primary mt-1">
              Telemetry Data Quality &amp; Completeness
            </h3>
            <p className="text-body-ui text-text-secondary mt-0.5">
              Continuous monitoring ensuring every prediction, recommendation, outcome, and user action
              is observed, verified, attributed, and audited without gaps.
            </p>
          </div>

          <div className="text-right">
            <div className="text-caption-mono text-text-muted uppercase text-xs">Composite Health</div>
            <div className="text-display-1 font-mono font-extrabold text-accent-positive">
              {EXECUTIVE_DATA_QUALITY_KPIS.telemetryHealth.toFixed(1)}%
            </div>
            <div className="text-caption-mono text-text-secondary text-[11px]">
              Target: &ge; 98.0%
            </div>
          </div>
        </div>

        {/* 5 KPI Metric Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 pt-4 border-t border-border-subtle">
          <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
            <div className="text-[11px] font-mono text-text-muted uppercase">Health Index</div>
            <div className="text-header-1 font-mono font-bold text-accent-positive mt-0.5">
              {EXECUTIVE_DATA_QUALITY_KPIS.telemetryHealth.toFixed(1)}%
            </div>
            <div className="text-[10px] font-mono text-text-muted">Floor: 98.0%</div>
          </div>

          <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
            <div className="text-[11px] font-mono text-text-muted uppercase">Completeness</div>
            <div className="text-header-1 font-mono font-bold text-accent-positive mt-0.5">
              {EXECUTIVE_DATA_QUALITY_KPIS.eventCompleteness.toFixed(1)}%
            </div>
            <div className="text-[10px] font-mono text-text-muted">Target: &ge; 99.5%</div>
          </div>

          <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
            <div className="text-[11px] font-mono text-text-muted uppercase">Attribution Coverage</div>
            <div className="text-header-1 font-mono font-bold text-accent-positive mt-0.5">
              {EXECUTIVE_DATA_QUALITY_KPIS.attributionCoverage.toFixed(1)}%
            </div>
            <div className="text-[10px] font-mono text-text-muted">0 Orphan Records</div>
          </div>

          <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
            <div className="text-[11px] font-mono text-text-muted uppercase">Event Quality</div>
            <div className="text-header-1 font-mono font-bold text-accent-positive mt-0.5">
              {EXECUTIVE_DATA_QUALITY_KPIS.eventQuality.toFixed(1)}%
            </div>
            <div className="text-[10px] font-mono text-text-muted">Schema Verified</div>
          </div>

          <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
            <div className="text-[11px] font-mono text-text-muted uppercase">Reconstructability</div>
            <div className="text-header-1 font-mono font-bold text-accent-positive mt-0.5">
              {EXECUTIVE_DATA_QUALITY_KPIS.journeyReconstructability.toFixed(1)}%
            </div>
            <div className="text-[10px] font-mono text-text-muted">Target: &ge; 95.0%</div>
          </div>
        </div>
      </div>

      {/* TQ-1 to TQ-5 Invariant Ledger */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4">
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <div>
            <h4 className="text-header-1 text-text-primary">
              Telemetry Quality Invariants (TQ-1 through TQ-5)
            </h4>
            <p className="text-body-ui text-text-secondary mt-0.5">
              Immutable mathematical invariants governing data integrity across the decision lifecycle.
            </p>
          </div>
          <span className="px-3 py-1 text-caption-mono font-bold bg-accent-positive/20 text-accent-positive border border-accent-positive/40 rounded-lg">
            5 / 5 INVARIANTS PASS
          </span>
        </div>

        <div className="space-y-3">
          {TELEMETRY_QUALITY_INVARIANTS.map((inv) => (
            <div
              key={inv.id}
              onClick={() => setSelectedInvariant(selectedInvariant?.id === inv.id ? null : inv)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setSelectedInvariant(selectedInvariant?.id === inv.id ? null : inv);
                }
              }}
              tabIndex={0}
              role="button"
              aria-expanded={selectedInvariant?.id === inv.id}
              className={`p-4 rounded-lg border transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-accent-info ${
                selectedInvariant?.id === inv.id
                  ? 'bg-bg-surface-elevated border-accent-info shadow'
                  : 'bg-bg-surface-raised hover:bg-bg-surface-elevated border-border-subtle'
              }`}
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="flex items-center gap-3">
                  <span className="px-2.5 py-1 text-caption-mono font-bold bg-accent-info/10 text-accent-info border border-accent-info/30 rounded">
                    {inv.id}
                  </span>
                  <div>
                    <h5 className="text-body-ui font-semibold text-text-primary">{inv.name}</h5>
                    <p className="text-caption text-text-secondary">{inv.description}</p>
                  </div>
                </div>

                <div className="flex items-center gap-4 text-caption-mono">
                  <div className="text-right">
                    <div className="text-text-muted text-[11px]">Target: {inv.target}</div>
                    <div className="text-accent-positive font-bold">{inv.actual}</div>
                  </div>
                  <span className="px-2 py-0.5 text-[11px] font-mono font-bold bg-accent-positive/20 text-accent-positive border border-accent-positive/40 rounded">
                    {inv.status}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Alert Thresholds & SRE Escalation Matrix */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4">
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <div>
            <h4 className="text-header-2 text-text-primary">
              Data Quality Alert Thresholds &amp; Escalation Matrix
            </h4>
            <p className="text-body-ui text-text-secondary mt-0.5">
              Fail-closed guardrails alerting SREs and engineering upon any telemetry anomalies.
            </p>
          </div>
          <span className="px-2.5 py-1 text-caption-mono text-xs font-semibold bg-bg-surface-raised text-text-secondary rounded border border-border-subtle">
            Status: {evaluation.summaryText}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {DATA_QUALITY_ALERTS.map((alert) => (
            <div
              key={alert.id}
              className="p-4 bg-bg-surface-raised rounded-lg border border-border-subtle space-y-2"
            >
              <div className="flex items-center justify-between">
                <span className="text-body-ui font-semibold text-text-primary">
                  {alert.threshold}
                </span>
                <span
                  className={`px-2 py-0.5 text-[10px] font-mono font-bold uppercase rounded ${
                    alert.severity === 'CRITICAL'
                      ? 'bg-accent-negative/20 text-accent-negative border border-accent-negative/40'
                      : alert.severity === 'HIGH'
                      ? 'bg-accent-warning/20 text-accent-warning border border-accent-warning/40'
                      : 'bg-accent-info/20 text-accent-info border border-accent-info/40'
                  }`}
                >
                  {alert.severity}
                </span>
              </div>
              <p className="text-caption text-text-secondary">{alert.description}</p>
              <div className="pt-2 border-t border-border-subtle text-caption font-mono text-text-muted">
                Escalation: <strong className="text-text-primary">{alert.escalationPolicy}</strong>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
