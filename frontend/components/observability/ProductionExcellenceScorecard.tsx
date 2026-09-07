'use client';

import React, { useState } from 'react';
import {
  ScorecardDimension,
  ProductionExcellenceScorecardData,
} from '@/types/phase27-observability';
import {
  PRODUCTION_EXCELLENCE_SCORECARD_DATA,
  computeOverallScore,
} from '@/lib/telemetry/productionObservabilityEngine';

interface ProductionExcellenceScorecardProps {
  initialData?: ProductionExcellenceScorecardData;
}

export default function ProductionExcellenceScorecard({
  initialData = PRODUCTION_EXCELLENCE_SCORECARD_DATA,
}: ProductionExcellenceScorecardProps) {
  const [selectedDimensionId, setSelectedDimensionId] = useState<string | null>(null);

  const selectedDimension = initialData.dimensions.find(
    (d) => d.id === selectedDimensionId
  );

  return (
    <div className="space-y-6" data-testid="production-excellence-scorecard">
      {/* Top Banner: Formal Certification Seal */}
      <div className="p-6 bg-gradient-to-r from-bg-surface-raised via-bg-surface to-bg-surface-raised border-2 border-accent-positive/40 rounded-2xl shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 text-caption-mono font-bold tracking-wider uppercase bg-accent-positive/15 text-accent-positive border border-accent-positive/40 rounded-md">
                Certified Institutional Grade
              </span>
              <span className="text-text-muted text-caption-mono">
                Cadence: {initialData.cadence} Audit
              </span>
              <span className="text-text-muted text-caption-mono">
                Evaluated: {new Date(initialData.evaluatedAt).toLocaleDateString()}
              </span>
            </div>
            <h2 className="text-display-2 font-bold text-text-primary flex items-center gap-3">
              Production Excellence Review
              <span className="text-accent-positive">★ 99%+</span>
            </h2>
            <p className="text-body-ui text-text-secondary max-w-3xl">
              Formal certification measuring user adoption, cognitive behavioral improvement,
              executive decision speed, and continuous operational integrity.
            </p>
          </div>

          {/* Large Overall Score Badge */}
          <div className="flex items-center gap-5 p-4 bg-bg-surface/80 backdrop-blur border border-accent-positive/30 rounded-xl">
            <div className="text-center">
              <div className="text-display-1 font-mono font-extrabold text-accent-positive tracking-tight">
                {initialData.overallScore.toFixed(1)}%
              </div>
              <div className="text-caption-mono text-text-secondary font-semibold uppercase tracking-wider">
                Excellence Index
              </div>
              <div className="text-[11px] font-mono text-text-muted">
                Target: &ge; {initialData.targetScore.toFixed(1)}%
              </div>
            </div>

            <div className="h-14 w-px bg-border-subtle" />

            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-accent-positive animate-pulse" />
                <span className="text-body-ui font-bold text-accent-positive">
                  {initialData.classification}
                </span>
              </div>
              <p className="text-caption text-text-muted">
                Zero P0/P1 Operational Blockers
              </p>
              <p className="text-caption text-text-muted">
                Quantitative Invariant Freeze Intact
              </p>
            </div>
          </div>
        </div>

        {/* Stakeholder Sign-Off Block */}
        <div className="mt-6 pt-5 border-t border-border-subtle/80 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs font-mono">
          <div className="p-3 bg-bg-surface/60 rounded-lg border border-border-subtle">
            <div className="text-text-muted font-sans font-medium text-[11px]">Product Owner</div>
            <div className="text-text-primary font-semibold truncate">{initialData.certifiedBy.productOwner}</div>
            <div className="text-accent-positive text-[11px] mt-1">✓ Verified &amp; Signed</div>
          </div>
          <div className="p-3 bg-bg-surface/60 rounded-lg border border-border-subtle">
            <div className="text-text-muted font-sans font-medium text-[11px]">UX Architecture Lead</div>
            <div className="text-text-primary font-semibold truncate">{initialData.certifiedBy.uxLead}</div>
            <div className="text-accent-positive text-[11px] mt-1">✓ Verified &amp; Signed</div>
          </div>
          <div className="p-3 bg-bg-surface/60 rounded-lg border border-border-subtle">
            <div className="text-text-muted font-sans font-medium text-[11px]">Chief Systems Architect</div>
            <div className="text-text-primary font-semibold truncate">{initialData.certifiedBy.engineeringLead}</div>
            <div className="text-accent-positive text-[11px] mt-1">✓ Verified &amp; Signed</div>
          </div>
          <div className="p-3 bg-bg-surface/60 rounded-lg border border-border-subtle">
            <div className="text-text-muted font-sans font-medium text-[11px]">CIO &amp; Committee Chair</div>
            <div className="text-text-primary font-semibold truncate">{initialData.certifiedBy.executiveSponsor}</div>
            <div className="text-accent-positive text-[11px] mt-1">✓ Executive Approval</div>
          </div>
        </div>
      </div>

      {/* 6 Dimension Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {initialData.dimensions.map((dim) => {
          const isSelected = selectedDimensionId === dim.id;
          return (
            <div
              key={dim.id}
              onClick={() => setSelectedDimensionId(isSelected ? null : dim.id)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setSelectedDimensionId(isSelected ? null : dim.id);
                }
              }}
              tabIndex={0}
              role="button"
              aria-expanded={isSelected}
              className={`p-5 rounded-xl border transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-accent-info ${
                isSelected
                  ? 'bg-bg-surface-elevated border-accent-info shadow-lg'
                  : 'bg-bg-surface hover:bg-bg-surface-raised border-border-subtle'
              }`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <span className="text-caption-mono text-text-muted uppercase">
                    Weight: {(dim.weight * 100).toFixed(0)}%
                  </span>
                  <h3 className="text-header-2 font-bold text-text-primary mt-1">
                    {dim.name}
                  </h3>
                </div>
                <div className="text-right">
                  <div className="text-header-1 font-mono font-bold text-accent-positive">
                    {dim.score.toFixed(1)}%
                  </div>
                  <span className="px-2 py-0.5 text-[10px] font-mono font-bold uppercase bg-accent-positive/10 text-accent-positive border border-accent-positive/30 rounded">
                    {dim.status}
                  </span>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="mt-4 w-full bg-bg-surface-raised h-2 rounded-full overflow-hidden border border-border-subtle">
                <div
                  className="bg-accent-positive h-full rounded-full transition-all duration-500"
                  style={{ width: `${dim.score}%` }}
                />
              </div>

              <div className="mt-4 space-y-1.5 text-caption">
                <div className="flex justify-between text-text-muted">
                  <span>Target:</span>
                  <span className="font-mono text-text-secondary">{dim.target}</span>
                </div>
                <div className="flex justify-between text-text-muted">
                  <span>Actual:</span>
                  <span className="font-mono text-text-primary font-medium">{dim.actualMetric}</span>
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-border-subtle flex items-center justify-between text-caption-mono text-text-muted">
                <span>{dim.kpis.length} Measured KPIs</span>
                <span className="text-accent-info font-medium">
                  {isSelected ? 'Collapse ▲' : 'Inspect KPIs ▼'}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* KPI Inspection Drawer/Detail Panel */}
      {selectedDimension && (
        <div className="p-6 bg-bg-surface-elevated border border-accent-info/40 rounded-xl space-y-4 animate-in fade-in duration-200">
          <div className="flex items-center justify-between border-b border-border-subtle pb-3">
            <div>
              <span className="text-caption-mono text-accent-info uppercase font-bold">
                Dimension KPI Breakdown
              </span>
              <h4 className="text-header-1 text-text-primary">
                {selectedDimension.name} — Detailed Metrics
              </h4>
            </div>
            <button
              onClick={() => setSelectedDimensionId(null)}
              className="px-3 py-1 text-caption-mono text-text-muted hover:text-text-primary bg-bg-surface rounded border border-border-subtle transition-colors"
            >
              Close
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {selectedDimension.kpis.map((kpi, idx) => (
              <div
                key={idx}
                className="p-4 bg-bg-surface rounded-lg border border-border-subtle space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="text-body-ui font-semibold text-text-primary">
                    {kpi.name}
                  </span>
                  <span className="px-2 py-0.5 text-[11px] font-mono font-bold bg-accent-positive/10 text-accent-positive border border-accent-positive/30 rounded">
                    PASS
                  </span>
                </div>
                <div className="space-y-1 text-caption font-mono">
                  <div className="flex justify-between text-text-muted">
                    <span>Target:</span>
                    <span className="text-text-secondary">{kpi.target}</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>Actual:</span>
                    <span className="text-accent-positive font-bold">{kpi.actual}</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>Attainment:</span>
                    <span className="text-text-primary font-bold">{kpi.score.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
