'use client';

import React, { useState } from 'react';
import {
  DIRResult,
  UserDIRInputs,
} from '@/types/dir-framework';
import {
  CANONICAL_DIR_RESULT,
  CANONICAL_USER_DIR_INPUTS,
} from '@/lib/telemetry/dirEngine';

interface DIRHeroCardProps {
  result?: DIRResult;
  inputs?: UserDIRInputs;
}

export default function DIRHeroCard({
  result = CANONICAL_DIR_RESULT,
  inputs = CANONICAL_USER_DIR_INPUTS,
}: DIRHeroCardProps) {
  const [showFormulas, setShowFormulas] = useState(false);
  const { finalDIR, classification, percentileCohort, components, projection, validationRules, isDataSufficient, isLowConfidenceDataset } = result;

  const componentList = [
    components.dqg,
    components.bas,
    components.ras,
    components.drs,
    components.lvi,
  ];

  return (
    <div className="space-y-6" data-testid="dir-hero-card">
      {/* 1. Primary DIR Hero Banner */}
      <div className="p-6 md:p-8 bg-gradient-to-br from-bg-surface-raised via-bg-surface to-bg-surface-raised border border-border-subtle rounded-2xl shadow-sm space-y-6">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="px-3 py-1 text-caption-mono font-bold uppercase tracking-wider text-xs rounded bg-accent-positive/10 text-accent-positive border border-accent-positive/30">
                North Star Metric
              </span>
              <span className="px-2.5 py-0.5 text-caption-mono text-xs rounded bg-bg-surface border border-border-subtle text-text-secondary font-mono">
                Phase 28 Certified
              </span>
              {isLowConfidenceDataset && (
                <span className="px-2.5 py-0.5 text-caption-mono text-xs rounded bg-accent-warning/10 text-accent-warning border border-accent-warning/30 font-semibold">
                  Low Confidence Dataset
                </span>
              )}
            </div>
            <h1 className="text-display-1 md:text-display-2 font-extrabold text-text-primary">
              Decision Improvement Rating (DIR)
            </h1>
            <p className="text-body-ui text-text-secondary max-w-2xl">
              Measures longitudinal decision-maker improvement across quality growth, behavioral adoption, rule adherence, drift resistance, and learning velocity.
            </p>
          </div>

          {/* Large DIR Display Score */}
          <div className="flex items-center gap-6 p-4 md:p-6 bg-bg-surface border border-border-subtle rounded-2xl shadow-inner min-w-[280px] justify-between">
            <div>
              <div className="text-caption-mono text-text-muted uppercase text-xs">Current DIR</div>
              <div className="flex items-baseline gap-2">
                <span className="text-display-2 md:text-display-1 font-mono font-black text-accent-positive">
                  {finalDIR}
                </span>
                <span className="text-header-2 font-mono text-text-muted">/ 100</span>
              </div>
              <div className="flex items-center gap-1.5 mt-1">
                <span className="px-2 py-0.5 text-caption-mono font-bold uppercase text-[11px] rounded bg-accent-positive/20 text-accent-positive border border-accent-positive/40">
                  {classification}
                </span>
                <span className="text-caption-mono text-xs text-text-secondary">
                  Top {100 - percentileCohort}%
                </span>
              </div>
            </div>

            <div className="text-right border-l border-border-subtle pl-4 space-y-1">
              <div className="text-caption-mono text-text-muted text-xs">Cohort Velocity</div>
              <div className="text-body-ui font-mono font-bold text-accent-positive">
                +{finalDIR - 55} pts
              </div>
              <div className="text-caption text-text-muted text-[11px]">
                Faster than {percentileCohort}% of peers
              </div>
            </div>
          </div>
        </div>

        {/* 2. 90-Day Trajectory Projection Ribbon */}
        <div className="p-4 bg-bg-surface/80 border border-border-subtle/80 rounded-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-accent-info/15 text-accent-info border border-accent-info/30 flex items-center justify-center font-bold font-mono text-sm">
              90d
            </div>
            <div>
              <div className="text-caption-mono text-xs text-text-muted uppercase">
                Forward Forecast
              </div>
              <div className="text-body-ui font-medium text-text-primary">
                Projected to reach <span className="font-mono font-bold text-accent-positive">{projection.projectedScore90d}</span> within 90 days ({projection.projectedConfidence}% confidence).
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4 text-sm">
            <div>
              <span className="text-caption text-text-muted">Target Tier: </span>
              <span className="font-semibold text-text-primary">{projection.targetLabel} ({projection.targetScore})</span>
            </div>
            <div className="hidden sm:block h-4 w-px bg-border-subtle" />
            <div>
              <span className="text-caption text-text-muted">Quarterly Delta: </span>
              <span className="font-mono font-bold text-accent-positive">+{projection.projectedQuarterlyGain}.0</span>
            </div>
          </div>
        </div>

        {/* 3. Five Component Breakdown Bars */}
        <div className="space-y-3 pt-2">
          <div className="flex items-center justify-between">
            <h3 className="text-header-2 text-text-primary font-bold">
              Component Weights &amp; Weighted Contributions
            </h3>
            <button
              onClick={() => setShowFormulas(!showFormulas)}
              className="text-caption-mono text-xs text-accent-info hover:underline font-semibold"
            >
              {showFormulas ? 'Hide Formulas' : 'Show Mathematical Formulas'}
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
            {componentList.map((c) => (
              <div
                key={c.key}
                className="p-4 bg-bg-surface border border-border-subtle rounded-xl space-y-2 hover:border-border transition-colors"
              >
                <div className="flex items-center justify-between">
                  <span className="text-caption-mono text-text-muted font-bold text-xs uppercase">
                    {c.key.toUpperCase()} ({(c.weight * 100).toFixed(0)}%)
                  </span>
                  <span className="text-caption-mono font-mono text-xs text-accent-positive font-bold">
                    +{c.weightedContribution.toFixed(1)} pts
                  </span>
                </div>

                <div>
                  <div className="text-body-ui font-bold text-text-primary text-sm truncate" title={c.label}>
                    {c.label}
                  </div>
                  <div className="text-header-2 font-mono font-extrabold text-text-primary mt-0.5">
                    {c.score}{c.key === 'bas' || c.key === 'ras' || c.key === 'drs' ? '%' : ''}
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="w-full bg-bg-surface-raised h-2 rounded-full overflow-hidden border border-border-subtle/50">
                  <div
                    className="bg-accent-positive h-full rounded-full transition-all duration-500"
                    style={{ width: `${Math.min(100, Math.max(0, c.score))}%` }}
                  />
                </div>

                {showFormulas && (
                  <div className="pt-1 text-[11px] font-mono text-text-muted border-t border-border-subtle/50 leading-tight">
                    {c.formula}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* 4. Strongest Driver & Largest Obstacle */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          {/* Driver */}
          <div className="p-4 bg-bg-surface border border-accent-positive/30 rounded-xl space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-caption-mono text-accent-positive font-bold uppercase text-xs">
                Strongest Driver
              </span>
              <span className="text-caption-mono font-mono font-extrabold text-accent-positive text-sm">
                +{projection.strongestDriver.impact.toFixed(1)} pts
              </span>
            </div>
            <div className="text-body-ui font-bold text-text-primary text-sm">
              {projection.strongestDriver.name}
            </div>
            <p className="text-caption text-text-secondary text-xs">
              {projection.strongestDriver.description}
            </p>
          </div>

          {/* Obstacle */}
          <div className="p-4 bg-bg-surface border border-accent-warning/30 rounded-xl space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-caption-mono text-accent-warning font-bold uppercase text-xs">
                Largest Obstacle
              </span>
              <span className="text-caption-mono font-mono font-extrabold text-accent-warning text-sm">
                {projection.largestObstacle.impact.toFixed(1)} pts
              </span>
            </div>
            <div className="text-body-ui font-bold text-text-primary text-sm">
              {projection.largestObstacle.name}
            </div>
            <p className="text-caption text-text-secondary text-xs">
              {projection.largestObstacle.description}
            </p>
          </div>
        </div>

        {/* 5. Validation Rules Checklist */}
        <div className="p-4 bg-bg-surface border border-border-subtle rounded-xl space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-caption-mono text-text-muted font-bold text-xs uppercase">
              Institutional Validation Gates (6 Rules)
            </span>
            <span className="text-caption-mono text-accent-positive font-bold text-xs">
              {validationRules.filter((r) => r.passed).length} / {validationRules.length} Passing
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {validationRules.map((rule) => (
              <div
                key={rule.id}
                className="p-2.5 bg-bg-surface-raised/60 border border-border-subtle/70 rounded-lg flex items-start gap-2 text-xs"
              >
                <span
                  className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0 mt-0.5 ${
                    rule.passed
                      ? 'bg-accent-positive/20 text-accent-positive'
                      : 'bg-accent-warning/20 text-accent-warning'
                  }`}
                >
                  {rule.passed ? '✓' : '!'}
                </span>
                <div className="min-w-0">
                  <div className="font-semibold text-text-primary truncate">{rule.name}</div>
                  <div className="text-text-muted text-[11px]">
                    Threshold: {rule.threshold} ({rule.actualValue})
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
