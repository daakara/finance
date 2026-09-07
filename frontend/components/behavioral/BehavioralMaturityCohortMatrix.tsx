'use client';

import React, { useState } from 'react';
import { MATURITY_TIERS, ROLE_COHORTS } from '@/lib/telemetry/statisticalConfidenceEngine';
import { BehavioralMaturityTier } from '@/types/behavioral-intelligence';

export default function BehavioralMaturityCohortMatrix() {
  const [selectedTier, setSelectedTier] = useState<BehavioralMaturityTier>('OPTIMIZER');

  const outcomeComparisons = [
    { cohort: 'Optimizers', quality: 82, benchmarkPct: 100, barColor: 'bg-accent-positive' },
    { cohort: 'Learners', quality: 74, benchmarkPct: 90, barColor: 'bg-accent-info' },
    { cohort: 'Consumers', quality: 61, benchmarkPct: 74, barColor: 'bg-text-muted' },
  ];

  const optimizerTrends = [
    { quarter: 'Q1', share: 8, isTarget: false },
    { quarter: 'Q2', share: 11, isTarget: false },
    { quarter: 'Q3', share: 15, isTarget: true },
  ];

  return (
    <div className="space-y-6" data-testid="behavioral-maturity-cohort-matrix">
      {/* 1. USER EVOLUTION DISTRIBUTION */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-border-subtle pb-3">
          <div>
            <h3 className="text-header-1 font-bold text-text-primary">
              USER EVOLUTION DISTRIBUTION
            </h3>
            <p className="text-body-ui text-text-secondary text-sm">
              Cohort breakdown across the 5 maturity tiers (Consumers &rarr; Optimizers).
            </p>
          </div>
          <span className="text-caption-mono text-accent-positive font-bold text-xs">
            100% User Base Profiled
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {MATURITY_TIERS.map((tier) => {
            const isSelected = selectedTier === tier.tier;
            return (
              <div
                key={tier.tier}
                onClick={() => setSelectedTier(tier.tier)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setSelectedTier(tier.tier);
                  }
                }}
                tabIndex={0}
                role="button"
                aria-expanded={isSelected}
                className={`p-4 rounded-xl border transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-accent-info ${
                  isSelected
                    ? 'bg-bg-surface-elevated border-accent-info shadow-md'
                    : 'bg-bg-surface-raised hover:bg-bg-surface-elevated border-border-subtle'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-caption-mono text-accent-info font-bold text-xs">
                    {tier.tier}
                  </span>
                  <span className="text-display-2 font-mono font-black text-text-primary">
                    {tier.userPercentage}%
                  </span>
                </div>
                <div className="text-caption text-text-secondary text-xs mt-1 line-clamp-2">
                  {tier.description}
                </div>
                <div className="mt-3 pt-2 border-t border-border-subtle text-[11px] font-mono text-accent-positive">
                  {tier.primaryAction}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 2. Trend View & Outcome Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trend View: Optimizer Cohort */}
        <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-border-subtle pb-3">
            <h4 className="text-header-2 font-bold text-text-primary">
              Trend View: Optimizer Cohort Growth
            </h4>
            <span className="text-caption-mono text-accent-positive font-bold text-xs">
              Target: +20% Expansion
            </span>
          </div>

          <p className="text-caption text-text-secondary text-xs">
            Longitudinal growth of users reaching autonomous edge calibration and stress-testing.
          </p>

          <div className="grid grid-cols-3 gap-3 pt-2">
            {optimizerTrends.map((ot) => (
              <div
                key={ot.quarter}
                className={`p-4 rounded-xl border text-center space-y-1 ${
                  ot.isTarget
                    ? 'bg-accent-positive/10 border-accent-positive/40'
                    : 'bg-bg-surface-raised border-border-subtle'
                }`}
              >
                <div className="text-caption-mono text-text-muted text-xs uppercase">{ot.quarter}</div>
                <div className="text-display-2 font-mono font-black text-accent-positive">
                  {ot.share}%
                </div>
                <div className="text-[11px] text-text-secondary">
                  {ot.isTarget ? 'Target Milestone' : 'Observed'}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Outcome Comparison */}
        <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-border-subtle pb-3">
            <h4 className="text-header-2 font-bold text-text-primary">
              Cohort Outcome Comparison
            </h4>
            <span className="text-caption-mono text-text-muted text-xs">
              Decision Quality Score
            </span>
          </div>

          <div className="space-y-4 pt-1">
            {outcomeComparisons.map((oc) => (
              <div key={oc.cohort} className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-body-ui font-bold text-text-primary text-sm">
                    {oc.cohort}
                  </span>
                  <span className="text-header-2 font-mono font-black text-accent-positive">
                    {oc.quality}
                  </span>
                </div>
                <div className="w-full bg-bg-surface-raised h-2 rounded-full overflow-hidden border border-border-subtle/60">
                  <div
                    className={`${oc.barColor} h-full rounded-full`}
                    style={{ width: `${(oc.quality / 100) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 3. Role-Based Behavioral Adoption Benchmarking */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm">
        <h4 className="text-header-2 font-bold text-text-primary">
          Role-Based Behavioral Adoption Benchmarking
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {ROLE_COHORTS.map((rc) => (
            <div key={rc.role} className="p-4 bg-bg-surface-raised rounded-xl border border-border-subtle space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-body-ui font-bold text-text-primary">{rc.label}</span>
                <span className="text-caption-mono font-mono text-text-muted text-xs">{rc.userCount} Users</span>
              </div>
              <div className="flex items-baseline justify-between pt-1">
                <div>
                  <span className="text-[11px] font-mono text-text-muted uppercase block">Adoption</span>
                  <span className="text-display-2 font-mono font-black text-accent-positive">{rc.adoptionRate}%</span>
                </div>
                <div className="text-right">
                  <span className="text-[11px] font-mono text-text-muted uppercase block">Avg Quality</span>
                  <span className="text-header-1 font-mono font-bold text-text-primary">{rc.decisionQuality}</span>
                </div>
              </div>
              <div className="w-full bg-bg-surface h-1.5 rounded-full overflow-hidden border border-border-subtle mt-2">
                <div
                  className="bg-accent-positive h-full rounded-full"
                  style={{ width: `${rc.adoptionRate}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
