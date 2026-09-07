'use client';

import React, { useState } from 'react';
import { MATURITY_TIERS, ROLE_COHORTS } from '@/lib/telemetry/statisticalConfidenceEngine';
import { BehavioralMaturityTier } from '@/types/behavioral-intelligence';

export default function BehavioralMaturityCohortMatrix() {
  const [selectedTier, setSelectedTier] = useState<BehavioralMaturityTier>('OPTIMIZER');

  return (
    <div className="space-y-6" data-testid="behavioral-maturity-cohort-matrix">
      {/* 5 Maturity Tiers */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4">
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <div>
            <h4 className="text-header-1 text-text-primary">
              Behavioral Maturity Tiers (Levels 1 through 5)
            </h4>
            <p className="text-body-ui text-text-secondary mt-0.5">
              Distribution of institutional decision-makers categorized by autonomous adherence and learning loops.
            </p>
          </div>
          <span className="text-caption-mono text-text-muted text-xs">
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
                    {tier.scoreRange}
                  </span>
                  <span className="text-caption-mono font-bold text-accent-positive font-mono">
                    {tier.userPercentage}%
                  </span>
                </div>
                <h5 className="text-body-ui font-semibold text-text-primary mt-1">
                  {tier.label.split('(')[0]}
                </h5>
                <p className="text-caption text-text-muted mt-1 line-clamp-2">
                  {tier.description}
                </p>
                <div className="mt-3 pt-2 border-t border-border-subtle text-[11px] font-mono text-accent-positive">
                  {tier.primaryAction}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Role Cohort Benchmarking Breakdown */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4">
        <h4 className="text-header-2 text-text-primary">
          Role-Based Behavioral Adoption Benchmarking
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {ROLE_COHORTS.map((rc) => (
            <div key={rc.role} className="p-4 bg-bg-surface-raised rounded-lg border border-border-subtle space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-body-ui font-semibold text-text-primary">{rc.label}</span>
                <span className="text-caption-mono font-mono text-text-muted text-xs">{rc.userCount} Users</span>
              </div>
              <div className="flex items-baseline justify-between pt-1">
                <div>
                  <span className="text-[11px] font-mono text-text-muted uppercase block">Adoption</span>
                  <span className="text-display-2 font-mono font-bold text-accent-positive">{rc.adoptionRate}%</span>
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
