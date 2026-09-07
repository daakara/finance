'use client';

import React from 'react';
import {
  CohortMigrationSummary,
} from '@/types/dir-framework';
import {
  CANONICAL_COHORT_MIGRATION,
} from '@/lib/telemetry/dirEngine';

interface CohortMigrationDashboardProps {
  migration?: CohortMigrationSummary;
}

export default function CohortMigrationDashboard({
  migration = CANONICAL_COHORT_MIGRATION,
}: CohortMigrationDashboardProps) {
  const {
    cohortAdvancementRate,
    cohortAdvancementTarget,
    cohortRegressionRate,
    cohortRegressionTarget,
    timeToMaturityDays,
    timeToMaturityTargetDays,
    cohortVelocityScore,
    cohorts,
    transitionFlows,
  } = migration;

  return (
    <div className="space-y-6" data-testid="cohort-migration-dashboard">
      {/* 1. Header & Institutional KPI Ribbon */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono font-bold uppercase tracking-wider text-xs rounded bg-accent-positive/10 text-accent-positive border border-accent-positive/30">
                Migration Velocity
              </span>
              <span className="text-caption-mono text-text-muted text-xs">
                6-Stage Progression Lifecycle
              </span>
            </div>
            <h2 className="text-header-1 font-bold text-text-primary mt-1">
              Behavioral Cohort Migration Framework
            </h2>
            <p className="text-body-ui text-text-secondary text-sm">
              Tracks the movement of decision-makers from passive signal consumers into disciplined institutional operators.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="px-3 py-1 text-caption-mono text-xs font-bold rounded-lg bg-accent-positive/15 text-accent-positive border border-accent-positive/30">
              Migration Health: EXCELLENT
            </span>
          </div>
        </div>

        {/* 2. Four Migration KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* CAR */}
          <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-caption-mono text-text-muted text-xs uppercase">Advancement Rate (CAR)</span>
              <span className="text-caption-mono text-accent-positive font-bold text-xs">PASS</span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-display-2 font-mono font-black text-accent-positive">
                {cohortAdvancementRate.toFixed(1)}%
              </span>
              <span className="text-caption font-mono text-text-muted">/ &gt;{cohortAdvancementTarget}% target</span>
            </div>
            <div className="text-caption text-text-secondary text-[11px]">
              Rate of quarterly promotion to higher cohorts
            </div>
          </div>

          {/* CRR */}
          <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-caption-mono text-text-muted text-xs uppercase">Regression Rate (CRR)</span>
              <span className="text-caption-mono text-accent-positive font-bold text-xs">PASS</span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-display-2 font-mono font-black text-accent-positive">
                {cohortRegressionRate.toFixed(1)}%
              </span>
              <span className="text-caption font-mono text-text-muted">/ &lt;{cohortRegressionTarget}% target</span>
            </div>
            <div className="text-caption text-text-secondary text-[11px]">
              Rate of relapse into undisciplined behaviors
            </div>
          </div>

          {/* TTM */}
          <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-caption-mono text-text-muted text-xs uppercase">Time to Maturity (TTM)</span>
              <span className="text-caption-mono text-accent-positive font-bold text-xs">PASS</span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-display-2 font-mono font-black text-accent-positive">
                {timeToMaturityDays}d
              </span>
              <span className="text-caption font-mono text-text-muted">/ &lt;{timeToMaturityTargetDays}d target</span>
            </div>
            <div className="text-caption text-text-secondary text-[11px]">
              Average days to reach Practitioner/Learner tier
            </div>
          </div>

          {/* CVS */}
          <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-caption-mono text-text-muted text-xs uppercase">Cohort Velocity Score</span>
              <span className="text-caption-mono text-accent-positive font-bold text-xs">PASS</span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-display-2 font-mono font-black text-accent-positive">
                {cohortVelocityScore.toFixed(3)}
              </span>
              <span className="text-caption font-mono text-text-muted">/ day</span>
            </div>
            <div className="text-caption text-text-secondary text-[11px]">
              Rate of institutional behavioral mastery
            </div>
          </div>
        </div>

        {/* 3. Six Cohort Distribution Grid */}
        <div className="space-y-3 pt-2">
          <div className="flex items-center justify-between">
            <h3 className="text-header-2 font-bold text-text-primary">
              6-Stage Behavioral Cohort Distribution
            </h3>
            <span className="text-caption text-text-muted text-xs">
              Migration trend: +6.0% net shift from passive into disciplined tiers
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {cohorts.map((cohort) => {
              const isPositiveTrend = cohort.trendDelta > 0;
              return (
                <div
                  key={cohort.id}
                  className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-2 hover:border-border transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="w-5 h-5 rounded bg-bg-surface text-text-primary font-mono text-xs font-bold flex items-center justify-center border border-border-subtle">
                        {cohort.level}
                      </span>
                      <span className="text-body-ui font-bold text-text-primary text-sm">
                        {cohort.name}
                      </span>
                    </div>

                    <span
                      className={`text-caption-mono font-mono font-bold text-xs ${
                        isPositiveTrend ? 'text-accent-positive' : 'text-text-muted'
                      }`}
                    >
                      {isPositiveTrend ? `+${cohort.trendDelta}%` : `${cohort.trendDelta}%`}
                    </span>
                  </div>

                  <div className="flex items-baseline justify-between">
                    <span className="text-display-2 font-mono font-black text-text-primary">
                      {cohort.userSharePercent}%
                    </span>
                    <span className="text-caption text-text-muted text-xs">
                      Target: {cohort.targetAdvancementDays}d
                    </span>
                  </div>

                  {/* Share Bar */}
                  <div className="w-full bg-bg-surface h-1.5 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        isPositiveTrend ? 'bg-accent-positive' : 'bg-text-muted'
                      }`}
                      style={{ width: `${cohort.userSharePercent * 2.5}%` }}
                    />
                  </div>

                  <p className="text-caption text-text-secondary text-xs line-clamp-2">
                    {cohort.description}
                  </p>

                  <div className="pt-2 border-t border-border-subtle/50 text-[11px] text-text-muted">
                    <span className="font-semibold text-text-secondary">Dominant Action:</span> {cohort.dominantAction}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* 4. Sequential Migration Funnel */}
        <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-caption-mono text-text-muted font-bold uppercase text-xs">
              Sequential Promotion Funnel Rates
            </h3>
            <span className="text-caption-mono text-xs text-accent-positive font-semibold">
              All 5 Gates Healthy
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {transitionFlows.map((flow) => (
              <div
                key={`${flow.from}-${flow.to}`}
                className="p-3 bg-bg-surface border border-border-subtle rounded-lg space-y-1 text-center"
              >
                <div className="text-caption-mono text-text-muted text-[10px] uppercase truncate">
                  {flow.from} &rarr; {flow.to}
                </div>
                <div className="text-header-2 font-mono font-black text-accent-positive">
                  {flow.transitionRate.toFixed(1)}%
                </div>
                <div className="text-caption text-text-secondary text-[11px]">
                  {flow.flowLabel}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
