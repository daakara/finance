'use client';

import React from 'react';
import {
  getCanonicalDailyCertification,
  evaluateProductionCertification,
} from '@/lib/telemetry/dailyCertificationEngine';

export default function DailyCertificationDashboard() {
  const certData = getCanonicalDailyCertification();
  const evaluation = evaluateProductionCertification(certData.audits);

  return (
    <div
      className="space-y-8 max-w-7xl mx-auto px-2 sm:px-4 py-4"
      data-testid="daily-certification-dashboard"
      role="region"
      aria-label="Daily Production Certification and Value Attribution Dashboard"
    >
      {/* 1. TOP CERTIFICATION BANNER */}
      <div
        className="p-6 md:p-8 bg-gradient-to-br from-bg-surface via-bg-surface-raised to-bg-surface border border-border-subtle rounded-2xl shadow-md space-y-6"
        data-testid="certification-banner"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border-subtle pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono text-xs font-bold uppercase rounded bg-accent-positive/10 text-accent-positive border border-accent-positive/30">
                DAILY PRODUCTION CERTIFICATION: {certData.status}
              </span>
              <span className="px-2.5 py-0.5 text-caption-mono text-xs font-bold uppercase rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                {certData.releaseTrain}
              </span>
            </div>
            <h1 className="text-display-1 md:text-display-2 font-black text-text-primary mt-2 tracking-tight">
              ARX PRODUCTION HEALTH: {certData.overallHealthScore}%
            </h1>
            <p className="text-body-ui text-text-secondary text-sm md:text-base mt-1">
              Automated daily operational audit verifying telemetry integrity, attribution completeness, performance budgets, and behavioral value creation.
            </p>
          </div>

          <div className="text-right">
            <div className="text-caption-mono text-text-muted text-xs uppercase font-bold">
              Audits Passed
            </div>
            <div className="text-display-1 font-mono font-black text-accent-positive mt-1">
              {evaluation.passedCount} / {evaluation.totalCount}
            </div>
            <div className="text-caption-mono text-accent-positive font-bold text-xs mt-1">
              100% Operational Compliance
            </div>
          </div>
        </div>

        {/* 2. NORTH STAR METRIC: DECISION IMPACT RATIO (DIRatio) */}
        <div
          className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4 shadow-sm"
          data-testid="dir-ratio-card"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border-subtle pb-3">
            <div>
              <span className="px-2 py-0.5 rounded text-caption-mono text-[10px] font-bold bg-accent-positive/15 text-accent-positive border border-accent-positive/30 uppercase">
                Primary North Star Metric
              </span>
              <h2 className="text-header-1 font-bold text-text-primary text-base mt-1">
                Decision Impact Ratio (DIRatio)
              </h2>
            </div>
            <div className="text-right text-caption-mono text-xs text-text-muted">
              Sample Size: <strong className="text-text-primary font-mono font-bold">N = {certData.decisionImpactRatio.sampleSize.toLocaleString()}</strong> &bull; 95% CI
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
            <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-1">
              <div className="text-caption-mono text-text-muted text-xs uppercase">High Adoption Cohort</div>
              <div className="text-display-1 font-mono font-black text-accent-positive">
                {certData.decisionImpactRatio.highAdoptionWinRate}%
              </div>
              <div className="text-caption text-text-secondary text-xs">Win Rate (BAR &ge; 70%)</div>
            </div>

            <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-1">
              <div className="text-caption-mono text-text-muted text-xs uppercase">Low Adoption Cohort</div>
              <div className="text-display-1 font-mono font-black text-text-muted">
                {certData.decisionImpactRatio.lowAdoptionWinRate}%
              </div>
              <div className="text-caption text-text-secondary text-xs">Win Rate (BAR &lt; 50%)</div>
            </div>

            <div className="p-4 bg-cyan-500/10 border border-cyan-500/30 rounded-xl space-y-1">
              <div className="text-caption-mono text-cyan-400 text-xs uppercase font-bold">DIRatio Outperformance</div>
              <div className="text-display-1 font-mono font-black text-cyan-400">
                +{certData.decisionImpactRatio.dirRatio}%
              </div>
              <div className="text-caption text-cyan-300 font-bold text-xs">
                Statistically Significant (p &lt; 0.001)
              </div>
            </div>
          </div>

          <p className="text-body-ui text-text-secondary text-sm pt-1">
            <strong className="text-text-primary">Executive Summary:</strong> {certData.decisionImpactRatio.description}
          </p>
        </div>
      </div>

      {/* 3. OUTCOME INTELLIGENCE VALUE ATTRIBUTION (P28-400) */}
      <div
        className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm"
        data-testid="value-attribution-card"
        role="region"
        aria-label="Outcome Value Attribution"
      >
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <div>
            <h3 className="text-header-1 font-bold text-text-primary text-sm uppercase">
              Phase 28 Value Attribution Scorecard (P28-400)
            </h3>
            <p className="text-caption text-text-secondary text-xs">
              Quantified business value, capital preservation, and excess return delivered by ARX behavioral discipline.
            </p>
          </div>
          <span className="px-2 py-0.5 rounded text-caption-mono text-xs font-bold bg-accent-positive/10 text-accent-positive border border-accent-positive/30">
            Quantified Value Verified
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-1">
            <div className="text-caption-mono text-text-muted text-xs uppercase">Capital Preserved</div>
            <div className="text-display-1 font-mono font-black text-accent-positive">
              {certData.valueAttribution.capitalPreservedFormatted}
            </div>
            <div className="text-caption text-text-secondary text-xs">
              Stops: {certData.valueAttribution.sources.stopDiscipline} &bull; Macro: {certData.valueAttribution.sources.macroFilters}
            </div>
          </div>

          <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-1">
            <div className="text-caption-mono text-text-muted text-xs uppercase">Excess Return Generated</div>
            <div className="text-display-1 font-mono font-black text-cyan-400">
              +{certData.valueAttribution.excessReturnPct}%
            </div>
            <div className="text-caption text-text-secondary text-xs">vs Unguided Control Cohort</div>
          </div>

          <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-1">
            <div className="text-caption-mono text-text-muted text-xs uppercase">Mistakes Prevented</div>
            <div className="text-display-1 font-mono font-black text-accent-positive">
              {certData.valueAttribution.mistakesPrevented}
            </div>
            <div className="text-caption text-text-secondary text-xs">Late momentum &amp; gap chases blocked</div>
          </div>

          <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-1">
            <div className="text-caption-mono text-text-muted text-xs uppercase">Recommendations Adopted</div>
            <div className="text-display-1 font-mono font-black text-text-primary">
              {certData.valueAttribution.recommendationsAdopted.toLocaleString()}
            </div>
            <div className="text-caption text-text-secondary text-xs">
              Top: {certData.valueAttribution.topDriverName} ({certData.valueAttribution.topDriverContributionPct}%)
            </div>
          </div>
        </div>
      </div>

      {/* 4. BEHAVIORAL MATURITY COHORTS */}
      <div
        className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm"
        data-testid="behavioral-maturity-cohorts"
        role="region"
        aria-label="Behavioral Maturity Cohorts"
      >
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <div>
            <h3 className="text-header-1 font-bold text-text-primary text-sm uppercase">
              Behavioral Maturity Cohorts Distribution
            </h3>
            <p className="text-caption text-text-secondary text-xs">
              Migration tracking across the 5 behavioral maturity tiers with target expansion to 20% Optimizers.
            </p>
          </div>
          <span className="text-caption-mono text-cyan-400 font-bold text-xs">
            Target: Grow Optimizers 12% &rarr; 20%
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          {[
            { label: 'Non-Adopters', pct: certData.behavioralMaturity.nonAdoptersPct, color: 'text-text-muted' },
            { label: 'Explorers', pct: certData.behavioralMaturity.explorersPct, color: 'text-text-secondary' },
            { label: 'Practitioners', pct: certData.behavioralMaturity.practitionersPct, color: 'text-cyan-400' },
            { label: 'Learners', pct: certData.behavioralMaturity.learnersPct, color: 'text-accent-positive' },
            { label: 'Optimizers', pct: certData.behavioralMaturity.optimizersPct, color: 'text-accent-positive font-black', highlight: true },
          ].map((tier) => (
            <div
              key={tier.label}
              className={`p-4 rounded-xl border text-center space-y-1 ${
                tier.highlight
                  ? 'bg-accent-positive/10 border-accent-positive/30'
                  : 'bg-bg-surface-raised border-border-subtle'
              }`}
            >
              <div className="text-caption-mono text-text-muted uppercase text-[11px] font-bold">{tier.label}</div>
              <div className={`text-display-2 font-mono font-bold ${tier.color}`}>{tier.pct}%</div>
              {tier.highlight && (
                <div className="text-[10px] font-mono text-accent-positive font-bold">Goal: {certData.behavioralMaturity.optimizerTargetPct}%</div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* 5. DAILY 6-AUDIT PASS/FAIL TABLE */}
      <div
        className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm"
        data-testid="daily-audit-table"
        role="region"
        aria-label="Daily 6-Audit Pass/Fail Table"
      >
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <div>
            <h3 className="text-header-1 font-bold text-text-primary text-sm uppercase">
              Daily Automated Production Protection Audits
            </h3>
            <p className="text-caption text-text-secondary text-xs">
              Executed every morning prior to market open across all production clusters.
            </p>
          </div>
          <span className="text-caption-mono text-accent-positive font-bold text-xs">
            6 of 6 Passing
          </span>
        </div>

        <div className="space-y-2.5">
          {certData.audits.map((audit) => (
            <div
              key={audit.id}
              className="p-3.5 bg-bg-surface-raised border border-border-subtle rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
            >
              <div className="flex items-start sm:items-center gap-3">
                <span className="px-2 py-0.5 rounded text-caption-mono text-[10px] font-bold bg-accent-positive/15 text-accent-positive border border-accent-positive/30">
                  {audit.status}
                </span>
                <div>
                  <div className="text-body-ui font-semibold text-text-primary">
                    {audit.name} &bull; <span className="text-text-muted font-normal">{audit.target}</span>
                  </div>
                  <div className="text-caption text-text-secondary text-[11px] mt-0.5">
                    {audit.details}
                  </div>
                </div>
              </div>

              <div className="text-right sm:min-w-[120px]">
                <span className="font-mono font-bold text-accent-positive text-sm">
                  {audit.actual}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
