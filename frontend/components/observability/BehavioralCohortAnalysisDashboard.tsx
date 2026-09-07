'use client';

import React, { useState } from 'react';
import {
  TIME_COHORTS,
  DECISION_MATURITY_COHORTS,
  BEHAVIORAL_IMPROVEMENT_BREAKDOWN,
  computeLearningMaturityIndex,
} from '@/lib/telemetry/behavioralCohortEngine';

export default function BehavioralCohortAnalysisDashboard() {
  const [selectedCohortId, setSelectedCohortId] = useState<string>('COHORT_B');
  const [selectedMaturityLevel, setSelectedMaturityLevel] = useState<number>(3);

  // Default interactive LMI inputs
  const [lmiInputs, setLmiInputs] = useState({
    outcomeReviews: 78,
    aiCoachingEngagement: 82,
    decisionJournalUsage: 74,
    recommendationAcceptance: 71,
  });

  const lmiResult = computeLearningMaturityIndex(lmiInputs);

  return (
    <div className="space-y-6" data-testid="behavioral-cohort-analysis-dashboard">
      {/* Top Banner: Executive Learning Dashboard */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl shadow-sm space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono font-bold bg-accent-positive/10 text-accent-positive border border-accent-positive/30 rounded">
                Behavioral Intelligence
              </span>
              <span className="text-caption-mono text-text-muted">
                Cohort Improvement Dynamics
              </span>
            </div>
            <h3 className="text-display-2 font-bold text-text-primary mt-1">
              Behavioral Cohort Analysis &amp; Learning Maturity
            </h3>
            <p className="text-body-ui text-text-secondary mt-0.5">
              Empirical tracking verifying that disciplined AI Coach engagement drives higher decision
              quality, lower drift, and reduced repeat mistakes over time.
            </p>
          </div>

          <div className="text-right">
            <div className="text-caption-mono text-text-muted uppercase text-xs">Average Decision Quality</div>
            <div className="text-display-1 font-mono font-extrabold text-accent-positive flex items-center justify-end gap-2">
              {BEHAVIORAL_IMPROVEMENT_BREAKDOWN.avgDecisionQualityScore}
              <span className="text-body-ui text-accent-positive font-bold">
                (&uarr; +{BEHAVIORAL_IMPROVEMENT_BREAKDOWN.decisionQualityDeltaQoQ} pts QoQ)
              </span>
            </div>
            <div className="text-caption-mono text-text-muted text-[11px]">
              Institutional Baseline: 68
            </div>
          </div>
        </div>

        {/* 3 Improvement Segments */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4 border-t border-border-subtle">
          <div className="p-4 bg-bg-surface-raised rounded-xl border border-accent-positive/30 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-caption-mono text-accent-positive uppercase font-bold">Rising Users</span>
              <span className="text-header-1 font-mono font-bold text-accent-positive">
                {BEHAVIORAL_IMPROVEMENT_BREAKDOWN.risingUsersPct}%
              </span>
            </div>
            <p className="text-caption text-text-secondary">
              Decision Quality &uarr;, PAR &uarr;, and Success Rate &uarr; across consecutive cycles.
            </p>
          </div>

          <div className="p-4 bg-bg-surface-raised rounded-xl border border-border-subtle space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-caption-mono text-text-muted uppercase font-bold">Plateau Users</span>
              <span className="text-header-1 font-mono font-bold text-text-secondary">
                {BEHAVIORAL_IMPROVEMENT_BREAKDOWN.plateauUsersPct}%
              </span>
            </div>
            <p className="text-caption text-text-secondary">
              Stable baseline outcomes without adopting AI Coach recommendations.
            </p>
          </div>

          <div className="p-4 bg-bg-surface-raised rounded-xl border border-accent-warning/30 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-caption-mono text-accent-warning uppercase font-bold">Regressing Users</span>
              <span className="text-header-1 font-mono font-bold text-accent-warning">
                {BEHAVIORAL_IMPROVEMENT_BREAKDOWN.regressingUsersPct}%
              </span>
            </div>
            <p className="text-caption text-text-secondary">
              Review activity &darr;, late momentum chasing, and declining decision quality.
            </p>
          </div>
        </div>
      </div>

      {/* Learning Maturity Index (LMI) Section */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border-subtle pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 text-caption-mono font-bold bg-accent-info/10 text-accent-info border border-accent-info/30 rounded">
                Formula
              </span>
              <h4 className="text-header-1 text-text-primary">
                Learning Maturity Index (LMI)
              </h4>
            </div>
            <p className="text-body-ui text-text-secondary mt-1 font-mono text-xs">
              LMI = 0.30(Outcome Reviews) + 0.25(AI Coach) + 0.25(Decision Journal) + 0.20(Rec Acceptance)
            </p>
          </div>

          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-display-2 font-mono font-bold text-accent-positive">
                {lmiResult.compositeScore.toFixed(1)} <span className="text-body-ui font-normal text-text-muted">/ 100</span>
              </div>
              <span className="px-2 py-0.5 text-[11px] font-mono font-bold bg-accent-positive/20 text-accent-positive border border-accent-positive/40 rounded">
                {lmiResult.classification}
              </span>
            </div>
          </div>
        </div>

        {/* Interactive Weight breakdown sliders / progress bars */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 bg-bg-surface-raised rounded-lg border border-border-subtle space-y-2">
            <div className="flex justify-between text-caption-mono text-text-muted">
              <span>Outcome Reviews (30%)</span>
              <span className="text-text-primary font-bold">{lmiInputs.outcomeReviews}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={lmiInputs.outcomeReviews}
              onChange={(e) => setLmiInputs({ ...lmiInputs, outcomeReviews: Number(e.target.value) })}
              className="w-full accent-accent-info"
            />
          </div>

          <div className="p-4 bg-bg-surface-raised rounded-lg border border-border-subtle space-y-2">
            <div className="flex justify-between text-caption-mono text-text-muted">
              <span>AI Coaching (25%)</span>
              <span className="text-text-primary font-bold">{lmiInputs.aiCoachingEngagement}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={lmiInputs.aiCoachingEngagement}
              onChange={(e) => setLmiInputs({ ...lmiInputs, aiCoachingEngagement: Number(e.target.value) })}
              className="w-full accent-accent-info"
            />
          </div>

          <div className="p-4 bg-bg-surface-raised rounded-lg border border-border-subtle space-y-2">
            <div className="flex justify-between text-caption-mono text-text-muted">
              <span>Decision Journal (25%)</span>
              <span className="text-text-primary font-bold">{lmiInputs.decisionJournalUsage}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={lmiInputs.decisionJournalUsage}
              onChange={(e) => setLmiInputs({ ...lmiInputs, decisionJournalUsage: Number(e.target.value) })}
              className="w-full accent-accent-info"
            />
          </div>

          <div className="p-4 bg-bg-surface-raised rounded-lg border border-border-subtle space-y-2">
            <div className="flex justify-between text-caption-mono text-text-muted">
              <span>Rec Acceptance (20%)</span>
              <span className="text-text-primary font-bold">{lmiInputs.recommendationAcceptance}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={lmiInputs.recommendationAcceptance}
              onChange={(e) => setLmiInputs({ ...lmiInputs, recommendationAcceptance: Number(e.target.value) })}
              className="w-full accent-accent-info"
            />
          </div>
        </div>
      </div>

      {/* Decision Maturity Cohorts (1 through 5) */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4">
        <h4 className="text-header-1 text-text-primary">
          Decision Maturity Cohorts (1 to 5)
        </h4>
        <p className="text-body-ui text-text-secondary">
          Segmentation based on user decision actions rather than simple time tenure.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {DECISION_MATURITY_COHORTS.map((cohort) => {
            const isSelected = selectedMaturityLevel === cohort.level;
            return (
              <div
                key={cohort.level}
                onClick={() => setSelectedMaturityLevel(cohort.level)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setSelectedMaturityLevel(cohort.level);
                  }
                }}
                tabIndex={0}
                role="button"
                aria-expanded={isSelected}
                className={`p-4 rounded-xl border transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-accent-info ${
                  isSelected
                    ? 'bg-bg-surface-elevated border-accent-info shadow'
                    : 'bg-bg-surface-raised hover:bg-bg-surface-elevated border-border-subtle'
                }`}
              >
                <div className="flex justify-between items-center">
                  <span className="text-caption-mono text-accent-info font-bold">L{cohort.level}</span>
                  <span className="text-caption-mono font-mono text-text-primary font-bold">{cohort.userPercentage}%</span>
                </div>
                <h5 className="text-body-ui font-semibold text-text-primary mt-1">{cohort.title.split(':')[1]}</h5>
                <p className="text-caption text-text-muted mt-1 line-clamp-2">{cohort.behaviorProfile}</p>
                <div className="mt-3 pt-2 border-t border-border-subtle text-[11px] font-mono text-accent-positive">
                  Adoption: {cohort.recommendationAdoptionRate}%
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Time-based Cohorts (A to D) */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4">
        <h4 className="text-header-2 text-text-primary">
          Time-Based Tenure Cohorts (Cohorts A through D)
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {TIME_COHORTS.map((c) => (
            <div key={c.id} className="p-4 bg-bg-surface-raised rounded-lg border border-border-subtle space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-caption-mono text-accent-info font-bold">{c.tenureRange}</span>
                <span className="px-2 py-0.5 text-[10px] font-mono font-bold bg-accent-positive/10 text-accent-positive border border-accent-positive/30 rounded">
                  {c.churnRisk} RISK
                </span>
              </div>
              <div className="text-body-ui font-semibold text-text-primary">{c.label}</div>
              <div className="space-y-1 text-caption font-mono text-text-muted">
                <div className="flex justify-between">
                  <span>Users:</span>
                  <span className="text-text-primary font-bold">{c.userCount}</span>
                </div>
                <div className="flex justify-between">
                  <span>Avg LMI:</span>
                  <span className="text-accent-positive font-bold">{c.avgLmiScore}</span>
                </div>
                <div className="flex justify-between">
                  <span>Avg Decision Score:</span>
                  <span className="text-text-primary font-bold">{c.avgDecisionScore}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Learning Effectiveness Model Journey Strip */}
      <div className="p-6 bg-gradient-to-r from-bg-surface via-bg-surface-raised to-bg-surface border border-border-subtle rounded-xl space-y-4">
        <h4 className="text-header-2 text-text-primary">
          Learning Effectiveness Model (Proven Decision Loop)
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-center">
          <div className="p-3 bg-bg-surface rounded-lg border border-border-subtle">
            <div className="text-caption-mono text-accent-info uppercase font-bold">1. Coach View</div>
            <div className="text-body-ui font-semibold text-text-primary mt-1">68% Weekly PM Reach</div>
            <div className="text-caption text-text-muted">Contextual Insight Ingested</div>
          </div>
          <div className="p-3 bg-bg-surface rounded-lg border border-border-subtle">
            <div className="text-caption-mono text-accent-info uppercase font-bold">2. Rec Accepted</div>
            <div className="text-body-ui font-semibold text-text-primary mt-1">70.5% BAR Rate</div>
            <div className="text-caption text-text-muted">Direct Rule Compliance</div>
          </div>
          <div className="p-3 bg-bg-surface rounded-lg border border-border-subtle">
            <div className="text-caption-mono text-accent-info uppercase font-bold">3. Outcome Reviewed</div>
            <div className="text-body-ui font-semibold text-text-primary mt-1">100% Attribution</div>
            <div className="text-caption text-text-muted">Attribution Analysis</div>
          </div>
          <div className="p-3 bg-bg-surface rounded-lg border border-accent-positive/40">
            <div className="text-caption-mono text-accent-positive uppercase font-bold">4. Win-Rate Delta</div>
            <div className="text-body-ui font-semibold text-accent-positive mt-1">+8.2% Learning Velocity</div>
            <div className="text-caption text-text-muted">Top Driver: Flow Accumulation (72%)</div>
          </div>
        </div>
      </div>
    </div>
  );
}
