'use client';

import React, { useState } from 'react';
import { computeDIR, computeBehavioralDIR } from '@/lib/telemetry/decisionIntelligenceEngine';
import { evaluateLearningVelocity, computeQoQProgression, computeAnnualGrowth } from '@/lib/telemetry/learningVelocityEngine';
import { classifyUserCohort, getCohortDistribution } from '@/lib/telemetry/behavioralCohortEngine';
import { evaluateExecutiveBenchmarks } from '@/lib/telemetry/executiveBenchmarkEngine';

export interface BehavioralIntelligenceDashboardProps {
  userId?: string;
}

export default function BehavioralIntelligenceDashboard({
  userId = 'usr_david_001',
}: BehavioralIntelligenceDashboardProps) {
  const [activeTab, setActiveTab] = useState<'OVERVIEW' | 'BENCHMARKS' | 'COHORTS'>('OVERVIEW');

  // 1. DIR Calculation
  const dirResult = computeDIR({
    decisionQualityScore: 74,
    outcomeScore: 72,
    learningScore: 84,
    governanceScore: 87,
    decisionCount: 42,
    resolvedOutcomeCount: 30,
    evidenceOpenRate: 44,
  });

  const behavioralDir = computeBehavioralDIR({
    decisionQuality: 74,
    behaviorAdoption: 70.5,
    ruleAdherence: 87.0,
    repeatMistakeReduction: 43.0,
    decisionDrift: 21.0,
  });

  // 2. Learning Velocity
  const velocityResult = evaluateLearningVelocity({
    dirTrend: 3.2,
    outcomeReviews: 78,
    learningCoachUsage: 82,
    decisionJournalActivity: 74,
    recommendationAdoption: 71,
    historicalGrowthPeriods: [2, 4, 6],
  });

  const qoq = computeQoQProgression(74, 68);
  const annual = computeAnnualGrowth(62, 74);

  // 3. Cohorts
  const cohortAssignment = classifyUserCohort({
    daysActive: 142,
    dirScore: dirResult.dirScore,
    ruleAdherence: 87.0,
    driftScore: 21.0,
    behaviorAdoptionRate: 70.5,
    evidenceUsageRate: 54.0,
    weeklySessionsCount: 8.4,
  });

  const distribution = getCohortDistribution();

  // 4. Benchmarks
  const benchmarkResult = evaluateExecutiveBenchmarks({
    currentScore: dirResult.dirScore,
  });

  return (
    <div
      className="space-y-6"
      data-testid="behavioral-intelligence-dashboard"
      role="region"
      aria-label="Behavioral Intelligence Dashboard"
    >
      {/* Institutional Top Header */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-3 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono font-bold uppercase bg-accent-positive/10 text-accent-positive border border-accent-positive/30 rounded">
                Phase 28 Milestone 1
              </span>
              <span className="text-caption-mono text-text-muted text-xs">
                Behavioral Intelligence Foundations (Measurement Layer)
              </span>
            </div>
            <h2 className="text-display-1 font-bold text-text-primary mt-1">
              BEHAVIORAL INTELLIGENCE FOUNDATIONS
            </h2>
            <p className="text-body-ui text-text-secondary mt-0.5">
              Empirical measurement of decision quality, learning velocity, and behavioral maturation with zero recommendation-generation logic.
            </p>
          </div>

          <div className="flex items-center gap-2 p-1.5 bg-bg-surface-raised rounded-xl border border-border-subtle self-start md:self-auto">
            <button
              onClick={() => setActiveTab('OVERVIEW')}
              role="button"
              aria-pressed={activeTab === 'OVERVIEW'}
              className={`px-3 py-1.5 text-caption-mono text-xs font-bold rounded-lg transition-colors ${
                activeTab === 'OVERVIEW'
                  ? 'bg-accent-positive/20 text-accent-positive border border-accent-positive/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab('BENCHMARKS')}
              role="button"
              aria-pressed={activeTab === 'BENCHMARKS'}
              className={`px-3 py-1.5 text-caption-mono text-xs font-bold rounded-lg transition-colors ${
                activeTab === 'BENCHMARKS'
                  ? 'bg-accent-positive/20 text-accent-positive border border-accent-positive/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              Benchmarks
            </button>
            <button
              onClick={() => setActiveTab('COHORTS')}
              role="button"
              aria-pressed={activeTab === 'COHORTS'}
              className={`px-3 py-1.5 text-caption-mono text-xs font-bold rounded-lg transition-colors ${
                activeTab === 'COHORTS'
                  ? 'bg-accent-positive/20 text-accent-positive border border-accent-positive/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              Cohorts
            </button>
          </div>
        </div>

        {/* 5 Invariant Certification Badges */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2 pt-3 border-t border-border-subtle text-[11px] font-mono">
          <div className="p-2 bg-bg-surface-raised rounded border border-border-subtle">
            <span className="text-text-muted block text-[10px]">INV-B1: Attribution</span>
            <span className="text-accent-positive font-bold">100% Attributable</span>
          </div>
          <div className="p-2 bg-bg-surface-raised rounded border border-border-subtle">
            <span className="text-text-muted block text-[10px]">INV-B2: DIR Determinism</span>
            <span className="text-accent-positive font-bold">Verified Pure Math</span>
          </div>
          <div className="p-2 bg-bg-surface-raised rounded border border-border-subtle">
            <span className="text-text-muted block text-[10px]">INV-B3: Transparency</span>
            <span className="text-accent-positive font-bold">95% CI Attached</span>
          </div>
          <div className="p-2 bg-bg-surface-raised rounded border border-border-subtle">
            <span className="text-text-muted block text-[10px]">INV-B4: Benchmarks</span>
            <span className="text-accent-positive font-bold">Frozen Immutable</span>
          </div>
          <div className="p-2 bg-bg-surface-raised rounded border border-border-subtle">
            <span className="text-text-muted block text-[10px]">INV-B5: Traceability</span>
            <span className="text-accent-positive font-bold">4-Stage Linked</span>
          </div>
        </div>
      </div>

      {/* SECTION 1: DIR HERO CARD */}
      <div
        className="p-6 bg-bg-surface border-2 border-accent-positive/40 rounded-2xl space-y-6 shadow-sm"
        data-testid="dir-hero"
      >
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border-subtle pb-4">
          <div>
            <span className="text-caption-mono text-accent-positive uppercase font-bold text-xs">
              Deliverable 1 &bull; Decision Intelligence Rate (DIR)
            </span>
            <h3 className="text-header-1 font-black text-text-primary mt-0.5">
              DECISION IMPROVEMENT RATING: <span className="text-accent-positive font-mono">{dirResult.dirScore}</span> / 100
            </h3>
            <p className="text-body-ui text-text-secondary mt-0.5">
              Composite score: <code className="text-accent-info font-mono text-xs">0.40(DQS) + 0.25(Outcome) + 0.20(Learning) + 0.15(Governance)</code>
            </p>
          </div>

          <div className="p-4 bg-bg-surface-raised rounded-xl border border-border-subtle text-right">
            <div className="text-caption-mono text-text-muted uppercase text-xs">95% Confidence Band</div>
            <div className="text-header-2 font-mono font-bold text-text-primary mt-0.5">
              [{dirResult.confidenceBand.lower} &mdash; {dirResult.confidenceBand.upper}]
            </div>
            <div className="text-caption-mono text-accent-positive text-xs font-semibold">
              Confidence: {dirResult.confidenceScore}% (N = {dirResult.sampleSize})
            </div>
          </div>
        </div>

        {/* 4 DIR Components */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="p-3 bg-bg-surface-raised rounded-xl border border-border-subtle">
            <div className="text-[11px] font-mono text-text-muted uppercase">1. Decision Quality (40%)</div>
            <div className="text-header-2 font-mono font-bold text-accent-positive mt-0.5">
              {dirResult.components?.dqsContribution.toFixed(1)} pts
            </div>
            <div className="text-[10px] font-mono text-text-muted">DQS = 74 &bull; Raw Contrib</div>
          </div>
          <div className="p-3 bg-bg-surface-raised rounded-xl border border-border-subtle">
            <div className="text-[11px] font-mono text-text-muted uppercase">2. Outcome Score (25%)</div>
            <div className="text-header-2 font-mono font-bold text-accent-positive mt-0.5">
              {dirResult.components?.outcomeContribution.toFixed(1)} pts
            </div>
            <div className="text-[10px] font-mono text-text-muted">Outcome = 72 &bull; 30 Resolved</div>
          </div>
          <div className="p-3 bg-bg-surface-raised rounded-xl border border-border-subtle">
            <div className="text-[11px] font-mono text-text-muted uppercase">3. Learning Score (20%)</div>
            <div className="text-header-2 font-mono font-bold text-accent-positive mt-0.5">
              {dirResult.components?.learningContribution.toFixed(1)} pts
            </div>
            <div className="text-[10px] font-mono text-text-muted">Learning = 84 &bull; High Velocity</div>
          </div>
          <div className="p-3 bg-bg-surface-raised rounded-xl border border-border-subtle">
            <div className="text-[11px] font-mono text-text-muted uppercase">4. Governance (15%)</div>
            <div className="text-header-2 font-mono font-bold text-accent-positive mt-0.5">
              {dirResult.components?.governanceContribution.toFixed(1)} pts
            </div>
            <div className="text-[10px] font-mono text-text-muted">Gov = 87 &bull; Audit Chain OK</div>
          </div>
        </div>

        {/* Behavioral Composite DIR Comparison */}
        <div className="p-4 bg-bg-surface-raised rounded-xl border border-border-subtle flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-sm">
          <div>
            <div className="text-caption-mono text-text-muted text-xs uppercase">Behavioral DIR Formulation</div>
            <div className="text-body-ui font-semibold text-text-primary mt-0.5">
              0.35(DQ) + 0.25(BA) + 0.20(RA) + 0.10(RM) + 0.10(100 - Drift) = <strong className="font-mono text-accent-positive">{behavioralDir.overallScore}</strong>
            </div>
          </div>
          <span className="px-3 py-1 text-caption-mono font-bold text-xs rounded bg-accent-positive/15 text-accent-positive border border-accent-positive/30">
            Classification: Proficient (70-79)
          </span>
        </div>
      </div>

      {/* SECTION 2: LEARNING VELOCITY */}
      <div
        className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm"
        data-testid="learning-velocity-section"
      >
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <div>
            <span className="text-caption-mono text-accent-info uppercase font-bold text-xs">
              Deliverable 2 &bull; Learning Velocity Engine
            </span>
            <h3 className="text-header-2 font-bold text-text-primary mt-0.5">
              LEARNING VELOCITY: <span className="font-mono text-accent-positive">{velocityResult.score} / 100</span>
            </h3>
          </div>
          <span className="px-2.5 py-1 text-caption-mono font-bold text-xs rounded bg-accent-positive/20 text-accent-positive border border-accent-positive/40">
            {velocityResult.direction} &bull; Acceleration: {velocityResult.acceleration}x
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
          <div className="p-4 bg-bg-surface-raised rounded-xl border border-border-subtle space-y-1">
            <div className="text-caption-mono text-text-muted text-xs uppercase">QoQ Progression</div>
            <div className="text-header-2 font-mono font-bold text-accent-positive">
              ▲ +{qoq.delta} pts
            </div>
            <div className="text-caption text-text-secondary text-xs">
              +{qoq.percentageGain}% Quarter-over-Quarter
            </div>
          </div>

          <div className="p-4 bg-bg-surface-raised rounded-xl border border-border-subtle space-y-1">
            <div className="text-caption-mono text-text-muted text-xs uppercase">Annual Growth</div>
            <div className="text-header-2 font-mono font-bold text-accent-positive">
              ▲ +{annual.annualDelta} pts
            </div>
            <div className="text-caption text-text-secondary text-xs">
              +{annual.compoundedAnnualRate}% Compounded Annual Gain
            </div>
          </div>

          <div className="p-4 bg-bg-surface-raised rounded-xl border border-border-subtle space-y-1">
            <div className="text-caption-mono text-text-muted text-xs uppercase">Plateau Detection</div>
            <div className="text-header-2 font-mono font-bold text-accent-positive">
              NO PLATEAU
            </div>
            <div className="text-caption text-text-secondary text-xs">
              Velocity sustained above 1.0 threshold
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 3: BEHAVIORAL MATURITY & COHORTS */}
      <div
        className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm"
        data-testid="behavioral-maturity-section"
      >
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <div>
            <span className="text-caption-mono text-accent-info uppercase font-bold text-xs">
              Deliverable 3 &bull; Behavioral Cohort Framework
            </span>
            <h3 className="text-header-2 font-bold text-text-primary mt-0.5">
              COHORT ASSIGNMENT: <span className="text-accent-info">{cohortAssignment.primaryCohort}</span> ({cohortAssignment.tenureCohort})
            </h3>
          </div>
          <span className="text-caption-mono text-text-muted text-xs">
            Confidence: {cohortAssignment.confidence}%
          </span>
        </div>

        <div className="space-y-3">
          <div className="text-caption-mono text-text-muted text-xs uppercase font-bold">
            Enterprise Cohort Distribution (Total: {distribution.totalPercentage}%)
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-center">
            <div className="p-3 bg-bg-surface-raised rounded-xl border border-border-subtle">
              <div className="text-header-2 font-mono font-bold text-text-primary">{distribution.consumers}%</div>
              <div className="text-[11px] font-mono text-text-muted uppercase mt-0.5">1. Consumer</div>
            </div>
            <div className="p-3 bg-bg-surface-raised rounded-xl border border-border-subtle">
              <div className="text-header-2 font-mono font-bold text-text-primary">{distribution.investigators}%</div>
              <div className="text-[11px] font-mono text-text-muted uppercase mt-0.5">2. Investigator</div>
            </div>
            <div className="p-3 bg-bg-surface-raised rounded-xl border-2 border-accent-info/50">
              <div className="text-header-2 font-mono font-bold text-accent-info">{distribution.practitioners}%</div>
              <div className="text-[11px] font-mono text-accent-info uppercase mt-0.5 font-bold">3. Practitioner (You)</div>
            </div>
            <div className="p-3 bg-bg-surface-raised rounded-xl border border-border-subtle">
              <div className="text-header-2 font-mono font-bold text-text-primary">{distribution.learners}%</div>
              <div className="text-[11px] font-mono text-text-muted uppercase mt-0.5">4. Learner</div>
            </div>
            <div className="p-3 bg-bg-surface-raised rounded-xl border border-border-subtle">
              <div className="text-header-2 font-mono font-bold text-accent-positive">{distribution.optimizers}%</div>
              <div className="text-[11px] font-mono text-accent-positive uppercase mt-0.5 font-bold">5. Optimizer</div>
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 4: IMPROVEMENT OPPORTUNITIES (Measurement Observation) */}
      <div
        className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm"
        data-testid="improvement-opportunities-section"
      >
        <div className="border-b border-border-subtle pb-3">
          <span className="text-caption-mono text-accent-warning uppercase font-bold text-xs">
            Deliverable 5 &bull; Improvement Opportunities & Measurement Attribution
          </span>
          <h3 className="text-header-2 font-bold text-text-primary mt-0.5">
            OBSERVED BEHAVIORAL DRIVERS
          </h3>
          <p className="text-body-ui text-text-secondary text-xs mt-0.5">
            Measurement attribution isolating behavioral strengths and friction areas. No automated trading advice generated.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Top Strength */}
          <div className="p-4 bg-bg-surface-raised rounded-xl border border-accent-positive/30 space-y-2">
            <div className="flex items-center justify-between">
              <span className="px-2 py-0.5 text-caption-mono font-bold text-[10px] rounded bg-accent-positive/15 text-accent-positive border border-accent-positive/30 uppercase">
                Top Strength
              </span>
              <span className="text-caption-mono text-accent-positive font-bold text-xs font-mono">
                +6.2 pts
              </span>
            </div>
            <div className="text-body-ui font-bold text-text-primary">
              Institutional Flow Discipline
            </div>
            <p className="text-caption text-text-secondary text-xs">
              Entering breakouts only when volume confirmation exceeds 2.0 sigma generated a 72% win rate across 38 evaluated decisions.
            </p>
          </div>

          {/* Top Improvement Area */}
          <div className="p-4 bg-bg-surface-raised rounded-xl border border-accent-warning/40 space-y-2">
            <div className="flex items-center justify-between">
              <span className="px-2 py-0.5 text-caption-mono font-bold text-[10px] rounded bg-accent-warning/15 text-accent-warning border border-accent-warning/30 uppercase">
                Top Improvement Area
              </span>
              <span className="text-caption-mono text-accent-risk font-bold text-xs font-mono">
                -28% drag
              </span>
            </div>
            <div className="text-body-ui font-bold text-text-primary">
              Late-Day Momentum Chasing
            </div>
            <p className="text-caption text-text-secondary text-xs">
              Orders placed after 2:30 PM EST represent 28% of total losses and account for the primary driver of decision drift.
            </p>
          </div>
        </div>
      </div>

      {/* SECTION 5: BENCHMARK COMPARISON */}
      <div
        className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm"
        data-testid="benchmark-comparison-section"
      >
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <div>
            <span className="text-caption-mono text-accent-info uppercase font-bold text-xs">
              Deliverable 4 &bull; Executive Behavioral Benchmarking
            </span>
            <h3 className="text-header-2 font-bold text-text-primary mt-0.5">
              INSTITUTIONAL COMPARISON LAYERS (Top {benchmarkResult.percentileRank}%)
            </h3>
          </div>
          <span className="text-caption-mono text-accent-positive font-bold text-xs">
            Target: 80 within {benchmarkResult.expectedProgression.targetHorizonMonths} months
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="p-4 bg-bg-surface-raised rounded-xl border border-border-subtle space-y-1">
            <div className="text-caption-mono text-text-muted text-xs uppercase">1. Personal Historical</div>
            <div className="text-header-1 font-mono font-bold text-text-primary mt-0.5">
              {benchmarkResult.benchmarks.personalHistorical}
            </div>
            <div className="text-[11px] font-mono text-accent-positive font-bold">
              ▲ +{benchmarkResult.layerDeltas.vsPersonalHistorical} points
            </div>
          </div>

          <div className="p-4 bg-bg-surface-raised rounded-xl border border-border-subtle space-y-1">
            <div className="text-caption-mono text-text-muted text-xs uppercase">2. Team Average</div>
            <div className="text-header-1 font-mono font-bold text-text-primary mt-0.5">
              {benchmarkResult.benchmarks.teamAverage}
            </div>
            <div className="text-[11px] font-mono text-accent-positive font-bold">
              ▲ +{benchmarkResult.layerDeltas.vsTeamAverage} points ahead
            </div>
          </div>

          <div className="p-4 bg-bg-surface-raised rounded-xl border border-border-subtle space-y-1">
            <div className="text-caption-mono text-text-muted text-xs uppercase">3. Institution Average</div>
            <div className="text-header-1 font-mono font-bold text-text-primary mt-0.5">
              {benchmarkResult.benchmarks.institutionAverage}
            </div>
            <div className="text-[11px] font-mono text-accent-positive font-bold">
              ▲ +{benchmarkResult.layerDeltas.vsInstitutionAverage} points ahead
            </div>
          </div>

          <div className="p-4 bg-bg-surface-raised rounded-xl border border-border-subtle space-y-1">
            <div className="text-caption-mono text-text-muted text-xs uppercase">4. Elite Quartile</div>
            <div className="text-header-1 font-mono font-bold text-text-primary mt-0.5">
              {benchmarkResult.benchmarks.eliteQuartile}
            </div>
            <div className="text-[11px] font-mono text-text-muted font-bold">
              {benchmarkResult.layerDeltas.vsEliteQuartile} points to elite
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
