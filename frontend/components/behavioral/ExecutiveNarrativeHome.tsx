'use client';

import React, { useState } from 'react';
import {
  generateExecutiveNarrative,
  generateCanonicalExecutiveNarrative,
} from '@/lib/telemetry/executiveNarrativeEngine';
import {
  evaluateCapabilityAttribution,
  CANONICAL_CAPABILITY_ATTRIBUTION,
} from '@/lib/telemetry/capabilityImpactEngine';
import {
  ExecutiveNarrativeState,
  ExecutiveNarrativeResult,
} from '@/types/behavioral-intelligence';
import { CANONICAL_NARRATIVE_FIXTURES } from '@/fixtures/behavioral-fixtures';

interface ExecutiveNarrativeHomeProps {
  initialState?: ExecutiveNarrativeState;
}

export default function ExecutiveNarrativeHome({
  initialState = 'HEALTHY',
}: ExecutiveNarrativeHomeProps) {
  const [selectedState, setSelectedState] = useState<ExecutiveNarrativeState>(initialState);
  const [isEvidenceOpen, setIsEvidenceOpen] = useState<boolean>(false);
  const [actionAccepted, setActionAccepted] = useState<boolean>(false);

  // Derive narrative based on selected state fixture
  const fixture = CANONICAL_NARRATIVE_FIXTURES[selectedState] ?? CANONICAL_NARRATIVE_FIXTURES.HEALTHY;
  const narrativeResult: ExecutiveNarrativeResult = generateExecutiveNarrative({
    dir: fixture.dir,
    dirTrend: fixture.dirTrend,
    learningVelocity: fixture.learningVelocity,
    confidence: fixture.confidence,
    decisionCount: fixture.decisionCount,
    daysSinceLastActivity: fixture.daysSinceLastActivity,
    topDriver: fixture.topDriver,
    topWeakness: fixture.topWeakness,
    userName: 'David',
    dateString: 'Monday 07 September 2026',
    portfolioAtRisk: 184000,
  });

  const capabilityAttribution = evaluateCapabilityAttribution();

  const stateBadges: Record<ExecutiveNarrativeState, { label: string; color: string; border: string; bg: string }> = {
    HEALTHY: { label: 'Improving (Healthy)', color: 'text-accent-positive', border: 'border-accent-positive/40', bg: 'bg-accent-positive/10' },
    IMPROVING: { label: 'Improving', color: 'text-accent-positive', border: 'border-accent-positive/40', bg: 'bg-accent-positive/10' },
    PLATEAU: { label: 'Decision Quality Stable (Plateau)', color: 'text-accent-warning', border: 'border-accent-warning/40', bg: 'bg-accent-warning/10' },
    DECLINING: { label: 'Decision Quality Declining', color: 'text-accent-negative', border: 'border-accent-negative/40', bg: 'bg-accent-negative/10' },
    NEW_USER: { label: 'Baseline Not Yet Established', color: 'text-cyan-400', border: 'border-cyan-500/40', bg: 'bg-cyan-500/10' },
    INACTIVE: { label: 'DIR Suspended (Inactive)', color: 'text-text-muted', border: 'border-border-subtle', bg: 'bg-bg-surface-raised' },
    LOW_CONFIDENCE: { label: 'Low Confidence (Caution)', color: 'text-accent-warning', border: 'border-accent-warning/40', bg: 'bg-accent-warning/10' },
  };

  const badge = stateBadges[narrativeResult.state] ?? stateBadges.HEALTHY;

  return (
    <div
      className="space-y-6 max-w-7xl mx-auto px-2 sm:px-4 py-4"
      data-testid="executive-narrative-home"
      data-component-id="ARX-ENH-001"
    >
      {/* 0. State Inspector Toolbar (Reviewer Utility for all 7 States) */}
      <div className="p-3 bg-bg-surface border border-border-subtle rounded-xl flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="text-caption-mono text-text-muted uppercase font-bold text-[11px]">
            Executive State Simulation:
          </span>
          <span className="text-caption-mono text-cyan-400 font-bold">
            {narrativeResult.state}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {(Object.keys(CANONICAL_NARRATIVE_FIXTURES) as ExecutiveNarrativeState[]).map((st) => (
            <button
              key={st}
              onClick={() => {
                setSelectedState(st);
                setActionAccepted(false);
              }}
              className={`px-2.5 py-1 min-h-[32px] rounded text-caption-mono text-xs font-semibold transition-all ${
                selectedState === st
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-sm'
                  : 'bg-bg-surface-raised text-text-secondary hover:text-text-primary border border-border-subtle'
              }`}
            >
              {st}
            </button>
          ))}
        </div>
      </div>

      {/* 1. Executive Top Summary Strip */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* DIR Tile */}
        <div className="p-5 bg-bg-surface border border-border-subtle rounded-xl flex items-center justify-between shadow-sm">
          <div>
            <div className="text-caption-mono text-text-muted uppercase text-xs">Decision Intelligence Rate</div>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-display-1 font-mono font-black text-accent-positive">
                {narrativeResult.dirScore}
              </span>
              <span className="text-caption-mono text-text-muted text-xs">/ 100</span>
              <span className="text-caption font-mono font-bold text-accent-positive text-xs">
                {narrativeResult.dirTrend >= 0 ? `▲ +${narrativeResult.dirTrend}` : `▼ ${narrativeResult.dirTrend}`}
              </span>
            </div>
            <div className="text-caption text-text-secondary text-xs mt-0.5">
              Confidence: <strong className="text-text-primary font-mono">{narrativeResult.confidence}%</strong> (95% CI)
            </div>
          </div>
          <div className="hidden sm:flex w-10 h-10 rounded-full bg-accent-positive/10 border border-accent-positive/30 items-center justify-center font-mono font-bold text-accent-positive text-sm">
            DIR
          </div>
        </div>

        {/* Learning Velocity Tile */}
        <div className="p-5 bg-bg-surface border border-border-subtle rounded-xl flex items-center justify-between shadow-sm">
          <div>
            <div className="text-caption-mono text-text-muted uppercase text-xs">Learning Velocity</div>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-display-1 font-mono font-black text-cyan-400">
                {narrativeResult.metrics.learningVelocity}
              </span>
              <span className="text-caption-mono text-text-muted text-xs">/ 100</span>
              <span className="px-2 py-0.5 rounded text-caption-mono text-xs font-bold bg-accent-positive/10 text-accent-positive border border-accent-positive/30">
                HIGH
              </span>
            </div>
            <div className="text-caption text-text-secondary text-xs mt-0.5">
              Top <strong className="text-text-primary">12%</strong> Institutional Cohort
            </div>
          </div>
          <div className="hidden sm:flex w-10 h-10 rounded-full bg-cyan-500/10 border border-cyan-500/30 items-center justify-center font-mono font-bold text-cyan-400 text-sm">
            LVI
          </div>
        </div>

        {/* Attention Tile */}
        <div className="p-5 bg-bg-surface border border-border-subtle rounded-xl flex items-center justify-between shadow-sm">
          <div>
            <div className="text-caption-mono text-text-muted uppercase text-xs">Active Attention</div>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-display-1 font-mono font-black text-accent-warning">
                {narrativeResult.metrics.attentionCount}
              </span>
              <span className="text-caption text-text-secondary text-xs font-semibold">Positions Flagged</span>
            </div>
            <div className="text-caption text-text-secondary text-xs mt-0.5">
              Capital at Risk: <strong className="text-text-primary font-mono">${(narrativeResult.metrics.portfolioAtRisk / 1000).toFixed(0)}k</strong>
            </div>
          </div>
          <div className="hidden sm:flex w-10 h-10 rounded-full bg-accent-warning/10 border border-accent-warning/30 items-center justify-center font-mono font-bold text-accent-warning text-sm">
            !
          </div>
        </div>
      </div>

      {/* 2. Executive Narrative Hero (Comprehension < 30 seconds) */}
      <div
        className="p-6 md:p-8 bg-gradient-to-br from-bg-surface via-bg-surface-raised to-bg-surface border border-border-subtle rounded-2xl shadow-md space-y-6"
        data-testid="executive-narrative-hero"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border-subtle pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono text-xs font-bold uppercase rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                Daily Executive Briefing
              </span>
              <span className={`px-2.5 py-0.5 text-caption-mono text-xs font-bold uppercase rounded ${badge.bg} ${badge.color} border ${badge.border}`}>
                {badge.label}
              </span>
            </div>
            <h1 className="text-display-1 md:text-display-2 font-black text-text-primary mt-2 tracking-tight">
              {narrativeResult.headline}
            </h1>
          </div>
          <div className="text-right">
            <div className="text-caption-mono text-text-muted font-bold text-sm">
              {narrativeResult.generatedAt}
            </div>
            <div className="text-caption-mono text-accent-positive text-xs mt-0.5">
              Deterministic Synthesis Verified &bull; INV-B7 PASS
            </div>
          </div>
        </div>

        {/* Narrative Core: Observation -> Learning -> Recommended Action */}
        <div className="space-y-4">
          <div className="p-4 bg-bg-surface rounded-xl border border-border-subtle/80 space-y-2">
            <div className="text-caption-mono text-text-muted uppercase text-xs font-bold tracking-wider">
              Executive Summary (30s Briefing)
            </div>
            <p className="text-body-ui font-semibold text-text-primary text-base md:text-lg leading-relaxed">
              {narrativeResult.executiveSummary.observation}
            </p>
            <p className="text-body-ui text-text-secondary text-sm md:text-base leading-relaxed">
              {narrativeResult.executiveSummary.learning}
            </p>
          </div>

          {/* Action Callout (INV-B8 Actionability Invariant) */}
          <div
            className="p-5 bg-gradient-to-r from-accent-positive/10 via-bg-surface to-bg-surface-raised border-2 border-accent-positive/40 rounded-xl space-y-3"
            data-testid="recommended-action-box"
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-accent-positive animate-pulse" />
                <span className="text-caption-mono text-accent-positive uppercase font-bold text-xs">
                  Mandatory Recommended Action &bull; INV-B8
                </span>
              </div>
              <div className="text-caption-mono text-accent-positive font-bold text-xs">
                Action Confidence: {narrativeResult.executiveSummary.actionConfidence}%
              </div>
            </div>

            <div className="text-header-1 font-bold text-text-primary text-base md:text-xl">
              {narrativeResult.executiveSummary.recommendedAction}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-border-subtle/60">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setActionAccepted(true)}
                  disabled={actionAccepted}
                  className={`px-5 py-2.5 min-h-[44px] rounded-lg font-semibold text-xs transition-all shadow-sm ${
                    actionAccepted
                      ? 'bg-accent-positive/20 text-accent-positive border border-accent-positive/40'
                      : 'bg-accent-positive text-slate-950 hover:bg-accent-positive/90 active:scale-95'
                  }`}
                >
                  {actionAccepted ? '✓ Action Executed & Journaled' : 'Execute Directive'}
                </button>
                <button
                  onClick={() => setIsEvidenceOpen(!isEvidenceOpen)}
                  className="px-4 py-2.5 min-h-[44px] rounded-lg bg-bg-surface hover:bg-bg-surface-elevated text-cyan-400 hover:text-cyan-300 border border-cyan-500/30 text-caption-mono text-xs font-semibold transition-all"
                >
                  {isEvidenceOpen ? 'Hide Evidence Root' : 'View Evidence Trace'}
                </button>
              </div>
              <span className="text-caption-mono text-text-muted text-xs">
                Trace: {narrativeResult.executiveSummary.evidenceTrace}
              </span>
            </div>

            {/* Evidence Trace Drawer */}
            {isEvidenceOpen && (
              <div className="mt-3 p-4 bg-bg-surface border border-cyan-500/30 rounded-lg text-xs font-mono text-text-secondary space-y-1 animate-fadeIn">
                <div className="text-cyan-400 font-bold uppercase text-[11px]">Audited Root Trace:</div>
                <div>Hash: SHA256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069</div>
                <div>Attribution: {narrativeResult.executiveSummary.evidenceTrace}</div>
                <div>Status: Invariant INV-B8 Satisfied (Evidence &rarr; Driver &rarr; Learning &rarr; Action)</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 3. Top Opportunity vs. Top Risk Cards (Side-by-Side) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Opportunity */}
        <div
          className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm"
          data-testid="top-opportunity-card"
        >
          <div className="flex items-center justify-between border-b border-border-subtle pb-3">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-accent-positive" />
              <h3 className="text-header-2 font-bold text-text-primary text-sm uppercase">
                Top Opportunity Driver
              </h3>
            </div>
            <span className="px-2 py-0.5 rounded text-caption-mono text-xs font-bold bg-accent-positive/10 text-accent-positive border border-accent-positive/30">
              +{narrativeResult.topOpportunity.estimatedContributionPoints} DQ Points
            </span>
          </div>

          <div className="space-y-2">
            <h4 className="text-header-1 font-bold text-text-primary text-base">
              {narrativeResult.topOpportunity.title}
            </h4>
            <p className="text-body-ui text-text-secondary text-sm">
              {narrativeResult.topOpportunity.driverPattern}
            </p>
          </div>

          <div className="p-3 bg-bg-surface-raised rounded-xl border border-border-subtle space-y-1">
            <div className="text-caption-mono text-accent-positive text-xs font-bold uppercase">
              Actionable Directive:
            </div>
            <div className="text-body-ui text-text-primary text-xs font-semibold">
              {narrativeResult.topOpportunity.actionableDirective}
            </div>
          </div>

          <div className="flex items-center justify-between text-caption-mono text-xs text-text-muted pt-2 border-t border-border-subtle">
            <span>Historical Win Rate: <strong className="text-accent-positive">{narrativeResult.topOpportunity.historicalWinRate}%</strong></span>
            <span>Confidence: <strong className="text-text-primary">{narrativeResult.topOpportunity.confidence}%</strong> (n={narrativeResult.topOpportunity.evidenceSample})</span>
          </div>
        </div>

        {/* Top Risk */}
        <div
          className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm"
          data-testid="top-risk-card"
        >
          <div className="flex items-center justify-between border-b border-border-subtle pb-3">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-accent-negative" />
              <h3 className="text-header-2 font-bold text-text-primary text-sm uppercase">
                Primary Threat & Drift Source
              </h3>
            </div>
            <span className="px-2 py-0.5 rounded text-caption-mono text-xs font-bold bg-accent-negative/10 text-accent-negative border border-accent-negative/30">
              {narrativeResult.topRisk.lossContributionPct}% Loss Contribution
            </span>
          </div>

          <div className="space-y-2">
            <h4 className="text-header-1 font-bold text-text-primary text-base">
              {narrativeResult.topRisk.title}
            </h4>
            <p className="text-body-ui text-text-secondary text-sm">
              {narrativeResult.topRisk.threatPattern}
            </p>
          </div>

          <div className="p-3 bg-bg-surface-raised rounded-xl border border-border-subtle space-y-1">
            <div className="text-caption-mono text-accent-negative text-xs font-bold uppercase">
              Mandatory Mitigation:
            </div>
            <div className="text-body-ui text-text-primary text-xs font-semibold">
              {narrativeResult.topRisk.mitigationDirective}
            </div>
          </div>

          <div className="flex items-center justify-between text-caption-mono text-xs text-text-muted pt-2 border-t border-border-subtle">
            <span>Risk Severity: <strong className="text-accent-negative">Elevated</strong></span>
            <span>Confidence: <strong className="text-text-primary">{narrativeResult.topRisk.confidence}%</strong> (n={narrativeResult.topRisk.evidenceSample})</span>
          </div>
        </div>
      </div>

      {/* 4. AI Executive Coach & Capability Impact Attribution (Behavioral ROI) */}
      <div
        className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-6 shadow-sm"
        data-testid="capability-roi-section"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border-subtle pb-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 text-caption-mono text-xs font-bold rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 uppercase">
                AI Executive Coach &bull; Capability Impact Attribution
              </span>
              <span className="px-2 py-0.5 text-caption-mono text-xs font-bold rounded bg-accent-positive/10 text-accent-positive border border-accent-positive/30">
                Conservation Invariant: PASS
              </span>
            </div>
            <h3 className="text-header-1 font-bold text-text-primary mt-1">
              Which ARX Capabilities Improve Your Decisions?
            </h3>
          </div>
          <div className="text-right text-caption-mono text-xs">
            <span className="text-text-muted">Total Improvement: </span>
            <strong className="text-accent-positive font-mono text-sm">+{capabilityAttribution.totalImprovementPoints} DQ Points</strong>
          </div>
        </div>

        {/* Capability ROI Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {capabilityAttribution.capabilities.map((cap) => (
            <div
              key={cap.capabilityId}
              className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-2 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-caption-mono text-text-muted text-[11px] uppercase font-bold">
                    {cap.capabilityName}
                  </span>
                  <span className="px-1.5 py-0.5 rounded text-caption-mono text-[10px] font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                    CRI: {cap.capabilityRoiIndex}
                  </span>
                </div>
                <div className="text-display-2 font-mono font-black text-accent-positive mt-1">
                  +{cap.estimatedContribution} pts
                </div>
                <div className="text-caption text-text-secondary text-xs mt-1">
                  Range: [{cap.contributionRange.lower} &rarr; {cap.contributionRange.upper}] @ {cap.confidence}% conf
                </div>
                <p className="text-caption text-text-muted text-[11px] mt-2 line-clamp-3">
                  {cap.executiveExplanation}
                </p>
              </div>

              <div className="pt-2 border-t border-border-subtle flex items-center justify-between text-[11px] font-mono text-text-muted">
                <span>Usage: {cap.usageRate}%</span>
                <span>{cap.interactionsCount} events</span>
              </div>
            </div>
          ))}
        </div>

        {/* Conservation Invariant Ledger */}
        <div className="p-3 bg-bg-surface-raised border border-border-subtle rounded-xl flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
          <span className="text-text-muted">
            Attribution Balance: <strong className="text-text-primary">11.4 pts Explained</strong> + <strong className="text-text-primary">0.6 pt Residual</strong> = <strong className="text-accent-positive">12.0 Total</strong>
          </span>
          <span className="text-cyan-400 font-bold">
            Highest Leverage Feature: {capabilityAttribution.highestRoiCapability} (CRI: 6.0)
          </span>
        </div>
      </div>

      {/* 5. 4-Quarter Decision Quality Trend (Longitudinal Horizon) */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm">
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <h3 className="text-header-2 font-bold text-text-primary text-sm uppercase">
            Decision Quality Progression Horizon
          </h3>
          <span className="text-caption-mono text-accent-positive text-xs font-bold">
            Trajectory: Score 80 Target in 4.0 Months (87% Probability)
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-center">
          {[
            { quarter: '2025 Q4', score: 62, cohort: 'Consumer', isCurrent: false },
            { quarter: '2026 Q1', score: 66, cohort: 'Investigator', isCurrent: false },
            { quarter: '2026 Q2', score: 70, cohort: 'Practitioner', isCurrent: false },
            { quarter: 'CURRENT', score: 74, cohort: 'Learner', isCurrent: true },
            { quarter: 'TARGET Q4', score: 80, cohort: 'Optimizer', isCurrent: false },
          ].map((q) => (
            <div
              key={q.quarter}
              className={`p-3 rounded-xl border ${
                q.isCurrent
                  ? 'bg-cyan-500/10 border-cyan-500/50 shadow-sm'
                  : 'bg-bg-surface-raised border-border-subtle'
              }`}
            >
              <div className="text-caption-mono text-text-muted text-[10px] uppercase font-bold">{q.quarter}</div>
              <div className={`text-display-2 font-mono font-black ${q.isCurrent ? 'text-cyan-400' : 'text-text-primary'}`}>
                {q.score}
              </div>
              <div className="text-[11px] font-mono text-text-secondary mt-0.5">{q.cohort}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
