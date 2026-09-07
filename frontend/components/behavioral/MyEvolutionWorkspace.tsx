'use client';

import React, { useState } from 'react';
import {
  getCanonicalEvolutionJourney,
  CANONICAL_EVOLUTION_MILESTONES,
  CANONICAL_BEHAVIOR_LEDGER,
  CANONICAL_CAPABILITY_LEADERBOARD,
} from '@/lib/telemetry/decisionEvolutionEngine';
import {
  EvolutionInteractionMode,
  EvolutionMilestone,
} from '@/types/behavioral-intelligence';

export default function MyEvolutionWorkspace() {
  const journey = getCanonicalEvolutionJourney();
  const [activeMode, setActiveMode] = useState<EvolutionInteractionMode>('EXPLORATION');
  const [selectedMilestone, setSelectedMilestone] = useState<EvolutionMilestone>(
    journey.milestones.find((m) => m.isCurrent) ?? journey.milestones[3]
  );

  return (
    <div
      className="space-y-8 max-w-7xl mx-auto px-2 sm:px-4 py-4"
      data-testid="my-evolution-workspace"
      data-component-id="ARX-EVO-001"
    >
      {/* 0. Interaction Mode Switcher Toolbar */}
      <div
        className="p-3 bg-bg-surface border border-border-subtle rounded-xl flex flex-wrap items-center justify-between gap-3 text-xs shadow-sm"
        data-testid="interaction-mode-selector"
        aria-label="Interaction Mode Selector"
      >
        <div className="flex items-center gap-2">
          <span className="text-caption-mono text-cyan-400 font-bold uppercase text-[11px]">
            Workspace Lens:
          </span>
          <span className="text-caption-mono text-text-muted">
            {activeMode === 'SUMMARY' && 'Fast-read summary (<5 seconds)'}
            {activeMode === 'EXPLORATION' && 'Full 4-quarter progression journey'}
            {activeMode === 'ANALYSIS' && 'Deep milestone audit & behavior change'}
            {activeMode === 'PROJECTION' && 'Forward trajectory to score 80'}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Evolution Interaction Modes">
          {(['SUMMARY', 'EXPLORATION', 'ANALYSIS', 'PROJECTION'] as EvolutionInteractionMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => setActiveMode(mode)}
              aria-label={`Select ${mode} view`}
              className={`px-3 py-1.5 min-h-[44px] rounded-lg text-caption-mono text-xs font-semibold transition-all ${
                activeMode === mode
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-sm'
                  : 'bg-bg-surface-raised text-text-secondary hover:text-text-primary border border-border-subtle'
              }`}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>

      {/* LAYER 1: EVOLUTION HERO */}
      <div
        className="p-6 md:p-8 bg-gradient-to-br from-bg-surface via-bg-surface-raised to-bg-surface border border-border-subtle rounded-2xl shadow-md space-y-6"
        data-testid="evolution-hero"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border-subtle pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono text-xs font-bold uppercase rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                Decision-Maker Evolution Profile
              </span>
              <span className="px-2.5 py-0.5 text-caption-mono text-xs font-bold uppercase rounded bg-accent-positive/10 text-accent-positive border border-accent-positive/30">
                Top 18% Institutional Peer Group
              </span>
            </div>
            <h1 className="text-display-1 md:text-display-2 font-black text-text-primary mt-2 tracking-tight">
              MY EVOLUTION
            </h1>
            <p className="text-body-ui text-text-secondary text-sm md:text-base mt-1">
              You are improving faster than <strong className="text-text-primary">82%</strong> of comparable decision makers.
              Largest contributor: <strong className="text-accent-positive">{journey.evolutionCoach.biggestWin}</strong>.
            </p>
          </div>

          <div className="text-right">
            <div className="text-caption-mono text-text-muted text-xs uppercase font-bold">Current Decision Quality</div>
            <div className="flex items-baseline gap-2 justify-end mt-1">
              <span className="text-display-1 font-mono font-black text-accent-positive">
                {journey.currentDir}
              </span>
              <span className="text-caption-mono text-text-muted text-sm">/ 100</span>
              <span className="text-caption font-mono font-bold text-accent-positive text-sm">
                ▲ +{journey.totalGain} Last 4 Quarters
              </span>
            </div>
            <div className="text-caption-mono text-text-secondary text-xs mt-1">
              Level: <strong className="text-cyan-400">{journey.maturityTier}</strong>
            </div>
          </div>
        </div>

        {/* Hero KPI Matrix & Progress to Target 80 */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Tile 1 */}
          <div className="p-4 bg-bg-surface border border-border-subtle rounded-xl space-y-1">
            <div className="text-caption-mono text-text-muted uppercase text-xs">Baseline DIR</div>
            <div className="text-display-2 font-mono font-black text-text-primary">
              {journey.baselineDir} <span className="text-xs font-normal text-text-muted">(2025 Q4)</span>
            </div>
            <div className="text-caption text-text-secondary text-xs">
              Signal Consumer tier
            </div>
          </div>

          {/* Tile 2 */}
          <div className="p-4 bg-bg-surface border border-border-subtle rounded-xl space-y-1">
            <div className="text-caption-mono text-text-muted uppercase text-xs">Current DIR</div>
            <div className="text-display-2 font-mono font-black text-accent-positive">
              {journey.currentDir} <span className="text-xs font-bold text-accent-positive">▲ +12 pts</span>
            </div>
            <div className="text-caption text-text-secondary text-xs">
              Autonomous Learner tier
            </div>
          </div>

          {/* Tile 3 */}
          <div className="p-4 bg-bg-surface border border-border-subtle rounded-xl space-y-1">
            <div className="text-caption-mono text-text-muted uppercase text-xs">Target Horizon</div>
            <div className="text-display-2 font-mono font-black text-cyan-400">
              {journey.targetDir} <span className="text-xs font-normal text-text-muted">DIR</span>
            </div>
            <div className="text-caption text-text-secondary text-xs">
              Expected: {journey.targetHorizon}
            </div>
          </div>

          {/* Tile 4 */}
          <div className="p-4 bg-bg-surface border border-border-subtle rounded-xl space-y-1">
            <div className="text-caption-mono text-text-muted uppercase text-xs">Learning Velocity</div>
            <div className="text-display-2 font-mono font-black text-cyan-400">
              {journey.learningVelocity} <span className="text-xs font-bold text-accent-positive">{journey.learningVelocityClass}</span>
            </div>
            <div className="text-caption text-text-secondary text-xs">
              Attribution Coverage: {journey.attributionCoveragePct}%
            </div>
          </div>
        </div>

        {/* Visual Progress Bar to Target 80 */}
        <div className="space-y-2 p-4 bg-bg-surface border border-border-subtle rounded-xl">
          <div className="flex items-center justify-between text-caption-mono text-xs">
            <span className="text-text-muted uppercase font-bold">
              Progress to Institutional Optimizer (Score 80)
            </span>
            <span className="text-accent-positive font-bold">
              {journey.currentDir} / {journey.targetDir} ({journey.targetProbability}% Probability)
            </span>
          </div>
          <div className="w-full bg-bg-surface-raised h-3 rounded-full overflow-hidden border border-border-subtle">
            <div
              className="bg-gradient-to-r from-cyan-500 to-accent-positive h-full rounded-full transition-all duration-500"
              style={{ width: `${Math.round(((journey.currentDir - journey.baselineDir) / (journey.targetDir - journey.baselineDir)) * 100)}%` }}
            />
          </div>
          <div className="flex items-center justify-between text-caption text-text-muted text-[11px]">
            <span>Baseline 62 (Consumer)</span>
            <span>Current 74 (Learner)</span>
            <span>Target 80 (Optimizer)</span>
          </div>
        </div>
      </div>

      {/* LAYER 2: PROGRESSION JOURNEY TIMELINE */}
      <div
        className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-6 shadow-sm"
        data-testid="progression-timeline"
        aria-label="Longitudinal Progression Journey Timeline"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border-subtle pb-3">
          <div>
            <h2 className="text-header-1 font-bold text-text-primary">
              Longitudinal Progression Journey
            </h2>
            <p className="text-body-ui text-text-secondary text-sm">
              Click any quarterly milestone to inspect the underlying behavioral problem, intervention, and evidence trace.
            </p>
          </div>
          <span className="text-caption-mono text-accent-positive font-bold text-xs">
            Invariant INV-B9: 100% Traceable
          </span>
        </div>

        {/* Interactive Milestone Nodes */}
        <div className="grid grid-cols-1 sm:grid-cols-5 gap-3" role="region" aria-label="Quarterly Milestones">
          {journey.milestones.map((m) => {
            const isSelected = selectedMilestone.quarter === m.quarter;
            return (
              <div
                key={m.quarter}
                onClick={() => setSelectedMilestone(m)}
                role="button"
                tabIndex={0}
                aria-label={`Inspect milestone ${m.quarter}`}
                className={`p-4 rounded-xl border transition-all cursor-pointer flex flex-col justify-between min-h-[140px] ${
                  isSelected
                    ? 'bg-bg-surface-elevated border-cyan-500 shadow-md ring-2 ring-cyan-500/30'
                    : m.isCurrent
                    ? 'bg-cyan-500/10 border-cyan-500/40 hover:bg-bg-surface-elevated'
                    : m.isTarget
                    ? 'bg-bg-surface-raised border-dashed border-border-subtle hover:bg-bg-surface-elevated'
                    : 'bg-bg-surface-raised border-border-subtle hover:bg-bg-surface-elevated'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-caption-mono text-text-muted uppercase text-[11px] font-bold">
                      {m.quarter}
                    </span>
                    {m.isCurrent && (
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-500/20 text-cyan-300">
                        NOW
                      </span>
                    )}
                    {m.isTarget && (
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-accent-warning/20 text-accent-warning">
                        GOAL
                      </span>
                    )}
                  </div>

                  <div className="flex items-baseline gap-1.5 mt-2">
                    <span className={`text-display-2 font-mono font-black ${m.isCurrent ? 'text-cyan-400' : 'text-text-primary'}`}>
                      {m.dirScore}
                    </span>
                    <span className="text-caption-mono text-text-muted text-xs">DIR</span>
                    {m.scoreDelta > 0 && (
                      <span className="text-caption font-mono font-bold text-accent-positive text-xs ml-auto">
                        +{m.scoreDelta}
                      </span>
                    )}
                  </div>
                </div>

                <div className="pt-2 border-t border-border-subtle/80 mt-2">
                  <div className="text-caption font-semibold text-text-primary text-xs truncate">
                    {m.cohortLabel}
                  </div>
                  <div className="text-[11px] font-mono text-accent-positive truncate">
                    {m.outcomeImpact.drawdown} DD &bull; {m.outcomeImpact.winRate} Win
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Milestone Detail Inspector (Analysis Mode) */}
        <div
          className="p-5 bg-bg-surface-raised border border-cyan-500/30 rounded-xl space-y-4 shadow-sm"
          data-testid="milestone-inspector"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border-subtle pb-3">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded text-caption-mono text-xs font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                {selectedMilestone.quarter} Milestone Audit
              </span>
              <span className="text-caption-mono text-text-muted text-xs">
                Cohort: <strong className="text-text-primary">{selectedMilestone.cohortLabel}</strong> ({selectedMilestone.dirScore} DIR)
              </span>
            </div>
            <span className="text-caption-mono text-accent-positive font-bold text-xs">
              Attribution Confidence: {selectedMilestone.confidence}%
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            {/* Left: Problem & Intervention */}
            <div className="space-y-3">
              <div className="p-3 bg-bg-surface rounded-lg border border-border-subtle space-y-1">
                <div className="text-caption-mono text-accent-negative uppercase font-bold text-[11px]">
                  Problem Diagnosed:
                </div>
                <div className="text-body-ui text-text-primary font-semibold">
                  {selectedMilestone.problem}
                </div>
              </div>

              <div className="p-3 bg-bg-surface rounded-lg border border-border-subtle space-y-1">
                <div className="text-caption-mono text-cyan-400 uppercase font-bold text-[11px]">
                  Action / Intervention Implemented:
                </div>
                <div className="text-body-ui text-text-primary font-semibold">
                  {selectedMilestone.actionTaken}
                </div>
              </div>
            </div>

            {/* Right: Behavior Shifts & Outcomes */}
            <div className="space-y-3">
              <div className="p-3 bg-bg-surface rounded-lg border border-border-subtle space-y-1">
                <div className="text-caption-mono text-accent-positive uppercase font-bold text-[11px]">
                  Behavior Adopted:
                </div>
                <div className="text-body-ui text-text-primary font-semibold">
                  {selectedMilestone.behaviorAdopted}
                </div>
                <div className="text-caption text-text-muted text-[11px] pt-1">
                  Stopped: <span className="line-through text-text-secondary">{selectedMilestone.behaviorStopped}</span>
                </div>
              </div>

              <div className="p-3 bg-bg-surface rounded-lg border border-border-subtle space-y-1">
                <div className="text-caption-mono text-text-muted uppercase font-bold text-[11px]">
                  Outcome &amp; Attribution Impact:
                </div>
                <div className="flex items-center justify-between font-mono text-text-primary">
                  <span>Win Rate: <strong className="text-accent-positive">{selectedMilestone.outcomeImpact.winRate}</strong></span>
                  <span>Max Drawdown: <strong className="text-accent-positive">{selectedMilestone.outcomeImpact.drawdown}</strong></span>
                  <span>Profit Factor: <strong className="text-accent-positive">{selectedMilestone.outcomeImpact.profitFactor}</strong></span>
                </div>
                <div className="text-[11px] font-mono text-text-secondary pt-1">
                  Primary Capability: {selectedMilestone.primaryCapability} (+{selectedMilestone.capabilityContribution} pts)
                </div>
              </div>
            </div>
          </div>

          <div className="pt-2 border-t border-border-subtle flex items-center justify-between text-caption-mono text-xs text-text-muted">
            <span>Root Trace: {selectedMilestone.evidenceTrace}</span>
            <span className="text-accent-positive">Invariant INV-B9 Traceability Verified</span>
          </div>
        </div>
      </div>

      {/* LAYER 3: WHAT CHANGED (BEHAVIOR LEDGER & CAPABILITY LEADERBOARD) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Behavior Ledger */}
        <div
          className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm"
          data-testid="behavior-ledger"
        >
          <div className="flex items-center justify-between border-b border-border-subtle pb-3">
            <div>
              <h3 className="text-header-1 font-bold text-text-primary text-sm uppercase">
                Behavior Ledger
              </h3>
              <p className="text-caption text-text-secondary text-xs">
                Tangible habits adopted and bad habits eliminated across quarters.
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-caption-mono text-xs font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
              INV-B10 Conserved
            </span>
          </div>

          <div className="space-y-2.5">
            {journey.behaviorLedger.map((item) => (
              <div
                key={item.id}
                className="p-3 bg-bg-surface-raised border border-border-subtle rounded-xl flex items-center justify-between gap-3 text-xs"
              >
                <div className="flex items-center gap-2.5">
                  <span
                    className={`w-6 h-6 rounded-full flex items-center justify-center font-bold text-xs ${
                      item.type === 'ADOPTED'
                        ? 'bg-accent-positive/15 text-accent-positive border border-accent-positive/30'
                        : 'bg-accent-negative/15 text-accent-negative border border-accent-negative/30'
                    }`}
                  >
                    {item.type === 'ADOPTED' ? '✓' : '✗'}
                  </span>
                  <div>
                    <div className="text-body-ui font-semibold text-text-primary">
                      {item.name}
                    </div>
                    <div className="text-caption-mono text-text-muted text-[11px]">
                      {item.quarter} &bull; {item.metricCorrelation}
                    </div>
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-caption-mono font-bold text-accent-positive">
                    +{item.impactPoints} pts
                  </div>
                  <div className="text-caption text-text-muted text-[10px]">
                    {item.confidence}% conf
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Capability ROI Leaderboard */}
        <div
          className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm"
          data-testid="capability-roi-leaderboard"
          aria-label="Capability ROI Leaderboard"
        >
          <div className="flex items-center justify-between border-b border-border-subtle pb-3">
            <div>
              <h3 className="text-header-1 font-bold text-text-primary text-sm uppercase">
                Capability ROI Leaderboard
              </h3>
              <p className="text-caption text-text-secondary text-xs">
                Which ARX capabilities produce the largest return per interaction?
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-caption-mono text-xs font-bold bg-accent-positive/10 text-accent-positive border border-accent-positive/30">
              CRI Ranked
            </span>
          </div>

          <div className="space-y-3">
            {journey.capabilityLeaderboard.map((cap) => (
              <div
                key={cap.capabilityId}
                className="p-3.5 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-1.5 text-xs"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold text-cyan-400 text-sm">#{cap.rank}</span>
                    <span className="text-body-ui font-bold text-text-primary">{cap.name}</span>
                  </div>
                  <span className="px-2 py-0.5 rounded text-caption-mono text-[11px] font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                    CRI: {cap.cri}
                  </span>
                </div>

                <div className="flex items-baseline justify-between text-caption-mono text-text-secondary">
                  <span>Impact: <strong className="text-accent-positive font-mono">+{cap.impactPoints} pts</strong></span>
                  <span>Usage: <strong className="text-text-primary">{cap.usageRate}%</strong></span>
                  <span>Confidence: <strong className="text-text-primary">{cap.confidence}%</strong></span>
                </div>

                <p className="text-caption text-text-muted text-[11px] pt-1 border-t border-border-subtle/60">
                  {cap.strategicNote}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* LAYER 4: AI EVOLUTION COACH */}
      <div
        className="p-6 md:p-8 bg-bg-surface border border-border-subtle rounded-2xl space-y-6 shadow-sm"
        data-testid="ai-evolution-coach"
        aria-label="AI Evolution Coach"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border-subtle pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono text-xs font-bold uppercase rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                AI Evolution Coach
              </span>
              <span className="px-2.5 py-0.5 text-caption-mono text-xs font-bold uppercase rounded bg-accent-positive/10 text-accent-positive border border-accent-positive/30">
                Trajectory Target: 80 DIR
              </span>
            </div>
            <h3 className="text-header-1 font-bold text-text-primary mt-1">
              Personal Decision Development Horizon
            </h3>
          </div>
          <div className="text-right text-caption-mono text-xs text-text-muted">
            Expected: <strong className="text-accent-positive font-mono">{journey.evolutionCoach.projectedMonthsToTarget} Months</strong> @ {journey.evolutionCoach.projectedConfidence}% confidence
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Card 1: Biggest Win */}
          <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-2">
            <div className="text-caption-mono text-accent-positive uppercase font-bold text-xs">
              1. Biggest Improvement Win:
            </div>
            <div className="text-body-ui font-bold text-text-primary text-sm">
              {journey.evolutionCoach.biggestWin}
            </div>
            <p className="text-caption text-text-secondary text-xs">
              Adopting volume surge criteria on Stage 2 breakouts delivered $+6.2$ points and reduced losing streaks by $38\%$.
            </p>
          </div>

          {/* Card 2: Biggest Risk */}
          <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-2">
            <div className="text-caption-mono text-accent-negative uppercase font-bold text-xs">
              2. Primary Obstacle &amp; Threat:
            </div>
            <div className="text-body-ui font-bold text-text-primary text-sm">
              {journey.evolutionCoach.biggestRisk}
            </div>
            <p className="text-caption text-text-secondary text-xs">
              Holding cyclical tech assets during sovereign yield breakouts remains responsible for $21\%$ of negative volatility.
            </p>
          </div>

          {/* Card 3: Next Habit */}
          <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-2">
            <div className="text-caption-mono text-cyan-400 uppercase font-bold text-xs">
              3. Next Habit to Lock In:
            </div>
            <div className="text-body-ui font-bold text-text-primary text-sm">
              {journey.evolutionCoach.nextHabit}
            </div>
            <p className="text-caption text-text-secondary text-xs">
              Enforce strict risk-per-trade limits ($1.0\%-1.5\%$) regardless of subjective confidence to achieve Optimizer tier.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
