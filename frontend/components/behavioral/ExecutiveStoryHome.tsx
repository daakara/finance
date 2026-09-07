'use client';

import React, { useState } from 'react';
import ExecutiveStatusStrip from './ExecutiveStatusStrip';
import {
  CANONICAL_BEHAVIORAL_PROFILE,
} from '@/lib/telemetry/behavioralStoryEngine';
import { BehavioralIntelligenceProfile } from '@/types/behavioral-intelligence';

interface ExecutiveStoryHomeProps {
  profile?: BehavioralIntelligenceProfile;
}

export default function ExecutiveStoryHome({
  profile = CANONICAL_BEHAVIORAL_PROFILE,
}: ExecutiveStoryHomeProps) {
  const [showEvidence, setShowEvidence] = useState<boolean>(false);
  const [showSimulation, setShowSimulation] = useState<boolean>(false);
  const { story, decisionQuality, strengths, risks } = profile;

  return (
    <div
      className="space-y-6"
      data-testid="executive-story-home"
      data-component-id="ARX-EXH-001"
    >
      {/* 1. Header Greeting Ribbon */}
      <div className="p-6 bg-gradient-to-r from-bg-surface-raised via-bg-surface to-bg-surface-raised border border-border-subtle rounded-2xl shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono text-accent-info font-bold uppercase tracking-wider text-xs rounded bg-accent-info/10 border border-accent-info/30">
                Executive Home · ARX-EXH-001
              </span>
              <span className="text-caption-mono text-text-muted text-xs">
                Comprehension: &le; 30s
              </span>
            </div>
            <h1 className="text-display-1 md:text-display-2 font-black text-text-primary mt-1 tracking-tight">
              GOOD MORNING {story.userName.toUpperCase()}
            </h1>
          </div>
          <div className="text-right">
            <div className="text-caption-mono text-text-muted font-semibold text-sm">
              {story.dateString}
            </div>
            <div className="text-caption-mono text-accent-positive font-bold text-xs mt-0.5">
              Institutional Production Excellence Certified (99.3%)
            </div>
          </div>
        </div>

        <div className="mt-3 pt-3 border-t border-border-subtle/80 flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-accent-positive animate-pulse" />
          <span className="text-body-ui font-semibold text-accent-positive">
            You are improving.
          </span>
          <span className="text-caption text-text-secondary">
            Decision quality up +{decisionQuality.annualChange} points over the last 12 months.
          </span>
        </div>
      </div>

      {/* Component A: Executive Status Strip (72px height) */}
      <ExecutiveStatusStrip />

      {/* 2. Current State vs. Today's Priorities (Wireframe Section 1) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Box Left: YOUR CURRENT STATE */}
        <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-border-subtle pb-3">
            <h2 className="text-header-2 font-bold text-text-primary uppercase tracking-wider text-xs">
              YOUR CURRENT STATE
            </h2>
            <span className="px-2.5 py-0.5 text-caption-mono text-xs font-bold rounded bg-accent-positive/10 text-accent-positive border border-accent-positive/30">
              Improving
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Metric 1 */}
            <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-1">
              <div className="text-caption-mono text-text-muted text-xs uppercase">Decision Quality</div>
              <div className="flex items-baseline gap-2">
                <span className="text-display-2 font-mono font-black text-accent-positive">
                  {decisionQuality.currentScore}
                </span>
                <span className="text-body-ui font-mono font-bold text-accent-positive text-sm">
                  ▲ +{decisionQuality.annualChange} Last Year
                </span>
              </div>
              <div className="text-caption text-text-secondary text-xs">
                Top <strong className="text-text-primary">{decisionQuality.percentileRank}%</strong> Cohort
              </div>
            </div>

            {/* Metric 2 */}
            <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-1">
              <div className="text-caption-mono text-text-muted text-xs uppercase">Decision Drift</div>
              <div className="flex items-baseline gap-2">
                <span className="text-display-2 font-mono font-black text-text-primary">
                  21%
                </span>
                <span className="px-2 py-0.5 text-caption-mono font-bold text-xs rounded bg-accent-positive/10 text-accent-positive border border-accent-positive/30">
                  LOW RISK
                </span>
              </div>
              <div className="text-caption text-text-secondary text-xs">
                Within mandate variance floor (&lt;25%)
              </div>
            </div>
          </div>
        </div>

        {/* Box Right: TODAY'S PRIORITIES */}
        <div className="p-6 bg-bg-surface border border-accent-warning/40 rounded-2xl space-y-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-border-subtle pb-3">
            <h2 className="text-header-2 font-bold text-accent-warning uppercase tracking-wider text-xs">
              TODAY&apos;S PRIORITIES
            </h2>
            <span className="px-2.5 py-0.5 text-caption-mono text-xs font-bold rounded bg-accent-warning/15 text-accent-warning border border-accent-warning/30">
              Action Required
            </span>
          </div>

          <div className="space-y-3 text-sm">
            <div className="flex items-center justify-between text-body-ui font-bold text-text-primary">
              <span>3 Positions Require Review</span>
              <span className="text-caption-mono text-text-muted text-xs">NVDA · AMD · CRWD</span>
            </div>

            <div className="p-3 bg-bg-surface-raised rounded-xl border border-border-subtle flex items-start justify-between gap-3">
              <div>
                <div className="text-caption-mono text-accent-risk font-bold text-xs uppercase">
                  Highest Risk
                </div>
                <div className="text-header-2 font-black text-text-primary mt-0.5">
                  NVDA
                </div>
                <div className="text-caption text-text-secondary text-xs">
                  Macro Deterioration Signal triggered overnight
                </div>
              </div>
              <div className="text-right">
                <div className="text-caption-mono text-text-muted text-[11px] uppercase">
                  Recommended Action
                </div>
                <div className="text-body-ui font-bold text-accent-info mt-0.5">
                  Reduce exposure 15%
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 3. YOUR STORY THIS WEEK (Wireframe Section 2) */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm">
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <h2 className="text-header-2 font-bold text-text-primary uppercase tracking-wider text-xs">
            YOUR STORY THIS WEEK
          </h2>
          <span className="px-2.5 py-0.5 text-caption-mono text-xs font-bold bg-accent-positive/10 text-accent-positive border border-accent-positive/30 rounded">
            84% Adherence
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-body-ui">
          <div className="space-y-3">
            <p className="text-text-primary font-medium">
              You followed <strong className="text-accent-positive font-bold">84%</strong> of recommendations.
            </p>
            <ul className="space-y-2 text-caption">
              <li className="flex items-start gap-2">
                <span className="text-accent-positive font-bold mt-0.5">✓</span>
                <span>Decision quality improved by <strong className="text-text-primary font-mono font-bold">2 points</strong> across evaluated trades.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent-positive font-bold mt-0.5">✓</span>
                <span>Repeat mistakes declined by <strong className="text-accent-positive font-mono font-bold">12%</strong>.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent-warning font-bold mt-0.5">!</span>
                <span>Macro risk exposure increased significantly during the last three sessions.</span>
              </li>
            </ul>
          </div>

          <div className="p-4 bg-bg-surface-raised rounded-xl border border-border-subtle space-y-3 text-caption">
            <div>
              <span className="text-caption-mono text-accent-positive font-bold text-xs uppercase block">
                Strongest Positive Change &bull; Biggest Positive Change
              </span>
              <span className="text-body-ui font-bold text-text-primary">
                Institutional Flow Discipline
              </span>
            </div>
            <div>
              <span className="text-caption-mono text-accent-info font-bold text-xs uppercase block">
                Recommended Focus
              </span>
              <span className="text-body-ui font-bold text-accent-info">
                Tighten macro filters.
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 4. WHAT'S WORKING vs. WHAT'S HURTING (Wireframe Section 3) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Working */}
        <div className="p-6 bg-bg-surface border border-accent-positive/30 rounded-2xl space-y-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-border-subtle pb-3">
            <h3 className="text-header-2 font-bold text-accent-positive uppercase tracking-wider text-xs">
              WHAT&apos;S WORKING
            </h3>
            <span className="text-caption-mono text-text-muted text-xs">
              Alpha Catalysts
            </span>
          </div>

          <div className="space-y-3">
            <div className="p-3 bg-bg-surface-raised border border-border-subtle rounded-xl flex items-center justify-between">
              <div>
                <div className="text-body-ui font-bold text-text-primary">Institutional Flow</div>
                <div className="text-caption text-text-secondary text-xs">Surge accumulation conviction</div>
              </div>
              <div className="text-right font-mono">
                <div className="text-header-2 font-bold text-accent-positive">72%</div>
                <div className="text-caption text-text-muted text-[10px] uppercase">Win Rate</div>
              </div>
            </div>

            <div className="p-3 bg-bg-surface-raised border border-border-subtle rounded-xl flex items-center justify-between">
              <div>
                <div className="text-body-ui font-bold text-text-primary">Sector Confirmation</div>
                <div className="text-caption text-text-secondary text-xs">Relative strength alignment</div>
              </div>
              <div className="text-right font-mono">
                <div className="text-header-2 font-bold text-accent-positive">69%</div>
                <div className="text-caption text-text-muted text-[10px] uppercase">Win Rate</div>
              </div>
            </div>
          </div>
        </div>

        {/* Hurting */}
        <div className="p-6 bg-bg-surface border border-accent-warning/40 rounded-2xl space-y-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-border-subtle pb-3">
            <h3 className="text-header-2 font-bold text-accent-warning uppercase tracking-wider text-xs">
              WHAT&apos;S HURTING
            </h3>
            <span className="text-caption-mono text-text-muted text-xs">
              Drag Factors &bull; Biggest Risk Exposure
            </span>
          </div>

          <div className="space-y-3">
            <div className="p-3 bg-bg-surface-raised border border-border-subtle rounded-xl flex items-center justify-between">
              <div>
                <div className="text-body-ui font-bold text-text-primary">Late Momentum Entries</div>
                <div className="text-caption text-text-secondary text-xs">Chasing extended breakouts</div>
              </div>
              <div className="text-right font-mono">
                <div className="text-header-2 font-bold text-accent-risk">28%</div>
                <div className="text-caption text-text-muted text-[10px] uppercase">Loss Contribution</div>
              </div>
            </div>

            <div className="p-3 bg-bg-surface-raised border border-border-subtle rounded-xl flex items-center justify-between">
              <div>
                <div className="text-body-ui font-bold text-text-primary">Macro Blindness</div>
                <div className="text-caption text-text-secondary text-xs">Trading growth during regime shift</div>
              </div>
              <div className="text-right font-mono">
                <div className="text-body-ui font-bold text-accent-warning">#1</div>
                <div className="text-caption text-text-muted text-[10px] uppercase">Drift Source</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 5. AI CHIEF OF STAFF (Wireframe Section 4) */}
      <div className="p-6 bg-gradient-to-r from-bg-surface via-bg-surface-raised to-bg-surface border-2 border-accent-info/40 rounded-2xl space-y-4 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 text-caption-mono font-bold uppercase bg-accent-info/20 text-accent-info border border-accent-info/40 rounded">
              AI CHIEF OF STAFF &bull; ARX Chief of Staff
            </span>
            <span className="text-caption-mono text-text-muted text-xs">
              Decision Optimization Engine
            </span>
          </div>
          <div className="text-caption-mono text-accent-positive text-xs font-bold">
            Confidence: 89%
          </div>
        </div>

        <div>
          <h4 className="text-body-ui font-mono uppercase text-accent-info font-bold">
            If you only do one thing today:
          </h4>
          <p className="text-header-1 font-bold text-text-primary mt-1">
            Review positions exposed to deteriorating macro signals.
          </p>
          <p className="text-body-ui text-text-secondary mt-1">
            Estimated improvement: <strong className="text-accent-positive font-mono font-bold">+3.4 Decision Quality Points</strong>.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 pt-2">
          <button
            onClick={() => setShowEvidence(!showEvidence)}
            className="px-4 py-2 text-body-ui font-medium rounded-lg bg-accent-info/15 text-accent-info border border-accent-info/40 hover:bg-accent-info/25 transition-colors"
          >
            {showEvidence ? 'Hide Evidence ▲' : 'View Evidence'}
          </button>
          <a
            href="/design-system-preview?tab=sprint-8"
            className="px-4 py-2 text-body-ui font-medium rounded-lg bg-bg-surface-raised text-text-primary border border-border-subtle hover:bg-bg-surface-elevated transition-colors"
          >
            Open Playbook
          </a>
          <button
            onClick={() => setShowSimulation(!showSimulation)}
            className="px-4 py-2 text-body-ui font-medium rounded-lg bg-bg-surface-raised text-text-secondary border border-border-subtle hover:text-text-primary transition-colors"
          >
            {showSimulation ? 'Close Simulation' : 'See Impact Simulation'}
          </button>
        </div>

        {showEvidence && (
          <div className="p-4 bg-bg-surface rounded-lg border border-border-subtle text-caption text-text-secondary font-mono animate-in fade-in duration-200">
            {story.chiefOfStaffHighlight.evidenceDetail}
          </div>
        )}

        {showSimulation && (
          <div className="p-4 bg-bg-surface rounded-lg border border-accent-positive/30 text-caption text-text-secondary space-y-2 animate-in fade-in duration-200">
            <div className="font-bold text-text-primary">Impact Simulation: 15% Macro Exposure Reduction</div>
            <div className="text-accent-positive font-mono font-bold">Estimated downside protected: $184,000 under 2-sigma shock</div>
            <div>Projected Sharpe increase: +0.28 over rolling 90 days.</div>
          </div>
        )}
      </div>

      {/* 6. WHERE YOU ARE GOING (Wireframe Section 5 & Component E: Decision Quality Trend) */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm">
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <h2 className="text-header-2 font-bold text-text-primary uppercase tracking-wider text-xs">
            WHERE YOU ARE GOING
          </h2>
          <span className="text-caption-mono text-accent-positive font-bold text-xs">
            Target: 80 · 87% Confidence in 4 Months
          </span>
        </div>

        <div className="space-y-4 pt-2">
          {/* Progression Step Ribbon */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {[
              { score: 62, label: 'Baseline', status: 'past' },
              { score: 64, label: 'Q2', status: 'past' },
              { score: 67, label: 'Q3', status: 'past' },
              { score: 71, label: 'Q4', status: 'past' },
              { score: 74, label: 'Current', status: 'current' },
            ].map((step) => (
              <div
                key={step.score}
                className={`p-3 rounded-xl border text-center ${
                  step.status === 'current'
                    ? 'bg-accent-positive/15 border-accent-positive text-accent-positive shadow-sm'
                    : 'bg-bg-surface-raised border-border-subtle text-text-secondary'
                }`}
              >
                <div className="text-display-2 font-mono font-black">{step.score}</div>
                <div className="text-caption-mono text-xs uppercase mt-0.5">{step.label}</div>
              </div>
            ))}
          </div>

          <div className="p-4 bg-bg-surface-raised rounded-xl border border-border-subtle flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-sm">
            <div className="space-y-0.5">
              <div className="font-bold text-text-primary">Next Horizon: Quality Score 80</div>
              <div className="text-caption text-text-secondary text-xs">
                Calibrated across 549 peer portfolios. Current velocity index: 84 (High).
              </div>
            </div>
            <div className="flex items-center gap-2 font-mono font-bold text-accent-positive">
              <span>Confidence: 87%</span>
              <span className="text-text-muted">·</span>
              <span>4.0 Months Remaining</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
