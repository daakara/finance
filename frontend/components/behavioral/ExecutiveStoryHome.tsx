'use client';

import React, { useState } from 'react';
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
  const { story, decisionQuality, strengths, risks } = profile;

  const topStrength = strengths[0];
  const topRisk = risks[0];

  return (
    <div className="space-y-6" data-testid="executive-story-home">
      {/* 1. Header Greeting Ribbon */}
      <div className="p-6 bg-gradient-to-r from-bg-surface-raised via-bg-surface to-bg-surface-raised border border-border-subtle rounded-2xl shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <span className="text-caption-mono text-accent-info font-bold uppercase tracking-wider text-xs">
              Executive Decision Intelligence
            </span>
            <h2 className="text-display-2 font-bold text-text-primary mt-0.5">
              GOOD MORNING {story.userName.toUpperCase()}
            </h2>
          </div>
          <div className="text-caption-mono text-text-muted font-semibold text-sm">
            {story.dateString}
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

      {/* 2. Immediate 5-Second Glanceable Matrix */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Decision Quality */}
        <div className="p-5 bg-bg-surface border border-border-subtle rounded-xl space-y-2">
          <div className="text-caption-mono text-text-muted uppercase text-xs">Decision Quality</div>
          <div className="flex items-baseline gap-2">
            <span className="text-display-1 font-mono font-extrabold text-accent-positive">
              {decisionQuality.currentScore}
            </span>
            <span className="text-body-ui font-mono font-bold text-accent-positive">
              ▲ +{decisionQuality.annualChange}
            </span>
          </div>
          <div className="text-caption text-text-secondary">
            Top <strong className="text-text-primary">{decisionQuality.percentileRank}%</strong> cohort
          </div>
          <div className="w-full bg-bg-surface-raised h-2 rounded-full overflow-hidden border border-border-subtle mt-2">
            <div
              className="bg-accent-positive h-full rounded-full"
              style={{ width: `${(decisionQuality.currentScore / decisionQuality.targetScore) * 100}%` }}
            />
          </div>
          <div className="text-[11px] font-mono text-text-muted flex justify-between">
            <span>Score: {decisionQuality.currentScore}</span>
            <span>Target: {decisionQuality.targetScore}</span>
          </div>
        </div>

        {/* Card 2: Biggest Strength */}
        <div className="p-5 bg-bg-surface border border-accent-positive/30 rounded-xl space-y-2">
          <div className="text-caption-mono text-accent-positive uppercase text-xs font-bold">
            Biggest Positive Change
          </div>
          <h3 className="text-header-2 font-bold text-text-primary">
            {topStrength.title}
          </h3>
          <div className="text-body-ui font-mono font-bold text-accent-positive">
            +{topStrength.qualityPointContribution} quality points
          </div>
          <p className="text-caption text-text-secondary line-clamp-2">
            {topStrength.description}
          </p>
          <div className="text-[11px] font-mono text-text-muted pt-1">
            Confidence: <strong className="text-text-primary">{topStrength.confidence}%</strong>
          </div>
        </div>

        {/* Card 3: Biggest Risk */}
        <div className="p-5 bg-bg-surface border border-accent-warning/30 rounded-xl space-y-2">
          <div className="text-caption-mono text-accent-warning uppercase text-xs font-bold">
            Biggest Risk Exposure
          </div>
          <h3 className="text-header-2 font-bold text-text-primary">
            {topRisk.title}
          </h3>
          <div className="text-body-ui font-mono font-bold text-accent-warning">
            {topRisk.exposurePercentage}% current risk exposure
          </div>
          <p className="text-caption text-text-secondary line-clamp-2">
            {topRisk.description}
          </p>
          <div className="text-[11px] font-mono text-text-muted pt-1">
            Confidence: <strong className="text-text-primary">{topRisk.confidence}%</strong>
          </div>
        </div>

        {/* Card 4: Recommended Focus Today */}
        <div className="p-5 bg-bg-surface border border-accent-info/30 rounded-xl space-y-2">
          <div className="text-caption-mono text-accent-info uppercase text-xs font-bold">
            Recommended Focus Today
          </div>
          <h3 className="text-header-2 font-bold text-text-primary">
            Reduce Regime Weakness
          </h3>
          <div className="text-body-ui font-mono font-bold text-accent-info">
            Expected: +3.4 quality pts
          </div>
          <p className="text-caption text-text-secondary line-clamp-2">
            {story.recommendedActionToday}
          </p>
          <div className="text-[11px] font-mono text-text-muted pt-1">
            Confidence: <strong className="text-text-primary">89%</strong>
          </div>
        </div>
      </div>

      {/* 3. Narrative Sections: Today's Story & Outcome Narrative */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Today's Story */}
        <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4">
          <div className="flex items-center justify-between border-b border-border-subtle pb-3">
            <h3 className="text-header-1 text-text-primary">
              Your Story This Week
            </h3>
            <span className="px-2.5 py-0.5 text-caption-mono text-xs font-bold bg-accent-positive/10 text-accent-positive border border-accent-positive/30 rounded">
              84% Compliance
            </span>
          </div>

          <div className="space-y-3 text-body-ui text-text-secondary">
            <p className="text-text-primary font-medium">
              You followed <strong className="text-accent-positive font-bold">84%</strong> of AI recommendations this week.
            </p>
            <ul className="space-y-2 text-caption">
              <li className="flex items-start gap-2">
                <span className="text-accent-positive font-bold mt-0.5">✓</span>
                <span>Decision quality improved <strong className="text-text-primary font-mono font-bold">+{story.weeklyScoreDelta} points</strong> across evaluated setups.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent-positive font-bold mt-0.5">✓</span>
                <span>Repeat mistakes fell <strong className="text-accent-positive font-mono font-bold">{story.repeatMistakeDelta}%</strong> relative to previous 30-day baseline.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent-warning font-bold mt-0.5">!</span>
                <span>{story.macroExposureTrend}</span>
              </li>
            </ul>

            <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle text-caption">
              <span className="text-accent-info font-bold font-mono uppercase text-xs block mb-1">
                Recommended Action
              </span>
              <span className="text-text-primary font-medium">
                {story.recommendedActionToday}
              </span>
            </div>
          </div>
        </div>

        {/* Outcome Narrative */}
        <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4">
          <div className="flex items-center justify-between border-b border-border-subtle pb-3">
            <h3 className="text-header-1 text-text-primary">
              Monthly Outcome Narrative
            </h3>
            <span className="text-caption-mono text-text-muted text-xs">
              42 Decisions Recorded
            </span>
          </div>

          <div className="space-y-3 text-caption">
            <div className="flex items-center justify-between p-3 bg-bg-surface-raised rounded-lg border border-border-subtle font-mono">
              <span>Successes: <strong className="text-accent-positive">{story.monthlyStats.successCount}</strong></span>
              <span>Failures: <strong className="text-accent-warning">{story.monthlyStats.failureCount}</strong></span>
              <span>Win Rate: <strong className="text-text-primary">{((story.monthlyStats.successCount / story.monthlyStats.totalDecisions) * 100).toFixed(1)}%</strong></span>
            </div>

            <div className="space-y-2">
              <div className="text-text-secondary">
                <span className="text-text-muted block text-[11px] font-mono uppercase">Largest Success</span>
                <span className="text-text-primary font-medium">{story.monthlyStats.largestSuccessDriver}</span>
              </div>
              <div className="text-text-secondary">
                <span className="text-text-muted block text-[11px] font-mono uppercase">Largest Failure</span>
                <span className="text-text-primary font-medium">{story.monthlyStats.largestFailureDriver}</span>
              </div>
              <div className="p-3 bg-accent-info/10 rounded-lg border border-accent-info/30 text-text-primary">
                <span className="text-accent-info font-mono text-xs uppercase font-bold block mb-0.5">Net Learning</span>
                {story.monthlyStats.netLearningTakeaway}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 4. AI Chief of Staff Directive */}
      <div className="p-6 bg-gradient-to-r from-bg-surface via-bg-surface-raised to-bg-surface border-2 border-accent-info/40 rounded-2xl space-y-4 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 text-caption-mono font-bold uppercase bg-accent-info/20 text-accent-info border border-accent-info/40 rounded">
              ARX Chief of Staff
            </span>
            <span className="text-caption-mono text-text-muted text-xs">
              Autonomous Behavioral Oversight
            </span>
          </div>
          <div className="text-caption-mono text-accent-positive text-xs font-bold">
            Confidence: {story.chiefOfStaffHighlight.confidence}%
          </div>
        </div>

        <div>
          <h4 className="text-body-ui font-mono uppercase text-accent-info font-bold">
            If you only do one thing today:
          </h4>
          <p className="text-header-1 font-bold text-text-primary mt-1">
            {story.chiefOfStaffHighlight.actionTitle}.
          </p>
          <p className="text-body-ui text-text-secondary mt-1">
            Estimated capital drawdown reduction: <strong className="text-accent-positive font-mono font-bold">-{story.chiefOfStaffHighlight.drawdownReductionPct}%</strong> under stress testing.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 pt-2">
          <button
            onClick={() => setShowEvidence(!showEvidence)}
            className="px-4 py-2 text-body-ui font-medium rounded-lg bg-accent-info/15 text-accent-info border border-accent-info/40 hover:bg-accent-info/25 transition-colors"
          >
            {showEvidence ? 'Hide Evidence ▲' : 'Show Evidence ▼'}
          </button>
        </div>

        {showEvidence && (
          <div className="p-4 bg-bg-surface rounded-lg border border-border-subtle text-caption text-text-secondary font-mono animate-in fade-in duration-200">
            {story.chiefOfStaffHighlight.evidenceDetail}
          </div>
        )}
      </div>
    </div>
  );
}
