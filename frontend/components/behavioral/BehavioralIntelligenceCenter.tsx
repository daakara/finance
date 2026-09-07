'use client';

import React from 'react';
import { CANONICAL_BEHAVIORAL_PROFILE } from '@/lib/telemetry/behavioralStoryEngine';
import { BehavioralIntelligenceProfile } from '@/types/behavioral-intelligence';

interface BehavioralIntelligenceCenterProps {
  profile?: BehavioralIntelligenceProfile;
}

export default function BehavioralIntelligenceCenter({
  profile = CANONICAL_BEHAVIORAL_PROFILE,
}: BehavioralIntelligenceCenterProps) {
  const { decisionQuality, learningVelocity, behaviorAdoption, ruleAdherence, decisionDrift, timeline } = profile;

  return (
    <div className="space-y-6" data-testid="behavioral-intelligence-center">
      {/* Top Banner: Evolution Header */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono font-bold uppercase bg-accent-positive/10 text-accent-positive border border-accent-positive/30 rounded">
                Behavioral Intelligence Center
              </span>
              <span className="text-caption-mono text-text-muted">
                Decision Improvement Intelligence
              </span>
            </div>
            <h3 className="text-display-2 font-bold text-text-primary mt-1">
              YOUR DECISION EVOLUTION
            </h3>
            <p className="text-body-ui text-text-secondary mt-0.5">
              Empirical tracking measuring the elimination of systemic mistakes, rule adherence, and decision velocity.
            </p>
          </div>

          <div className="text-right">
            <div className="text-caption-mono text-text-muted uppercase text-xs">Learning Velocity (LVI)</div>
            <div className="text-display-1 font-mono font-extrabold text-accent-positive">
              {learningVelocity.velocityIndex} <span className="text-body-ui font-normal text-text-muted">/ 100</span>
            </div>
            <div className="text-caption-mono text-accent-positive font-bold text-xs">
              {learningVelocity.velocityTier} &bull; Top 12% Peer Group
            </div>
          </div>
        </div>

        {/* 5 Behavioral Pillars Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 pt-4 border-t border-border-subtle">
          <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
            <div className="text-[11px] font-mono text-text-muted uppercase">1. Decision Quality</div>
            <div className="text-header-1 font-mono font-bold text-accent-positive mt-0.5">
              {decisionQuality.currentScore}
            </div>
            <div className="text-[10px] font-mono text-accent-positive">▲ +{decisionQuality.annualChange} (12 Mo)</div>
          </div>

          <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
            <div className="text-[11px] font-mono text-text-muted uppercase">2. Adoption Rate (BAR)</div>
            <div className="text-header-1 font-mono font-bold text-accent-positive mt-0.5">
              {behaviorAdoption.value.toFixed(1)}%
            </div>
            <div className="text-[10px] font-mono text-text-muted">Target: &ge; 70.0%</div>
          </div>

          <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
            <div className="text-[11px] font-mono text-text-muted uppercase">3. Rule Adherence</div>
            <div className="text-header-1 font-mono font-bold text-accent-positive mt-0.5">
              {ruleAdherence.value.toFixed(1)}%
            </div>
            <div className="text-[10px] font-mono text-text-muted">Target: &ge; 85.0%</div>
          </div>

          <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
            <div className="text-[11px] font-mono text-text-muted uppercase">4. Repeat Mistakes</div>
            <div className="text-header-1 font-mono font-bold text-accent-positive mt-0.5">
              21 &rarr; 12
            </div>
            <div className="text-[10px] font-mono text-accent-positive">-43% Reduction</div>
          </div>

          <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
            <div className="text-[11px] font-mono text-text-muted uppercase">5. Decision Drift</div>
            <div className="text-header-1 font-mono font-bold text-accent-positive mt-0.5">
              {decisionDrift.driftScore.toFixed(1)}%
            </div>
            <div className="text-[10px] font-mono text-accent-positive">{decisionDrift.driftCategory} RISK</div>
          </div>
        </div>
      </div>

      {/* Chronological Behavioral Timeline */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4">
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <div>
            <h4 className="text-header-1 text-text-primary">
              Behavioral Milestone Timeline
            </h4>
            <p className="text-body-ui text-text-secondary mt-0.5">
              Chronological ledger tracking behavioral improvements and their measured impact on decision quality.
            </p>
          </div>
          <span className="text-caption-mono text-text-muted text-xs">
            3 Major Breakthroughs
          </span>
        </div>

        <div className="space-y-4">
          {timeline.map((event, idx) => (
            <div
              key={idx}
              className="p-4 bg-bg-surface-raised rounded-xl border border-border-subtle space-y-2"
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="flex items-center gap-3">
                  <span className="px-2.5 py-1 text-caption-mono font-bold bg-accent-info/10 text-accent-info border border-accent-info/30 rounded">
                    {event.quarter}
                  </span>
                  <span className="text-caption-mono text-text-muted uppercase text-xs">
                    Category: {event.category}
                  </span>
                </div>

                <div className="flex items-center gap-2 text-caption-mono">
                  <span className="text-text-muted">Quality Delta:</span>
                  <span className="px-2 py-0.5 font-bold text-xs rounded bg-accent-positive/15 text-accent-positive border border-accent-positive/30">
                    +{event.qualityDelta.toFixed(1)} Points
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div>
                  <span className="text-[11px] font-mono text-accent-warning uppercase font-bold block">
                    Problem Identified
                  </span>
                  <p className="text-body-ui text-text-primary mt-0.5">
                    {event.problemIdentified}
                  </p>
                </div>
                <div>
                  <span className="text-[11px] font-mono text-accent-positive uppercase font-bold block">
                    Governance Improvement Applied
                  </span>
                  <p className="text-body-ui text-text-primary mt-0.5">
                    {event.governanceImprovement}
                  </p>
                </div>
              </div>

              <div className="pt-2 border-t border-border-subtle text-caption text-text-muted font-mono">
                Empirical Evidence: {event.supportingEvidence}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
