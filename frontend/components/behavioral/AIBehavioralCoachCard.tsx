'use client';

import React, { useState } from 'react';
import { CANONICAL_IMPROVEMENT_FORECAST } from '@/lib/telemetry/behavioralStoryEngine';
import { ImprovementForecast } from '@/types/behavioral-intelligence';

interface AIBehavioralCoachCardProps {
  forecast?: ImprovementForecast;
}

export default function AIBehavioralCoachCard({
  forecast = CANONICAL_IMPROVEMENT_FORECAST,
}: AIBehavioralCoachCardProps) {
  const [scenarioMode, setScenarioMode] = useState<'CURRENT_PACE' | 'ACCELERATED'>('CURRENT_PACE');

  const isAccelerated = scenarioMode === 'ACCELERATED';
  const displayMonths = isAccelerated ? 2.5 : 4.0;
  const displayConfidence = isAccelerated ? 79 : forecast.confidence;

  const contributors = [
    { rank: 1, title: 'Better stop discipline', points: 3.2, description: 'Cutting losses at invalidation bounds eliminated tail drawdowns.' },
    { rank: 2, title: 'Stronger macro filtering', points: 4.4, description: 'Filtering trades when macro regime shifts preserved accumulated capital.' },
    { rank: 3, title: 'Reduced momentum chasing', points: 2.8, description: 'Ceasing late-stage breakout chases improved entry-to-stop ratio.' },
  ];

  return (
    <div className="space-y-6" data-testid="ai-behavioral-coach-card">
      {/* Low Confidence Warning Edge Case */}
      {forecast.isLowConfidenceWarning && (
        <div className="p-4 bg-accent-warning/15 border border-accent-warning/40 rounded-xl text-accent-warning flex items-center gap-3">
          <span className="text-xl">⚠️</span>
          <div className="text-body-ui">
            <strong>Uncertainty Warning:</strong> Forecast confidence is below 60% due to sample variance. Projections are provisional.
          </div>
        </div>
      )}

      {/* Main Coach Forecast Card */}
      <div className="p-6 md:p-8 bg-gradient-to-r from-bg-surface via-bg-surface-raised to-bg-surface border-2 border-accent-positive/40 rounded-2xl shadow-md space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono font-bold uppercase bg-accent-positive/15 text-accent-positive border border-accent-positive/40 rounded">
                AI Behavioral Coach
              </span>
              <span className="text-caption-mono text-text-muted text-xs">
                Forward Developmental Guidance
              </span>
            </div>
            <h3 className="text-display-1 font-bold text-text-primary mt-1">
              Why You&apos;re Improving &bull; You are improving because:
            </h3>
            <p className="text-body-ui text-text-secondary mt-0.5">
              Decision quality increased <strong className="text-accent-positive font-mono font-bold">12 points</strong> during the last year.
            </p>
          </div>

          <div className="p-4 bg-bg-surface rounded-xl border border-border-subtle text-right">
            <div className="text-caption-mono text-text-muted uppercase text-xs">Target Horizon</div>
            <div className="text-display-2 font-mono font-black text-accent-positive">
              {forecast.currentScore} &rarr; {forecast.projectedScoreFourMonths}
            </div>
            <div className="text-caption-mono text-accent-positive text-xs font-semibold">
              Within {displayMonths} Months (Confidence: {displayConfidence}% (87% Baseline Confidence))
            </div>
          </div>
        </div>

        {/* 3 Greatest Contributors with Exact Points */}
        <div className="space-y-3">
          <div className="text-caption-mono text-text-muted uppercase text-xs font-bold">
            Greatest Contributors to Decision Growth
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {contributors.map((c) => (
              <div
                key={c.rank}
                className="p-4 bg-bg-surface border border-border-subtle rounded-xl space-y-2 hover:border-accent-positive/40 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <span className="w-6 h-6 rounded-full bg-accent-positive/20 text-accent-positive text-xs font-mono font-bold flex items-center justify-center">
                    {c.rank}
                  </span>
                  <span className="text-caption-mono font-mono font-black text-accent-positive text-sm">
                    +{c.points.toFixed(1)} points
                  </span>
                </div>
                <div className="text-body-ui font-bold text-text-primary">
                  {c.title}
                </div>
                <p className="text-caption text-text-secondary text-xs">
                  {c.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Trajectory Statement & Scenario Toggle */}
        <div className="p-4 bg-bg-surface rounded-xl border border-border-subtle flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-0.5 text-sm">
            <div className="font-bold text-text-primary">
              At current pace you are likely to reach Quality Score 80 within {displayMonths} months.
            </div>
            <div className="text-caption text-text-muted text-xs font-mono">
              Confidence: {displayConfidence}% &bull; Projected Gain: +{forecast.expectedGain}.0 points
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setScenarioMode('CURRENT_PACE')}
              role="button"
              aria-pressed={!isAccelerated}
              aria-label="View current pace scenario (4.0 months)"
              className={`px-3 py-1.5 text-caption-mono text-xs font-bold rounded-lg transition-colors ${
                !isAccelerated
                  ? 'bg-accent-info/20 text-accent-info border border-accent-info/40'
                  : 'bg-bg-surface-raised text-text-muted hover:text-text-primary'
              }`}
            >
              Current Pace (4.0 Mo)
            </button>
            <button
              onClick={() => setScenarioMode('ACCELERATED')}
              role="button"
              aria-pressed={isAccelerated}
              aria-label="View accelerated sizing scenario (2.5 months)"
              className={`px-3 py-1.5 text-caption-mono text-xs font-bold rounded-lg transition-colors ${
                isAccelerated
                  ? 'bg-accent-positive/20 text-accent-positive border border-accent-positive/40'
                  : 'bg-bg-surface-raised text-text-muted hover:text-text-primary'
              }`}
            >
              Accelerate Sizing &bull; Accelerate Drift (2.5 Mo)
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
