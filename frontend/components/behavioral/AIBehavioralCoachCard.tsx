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
      <div className="p-6 bg-gradient-to-r from-bg-surface via-bg-surface-raised to-bg-surface border-2 border-accent-positive/40 rounded-2xl shadow-md space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono font-bold uppercase bg-accent-positive/15 text-accent-positive border border-accent-positive/40 rounded">
                AI Behavioral Coach
              </span>
              <span className="text-caption-mono text-text-muted">
                Forward Trajectory Guidance
              </span>
            </div>
            <h3 className="text-display-2 font-bold text-text-primary mt-1">
              Personal Developmental Trajectory
            </h3>
            <p className="text-body-ui text-text-secondary mt-0.5">
              Empirical forecast projecting score evolution toward Institutional Excellence (80+).
            </p>
          </div>

          <div className="p-4 bg-bg-surface rounded-xl border border-border-subtle text-right">
            <div className="text-caption-mono text-text-muted uppercase text-xs">Target Milestone</div>
            <div className="text-display-1 font-mono font-extrabold text-accent-positive">
              {forecast.currentScore} &rarr; {forecast.projectedScoreFourMonths}
            </div>
            <div className="text-caption-mono text-accent-positive text-xs font-semibold">
              Within {displayMonths} Months (Confidence: {displayConfidence}%)
            </div>
          </div>
        </div>

        {/* Why you are improving */}
        <div className="p-5 bg-bg-surface rounded-xl border border-border-subtle space-y-3">
          <h4 className="text-header-2 text-text-primary">
            You are improving because:
          </h4>
          <ol className="space-y-2 text-body-ui">
            {forecast.keyCatalysts.map((catalyst, idx) => (
              <li key={idx} className="flex items-start gap-3">
                <span className="w-5 h-5 rounded-full bg-accent-positive/20 text-accent-positive border border-accent-positive/40 text-xs font-mono font-bold flex items-center justify-center mt-0.5">
                  {idx + 1}
                </span>
                <span className="text-text-primary font-medium">{catalyst}</span>
              </li>
            ))}
          </ol>
        </div>

        {/* Scenario Toggle */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pt-2">
          <div className="text-caption text-text-secondary">
            Projected gain: <strong className="text-accent-positive">+{forecast.expectedGain} points</strong> on current trajectory.
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setScenarioMode('CURRENT_PACE')}
              className={`px-3 py-1.5 text-caption-mono font-bold rounded-lg transition-colors ${
                !isAccelerated
                  ? 'bg-accent-info/20 text-accent-info border border-accent-info/40'
                  : 'bg-bg-surface-raised text-text-muted hover:text-text-primary'
              }`}
            >
              Current Pace (4.0 Mo)
            </button>
            <button
              onClick={() => setScenarioMode('ACCELERATED')}
              className={`px-3 py-1.5 text-caption-mono font-bold rounded-lg transition-colors ${
                isAccelerated
                  ? 'bg-accent-positive/20 text-accent-positive border border-accent-positive/40'
                  : 'bg-bg-surface-raised text-text-muted hover:text-text-primary'
              }`}
            >
              Accelerate Drift &le; 15% (2.5 Mo)
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
