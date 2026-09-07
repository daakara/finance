'use client';

import React from 'react';
import { MetricWithConfidence } from '@/types/behavioral-intelligence';

interface ConfidenceBandMetricProps {
  metric: MetricWithConfidence;
}

export default function ConfidenceBandMetric({ metric }: ConfidenceBandMetricProps) {
  const { name, value, unit, ci, trend, delta7d, target } = metric;

  // Calculate relative position within a 50% - 100% display window
  const windowMin = 50;
  const windowMax = 100;
  const windowSpan = windowMax - windowMin;

  const leftPct = Math.max(0, Math.min(100, ((ci.lowerBound - windowMin) / windowSpan) * 100));
  const widthPct = Math.max(4, Math.min(100 - leftPct, ((ci.upperBound - ci.lowerBound) / windowSpan) * 100));
  const pointPct = Math.max(0, Math.min(100, ((ci.pointEstimate - windowMin) / windowSpan) * 100));

  return (
    <div className="p-4 bg-bg-surface rounded-xl border border-border-subtle space-y-3" data-testid="confidence-band-metric">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-body-ui font-semibold text-text-primary block">{name}</span>
          <span className="text-caption text-text-muted">Target: &ge; {target.toFixed(1)}{unit}</span>
        </div>

        <div className="text-right">
          <div className="text-header-1 font-mono font-bold text-accent-positive">
            {value.toFixed(1)}{unit}
          </div>
          <span
            className={`px-2 py-0.5 text-[10px] font-mono font-bold uppercase rounded ${
              trend === 'RAPID_IMPROVEMENT' || trend === 'IMPROVING'
                ? 'bg-accent-positive/15 text-accent-positive border border-accent-positive/30'
                : trend === 'STABLE'
                ? 'bg-accent-info/15 text-accent-info border border-accent-info/30'
                : 'bg-accent-warning/15 text-accent-warning border border-accent-warning/30'
            }`}
          >
            ▲ {delta7d > 0 ? `+${delta7d.toFixed(1)}%` : `${delta7d.toFixed(1)}%`} ({trend})
          </span>
        </div>
      </div>

      {/* Visual Confidence Band */}
      <div className="space-y-1">
        <div className="flex justify-between text-[11px] font-mono text-text-muted">
          <span>95% Confidence Interval</span>
          <span className="text-text-secondary">{ci.displayString}</span>
        </div>

        {/* Graphical Representation: [━━━━━━●━━━━━━] */}
        <div className="relative w-full bg-bg-surface-raised h-3 rounded-full overflow-hidden border border-border-subtle">
          {/* Shaded interval band */}
          <div
            className="absolute top-0 bottom-0 bg-accent-positive/25 rounded-full"
            style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
          />
          {/* Point estimate dot */}
          <div
            className="absolute top-0 bottom-0 w-2.5 h-2.5 -ml-1 rounded-full bg-accent-positive border border-bg-surface my-auto"
            style={{ left: `${pointPct}%` }}
          />
        </div>

        <div className="flex justify-between text-[10px] font-mono text-text-muted pt-0.5">
          <span>{windowMin}%</span>
          <span>Bound: [{ci.lowerBound.toFixed(1)}% &bull; {ci.upperBound.toFixed(1)}%]</span>
          <span>{windowMax}%</span>
        </div>
      </div>
    </div>
  );
}
