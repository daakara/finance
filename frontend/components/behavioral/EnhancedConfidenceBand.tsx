'use client';

import React from 'react';
import { MetricWithConfidence } from '@/types/behavioral-intelligence';

interface EnhancedConfidenceBandProps {
  metric: MetricWithConfidence;
  sampleSize?: number;
}

export default function EnhancedConfidenceBand({
  metric,
  sampleSize = 42,
}: EnhancedConfidenceBandProps) {
  const { name, value, unit, ci, target, delta30d } = metric;
  const { lowerBound, upperBound, marginOfError, confidenceLevel } = ci;

  // Strict sample size guard: minimum n >= 30 for reliable statistical bounds
  if (sampleSize < 30) {
    return (
      <div
        className="p-5 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-2 opacity-80"
        data-testid="confidence-band-insufficient"
      >
        <div className="flex items-center justify-between">
          <span className="text-body-ui font-semibold text-text-primary text-sm">
            {name}
          </span>
          <span className="px-2 py-0.5 text-caption-mono text-[10px] font-bold rounded bg-accent-warning/15 text-accent-warning border border-accent-warning/30">
            Data Insufficient (n &lt; 30)
          </span>
        </div>
        <div className="text-display-2 font-mono font-black text-text-muted">
          {value}{unit}
        </div>
        <p className="text-caption text-text-muted text-xs">
          Minimum 30 observations required for Wilson score statistical interval.
        </p>
      </div>
    );
  }

  // Three-Tier Color Coding
  // 1. Green: entire interval above target (lowerBound > target)
  // 2. Yellow: target is inside interval (lowerBound <= target && target <= upperBound)
  // 3. Red: entire interval below target (upperBound < target)
  let statusColor = 'text-accent-positive';
  let statusBg = 'bg-accent-positive';
  let statusBorder = 'border-accent-positive/30';
  let statusLabel = 'Above Target (Strong)';

  if (upperBound < target) {
    statusColor = 'text-accent-negative';
    statusBg = 'bg-accent-negative';
    statusBorder = 'border-accent-negative/30';
    statusLabel = 'Below Target (Deficit)';
  } else if (lowerBound <= target && target <= upperBound) {
    statusColor = 'text-accent-warning';
    statusBg = 'bg-accent-warning';
    statusBorder = 'border-accent-warning/30';
    statusLabel = 'Target In Interval (Indeterminate)';
  }

  // Normalize positions for a 0-100 scale visual bar
  const leftPercent = Math.max(0, Math.min(100, lowerBound));
  const widthPercent = Math.max(2, Math.min(100 - leftPercent, upperBound - lowerBound));
  const pointPercent = Math.max(0, Math.min(100, value));
  const targetPercent = Math.max(0, Math.min(100, target));

  return (
    <div
      className="p-5 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-3"
      data-testid="enhanced-confidence-band"
    >
      <div className="flex items-center justify-between">
        <span className="text-body-ui font-bold text-text-primary text-sm">
          {name}
        </span>
        <span
          className={`px-2 py-0.5 text-caption-mono text-[10px] font-bold rounded border ${statusColor} ${statusBorder} bg-bg-surface`}
        >
          {statusLabel}
        </span>
      </div>

      <div className="flex items-baseline justify-between">
        <div className="flex items-baseline gap-2">
          <span className={`text-display-2 font-mono font-black ${statusColor}`}>
            {value}{unit}
          </span>
          <span className="text-caption-mono text-text-muted text-xs">
            Target: {target}{unit}
          </span>
        </div>

        <div className="text-right">
          <div className="text-caption-mono text-text-secondary text-xs">
            {delta30d >= 0 ? `▲ +${delta30d}%` : `▼ ${delta30d}%`} 30d
          </div>
          <div className="text-caption text-text-muted text-[10px]">
            ±{marginOfError.toFixed(1)}% MoE
          </div>
        </div>
      </div>

      {/* Visual Confidence Band */}
      <div className="space-y-1.5 pt-1">
        <div className="relative h-4 bg-bg-surface rounded-md border border-border-subtle overflow-hidden">
          {/* CI Range Area */}
          <div
            className={`absolute top-1 bottom-1 rounded opacity-30 ${statusBg}`}
            style={{
              left: `${leftPercent}%`,
              width: `${widthPercent}%`,
            }}
          />

          {/* Point Estimate Marker */}
          <div
            className={`absolute top-0.5 bottom-0.5 w-1.5 rounded-full -ml-0.5 ${statusBg}`}
            style={{ left: `${pointPercent}%` }}
            title={`Point estimate: ${value}${unit}`}
          />

          {/* Target Benchmark Line */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-text-primary/70 -ml-[1px]"
            style={{ left: `${targetPercent}%` }}
            title={`Target: ${target}${unit}`}
          />
        </div>

        <div className="flex items-center justify-between text-[11px] font-mono text-text-muted">
          <span>{lowerBound.toFixed(1)}% (Lower)</span>
          <span className="text-text-secondary font-semibold">{(confidenceLevel * 100).toFixed(0)}% CI</span>
          <span>{upperBound.toFixed(1)}% (Upper)</span>
        </div>
      </div>
    </div>
  );
}
