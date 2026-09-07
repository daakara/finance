'use client';

import React from 'react';
import {
  DIRQuarterlyHistory,
  DIRPeerBenchmark,
} from '@/types/dir-framework';
import {
  CANONICAL_DIR_HISTORY,
  CANONICAL_DIR_PEER_BENCHMARKS,
} from '@/lib/telemetry/dirEngine';

interface DIREvolutionTimelineProps {
  history?: DIRQuarterlyHistory[];
  benchmarks?: DIRPeerBenchmark[];
  currentScore?: number;
}

export default function DIREvolutionTimeline({
  history = CANONICAL_DIR_HISTORY,
  benchmarks = CANONICAL_DIR_PEER_BENCHMARKS,
  currentScore = 63,
}: DIREvolutionTimelineProps) {
  return (
    <div className="space-y-6" data-testid="dir-evolution-timeline">
      {/* 1. Header */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono font-bold uppercase tracking-wider text-xs rounded bg-accent-positive/10 text-accent-positive border border-accent-positive/30">
                Longitudinal Tracking
              </span>
              <span className="text-caption-mono text-text-muted text-xs">
                4 Quarters Monitored
              </span>
            </div>
            <h2 className="text-header-1 font-bold text-text-primary mt-1">
              DIR Evolution &amp; Peer Cohort Benchmarks
            </h2>
            <p className="text-body-ui text-text-secondary text-sm">
              Demonstrates consistent positive velocity across 4 consecutive quarters (+8.0 pts total gain).
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="px-4 py-2 bg-bg-surface-raised border border-border-subtle rounded-xl text-center">
              <div className="text-caption-mono text-text-muted text-[11px] uppercase">Annual Delta</div>
              <div className="text-header-2 font-mono font-bold text-accent-positive">+8.0 pts</div>
            </div>
            <div className="px-4 py-2 bg-bg-surface-raised border border-border-subtle rounded-xl text-center">
              <div className="text-caption-mono text-text-muted text-[11px] uppercase">Target Milestone</div>
              <div className="text-header-2 font-mono font-bold text-accent-info">75.0</div>
            </div>
          </div>
        </div>

        {/* 2. Timeline Step Progression */}
        <div className="space-y-4 pt-2">
          <h3 className="text-caption-mono text-text-muted font-bold uppercase text-xs">
            Quarterly Trajectory Progression
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {history.map((item, index) => {
              const isCurrent = index === history.length - 1;
              return (
                <div
                  key={item.quarter}
                  className={`p-4 rounded-xl border relative space-y-3 transition-all ${
                    isCurrent
                      ? 'bg-accent-positive/5 border-accent-positive/40 shadow-sm'
                      : 'bg-bg-surface-raised border-border-subtle'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-caption-mono font-bold text-xs text-text-muted uppercase">
                      {item.quarter}
                    </span>
                    {isCurrent && (
                      <span className="px-2 py-0.5 text-caption-mono text-[10px] font-bold rounded bg-accent-positive/20 text-accent-positive">
                        Current
                      </span>
                    )}
                  </div>

                  <div className="flex items-baseline gap-2">
                    <span
                      className={`text-display-2 font-mono font-black ${
                        isCurrent ? 'text-accent-positive' : 'text-text-primary'
                      }`}
                    >
                      {item.dirScore}
                    </span>
                    {item.delta > 0 && (
                      <span className="text-body-ui font-mono font-bold text-accent-positive text-sm">
                        +{item.delta}
                      </span>
                    )}
                  </div>

                  <div className="space-y-1 text-xs">
                    <div className="text-text-muted font-medium">
                      DQS Score: <span className="text-text-primary font-mono font-semibold">{item.dqs}</span>
                    </div>
                    <div className="text-text-primary font-medium text-xs line-clamp-2">
                      {item.dominantBehavior}
                    </div>
                    <div className="text-text-muted text-[11px] pt-1 border-t border-border-subtle/60">
                      Milestone: {item.keyMilestone}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* 3. Peer Cohort Comparison Matrix */}
        <div className="pt-4 border-t border-border-subtle space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-header-2 font-bold text-text-primary">
              Peer Cohort Comparison
            </h3>
            <span className="text-caption text-text-muted text-xs">
              Calibrated against 549 institutional portfolios
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {benchmarks.map((peer) => {
              const delta = currentScore - peer.avgDIR;
              const isAhead = delta >= 0;
              return (
                <div
                  key={peer.cohortName}
                  className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-caption-mono text-text-muted text-xs font-bold truncate">
                      {peer.cohortName}
                    </span>
                    <span className="text-caption text-text-muted text-[10px]">
                      n={peer.sampleSize}
                    </span>
                  </div>

                  <div className="flex items-baseline justify-between">
                    <span className="text-display-2 font-mono font-black text-text-primary">
                      {peer.avgDIR}
                    </span>
                    <span
                      className={`text-caption-mono font-mono font-bold text-xs ${
                        isAhead ? 'text-accent-positive' : 'text-accent-warning'
                      }`}
                    >
                      {isAhead ? `+${delta}` : `${delta}`} vs you
                    </span>
                  </div>

                  <div className="w-full bg-bg-surface h-1.5 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min(100, (peer.avgDIR / 100) * 100)}%`,
                        backgroundColor: peer.colorHex,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* 4. AI Coach Next Horizon Prescription */}
        <div className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <span className="text-caption-mono text-accent-info font-bold uppercase text-xs">
              AI Coach Horizon Target
            </span>
            <div className="text-body-ui font-bold text-text-primary text-sm">
              Path to 75 (High Performer) within 180 days:
            </div>
            <p className="text-caption text-text-secondary text-xs max-w-2xl">
              Compress behavioral drift on winning trades from 21.0% to &lt; 15.0%, and maintain current stop-loss execution discipline (91%).
            </p>
          </div>

          <div className="flex-shrink-0">
            <span className="px-3 py-1.5 text-caption-mono text-xs font-bold rounded-lg bg-accent-positive/15 text-accent-positive border border-accent-positive/30">
              Trajectory: On Track
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
