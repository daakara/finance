'use client';

import React from 'react';
import { ExecutiveJourneyAnalyticsData } from '@/types/phase27-observability';
import { EXECUTIVE_JOURNEY_ANALYTICS } from '@/lib/telemetry/productionObservabilityEngine';

interface ExecutiveUsageAnalyticsProps {
  data?: ExecutiveJourneyAnalyticsData;
}

export default function ExecutiveUsageAnalytics({
  data = EXECUTIVE_JOURNEY_ANALYTICS,
}: ExecutiveUsageAnalyticsProps) {
  return (
    <div className="space-y-6" data-testid="executive-usage-analytics">
      {/* Top Header & Key Speed Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 bg-bg-surface border border-border-subtle rounded-xl">
          <div className="text-caption-mono text-text-muted uppercase">Executive Sessions</div>
          <div className="text-display-2 font-mono font-bold text-text-primary mt-1">
            {data.totalExecutiveSessions}
          </div>
          <div className="text-caption text-text-secondary mt-1">
            30-Day Cohort Tracking
          </div>
        </div>

        <div className="p-4 bg-bg-surface border border-border-subtle rounded-xl">
          <div className="text-caption-mono text-text-muted uppercase">Avg Session Dwell</div>
          <div className="text-display-2 font-mono font-bold text-text-primary mt-1">
            {data.avgSessionDurationMin} <span className="text-body-ui font-normal text-text-muted">min</span>
          </div>
          <div className="text-caption text-accent-positive mt-1">
            Active Attention Focus
          </div>
        </div>

        <div className="p-4 bg-bg-surface border border-accent-positive/30 rounded-xl">
          <div className="text-caption-mono text-accent-positive uppercase font-bold">CEO Speed Test</div>
          <div className="text-display-2 font-mono font-bold text-accent-positive mt-1">
            48 <span className="text-body-ui font-normal text-text-muted">sec</span>
          </div>
          <div className="text-caption text-text-secondary mt-1">
            Target: &le; 120s (Briefing to Action)
          </div>
        </div>

        <div className="p-4 bg-bg-surface border border-border-subtle rounded-xl">
          <div className="text-caption-mono text-text-muted uppercase">End-to-End Retention</div>
          <div className="text-display-2 font-mono font-bold text-accent-positive mt-1">
            76.4%
          </div>
          <div className="text-caption text-text-secondary mt-1">
            Command Center &rarr; Playbook
          </div>
        </div>
      </div>

      {/* Funnel Drop-off Diagram */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-6">
        <div>
          <h3 className="text-header-1 text-text-primary">
            Executive Journey Funnel &amp; Drop-off Analysis
          </h3>
          <p className="text-body-ui text-text-secondary mt-1">
            Step-by-step funnel measuring executive navigation from Morning Briefing through to
            the Personal Decision Playbook.
          </p>
        </div>

        <div className="space-y-4">
          {data.funnelSteps.map((step, idx) => {
            const widthPct = Math.max(step.conversionRate, 15);
            return (
              <div key={idx} className="space-y-1.5">
                <div className="flex items-center justify-between text-body-ui">
                  <div className="flex items-center gap-3">
                    <span className="w-6 h-6 rounded-full bg-accent-info/20 text-accent-info border border-accent-info/40 text-xs font-mono font-bold flex items-center justify-center">
                      {step.stepNumber}
                    </span>
                    <span className="font-semibold text-text-primary">{step.name}</span>
                  </div>

                  <div className="flex items-center gap-4 text-caption-mono">
                    <span className="text-text-muted">
                      Dwell: <strong className="text-text-secondary">{step.avgDwellTimeSec}s</strong>
                    </span>
                    <span className="text-text-muted">
                      Visitors: <strong className="text-text-primary">{step.visitors}</strong>
                    </span>
                    <span className="text-accent-positive font-bold text-body-ui">
                      {step.conversionRate.toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Progress Visual with Drop-off indicator */}
                <div className="relative w-full bg-bg-surface-raised h-8 rounded-lg overflow-hidden border border-border-subtle flex items-center px-3">
                  <div
                    className="absolute left-0 top-0 bottom-0 bg-gradient-to-r from-accent-info/30 to-accent-positive/40 transition-all duration-500 rounded-lg"
                    style={{ width: `${widthPct}%` }}
                  />
                  <div className="relative z-10 flex items-center justify-between w-full text-caption-mono font-semibold">
                    <span className="text-text-primary">Step {step.stepNumber}</span>
                    {step.dropOffRate > 0 && (
                      <span className="text-accent-warning text-xs">
                        Drop-off: -{step.dropOffRate.toFixed(1)}%
                      </span>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Friction Points & Mitigations */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4">
        <h3 className="text-header-2 text-text-primary">
          Identified Friction Points &amp; Applied UX Resolutions
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {data.frictionPoints.map((pt, idx) => (
            <div
              key={idx}
              className="p-4 bg-bg-surface-raised rounded-lg border border-border-subtle space-y-2"
            >
              <div className="flex items-center justify-between">
                <span className="text-body-ui font-semibold text-text-primary">
                  {pt.location}
                </span>
                <span className="px-2 py-0.5 text-[10px] font-mono font-bold uppercase bg-accent-info/10 text-accent-info border border-accent-info/30 rounded">
                  {pt.severity} Severity
                </span>
              </div>
              <p className="text-caption text-text-secondary">
                {pt.description}
              </p>
              <div className="pt-2 border-t border-border-subtle text-caption font-mono text-accent-positive">
                Resolution: {pt.resolution}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
