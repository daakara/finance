'use client';

import React from 'react';
import { CANONICAL_MORNING_BRIEFING_V2 } from '@/lib/telemetry/behavioralStoryEngine';
import { MorningBriefingV2Story } from '@/types/behavioral-intelligence';

interface MorningBriefingV2Props {
  story?: MorningBriefingV2Story;
}

export default function MorningBriefingV2({
  story = CANONICAL_MORNING_BRIEFING_V2,
}: MorningBriefingV2Props) {
  return (
    <div className="space-y-6" data-testid="morning-briefing-v2">
      {/* Staleness Banner Edge Case */}
      {story.isFeedStale && (
        <div className="p-4 bg-accent-warning/15 border border-accent-warning/40 rounded-xl text-accent-warning flex items-center gap-3">
          <span className="text-xl">⚠️</span>
          <div className="text-body-ui">
            <strong>Market Feeds Unavailable:</strong> Displaying last known market state. Recommendations are suppressed until feed continuity is verified.
          </div>
        </div>
      )}

      {/* Top Banner: Market Context & Risk Shift */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono font-bold uppercase bg-accent-info/10 text-accent-info border border-accent-info/30 rounded">
                Morning Briefing 2.0
              </span>
              <span className="text-caption-mono text-text-muted">
                Narrative &rarr; Insight &rarr; Action
              </span>
            </div>
            <h3 className="text-display-2 font-bold text-text-primary mt-1">
              Overnight Market Risk Context
            </h3>
            <p className="text-body-ui text-text-secondary mt-0.5">
              {story.narrativeSummary}
            </p>
          </div>

          <div className="flex items-center gap-4 p-4 bg-bg-surface-raised rounded-xl border border-border-subtle">
            <div>
              <div className="text-[11px] font-mono text-text-muted uppercase">Market Risk Score</div>
              <div className="flex items-baseline gap-2 mt-0.5">
                <span className="text-display-2 font-mono font-bold text-text-muted line-through">
                  {story.marketRiskScorePrev}
                </span>
                <span className="text-display-1 font-mono font-extrabold text-accent-warning">
                  &rarr; {story.marketRiskScoreCurrent}
                </span>
              </div>
              <div className="text-caption text-accent-warning font-semibold mt-0.5">
                Risk Environment Deteriorated
              </div>
            </div>
          </div>
        </div>

        {/* 4-Step Narrative UX Flow */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 pt-4 border-t border-border-subtle">
          <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
            <div className="text-[11px] font-mono text-accent-info uppercase font-bold">1. Market Changed</div>
            <div className="text-body-ui font-semibold text-text-primary mt-1">SOX &amp; Tech Pullback</div>
            <div className="text-caption text-text-muted mt-0.5">Treasury flight to safety</div>
          </div>

          <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
            <div className="text-[11px] font-mono text-accent-info uppercase font-bold">2. Why It Matters</div>
            <div className="text-body-ui font-semibold text-text-primary mt-1">Beta Headwinds</div>
            <div className="text-caption text-text-muted mt-0.5">Growth breadth narrowed to 32%</div>
          </div>

          <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
            <div className="text-[11px] font-mono text-accent-info uppercase font-bold">3. What Is Affected</div>
            <div className="text-body-ui font-semibold text-text-primary mt-1">3 Exposed Holdings</div>
            <div className="text-caption text-accent-warning font-semibold mt-0.5">${(story.totalCapitalAtRisk / 1000).toFixed(0)}K Capital at Risk</div>
          </div>

          <div className="p-3 bg-bg-surface-raised rounded-lg border border-accent-positive/40">
            <div className="text-[11px] font-mono text-accent-positive uppercase font-bold">4. What To Do</div>
            <div className="text-body-ui font-semibold text-accent-positive mt-1">Tighten Regime Gates</div>
            <div className="text-caption text-text-muted mt-0.5">Confidence: {story.recommendationConfidence}%</div>
          </div>
        </div>
      </div>

      {/* Market Shifts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {story.marketShifts.map((shift, idx) => (
          <div key={idx} className="p-4 bg-bg-surface rounded-xl border border-border-subtle space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-caption-mono font-bold text-text-primary">{shift.dimension}</span>
              <span
                className={`px-2 py-0.5 text-[10px] font-mono font-bold uppercase rounded ${
                  shift.direction === 'STRENGTHENED'
                    ? 'bg-accent-positive/15 text-accent-positive border border-accent-positive/30'
                    : 'bg-accent-warning/15 text-accent-warning border border-accent-warning/30'
                }`}
              >
                {shift.direction}
              </span>
            </div>
            <p className="text-caption text-text-secondary">{shift.detail}</p>
          </div>
        ))}
      </div>

      {/* Affected Positions Table */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4">
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <div>
            <h4 className="text-header-1 text-text-primary">
              Your Exposure &amp; Impacted Positions
            </h4>
            <p className="text-body-ui text-text-secondary mt-0.5">
              Positions currently violating preferred macro regime filters based on overnight shift.
            </p>
          </div>
          <div className="text-right">
            <span className="text-caption-mono text-text-muted text-xs block">Estimated Capital At Risk</span>
            <span className="text-header-1 font-mono font-bold text-accent-warning">
              ${story.totalCapitalAtRisk.toLocaleString()}
            </span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-body-ui font-mono text-xs">
            <thead>
              <tr className="border-b border-border-subtle text-text-muted">
                <th className="pb-2">Ticker</th>
                <th className="pb-2">Shares</th>
                <th className="pb-2">Price</th>
                <th className="pb-2">Capital at Risk</th>
                <th className="pb-2">Violated Condition</th>
                <th className="pb-2 text-right">Suggested Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-subtle">
              {story.affectedPositions.map((pos, idx) => (
                <tr key={idx}>
                  <td className="py-3 font-sans font-bold text-text-primary text-sm">{pos.symbol}</td>
                  <td className="py-3 text-text-secondary">{pos.shares}</td>
                  <td className="py-3 text-text-secondary">${pos.currentPrice.toFixed(2)}</td>
                  <td className="py-3 text-accent-warning font-bold">${pos.capitalAtRisk.toLocaleString()}</td>
                  <td className="py-3 text-text-secondary font-sans">{pos.violatedCondition}</td>
                  <td className="py-3 text-right">
                    <span className="px-2.5 py-1 text-[11px] font-bold rounded bg-accent-info/15 text-accent-info border border-accent-info/30">
                      {pos.suggestedAction}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
