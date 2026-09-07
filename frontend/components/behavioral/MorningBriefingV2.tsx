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

      {/* Story-Based Flow Hero Card (Wireframe 2) */}
      <div className="p-6 md:p-8 bg-gradient-to-br from-bg-surface-raised via-bg-surface to-bg-surface-raised border border-border-subtle rounded-2xl space-y-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border-subtle pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono font-bold uppercase bg-accent-info/10 text-accent-info border border-accent-info/30 rounded">
                Morning Briefing 2.0
              </span>
              <span className="text-caption-mono text-text-muted text-xs">
                Context &bull; Meaning &bull; Impact &bull; Action
              </span>
            </div>
            <h2 className="text-display-1 font-black text-text-primary mt-1">
              GOOD MORNING
            </h2>
          </div>

          <div className="flex items-center gap-3 p-3 bg-bg-surface rounded-xl border border-border-subtle">
            <div className="text-right">
              <div className="text-caption-mono text-text-muted uppercase text-[10px]">Market Risk Score</div>
              <div className="flex items-baseline gap-2 mt-0.5">
                <span className="text-display-2 font-mono font-bold text-text-muted line-through">42</span>
                <span className="text-display-2 font-mono font-black text-accent-warning">&rarr; 56</span>
              </div>
            </div>
          </div>
        </div>

        {/* 4 Story-Based Flow Blocks */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Block 1: OVERNIGHT STORY */}
          <div className="p-5 bg-bg-surface border border-border-subtle rounded-xl space-y-2">
            <div className="text-caption-mono text-accent-info uppercase font-bold text-xs">
              1. Market Changed &bull; OVERNIGHT STORY
            </div>
            <div className="space-y-1 text-caption text-text-secondary">
              <p className="font-semibold text-text-primary">Risk conditions deteriorated.</p>
              <p>Treasuries strengthened.</p>
              <p>Semiconductor leadership weakened.</p>
            </div>
            <div className="pt-2 border-t border-border-subtle text-[11px] font-mono text-accent-warning font-bold">
              Market Risk: 42 &rarr; 56
            </div>
          </div>

          {/* Block 2: WHY IT MATTERS */}
          <div className="p-5 bg-bg-surface border border-border-subtle rounded-xl space-y-2">
            <div className="text-caption-mono text-accent-info uppercase font-bold text-xs">
              WHY IT MATTERS
            </div>
            <p className="text-caption text-text-secondary leading-relaxed">
              Three of your current positions depend on growth leadership. These positions are now operating outside preferred conditions.
            </p>
            <div className="pt-2 border-t border-border-subtle text-[11px] font-mono text-text-muted">
              Breadth: 32% growth stocks above 50d MA
            </div>
          </div>

          {/* Block 3: IMPACTED POSITIONS */}
          <div className="p-5 bg-bg-surface border border-accent-warning/30 rounded-xl space-y-2">
            <div className="text-caption-mono text-accent-warning uppercase font-bold text-xs">
              IMPACTED POSITIONS
            </div>
            <div className="space-y-1 font-mono font-bold text-text-primary text-sm">
              <div>NVDA <span className="text-text-muted font-normal text-xs">(Semis)</span></div>
              <div>AMD <span className="text-text-muted font-normal text-xs">(Semis)</span></div>
              <div>CRWD <span className="text-text-muted font-normal text-xs">(Software)</span></div>
            </div>
            <div className="pt-2 border-t border-border-subtle text-[11px] font-mono text-accent-warning font-semibold">
              Estimated Capital At Risk: ${(story.totalCapitalAtRisk / 1000).toFixed(0)}K ($184,000 Total at Risk)
            </div>
          </div>

          {/* Block 4: RECOMMENDED ACTION */}
          <div className="p-5 bg-bg-surface border border-accent-positive/40 rounded-xl space-y-2 flex flex-col justify-between">
            <div className="space-y-1">
              <div className="text-caption-mono text-accent-positive uppercase font-bold text-xs">
                RECOMMENDED ACTION
              </div>
              <div className="text-body-ui font-bold text-text-primary text-sm">
                Reduce allocation by 15%.
              </div>
              <div className="text-caption text-text-secondary text-xs">
                Estimated downside protected: <strong className="text-accent-positive font-mono">${story.totalCapitalAtRisk.toLocaleString()}</strong>
              </div>
              <div className="text-[11px] font-mono text-text-muted">
                Confidence: {story.recommendationConfidence}%
              </div>
            </div>
            <button
              onClick={() => alert('Opening Macro Invalidation Review for NVDA, AMD, CRWD')}
              className="mt-2 w-full py-2 px-3 text-caption-mono font-bold text-xs rounded-lg bg-accent-positive text-bg-app hover:opacity-90 transition-opacity text-center"
            >
              [Review Now]
            </button>
          </div>
        </div>
      </div>

      {/* Affected Positions Breakdown Table */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm">
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <div>
            <h3 className="text-header-2 font-bold text-text-primary">
              Impacted Positions Detail
            </h3>
            <p className="text-body-ui text-text-secondary text-sm">
              Positions currently violating macro filters based on overnight risk shift.
            </p>
          </div>
          <div className="text-right">
            <span className="text-caption-mono text-text-muted text-xs block">Downside Risk Exposure</span>
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
                <th className="pb-2">Current Price</th>
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
