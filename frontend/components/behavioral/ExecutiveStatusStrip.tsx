'use client';

import React from 'react';

interface ExecutiveStatusStripProps {
  marketState?: string;
  portfolioStatus?: string;
  attentionCount?: number;
  criticalAlert?: string;
}

export default function ExecutiveStatusStrip({
  marketState = 'Weakening Growth · Flight to Quality',
  portfolioStatus = '3 Positions Require Review',
  attentionCount = 3,
  criticalAlert = 'Macro Invalidation Triggered',
}: ExecutiveStatusStripProps) {
  return (
    <div
      className="min-h-[72px] p-3 md:px-6 bg-bg-surface border border-border-subtle rounded-xl flex flex-wrap items-center justify-between gap-4 shadow-sm"
      data-testid="executive-status-strip"
    >
      {/* 1. Market State */}
      <div className="flex items-center gap-3">
        <div className="w-2.5 h-2.5 rounded-full bg-accent-warning animate-pulse flex-shrink-0" />
        <div>
          <div className="text-caption-mono text-text-muted uppercase text-[10px]">
            Market Regime
          </div>
          <div className="text-body-ui font-bold text-text-primary text-xs md:text-sm">
            {marketState}
          </div>
        </div>
      </div>

      <div className="hidden lg:block h-8 w-px bg-border-subtle" />

      {/* 2. Portfolio Status */}
      <div className="flex items-center gap-3">
        <span className="w-6 h-6 rounded bg-accent-risk/15 text-accent-risk border border-accent-risk/30 flex items-center justify-center font-mono font-bold text-xs flex-shrink-0">
          !
        </span>
        <div>
          <div className="text-caption-mono text-text-muted uppercase text-[10px]">
            Portfolio Status
          </div>
          <div className="text-body-ui font-semibold text-text-primary text-xs md:text-sm">
            {portfolioStatus}
          </div>
        </div>
      </div>

      <div className="hidden lg:block h-8 w-px bg-border-subtle" />

      {/* 3. Attention Count */}
      <div className="flex items-center gap-3">
        <div className="px-2 py-0.5 rounded bg-bg-surface-raised border border-border-subtle font-mono text-xs font-bold text-accent-info">
          {attentionCount}
        </div>
        <div>
          <div className="text-caption-mono text-text-muted uppercase text-[10px]">
            Attention Queue
          </div>
          <div className="text-body-ui font-medium text-text-secondary text-xs md:text-sm">
            Urgent Items Pending
          </div>
        </div>
      </div>

      <div className="hidden lg:block h-8 w-px bg-border-subtle" />

      {/* 4. Critical Alert */}
      <div className="flex items-center gap-2 bg-accent-risk/10 px-3 py-1.5 rounded-lg border border-accent-risk/30">
        <span className="w-2 h-2 rounded-full bg-accent-risk" />
        <span className="text-caption-mono font-bold text-accent-risk text-xs">
          {criticalAlert}
        </span>
      </div>
    </div>
  );
}
