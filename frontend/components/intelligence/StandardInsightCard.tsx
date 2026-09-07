'use client';

import React from 'react';
import { StandardInsightData } from '@/types/ux-foundations';

interface StandardInsightCardProps {
  insight: StandardInsightData;
  onClick?: (insight: StandardInsightData) => void;
}

const CATEGORY_COLORS: Record<string, string> = {
  FLOW: 'bg-emerald-950/40 text-emerald-300 border-emerald-800/60',
  REGIME: 'bg-indigo-950/40 text-indigo-300 border-indigo-800/60',
  MOMENTUM: 'bg-cyan-950/40 text-cyan-300 border-cyan-800/60',
  VOLATILITY: 'bg-amber-950/40 text-amber-300 border-amber-800/60',
  VALUATION: 'bg-purple-950/40 text-purple-300 border-purple-800/60',
  BEHAVIOR: 'bg-rose-950/40 text-rose-300 border-rose-800/60',
};

export const StandardInsightCard: React.FC<StandardInsightCardProps> = ({
  insight,
  onClick,
}) => {
  return (
    <article
      role="region"
      aria-label={`Insight: ${insight.headline}`}
      onClick={() => onClick && onClick(insight)}
      className="bg-slate-900 border border-slate-800 hover:border-slate-700 rounded-xl p-4 transition-all duration-150 shadow-sm hover:shadow-md cursor-pointer group focus-within:ring-2 focus-within:ring-cyan-500"
    >
      {/* Header: Category, Ticker, Confidence, Delta */}
      <div className="flex items-center justify-between gap-2 mb-2.5">
        <div className="flex items-center space-x-2">
          {insight.ticker && (
            <span className="font-mono font-bold text-sm text-white px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
              {insight.ticker}
            </span>
          )}
          <span
            className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded border ${
              CATEGORY_COLORS[insight.category] || CATEGORY_COLORS.FLOW
            }`}
          >
            {insight.category}
          </span>
        </div>

        <div className="flex items-center space-x-2">
          {insight.metricDelta && (
            <span
              className={`text-xs font-mono font-semibold px-2 py-0.5 rounded ${
                insight.metricDelta.isPositive
                  ? 'bg-emerald-950/50 text-emerald-400 border border-emerald-800/40'
                  : 'bg-rose-950/50 text-rose-400 border border-rose-800/40'
              }`}
            >
              {insight.metricDelta.label}: {insight.metricDelta.value}
            </span>
          )}
          <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
            {insight.confidence}% Conf.
          </span>
        </div>
      </div>

      {/* Body: Headline & Observation */}
      <h3 className="text-sm font-semibold text-white group-hover:text-cyan-300 transition-colors">
        {insight.headline}
      </h3>
      <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
        {insight.observation}
      </p>

      {/* Footer: Provenance & Attribution */}
      <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex items-center justify-between text-[10px] text-slate-400 font-mono">
        <span>Source: {insight.source}</span>
        <span>{insight.timestamp}</span>
      </div>
    </article>
  );
};
