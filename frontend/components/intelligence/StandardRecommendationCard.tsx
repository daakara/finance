'use client';

import React, { useState } from 'react';
import { StandardRecommendationData, RecommendationCategory } from '@/types/ux-foundations';

interface StandardRecommendationCardProps {
  recommendation: StandardRecommendationData;
  onExecute?: (recommendation: StandardRecommendationData) => void;
  onDismiss?: (recommendation: StandardRecommendationData) => void;
}

const CATEGORY_STYLES: Record<RecommendationCategory, { label: string; badge: string; cardTint: string }> = {
  DO_MORE: {
    label: 'DO MORE',
    badge: 'bg-emerald-950/60 text-emerald-300 border-emerald-700/60',
    cardTint: 'border-emerald-500/20 bg-slate-900',
  },
  STOP_DOING: {
    label: 'STOP DOING',
    badge: 'bg-rose-950/60 text-rose-300 border-rose-700/60',
    cardTint: 'border-rose-500/20 bg-slate-900',
  },
  CALIBRATE: {
    label: 'CALIBRATE',
    badge: 'bg-amber-950/60 text-amber-300 border-amber-700/60',
    cardTint: 'border-amber-500/20 bg-slate-900',
  },
};

export const StandardRecommendationCard: React.FC<StandardRecommendationCardProps> = ({
  recommendation,
  onExecute,
  onDismiss,
}) => {
  const [executed, setExecuted] = useState(false);
  const style = CATEGORY_STYLES[recommendation.category] || CATEGORY_STYLES.CALIBRATE;

  const handleExecute = () => {
    setExecuted(true);
    if (onExecute) onExecute(recommendation);
  };

  return (
    <div
      role="region"
      aria-label={`Recommendation: ${recommendation.action}`}
      className={`border ${style.cardTint} rounded-xl p-4 shadow-sm transition-colors duration-150`}
    >
      {/* Top row: Category, Urgency, Projected Impact */}
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center space-x-2">
          {recommendation.ticker && (
            <span className="font-mono font-bold text-xs text-white px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
              {recommendation.ticker}
            </span>
          )}
          <span
            className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${style.badge}`}
          >
            {style.label}
          </span>
          <span
            className={`text-[10px] font-mono px-1.5 py-0.2 rounded ${
              recommendation.urgency === 'HIGH'
                ? 'bg-rose-950 text-rose-400 border border-rose-800'
                : 'bg-slate-800 text-slate-400'
            }`}
          >
            {recommendation.urgency} Priority
          </span>
        </div>

        <div className="flex items-center space-x-1.5 bg-slate-950 border border-slate-800 rounded px-2 py-0.5">
          <span className="text-[10px] uppercase font-mono text-slate-400">Impact</span>
          <span className="text-xs font-mono font-bold text-emerald-400">
            {recommendation.projectedImpact}
          </span>
        </div>
      </div>

      {/* Action headline & Rationale */}
      <h3 className="text-sm font-semibold text-white">
        {recommendation.action}
      </h3>
      <p className="text-xs text-slate-400 mt-1 leading-relaxed">
        {recommendation.rationale}
      </p>

      {/* Action Footer */}
      <div className="mt-3.5 pt-2.5 border-t border-slate-800/80 flex items-center justify-between">
        <span className="text-[11px] font-mono text-slate-400">
          Confidence: <strong className="text-slate-300">{recommendation.confidence}%</strong>
        </span>

        <div className="flex items-center space-x-2">
          {recommendation.secondaryActionLabel && (
            <button
              type="button"
              disabled={executed}
              onClick={() => onDismiss && onDismiss(recommendation)}
              className="px-2.5 py-1 text-xs font-medium text-slate-400 hover:text-white bg-slate-800/60 hover:bg-slate-800 rounded border border-slate-700/60 transition-colors focus:outline-none focus:ring-2 focus:ring-slate-500"
            >
              {recommendation.secondaryActionLabel}
            </button>
          )}

          <button
            type="button"
            disabled={executed}
            onClick={handleExecute}
            className={`px-3 py-1 text-xs font-semibold rounded border transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500 ${
              executed
                ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300 cursor-default'
                : 'bg-cyan-600 hover:bg-cyan-500 border-cyan-500 text-white shadow-sm'
            }`}
          >
            {executed ? '✓ Enforced in Playbook' : recommendation.primaryActionLabel}
          </button>
        </div>
      </div>
    </div>
  );
};
