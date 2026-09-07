'use client';

import React from 'react';
import { StandardLearningData } from '@/types/ux-foundations';

interface StandardLearningCardProps {
  learning: StandardLearningData;
  onAdoptRule?: (learning: StandardLearningData) => void;
}

export const StandardLearningCard: React.FC<StandardLearningCardProps> = ({
  learning,
  onAdoptRule,
}) => {
  return (
    <div
      role="region"
      aria-label={`Learning: ${learning.title}`}
      className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm hover:border-slate-700 transition-all"
    >
      {/* Top Header */}
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center space-x-2">
          {learning.ticker && (
            <span className="font-mono font-bold text-xs text-white px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
              {learning.ticker}
            </span>
          )}
          <span className="text-xs font-semibold text-slate-300">
            Causal Takeaway
          </span>
        </div>

        <span
          className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${
            learning.statisticalStrength === 'VERY_HIGH'
              ? 'bg-emerald-950/60 text-emerald-300 border-emerald-800/60'
              : learning.statisticalStrength === 'HIGH'
              ? 'bg-cyan-950/60 text-cyan-300 border-cyan-800/60'
              : 'bg-slate-800 text-slate-300 border-slate-700'
          }`}
        >
          {learning.statisticalStrength} Evidence
        </span>
      </div>

      {/* Main Title & Takeaway */}
      <h3 className="text-sm font-semibold text-white">
        {learning.title}
      </h3>
      <p className="text-xs text-slate-400 mt-1 leading-relaxed">
        {learning.takeaway}
      </p>

      {/* Metric Impact Grid */}
      <div className="grid grid-cols-2 gap-2 my-3 p-2.5 bg-slate-950/70 border border-slate-800/80 rounded-lg">
        <div>
          <span className="text-[10px] uppercase font-mono text-slate-400 block">Win Rate Correlation</span>
          <span className="text-xs font-mono font-bold text-emerald-400">{learning.winRateImpact}</span>
        </div>
        <div>
          <span className="text-[10px] uppercase font-mono text-slate-400 block">Error Reduction</span>
          <span className="text-xs font-mono font-bold text-cyan-400">{learning.errorElimination}</span>
        </div>
      </div>

      {/* Rule Update Proposal */}
      <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between gap-3">
        <div className="text-[11px] text-slate-300 truncate">
          <strong className="text-amber-400 font-medium">Proposed Rule:</strong> {learning.ruleRefinementProposal}
        </div>
        {onAdoptRule && (
          <button
            type="button"
            onClick={() => onAdoptRule(learning)}
            className="shrink-0 px-2.5 py-1 text-xs font-medium text-cyan-300 hover:text-white bg-cyan-950/60 hover:bg-cyan-900/60 border border-cyan-800/60 rounded focus:outline-none focus:ring-2 focus:ring-cyan-500 transition-colors"
          >
            Add to Playbook
          </button>
        )}
      </div>
    </div>
  );
};
