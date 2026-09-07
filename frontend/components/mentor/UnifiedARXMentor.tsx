'use client';

import React, { useState } from 'react';
import { MentorContext, MentorInsight } from '@/types/ux-foundations';

interface UnifiedARXMentorProps {
  insight: MentorInsight;
  onApplyRecommendation?: (insight: MentorInsight) => void;
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

const CONTEXT_BADGES: Record<MentorContext, { label: string; border: string; bg: string }> = {
  ATTENTION: { label: 'Attention Coach', border: 'border-cyan-500/40', bg: 'bg-cyan-950/30' },
  DECISION: { label: 'Decision Coach', border: 'border-emerald-500/40', bg: 'bg-emerald-950/30' },
  ATTRIBUTION: { label: 'Attribution Coach', border: 'border-purple-500/40', bg: 'bg-purple-950/30' },
  LEARNING: { label: 'Learning Coach', border: 'border-indigo-500/40', bg: 'bg-indigo-950/30' },
  PLAYBOOK: { label: 'Playbook Coach', border: 'border-blue-500/40', bg: 'bg-blue-950/30' },
  GOVERNANCE: { label: 'Governance Guardian', border: 'border-amber-500/40', bg: 'bg-amber-950/30' },
};

export const UnifiedARXMentor: React.FC<UnifiedARXMentorProps> = ({
  insight,
  onApplyRecommendation,
  collapsed = false,
  onToggleCollapse,
}) => {
  const [showEvidence, setShowEvidence] = useState(false);
  const [applied, setApplied] = useState(false);

  const contextConfig = CONTEXT_BADGES[insight.context] || CONTEXT_BADGES.DECISION;

  const handleApply = () => {
    setApplied(true);
    if (onApplyRecommendation) onApplyRecommendation(insight);
  };

  return (
    <aside
      aria-label={`${insight.roleName}: ${insight.recommendation}`}
      className={`bg-slate-900/95 border ${contextConfig.border} rounded-xl p-4 shadow-lg backdrop-blur-sm transition-all duration-200 flex flex-col justify-between`}
    >
      <div>
        {/* Mentor Header: Context Badge, Confidence, Collapse */}
        <div className="flex items-center justify-between gap-2 mb-3">
          <div className="flex items-center space-x-2">
            <div className="h-6 w-6 rounded-md bg-gradient-to-tr from-cyan-500 to-indigo-600 flex items-center justify-center text-white text-[11px] font-bold shadow-sm">
              AI
            </div>
            <div>
              <span className="text-xs font-bold text-white block leading-tight">
                {insight.roleName}
              </span>
              <span className="text-[10px] font-mono text-cyan-400">
                Cognitive Assistance
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-950/60 border border-emerald-800/40 px-2 py-0.5 rounded">
              {insight.confidence}% Conf.
            </span>
            {onToggleCollapse && (
              <button
                type="button"
                onClick={onToggleCollapse}
                aria-label={collapsed ? 'Expand Mentor' : 'Collapse Mentor'}
                className="text-slate-400 hover:text-white p-1 rounded focus:outline-none focus:ring-2 focus:ring-cyan-500"
              >
                {collapsed ? '➕' : '➖'}
              </button>
            )}
          </div>
        </div>

        {/* 5-Stage Cognitive Interaction Pattern */}
        <div className="space-y-3">
          {/* 1. Observation ("What we see") */}
          <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800/80">
            <div className="text-[10px] uppercase font-mono font-bold tracking-wider text-slate-400 mb-1 flex items-center space-x-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-slate-400" />
              <span>1. Observation</span>
            </div>
            <p className="text-xs text-slate-200 leading-relaxed font-sans">
              {insight.observation}
            </p>
          </div>

          {/* 2. Understanding ("Why it matters") */}
          <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800/80">
            <div className="text-[10px] uppercase font-mono font-bold tracking-wider text-indigo-400 mb-1 flex items-center space-x-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-indigo-400" />
              <span>2. Systemic Understanding</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed font-sans">
              {insight.understanding}
            </p>
          </div>

          {/* 3. Recommendation ("What to do") */}
          <div className="p-2.5 rounded-lg bg-cyan-950/30 border border-cyan-800/40">
            <div className="flex items-center justify-between mb-1">
              <div className="text-[10px] uppercase font-mono font-bold tracking-wider text-cyan-300 flex items-center space-x-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
                <span>3. Recommended Action</span>
              </div>
              <span className="text-[10px] font-mono font-semibold text-emerald-400">
                Impact: {insight.projectedImpact}
              </span>
            </div>
            <p className="text-xs font-medium text-white leading-relaxed">
              {insight.recommendation}
            </p>
          </div>

          {/* 4. Justification ("Why this recommendation") */}
          <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800/80">
            <div className="text-[10px] uppercase font-mono font-bold tracking-wider text-slate-400 mb-1 flex items-center space-x-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              <span>4. Mathematical Justification</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              {insight.justification}
            </p>
          </div>
        </div>
      </div>

      {/* 5. Learning & Evidence Footer */}
      <div className="mt-4 pt-3 border-t border-slate-800">
        <div className="flex items-center justify-between gap-2">
          <button
            type="button"
            onClick={() => setShowEvidence(!showEvidence)}
            className="text-xs text-cyan-400 hover:text-cyan-300 font-medium flex items-center space-x-1 focus:outline-none focus:ring-2 focus:ring-cyan-500 rounded px-1.5 py-0.5"
          >
            <span>{showEvidence ? 'Hide Audit Evidence' : '5. Inspect Evidence'}</span>
            <span className="text-[10px] font-mono">({insight.sampleSize} samples)</span>
          </button>

          <button
            type="button"
            disabled={applied}
            onClick={handleApply}
            className={`px-3 py-1.5 text-xs font-semibold rounded border transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500 ${
              applied
                ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
                : 'bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white border-cyan-500/60 shadow'
            }`}
          >
            {applied ? '✓ Adopted by User' : 'Adopt Recommendation'}
          </button>
        </div>

        {/* Expandable Evidence Details */}
        {showEvidence && (
          <div className="mt-3 p-2 bg-slate-950 rounded border border-slate-800 text-[11px] font-mono space-y-1">
            <div className="flex justify-between text-slate-400">
              <span>Sample Size:</span>
              <span className="text-white">N = {insight.sampleSize}</span>
            </div>
            <div className="flex justify-between text-slate-400">
              <span>Statistical P-Value:</span>
              <span className="text-emerald-400">
                {insight.pValue < 0.001 ? 'p < 0.001' : `p = ${insight.pValue.toFixed(3)}`}
              </span>
            </div>
            <div className="text-slate-400 truncate mt-1 pt-1 border-t border-slate-800">
              <span className="block text-[9px] uppercase text-slate-400">Ledger Verification Hash:</span>
              <code className="text-[10px] text-cyan-400 truncate block">{insight.evidenceHash}</code>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};
