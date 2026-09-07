'use client';

import React, { useState } from 'react';
import { DecisionProfile } from '@/types/ux-foundations';

interface DecisionProfileHeaderProps {
  profile: DecisionProfile;
  compact?: boolean;
}

export const DecisionProfileHeader: React.FC<DecisionProfileHeaderProps> = ({
  profile,
  compact = false,
}) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      role="region"
      aria-label="Personal Decision Identity"
      className="w-full bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-md"
    >
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        {/* Left: Identity & Quality Score */}
        <div className="flex items-center space-x-4">
          <div className="relative">
            <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white font-bold text-lg shadow-inner">
              {profile.userName.charAt(0)}
            </div>
            <div className="absolute -bottom-1 -right-1 bg-emerald-500 h-3.5 w-3.5 rounded-full border-2 border-slate-900" title="Active Session" />
          </div>

          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-base font-bold text-white tracking-tight">
                {profile.userName}
              </h2>
              <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 border border-slate-700 text-slate-300 font-medium">
                {profile.role}
              </span>
            </div>
            <div className="flex items-center space-x-3 mt-1 text-xs text-slate-400 font-mono">
              <span>{profile.completedDecisionsCount} Decisions Resolved</span>
              <span>•</span>
              <span className="text-cyan-400 font-semibold">{profile.decileRank} Peer Decile</span>
            </div>
          </div>
        </div>

        {/* Center: Core Behavioral Metrics */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {/* Decision Quality Score */}
          <div className="bg-slate-950/70 border border-slate-800/80 rounded-lg px-3 py-2">
            <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400">
              Quality Score
            </div>
            <div className="flex items-baseline space-x-1.5 mt-0.5">
              <span className="text-xl font-bold font-mono text-white">
                {profile.qualityScore}
              </span>
              <span className="text-xs font-mono font-medium text-emerald-400">
                +{profile.qualityScoreDelta}
              </span>
            </div>
          </div>

          {/* Primary Edge */}
          <div className="bg-slate-950/70 border border-slate-800/80 rounded-lg px-3 py-2">
            <div className="text-[10px] uppercase font-mono tracking-wider text-emerald-400/90">
              Primary Edge
            </div>
            <div className="text-xs font-semibold text-slate-200 mt-0.5 truncate" title={profile.primaryEdge}>
              {profile.primaryEdge}
            </div>
          </div>

          {/* Systemic Weakness */}
          <div className="bg-slate-950/70 border border-slate-800/80 rounded-lg px-3 py-2">
            <div className="text-[10px] uppercase font-mono tracking-wider text-rose-400/90">
              Systemic Trap
            </div>
            <div className="text-xs font-semibold text-slate-200 mt-0.5 truncate" title={profile.primaryWeakness}>
              {profile.primaryWeakness}
            </div>
          </div>

          {/* Active Rules */}
          <div className="bg-slate-950/70 border border-slate-800/80 rounded-lg px-3 py-2">
            <div className="text-[10px] uppercase font-mono tracking-wider text-indigo-400/90">
              Active Playbook
            </div>
            <div className="text-xs font-semibold font-mono text-slate-200 mt-0.5">
              {profile.activeRulesCount} Enforced Rules
            </div>
          </div>
        </div>

        {/* Right: Expand Details */}
        {!compact && (
          <div className="flex lg:flex-col items-center lg:items-end justify-between lg:justify-center border-t lg:border-t-0 border-slate-800 pt-2 lg:pt-0">
            <button
              type="button"
              onClick={() => setExpanded(!expanded)}
              aria-expanded={expanded}
              className="text-xs text-cyan-400 hover:text-cyan-300 font-medium flex items-center space-x-1 focus:outline-none focus:ring-2 focus:ring-cyan-500 rounded px-2 py-1"
            >
              <span>{expanded ? 'Hide History' : 'View Evolution'}</span>
              <span className="text-[10px]">{expanded ? '▲' : '▼'}</span>
            </button>
            <span className="text-[10px] text-slate-500 font-mono">
              Next Tier: 80 (+6 pts)
            </span>
          </div>
        )}
      </div>

      {/* Expandable History Drawer */}
      {expanded && (
        <div className="mt-4 pt-3 border-t border-slate-800/80">
          <div className="text-xs font-semibold text-slate-300 mb-2">
            Decision Quality Score Evolution (Rolling Trajectory)
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
            {profile.scoreHistory.map((h, i) => (
              <div key={i} className="bg-slate-950 border border-slate-800 rounded p-2 text-center">
                <div className="text-[10px] font-mono text-slate-400">{h.date}</div>
                <div className="text-sm font-bold font-mono text-white mt-0.5">{h.score}</div>
                {h.milestone && (
                  <div className="text-[9px] text-cyan-400 truncate mt-0.5">{h.milestone}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
