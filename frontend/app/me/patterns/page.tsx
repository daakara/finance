"use client";

import React, { useState } from 'react';
import Link from 'next/link';
import IntelligenceHeader from '../../../components/ui/IntelligenceHeader';
import {
  CANONICAL_BEHAVIORAL_PROFILE,
  PersonalBehavioralProfile,
} from '../../../lib/simulation/adaptiveBehaviorEngine';

export default function PatternsPage() {
  const [profile] = useState<PersonalBehavioralProfile>(CANONICAL_BEHAVIORAL_PROFILE);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-10 space-y-8 max-w-5xl mx-auto">
      <IntelligenceHeader
        certification="HORIZON-12-CERTIFIED"
        title="Personal Behavioral Patterns"
        subtitle="What We've Learned About You. Real-world execution chronotypes, friction traps, and active adaptive rules."
        breadcrumbs={[
          { label: 'Life OS', href: '/me' },
          { label: 'Behavioral Patterns' },
        ]}
      />

      {/* Navigation Breadcrumb */}
      <div className="flex items-center justify-between text-xs text-slate-400">
        <Link href="/me" className="text-emerald-400 hover:underline">
          ← Back to 30-Second Cockpit
        </Link>
        <Link href="/me/execute" className="text-cyan-400 hover:underline">
          Go to Execution Cockpit →
        </Link>
      </div>

      {/* Top Banner: Overall Adherence Index */}
      <div className="p-6 rounded-3xl border border-slate-800 bg-gradient-to-br from-slate-900 via-slate-900/90 to-indigo-950/20 shadow-xl flex flex-col sm:flex-row sm:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
              Personal Behavioral Model
            </span>
          </div>
          <h2 className="text-2xl font-black text-white">How You Actually Operate</h2>
          <p className="text-sm text-slate-300 mt-1 max-w-xl leading-relaxed">
            ARX learns from your observed execution history. The platform automatically adjusts task sizes and timing to match your biological energy peaks.
          </p>
        </div>

        <div className="flex items-center gap-4 bg-slate-950/80 p-4 rounded-2xl border border-slate-800/80 text-center">
          <div>
            <span className="text-xs text-slate-400 block">Adherence Index</span>
            <span className="text-3xl font-black text-emerald-400">
              {profile.overallAdherenceIndex} / 100
            </span>
          </div>
          <div className="h-10 w-px bg-slate-800" />
          <div>
            <span className="text-xs text-slate-400 block">Peak Window</span>
            <span className="text-base font-bold text-cyan-400">
              08:00 - 11:00
            </span>
          </div>
        </div>
      </div>

      {/* Grid: Superpowers vs. Friction Traps */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Superpowers Card */}
        <div className="p-6 rounded-2xl border border-emerald-500/30 bg-gradient-to-br from-slate-900 to-emerald-950/10 shadow-lg space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <span className="text-emerald-400 text-lg">✓</span>
              <h3 className="text-base font-bold text-white">Your Execution Superpowers</h3>
            </div>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 font-semibold">
              High Conversion
            </span>
          </div>
          <ul className="space-y-2.5 text-xs text-slate-300">
            {profile.superpowerStrengths.map((s, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-emerald-400 font-bold">✓</span>
                <span>{s}</span>
              </li>
            ))}
          </ul>
          <p className="text-[11px] text-slate-400 pt-2 border-t border-slate-800/60 leading-relaxed">
            &ldquo;When actions are scheduled before noon and scoped to under 30 minutes, your follow-through rate is in the 90th percentile.&rdquo;
          </p>
        </div>

        {/* Friction Traps Card */}
        <div className="p-6 rounded-2xl border border-amber-500/30 bg-gradient-to-br from-slate-900 to-amber-950/10 shadow-lg space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <span className="text-amber-400 text-lg">⚠</span>
              <h3 className="text-base font-bold text-white">Your Observed Friction Traps</h3>
            </div>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-amber-500/20 text-amber-300 font-semibold">
              Non-Punitive
            </span>
          </div>
          <ul className="space-y-2.5 text-xs text-slate-300">
            {profile.frictionTraps.map((f, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-amber-400 font-bold">⚠</span>
                <span>{f}</span>
              </li>
            ))}
          </ul>
          <p className="text-[11px] text-slate-400 pt-2 border-t border-slate-800/60 leading-relaxed">
            &ldquo;Zero shame or moralizing: your biological recovery is simply depleted by 19:00. Willpower is naturally lowest when cognitive fatigue peaks.&rdquo;
          </p>
        </div>
      </div>

      {/* Domain Adherence Breakdown */}
      <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/60 space-y-4">
        <h3 className="text-base font-bold text-white">Domain Follow-Through Conversion</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {Object.values(profile.domainAdherence).map((dom) => (
            <div key={dom.domain} className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-slate-300">{dom.domain}</span>
                <span
                  className={`font-mono font-bold ${
                    dom.completionRatePct >= 80
                      ? 'text-emerald-400'
                      : dom.completionRatePct >= 60
                      ? 'text-cyan-400'
                      : 'text-amber-400'
                  }`}
                >
                  {dom.completionRatePct}%
                </span>
              </div>
              <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    dom.completionRatePct >= 80
                      ? 'bg-emerald-500'
                      : dom.completionRatePct >= 60
                      ? 'bg-cyan-500'
                      : 'bg-amber-500'
                  }`}
                  style={{ width: `${dom.completionRatePct}%` }}
                />
              </div>
              <div className="text-[10px] text-slate-500 flex justify-between">
                <span>{dom.completions}/{dom.attempts} Finished</span>
                <span>{dom.adherenceRating.replace('_', ' ')}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Active Adaptive Rules Ledger */}
      <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/80 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-base font-bold text-white">How ARX Has Adapted to You</h3>
            <p className="text-xs text-slate-400">Live ledger of active personalization rules modifying the Next Best Action engine</p>
          </div>
          <span className="text-xs px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 font-mono font-semibold">
            {profile.activeAdaptiveRules.length} Rules Active
          </span>
        </div>

        <div className="space-y-3">
          {profile.activeAdaptiveRules.map((rule) => (
            <div
              key={rule.id}
              className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 hover:border-slate-700 transition-all flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs"
            >
              <div className="space-y-1 max-w-2xl">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[10px] text-emerald-400 font-semibold px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-800/60">
                    {rule.id}
                  </span>
                  <span className="text-slate-300 font-bold">{rule.adaptationAction}</span>
                </div>
                <p className="text-slate-400 text-[11px]">
                  <strong>Trigger:</strong> {rule.triggerPattern} · <strong>Friction Diagnosed:</strong> {rule.observedFriction}
                </p>
              </div>
              <div className="text-right text-[11px] text-slate-500 font-mono whitespace-nowrap">
                Active since {rule.appliedDate}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
