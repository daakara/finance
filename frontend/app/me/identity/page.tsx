"use client";

import React, { useState } from 'react';
import Link from 'next/link';
import {
  CANONICAL_IDENTITY_TWIN,
  IdentityTwin,
  IdentityTrait,
} from '../../../lib/simulation/identityIntelligenceEngine';

export default function IdentityIntelligencePage() {
  const [twin] = useState<IdentityTwin>(CANONICAL_IDENTITY_TWIN);
  const [selectedTrait, setSelectedTrait] = useState<IdentityTrait | null>(
    twin.currentIdentity.traits[0] || null
  );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Navigation & Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-indigo-400 mb-1">
              <Link href="/me" className="hover:underline text-slate-400">
                Life OS
              </Link>
              <span>/</span>
              <span>Identity Intelligence (H13)</span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
              <span>Who Am I Becoming?</span>
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 font-mono border border-indigo-500/30">
                Identity Twin
              </span>
            </h1>
            <p className="text-sm text-slate-400 mt-1">
              Moving beyond daily tasks toward long-term identity trajectory. Aligning daily habits with your future self.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Link
              href="/me"
              className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-xs font-medium text-slate-300 border border-slate-800 transition-colors"
            >
              Back to Cockpit
            </Link>
            <Link
              href="/me/patterns"
              className="px-3 py-1.5 rounded-lg bg-amber-950/60 hover:bg-amber-900/60 text-xs font-medium text-amber-300 border border-amber-800/60 transition-colors"
            >
              Behavioral Patterns
            </Link>
          </div>
        </div>

        {/* TOP METRICS: THE TRIAD SUMMARY & IAI */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl border border-indigo-900/60 bg-gradient-to-br from-indigo-950/40 to-slate-900 shadow-sm">
            <div className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-1">
              Identity Alignment (IAI)
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-extrabold text-white">{twin.identityAlignmentIndex}</span>
              <span className="text-xs text-indigo-300">/ 100</span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Actions reinforce target trajectory with developing consistency.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 shadow-sm">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Identity Momentum
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-extrabold text-emerald-400">{twin.identityMomentum}%</span>
              <span className="text-xs text-slate-400">Compounding</span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Positive habit velocity over 90-day execution window.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 shadow-sm">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Overall Identity Gap
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-extrabold text-amber-400">+{twin.identityGap.overallGap} pts</span>
              <span className="text-xs text-slate-400">to Target</span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Estimated ~{twin.identityGap.estimatedEffortWeeks} weeks of focused progression.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 shadow-sm">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Life Intelligence Triad
            </div>
            <div className="grid grid-cols-3 gap-1 pt-1 text-center font-mono">
              <div className="bg-slate-950 p-1.5 rounded border border-slate-800">
                <div className="text-[10px] text-slate-500">LHI</div>
                <div className="text-sm font-bold text-emerald-400">84</div>
              </div>
              <div className="bg-slate-950 p-1.5 rounded border border-slate-800">
                <div className="text-[10px] text-slate-500">HHI</div>
                <div className="text-sm font-bold text-indigo-400">89</div>
              </div>
              <div className="bg-slate-950 p-1.5 rounded border border-slate-800">
                <div className="text-[10px] text-slate-500">IAI</div>
                <div className="text-sm font-bold text-cyan-400">{twin.identityAlignmentIndex}</div>
              </div>
            </div>
            <p className="text-[11px] text-slate-400 mt-1 text-center">
              Life stable · Household safe · Identity advancing
            </p>
          </div>
        </div>

        {/* IDENTITY DRIFT ALERT (IF ACTIVE) */}
        {twin.driftAlerts.length > 0 && (
          <div className="p-4 rounded-xl border border-amber-800/80 bg-gradient-to-r from-amber-950/40 via-slate-900 to-slate-900 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-amber-400 animate-ping" />
                <span className="text-xs font-bold text-amber-300 uppercase tracking-wider font-mono">
                  Identity Drift Detected (INV-OI110-P)
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30">
                  {twin.driftAlerts[0].driftSeverity} Severity
                </span>
              </div>
              <p className="text-sm text-slate-200">
                {twin.driftAlerts[0].causalExplanation}
              </p>
              <p className="text-xs text-amber-200/80">
                <strong>Recommended Remedy:</strong> {twin.driftAlerts[0].correctiveActionRecommendation}
              </p>
            </div>
            <button
              onClick={() => alert('Micro-commitment added to queue: Draft 1 technical briefing.')}
              className="px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-slate-950 font-semibold text-xs whitespace-nowrap transition-colors shadow-sm"
            >
              Accept Micro-Commitment
            </button>
          </div>
        )}

        {/* QUESTION 1 & 2: WHO AM I TODAY VS WHO AM I BECOMING? */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* WHO AM I TODAY? */}
          <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                1. Who Am I Today?
              </span>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                Competence: {twin.currentIdentity.competenceScore}/100
              </span>
            </div>

            <div>
              <h2 className="text-xl font-bold text-white tracking-tight">
                {twin.currentIdentity.roleTitle}
              </h2>
              <p className="text-xs font-medium text-indigo-300 mt-0.5">
                {twin.currentIdentity.archetype}
              </p>
            </div>

            <div className="flex flex-wrap gap-1.5 pt-1">
              {twin.currentIdentity.domainFoci.map((focus) => (
                <span
                  key={focus}
                  className="text-[11px] px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700/60"
                >
                  {focus}
                </span>
              ))}
            </div>

            <div className="space-y-3 pt-2">
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Core Traits & Progression
              </div>
              {twin.currentIdentity.traits.map((trait) => (
                <div
                  key={trait.id}
                  onClick={() => setSelectedTrait(trait)}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedTrait?.id === trait.id
                      ? 'border-indigo-500 bg-indigo-950/30'
                      : 'border-slate-800/80 bg-slate-950/50 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs mb-1.5">
                    <span className="font-semibold text-slate-200">{trait.name}</span>
                    <span className="font-mono text-indigo-400">
                      {trait.currentLevel} → {trait.targetLevel} (+{trait.delta})
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <div
                      className="bg-indigo-500 h-full rounded-full transition-all duration-500"
                      style={{ width: `${trait.currentLevel}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* WHO AM I BECOMING? */}
          <div className="p-6 rounded-xl border border-indigo-900/60 bg-gradient-to-br from-indigo-950/20 to-slate-900/40 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-indigo-400 uppercase tracking-wider">
                2. Who Am I Becoming?
              </span>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-indigo-900/40 text-indigo-200 border border-indigo-800/60">
                Target Competence: {twin.targetIdentity.competenceScore}/100
              </span>
            </div>

            <div>
              <h2 className="text-xl font-bold text-white tracking-tight">
                {twin.targetIdentity.roleTitle}
              </h2>
              <p className="text-xs font-medium text-indigo-300 mt-0.5">
                {twin.targetIdentity.archetype}
              </p>
            </div>

            <div className="flex flex-wrap gap-1.5 pt-1">
              {twin.targetIdentity.domainFoci.map((focus) => (
                <span
                  key={focus}
                  className="text-[11px] px-2.5 py-1 rounded-md bg-indigo-900/30 text-indigo-300 border border-indigo-700/50"
                >
                  {focus}
                </span>
              ))}
            </div>

            <div className="space-y-3 pt-2">
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Identified Skill Trajectory Gaps
              </div>
              <div className="space-y-2">
                {twin.identityGap.skillGaps.map((sg) => (
                  <div
                    key={sg.skill}
                    className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800 text-xs flex items-center justify-between"
                  >
                    <div>
                      <div className="font-semibold text-slate-200">{sg.skill}</div>
                      <div className="text-[10px] text-slate-400">
                        {sg.domain} · Est. {sg.estimatedWeeks} weeks
                      </div>
                    </div>
                    <div className="text-right font-mono">
                      <span className="text-amber-400 font-bold">+{sg.gap} pts</span>
                      <div className="text-[10px] text-slate-500">
                        {sg.current} → {sg.required}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* QUESTION 3: WHAT EVIDENCE SUPPORTS THAT? (TRACEABILITY CHAIN) */}
        {selectedTrait && (
          <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">
                  3. What Evidence Supports That?
                </span>
                <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono border border-emerald-500/30">
                  INV-OI111-P Traceability
                </span>
              </div>
              <span className="text-xs font-mono text-slate-400">
                Trait: {selectedTrait.name}
              </span>
            </div>

            <p className="text-xs text-slate-300">
              Under invariant <code>INV-OI111-P</code>, every identity score shift is strictly backed by verifiable, causal proof points from real execution history.
            </p>

            <div className="space-y-2">
              {selectedTrait.evidenceChain.map((ev, idx) => (
                <div
                  key={idx}
                  className="flex items-start gap-3 p-3 rounded-lg bg-slate-950/80 border border-slate-800/80 text-xs"
                >
                  <span className="h-5 w-5 rounded-full bg-emerald-950 text-emerald-400 flex items-center justify-center font-bold border border-emerald-700/60 shrink-0">
                    ✓
                  </span>
                  <div>
                    <div className="font-semibold text-slate-200">{ev}</div>
                    <div className="text-[11px] text-slate-500 mt-0.5">
                      Verified via Execution Engine · Lineage Hash Confirmed
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* QUESTION 4 & 5: EMERGING VS FADING IDENTITIES */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* EMERGING IDENTITIES */}
          <div className="p-6 rounded-xl border border-emerald-900/50 bg-gradient-to-br from-emerald-950/20 to-slate-900/40 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">
                4. What Identities Are Emerging?
              </span>
              <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono border border-emerald-500/30">
                Rising Patterns
              </span>
            </div>

            <div className="space-y-3">
              {twin.emergingIdentities.map((ei) => (
                <div
                  key={ei.name}
                  className="p-3.5 rounded-lg bg-slate-950/80 border border-emerald-900/40 space-y-2 text-xs"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white text-sm">{ei.name}</span>
                    <span className="font-mono text-emerald-400 font-semibold">
                      {ei.confidenceScore}% Confidence
                    </span>
                  </div>
                  <p className="text-slate-300 text-xs">
                    <span className="text-emerald-400 font-semibold">Catalyst: </span>
                    {ei.catalyst}
                  </p>
                  <div className="space-y-1 pt-1 border-t border-slate-800">
                    {ei.evidencePoints.map((ep, i) => (
                      <div key={i} className="text-[11px] text-slate-400 flex items-center gap-1.5">
                        <span className="h-1 w-1 rounded-full bg-emerald-400" />
                        <span>{ep}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* FADING IDENTITIES */}
          <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                5. What Identities Are Fading?
              </span>
              <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                Decaying Habits
              </span>
            </div>

            <div className="space-y-3">
              {twin.fadingIdentities.map((fi) => (
                <div
                  key={fi.name}
                  className="p-3.5 rounded-lg bg-slate-950/80 border border-slate-800 space-y-2 text-xs"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-slate-300 text-sm line-through decoration-rose-500/60">
                      {fi.name}
                    </span>
                    <span className="font-mono text-rose-400 font-semibold">
                      -{fi.decayRatePct}% Shed
                    </span>
                  </div>
                  <p className="text-slate-300 text-xs">
                    <span className="text-indigo-400 font-semibold">Superseded By: </span>
                    {fi.supersededBy}
                  </p>
                  <p className="text-[11px] text-slate-500 pt-1 border-t border-slate-800">
                    {fi.observedReduction}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* BOTTOM FOOTER LINK TO 30-SECOND COCKPIT */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs text-slate-400">
          <div>
            <span className="font-semibold text-slate-200">The 30-Second Cockpit Invariant:</span> Every recommendation in your daily queue compounds your target identity while strictly preserving the 1 Primary Action limit.
          </div>
          <Link
            href="/me"
            className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium whitespace-nowrap transition-colors"
          >
            Open Today&apos;s Cockpit
          </Link>
        </div>
      </div>
    </div>
  );
}
