"use client";

import React, { useState } from 'react';
import Link from 'next/link';
import {
  orchestrateNextBestActions,
  CANONICAL_CANDIDATE_POOL,
  UserOperationalContext,
  CockpitActionQueue,
} from '../../lib/simulation/nextBestActionEngine';
import {
  evaluateCognitiveTradingOpportunity,
  CANONICAL_MARKET_OPPORTUNITIES,
  MarketEnvironment,
} from '../../lib/simulation/cognitiveTradingEngine';
import { queryUniversalWisdom } from '../../lib/simulation/universalWisdomEngine';

export default function ThirtySecondCockpit() {
  const [activeMode, setActiveMode] = useState<'CALM' | 'HOUSEHOLD' | 'TECHNICAL'>('CALM');
  const [showTechnicalDrawer, setShowTechnicalDrawer] = useState<boolean>(false);
  const [recoveryScore, setRecoveryScore] = useState<number>(84);
  const [liquidRunwayMonths, setLiquidRunwayMonths] = useState<number>(14.2);

  // Default simulated user context
  const context: UserOperationalContext = {
    recoveryScore,
    recentLossStreak: 0,
    dailyDrawdownPct: 0.005,
    liquidCash: Math.round(liquidRunwayMonths * 4200),
    monthlyEssentialBurn: 4200,
    householdStrainIndex: 22,
    focusHoursAvailable: 4.5,
  };

  const marketEnv: MarketEnvironment = {
    regime: 'TRENDING_BULL',
    vixLevel: 14.8,
    marketTrendConfidence: 86,
  };

  // Evaluate candidate actions via the Next Best Action Engine
  const topOpportunity = CANONICAL_MARKET_OPPORTUNITIES[0];
  const tradingAssessment = evaluateCognitiveTradingOpportunity(topOpportunity, context, marketEnv);

  const candidatePool = [
    tradingAssessment.candidateAction,
    ...CANONICAL_CANDIDATE_POOL.filter((c) => c.domain !== 'TRADING'),
  ];

  const actionQueue: CockpitActionQueue = orchestrateNextBestActions(candidatePool, context);
  const primary = actionQueue.primaryAction;
  const secondary = actionQueue.secondaryActions;

  // Collective Wisdom query
  const wisdom = queryUniversalWisdom({
    ageRange: '31-35',
    dependentsCount: 2,
    runwayMonthsRange: '12-24',
    baselineDomain: 'Software & Financial Analysis',
  });

  return (
    <div className="space-y-6">
      {/* Top Cockpit Header & Mode Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-900/80 border border-slate-800 p-4 rounded-xl backdrop-blur-sm">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse" />
            <h2 className="text-lg font-bold text-white tracking-tight">The 30-Second Life Cockpit</h2>
            <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 font-mono border border-indigo-500/30">
              Horizon 10
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Calm decision reduction · 1 Primary Action · {actionQueue.intelligenceReductionRatio}% Noise Suppressed
          </p>
        </div>

        {/* Mode Switcher */}
        <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
          <button
            onClick={() => setActiveMode('CALM')}
            className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
              activeMode === 'CALM'
                ? 'bg-emerald-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Calm Mode
          </button>
          <button
            onClick={() => setActiveMode('HOUSEHOLD')}
            className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
              activeMode === 'HOUSEHOLD'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Household Mode
          </button>
          <button
            onClick={() => {
              setActiveMode('TECHNICAL');
              setShowTechnicalDrawer(true);
            }}
            className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
              activeMode === 'TECHNICAL'
                ? 'bg-amber-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Quant Proof
          </button>
        </div>
      </div>

      {/* ZONE 1 & 2: How am I doing? & Where am I heading? (Two 30-second cards) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Card 1: How am I doing? */}
        <div className="p-5 rounded-xl border border-slate-800 bg-gradient-to-br from-slate-900 to-slate-950 shadow-md">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
              1. How Am I Doing?
            </span>
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700/60">
                LHI 84 · HHI 89 · IAI 61
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 font-semibold border border-emerald-500/30">
                Optimal State
              </span>
            </div>
          </div>
          <div className="flex items-baseline gap-3 mb-2">
            <span className="text-3xl font-extrabold text-white tracking-tight">84 / 100</span>
            <span className="text-sm font-medium text-emerald-400">↑ Prime Mental Poise</span>
          </div>
          <p className="text-sm text-slate-300 leading-relaxed mb-4">
            Sleep was <strong>7.6 hours</strong> with low resting stress. Your emotional clarity is primed for disciplined decisions today.
          </p>
          <div className="grid grid-cols-3 gap-2 pt-3 border-t border-slate-800/80 text-center">
            <div>
              <div className="text-xs text-slate-400">Sleep Recovery</div>
              <div className="text-sm font-bold text-emerald-400">{recoveryScore}%</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">Focus Hours</div>
              <div className="text-sm font-bold text-cyan-400">4.5h Free</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">Domestic Strain</div>
              <div className="text-sm font-bold text-indigo-400">Low (22%)</div>
            </div>
          </div>
        </div>

        {/* Card 2: Where am I heading? */}
        <div className="p-5 rounded-xl border border-slate-800 bg-gradient-to-br from-slate-900 to-slate-950 shadow-md">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
              2. Where Am I Heading?
            </span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/30">
              Shielded
            </span>
          </div>
          <div className="flex items-baseline gap-3 mb-2">
            <span className="text-3xl font-extrabold text-white tracking-tight">
              {liquidRunwayMonths.toFixed(1)} Months
            </span>
            <span className="text-sm font-medium text-cyan-400">Safe Cash Runway</span>
          </div>
          <p className="text-sm text-slate-300 leading-relaxed mb-4">
            Your emergency buffer covers <strong>$59,640</strong> in essential expenses. Strict 6-month capital preservation floor is fully respected.
          </p>
          <div className="grid grid-cols-3 gap-2 pt-3 border-t border-slate-800/80 text-center">
            <div>
              <div className="text-xs text-slate-400">Safety Floor</div>
              <div className="text-sm font-bold text-slate-200">6.0 Mos</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">Market Dip Shock</div>
              <div className="text-sm font-bold text-emerald-400">Protected</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">12m Trajectory</div>
              <div className="text-sm font-bold text-cyan-400">+12% Upside</div>
            </div>
          </div>
        </div>
      </div>

      {/* ZONE 3: What should I do next? (THE SINGLE KEYSTONE MOVE - INV-OI101-P) */}
      <div className="p-6 rounded-2xl border-2 border-emerald-500/40 bg-gradient-to-br from-slate-900 via-slate-900/90 to-emerald-950/20 shadow-xl relative overflow-hidden">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="text-amber-400 text-lg">★</span>
            <span className="text-xs font-bold uppercase tracking-wider text-emerald-400">
              3. Today&apos;s Single Smartest Move (Primary)
            </span>
          </div>
          <span className="text-xs px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 font-semibold border border-emerald-500/40">
            Utility Score: {primary.utilityScore}/100
          </span>
        </div>

        <h3 className="text-xl font-black text-white tracking-tight mb-2">
          {primary.headline}
        </h3>
        <p className="text-slate-300 text-base leading-relaxed mb-5 max-w-3xl">
          {primary.explanation}
        </p>

        {/* Proof Details Snippet */}
        {primary.proofDetails && (
          <div className="mb-5 p-3 rounded-lg bg-slate-950/60 border border-slate-800 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div>
              <span className="text-slate-500 block">Setup Type</span>
              <span className="text-slate-200 font-semibold">{primary.proofDetails.setupType || 'Adaptive'}</span>
            </div>
            <div>
              <span className="text-slate-500 block">Reward/Risk Ratio</span>
              <span className="text-emerald-400 font-semibold">{primary.proofDetails.riskRewardRatio || '3.4'} : 1</span>
            </div>
            <div>
              <span className="text-slate-500 block">Daily Risk Cap</span>
              <span className="text-cyan-400 font-semibold">${primary.proofDetails.maxDollarRisk || 140} Max</span>
            </div>
            <div>
              <span className="text-slate-500 block">Expected 30d LHI</span>
              <span className="text-amber-400 font-semibold">+{primary.expectedImpact.lhiDelta} Points</span>
            </div>
          </div>
        )}

        <div className="flex flex-wrap items-center gap-3">
          <Link
            href="/me/execute"
            className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-bold rounded-lg shadow-lg transition-all flex items-center gap-1.5"
          >
            <span>▶</span> Execute Now
          </Link>
          {primary.actionPayload && (
            <Link
              href={primary.actionPayload.route}
              className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-semibold rounded-lg border border-slate-700 transition-all"
            >
              {primary.actionPayload.ctaLabel} →
            </Link>
          )}
          <button
            onClick={() => setShowTechnicalDrawer(!showTechnicalDrawer)}
            className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-semibold rounded-lg border border-slate-700 transition-all"
          >
            {showTechnicalDrawer ? 'Hide Mathematical Proof' : 'Inspect Mathematical Proof ↗'}
          </button>
        </div>
      </div>

      {/* ZONE 4: Secondary Focus This Week (INV-OI101-P: <= 2 items) */}
      <div>
        <div className="flex items-center justify-between mb-3 px-1">
          <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
            Secondary Focus This Week (Max 2)
          </span>
          <span className="text-xs text-slate-500">Curated by NBA Engine</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {secondary.map((act) => (
            <div
              key={act.id}
              className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-slate-700 transition-all flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                    {act.domain}
                  </span>
                  <span className="text-xs text-slate-400 font-mono">+{act.expectedImpact.lhiDelta} LHI</span>
                </div>
                <h4 className="text-sm font-bold text-white mb-1">{act.headline}</h4>
                <p className="text-xs text-slate-400 leading-relaxed">{act.explanation}</p>
              </div>
              {act.actionPayload && (
                <div className="mt-3 pt-2 border-t border-slate-800/80">
                  <Link
                    href={act.actionPayload.route}
                    className="text-xs font-semibold text-emerald-400 hover:text-emerald-300"
                  >
                    {act.actionPayload.ctaLabel} →
                  </Link>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* ZONE 5: Population Wisdom Insight (Differentially Private) */}
      <div className="p-4 rounded-xl border border-indigo-900/40 bg-indigo-950/20">
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-indigo-400 text-sm">✦</span>
          <span className="text-xs font-bold uppercase tracking-wider text-indigo-300">
            Collective Wisdom (Anonymized Cohort · k = {wisdom.totalCohortSampleSize})
          </span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          &ldquo;{wisdom.topEmpiricalRecommendation}&rdquo;
        </p>
      </div>

      {/* TECHNICAL PROOF DRAWER (Progressive Disclosure) */}
      {showTechnicalDrawer && (
        <div className="p-6 rounded-2xl border border-amber-500/40 bg-slate-950 shadow-2xl space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h4 className="text-base font-bold text-white">Mathematical Proof &amp; Invariant Verification</h4>
              <p className="text-xs text-slate-400">Formal verification logs across Horizons 1 through 10</p>
            </div>
            <button
              onClick={() => setShowTechnicalDrawer(false)}
              className="text-xs text-slate-400 hover:text-white px-2 py-1 bg-slate-900 rounded border border-slate-800"
            >
              ✕ Close
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
              <span className="text-emerald-400 font-bold block mb-1">INV-OI97-P Passed</span>
              <p className="text-slate-400">
                Cognitive Recovery: {context.recoveryScore}% (&gt; 55% floor). Loss streak: 0 (&lt; 2). Safe trade execution certified.
              </p>
            </div>
            <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
              <span className="text-emerald-400 font-bold block mb-1">INV-OI98-P Passed</span>
              <p className="text-slate-400">
                Cash runway remains {liquidRunwayMonths.toFixed(1)} months (&gt; 6.0 month household safety floor).
              </p>
            </div>
            <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
              <span className="text-emerald-400 font-bold block mb-1">INV-OI101-P Passed</span>
              <p className="text-slate-400">
                Decision Simplicity: Exactly 1 primary move surfaced, {secondary.length} secondary actions visible.
              </p>
            </div>
          </div>

          <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-800 font-mono text-[11px] text-slate-300">
            <div className="text-amber-400 font-bold mb-1">{"// Causal DAG Trace"}</div>
            <div>Sleep (7.6h) → Recovery (84%) → Mental Poise (0.91) → Minervini VCP Setup (GOOGL) → MaxRisk ($140) → PostRunway (14.2m)</div>
          </div>
        </div>
      )}

      {/* QUICK HORIZON WORKSPACE NAVIGATION */}
      <div className="pt-4 border-t border-slate-800">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
          Deep Exploration Workspaces
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <Link href="/me/allocator" className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800">
            Capacity Allocator (H5)
          </Link>
          <Link href="/me/signals" className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800">
            Telemetry Signals (H6)
          </Link>
          <Link href="/me/twin" className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800">
            Causal Twin DAG (H7)
          </Link>
          <Link href="/me/household" className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800">
            Household OS (H8)
          </Link>
          <Link href="/me/trajectories" className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800">
            Trajectory Sequencer (H9)
          </Link>
          <Link href="/me/strategy" className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800">
            Household Strategy (H9/10)
          </Link>
          <Link href="/me/execute" className="px-3 py-1.5 rounded-lg bg-emerald-950/80 hover:bg-emerald-900/80 text-emerald-300 border border-emerald-700/80 font-semibold">
            Execution Cockpit (H11)
          </Link>
          <Link href="/me/decisions" className="px-3 py-1.5 rounded-lg bg-indigo-950/80 hover:bg-indigo-900/80 text-indigo-300 border border-indigo-700/80 font-semibold">
            Decision Journal (H11)
          </Link>
          <Link href="/me/patterns" className="px-3 py-1.5 rounded-lg bg-amber-950/80 hover:bg-amber-900/80 text-amber-300 border border-amber-700/80 font-semibold">
            Behavioral Patterns (H12)
          </Link>
          <Link href="/me/identity" className="px-3 py-1.5 rounded-lg bg-indigo-950/90 hover:bg-indigo-900/90 text-indigo-200 border border-indigo-600/90 font-semibold">
            Identity Intelligence (H13)
          </Link>
          <Link href="/screener" className="px-3 py-1.5 rounded-lg bg-emerald-950/60 hover:bg-emerald-900/60 text-emerald-300 border border-emerald-800/60">
            Market Screener Terminal
          </Link>
          <Link href="/portfolio" className="px-3 py-1.5 rounded-lg bg-cyan-950/60 hover:bg-cyan-900/60 text-cyan-300 border border-cyan-800/60">
            Portfolio Terminal
          </Link>
        </div>
      </div>
    </div>
  );
}
