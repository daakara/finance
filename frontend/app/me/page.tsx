'use client';

/**
 * Horizon 5: /me — The Personal Life Command Center & Life Operating System
 *
 * Implements the 5 Core Consumer Layers:
 * - Layer 1: Life Command Center (LHI 82.4, Trajectory, Top Goals, Risks)
 * - Layer 2: Future Self Navigator (Current -> 6m -> 12m -> 24m)
 * - Layer 3: Personal Portfolio Engine (Paths A, B, C, D)
 * - Layer 4: Daily Action Engine (Micro-commitments & impact traceability)
 * - Layer 5: Personal Drift Dashboard & Future Recovery Projections (INV-OI83-P)
 */

import React, { useState, Suspense } from 'react';
import Link from 'next/link';
import IntelligenceHeader from '../../components/ui/IntelligenceHeader';
import HorizonCard from '../../components/ui/HorizonCard';
import HorizonMetricCard from '../../components/ui/HorizonMetricCard';
import SeverityBadge from '../../components/ui/SeverityBadge';
import {
  CANONICAL_PERSONAL_DRIFT_CARDS,
  getDriftGaugeProperties,
  calculateProgressRail,
} from '../../lib/simulation/personalDriftEngine';
import {
  CANONICAL_RECOVERY_PROJECTION,
  rankRecoveryStrategies,
} from '../../lib/simulation/futureRecoveryEngine';
import { DEFAULT_PERSONAL_CAPACITY } from '../../lib/simulation/personalCapacityEngine';

function LifeCommandCenterContent() {
  const [selectedStrategyId, setSelectedStrategyId] = useState<string>('STRAT-D-COMBINED');
  const [expandedDriftIndex, setExpandedDriftIndex] = useState<number | null>(0);
  const [selectedPortfolioPath, setSelectedPortfolioPath] = useState<string>('PATH-B');

  const recoveryData = CANONICAL_RECOVERY_PROJECTION;
  const rankedStrategies = rankRecoveryStrategies(
    recoveryData.strategies,
    DEFAULT_PERSONAL_CAPACITY
  );
  const activeStrategy =
    rankedStrategies.find((s) => s.strategyId === selectedStrategyId) || rankedStrategies[0];

  const portfolioPaths = [
    {
      id: 'PATH-A',
      name: 'Path A: Retain Staff Engineer Baseline',
      lhi: 80,
      incomeTrend: 'Stable Baseline ($220k)',
      wellbeing: 'Moderate (Low stress)',
      timeCost: '40h / wk',
      risk: 'LOW',
      probability: '95%',
      recommended: false,
    },
    {
      id: 'PATH-B',
      name: 'Path B: Move Into AI Systems Architecture',
      lhi: 88,
      incomeTrend: '+35% Upside ($310k)',
      wellbeing: 'High Growth',
      timeCost: '46h / wk',
      risk: 'MEDIUM',
      probability: '76%',
      recommended: true,
    },
    {
      id: 'PATH-C',
      name: 'Path C: Launch Bootstrapped AI Product',
      lhi: 84,
      incomeTrend: 'High Variance ($0-$400k)',
      wellbeing: 'High Autonomy',
      timeCost: '54h / wk',
      risk: 'HIGH',
      probability: '58%',
      recommended: false,
    },
    {
      id: 'PATH-D',
      name: 'Path D: Part-Time Graduate Specialization',
      lhi: 79,
      incomeTrend: 'Deferred Yield',
      wellbeing: 'Moderate (Time strain)',
      timeCost: '52h / wk',
      risk: 'MEDIUM',
      probability: '84%',
      recommended: false,
    },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 space-y-8 max-w-7xl mx-auto">
      {/* Top Bar with Primary Actions & Allocator Link */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs uppercase tracking-wider font-semibold text-slate-300">
            ARX Horizon // Personal Operating System
          </span>
          <span className="text-slate-600">|</span>
          <span className="text-xs text-slate-400 font-mono">Self-Sovereign Node #1</span>
        </div>

        <div className="flex items-center gap-3">
          <Link
            href="/me/signals"
            className="px-3 py-1.5 bg-cyan-600/20 hover:bg-cyan-600/30 text-cyan-300 border border-cyan-500/30 rounded text-xs font-semibold uppercase tracking-wider transition-colors flex items-center gap-2"
          >
            <span>📡</span> Signal Health (91%)
          </Link>
          <Link
            href="/me/allocator"
            className="px-3 py-1.5 bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/30 rounded text-xs font-semibold uppercase tracking-wider transition-colors flex items-center gap-2"
          >
            <span>⚖</span> Open Resource Allocator
          </Link>
          <SeverityBadge level="LOW" status="HEALTHY" />
        </div>
      </div>

      {/* Primary Header */}
      <IntelligenceHeader
        title="Personal Life Operating System"
        subtitle="The individual personal intelligence layer. Real-time Life Health Index, future self trajectory navigation, personal strategy portfolio ranking, and adaptive recovery projections."
        certification="HORIZON-6-CERTIFIED"
        status="CERTIFIED"
        breadcrumbs={[
          { label: 'Life OS', href: '/me' },
          { label: 'Command Center' },
        ]}
      />

      {/* ========================================================================= */}
      {/* LAYER 1: LIFE COMMAND CENTER (Vital Signs) */}
      {/* ========================================================================= */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xs uppercase tracking-wider font-semibold text-slate-400">
            Layer 1 // Life Command Center (Core Vitality)
          </h2>
          <span className="text-xs text-emerald-400 font-mono">Trajectory: ON TRACK</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <HorizonMetricCard
            label="LIFE HEALTH INDEX (LHI)"
            value="82.4 / 100"
            delta="+3.1 last 30 days"
            deltaPositive={true}
            severity="PASS"
          />
          <HorizonMetricCard
            label="FINANCIAL RUNWAY"
            value="14.2 Months"
            delta="Liquid survival buffer"
            deltaPositive={true}
            severity="PASS"
          />
          <HorizonMetricCard
            label="AUTONOMIC RECOVERY"
            value="74% Battery"
            delta="HRV: 68ms (Stable)"
            deltaPositive={true}
            severity="PASS"
          />
          <HorizonMetricCard
            label="DISCRETIONARY SLACK"
            value="23.5 hrs / wk"
            delta="Safe from burnout"
            deltaPositive={true}
            severity="PASS"
          />
        </div>
      </div>

      {/* ========================================================================= */}
      {/* LAYER 2: FUTURE SELF NAVIGATOR */}
      {/* ========================================================================= */}
      <HorizonCard
        title="Layer 2 // Future Self Navigator (Trajectory Timeline)"
        subtitle="Probabilistic multi-horizon projection from current state to desired destination"
      >
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 relative">
          <div className="p-4 bg-slate-900/90 rounded-lg border border-slate-800 space-y-2 relative">
            <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
              CURRENT SELF (NOW)
            </div>
            <div className="text-base font-bold text-slate-200">Software Engineer</div>
            <div className="text-xs text-slate-400">Baseline IC execution & delivery</div>
            <div className="text-xs font-mono text-emerald-400 pt-2 border-t border-slate-800">
              LHI: 82.4 (Active)
            </div>
          </div>

          <div className="p-4 bg-slate-900/90 rounded-lg border border-slate-800 space-y-2 relative">
            <div className="text-[10px] uppercase font-bold text-blue-400 tracking-wider">
              +6 MONTHS
            </div>
            <div className="text-base font-bold text-slate-200">Senior Engineer</div>
            <div className="text-xs text-slate-400">Leading multi-service design</div>
            <div className="text-xs font-mono text-blue-400 pt-2 border-t border-slate-800 flex justify-between">
              <span>LHI: 85.1</span>
              <span>88% Prob</span>
            </div>
          </div>

          <div className="p-4 bg-slate-900/90 rounded-lg border border-slate-800 space-y-2 relative">
            <div className="text-[10px] uppercase font-bold text-purple-400 tracking-wider">
              +12 MONTHS
            </div>
            <div className="text-base font-bold text-slate-200">Staff AI Specialist</div>
            <div className="text-xs text-slate-400">Production agentic systems</div>
            <div className="text-xs font-mono text-purple-400 pt-2 border-t border-slate-800 flex justify-between">
              <span>LHI: 88.6</span>
              <span>76% Prob</span>
            </div>
          </div>

          <div className="p-4 bg-slate-900/90 rounded-lg border border-emerald-500/30 bg-emerald-950/10 space-y-2 relative">
            <div className="text-[10px] uppercase font-bold text-emerald-400 tracking-wider">
              +24 MONTHS (TARGET)
            </div>
            <div className="text-base font-bold text-emerald-200">Principal AI Architect</div>
            <div className="text-xs text-slate-400">Strategic domain leadership</div>
            <div className="text-xs font-mono text-emerald-400 pt-2 border-t border-slate-800 flex justify-between">
              <span>LHI: 92.4</span>
              <span>71% Prob</span>
            </div>
          </div>
        </div>
      </HorizonCard>

      {/* ========================================================================= */}
      {/* LAYER 3: PERSONAL PORTFOLIO ENGINE (Comparing Life Paths) */}
      {/* ========================================================================= */}
      <HorizonCard
        title="Layer 3 // Personal Strategy Portfolio Engine"
        subtitle="Ranked evaluation of 4 strategic life trajectories across income, wellbeing, risk, and LHI"
      >
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {portfolioPaths.map((path) => {
              const isSelected = selectedPortfolioPath === path.id;
              return (
                <div
                  key={path.id}
                  onClick={() => setSelectedPortfolioPath(path.id)}
                  className={`p-4 rounded-lg border cursor-pointer transition-all space-y-3 ${
                    isSelected
                      ? 'bg-slate-900 border-emerald-500 shadow-lg shadow-emerald-950/30'
                      : 'bg-slate-900/50 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <span className="text-xs font-bold text-slate-300">{path.name}</span>
                    {path.recommended && (
                      <span className="text-[9px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-1.5 py-0.5 rounded font-bold">
                        RECOMMENDED
                      </span>
                    )}
                  </div>

                  <div className="space-y-1 text-xs text-slate-400">
                    <div className="flex justify-between">
                      <span>Expected LHI:</span>
                      <span className="font-mono font-bold text-emerald-400">{path.lhi}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Success Prob:</span>
                      <span className="font-mono text-slate-200">{path.probability}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Time Cost:</span>
                      <span className="font-mono text-slate-300">{path.timeCost}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Financial Yield:</span>
                      <span className="font-mono text-slate-300">{path.incomeTrend}</span>
                    </div>
                  </div>

                  <div className="pt-2 border-t border-slate-800 flex justify-between items-center text-[10px]">
                    <span className="text-slate-500">Risk Profile</span>
                    <span
                      className={`font-semibold ${
                        path.risk === 'LOW'
                          ? 'text-emerald-400'
                          : path.risk === 'MEDIUM'
                          ? 'text-amber-400'
                          : 'text-rose-400'
                      }`}
                    >
                      {path.risk}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </HorizonCard>

      {/* ========================================================================= */}
      {/* LAYER 4: DAILY ACTION ENGINE (Actionable Micro-Commitments) */}
      {/* ========================================================================= */}
      <HorizonCard
        title="Layer 4 // Daily Action Engine"
        subtitle="High-leverage micro-commitments with explicit causal outcome impact"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-slate-900/80 rounded-lg border border-slate-800 space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-xs font-semibold text-blue-400 uppercase tracking-wider">
                Craft Mastery
              </span>
              <span className="text-xs font-mono text-slate-400">4h this week</span>
            </div>
            <div className="text-sm font-semibold text-slate-200">
              System Design & Distributed Consensus Practice
            </div>
            <div className="text-xs text-slate-400">
              Complete 2 mock architecture walkthroughs for large-scale event streams.
            </div>
            <div className="pt-2 border-t border-slate-800 flex justify-between text-xs font-mono">
              <span className="text-emerald-400">+2.1 Career Impact</span>
              <span className="text-cyan-400">94% Trust</span>
            </div>
          </div>

          <div className="p-4 bg-slate-900/80 rounded-lg border border-slate-800 space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">
                Attention Defense
              </span>
              <span className="text-xs font-mono text-slate-400">Saves 45m</span>
            </div>
            <div className="text-sm font-semibold text-slate-200">
              Decline Non-Essential Friday Status Review
            </div>
            <div className="text-xs text-slate-400">
              Asynchronously post weekly bullet notes; protect deep cognitive block.
            </div>
            <div className="pt-2 border-t border-slate-800 flex justify-between text-xs font-mono">
              <span className="text-emerald-400">+12% Focus Reserve</span>
              <span className="text-cyan-400">96% Trust</span>
            </div>
          </div>

          <div className="p-4 bg-slate-900/80 rounded-lg border border-slate-800 space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-xs font-semibold text-pink-400 uppercase tracking-wider">
                Autonomic Balance
              </span>
              <span className="text-xs font-mono text-slate-400">35m today</span>
            </div>
            <div className="text-sm font-semibold text-slate-200">
              Aerobic Zone-2 Trail Run & Mobility
            </div>
            <div className="text-xs text-slate-400">
              Keep heart rate under 135 bpm to optimize mitochondrial recovery.
            </div>
            <div className="pt-2 border-t border-slate-800 flex justify-between text-xs font-mono">
              <span className="text-emerald-400">+8ms HRV Lift</span>
              <span className="text-cyan-400">98% Trust</span>
            </div>
          </div>
        </div>
      </HorizonCard>

      {/* ========================================================================= */}
      {/* LAYER 5: PERSONAL DRIFT DASHBOARD & FUTURE RECOVERY PROJECTIONS */}
      {/* ========================================================================= */}
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm uppercase tracking-wider font-bold text-slate-200">
              Layer 5 // Personal Drift & Future Recovery Projections
            </h2>
            <p className="text-xs text-slate-400">
              Continuous drift detection across 4 pillars and multi-strategy recovery simulations
              starting from the drifted state.
            </p>
          </div>
          <SeverityBadge level="LOW" status="HEALTHY" />
        </div>

        {/* 4 Drift Cards with Rails, Gauges, and Waterfalls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {CANONICAL_PERSONAL_DRIFT_CARDS.map((drift, idx) => {
            const gaugeProps = getDriftGaugeProperties(drift.driftPct);
            const rail = calculateProgressRail(drift.expected, drift.actual, drift.expected * 1.2);
            const isExpanded = expandedDriftIndex === idx;

            return (
              <div
                key={drift.domain}
                className="p-4 bg-slate-900/90 rounded-lg border border-slate-800 space-y-4"
              >
                <div className="flex justify-between items-start">
                  <div>
                    <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                      {drift.domain} DRIFT
                    </span>
                    <div className="text-sm font-semibold text-slate-200">{drift.metric}</div>
                  </div>
                  <span
                    className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${gaugeProps.badgeClass}`}
                  >
                    {drift.driftPct}% DRIFT
                  </span>
                </div>

                {/* Progress Rail */}
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between text-slate-400 font-mono text-[11px]">
                    <span>
                      Actual:{' '}
                      <strong className="text-slate-200">
                        {drift.actual} {drift.unit}
                      </strong>
                    </span>
                    <span>
                      Expected:{' '}
                      <strong className="text-slate-200">
                        {drift.expected} {drift.unit}
                      </strong>
                    </span>
                  </div>
                  <div className="relative w-full h-3 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="absolute top-0 bottom-0 bg-slate-700 w-1 rounded"
                      style={{ left: `${rail.expectedOffsetPct}%` }}
                      title="Expected Point"
                    />
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${rail.actualOffsetPct}%`,
                        backgroundColor: gaugeProps.color,
                      }}
                    />
                  </div>
                </div>

                {/* Recommendation */}
                <div className="text-xs text-slate-300 bg-slate-950/60 p-2.5 rounded border border-slate-800/80">
                  <span className="font-semibold text-emerald-400">Recalibration: </span>
                  {drift.recommendation}
                </div>

                {/* Toggleable Waterfall Root Causes */}
                <div>
                  <button
                    onClick={() => setExpandedDriftIndex(isExpanded ? null : idx)}
                    className="text-[11px] text-slate-400 hover:text-slate-200 transition-colors flex items-center gap-1 font-semibold"
                  >
                    <span>{isExpanded ? '▼ Hide' : '► View'} Causal Drift Waterfall</span>
                  </button>

                  {isExpanded && (
                    <div className="mt-2 pt-2 border-t border-slate-800 space-y-1.5 text-xs font-mono">
                      {drift.waterfallCauses.map((cause, cIdx) => (
                        <div key={cIdx} className="flex justify-between text-slate-400">
                          <span>• {cause.cause}</span>
                          <span className="text-rose-400">{cause.impact} pts</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Future Recovery Projection Lab */}
        <HorizonCard
          title="Future Recovery Projection Lab"
          subtitle="Simulating recovery trajectories from the drifted state: Which path closes the gap fastest without exceeding personal capacity?"
        >
          <div className="space-y-6">
            {/* Context Summary */}
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 text-center">
              <div className="p-3 bg-slate-900/60 rounded border border-slate-800">
                <div className="text-xs text-slate-400">CURRENT DRIFTED LHI</div>
                <div className="text-xl font-mono font-bold text-amber-400">
                  {recoveryData.currentLhi}
                </div>
                <div className="text-[10px] text-slate-500">Baseline before intervention</div>
              </div>

              <div className="p-3 bg-slate-900/60 rounded border border-slate-800">
                <div className="text-xs text-slate-400">DO-NOTHING 12M PROJECTION</div>
                <div className="text-xl font-mono font-bold text-slate-400">
                  {recoveryData.baselineProjectedLhi}
                </div>
                <div className="text-[10px] text-slate-500">+6 pts without re-optimization</div>
              </div>

              <div className="p-3 bg-slate-900/60 rounded border border-slate-800">
                <div className="text-xs text-slate-400">BEST RECOVERY LHI</div>
                <div className="text-xl font-mono font-bold text-emerald-400">
                  {activeStrategy.projectedLhi}
                </div>
                <div className="text-[10px] text-slate-500">Under {activeStrategy.name}</div>
              </div>

              <div className="p-3 bg-slate-900/60 rounded border border-slate-800">
                <div className="text-xs text-slate-400">MONTE CARLO RECOVERY</div>
                <div className="text-xl font-mono font-bold text-blue-400">
                  {recoveryData.monteCarloSimulations.p50Months} Months
                </div>
                <div className="text-[10px] text-slate-500">
                  {recoveryData.monteCarloSimulations.confidencePct}% Confidence (10k runs)
                </div>
              </div>
            </div>

            {/* Recovery Strategies Comparison Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border border-slate-800 rounded-lg overflow-hidden">
                <thead className="bg-slate-900 text-slate-400 uppercase tracking-wider text-[10px] font-mono">
                  <tr>
                    <th className="p-3">Strategy</th>
                    <th className="p-3">Time Cost</th>
                    <th className="p-3">Monthly Spend</th>
                    <th className="p-3">Recovery SLA</th>
                    <th className="p-3">Velocity (pts/mo)</th>
                    <th className="p-3">Projected LHI</th>
                    <th className="p-3">INV-OI83-P Status</th>
                    <th className="p-3">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {rankedStrategies.map((strat) => {
                    const isSelected = selectedStrategyId === strat.strategyId;
                    return (
                      <tr
                        key={strat.strategyId}
                        className={`transition-colors ${
                          isSelected ? 'bg-slate-900/90 font-semibold' : 'hover:bg-slate-900/40'
                        }`}
                      >
                        <td className="p-3 text-slate-200">
                          <div>{strat.name}</div>
                          <div className="text-[10px] text-slate-400 font-normal">
                            {strat.description}
                          </div>
                        </td>
                        <td className="p-3 font-mono text-slate-300">
                          {strat.weeklyHoursRequired}h / wk
                        </td>
                        <td className="p-3 font-mono text-slate-300">€{strat.monthlyCost}</td>
                        <td className="p-3 font-mono text-slate-300">
                          {strat.recoveryTimeMonths} Months
                        </td>
                        <td className="p-3 font-mono text-emerald-400 font-bold">
                          {strat.recoveryVelocity}
                        </td>
                        <td className="p-3 font-mono text-purple-300 font-bold">
                          {strat.projectedLhi}
                        </td>
                        <td className="p-3">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                              strat.isFeasible
                                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                                : 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
                            }`}
                          >
                            {strat.isFeasible ? 'FEASIBLE' : 'INFEASIBLE'}
                          </span>
                        </td>
                        <td className="p-3">
                          <button
                            onClick={() => setSelectedStrategyId(strat.strategyId)}
                            className={`px-2.5 py-1 rounded text-[10px] font-semibold uppercase tracking-wider transition-colors ${
                              isSelected
                                ? 'bg-emerald-600 text-white'
                                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                            }`}
                          >
                            {isSelected ? 'Active Plan' : 'Select'}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Selected Strategy Traceability Waterfall */}
            <div className="p-4 bg-slate-900/80 rounded-lg border border-slate-800 space-y-3">
              <div className="flex justify-between items-center text-xs">
                <span className="font-semibold text-slate-200 uppercase tracking-wider">
                  Causal Recovery Lineage // {activeStrategy.name}
                </span>
                <span className="text-xs text-emerald-400 font-mono font-bold">
                  Total LHI Lift: +{activeStrategy.projectedLhi - recoveryData.currentLhi} points
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
                {activeStrategy.traceabilityLineage.map((node, nIdx) => (
                  <div
                    key={nIdx}
                    className="p-2.5 bg-slate-950/60 rounded border border-slate-800/80 text-xs space-y-1 font-mono"
                  >
                    <div className="text-[10px] text-slate-500">STEP {nIdx + 1}</div>
                    <div className="text-slate-200 font-semibold">{node.step}</div>
                    <div className="text-emerald-400 text-[11px]">{node.delta}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </HorizonCard>
      </div>
    </div>
  );
}

export default function LifeCommandCenterPage() {
  return (
    <Suspense
      fallback={
        <div className="p-8 text-slate-400 bg-slate-950 min-h-screen">
          Loading Life Command Center...
        </div>
      }
    >
      <LifeCommandCenterContent />
    </Suspense>
  );
}
