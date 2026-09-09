'use client';

/**
 * Horizon 9: /me/trajectories — Multi-Year Household Trajectory Sequencing & Proactive Copilot
 *
 * Implements:
 * - Multi-Year State Transitions: State(t+1) = State(t) + Impact(t) + Compounding(t)
 * - Trajectory Candidate Permutations & Multi-Criteria Ranking
 * - Pareto Frontier Identification (Non-Dominated Trajectories)
 * - Interactive Causal Ripple Map (Magnitude, Impact Width, Polarity, Latency)
 * - Proactive Household Copilot with Explainable Recommendations (INV-OI92-P)
 * - Full Invariant Suite: INV-OI92-P through INV-OI96-P
 */

import React, { useState, useMemo, Suspense } from 'react';
import Link from 'next/link';
import IntelligenceHeader from '../../../components/ui/IntelligenceHeader';
import HorizonCard from '../../../components/ui/HorizonCard';
import HorizonMetricCard from '../../../components/ui/HorizonMetricCard';
import SeverityBadge from '../../../components/ui/SeverityBadge';
import {
  CANONICAL_TRAJECTORIES,
  calculateMultiCriteriaScore,
  identifyParetoFront,
  buildRippleMap,
  detectProactiveOpportunities,
} from '../../../lib/simulation/householdTrajectoryEngine';
import {
  HouseholdTrajectory,
  ExplainableRecommendation,
  RippleMapData,
} from '../../../types/personal-digital-twin';

function TrajectoriesContent() {
  const [selectedTrajectoryId, setSelectedTrajectoryId] = useState<string>('SEQ_ALPHA');
  const [selectedRippleIntervention, setSelectedRippleIntervention] = useState<string>('STRATEGIC_RELOCATION');

  const paretoTrajectories = useMemo(() => {
    return identifyParetoFront(CANONICAL_TRAJECTORIES);
  }, []);

  const activeTrajectory = useMemo(() => {
    return (
      paretoTrajectories.find((t) => t.trajectoryId === selectedTrajectoryId) ||
      paretoTrajectories[0]
    );
  }, [paretoTrajectories, selectedTrajectoryId]);

  const rippleData: RippleMapData = useMemo(() => {
    return buildRippleMap(selectedRippleIntervention);
  }, [selectedRippleIntervention]);

  const copilotRecommendations = useMemo(() => {
    return detectProactiveOpportunities({
      energyLevel: 88,
      allocatedWeeklyHours: 28,
      marketDemandIndex: 92,
      financialRunwayMonths: 14,
    });
  }, []);

  const activeRec: ExplainableRecommendation = copilotRecommendations[0];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 space-y-6 max-w-7xl mx-auto">
      {/* Top Breadcrumb Nav */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <Link
            href="/me"
            className="text-xs uppercase tracking-wider font-semibold text-slate-400 hover:text-white transition-colors"
          >
            ← Back to Life Command Center
          </Link>
          <span className="text-slate-600">/</span>
          <span className="text-xs uppercase tracking-wider font-semibold text-amber-400">
            Multi-Year Trajectory Sequencer (Y1-Y3)
          </span>
        </div>
        <div className="flex items-center gap-3">
          <Link
            href="/me/household"
            className="text-xs bg-slate-900 border border-slate-700 hover:border-slate-500 px-3 py-1.5 rounded-lg text-slate-300 hover:text-white transition-colors"
          >
            Household Twin →
          </Link>
          <Link
            href="/me/twin"
            className="text-xs bg-slate-900 border border-slate-700 hover:border-slate-500 px-3 py-1.5 rounded-lg text-slate-300 hover:text-white transition-colors"
          >
            Personal Twin →
          </Link>
          <SeverityBadge
            level={activeTrajectory.isParetoOptimal ? 'LOW' : 'MEDIUM'}
            status={activeTrajectory.isParetoOptimal ? 'PARETO EFFICIENT' : 'SUB-OPTIMAL'}
          />
        </div>
      </div>

      {/* Intelligence Header */}
      <IntelligenceHeader
        title="Multi-Year Household Trajectory Sequencer"
        subtitle="Dynamic State Transitions, Compounding Loops & Proactive Copilot · INV-OI92-P through INV-OI96-P"
        certification="CERTIFIED MULTI-YEAR SEQUENCER"
        status={activeTrajectory.stabilityScore >= 4.0 ? 'OPTIMAL' : 'WARNING'}
        replayHash="0xTRAJECTORY_H9"
        breadcrumbs={[
          { label: 'Home', href: '/' },
          { label: 'Life OS', href: '/me' },
          { label: 'Trajectories', href: '/me/trajectories' },
        ]}
      />

      {/* Metrics Banner */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <HorizonMetricCard
          label="Top Trajectory Score"
          value={`${activeTrajectory.multiCriteriaScore.toFixed(1)} / 100`}
          delta={activeTrajectory.isParetoOptimal ? 'Pareto Optimal' : 'Dominated'}
          deltaPositive={activeTrajectory.isParetoOptimal}
          subtext={`LHI: ${activeTrajectory.projectedLhi} | HHI: ${activeTrajectory.projectedHhi}`}
          severity={activeTrajectory.isParetoOptimal ? 'PASS' : 'WARN'}
        />
        <HorizonMetricCard
          label="Trajectory Stability (INV-OI93-P)"
          value={`${activeTrajectory.stabilityScore.toFixed(1)} μ/σ`}
          delta={activeTrajectory.stabilityScore >= 4.0 ? 'Top-Quartile Under 10k MC' : 'High Volatility'}
          deltaPositive={activeTrajectory.stabilityScore >= 4.0}
          subtext="Expected outcome divided by Monte Carlo variance"
          severity={activeTrajectory.stabilityScore >= 4.0 ? 'PASS' : 'WARN'}
        />
        <HorizonMetricCard
          label="Future Optionality (INV-OI95-P)"
          value={`${activeTrajectory.optionalityScore} / 100`}
          delta="Uncapped Career Paths"
          deltaPositive={true}
          subtext="24 mo runway + distributed AI specialization"
          severity="PASS"
        />
        <HorizonMetricCard
          label="Proactive Copilot (INV-OI92-P)"
          value="1 Active Opportunity"
          delta="100% Explainable"
          deltaPositive={true}
          subtext="Authoritative signal trigger & p10/p50/p90 bounds"
          severity="INFO"
        />
      </div>

      {/* Proactive Copilot Alert Card (INV-OI92-P) */}
      {activeRec && (
        <HorizonCard
          title="Proactive Household Copilot (INV-OI92-P)"
          badge="Live Opportunity Detected"
        >
          <div className="space-y-4 text-xs">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 p-3 bg-cyan-950/30 border border-cyan-700/50 rounded-xl">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[10px] bg-cyan-900/60 text-cyan-300 px-2 py-0.5 rounded font-bold">
                    SIGNAL DETECTED
                  </span>
                  <span className="font-bold text-white text-sm">{activeRec.title}</span>
                </div>
                <p className="text-slate-300 leading-relaxed text-[11px]">{activeRec.whyNow}</p>
              </div>
              <div className="flex items-center gap-3 font-mono">
                <div className="text-right">
                  <div className="text-[10px] text-slate-400">Expected Gain</div>
                  <div className="text-emerald-400 font-bold">
                    +{activeRec.expectedLhiImpact} LHI / +{activeRec.expectedHhiImpact} HHI
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[10px] text-slate-400">Confidence</div>
                  <div className="text-cyan-400 font-bold">{activeRec.confidencePct}%</div>
                </div>
              </div>
            </div>

            {/* Causal Lineage Path & Uncertainty Bounds */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 space-y-1.5">
                <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                  Causal Trace Path (Zero Black-Box)
                </span>
                <div className="flex flex-wrap items-center gap-1.5 font-mono text-[10px]">
                  {activeRec.tracePath.map((step, idx) => (
                    <React.Fragment key={idx}>
                      <span className="bg-slate-800 text-slate-300 px-2 py-0.5 rounded border border-slate-700">
                        {step}
                      </span>
                      {idx < activeRec.tracePath.length - 1 && <span className="text-slate-600">→</span>}
                    </React.Fragment>
                  ))}
                </div>
              </div>

              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 space-y-1.5">
                <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                  Monte Carlo Uncertainty Bounds
                </span>
                <div className="grid grid-cols-3 gap-2 text-center font-mono">
                  <div className="bg-slate-950 p-1.5 rounded border border-slate-800">
                    <div className="text-[9px] text-slate-500">p10 Downside</div>
                    <div className="font-bold text-amber-400">+{activeRec.p10.toFixed(1)}</div>
                  </div>
                  <div className="bg-slate-950 p-1.5 rounded border border-slate-800">
                    <div className="text-[9px] text-slate-500">p50 Expected</div>
                    <div className="font-bold text-cyan-400">+{activeRec.p50.toFixed(1)}</div>
                  </div>
                  <div className="bg-slate-950 p-1.5 rounded border border-slate-800">
                    <div className="text-[9px] text-slate-500">p90 Upside</div>
                    <div className="font-bold text-emerald-400">+{activeRec.p90.toFixed(1)}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </HorizonCard>
      )}

      {/* Trajectory Sequence Permutations */}
      <HorizonCard
        title="Multi-Year Trajectory Candidates (3-Year Horizon)"
        badge="Combinatorial Sequencing"
      >
        <div className="space-y-4">
          <p className="text-xs text-slate-400">
            Order changes everything. Compare multi-year sequence permutations to discover how timing and state
            transitions compound over 36 months.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {paretoTrajectories.map((tr) => {
              const isSelected = tr.trajectoryId === selectedTrajectoryId;
              return (
                <button
                  key={tr.trajectoryId}
                  onClick={() => setSelectedTrajectoryId(tr.trajectoryId)}
                  className={`text-left p-4 rounded-xl border transition-all ${
                    isSelected
                      ? 'bg-amber-950/40 border-amber-500 shadow-lg shadow-amber-950/30'
                      : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span
                      className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
                        tr.isParetoOptimal
                          ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                          : 'bg-slate-800 text-slate-400 border border-slate-700'
                      }`}
                    >
                      {tr.isParetoOptimal ? '★ PARETO FRONT' : 'SUB-OPTIMAL'}
                    </span>
                    <span className="text-xs font-mono font-bold text-white">
                      Score: {tr.multiCriteriaScore}
                    </span>
                  </div>
                  <h4 className="text-sm font-semibold text-white mb-1">{tr.name}</h4>
                  <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed mb-3">
                    {tr.description}
                  </p>
                  <div className="grid grid-cols-2 gap-1 font-mono text-[10px] text-slate-400 pt-2 border-t border-slate-800">
                    <div>LHI: <span className="text-white font-bold">{tr.projectedLhi}</span></div>
                    <div>HHI: <span className="text-white font-bold">{tr.projectedHhi}</span></div>
                    <div>Stability: <span className="text-cyan-400 font-bold">{tr.stabilityScore}</span></div>
                    <div>Optionality: <span className="text-amber-400 font-bold">{tr.optionalityScore}</span></div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </HorizonCard>

      {/* Trajectory Step-by-Step State Evolution & Visual Ripple Map */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Col: Step-by-Step 3-Year State Progression (6 cols) */}
        <div className="lg:col-span-6 space-y-6">
          <HorizonCard
            title={`${activeTrajectory.name} — 3-Year State Evolution`}
            badge="Sequential Compounding"
          >
            <div className="space-y-4">
              <div className="space-y-3">
                {activeTrajectory.steps.map((step) => (
                  <div
                    key={step.stepId}
                    className="bg-slate-900/80 p-3.5 rounded-xl border border-slate-800 space-y-2"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-[10px] bg-amber-950/60 text-amber-300 px-2 py-0.5 rounded font-bold border border-amber-800/60">
                          YEAR {step.year}
                        </span>
                        <span className="font-bold text-white">{step.title}</span>
                      </div>
                      <div className="font-mono text-[11px] text-slate-300">
                        {step.annualCapitalDelta >= 0 ? `+$${step.annualCapitalDelta.toLocaleString()}` : `-$${Math.abs(step.annualCapitalDelta).toLocaleString()}`}
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-2 font-mono text-[10px] pt-1 border-t border-slate-800/60">
                      <div>
                        <span className="text-slate-500">Δ Personal LHI: </span>
                        <span className={step.personalLhiImpact >= 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                          {step.personalLhiImpact >= 0 ? '+' : ''}{step.personalLhiImpact}
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500">Δ Household HHI: </span>
                        <span className={step.householdHhiImpact >= 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                          {step.householdHhiImpact >= 0 ? '+' : ''}{step.householdHhiImpact}
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500">Discretionary: </span>
                        <span className={step.discretionaryHoursDelta >= 0 ? 'text-cyan-400' : 'text-amber-400'}>
                          {step.discretionaryHoursDelta >= 0 ? '+' : ''}{step.discretionaryHoursDelta}h/wk
                        </span>
                      </div>
                    </div>

                    {/* Post-step state snapshot */}
                    <div className="flex flex-wrap gap-1.5 pt-1">
                      {Object.entries(step.stateAfterStep).map(([k, v]) => (
                        <span key={k} className="text-[9px] font-mono bg-slate-950 text-slate-400 px-1.5 py-0.5 rounded border border-slate-800">
                          {k}: <span className="text-white font-bold">{v}</span>
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              {/* Unintended consequences */}
              {activeTrajectory.unintendedConsequences.length > 0 && (
                <div className="pt-2 border-t border-slate-800 space-y-1.5">
                  <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                    Audited Multi-Year Trajectory Trade-Offs:
                  </div>
                  {activeTrajectory.unintendedConsequences.map((c, i) => (
                    <div key={i} className="text-[11px] text-slate-300 bg-amber-950/20 border border-amber-900/30 p-2 rounded flex items-start gap-1.5">
                      <span className="text-amber-400 font-bold">⚠</span>
                      <span>{c}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </HorizonCard>
        </div>

        {/* Right Col: Interactive Visual Ripple Map (6 cols) */}
        <div className="lg:col-span-6 space-y-6">
          <HorizonCard
            title="Interactive Visual Ripple Map"
            badge="Node Magnitude & Impact Width"
          >
            <div className="space-y-4 text-xs">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <span className="text-slate-400">Select Causal Trigger:</span>
                <div className="flex gap-2">
                  <button
                    onClick={() => setSelectedRippleIntervention('STRATEGIC_RELOCATION')}
                    className={`px-2 py-1 rounded text-[11px] font-semibold transition-all ${
                      selectedRippleIntervention === 'STRATEGIC_RELOCATION'
                        ? 'bg-rose-500 text-slate-950'
                        : 'bg-slate-900 text-slate-400 hover:text-white'
                    }`}
                  >
                    Relocation Ripple
                  </button>
                  <button
                    onClick={() => setSelectedRippleIntervention('EXECUTIVE_MASTERS')}
                    className={`px-2 py-1 rounded text-[11px] font-semibold transition-all ${
                      selectedRippleIntervention === 'EXECUTIVE_MASTERS'
                        ? 'bg-indigo-500 text-white'
                        : 'bg-slate-900 text-slate-400 hover:text-white'
                    }`}
                  >
                    Master&apos;s Ripple
                  </button>
                </div>
              </div>

              {/* Ripple Network View */}
              <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800 space-y-3">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="font-bold text-white uppercase tracking-wider">
                    Propagation Tree: {rippleData.rootIntervention}
                  </span>
                  <div className="flex items-center gap-2 font-mono">
                    <span className="text-emerald-400">Net LHI: {rippleData.netLhiDelta >= 0 ? '+' : ''}{rippleData.netLhiDelta}</span>
                    <span className="text-slate-600">|</span>
                    <span className={rippleData.netHhiDelta >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      Net HHI: {rippleData.netHhiDelta >= 0 ? '+' : ''}{rippleData.netHhiDelta}
                    </span>
                  </div>
                </div>

                {/* Nodes List */}
                <div className="space-y-2">
                  {rippleData.nodes.map((node) => {
                    const isRoot = node.id === 'ROOT';
                    const isPositive = node.delta >= 0;
                    return (
                      <div
                        key={node.id}
                        className={`flex items-center justify-between p-2 rounded-lg border ${
                          isRoot
                            ? 'bg-slate-800/90 border-slate-700 font-bold text-white'
                            : 'bg-slate-950/60 border-slate-800/80 text-slate-300'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <div
                            className={`w-2 h-2 rounded-full ${
                              isRoot
                                ? 'bg-amber-400'
                                : isPositive
                                ? 'bg-emerald-400'
                                : 'bg-rose-400'
                            }`}
                          />
                          <span className="text-[11px]">{node.label}</span>
                          <span className="text-[9px] font-mono text-slate-500 uppercase">
                            ({node.domain})
                          </span>
                        </div>
                        {!isRoot && (
                          <div className="font-mono text-[11px] font-bold">
                            <span className={isPositive ? 'text-emerald-400' : 'text-rose-400'}>
                              {isPositive ? '+' : ''}{node.delta} {node.unit}
                            </span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Edges List with Latency & Polarity */}
                <div className="pt-2 border-t border-slate-800 space-y-1">
                  <div className="text-[10px] uppercase font-semibold text-slate-500 tracking-wider">
                    Causal Coupling & Latency:
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 text-[10px] font-mono">
                    {rippleData.edges.map((e) => (
                      <div
                        key={e.id}
                        className="bg-slate-950/40 p-1.5 rounded border border-slate-800/60 flex items-center justify-between"
                      >
                        <div className="flex items-center gap-1">
                          <span className={e.isPositive ? 'text-emerald-400' : 'text-rose-400'}>
                            {e.isPositive ? '↗' : '↘'}
                          </span>
                          <span className="text-slate-400">{e.from} → {e.to}</span>
                        </div>
                        <span className="text-slate-500">{e.latencyWeeks}w lag</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </HorizonCard>
        </div>
      </div>
    </div>
  );
}

export default function TrajectoriesPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-slate-500">Loading Trajectories...</div>}>
      <TrajectoriesContent />
    </Suspense>
  );
}
