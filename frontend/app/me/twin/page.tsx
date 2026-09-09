'use client';

/**
 * Horizon 7: /me/twin — Integrated Life Twin & Unified Life Causal Graph
 *
 * Implements:
 * - One Causal Model of Life (Sleep -> Recovery -> Energy -> Focus -> Practice -> Skill -> Poise -> Comp -> Savings -> LHI)
 * - 18 Nodes across 6 Domains (HEALTH, CAREER, LEARNING, FINANCE, RELATIONSHIPS, TIME)
 * - 24 Directed Causal Edges with Sensitivities & Latencies (Strict DAG)
 * - INV-OI88-P (Cross-Domain Traceability Invariant)
 * - INV-OI89-P (Cross-Domain Consistency Invariant & Collateral Drag Auditor)
 * - Multi-Domain Monte Carlo Uncertainty Distribution (10,000 runs)
 */

import React, { useState, useMemo, Suspense } from 'react';
import Link from 'next/link';
import IntelligenceHeader from '../../../components/ui/IntelligenceHeader';
import HorizonCard from '../../../components/ui/HorizonCard';
import HorizonMetricCard from '../../../components/ui/HorizonMetricCard';
import SeverityBadge from '../../../components/ui/SeverityBadge';
import {
  CANONICAL_LIFE_NODES,
  CANONICAL_LIFE_EDGES,
  CANONICAL_SCENARIOS,
  buildUnifiedLifeGraph,
  computeDomainNormalizedScores,
  simulateCrossDomainScenario,
} from '../../../lib/simulation/unifiedLifeGraphEngine';
import {
  LifeDomainType,
  CrossDomainScenario,
  CrossDomainSimulationResult,
} from '../../../types/personal-digital-twin';

const DOMAIN_COLORS: Record<LifeDomainType, { text: string; bg: string; border: string; bar: string }> = {
  HEALTH: { text: 'text-emerald-400', bg: 'bg-emerald-950/40', border: 'border-emerald-700/50', bar: 'bg-emerald-500' },
  CAREER: { text: 'text-sky-400', bg: 'bg-sky-950/40', border: 'border-sky-700/50', bar: 'bg-sky-500' },
  LEARNING: { text: 'text-indigo-400', bg: 'bg-indigo-950/40', border: 'border-indigo-700/50', bar: 'bg-indigo-500' },
  FINANCE: { text: 'text-amber-400', bg: 'bg-amber-950/40', border: 'border-amber-700/50', bar: 'bg-amber-500' },
  RELATIONSHIPS: { text: 'text-rose-400', bg: 'bg-rose-950/40', border: 'border-rose-700/50', bar: 'bg-rose-500' },
  TIME: { text: 'text-purple-400', bg: 'bg-purple-950/40', border: 'border-purple-700/50', bar: 'bg-purple-500' },
};

function TwinContent() {
  const [scenarios] = useState<CrossDomainScenario[]>(CANONICAL_SCENARIOS);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('ai_architect_pivot');
  const [selectedDomainFilter, setSelectedDomainFilter] = useState<LifeDomainType | 'ALL'>('ALL');
  const [showAllTraceSteps, setShowAllTraceSteps] = useState<boolean>(false);

  const graph = useMemo(() => buildUnifiedLifeGraph(), []);

  const activeScenario = useMemo(() => {
    return scenarios.find((s) => s.id === selectedScenarioId) || scenarios[0];
  }, [scenarios, selectedScenarioId]);

  const simResult: CrossDomainSimulationResult = useMemo(() => {
    return simulateCrossDomainScenario(activeScenario, graph);
  }, [activeScenario, graph]);

  // Filtered nodes for the topology inspector
  const filteredNodes = useMemo(() => {
    const all = Object.values(graph.nodes);
    if (selectedDomainFilter === 'ALL') return all;
    return all.filter((n) => n.domain === selectedDomainFilter);
  }, [graph.nodes, selectedDomainFilter]);

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
          <span className="text-xs uppercase tracking-wider font-semibold text-cyan-400">
            Integrated Life Twin & Causal Graph
          </span>
        </div>
        <div className="flex items-center gap-3">
          <Link
            href="/me/signals"
            className="text-xs bg-slate-900 border border-slate-700 hover:border-slate-500 px-3 py-1.5 rounded-lg text-slate-300 hover:text-white transition-colors"
          >
            Signal Layer →
          </Link>
          <Link
            href="/me/allocator"
            className="text-xs bg-slate-900 border border-slate-700 hover:border-slate-500 px-3 py-1.5 rounded-lg text-slate-300 hover:text-white transition-colors"
          >
            168h Allocator →
          </Link>
          <SeverityBadge
            level={simResult.isConsistent && simResult.isTraceable ? 'LOW' : 'HIGH'}
            status={simResult.isConsistent && simResult.isTraceable ? 'PASS' : 'FAIL'}
          />
        </div>
      </div>

      {/* Enterprise / Life Intelligence Header */}
      <IntelligenceHeader
        title="Unified Life Causal Graph"
        subtitle="18-Node Cross-Domain Causal Simulation & Uncertainty Architecture · INV-OI88-P & INV-OI89-P"
        certification="CERTIFIED LIFE CAUSAL GRAPH (DAG)"
        status={simResult.isConsistent ? 'OPTIMAL' : 'WARNING'}
        replayHash="0xCAUSAL_TWIN_H7"
        breadcrumbs={[
          { label: 'Home', href: '/' },
          { label: 'Life OS', href: '/me' },
          { label: 'Twin Lab', href: '/me/twin' },
        ]}
      />

      {/* Metric Cards Banner */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <HorizonMetricCard
          label="Composite Life Health Index"
          value={`${simResult.projectedLhi.toFixed(1)} / 100`}
          delta={`${simResult.lhiDelta >= 0 ? '+' : ''}${simResult.lhiDelta.toFixed(1)} pts`}
          deltaPositive={simResult.lhiDelta >= 0}
          subtext={`Baseline: ${simResult.baselineLhi.toFixed(1)} across 6 domains`}
          severity={simResult.lhiDelta >= 0 ? 'PASS' : 'WARN'}
        />
        <HorizonMetricCard
          label="Causal Graph Architecture"
          value="18 Nodes · 24 Edges"
          delta="DAG: 0 Cycles"
          deltaPositive={true}
          subtext="6 Domains: Health, Career, Learning, Finance, Rel, Time"
          severity="PASS"
        />
        <HorizonMetricCard
          label="Traceability (INV-OI88-P)"
          value={simResult.isTraceable ? '100% Causal Chain' : 'Trace Broken'}
          delta={`${simResult.traceLineage.length} Step Propagation`}
          deltaPositive={simResult.isTraceable}
          subtext="Direct lineage from levers to downstream nodes"
          severity={simResult.isTraceable ? 'PASS' : 'CRITICAL'}
        />
        <HorizonMetricCard
          label="Monte Carlo Uncertainty (p50)"
          value={`${simResult.monteCarloDistribution.p50 >= 0 ? '+' : ''}${simResult.monteCarloDistribution.p50.toFixed(1)} pts`}
          delta={`p10: ${simResult.monteCarloDistribution.p10.toFixed(1)} | p90: ${simResult.monteCarloDistribution.p90.toFixed(1)}`}
          deltaPositive={simResult.monteCarloDistribution.p50 >= 0}
          subtext="10,000 deterministic seeded iterations"
          severity="INFO"
        />
      </div>

      {/* Scenario Selector & Intervention Workbench */}
      <HorizonCard
        title="Cross-Domain Strategic Scenarios"
        badge="Multi-Domain Levers"
      >
        <div className="space-y-4">
          <p className="text-xs text-slate-400">
            Select a multi-domain intervention scenario to trigger the causal propagation engine across all 18 nodes.
            Observe real-time primary benefits alongside involuntary second-order collateral drags.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {scenarios.map((sc) => {
              const isSelected = sc.id === selectedScenarioId;
              return (
                <button
                  key={sc.id}
                  onClick={() => setSelectedScenarioId(sc.id)}
                  className={`text-left p-4 rounded-xl border transition-all ${
                    isSelected
                      ? 'bg-cyan-950/40 border-cyan-500 shadow-lg shadow-cyan-950/30'
                      : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                      {sc.horizonWeeks}w Horizon
                    </span>
                    <span
                      className={`text-xs font-bold px-2 py-0.5 rounded ${
                        sc.lhiDelta >= 0
                          ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-800/60'
                          : 'bg-rose-950/80 text-rose-300 border border-rose-800/60'
                      }`}
                    >
                      {sc.lhiDelta >= 0 ? '+' : ''}
                      {sc.lhiDelta.toFixed(1)} LHI
                    </span>
                  </div>
                  <h4 className="text-sm font-semibold text-white mb-1.5">{sc.title}</h4>
                  <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed mb-3">
                    {sc.description}
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {Object.entries(sc.leverChanges).map(([lever, val]) => (
                      <span
                        key={lever}
                        className="text-[10px] font-mono bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded border border-slate-700/60"
                      >
                        {lever.replace('_HOURS', '').replace('_SCORE', '').replace('_CONTRIBUTION', '')}: {val > 0 ? `+${val}` : val}
                      </span>
                    ))}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </HorizonCard>

      {/* Main Dual Grid: Multi-Domain Decomposition & Causal Lineage */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Col: Multi-Domain Decomposition & Invariant Audit (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Domain Score Breakdown Card */}
          <HorizonCard
            title="LHI Multi-Domain Decomposition"
            badge="0.25H + 0.20C + 0.15F + 0.15L + 0.15R + 0.10T"
          >
            <div className="space-y-4">
              <div className="space-y-3">
                {(Object.keys(simResult.domainScores) as LifeDomainType[]).map((domain) => {
                  const data = simResult.domainScores[domain];
                  const cfg = DOMAIN_COLORS[domain];
                  const isPositive = data.delta >= 0;
                  return (
                    <div key={domain} className="bg-slate-900/70 p-3 rounded-lg border border-slate-800">
                      <div className="flex items-center justify-between text-xs mb-1.5">
                        <div className="flex items-center gap-2">
                          <span className={`font-bold ${cfg.text}`}>{domain}</span>
                          <span className="text-slate-500 font-mono text-[10px]">
                            {domain === 'HEALTH' && 'wt: 25%'}
                            {domain === 'CAREER' && 'wt: 20%'}
                            {domain === 'LEARNING' && 'wt: 15%'}
                            {domain === 'FINANCE' && 'wt: 15%'}
                            {domain === 'RELATIONSHIPS' && 'wt: 15%'}
                            {domain === 'TIME' && 'wt: 10%'}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 font-mono">
                          <span className="text-slate-400">{data.baseline.toFixed(1)}</span>
                          <span className="text-slate-600">→</span>
                          <span className="text-white font-bold">{data.projected.toFixed(1)}</span>
                          <span
                            className={`text-[11px] font-bold px-1.5 py-0.2 rounded ${
                              isPositive
                                ? 'text-emerald-400 bg-emerald-950/40'
                                : 'text-rose-400 bg-rose-950/40'
                            }`}
                          >
                            {isPositive ? '+' : ''}
                            {data.delta.toFixed(1)}
                          </span>
                        </div>
                      </div>
                      {/* Bar comparison */}
                      <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden relative">
                        <div
                          className={`h-full ${cfg.bar} transition-all duration-500`}
                          style={{ width: `${Math.min(100, Math.max(0, data.projected))}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Monte Carlo Uncertainty Distribution */}
              <div className="pt-3 border-t border-slate-800">
                <div className="flex items-center justify-between text-xs mb-2">
                  <span className="text-slate-400 font-semibold uppercase tracking-wider">
                    Uncertainty Distribution (10,000 Runs)
                  </span>
                  <span className="text-slate-500 font-mono text-[11px]">
                    {activeScenario.horizonWeeks} Weeks Horizon
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono">
                  <div className="bg-slate-900 p-2 rounded border border-slate-800">
                    <div className="text-slate-500 text-[10px]">p10 (Downside)</div>
                    <div className="font-bold text-amber-400">
                      {simResult.monteCarloDistribution.p10 >= 0 ? '+' : ''}
                      {simResult.monteCarloDistribution.p10.toFixed(1)} pts
                    </div>
                  </div>
                  <div className="bg-slate-900 p-2 rounded border border-slate-800">
                    <div className="text-slate-500 text-[10px]">p50 (Expected)</div>
                    <div className="font-bold text-cyan-400">
                      {simResult.monteCarloDistribution.p50 >= 0 ? '+' : ''}
                      {simResult.monteCarloDistribution.p50.toFixed(1)} pts
                    </div>
                  </div>
                  <div className="bg-slate-900 p-2 rounded border border-slate-800">
                    <div className="text-slate-500 text-[10px]">p90 (Upside)</div>
                    <div className="font-bold text-emerald-400">
                      {simResult.monteCarloDistribution.p90 >= 0 ? '+' : ''}
                      {simResult.monteCarloDistribution.p90.toFixed(1)} pts
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </HorizonCard>

          {/* Invariant Governance & Collateral Drag Audit */}
          <HorizonCard
            title="Invariant Governance & Collateral Drags"
            badge="INV-OI88-P & INV-OI89-P"
          >
            <div className="space-y-4 text-xs">
              {/* INV-OI88-P Traceability */}
              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-300">
                    INV-OI88-P: Cross-Domain Traceability
                  </span>
                  <span
                    className={`font-bold px-2 py-0.5 rounded text-[10px] ${
                      simResult.isTraceable
                        ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                        : 'bg-rose-950 text-rose-400 border border-rose-800'
                    }`}
                  >
                    {simResult.isTraceable ? 'CERTIFIED PASS' : 'FAIL'}
                  </span>
                </div>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Every node perturbation has an unbroken causal chain traced back to primary scenario levers.
                </p>
              </div>

              {/* INV-OI89-P Consistency */}
              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-300">
                    INV-OI89-P: Cross-Domain Consistency
                  </span>
                  <span
                    className={`font-bold px-2 py-0.5 rounded text-[10px] ${
                      simResult.isConsistent
                        ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                        : 'bg-amber-950 text-amber-400 border border-amber-800'
                    }`}
                  >
                    {simResult.isConsistent ? 'CERTIFIED PASS' : 'CHECK REQUIRED'}
                  </span>
                </div>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Physical limits and biological sleep floors enforced. No simulation is allowed to hide collateral drags.
                </p>
                {simResult.consistencyViolations.length > 0 && (
                  <div className="space-y-1 pt-1">
                    {simResult.consistencyViolations.map((v, i) => (
                      <div key={i} className="text-rose-400 bg-rose-950/30 p-1.5 rounded text-[11px] border border-rose-900/40">
                        {v}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Acknowledged Unintended Consequences */}
              <div className="space-y-2">
                <h5 className="font-semibold text-slate-300 uppercase tracking-wider text-[11px]">
                  Audited Collateral Consequences ({activeScenario.unintendedConsequences.length})
                </h5>
                <div className="space-y-1.5">
                  {activeScenario.unintendedConsequences.map((consequence, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-2 bg-amber-950/20 border border-amber-900/30 p-2 rounded text-slate-300 text-[11px]"
                    >
                      <span className="text-amber-400 font-bold">⚠</span>
                      <span>{consequence}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </HorizonCard>
        </div>

        {/* Right Col: Causal Lineage & Node Graph Explorer (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Step-by-Step Causal Trace Card */}
          <HorizonCard
            title={`Causal Propagation Lineage (${simResult.traceLineage.length} Steps)`}
            badge="Topological Propagation"
          >
            <div className="space-y-4">
              <p className="text-xs text-slate-400">
                Deterministic DAG forward simulation from initial levers through all downstream causal dependencies.
              </p>

              <div className="space-y-2 max-h-[460px] overflow-y-auto pr-1">
                {(showAllTraceSteps ? simResult.traceLineage : simResult.traceLineage.slice(0, 10)).map((trace) => {
                  const cfg = DOMAIN_COLORS[trace.domain];
                  const isDeltaPos = trace.delta >= 0;
                  return (
                    <div
                      key={trace.step}
                      className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 hover:border-slate-700 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-1 text-xs">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-[10px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded">
                            Step {trace.step}
                          </span>
                          <span className={`font-bold ${cfg.text}`}>{trace.nodeName}</span>
                          <span className="text-slate-500 text-[10px]">({trace.domain})</span>
                        </div>
                        <div className="flex items-center gap-2 font-mono text-xs">
                          <span className="text-slate-400">{trace.priorValue}</span>
                          <span className="text-slate-600">→</span>
                          <span className="text-white font-bold">{trace.newValue}</span>
                          <span
                            className={`font-bold px-1.5 py-0.2 rounded text-[11px] ${
                              isDeltaPos
                                ? 'text-emerald-400 bg-emerald-950/40'
                                : 'text-rose-400 bg-rose-950/40'
                            }`}
                          >
                            {isDeltaPos ? '+' : ''}
                            {trace.delta}
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1">
                        <span className="italic">{trace.mechanism || 'Direct intervention'}</span>
                        {trace.causedByEdgeId && (
                          <span className="font-mono text-[10px] text-cyan-400/80">
                            Edge: {trace.causedByEdgeId} ({trace.latencyWeeksCumulative}w lag)
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {simResult.traceLineage.length > 10 && (
                <button
                  onClick={() => setShowAllTraceSteps(!showAllTraceSteps)}
                  className="w-full text-center py-2 text-xs text-cyan-400 hover:text-cyan-300 border border-dashed border-slate-800 hover:border-slate-700 rounded-lg transition-colors"
                >
                  {showAllTraceSteps
                    ? 'Show Fewer Steps'
                    : `Show All ${simResult.traceLineage.length} Causal Steps (+${simResult.traceLineage.length - 10} more)`}
                </button>
              )}
            </div>
          </HorizonCard>

          {/* Causal Graph Node Directory */}
          <HorizonCard
            title={`Life Graph Topology (${filteredNodes.length} Nodes)`}
            badge="Cycle-Free DAG"
          >
            <div className="space-y-4">
              {/* Domain Filter Pills */}
              <div className="flex flex-wrap gap-1.5">
                {(['ALL', 'HEALTH', 'CAREER', 'LEARNING', 'FINANCE', 'RELATIONSHIPS', 'TIME'] as const).map((dom) => (
                  <button
                    key={dom}
                    onClick={() => setSelectedDomainFilter(dom)}
                    className={`text-xs px-2.5 py-1 rounded-full font-semibold transition-all ${
                      selectedDomainFilter === dom
                        ? 'bg-cyan-500 text-slate-950 shadow-sm'
                        : 'bg-slate-900 text-slate-400 border border-slate-800 hover:text-white'
                    }`}
                  >
                    {dom}
                  </button>
                ))}
              </div>

              {/* Node List Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-h-[380px] overflow-y-auto pr-1">
                {filteredNodes.map((node) => {
                  const cfg = DOMAIN_COLORS[node.domain];
                  return (
                    <div
                      key={node.id}
                      className="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800/80 space-y-1"
                    >
                      <div className="flex items-center justify-between text-xs">
                        <span className={`font-semibold ${cfg.text} truncate`}>{node.name}</span>
                        <span className="font-mono text-slate-300 font-bold">
                          {node.baselineValue} <span className="text-[10px] text-slate-500">{node.unit}</span>
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-400 line-clamp-1">{node.description}</p>
                      <div className="flex items-center justify-between text-[9px] text-slate-500 font-mono pt-1 border-t border-slate-800/50">
                        <span>Safe: {node.minSafeValue} - {node.maxSafeValue}</span>
                        <span>{node.domain}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </HorizonCard>
        </div>
      </div>
    </div>
  );
}

export default function TwinPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-slate-500">Loading Life Twin...</div>}>
      <TwinContent />
    </Suspense>
  );
}
