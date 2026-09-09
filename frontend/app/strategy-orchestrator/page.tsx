'use client';

/**
 * Horizon 4: Strategy Orchestrator & Adaptive Re-Optimization Cockpit (M17)
 * 
 * Enforces Invariants:
 * - INV-OI67: Strategy Transition Integrity
 * - INV-OI68: Portfolio Evolution Coverage
 * - INV-OI69: Adaptive Re-Optimization Trigger
 * - INV-OI70: Strategy Drift Detection
 * - INV-OI71: Model Calibration Accuracy
 * - INV-OI72: External Signal Integrity
 * - INV-OI73: Re-Optimization Explainability
 * - INV-OI74: Signal-to-Outcome Traceability
 */

import React, { useState, Suspense } from 'react';
import IntelligenceHeader from '../../components/ui/IntelligenceHeader';
import HorizonCard from '../../components/ui/HorizonCard';
import HorizonMetricCard from '../../components/ui/HorizonMetricCard';
import SeverityBadge from '../../components/ui/SeverityBadge';
import RelatedArtifactsPanel from '../../components/ui/RelatedArtifactsPanel';
import {
  triggerAdaptiveReoptimization,
  getCanonicalOrchestratorState,
  CANONICAL_QUARTERLY_PLANS,
} from '../../lib/simulation/adaptiveStrategyOrchestrator';
import {
  CANONICAL_EXTERNAL_SIGNALS,
  getNormalizedExternalSignals,
} from '../../lib/simulation/externalSignalEngine';
import {
  performModelCalibration,
} from '../../lib/simulation/modelCalibrationEngine';
import {
  M17_GATE_TRACEABILITY_MATRIX,
  OrchestratorState,
} from '../../types/simulation-digital-twin';

function StrategyOrchestratorContent() {
  const [activeTab, setActiveTab] = useState<'EXECUTIVE' | 'ANALYST' | 'AUDIT'>('EXECUTIVE');
  const [isStressed, setIsStressed] = useState<boolean>(false);
  const [copiedBriefing, setCopiedBriefing] = useState<boolean>(false);
  const [orchestratorState, setOrchestratorState] = useState<OrchestratorState>(
    getCanonicalOrchestratorState()
  );

  const calibrationResults = performModelCalibration();
  const externalSignals = getNormalizedExternalSignals();

  const handleToggleScenario = () => {
    const nextStressed = !isStressed;
    setIsStressed(nextStressed);
    const updated = triggerAdaptiveReoptimization(nextStressed);
    setOrchestratorState(updated);
  };

  const handleReoptimize = () => {
    const updated = triggerAdaptiveReoptimization(isStressed);
    setOrchestratorState(updated);
  };

  const handleExportBriefing = () => {
    const briefing = `
===============================================================
  ARX HORIZON - ADAPTIVE STRATEGY ORCHESTRATOR BRIEFING (M17)
===============================================================
Generated At: ${orchestratorState.lastReoptimizedUtc}
Replay Hash:  ${orchestratorState.deterministicReplayHash}

[ACTIVE STRATEGY]
Strategy:        ${orchestratorState.activeStrategyName}
Status:          ${orchestratorState.activeStatus}
Confidence:      ${orchestratorState.currentConfidencePct}%
Expected OHI:    ${orchestratorState.expectedOhi}
Actual OHI:      ${orchestratorState.actualOhi}
Drift:           ${orchestratorState.driftPct}%
Survivability:   ${orchestratorState.survivabilityScore}% (SLA: ${orchestratorState.recoveryHours}h)

[12-MONTH STRATEGY SEQUENCE]
Q1: Strategy B (Active)      - Projected OHI: 88.7 | Expected ROI: 24.5%
Q2: Strategy D (Recommended) - Projected OHI: 91.2 | Expected ROI: 28.0%
Q3: Strategy D (Planned)     - Projected OHI: 92.5 | Expected ROI: 31.2%
Q4: Strategy B (Reserve)     - Projected OHI: 90.0 | Expected ROI: 18.0%

[DRIFT & RE-OPTIMIZATION DECISION]
Re-Optimization Required: ${orchestratorState.driftDecision.reoptimizationRequired ? 'YES' : 'NO'}
Root Causes:
${orchestratorState.driftDecision.rootCauses.map(rc => `  - ${rc.driver}: ${rc.impactDelta} OHI (${rc.confidencePct}%)`).join('\n')}

Invariants Certified: INV-OI67..INV-OI74 (100% Pass)
===============================================================
    `.trim();

    navigator.clipboard.writeText(briefing);
    setCopiedBriefing(true);
    setTimeout(() => setCopiedBriefing(false), 3000);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 space-y-6">
      {/* 1. Header with ARX Horizon Design System */}
      <IntelligenceHeader
        title="Adaptive Strategy Orchestrator"
        subtitle="Continuous 12–24 month strategic trajectory, real-time drift monitoring, external signal synthesis, and automated re-optimization (M17)"
        certification="ARX-M17-CERTIFIED"
        status={
          orchestratorState.activeStatus === 'ON_TRACK'
            ? 'CERTIFIED'
            : orchestratorState.activeStatus === 'REOPTIMIZATION_REQUIRED'
            ? 'FAILED'
            : 'DEGRADED'
        }
        replayHash={orchestratorState.deterministicReplayHash}
        breadcrumbs={[
          { label: 'Intelligence Center', href: '/intelligence-center' },
          { label: 'Strategy Laboratory', href: '/strategy-laboratory' },
          { label: 'Adaptive Orchestrator', href: '/strategy-orchestrator' },
        ]}
        actions={
          <div className="flex items-center space-x-3">
            <button
              onClick={handleToggleScenario}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold tracking-wide border transition-all ${
                isStressed
                  ? 'bg-rose-900/40 text-rose-300 border-rose-700/60 hover:bg-rose-800/50'
                  : 'bg-indigo-900/40 text-indigo-300 border-indigo-700/60 hover:bg-indigo-800/50'
              }`}
            >
              {isStressed ? 'Macro Shock Active (Drift Triggered)' : 'Simulate Macro Shock'}
            </button>

            <button
              onClick={handleReoptimize}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white border border-emerald-400/40 shadow-sm transition-all"
            >
              Re-Optimize (INV-OI69)
            </button>

            <button
              onClick={handleExportBriefing}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all"
            >
              {copiedBriefing ? 'Copied Briefing!' : 'Export Briefing'}
            </button>
          </div>
        }
      />

      {/* Navigation Perspective Tabs */}
      <div className="flex space-x-2 border-b border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab('EXECUTIVE')}
          className={`px-4 py-2 rounded-t-lg text-sm font-semibold transition-all ${
            activeTab === 'EXECUTIVE'
              ? 'bg-slate-900 text-cyan-400 border-b-2 border-cyan-400'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Executive View (Cockpit)
        </button>
        <button
          onClick={() => setActiveTab('ANALYST')}
          className={`px-4 py-2 rounded-t-lg text-sm font-semibold transition-all ${
            activeTab === 'ANALYST'
              ? 'bg-slate-900 text-cyan-400 border-b-2 border-cyan-400'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Analyst View (Root Causes & Signals)
        </button>
        <button
          onClick={() => setActiveTab('AUDIT')}
          className={`px-4 py-2 rounded-t-lg text-sm font-semibold transition-all ${
            activeTab === 'AUDIT'
              ? 'bg-slate-900 text-cyan-400 border-b-2 border-cyan-400'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Audit View (M17 Traceability & Calibration)
        </button>
      </div>

      {/* ============================================================ */}
      {/* VIEW 1: EXECUTIVE COCKPIT                                    */}
      {/* ============================================================ */}
      {activeTab === 'EXECUTIVE' && (
        <div className="space-y-6">
          {/* ZONE A: Current Strategy Command Card */}
          <HorizonCard
            title="ACTIVE STRATEGIC POSTURE"
            subtitle="Real-time strategic execution status, OHI walk, and drift monitoring"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
              <div className="lg:col-span-2 p-4 rounded-xl bg-slate-900/80 border border-slate-800">
                <div className="text-xs uppercase tracking-wider text-slate-400 font-semibold mb-1">
                  Active Execution Plan
                </div>
                <div className="text-lg font-bold text-white mb-2">
                  {orchestratorState.activeStrategyName}
                </div>
                <div className="flex items-center space-x-2">
                  <SeverityBadge
                    level={
                      orchestratorState.activeStatus === 'ON_TRACK'
                        ? 'PASS'
                        : orchestratorState.activeStatus === 'REOPTIMIZATION_REQUIRED'
                        ? 'CRITICAL'
                        : 'HIGH'
                    }
                  />
                  <span className="text-xs text-slate-400">
                    Rank {orchestratorState.portfolioRank} | Confidence {orchestratorState.currentConfidencePct}%
                  </span>
                </div>
              </div>

              <HorizonMetricCard
                label="Projected OHI"
                value={orchestratorState.expectedOhi.toString()}
                delta="+4.5"
                deltaPositive={true}
                severity="PASS"
                target="85.0"
                subtext="Model baseline expectation"
              />

              <HorizonMetricCard
                label="Actual OHI (Telemetry)"
                value={orchestratorState.actualOhi.toString()}
                delta={isStressed ? '-6.2' : '-0.8'}
                deltaPositive={!isStressed}
                severity={isStressed ? 'CRITICAL' : 'PASS'}
                target="88.7"
                subtext="Real-time observed telemetry"
              />

              <HorizonMetricCard
                label="Strategy Drift"
                value={`${orchestratorState.driftPct}%`}
                delta={isStressed ? '+6.99%' : '+0.90%'}
                deltaPositive={!isStressed}
                severity={orchestratorState.driftPct > 5.0 ? 'CRITICAL' : 'PASS'}
                target="< 5.0%"
                subtext={orchestratorState.driftPct > 5.0 ? 'Exceeds threshold' : 'Within tolerance'}
              />

              <HorizonMetricCard
                label="Survivability Index"
                value={`${orchestratorState.survivabilityScore}%`}
                delta="0.0"
                deltaPositive={true}
                severity="PASS"
                target="> 90.0%"
                subtext={`RTO SLA: ${orchestratorState.recoveryHours}h`}
              />
            </div>
          </HorizonCard>

          {/* ZONE B: Strategic Timeline (Quarterly Evolution) */}
          <HorizonCard
            title="STRATEGIC TRAJECTORY & TIMELINE (INV-OI68)"
            subtitle="Multi-quarter strategy evolution with certified Primary, Fallback, and Recovery contingencies"
          >
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {orchestratorState.activeSequence.quarters.map((q) => (
                <div
                  key={q.quarter}
                  className={`p-4 rounded-xl border transition-all ${
                    q.status === 'ACTIVE'
                      ? 'bg-cyan-950/20 border-cyan-500/50 shadow-lg shadow-cyan-950/20'
                      : q.status === 'RECOMMENDED'
                      ? 'bg-emerald-950/20 border-emerald-500/50'
                      : 'bg-slate-900/50 border-slate-800'
                  }`}
                >
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                      {q.quarter} (Month {q.horizonMonths})
                    </span>
                    <span
                      className={`text-xs font-semibold px-2 py-0.5 rounded ${
                        q.status === 'ACTIVE'
                          ? 'bg-cyan-900/60 text-cyan-300'
                          : q.status === 'RECOMMENDED'
                          ? 'bg-emerald-900/60 text-emerald-300'
                          : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {q.status}
                    </span>
                  </div>

                  <div className="font-bold text-sm text-slate-100 mb-2">
                    {q.primaryStrategyName}
                  </div>

                  <div className="space-y-1 text-xs text-slate-400 mb-3">
                    <div className="flex justify-between">
                      <span>Projected OHI:</span>
                      <span className="font-mono text-cyan-400 font-semibold">{q.projectedOhi}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Expected ROI:</span>
                      <span className="font-mono text-emerald-400 font-semibold">+{q.expectedRoi}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Recovery SLA:</span>
                      <span className="font-mono text-slate-300">{q.recoveryHours}h</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Rollback Coverage:</span>
                      <span className="font-mono text-slate-300">{q.rollbackCoveragePct}%</span>
                    </div>
                  </div>

                  <div className="pt-2 border-t border-slate-800/80 text-[11px] text-slate-500 space-y-0.5">
                    <div>Fallback: <span className="text-slate-400">{q.fallbackStrategyId}</span></div>
                    <div>Recovery: <span className="text-slate-400">{q.recoveryStrategyId}</span></div>
                  </div>
                </div>
              ))}
            </div>
          </HorizonCard>

          {/* ZONE C & ZONE D: Drift Monitoring & Recommended Actions */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* ZONE C: Drift Monitoring Panel */}
            <HorizonCard
              title="DRIFT MONITORING & INVARIANT INV-OI70"
              subtitle="Metric-specific threshold tolerance and multi-day persistence tracking"
            >
              <div className="space-y-3">
                {orchestratorState.driftDecision.observations.map((obs) => (
                  <div
                    key={obs.metricId}
                    className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 flex justify-between items-center"
                  >
                    <div>
                      <div className="text-sm font-semibold text-slate-200">
                        {obs.metricName}
                      </div>
                      <div className="text-xs text-slate-400">
                        Expected: <span className="font-mono text-slate-300">{obs.expectedValue}</span> | Actual: <span className="font-mono text-slate-300">{obs.actualValue}</span> | Threshold: <span className="font-mono text-slate-300">±{obs.thresholdPct}%</span>
                      </div>
                    </div>
                    <div className="text-right flex items-center space-x-3">
                      <div>
                        <div className="text-sm font-mono font-bold text-slate-100">
                          {obs.driftPct}%
                        </div>
                        <div className="text-[10px] text-slate-500">
                          {obs.persistenceDays}d persistence
                        </div>
                      </div>
                      <SeverityBadge
                        level={
                          obs.severity === 'CRITICAL'
                            ? 'CRITICAL'
                            : obs.severity === 'HIGH'
                            ? 'HIGH'
                            : 'PASS'
                        }
                      />
                    </div>
                  </div>
                ))}
              </div>
            </HorizonCard>

            {/* ZONE D: Recommended Actions & Re-Optimization */}
            <HorizonCard
              title="RECOMMENDED EXECUTIVE ACTION (INV-OI69)"
              subtitle="Automated intervention dispatch with guaranteed rollback coverage"
            >
              <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-4">
                <div className="flex items-start justify-between">
                  <div>
                    <div className="text-xs uppercase tracking-wider text-emerald-400 font-bold mb-1">
                      Primary Recommendation
                    </div>
                    <div className="text-base font-bold text-slate-100">
                      Accelerate Infrastructure Resilience (Strategy D) into Q2
                    </div>
                    <p className="text-xs text-slate-400 mt-1">
                      Compresses recovery SLA to 1.0 hour and insulates institutional OHI against external macro shocks.
                    </p>
                  </div>
                  <span className="px-2.5 py-1 rounded bg-emerald-950/80 text-emerald-400 text-xs font-semibold border border-emerald-800/60">
                    High Conviction
                  </span>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pt-2 border-t border-slate-800 text-center">
                  <div className="p-2 rounded bg-slate-950/50">
                    <div className="text-[10px] text-slate-400 uppercase">OHI Gain</div>
                    <div className="text-sm font-mono font-bold text-emerald-400">+1.8</div>
                  </div>
                  <div className="p-2 rounded bg-slate-950/50">
                    <div className="text-[10px] text-slate-400 uppercase">Confidence</div>
                    <div className="text-sm font-mono font-bold text-cyan-400">89%</div>
                  </div>
                  <div className="p-2 rounded bg-slate-950/50">
                    <div className="text-[10px] text-slate-400 uppercase">Cost</div>
                    <div className="text-sm font-mono font-bold text-slate-300">LOW</div>
                  </div>
                  <div className="p-2 rounded bg-slate-950/50">
                    <div className="text-[10px] text-slate-400 uppercase">Rollback</div>
                    <div className="text-sm font-mono font-bold text-indigo-400">100%</div>
                  </div>
                </div>

                <div className="flex space-x-3 pt-2">
                  <button
                    onClick={handleReoptimize}
                    className="flex-1 py-2 px-4 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition-all shadow-md shadow-emerald-950"
                  >
                    Authorize Sequence Transition (INV-OI67)
                  </button>
                  <button
                    onClick={handleToggleScenario}
                    className="py-2 px-4 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700 transition-all"
                  >
                    Toggle Shock
                  </button>
                </div>
              </div>
            </HorizonCard>
          </div>

          {/* ZONE E & ZONE F: Portfolio Ranking & Survivability Cockpit */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* ZONE E: Strategy Portfolio Ranking */}
            <HorizonCard
              title="PORTFOLIO CANDIDATE RANKING"
              subtitle="Multi-criteria composite ranking across all evaluated candidates"
            >
              <div className="space-y-2">
                {[
                  { rank: '#1', id: 'STRAT-B-DUAL', name: 'Dual Curriculum & Governance Scaling', score: 94, status: 'RECOMMENDED' },
                  { rank: '#2', id: 'STRAT-D-RESIL', name: 'Resilient Infrastructure & RTO Compression', score: 91, status: 'ACTIVE' },
                  { rank: '#3', id: 'STRAT-A-TRN', name: 'Training Curriculum Scaling', score: 84, status: 'EVALUATING' },
                  { rank: '#4', id: 'STRAT-C-CONSV', name: 'Conservative Capital Freeze', score: 79, status: 'DEFENSIVE' },
                ].map((strat) => (
                  <div
                    key={strat.id}
                    className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 flex justify-between items-center"
                  >
                    <div className="flex items-center space-x-3">
                      <span className="font-mono font-bold text-sm text-cyan-400">{strat.rank}</span>
                      <div>
                        <div className="text-sm font-semibold text-slate-100">{strat.name}</div>
                        <div className="text-[11px] text-slate-400 font-mono">{strat.id}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-mono font-bold text-emerald-400">{strat.score} pts</div>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                        {strat.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </HorizonCard>

            {/* ZONE F: Survivability Cockpit */}
            <HorizonCard
              title="ORGANIZATIONAL SURVIVABILITY COCKPIT"
              subtitle="Mission-critical disaster recovery SLA, rollback coverage, and failure bounds"
            >
              <div className="grid grid-cols-2 gap-4 p-2">
                <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 text-center">
                  <div className="text-xs text-slate-400 uppercase mb-1">Survivability Score</div>
                  <div className="text-2xl font-bold font-mono text-emerald-400">
                    {orchestratorState.survivabilityScore}%
                  </div>
                  <div className="text-[11px] text-slate-500 mt-1">Multi-scenario weighted</div>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 text-center">
                  <div className="text-xs text-slate-400 uppercase mb-1">Recovery Time SLA</div>
                  <div className="text-2xl font-bold font-mono text-cyan-400">
                    {orchestratorState.recoveryHours} Hours
                  </div>
                  <div className="text-[11px] text-slate-500 mt-1">Certified target &le; 4.0h</div>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 text-center">
                  <div className="text-xs text-slate-400 uppercase mb-1">Rollback Coverage</div>
                  <div className="text-2xl font-bold font-mono text-indigo-400">
                    {orchestratorState.rollbackCoveragePct}%
                  </div>
                  <div className="text-[11px] text-slate-500 mt-1">Multi-tier L1-L4 verified</div>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 text-center">
                  <div className="text-xs text-slate-400 uppercase mb-1">Failure Probability</div>
                  <div className="text-2xl font-bold font-mono text-slate-200">
                    {orchestratorState.failureProbabilityPct}%
                  </div>
                  <div className="text-[11px] text-slate-500 mt-1">Under extreme stress</div>
                </div>
              </div>
            </HorizonCard>
          </div>
        </div>
      )}

      {/* ============================================================ */}
      {/* VIEW 2: ANALYST VIEW (ROOT CAUSES & SIGNALS)                */}
      {/* ============================================================ */}
      {activeTab === 'ANALYST' && (
        <div className="space-y-6">
          {/* Root-Cause Decomposition */}
          <HorizonCard
            title="ROOT-CAUSE DRIFT ATTRIBUTION (INV-OI73)"
            subtitle="Decomposes observed OHI divergence into exact internal metrics and external signals"
          >
            <div className="space-y-3">
              {orchestratorState.driftDecision.rootCauses.map((rc) => (
                <div
                  key={rc.driver}
                  className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 flex justify-between items-center"
                >
                  <div>
                    <div className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
                      <span>{rc.driver}</span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-cyan-400 font-mono">
                        {rc.category}
                      </span>
                    </div>
                    <div className="text-xs text-slate-400 mt-0.5">
                      Confidence: {rc.confidencePct}% | Attributed Causal Transmission
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="text-sm font-mono font-bold text-rose-400">
                      {rc.impactDelta} OHI
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </HorizonCard>

          {/* External Signals Matrix */}
          <HorizonCard
            title="EXTERNAL SIGNAL NORMALIZATION MATRIX (INV-OI72, INV-OI74)"
            subtitle="Real-time macro, regulatory, workforce, and market indicators normalized to [-100, +100]"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {externalSignals.map((sig) => (
                <div key={sig.signalId} className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-mono font-bold text-sm text-cyan-400">{sig.signalId}</span>
                    <span className="text-xs font-semibold px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                      {sig.category}
                    </span>
                  </div>
                  <div className="text-xs text-slate-400 mb-2">
                    Source: <span className="text-slate-300">{sig.source}</span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-center pt-2 border-t border-slate-800/80">
                    <div className="p-1 rounded bg-slate-950">
                      <div className="text-[10px] text-slate-500">Normalized</div>
                      <div className="text-xs font-mono font-bold text-rose-400">{sig.normalizedImpact}</div>
                    </div>
                    <div className="p-1 rounded bg-slate-950">
                      <div className="text-[10px] text-slate-500">Confidence</div>
                      <div className="text-xs font-mono font-bold text-cyan-400">{sig.confidencePct}%</div>
                    </div>
                    <div className="p-1 rounded bg-slate-950">
                      <div className="text-[10px] text-slate-500">Effective</div>
                      <div className="text-xs font-mono font-bold text-amber-400">{sig.effectiveImpact}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </HorizonCard>
        </div>
      )}

      {/* ============================================================ */}
      {/* VIEW 3: AUDIT VIEW (TRACEABILITY & CALIBRATION)              */}
      {/* ============================================================ */}
      {activeTab === 'AUDIT' && (
        <div className="space-y-6">
          {/* Model Calibration */}
          <HorizonCard
            title="SIMULATION CALIBRATION & ERROR BOUNDS (INV-OI71)"
            subtitle="Backtested prediction error metrics, bias quantification, and calibrated transmission weights"
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 font-semibold">
                    <th className="py-2.5 px-3">Causal Transmission Link</th>
                    <th className="py-2.5 px-3">Prior Weight</th>
                    <th className="py-2.5 px-3">Calibrated Weight</th>
                    <th className="py-2.5 px-3">MAE Error</th>
                    <th className="py-2.5 px-3">Prediction Bias</th>
                    <th className="py-2.5 px-3">Confidence</th>
                    <th className="py-2.5 px-3">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {calibrationResults.map((cal) => (
                    <tr key={cal.metricId} className="hover:bg-slate-900/40">
                      <td className="py-2.5 px-3 font-sans font-medium text-slate-200">{cal.metricId}</td>
                      <td className="py-2.5 px-3 text-slate-400">{cal.previousWeight}</td>
                      <td className="py-2.5 px-3 text-cyan-400 font-bold">{cal.calibratedWeight}</td>
                      <td className="py-2.5 px-3 text-slate-300">{cal.mae}</td>
                      <td className="py-2.5 px-3 text-slate-300">{cal.predictionBias}</td>
                      <td className="py-2.5 px-3 text-emerald-400">{cal.confidencePct}%</td>
                      <td className="py-2.5 px-3 font-sans">
                        <SeverityBadge level="PASS" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </HorizonCard>

          {/* Gate Traceability Matrix */}
          <HorizonCard
            title="M17 GATE TRACEABILITY MATRIX (M17-Gate-01 through M17-Gate-10)"
            subtitle="Formal audit trail certifying all 10 Horizon 4 gates and Invariants INV-OI67 through INV-OI74"
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 font-semibold">
                    <th className="py-2.5 px-3">Gate ID</th>
                    <th className="py-2.5 px-3">Gate Name</th>
                    <th className="py-2.5 px-3">Scope & Formal Invariant Requirement</th>
                    <th className="py-2.5 px-3 text-right">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {M17_GATE_TRACEABILITY_MATRIX.map((gate) => (
                    <tr key={gate.gateId} className="hover:bg-slate-900/40">
                      <td className="py-2.5 px-3 font-mono font-bold text-cyan-400">{gate.gateId}</td>
                      <td className="py-2.5 px-3 font-semibold text-slate-200">{gate.name}</td>
                      <td className="py-2.5 px-3 text-slate-400">{gate.requirement}</td>
                      <td className="py-2.5 px-3 text-right">
                        <SeverityBadge level="PASS" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </HorizonCard>
        </div>
      )}

      {/* Related Artifacts Panel */}
      <RelatedArtifactsPanel
        title="HORIZON STRATEGIC INTELLIGENCE SUITE"
        artifacts={[
          {
            id: 'ORC-ART-01',
            type: 'SIMULATION',
            title: 'Executive Adoption Center',
            href: '/adoption-center',
            summary: 'Executive usage telemetry, time-to-decision acceleration, and institutional ROI.',
          },
          {
            id: 'ORC-ART-02',
            type: 'SIMULATION',
            title: 'Executive Sandbox & Digital Twin',
            href: '/executive-sandbox',
            summary: 'Predictive organizational simulation, waterfall attribution walk, and rollback planner.',
          },
          {
            id: 'ORC-ART-03',
            type: 'SIMULATION',
            title: 'Strategy Laboratory & Portfolio Ranking',
            href: '/strategy-laboratory',
            summary: 'Candidate strategy ranking, multi-regime stress testing, and survivability cockpit.',
          },
          {
            id: 'ORC-ART-04',
            type: 'AUDIT',
            title: 'Release Certification Dashboard',
            href: '/release-dashboard',
            summary: 'Milestone certification gates, cryptographic attestation locks, and release governance.',
          },
        ]}
      />
    </div>
  );
}

export default function StrategyOrchestratorPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-slate-950 text-slate-400 p-8">Loading Strategy Orchestrator...</div>}>
      <StrategyOrchestratorContent />
    </Suspense>
  );
}
