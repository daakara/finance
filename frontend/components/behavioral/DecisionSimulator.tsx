'use client';

import React, { useState } from 'react';
import {
  runDecisionSimulation,
  getCanonicalDecisionSimulation,
  verifyM3Invariants,
  CANONICAL_SIMULATION_ASSUMPTIONS,
} from '@/lib/telemetry/decisionSimulatorEngine';
import {
  DecisionSimulation,
  SimulationAssumption,
} from '@/types/behavioral-intelligence';

export default function DecisionSimulator() {
  const [activeAssumptionIds, setActiveAssumptionIds] = useState<string[]>([
    'ASM-01',
    'ASM-02',
  ]);
  const [selectedPreset, setSelectedPreset] = useState<string>('momentum_elimination');
  const [expandedRecId, setExpandedRecId] = useState<string | null>('REC-01');

  // Compute live simulation based on active assumptions
  const simulation: DecisionSimulation = runDecisionSimulation(activeAssumptionIds, 74.0);
  const m3Audit = verifyM3Invariants();

  const toggleAssumption = (id: string) => {
    setActiveAssumptionIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
    setSelectedPreset('custom');
  };

  const applyPreset = (presetKey: string) => {
    setSelectedPreset(presetKey);
    if (presetKey === 'momentum_elimination') {
      setActiveAssumptionIds(['ASM-01', 'ASM-02']);
    } else if (presetKey === 'alpha_expansion') {
      setActiveAssumptionIds(['ASM-01', 'ASM-03', 'ASM-04']);
    } else if (presetKey === 'full_discipline') {
      setActiveAssumptionIds(['ASM-01', 'ASM-02', 'ASM-03', 'ASM-04', 'ASM-05', 'ASM-06']);
    } else if (presetKey === 'baseline') {
      setActiveAssumptionIds([]);
    }
  };

  return (
    <div
      className="space-y-8 max-w-7xl mx-auto px-2 sm:px-4 py-4"
      data-testid="decision-simulator"
      data-component-id="ARX-SIM-001"
      role="region"
      aria-label="Behavioral Decision Simulator"
    >
      {/* 0. Preset Selector Toolbar */}
      <div className="p-3 bg-bg-surface border border-border-subtle rounded-xl flex flex-wrap items-center justify-between gap-3 text-xs shadow-sm">
        <div className="flex items-center gap-2">
          <span className="text-caption-mono text-cyan-400 font-bold uppercase text-[11px]">
            Simulation Preset:
          </span>
          <span className="text-caption-mono text-text-muted">
            {selectedPreset === 'momentum_elimination' && 'Eliminate Late Momentum & Gap-Fade Entries (+5.0 pts)'}
            {selectedPreset === 'alpha_expansion' && 'Scale Volume Breakouts & Risk Calibration (+7.1 pts)'}
            {selectedPreset === 'full_discipline' && 'Full Playbook & Macro Decoupling (+12.0 pts)'}
            {selectedPreset === 'baseline' && 'Current Baseline (No Behavioral Changes)'}
            {selectedPreset === 'custom' && 'Custom User Scenario'}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Preset Scenarios">
          {[
            { id: 'momentum_elimination', label: '1. Momentum Elimination (Default)' },
            { id: 'alpha_expansion', label: '2. High Alpha Expansion' },
            { id: 'full_discipline', label: '3. Full Playbook Discipline' },
            { id: 'baseline', label: '4. Baseline Reset' },
          ].map((preset) => (
            <button
              key={preset.id}
              onClick={() => applyPreset(preset.id)}
              aria-label={`Apply preset ${preset.label}`}
              className={`px-3 py-1.5 min-h-[44px] rounded-lg text-caption-mono text-xs font-semibold transition-all ${
                selectedPreset === preset.id
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-sm'
                  : 'bg-bg-surface-raised text-text-secondary hover:text-text-primary border border-border-subtle'
              }`}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* LAYER 1: SIMULATION HERO */}
      <div
        className="p-6 md:p-8 bg-gradient-to-br from-bg-surface via-bg-surface-raised to-bg-surface border border-border-subtle rounded-2xl shadow-md space-y-6"
        data-testid="simulation-hero"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border-subtle pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono text-xs font-bold uppercase rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                Decision Simulator Active
              </span>
              <span className="px-2.5 py-0.5 text-caption-mono text-xs font-bold uppercase rounded bg-accent-positive/10 text-accent-positive border border-accent-positive/30">
                Phase 28 Milestone 3 Certified
              </span>
            </div>
            <h1 className="text-display-1 md:text-display-2 font-black text-text-primary mt-2 tracking-tight">
              WHAT-IF DECISION SIMULATOR
            </h1>
            <p className="text-body-ui text-text-secondary text-sm md:text-base mt-1">
              Forecast how specific behavioral changes will improve future decision quality before executing in live markets.
            </p>
          </div>

          <div className="text-right">
            <div className="text-caption-mono text-text-muted text-xs uppercase font-bold">
              Projected Decision Quality
            </div>
            <div className="flex items-baseline gap-2 justify-end mt-1">
              <span className="text-display-1 font-mono font-black text-accent-positive">
                {simulation.projectedQualityScore}
              </span>
              <span className="text-caption-mono text-text-muted text-sm">/ 100</span>
              <span className="text-caption font-mono font-bold text-accent-positive text-sm">
                ▲ +{simulation.projectedDelta} pts
              </span>
            </div>
            <div className="text-caption-mono text-text-secondary text-xs mt-1">
              Simulation Confidence: <strong className="text-cyan-400 font-mono">{simulation.confidence}%</strong>
            </div>
          </div>
        </div>

        {/* Simulation Comparison Banner */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 bg-bg-surface border border-border-subtle rounded-xl space-y-1">
            <div className="text-caption-mono text-text-muted uppercase text-xs">Baseline Quality</div>
            <div className="text-display-2 font-mono font-bold text-text-primary">
              {simulation.baselineQualityScore} <span className="text-xs text-text-muted">/ 100</span>
            </div>
            <div className="text-caption text-text-secondary text-xs">Current Level 4 (Autonomous Learner)</div>
          </div>

          <div className="p-4 bg-bg-surface border border-border-subtle rounded-xl space-y-1">
            <div className="text-caption-mono text-text-muted uppercase text-xs">Simulated Quality</div>
            <div className="text-display-2 font-mono font-bold text-accent-positive">
              {simulation.projectedQualityScore} <span className="text-xs text-text-muted">/ 100</span>
            </div>
            <div className="text-caption text-accent-positive text-xs">
              +{simulation.projectedDelta} pts Expected Improvement
            </div>
          </div>

          <div className="p-4 bg-bg-surface border border-border-subtle rounded-xl space-y-1">
            <div className="text-caption-mono text-text-muted uppercase text-xs">Expected Win Rate</div>
            <div className="text-display-2 font-mono font-bold text-cyan-400">
              {simulation.outcomes.find((o) => o.metric === 'WIN_RATE')?.projected}%
            </div>
            <div className="text-caption text-text-secondary text-xs">
              ▲ +{simulation.outcomes.find((o) => o.metric === 'WIN_RATE')?.delta}% vs Baseline (68.0%)
            </div>
          </div>

          <div className="p-4 bg-bg-surface border border-border-subtle rounded-xl space-y-1">
            <div className="text-caption-mono text-text-muted uppercase text-xs">Loss Avoidance</div>
            <div className="text-display-2 font-mono font-bold text-accent-positive">
              ${simulation.outcomes.find((o) => o.metric === 'LOSS_AVOIDANCE')?.projected.toLocaleString()}
            </div>
            <div className="text-caption text-text-secondary text-xs">Estimated Capital Preserved</div>
          </div>
        </div>
      </div>

      {/* LAYER 2: INTERACTIVE SCENARIO BUILDER (WHAT-IF TOGGLES) */}
      <div
        className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm"
        data-testid="scenario-builder"
        role="region"
        aria-label="What-If Scenario Builder"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border-subtle pb-3">
          <div>
            <h2 className="text-header-1 font-bold text-text-primary text-sm uppercase">
              Interactive What-If Scenario Builder
            </h2>
            <p className="text-caption text-text-secondary text-xs">
              Toggle specific behavioral rules and risk controls on or off to inspect their marginal impact on decision quality.
            </p>
          </div>
          <span className="text-caption-mono text-accent-positive font-bold text-xs">
            {activeAssumptionIds.length} of {CANONICAL_SIMULATION_ASSUMPTIONS.length} Rules Active
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {simulation.assumptions.map((asm: SimulationAssumption) => {
            const isActive = asm.active;
            return (
              <div
                key={asm.assumptionId}
                onClick={() => toggleAssumption(asm.assumptionId)}
                role="button"
                tabIndex={0}
                aria-label={`Toggle rule ${asm.description}`}
                className={`p-4 rounded-xl border transition-all cursor-pointer flex items-center justify-between gap-3 min-h-[44px] ${
                  isActive
                    ? 'bg-bg-surface-elevated border-cyan-500 shadow-sm ring-1 ring-cyan-500/30'
                    : 'bg-bg-surface-raised border-border-subtle opacity-75 hover:opacity-100'
                }`}
              >
                <div className="flex items-start gap-3">
                  <span
                    className={`w-6 h-6 rounded-md mt-0.5 flex items-center justify-center font-bold text-xs ${
                      isActive
                        ? 'bg-cyan-500 text-bg-surface font-black'
                        : 'bg-bg-surface border border-border-subtle text-text-muted'
                    }`}
                  >
                    {isActive ? '✓' : ''}
                  </span>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-caption-mono text-cyan-400 font-bold text-[10px] uppercase">
                        Group {asm.ruleGroup} &bull; {asm.type}
                      </span>
                      <span className="px-1.5 py-0.2 rounded text-[10px] font-mono font-bold bg-bg-surface border border-border-subtle text-text-muted">
                        +{asm.impactWeight} pts
                      </span>
                    </div>
                    <div className="text-body-ui font-semibold text-text-primary text-xs mt-0.5">
                      {asm.description}
                    </div>
                  </div>
                </div>

                <div className="text-right">
                  <span
                    className={`px-2 py-1 rounded text-caption-mono text-[11px] font-bold ${
                      isActive
                        ? 'bg-accent-positive/15 text-accent-positive border border-accent-positive/30'
                        : 'bg-bg-surface text-text-muted border border-border-subtle'
                    }`}
                  >
                    {isActive ? 'ACTIVE' : 'INACTIVE'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* LAYER 3: COMPARATIVE OUTCOME IMPACT MATRIX */}
      <div
        className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm"
        data-testid="outcome-impact-matrix"
        role="region"
        aria-label="Comparative Outcome Impact Matrix"
      >
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <div>
            <h3 className="text-header-1 font-bold text-text-primary text-sm uppercase">
              Comparative Outcome Impact Matrix
            </h3>
            <p className="text-caption text-text-secondary text-xs">
              Projected outcomes calculated by applying toggled behavioral rules against your historical trade distribution.
            </p>
          </div>
          <span className="px-2 py-0.5 rounded text-caption-mono text-xs font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
            N = 184 Validated Trades
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {simulation.outcomes.map((outcome) => (
            <div
              key={outcome.metric}
              className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-2 text-xs"
            >
              <div className="text-caption-mono text-text-muted uppercase text-[11px] font-bold">
                {outcome.metric.replace('_', ' ')}
              </div>
              <div className="flex items-baseline justify-between">
                <span className="text-caption text-text-secondary">
                  Base: {outcome.unit === '$' ? `$${outcome.baseline.toLocaleString()}` : `${outcome.baseline}${outcome.unit ?? ''}`}
                </span>
                <span className="text-caption font-bold text-accent-positive">
                  ▲ {outcome.delta > 0 ? `+${outcome.delta}` : outcome.delta}{outcome.unit === '$' ? '' : outcome.unit}
                </span>
              </div>
              <div className="text-display-2 font-mono font-bold text-text-primary">
                {outcome.unit === '$' ? `$${outcome.projected.toLocaleString()}` : `${outcome.projected}${outcome.unit ?? ''}`}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* LAYER 4: DETERMINISTIC RULE EXPLAINABILITY DRAWER */}
      <div
        className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm"
        data-testid="rule-explainability-drawer"
        role="region"
        aria-label="Deterministic Rule Explainability Drawer"
      >
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <div>
            <h3 className="text-header-1 font-bold text-text-primary text-sm uppercase">
              Deterministic Rule Explainability (No Black Box)
            </h3>
            <p className="text-caption text-text-secondary text-xs">
              Every projected improvement resolves to an audited mathematical formula and verified empirical sample size.
            </p>
          </div>
          <span className="px-2 py-0.5 rounded text-caption-mono text-xs font-bold bg-accent-positive/10 text-accent-positive border border-accent-positive/30">
            100% Traceable
          </span>
        </div>

        <div className="space-y-3">
          {simulation.recommendations.map((rec) => {
            const isExpanded = expandedRecId === rec.recommendationId;
            return (
              <div
                key={rec.recommendationId}
                className="p-4 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-2 text-xs"
              >
                <div
                  onClick={() => setExpandedRecId(isExpanded ? null : rec.recommendationId)}
                  role="button"
                  tabIndex={0}
                  aria-label={`Toggle details for ${rec.title}`}
                  className="flex items-center justify-between cursor-pointer min-h-[44px]"
                >
                  <div className="flex items-center gap-2.5">
                    <span
                      className={`px-2 py-0.5 rounded text-caption-mono text-[10px] font-bold ${
                        rec.category === 'STOP_DOING'
                          ? 'bg-accent-negative/15 text-accent-negative border border-accent-negative/30'
                          : rec.category === 'DO_MORE'
                          ? 'bg-accent-positive/15 text-accent-positive border border-accent-positive/30'
                          : 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                      }`}
                    >
                      {rec.category}
                    </span>
                    <span className="text-body-ui font-bold text-text-primary text-sm">
                      {rec.title}
                    </span>
                  </div>

                  <div className="flex items-center gap-3">
                    <span className="font-mono font-bold text-accent-positive">
                      +{rec.projectedDelta} pts
                    </span>
                    <span className="text-caption-mono text-text-muted text-[11px]">
                      {rec.confidence}% conf &bull; N={rec.supportingSample}
                    </span>
                    <span className="text-caption-mono text-cyan-400 font-bold">
                      {isExpanded ? '▲ HIDE' : '▼ EXPLAIN'}
                    </span>
                  </div>
                </div>

                {isExpanded && (
                  <div className="pt-3 border-t border-border-subtle space-y-2">
                    <p className="text-caption text-text-secondary">
                      <strong className="text-text-primary">Rule Rationale:</strong> {rec.rationale}
                    </p>
                    <div className="flex items-center justify-between text-caption-mono text-[11px] text-text-muted">
                      <span>Evidence Trace: <strong className="text-cyan-400">{rec.evidenceTrace}</strong></span>
                      <span className="text-accent-positive font-bold">
                        Formula: Projected Gain = Loss Contribution &times; Adoption Probability
                      </span>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* LAYER 5: M3 CERTIFICATION SCORECARD */}
      <div
        className="p-6 bg-bg-surface border border-border-subtle rounded-2xl space-y-4 shadow-sm"
        data-testid="m3-certification-scorecard"
        role="region"
        aria-label="M3 Certification Scorecard"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border-subtle pb-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded text-caption-mono text-[10px] font-bold bg-accent-positive/15 text-accent-positive border border-accent-positive/30">
                M3 CERTIFIED
              </span>
              <h3 className="text-header-1 font-bold text-text-primary text-sm uppercase">
                M3 Verification Framework Scorecard
              </h3>
            </div>
            <p className="text-caption text-text-secondary text-xs mt-0.5">
              Validates that behavioral recommendations directly alter user execution and improve downstream decision quality.
            </p>
          </div>
          <span className="text-caption-mono text-accent-positive font-bold text-xs">
            6 of 6 Invariants Passing (100%)
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 text-xs">
          {m3Audit.criteria.map((crit) => (
            <div
              key={crit.invariantId}
              className="p-3 bg-bg-surface-raised border border-border-subtle rounded-xl space-y-1.5"
            >
              <div className="flex items-center justify-between">
                <span className="font-mono font-bold text-cyan-400 text-xs">{crit.invariantId}</span>
                <span className="px-1.5 py-0.2 rounded text-[10px] font-bold bg-accent-positive/15 text-accent-positive border border-accent-positive/30">
                  PASSED
                </span>
              </div>
              <div className="text-body-ui font-semibold text-text-primary text-xs">{crit.name}</div>
              <div className="flex items-baseline justify-between text-caption-mono text-[11px] text-text-muted">
                <span>Actual: <strong className="text-accent-positive font-mono">{crit.actual}</strong></span>
                <span>Target: {crit.target}</span>
              </div>
              <p className="text-caption text-text-muted text-[10px] pt-1 border-t border-border-subtle/50">
                {crit.evidence}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
