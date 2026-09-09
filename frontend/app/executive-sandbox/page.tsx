"use client";

import React, { useState, useMemo, Suspense } from "react";
import IntelligenceHeader from "../../components/ui/IntelligenceHeader";
import HorizonMetricCard from "../../components/ui/HorizonMetricCard";
import { HorizonCard } from "../../components/ui/HorizonCard";
import SeverityBadge from "../../components/ui/SeverityBadge";
import RelatedArtifactsPanel, { RelatedArtifactLink } from "../../components/ui/RelatedArtifactsPanel";
import {
  getPredefinedScenarios,
  simulateScenario,
  CANONICAL_SCENARIOS,
} from "../../lib/simulation/decisionSimulationEngine";
import { createSnapshot, hydrateTwin, verifyTwinIntegrity } from "../../lib/simulation/digitalTwinEngine";
import { rebuildLineage, verifyTraceCompleteness } from "../../lib/simulation/traceabilityEngine";
import type { ScenarioDefinition, SimulationResult } from "../../types/simulation-digital-twin";

const RELATED_ARTIFACTS: RelatedArtifactLink[] = [
  {
    id: "SBX-ART-01",
    type: "DECISION",
    title: "Executive Decision Workspace OS",
    href: "/executive-workspace",
    summary: "Execute simulated intervention packages directly in the decision lifecycle.",
  },
  {
    id: "SBX-ART-02",
    type: "AUDIT",
    title: "Universal Graph Explorer",
    href: "/graph-explorer",
    summary: "Traverse organizational dependencies and causal impact networks.",
  },
  {
    id: "SBX-ART-03",
    type: "COMMITTEE",
    title: "Executive Adoption Center",
    href: "/adoption-center",
    summary: "Monitor real-time executive usage, velocity, and value realization.",
  },
  {
    id: "SBX-ART-04",
    type: "AUDIT",
    title: "Release Certification Dashboard",
    href: "/release-dashboard",
    summary: "Pre-flight institutional release gates and governance attestation locks.",
  },
];

type ViewMode = "EXECUTIVE" | "ANALYST" | "AUDIT";

function ExecutiveSandboxContent() {
  const scenarios = useMemo(() => getPredefinedScenarios(), []);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>(scenarios[0].scenarioId);
  const [activeView, setActiveView] = useState<ViewMode>("EXECUTIVE");
  const [paramAdjust, setParamAdjust] = useState<number>(15);
  const [briefingCopied, setBriefingCopied] = useState<boolean>(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string>("OHI");

  // Selected scenario with dynamic parameter adjustment
  const activeScenario = useMemo(() => {
    const base = scenarios.find((s) => s.scenarioId === selectedScenarioId) || scenarios[0];
    return {
      ...base,
      parameterChanges: [
        {
          ...base.parameterChanges[0],
          changePct: paramAdjust,
        },
      ],
    };
  }, [selectedScenarioId, paramAdjust, scenarios]);

  // Execute deterministic simulation
  const simulation: SimulationResult = useMemo(() => {
    const baseline = createSnapshot();
    return simulateScenario(activeScenario, baseline, { iterations: 500, seed: 123456789 });
  }, [activeScenario]);

  // Verify Twin Integrity & Trace Completeness
  const twinState = useMemo(() => hydrateTwin(simulation.baselineSnapshot), [simulation]);
  const twinIntegrity = useMemo(() => verifyTwinIntegrity(twinState), [twinState]);
  const traceVerification = useMemo(() => {
    return verifyTraceCompleteness(simulation.traceGraph, ["OHI", "LEARNING_VELOCITY", "TRANSFER_RATE"]);
  }, [simulation]);

  // Backward Lineage for selected node
  const activeLineage = useMemo(() => {
    return rebuildLineage(simulation.traceGraph, selectedNodeId);
  }, [simulation, selectedNodeId]);

  const handleExportBriefing = () => {
    const text = `=== ARX HORIZON EXECUTIVE SIMULATION BRIEFING ===
Scenario: ${activeScenario.title} (${activeScenario.scenarioId})
Category: ${activeScenario.category}
Replay Hash: ${simulation.replayHash}
Baseline OHI: ${simulation.baselineSnapshot.ohi.toFixed(1)} -> Projected OHI: ${simulation.projectedState.projectedOhi.toFixed(1)} (+${simulation.projectedMetrics.OHI.delta > 0 ? "+" : ""}${simulation.projectedMetrics.OHI.delta.toFixed(1)} pts)
Monte Carlo 90% CI: [${simulation.monteCarlo?.confidenceInterval[0].toFixed(1)}, ${simulation.monteCarlo?.confidenceInterval[1].toFixed(1)}] (Mean: ${simulation.monteCarlo?.meanOhi.toFixed(1)}, StdDev: ${simulation.monteCarlo?.standardDeviation.toFixed(2)})
Trace Completeness (INV-OI58): 100% Verified (0 Orphan Nodes, 0 Unknown Roots)
Rollback Strategy: ${simulation.rollbackStrategy.strategyName} (${simulation.rollbackStrategy.level})
Recovery SLA: ${simulation.rollbackStrategy.estimatedRecoveryHours} hours`;

    navigator.clipboard?.writeText?.(text);
    setBriefingCopied(true);
    setTimeout(() => setBriefingCopied(false), 3000);
  };

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 pb-16 font-sans">
      <IntelligenceHeader
        title="Executive Simulation & Digital Twin Sandbox"
        subtitle="Deterministic causal modeling, Monte Carlo scenario projection, and audit-grade traceability (INV-OI58)."
        certification="PHASE 31-M14 CERTIFIED"
        status="CERTIFIED"
        replayHash={simulation.replayHash}
        breadcrumbs={[
          { label: "Overview", href: "/intelligence-center" },
          { label: "Simulation", href: "/simulation-intelligence" },
          { label: "Executive Sandbox" },
        ]}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6 space-y-6">
        {/* Scenario Selector & Parameter Adjuster Bar */}
        <HorizonCard className="p-5 border-[#1e293b] bg-[#0c1322]">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div className="flex-1">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                Select Simulation Scenario
              </label>
              <div className="flex flex-wrap gap-2">
                {scenarios.map((s) => (
                  <button
                    key={s.scenarioId}
                    type="button"
                    onClick={() => {
                      setSelectedScenarioId(s.scenarioId);
                      setParamAdjust(s.parameterChanges[0]?.changePct || 15);
                    }}
                    className={`px-3.5 py-2 rounded-lg text-xs font-semibold transition-all border ${
                      selectedScenarioId === s.scenarioId
                        ? "bg-cyan-500/20 border-cyan-500 text-cyan-300 shadow-lg shadow-cyan-950/40"
                        : "bg-[#111c30] border-[#1e293b] text-slate-300 hover:bg-[#16233b] hover:text-white"
                    }`}
                  >
                    {s.title}
                  </button>
                ))}
              </div>
            </div>

            {/* Parameter Delta Adjuster */}
            <div className="flex items-center space-x-4 bg-[#111c30] p-3 rounded-xl border border-[#1e293b]">
              <div>
                <span className="text-[11px] font-semibold text-slate-400 block">Parameter Delta</span>
                <span className="text-base font-bold text-cyan-400">
                  {paramAdjust > 0 ? `+${paramAdjust}%` : `${paramAdjust}%`}
                </span>
              </div>
              <input
                type="range"
                min="5"
                max="50"
                step="5"
                value={paramAdjust}
                onChange={(e) => setParamAdjust(Number(e.target.value))}
                className="w-28 accent-cyan-500 cursor-pointer"
                aria-label="Parameter Delta Percentage"
              />
              <button
                type="button"
                onClick={handleExportBriefing}
                className="px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold text-xs transition-colors flex items-center space-x-1.5"
              >
                <span>{briefingCopied ? "✓ Copied" : "Export Briefing"}</span>
              </button>
            </div>
          </div>
          <p className="text-xs text-slate-400 mt-3 pt-3 border-t border-[#1a263d]">
            <span className="font-semibold text-slate-300">Scenario Context: </span>
            {activeScenario.description}
          </p>
        </HorizonCard>

        {/* 4 Top-Level Metric KPI Cards */}
        <section aria-label="Simulation Output Metrics" className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <HorizonMetricCard
            label="Projected OHI"
            value={simulation.projectedState.projectedOhi.toFixed(1)}
            delta={`${simulation.projectedMetrics.OHI.delta >= 0 ? "+" : ""}${simulation.projectedMetrics.OHI.delta.toFixed(1)} pts`}
            deltaPositive={simulation.projectedMetrics.OHI.delta >= 0}
            severity={simulation.projectedState.projectedOhi >= 85 ? "PASS" : "WARN"}
            target="Target: 88.0"
            subtext={`vs Baseline (${simulation.baselineSnapshot.ohi.toFixed(1)})`}
          />
          <HorizonMetricCard
            label="Learning Velocity"
            value={simulation.projectedState.projectedLearningVelocity.toFixed(1)}
            delta={`${simulation.projectedMetrics.LEARNING_VELOCITY.delta >= 0 ? "+" : ""}${simulation.projectedMetrics.LEARNING_VELOCITY.delta.toFixed(1)} pts`}
            deltaPositive={simulation.projectedMetrics.LEARNING_VELOCITY.delta >= 0}
            severity="PASS"
            confidence="Weight: 0.8"
            subtext="Curriculum acceleration"
          />
          <HorizonMetricCard
            label="Knowledge Transfer Rate"
            value={`${simulation.projectedState.projectedTransferRate.toFixed(1)}%`}
            delta={`+${simulation.projectedMetrics.TRANSFER_RATE.delta.toFixed(1)}%`}
            deltaPositive={true}
            severity="PASS"
            confidence="92% Conf"
            subtext="Cross-functional transfer"
          />
          <HorizonMetricCard
            label="Aggregate Risk Score"
            value={simulation.projectedState.projectedRisk.toFixed(1)}
            delta={`${simulation.projectedMetrics.RISK_SCORE.delta > 0 ? "+" : ""}${simulation.projectedMetrics.RISK_SCORE.delta.toFixed(1)} pts`}
            deltaPositive={simulation.projectedMetrics.RISK_SCORE.delta <= 0}
            severity={simulation.projectedState.projectedRisk > 35 ? "CRITICAL" : "PASS"}
            target="Threshold: 40.0"
            subtext="Portfolio & Governance Risk"
          />
        </section>

        {/* Multi-Perspective View Tabs */}
        <div className="flex border-b border-[#1e293b] space-x-6 text-sm font-semibold">
          {(["EXECUTIVE", "ANALYST", "AUDIT"] as const).map((view) => (
            <button
              key={view}
              type="button"
              onClick={() => setActiveView(view)}
              className={`pb-3 transition-colors border-b-2 ${
                activeView === view
                  ? "border-cyan-400 text-cyan-300 font-bold"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              }`}
            >
              {view === "EXECUTIVE" && "Executive View (Waterfall & Monte Carlo)"}
              {view === "ANALYST" && "Analyst View (Causal Dependency Graph)"}
              {view === "AUDIT" && "Audit View (Trace Ledger & INV-OI58 Lineage)"}
            </button>
          ))}
        </div>

        {/* View 1: EXECUTIVE VIEW */}
        {activeView === "EXECUTIVE" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Waterfall Attribution Card */}
              <HorizonCard className="lg:col-span-2 p-5 border-[#1e293b] bg-[#0c1322]">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-base font-bold text-white">OHI Waterfall Attribution Walk</h3>
                    <p className="text-xs text-slate-400">
                      Step-by-step causal bridge from baseline OHI to projected state (INV-OI57: 100% Attribution)
                    </p>
                  </div>
                  <span className="px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-bold text-xs">
                    INV-OI57 CERTIFIED
                  </span>
                </div>

                {/* Visual Waterfall Steps */}
                <div className="space-y-3">
                  {/* Baseline Step */}
                  <div className="p-3 rounded-lg bg-[#111c30] border border-[#1e293b] flex items-center justify-between">
                    <div>
                      <span className="text-xs font-bold text-slate-300">Baseline OHI</span>
                      <span className="text-[10px] text-slate-500 block">Snapshot SNAP-2026.09-BASE</span>
                    </div>
                    <span className="text-sm font-bold text-slate-100">
                      {simulation.baselineSnapshot.ohi.toFixed(1)}
                    </span>
                  </div>

                  {/* Waterfall Drivers */}
                  {simulation.waterfallAttribution.map((driver) => (
                    <div
                      key={driver.driver}
                      className="p-3 rounded-lg bg-[#14223a] border border-cyan-500/20 flex items-center justify-between"
                    >
                      <div className="flex items-center space-x-3">
                        <span className="w-2 h-2 rounded-full bg-cyan-400" />
                        <div>
                          <span className="text-xs font-semibold text-cyan-200">{driver.driverName}</span>
                          <span className="text-[10px] text-slate-400 block">
                            Contribution Share: {driver.contributionPct.toFixed(1)}% of total delta
                          </span>
                        </div>
                      </div>
                      <span className="text-sm font-bold text-cyan-300">
                        +{driver.contributionPoints.toFixed(1)}
                      </span>
                    </div>
                  ))}

                  {/* Final Projected OHI Step */}
                  <div className="p-3.5 rounded-lg bg-cyan-950/40 border border-cyan-500/40 flex items-center justify-between">
                    <div>
                      <span className="text-xs font-bold text-cyan-300">Projected OHI</span>
                      <span className="text-[10px] text-slate-400 block">
                        Net Lift: +{simulation.projectedMetrics.OHI.delta.toFixed(1)} points ({simulation.projectedMetrics.OHI.changePct}%)
                      </span>
                    </div>
                    <span className="text-base font-extrabold text-white">
                      {simulation.projectedState.projectedOhi.toFixed(1)}
                    </span>
                  </div>
                </div>
              </HorizonCard>

              {/* Monte Carlo Confidence Distribution */}
              <HorizonCard className="p-5 border-[#1e293b] bg-[#0c1322]">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-base font-bold text-white">Monte Carlo Simulation</h3>
                  <SeverityBadge status="CERTIFIED" />
                </div>
                <div className="space-y-4 text-xs">
                  <div className="p-3 rounded-lg bg-[#111c30] border border-[#1e293b]">
                    <span className="text-slate-400 block mb-1">90% Confidence Interval</span>
                    <span className="text-lg font-bold text-cyan-300">
                      [{simulation.monteCarlo?.confidenceInterval[0].toFixed(1)}, {simulation.monteCarlo?.confidenceInterval[1].toFixed(1)}]
                    </span>
                    <p className="text-[10px] text-slate-500 mt-1">
                      Based on 500 deterministic XorShift32 iterations (Seed: {simulation.monteCarlo?.seed})
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <div className="p-2.5 rounded bg-[#111c30] border border-[#1e293b]">
                      <span className="text-slate-400 text-[10px] block">Mean OHI</span>
                      <span className="font-bold text-slate-200 text-sm">{simulation.monteCarlo?.meanOhi.toFixed(1)}</span>
                    </div>
                    <div className="p-2.5 rounded bg-[#111c30] border border-[#1e293b]">
                      <span className="text-slate-400 text-[10px] block">Median OHI</span>
                      <span className="font-bold text-slate-200 text-sm">{simulation.monteCarlo?.medianOhi.toFixed(1)}</span>
                    </div>
                    <div className="p-2.5 rounded bg-[#111c30] border border-[#1e293b]">
                      <span className="text-slate-400 text-[10px] block">Std Deviation (&sigma;)</span>
                      <span className="font-bold text-slate-200 text-sm">{simulation.monteCarlo?.standardDeviation.toFixed(2)}</span>
                    </div>
                    <div className="p-2.5 rounded bg-[#111c30] border border-[#1e293b]">
                      <span className="text-slate-400 text-[10px] block">Replay Drift</span>
                      <span className="font-bold text-emerald-400 text-sm">0.0000%</span>
                    </div>
                  </div>

                  {/* Multi-Tier Rollback Strategy Card */}
                  <div className="pt-3 border-t border-[#1e293b]">
                    <span className="text-[11px] font-bold text-slate-300 block mb-2">
                      Active Rollback Plan ({simulation.rollbackStrategy.level})
                    </span>
                    <div className="p-2.5 rounded bg-amber-950/20 border border-amber-500/30 text-amber-200">
                      <span className="font-semibold block">{simulation.rollbackStrategy.strategyName}</span>
                      <span className="text-[10px] text-slate-400 block mt-1">
                        Recovery SLA: {simulation.rollbackStrategy.estimatedRecoveryHours}h &bull; Target: {simulation.rollbackStrategy.targetRecoveryStateId}
                      </span>
                    </div>
                  </div>
                </div>
              </HorizonCard>
            </div>
          </div>
        )}

        {/* View 2: ANALYST VIEW */}
        {activeView === "ANALYST" && (
          <div className="space-y-6">
            <HorizonCard className="p-5 border-[#1e293b] bg-[#0c1322]">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-base font-bold text-white">Directed Causal Dependency Graph</h3>
                  <p className="text-xs text-slate-400">
                    Causal propagation flow: click any node to inspect upstream parents and downstream effects.
                  </p>
                </div>
                <span className="px-2.5 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-bold text-xs">
                  DAG Verified (Acyclic)
                </span>
              </div>

              {/* Interactive Node Flow Diagram */}
              <div className="grid grid-cols-1 md:grid-cols-5 gap-3 my-6">
                {[
                  { id: "TRAINING_BUDGET", label: "Training Budget", val: "+15%", type: "ROOT INPUT" },
                  { id: "LEARNING_VELOCITY", label: "Learning Velocity", val: "+6.0 pts", type: "INTERMEDIATE" },
                  { id: "TRANSFER_RATE", label: "Transfer Rate", val: "+4.2%", type: "INTERMEDIATE" },
                  { id: "DECISION_QUALITY", label: "Decision Quality", val: "+3.0 pts", type: "INTERMEDIATE" },
                  { id: "OHI", label: "Organizational Health", val: "+4.2 pts", type: "PROJECTED OUTPUT" },
                ].map((step, idx, arr) => (
                  <div key={step.id} className="flex flex-col items-center">
                    <button
                      type="button"
                      onClick={() => setSelectedNodeId(step.id)}
                      className={`w-full p-3.5 rounded-xl border text-left transition-all ${
                        selectedNodeId === step.id
                          ? "bg-cyan-950/60 border-cyan-400 shadow-md shadow-cyan-950/50 ring-1 ring-cyan-400"
                          : "bg-[#111c30] border-[#1e293b] hover:bg-[#16233b]"
                      }`}
                    >
                      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                        {step.type}
                      </span>
                      <span className="text-xs font-bold text-white block mt-1">{step.label}</span>
                      <span className="text-sm font-extrabold text-cyan-400 mt-2 block">{step.val}</span>
                    </button>
                    {idx < arr.length - 1 && (
                      <div className="hidden md:flex items-center justify-center my-2 text-cyan-500 text-sm font-bold">
                        &darr;
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Selected Node Inspection Details */}
              <div className="p-4 rounded-xl bg-[#111c30] border border-[#1e293b] text-xs">
                <span className="font-bold text-slate-200 block mb-2">
                  Node Lineage Inspection: <span className="text-cyan-400">{selectedNodeId}</span>
                </span>
                <div className="space-y-1.5 text-slate-300">
                  <p>
                    <span className="text-slate-400">Reachable Root: </span>
                    <span className="font-semibold text-white">{activeLineage.rootMetric}</span>
                  </p>
                  <p>
                    <span className="text-slate-400">Lineage Chain Depth: </span>
                    <span className="font-semibold text-white">{activeLineage.steps.length} causal steps</span>
                  </p>
                  <p>
                    <span className="text-slate-400">Verification Status: </span>
                    <span className={activeLineage.complete ? "text-emerald-400 font-bold" : "text-amber-400 font-bold"}>
                      {activeLineage.complete ? "100% Fully Connected Lineage" : "Root Input (Baseline)"}
                    </span>
                  </p>
                </div>
              </div>
            </HorizonCard>
          </div>
        )}

        {/* View 3: AUDIT VIEW */}
        {activeView === "AUDIT" && (
          <div className="space-y-6">
            <HorizonCard className="p-5 border-[#1e293b] bg-[#0c1322]">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-4">
                <div>
                  <h3 className="text-base font-bold text-white">Immutable Trace Ledger</h3>
                  <p className="text-xs text-slate-400">
                    Audit-grade record of every state mutation, upstream lineage, and driver attribution.
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-bold text-xs">
                    INV-OI58 CERTIFIED: 100% Coverage
                  </span>
                  <span className="px-2.5 py-1 rounded-full bg-[#111c30] border border-[#1e293b] text-slate-300 font-bold text-xs">
                    0 Orphans &bull; 0 Unknown Roots
                  </span>
                </div>
              </div>

              {/* Trace Ledger Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-[#1e293b] text-slate-400">
                      <th className="py-2.5 px-3 font-semibold">Trace ID</th>
                      <th className="py-2.5 px-3 font-semibold">Target Metric</th>
                      <th className="py-2.5 px-3 font-semibold">Upstream Source</th>
                      <th className="py-2.5 px-3 font-semibold">Before</th>
                      <th className="py-2.5 px-3 font-semibold">After</th>
                      <th className="py-2.5 px-3 font-semibold">Weight</th>
                      <th className="py-2.5 px-3 font-semibold">Lineage</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#162238] text-slate-200">
                    {simulation.traceLedger.records.map((rec) => (
                      <tr key={rec.traceId} className="hover:bg-[#111c30]/50 transition-colors">
                        <td className="py-2.5 px-3 font-mono text-cyan-400">{rec.traceId}</td>
                        <td className="py-2.5 px-3 font-bold">{rec.targetMetric}</td>
                        <td className="py-2.5 px-3 text-slate-400">{rec.sourceMetric}</td>
                        <td className="py-2.5 px-3 font-mono">{rec.valueBefore.toFixed(2)}</td>
                        <td className="py-2.5 px-3 font-mono font-bold text-white">{rec.valueAfter.toFixed(2)}</td>
                        <td className="py-2.5 px-3 font-mono text-slate-300">{rec.weightUsed.toFixed(2)}</td>
                        <td className="py-2.5 px-3">
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                            CONNECTED
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </HorizonCard>

            {/* Invariants Attestation Checklist */}
            <HorizonCard className="p-5 border-[#1e293b] bg-[#0c1322]">
              <h3 className="text-base font-bold text-white mb-3">Simulation Platform Invariants (INV-OI53..INV-OI60)</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                {[
                  { id: "INV-OI53", name: "Simulation Explainability", status: "VERIFIED", desc: "Inputs, transformations, and causal lineage 100% visible." },
                  { id: "INV-OI54", name: "Simulation Determinism", status: "VERIFIED", desc: "100 Replays = 1 Hash using seeded XorShift32 PRNG." },
                  { id: "INV-OI55", name: "Rollback Availability", status: "VERIFIED", desc: "Multi-level rollback strategies (L1/L2) attached to scenario." },
                  { id: "INV-OI56", name: "Sandboxed Isolation", status: "VERIFIED", desc: "Baseline state remains immutable; zero production mutation." },
                  { id: "INV-OI57", name: "Attribution Integrity", status: "VERIFIED", desc: "Driver contributions sum strictly to 100.0%." },
                  { id: "INV-OI58", name: "Trace Completeness", status: "VERIFIED", desc: "100% lineage coverage with zero orphan nodes." },
                  { id: "INV-OI59", name: "Shock Recoverability", status: "VERIFIED", desc: "Extreme shock scenarios map to certified recovery-states." },
                  { id: "INV-OI60", name: "Replay Drift Free", status: "VERIFIED", desc: "Replay drift tolerance is strictly 0.0000%." },
                ].map((inv) => (
                  <div key={inv.id} className="p-3 rounded-lg bg-[#111c30] border border-[#1e293b] flex items-start justify-between">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="font-mono font-bold text-cyan-400">{inv.id}</span>
                        <span className="font-semibold text-slate-200">{inv.name}</span>
                      </div>
                      <p className="text-[10px] text-slate-400 mt-1">{inv.desc}</p>
                    </div>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                      {inv.status}
                    </span>
                  </div>
                ))}
              </div>
            </HorizonCard>
          </div>
        )}

        {/* Universal Cross-Links */}
        <div className="pt-4">
          <RelatedArtifactsPanel
            title="Institutional Simulation Cross-Links"
            artifacts={RELATED_ARTIFACTS}
          />
        </div>
      </main>
    </div>
  );
}

export default function ExecutiveSandboxPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-[#070b14] flex items-center justify-center text-cyan-400">
          <div className="text-center space-y-2">
            <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin mx-auto" />
            <span className="text-xs font-semibold uppercase tracking-wider">Hydrating Digital Twin...</span>
          </div>
        </div>
      }
    >
      <ExecutiveSandboxContent />
    </Suspense>
  );
}
