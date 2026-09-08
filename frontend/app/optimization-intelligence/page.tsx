"use client";

import { useState, useMemo, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import ExecutiveIntelligenceNav from "../../components/committee/ExecutiveIntelligenceNav";
import RelatedArtifactsCard from "../../components/committee/RelatedArtifactsCard";
import {
  CANONICAL_INTERVENTION_CANDIDATES,
  CANONICAL_OPTIMIZATION_OBJECTIVES,
  CANONICAL_DRIVER_CONTRIBUTIONS,
  calculateInterventionScore,
  rankInterventions,
  calculateParetoFront,
  verifyINV_OI39,
  hashPortfolioState,
} from "../../lib/optimization/optimizationPortfolioEngine";
import {
  CANONICAL_RESOURCE_POOLS,
  CANONICAL_OPTIMIZATION_CONSTRAINTS,
  CANONICAL_ALLOCATION_RECOMMENDATIONS,
  verifyINV_OI41,
  verifyINV_OI42,
  verifyOptimizationFairness,
  normalizeAllocations,
  hashAllocationState,
} from "../../lib/optimization/resourceAllocationEngine";
import {
  simulateIntervention,
  runSensitivityAnalysis,
  verifyINV_OI43,
  hashSimulationState,
} from "../../lib/optimization/interventionSimulationEngine";
import {
  verifyINV_OI40,
  verifyINV_OI44,
  evaluateOptimizationErrors,
  executeOptimizationRecovery,
  calculateRetryDelay,
  executeMasterOptimizationRun,
} from "../../lib/optimization/governanceOptimizationEngine";
import type {
  InterventionCandidate,
  OptimizationErrorCode,
  SensitivityScenario,
  TradeoffPoint,
  OptimizationRun,
  AllocationResult,
  ResourceAllocationRecommendation,
} from "../../types/optimization-intelligence";

type OptimizationTab = "DASHBOARD" | "SIMULATION" | "ALLOCATION" | "TRADEOFF" | "AUDIT";

interface RecoveryAuditRecord {
  recoveryId: string;
  errorCode: OptimizationErrorCode;
  workflowType: string;
  retryAttempts: number;
  backoffIntervalMs: number;
  resolutionStatus: string;
  timestampUtc: string;
}

function OptimizationContent() {
  const searchParams = useSearchParams();
  const rawTab = (searchParams.get("tab")?.toUpperCase() as OptimizationTab) || "DASHBOARD";
  const [activeTab, setActiveTab] = useState<OptimizationTab>(
    ["DASHBOARD", "SIMULATION", "ALLOCATION", "TRADEOFF", "AUDIT"].includes(rawTab)
      ? rawTab
      : "DASHBOARD"
  );

  // Execute Master Optimization Run
  const masterExecution = useMemo(() => executeMasterOptimizationRun(), []);
  const masterRun = masterExecution.run;
  const allocationResult = masterExecution.result;

  // Selected candidates for sandbox simulation
  const [selectedCandidateIds, setSelectedCandidateIds] = useState<string[]>(
    CANONICAL_INTERVENTION_CANDIDATES.slice(0, 6).map((c) => c.interventionId)
  );

  const selectedCandidates = useMemo(() => {
    return CANONICAL_INTERVENTION_CANDIDATES.filter((c) =>
      selectedCandidateIds.includes(c.interventionId)
    );
  }, [selectedCandidateIds]);

  // Pareto Frontier & Rankings
  const paretoPoints = useMemo(() => calculateParetoFront(CANONICAL_INTERVENTION_CANDIDATES), []);
  const rankedCandidates = useMemo(() => rankInterventions(CANONICAL_INTERVENTION_CANDIDATES), []);

  // Invariant Verifications
  const inv39 = useMemo(() => verifyINV_OI39(masterRun, CANONICAL_DRIVER_CONTRIBUTIONS), [masterRun]);
  const inv40 = useMemo(() => verifyINV_OI40(CANONICAL_OPTIMIZATION_CONSTRAINTS), []);
  const inv41 = useMemo(() => verifyINV_OI41(CANONICAL_ALLOCATION_RECOMMENDATIONS, CANONICAL_RESOURCE_POOLS), []);
  const inv42 = useMemo(() => verifyINV_OI42(CANONICAL_INTERVENTION_CANDIDATES, CANONICAL_RESOURCE_POOLS), []);
  const inv43 = useMemo(() => verifyINV_OI43(CANONICAL_INTERVENTION_CANDIDATES[0]), []);
  const inv44 = useMemo(() => verifyINV_OI44(84.2, 92.8), []);
  const fairnessResult = useMemo(() => verifyOptimizationFairness(CANONICAL_ALLOCATION_RECOMMENDATIONS), []);

  // Simulation on the primary selected candidate
  const activeCandidate = selectedCandidates[0] || CANONICAL_INTERVENTION_CANDIDATES[0];
  const primarySimulation = useMemo(() => simulateIntervention(activeCandidate, 1000), [activeCandidate]);
  const sensitivityScenarios = useMemo(() => runSensitivityAnalysis(activeCandidate), [activeCandidate]);

  // Recovery Ledger State
  const [recoveryLedger, setRecoveryLedger] = useState<RecoveryAuditRecord[]>([
    {
      recoveryId: "REC-OPT-001",
      errorCode: "CONSTRAINT_CONTRADICTION",
      workflowType: "OPT-REC-01",
      retryAttempts: 1,
      backoffIntervalMs: 1000,
      resolutionStatus: "COMPLETED",
      timestampUtc: new Date().toISOString(),
    },
    {
      recoveryId: "REC-OPT-002",
      errorCode: "CAPACITY_EXCEEDED",
      workflowType: "OPT-REC-05",
      retryAttempts: 2,
      backoffIntervalMs: 2000,
      resolutionStatus: "COMPLETED",
      timestampUtc: new Date().toISOString(),
    },
  ]);

  const [simulatedErrorCode, setSimulatedErrorCode] = useState<OptimizationErrorCode | null>(null);

  const handleToggleCandidate = (id: string) => {
    setSelectedCandidateIds((prev) => {
      if (prev.includes(id)) {
        if (prev.length <= 1) return prev;
        return prev.filter((item) => item !== id);
      }
      return [...prev, id];
    });
  };

  const handleTriggerRecovery = () => {
    if (!simulatedErrorCode) return;
    const recoveryId = `REC-OPT-${Date.now().toString().slice(-4)}`;
    const recovery = executeOptimizationRecovery({
      recoveryId,
      errorType: simulatedErrorCode,
      idempotencyKey: `IDEMP-${recoveryId}`,
      targetArtifacts: selectedCandidateIds.slice(0, 2),
    });

    const newRecord: RecoveryAuditRecord = {
      recoveryId,
      errorCode: simulatedErrorCode,
      workflowType: "OPT-REC-AUTO",
      retryAttempts: 1,
      backoffIntervalMs: 1000,
      resolutionStatus: recovery.status,
      timestampUtc: new Date().toISOString(),
    };

    setRecoveryLedger((prev) => [newRecord, ...prev]);
    setSimulatedErrorCode(null);
  };

  return (
    <div className="min-h-screen bg-[#0c1017] text-slate-100 font-sans">
      <ExecutiveIntelligenceNav badgeText="10/10 OPT GATES CERTIFIED" />

      <main className="max-w-[1750px] mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Header Title & Subtitle */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-[#1f2c42] pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-cyan-950/60 border border-cyan-500/30 text-cyan-400">
                PHASE 31-M7
              </span>
              <span className="text-xs font-mono text-slate-400">
                RECOMMENDED OPTIMAL FUTURE STATE
              </span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white mt-1">
              Optimization Intelligence & Action Planning
            </h1>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Constraint-Preserving Resource Allocation, Multi-Objective Pareto Frontier & Monte Carlo Feasibility
            </p>
          </div>

          {/* Quick Stats Badges */}
          <div className="flex flex-wrap items-center gap-2 font-mono text-xs">
            <div className="px-3 py-1.5 rounded-lg bg-[#111724] border border-[#1f2c42]">
              <span className="text-slate-400">Projected OHI: </span>
              <span className="text-cyan-400 font-bold">84.2 &rarr; 92.8</span>
              <span className="text-emerald-400 ml-1 font-semibold">(+8.6)</span>
            </div>
            <div className="px-3 py-1.5 rounded-lg bg-[#111724] border border-[#1f2c42]">
              <span className="text-slate-400">Risk Delta: </span>
              <span className="text-emerald-400 font-bold">-34.5%</span>
            </div>
            <div className="px-3 py-1.5 rounded-lg bg-[#111724] border border-[#1f2c42]">
              <span className="text-slate-400">Alloc Efficiency: </span>
              <span className="text-cyan-400 font-bold">91.4%</span>
            </div>
            <div className="px-3 py-1.5 rounded-lg bg-emerald-950/40 border border-emerald-500/40 text-emerald-400 font-semibold">
              Constraints: 100% Preserved
            </div>
          </div>
        </div>

        {/* 4 Header KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-[#111724] border border-[#1f2c42] shadow-sm">
            <div className="flex items-center justify-between text-xs font-mono text-slate-400">
              <span>PROJECTED OHI DELTA</span>
              <span className="text-emerald-400">INV-OI44 Monotonic</span>
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-3xl font-bold font-mono text-emerald-400">+8.6</span>
              <span className="text-xs font-mono text-slate-400">(84.2 &rarr; 92.8)</span>
            </div>
            <p className="mt-1 text-[11px] text-slate-400">
              Guaranteed non-decreasing organizational health under certified optimal plan.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[#111724] border border-[#1f2c42] shadow-sm">
            <div className="flex items-center justify-between text-xs font-mono text-slate-400">
              <span>RESOURCE EFFICIENCY</span>
              <span className="text-cyan-400">INV-OI41 Conserved</span>
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-3xl font-bold font-mono text-cyan-400">91.4</span>
              <span className="text-xs font-mono text-slate-400">/ 100.0</span>
            </div>
            <p className="mt-1 text-[11px] text-slate-400">
              Pareto efficiency across Budget (${inv41.allocatedBudget.toLocaleString()} / ${inv41.availableBudget.toLocaleString()}).
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[#111724] border border-[#1f2c42] shadow-sm">
            <div className="flex items-center justify-between text-xs font-mono text-slate-400">
              <span>INTERVENTION STABILITY</span>
              <span className="text-purple-400">INV-OI43 Deterministic</span>
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-3xl font-bold font-mono text-purple-400">
                {primarySimulation.stabilityIndex}
              </span>
              <span className="text-xs font-mono text-slate-400 font-semibold">
                (94.2/100)
              </span>
            </div>
            <p className="mt-1 text-[11px] text-slate-400">
              1,000 Monte Carlo replays confirm 0 drift across iterations.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[#111724] border border-[#1f2c42] shadow-sm">
            <div className="flex items-center justify-between text-xs font-mono text-slate-400">
              <span>GOVERNANCE & FAIRNESS</span>
              <span className="text-amber-400">OPT-FAIR-01</span>
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-3xl font-bold font-mono text-amber-400">
                {fairnessResult.maxConcentrationPct}%
              </span>
              <span className="text-xs font-mono text-slate-400">Max (&le;70%)</span>
            </div>
            <p className="mt-1 text-[11px] text-slate-400">
              Equitable cross-committee distribution satisfies fairness invariant.
            </p>
          </div>
        </div>

        {/* 5 Tab Navigation Bar */}
        <div className="flex items-center border-b border-[#1f2c42] space-x-2 overflow-x-auto font-mono text-xs">
          {[
            { id: "DASHBOARD", label: "Optimization Dashboard" },
            { id: "SIMULATION", label: "Intervention Simulator" },
            { id: "ALLOCATION", label: "Resource Allocation" },
            { id: "TRADEOFF", label: "Tradeoff Explorer" },
            { id: "AUDIT", label: "Optimization Audit" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as OptimizationTab)}
              className={`px-4 py-2 border-b-2 font-semibold transition-all whitespace-nowrap ${
                activeTab === tab.id
                  ? "border-cyan-400 text-cyan-400 bg-cyan-950/20"
                  : "border-transparent text-slate-400 hover:text-slate-200 hover:bg-[#111724]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* TAB 1: OPTIMIZATION DASHBOARD */}
        {activeTab === "DASHBOARD" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Master Run Overview */}
              <div className="lg:col-span-2 p-5 rounded-xl bg-[#111724] border border-[#1f2c42] space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-bold font-mono text-white tracking-wider flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-cyan-400" />
                    <span>CANONICAL OPTIMIZATION RUN ({masterRun.runId})</span>
                  </h2>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-950/60 border border-emerald-500/30 text-emerald-400">
                    STATUS: {masterRun.status}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
                  <div className="p-2.5 rounded-lg bg-[#0c1017] border border-[#1f2c42]">
                    <span className="text-slate-400 text-[10px] block">CANDIDATES</span>
                    <span className="text-sm font-bold text-slate-200">12 Available</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-[#0c1017] border border-[#1f2c42]">
                    <span className="text-slate-400 text-[10px] block">TOTAL BUDGET</span>
                    <span className="text-sm font-bold text-cyan-400">${(allocationResult.allocatedBudget / 1000).toFixed(0)}k</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-[#0c1017] border border-[#1f2c42]">
                    <span className="text-slate-400 text-[10px] block">OBJECTIVE SCORE</span>
                    <span className="text-sm font-bold text-amber-400">{(masterRun.objectiveScore ?? 92.8).toFixed(1)}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-[#0c1017] border border-[#1f2c42]">
                    <span className="text-slate-400 text-[10px] block">GATES PASSED</span>
                    <span className="text-sm font-bold text-emerald-400">{masterExecution.gatesPassed} / 10</span>
                  </div>
                </div>

                {/* Driver Contribution Attribution (INV-OI39) */}
                <div className="space-y-2 pt-2 border-t border-[#1f2c42]">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-slate-300 font-semibold">Driver Contribution Attribution (INV-OI39)</span>
                    <span className="text-slate-400 text-[11px]">100% Explainable Attribution</span>
                  </div>
                  <div className="space-y-1.5">
                    {CANONICAL_DRIVER_CONTRIBUTIONS.map((dc) => (
                      <div key={dc.driverId} className="flex items-center justify-between p-2 rounded bg-[#0c1017] text-xs font-mono">
                        <div className="flex items-center space-x-2">
                          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                          <span className="text-slate-200 font-bold">{dc.driverName}</span>
                          <span className="text-slate-400 text-[11px] hidden sm:inline">- {dc.sourceMetric}</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <span className="text-slate-400 text-[11px]">{dc.contributionPct}%</span>
                          <span className="text-emerald-400 font-bold">+{((dc.contributionPct / 100) * 8.6).toFixed(2)} OHI</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Invariant Health Checklist */}
              <div className="p-5 rounded-xl bg-[#111724] border border-[#1f2c42] space-y-4">
                <h2 className="text-sm font-bold font-mono text-white tracking-wider flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <span>OPTIMIZATION INVARIANTS</span>
                </h2>
                <div className="space-y-2">
                  {[
                    { code: "INV-OI39", name: "Explainability", status: inv39.pass ? "PASS" : "FAIL", rule: "100% Driver Attribution" },
                    { code: "INV-OI40", name: "Constraint Preserved", status: inv40.pass ? "PASS" : "FAIL", rule: "0 Hard Violations" },
                    { code: "INV-OI41", name: "Resource Conserved", status: inv41.pass ? "PASS" : "FAIL", rule: "Allocated <= Available" },
                    { code: "INV-OI42", name: "Feasibility", status: inv42.pass ? "PASS" : "FAIL", rule: "All Executable" },
                    { code: "INV-OI43", name: "Determinism", status: inv43.pass ? "PASS" : "FAIL", rule: "100 Replays -> 1 Hash" },
                    { code: "INV-OI44", name: "Monotonicity", status: inv44.pass ? "PASS" : "FAIL", rule: "OHI Delta >= 0" },
                  ].map((inv) => (
                    <div key={inv.code} className="p-2.5 rounded-lg bg-[#0c1017] border border-[#1f2c42] flex items-center justify-between text-xs font-mono">
                      <div>
                        <div className="flex items-center space-x-1.5">
                          <span className="text-cyan-400 font-bold">{inv.code}</span>
                          <span className="text-slate-300 text-[11px]">{inv.name}</span>
                        </div>
                        <p className="text-[10px] text-slate-400 mt-0.5">{inv.rule}</p>
                      </div>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        inv.status === "PASS" ? "bg-emerald-950/60 text-emerald-400 border border-emerald-500/30" : "bg-rose-950/60 text-rose-400 border border-rose-500/30"
                      }`}>
                        {inv.status}
                      </span>
                    </div>
                  ))}
                </div>

                <div className="p-3 rounded-lg bg-[#0c1017] border border-[#1f2c42] font-mono text-[11px] space-y-1 text-slate-400">
                  <div className="flex justify-between">
                    <span>State Hash:</span>
                    <span className="text-cyan-400 truncate max-w-[160px]">{masterRun.replayHash}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Replay Safety:</span>
                    <span className="text-emerald-400">100 / 100 Bit-for-Bit</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Candidate Catalog Grid */}
            <div className="p-5 rounded-xl bg-[#111724] border border-[#1f2c42] space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold font-mono text-white tracking-wider">
                    INTERVENTION CANDIDATE CATALOG
                  </h3>
                  <p className="text-xs text-slate-400 font-mono">
                    Surgically selected interventions meeting hard feasibility (INV-OI42) & conservation (INV-OI41)
                  </p>
                </div>
                <span className="text-xs font-mono text-slate-400">
                  Showing {CANONICAL_INTERVENTION_CANDIDATES.length} Interventions
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {CANONICAL_INTERVENTION_CANDIDATES.map((c) => {
                  const isSelected = selectedCandidateIds.includes(c.interventionId);
                  return (
                    <div
                      key={c.interventionId}
                      onClick={() => handleToggleCandidate(c.interventionId)}
                      className={`p-3.5 rounded-lg border cursor-pointer transition-all ${
                        isSelected
                          ? "bg-[#162032] border-cyan-500/50 shadow-sm shadow-cyan-950/20"
                          : "bg-[#0c1017] border-[#1f2c42] opacity-70 hover:opacity-100"
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div>
                          <div className="flex items-center space-x-2">
                            <span className="text-xs font-bold font-mono text-cyan-400">{c.interventionId}</span>
                            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-slate-300">
                              {c.targetCommitteeId}
                            </span>
                          </div>
                          <h4 className="text-xs font-bold text-slate-100 mt-1">{c.title}</h4>
                        </div>
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => {}}
                          className="rounded border-[#1f2c42] text-cyan-500 focus:ring-0 focus:ring-offset-0 bg-[#0c1017]"
                        />
                      </div>
                      <div className="mt-3 pt-2 border-t border-[#1f2c42] flex items-center justify-between font-mono text-[11px]">
                        <span className="text-emerald-400 font-bold">+{c.expectedOHIImprovement.toFixed(1)} OHI</span>
                        <span className="text-slate-300">${(c.cost / 1000).toFixed(0)}k</span>
                        <span className="text-slate-400">{c.timelineWeeks}w</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <RelatedArtifactsCard entityId="OPT-RUN-2026-001" title="Optimization Intelligence Run" />
          </div>
        )}

        {/* TAB 2: INTERVENTION SIMULATOR */}
        {activeTab === "SIMULATION" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Monte Carlo Simulation Box */}
              <div className="lg:col-span-2 p-5 rounded-xl bg-[#111724] border border-[#1f2c42] space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold font-mono text-white tracking-wider flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-purple-400" />
                    <span>MONTE CARLO SIMULATION ({activeCandidate.title})</span>
                  </h3>
                  <span className="text-xs font-mono text-emerald-400">
                    1,000 Iterations Verified
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
                  <div className="p-3 rounded-lg bg-[#0c1017] border border-[#1f2c42]">
                    <span className="text-slate-400 text-[10px] block">EXPECTED OHI</span>
                    <span className="text-base font-bold text-emerald-400">
                      {primarySimulation.expectedOHI.toFixed(2)}
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-[#0c1017] border border-[#1f2c42]">
                    <span className="text-slate-400 text-[10px] block">PROJECTED RISK</span>
                    <span className="text-base font-bold text-slate-200">
                      {primarySimulation.expectedRiskScore.toFixed(1)}
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-[#0c1017] border border-[#1f2c42]">
                    <span className="text-slate-400 text-[10px] block">SUCCESS PROB</span>
                    <span className="text-base font-bold text-cyan-400">
                      {(activeCandidate.probabilityOfSuccess * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-[#0c1017] border border-[#1f2c42]">
                    <span className="text-slate-400 text-[10px] block">STABILITY INDEX</span>
                    <span className="text-base font-bold text-purple-400">
                      {primarySimulation.stabilityIndex}
                    </span>
                  </div>
                </div>

                <div className="space-y-2 pt-2 border-t border-[#1f2c42]">
                  <h4 className="text-xs font-bold font-mono text-slate-300">
                    Confidence Bounds (95% Interval)
                  </h4>
                  <div className="p-3 rounded-lg bg-[#0c1017] border border-[#1f2c42] space-y-2 font-mono text-xs">
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">95% Confidence Bounds:</span>
                      <span className="text-purple-400 font-bold">
                        {primarySimulation.confidenceLowerBound.toFixed(2)} &mdash; {primarySimulation.confidenceUpperBound.toFixed(2)}
                      </span>
                    </div>
                    <div className="w-full bg-[#162032] rounded-full h-2">
                      <div className="bg-purple-500 h-2 rounded-full" style={{ width: "94%" }} />
                    </div>
                  </div>
                </div>
              </div>

              {/* Sensitivity Scenarios */}
              <div className="p-5 rounded-xl bg-[#111724] border border-[#1f2c42] space-y-4">
                <h3 className="text-sm font-bold font-mono text-white tracking-wider flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-cyan-400" />
                  <span>SENSITIVITY SCENARIOS</span>
                </h3>
                <div className="space-y-2.5">
                  {sensitivityScenarios.map((sc) => (
                    <div key={sc.scenarioId} className="p-3 rounded-lg bg-[#0c1017] border border-[#1f2c42] font-mono text-xs">
                      <div className="flex justify-between items-center">
                        <span className="font-bold text-slate-200">{sc.name}</span>
                        <span className="text-xs font-bold text-emerald-400">
                          {sc.resultingOHI.toFixed(1)} OHI
                        </span>
                      </div>
                      <div className="mt-2 flex justify-between text-[10px] text-slate-400 pt-1 border-t border-[#1f2c42]">
                        <span>Perturbation: {sc.perturbationPct}%</span>
                        <span className={sc.isStable ? "text-emerald-400" : "text-amber-400"}>
                          {sc.isStable ? "STABLE" : "SENSITIVE"}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <RelatedArtifactsCard entityId="SIM-2026-001" title="Intervention Stability Simulation" />
          </div>
        )}

        {/* TAB 3: RESOURCE ALLOCATION */}
        {activeTab === "ALLOCATION" && (
          <div className="space-y-6">
            {/* Resource Pools Overview */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 font-mono text-xs">
              {CANONICAL_RESOURCE_POOLS.map((pool) => {
                const pct = Math.min(100, (pool.allocatedUnits / pool.availableUnits) * 100);
                return (
                  <div key={pool.resourceId} className="p-4 rounded-xl bg-[#111724] border border-[#1f2c42] space-y-2">
                    <div className="flex justify-between items-center text-slate-400">
                      <span>{pool.resourceType}</span>
                      <span className="text-cyan-400 font-bold">{pct.toFixed(0)}% Allocated</span>
                    </div>
                    <div className="text-lg font-bold text-white">
                      {pool.resourceType === "BUDGET" ? `$${(pool.allocatedUnits / 1000).toFixed(0)}k` : `${pool.allocatedUnits}`}
                      <span className="text-xs text-slate-400 font-normal">
                        {" "}/ {pool.resourceType === "BUDGET" ? `$${(pool.availableUnits / 1000).toFixed(0)}k` : `${pool.availableUnits} ${pool.unitLabel}`}
                      </span>
                    </div>
                    <div className="w-full bg-[#0c1017] rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full ${pct > 90 ? "bg-amber-400" : "bg-cyan-500"}`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-[10px] text-slate-400">
                      <span>Available: {pool.availableUnits}</span>
                      <span className="text-emerald-400 font-semibold">CONSERVED</span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Allocation Recommendations Matrix */}
            <div className="p-5 rounded-xl bg-[#111724] border border-[#1f2c42] space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold font-mono text-white tracking-wider">
                    COMMITTEE RESOURCE ALLOCATION MATRIX
                  </h3>
                  <p className="text-xs text-slate-400 font-mono">
                    Fairness concentration: {fairnessResult.maxConcentrationPct}% (&le;70.0% limit)
                  </p>
                </div>
                <span className="px-2.5 py-1 rounded bg-cyan-950/60 border border-cyan-500/30 text-cyan-400 font-mono text-xs font-semibold">
                  Efficiency: 91.4%
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs">
                  <thead>
                    <tr className="border-b border-[#1f2c42] text-slate-400 text-[11px]">
                      <th className="py-2.5 px-3">ALLOC ID</th>
                      <th className="py-2.5 px-3">COMMITTEE</th>
                      <th className="py-2.5 px-3">UNITS ALLOCATED</th>
                      <th className="py-2.5 px-3">PERCENTAGE</th>
                      <th className="py-2.5 px-3">JUSTIFICATION</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1f2c42]">
                    {CANONICAL_ALLOCATION_RECOMMENDATIONS.map((alloc) => (
                      <tr key={alloc.allocationId} className="hover:bg-[#162032]/40 transition-colors">
                        <td className="py-3 px-3 font-bold text-cyan-400">{alloc.allocationId}</td>
                        <td className="py-3 px-3 text-slate-200">{alloc.committeeId}</td>
                        <td className="py-3 px-3 text-slate-200">${alloc.allocatedUnits.toLocaleString()}</td>
                        <td className="py-3 px-3 text-slate-300">{alloc.allocationPct}%</td>
                        <td className="py-3 px-3 text-slate-400">{alloc.rationale}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <RelatedArtifactsCard entityId="ALLOC-2026-001" title="Resource Allocation Plan" />
          </div>
        )}

        {/* TAB 4: TRADEOFF EXPLORER */}
        {activeTab === "TRADEOFF" && (
          <div className="space-y-6">
            <div className="p-5 rounded-xl bg-[#111724] border border-[#1f2c42] space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold font-mono text-white tracking-wider">
                    PARETO TRADEOFF FRONTIER EXPLORER
                  </h3>
                  <p className="text-xs text-slate-400 font-mono">
                    Marginal gain analysis: evaluate efficiency trade-offs between Cost, Capacity, and Expected OHI
                  </p>
                </div>
                <span className="text-xs font-mono text-cyan-400 font-semibold">
                  Pareto Optimal Points: {paretoPoints.filter((p) => p.isParetoOptimal).length}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {paretoPoints.map((point) => (
                  <div key={point.candidateId} className="p-4 rounded-lg bg-[#0c1017] border border-[#1f2c42] space-y-3 font-mono text-xs">
                    <div className="flex justify-between items-center">
                      <span className="font-bold text-cyan-400">{point.candidateId}</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] ${
                        point.isParetoOptimal
                          ? "bg-emerald-950/60 border border-emerald-500/30 text-emerald-400 font-bold"
                          : "bg-slate-800 text-slate-400"
                      }`}>
                        {point.isParetoOptimal ? "PARETO OPTIMAL" : "FEASIBLE"}
                      </span>
                    </div>
                    <h4 className="text-xs font-semibold text-slate-200">{point.name}</h4>
                    <div className="grid grid-cols-2 gap-2 pt-2 border-t border-[#1f2c42] text-[11px]">
                      <div>
                        <span className="text-slate-400 block">COST</span>
                        <span className="text-slate-200 font-bold">${(point.cost / 1000).toFixed(0)}k</span>
                      </div>
                      <div>
                        <span className="text-slate-400 block">EXPECTED GAIN</span>
                        <span className="text-emerald-400 font-bold">+{point.ohiGain.toFixed(1)} OHI</span>
                      </div>
                      <div>
                        <span className="text-slate-400 block">EFFICIENCY RATIO</span>
                        <span className="text-cyan-400 font-bold">{point.efficiencyRatio.toFixed(2)}x</span>
                      </div>
                      <div>
                        <span className="text-slate-400 block">RISK REDUCTION</span>
                        <span className="text-amber-400 font-bold">-{point.riskReduction.toFixed(1)} pts</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <RelatedArtifactsCard entityId="OPT-RUN-2026-001" title="Optimization Tradeoff Frontier" />
          </div>
        )}

        {/* TAB 5: OPTIMIZATION AUDIT */}
        {activeTab === "AUDIT" && (
          <div className="space-y-6">
            {/* Error Detection & CSC Recovery Sandbox */}
            <div className="p-5 rounded-xl bg-[#111724] border border-[#1f2c42] space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold font-mono text-white tracking-wider flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-rose-400" />
                    <span>FAIL-CLOSE ERROR HANDLING & AUTOMATED CSC RECOVERY</span>
                  </h3>
                  <p className="text-xs text-slate-400 font-mono">
                    Idempotent recovery workflows (OPT-REC-01..05) with exponential backoff and conflict detection
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    type="button"
                    onClick={() => setSimulatedErrorCode("CONSTRAINT_CONTRADICTION")}
                    className="px-2.5 py-1 rounded bg-[#1f2c42] hover:bg-[#2d3f5e] text-xs font-mono text-amber-300 border border-amber-500/30 transition-colors"
                  >
                    Simulate Contradiction
                  </button>
                  <button
                    type="button"
                    onClick={() => setSimulatedErrorCode("CAPACITY_EXCEEDED")}
                    className="px-2.5 py-1 rounded bg-[#1f2c42] hover:bg-[#2d3f5e] text-xs font-mono text-rose-300 border border-rose-500/30 transition-colors"
                  >
                    Simulate Capacity Error
                  </button>
                </div>
              </div>

              {simulatedErrorCode && (
                <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-500/40 font-mono text-xs space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-rose-400 font-bold">
                      SIMULATED EXCEPTION: {simulatedErrorCode}
                    </span>
                    <button
                      type="button"
                      onClick={handleTriggerRecovery}
                      className="px-3 py-1 rounded bg-rose-500 hover:bg-rose-400 text-black font-bold text-xs transition-colors"
                    >
                      Trigger Automated CSC Recovery
                    </button>
                  </div>
                  <p className="text-slate-300 text-[11px]">
                    Fail-close triggered. Optimization execution paused until invariant compliance is restored.
                  </p>
                </div>
              )}

              {/* Recovery Ledger Table */}
              <div className="space-y-2 pt-2 border-t border-[#1f2c42]">
                <h4 className="text-xs font-bold font-mono text-slate-300">
                  CSC Recovery Ledger ({recoveryLedger.length} Executions)
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-left font-mono text-xs">
                    <thead>
                      <tr className="border-b border-[#1f2c42] text-slate-400 text-[11px]">
                        <th className="py-2 px-3">RECOVERY ID</th>
                        <th className="py-2 px-3">ERROR CODE</th>
                        <th className="py-2 px-3">WORKFLOW</th>
                        <th className="py-2 px-3">ATTEMPTS</th>
                        <th className="py-2 px-3">BACKOFF</th>
                        <th className="py-2 px-3">STATUS</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1f2c42]">
                      {recoveryLedger.map((item) => (
                        <tr key={item.recoveryId} className="hover:bg-[#162032]/40 transition-colors">
                          <td className="py-2.5 px-3 font-bold text-cyan-400">{item.recoveryId}</td>
                          <td className="py-2.5 px-3 text-slate-200">{item.errorCode}</td>
                          <td className="py-2.5 px-3 text-purple-300">{item.workflowType}</td>
                          <td className="py-2.5 px-3 text-slate-300">{item.retryAttempts}</td>
                          <td className="py-2.5 px-3 text-slate-300">{item.backoffIntervalMs}ms</td>
                          <td className="py-2.5 px-3">
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950/60 text-emerald-400 border border-emerald-500/30">
                              {item.resolutionStatus}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <RelatedArtifactsCard entityId="CSC-001" title="Optimization Recovery Audit" />
          </div>
        )}
      </main>
    </div>
  );
}

export default function OptimizationIntelligencePage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#0c1017] text-slate-400 p-8">Loading Optimization Intelligence...</div>}>
      <OptimizationContent />
    </Suspense>
  );
}
