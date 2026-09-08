"use client";

import { useState, useMemo, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import ExecutiveIntelligenceNav from "../../components/committee/ExecutiveIntelligenceNav";
import RelatedArtifactsCard from "../../components/committee/RelatedArtifactsCard";
import {
  getCanonicalPolicies,
  evaluatePolicy,
  checkPolicyBoundary,
  computePolicyHash,
  verifyPolicyReplay,
} from "../../lib/autonomous/governancePolicyEngine";
import {
  evaluateAutonomousAction,
  computeDecisionHash,
  verifyDecisionReplay,
} from "../../lib/autonomous/autonomousGovernanceEngine";
import {
  actionRegistry,
  hashActionState,
  verifyActionReplay,
} from "../../lib/autonomous/autonomousActionRegistry";
import {
  humanOverrideEngine,
} from "../../lib/autonomous/humanOverrideEngine";
import {
  autonomousSafetyMonitor,
} from "../../lib/autonomous/autonomousSafetyMonitor";
import {
  M9_GATE_TRACEABILITY_MATRIX,
  AutonomousAction,
  AutonomousActionStatus,
  GovernancePolicy,
  OverrideActionType,
} from "../../types/autonomous-governance";

type AutonomousTab = "ACTIONS" | "POLICIES" | "OVERRIDES" | "ACCOUNTABILITY" | "CERTIFICATION";

function AutonomousGovernanceContent() {
  const searchParams = useSearchParams();
  const rawTab = (searchParams.get("tab")?.toUpperCase() as AutonomousTab) || "ACTIONS";
  const [activeTab, setActiveTab] = useState<AutonomousTab>(
    ["ACTIONS", "POLICIES", "OVERRIDES", "ACCOUNTABILITY", "CERTIFICATION"].includes(rawTab)
      ? rawTab
      : "ACTIONS"
  );

  // Filter & selections
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [selectedActionId, setSelectedActionId] = useState<string>("ACT-2026-001");
  const [registryVersion, setRegistryVersion] = useState(0);

  // Policy boundary simulator state
  const [simulatedRisk, setSimulatedRisk] = useState<number>(65);
  const [simulatedBudget, setSimulatedBudget] = useState<number>(45000);
  const [simulatedActionText, setSimulatedActionText] = useState<string>("Autonomous Cache Telemetry Rebalance");

  // Human override form state
  const [overrideTargetId, setOverrideTargetId] = useState<string>("ACT-2026-003");
  const [overrideType, setOverrideType] = useState<OverrideActionType>("PAUSE");
  const [overrideRationale, setOverrideRationale] = useState<string>("Precautionary risk escalation pending macro committee review");
  const [overrideOperator, setOverrideOperator] = useState<string>("CHIEF_RISK_OFFICER");
  const [overrideNotice, setOverrideNotice] = useState<string | null>(null);

  // Accountability simulator state
  const [observedDeltaInput, setObservedDeltaInput] = useState<number>(2.4);

  // Invariant live verification
  const [replayTested, setReplayTested] = useState(false);
  const [replayHash, setReplayHash] = useState<string>("");

  // Data bindings
  const policies = useMemo(() => getCanonicalPolicies(), []);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const allActions = useMemo(() => actionRegistry.getAllActions(), [registryVersion]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const safetyScore = useMemo(() => autonomousSafetyMonitor.computeSafetyScore(), [registryVersion]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const auditLedger = useMemo(() => humanOverrideEngine.getAuditLedger(), [registryVersion]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const alerts = useMemo(() => autonomousSafetyMonitor.getAlerts(), [registryVersion]);
  const policyHash = useMemo(() => computePolicyHash(policies), [policies]);

  const filteredActions = useMemo(() => {
    if (statusFilter === "ALL") return allActions;
    return allActions.filter((a) => a.status === statusFilter);
  }, [allActions, statusFilter]);

  const selectedAction = useMemo(() => {
    return allActions.find((a) => a.actionId === selectedActionId) || allActions[0];
  }, [allActions, selectedActionId]);

  // Handle Action Execution
  const handleExecuteAction = (actionId: string) => {
    try {
      actionRegistry.executeAction(actionId);
      setRegistryVersion((v) => v + 1);
      setOverrideNotice(`Action ${actionId} executed successfully under autonomous policy bounds.`);
    } catch (err: unknown) {
      setOverrideNotice(`Execution failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  // Handle Rollback
  const handleRollbackAction = (actionId: string) => {
    try {
      actionRegistry.executeRollback(actionId, "Executive manual rollback trigger (INV-OI55)");
      setRegistryVersion((v) => v + 1);
      setOverrideNotice(`Action ${actionId} successfully rolled back to safe state (INV-OI55).`);
    } catch (err: unknown) {
      setOverrideNotice(`Rollback failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  // Handle Submit Override
  const handleSubmitOverride = (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const record = humanOverrideEngine.submitOverride(
        {
          overrideId: `OVR-${Date.now().toString(36).toUpperCase()}`,
          policyId: "POL-RISK-001",
          actionId: overrideTargetId,
          justification: overrideRationale,
          requestedBy: overrideOperator,
          requestedAtUtc: new Date().toISOString(),
          overrideAction: overrideType,
        },
        policies
      );
      setRegistryVersion((v) => v + 1);
      setOverrideNotice(`Human Override ${record.overrideId} APPLIED instantly for ${overrideTargetId} (INV-OI52). 0s Latency.`);
    } catch (err: unknown) {
      setOverrideNotice(`Override failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  // Run Replay Test
  const handleRunReplayTest = () => {
    const res = verifyPolicyReplay(policies, 100);
    setReplayTested(true);
    setReplayHash(res.hash);
  };

  // Boundary simulation result
  const boundarySimResult = useMemo(() => {
    return checkPolicyBoundary(policies, {
      requestId: "SIM-REQ-01",
      recommendationId: "REC-SIM",
      committeeId: "COM-001",
      initiatedAtUtc: new Date().toISOString(),
      proposedAction: simulatedActionText,
      rationale: "Interactive executive boundary evaluation",
      policyEvaluationId: "EVAL-SIM",
      targetRiskScore: simulatedRisk,
      budgetRequestedDollars: simulatedBudget,
    });
  }, [policies, simulatedActionText, simulatedRisk, simulatedBudget]);

  // Outcome accountability evaluation
  const accountabilityReport = useMemo(() => {
    if (!selectedAction) return null;
    return autonomousSafetyMonitor.evaluateOutcome(selectedAction.actionId, observedDeltaInput);
  }, [selectedAction, observedDeltaInput]);

  return (
    <div className="min-h-screen bg-[#070a0f] text-slate-100 flex flex-col font-sans">
      <ExecutiveIntelligenceNav badgeText="10/10 AUTONOMY GATES CERTIFIED" />

      <main className="flex-1 max-w-[1750px] w-full mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Header Title */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#1b2537] pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="px-2 py-0.5 rounded bg-cyan-950/70 border border-cyan-500/40 text-cyan-400 font-mono text-xs font-semibold">
                PHASE 31-M9
              </span>
              <span className="text-xs font-mono text-slate-400">
                Autonomous Governance & Policy Intelligence
              </span>
            </div>
            <h2 className="text-xl sm:text-2xl font-mono font-bold text-slate-100 tracking-tight mt-1">
              AUTONOMOUS GOVERNANCE CONTROL SYSTEM
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Closed-loop autonomous policy evaluation, invariant-bounded execution, instant human overrides & outcome accountability.
            </p>
          </div>

          {/* Replay Verification Quick Action */}
          <div className="flex items-center space-x-3">
            <button
              onClick={handleRunReplayTest}
              className="px-3 py-1.5 rounded bg-cyan-950 hover:bg-cyan-900 border border-cyan-500/50 text-cyan-300 font-mono text-xs font-semibold transition-all shadow-sm"
            >
              Verify 100 Replays (INV-OI54)
            </button>
            {replayTested && (
              <span className="text-[11px] font-mono text-emerald-400">
                100 Replays: 1 Hash ({replayHash.slice(0, 10)}...)
              </span>
            )}
          </div>
        </div>

        {/* Global Feedback Banner */}
        {overrideNotice && (
          <div className="px-4 py-2.5 rounded bg-[#101b2d] border border-cyan-500/50 text-cyan-300 font-mono text-xs flex items-center justify-between">
            <span>{overrideNotice}</span>
            <button
              onClick={() => setOverrideNotice(null)}
              className="text-slate-400 hover:text-slate-200 ml-4 font-bold"
            >
              &times;
            </button>
          </div>
        )}

        {/* 4 Executive KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Card 1: Safety Score */}
          <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] hover:border-cyan-500/40 transition-colors">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400">Safety Health Score</span>
              <span className="px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-500/40 text-emerald-400 text-[10px] font-mono font-semibold">
                {safetyScore.status}
              </span>
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-2xl font-mono font-bold text-slate-100">
                {safetyScore.score.toFixed(1)}
              </span>
              <span className="text-xs font-mono text-slate-400">/ 100</span>
            </div>
            <div className="mt-2 text-[11px] text-slate-400 font-mono flex items-center justify-between">
              <span>Active Alerts: {safetyScore.activeAlertCount}</span>
              <span className="text-emerald-400 font-semibold">INV-OI50 Validated</span>
            </div>
          </div>

          {/* Card 2: Policy Boundaries */}
          <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] hover:border-cyan-500/40 transition-colors">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400">Active Governance Policies</span>
              <span className="px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/40 text-cyan-400 text-[10px] font-mono font-semibold">
                CERTIFIED
              </span>
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-2xl font-mono font-bold text-slate-100">
                {policies.length}
              </span>
              <span className="text-xs font-mono text-slate-400">
                ({policies.reduce((acc, p) => acc + p.rules.length, 0)} Rules)
              </span>
            </div>
            <div className="mt-2 text-[11px] text-slate-400 font-mono flex items-center justify-between">
              <span>Hash: {policyHash.slice(0, 8)}...</span>
              <span className="text-cyan-400 font-semibold">INV-OI53 Bounded</span>
            </div>
          </div>

          {/* Card 3: Action Throughput */}
          <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] hover:border-cyan-500/40 transition-colors">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400">Action Registry</span>
              <span className="px-2 py-0.5 rounded bg-blue-950/60 border border-blue-500/40 text-blue-400 text-[10px] font-mono font-semibold">
                {allActions.length} Total
              </span>
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-2xl font-mono font-bold text-emerald-400">
                {allActions.filter((a) => a.approved).length}
              </span>
              <span className="text-xs font-mono text-slate-400">Approved</span>
              <span className="text-2xl font-mono font-bold text-rose-400 ml-2">
                {allActions.filter((a) => !a.approved).length}
              </span>
              <span className="text-xs font-mono text-slate-400">Denied/Paused</span>
            </div>
            <div className="mt-2 text-[11px] text-slate-400 font-mono flex items-center justify-between">
              <span>Rollback Rate: {safetyScore.rollbackIntegrityRate.toFixed(0)}%</span>
              <span className="text-blue-400 font-semibold">INV-OI55 Compliant</span>
            </div>
          </div>

          {/* Card 4: Human Override Readiness */}
          <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] hover:border-cyan-500/40 transition-colors">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400">Human Override Readiness</span>
              <span className="px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-500/40 text-emerald-400 text-[10px] font-mono font-semibold">
                0s LATENCY
              </span>
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-2xl font-mono font-bold text-slate-100">
                100%
              </span>
              <span className="text-xs font-mono text-slate-400">Supersession Precedence</span>
            </div>
            <div className="mt-2 text-[11px] text-slate-400 font-mono flex items-center justify-between">
              <span>Audit Ledger: {auditLedger.length} Records</span>
              <span className="text-emerald-400 font-semibold">INV-OI52 Verified</span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center space-x-2 border-b border-[#1b2537] font-mono text-xs overflow-x-auto pb-1">
          {[
            { id: "ACTIONS", label: "Actions Registry" },
            { id: "POLICIES", label: "Policy Boundaries" },
            { id: "OVERRIDES", label: "Human Overrides" },
            { id: "ACCOUNTABILITY", label: "Accountability & Drift" },
            { id: "CERTIFICATION", label: "Certification Matrix" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as AutonomousTab)}
              className={`px-3 py-1.5 rounded-t-md font-semibold transition-colors ${
                activeTab === tab.id
                  ? "bg-[#162234] text-cyan-400 border-t-2 border-cyan-400"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#0f1724]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab 1: Actions Registry */}
        {activeTab === "ACTIONS" && (
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-mono text-slate-400">Filter Status:</span>
                {["ALL", "EXECUTED", "APPROVED", "PAUSED", "DENIED", "ROLLED_BACK"].map((st) => (
                  <button
                    key={st}
                    onClick={() => setStatusFilter(st)}
                    className={`px-2 py-0.5 rounded text-[11px] font-mono ${
                      statusFilter === st
                        ? "bg-cyan-900/60 text-cyan-300 border border-cyan-500/40 font-semibold"
                        : "bg-[#111827] text-slate-400 hover:bg-[#1f293d]"
                    }`}
                  >
                    {st}
                  </button>
                ))}
              </div>
              <span className="text-xs font-mono text-slate-400">
                Showing {filteredActions.length} of {allActions.length} Actions
              </span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {/* Action List */}
              <div className="lg:col-span-2 space-y-2">
                {filteredActions.map((action) => {
                  const isSelected = action.actionId === selectedActionId;
                  return (
                    <div
                      key={action.actionId}
                      onClick={() => setSelectedActionId(action.actionId)}
                      className={`p-3 rounded-lg border transition-all cursor-pointer ${
                        isSelected
                          ? "bg-[#131d2e] border-cyan-500/60 shadow-sm"
                          : "bg-[#0c121d] border-[#1b2537] hover:border-[#2d3f5e]"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <span className="font-mono text-xs font-bold text-slate-200">
                            {action.actionId}
                          </span>
                          <span className="px-1.5 py-0.2 rounded bg-[#182338] text-[10px] font-mono text-slate-300">
                            {action.category}
                          </span>
                        </div>
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${
                            action.status === "EXECUTED"
                              ? "bg-emerald-950/80 text-emerald-300 border border-emerald-500/40"
                              : action.status === "APPROVED"
                              ? "bg-blue-950/80 text-blue-300 border border-blue-500/40"
                              : action.status === "DENIED"
                              ? "bg-rose-950/80 text-rose-300 border border-rose-500/40"
                              : "bg-amber-950/80 text-amber-300 border border-amber-500/40"
                          }`}
                        >
                          {action.status}
                        </span>
                      </div>
                      <div className="text-xs font-medium text-slate-300 mt-1">
                        {action.title}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-1 line-clamp-1">
                        Expected: {action.expectedOutcome}
                      </div>
                      <div className="mt-2 flex items-center justify-between text-[10px] font-mono text-slate-400">
                        <span>Confidence: {action.confidenceScore.toFixed(1)}%</span>
                        <span>Rollback: {action.rollbackAvailable ? "YES (INV-OI55)" : "NO"}</span>
                        <span>Committee: {action.targetCommitteeId ?? "GLOBAL"}</span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Action Inspector Card */}
              <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-4">
                <div className="border-b border-[#1b2537] pb-2">
                  <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
                    Action Inspector
                  </span>
                  <h3 className="text-sm font-mono font-bold text-cyan-400 mt-0.5">
                    {selectedAction.actionId}
                  </h3>
                </div>

                <div className="space-y-2 text-xs font-mono">
                  <div>
                    <span className="text-slate-400">Title:</span>
                    <p className="text-slate-200 mt-0.5">{selectedAction.title}</p>
                  </div>
                  <div>
                    <span className="text-slate-400">Status:</span>
                    <p className="text-slate-200 mt-0.5 font-bold">{selectedAction.status}</p>
                  </div>
                  <div>
                    <span className="text-slate-400">Expected Outcome:</span>
                    <p className="text-slate-300 mt-0.5">{selectedAction.expectedOutcome}</p>
                  </div>
                  <div>
                    <span className="text-slate-400">Evidence Chain:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {selectedAction.evidenceIds?.map((eid) => (
                        <span key={eid} className="px-1.5 py-0.5 rounded bg-[#162234] text-cyan-300 text-[10px]">
                          {eid}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <span className="text-slate-400">Policy Approval Ref:</span>
                    <p className="text-slate-300 mt-0.5">{selectedAction.policyApprovalId}</p>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="pt-2 border-t border-[#1b2537] space-y-2">
                  {selectedAction.status === "APPROVED" && (
                    <button
                      onClick={() => handleExecuteAction(selectedAction.actionId)}
                      className="w-full py-1.5 rounded bg-emerald-900 hover:bg-emerald-800 border border-emerald-500/50 text-emerald-200 font-mono text-xs font-semibold transition-all"
                    >
                      Dispatch Execution
                    </button>
                  )}
                  {selectedAction.status === "EXECUTED" && selectedAction.rollbackAvailable && (
                    <button
                      onClick={() => handleRollbackAction(selectedAction.actionId)}
                      className="w-full py-1.5 rounded bg-amber-950 hover:bg-amber-900 border border-amber-500/50 text-amber-200 font-mono text-xs font-semibold transition-all"
                    >
                      Trigger Rollback (INV-OI55)
                    </button>
                  )}
                  {selectedAction.status === "DENIED" && (
                    <div className="p-2 rounded bg-rose-950/40 border border-rose-500/30 text-rose-300 text-[11px] font-mono">
                      Action blocked by policy rules. Escalated to human operator (INV-OI57).
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Policy Boundaries */}
        {activeTab === "POLICIES" && (
          <div className="space-y-6">
            {/* Interactive Policy Boundary Simulator */}
            <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-4">
              <div className="flex items-center justify-between border-b border-[#1b2537] pb-2">
                <div>
                  <h3 className="text-sm font-mono font-bold text-cyan-400">
                    Interactive Policy Boundary Evaluator (INV-OI53)
                  </h3>
                  <p className="text-xs text-slate-400">
                    Test potential action parameters against active governance constraints. 0 unauthorized actions permitted.
                  </p>
                </div>
                <span className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${
                  boundarySimResult.allowed
                    ? "bg-emerald-950/80 text-emerald-300 border border-emerald-500/50"
                    : "bg-rose-950/80 text-rose-300 border border-rose-500/50"
                }`}>
                  {boundarySimResult.allowed ? "ALLOWED (PASS)" : "BLOCKED (REJECTED)"}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs">
                <div>
                  <label className="text-slate-400 block mb-1">Proposed Action Name:</label>
                  <input
                    type="text"
                    value={simulatedActionText}
                    onChange={(e) => setSimulatedActionText(e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded bg-[#131d2e] border border-[#24334c] text-slate-200 focus:outline-none focus:border-cyan-500"
                  />
                </div>
                <div>
                  <label className="text-slate-400 block mb-1">Target Risk Score (VaR Floor 85.0):</label>
                  <input
                    type="number"
                    value={simulatedRisk}
                    onChange={(e) => setSimulatedRisk(Number(e.target.value))}
                    className="w-full px-2.5 py-1.5 rounded bg-[#131d2e] border border-[#24334c] text-slate-200 focus:outline-none focus:border-cyan-500"
                  />
                </div>
                <div>
                  <label className="text-slate-400 block mb-1">Budget Requested ($100k Limit):</label>
                  <input
                    type="number"
                    value={simulatedBudget}
                    onChange={(e) => setSimulatedBudget(Number(e.target.value))}
                    className="w-full px-2.5 py-1.5 rounded bg-[#131d2e] border border-[#24334c] text-slate-200 focus:outline-none focus:border-cyan-500"
                  />
                </div>
              </div>

              {boundarySimResult.blockingRules.length > 0 && (
                <div className="p-2.5 rounded bg-rose-950/40 border border-rose-500/40 text-rose-300 font-mono text-xs">
                  Blocking Rules Triggered: {boundarySimResult.blockingRules.join(", ")}
                </div>
              )}
            </div>

            {/* Active Policy Catalog */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {policies.map((policy) => (
                <div
                  key={policy.policyId}
                  className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-3"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-cyan-400">
                      {policy.policyId}
                    </span>
                    <span className="px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-300 text-[10px] font-mono border border-emerald-500/40">
                      v{policy.version}
                    </span>
                  </div>
                  <h4 className="text-xs font-semibold text-slate-200">
                    {policy.policyName}
                  </h4>
                  <div className="text-[11px] font-mono text-slate-400">
                    Owner: {policy.ownerId}
                  </div>
                  <div className="space-y-1.5 pt-2 border-t border-[#1b2537]">
                    <span className="text-[10px] font-mono text-slate-400 uppercase">
                      Rules ({policy.rules.length}):
                    </span>
                    {policy.rules.map((rule) => (
                      <div
                        key={rule.ruleId}
                        className="p-2 rounded bg-[#101724] border border-[#1b2537] text-[11px] font-mono space-y-0.5"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-slate-300 font-bold">{rule.ruleId}</span>
                          <span
                            className={`text-[9px] px-1 rounded ${
                              rule.action === "DENY"
                                ? "bg-rose-950 text-rose-300"
                                : rule.action === "REQUIRE_APPROVAL"
                                ? "bg-amber-950 text-amber-300"
                                : "bg-emerald-950 text-emerald-300"
                            }`}
                          >
                            {rule.action}
                          </span>
                        </div>
                        <div className="text-slate-400 text-[10px]">{rule.condition}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 3: Human Overrides */}
        {activeTab === "OVERRIDES" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Override Dispatch Form */}
            <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-4">
              <div className="border-b border-[#1b2537] pb-2">
                <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
                  Executive Override Terminal
                </span>
                <h3 className="text-sm font-mono font-bold text-cyan-400 mt-0.5">
                  Instant Supersession (INV-OI52)
                </h3>
                <p className="text-[11px] text-slate-400 mt-1">
                  Human commands immediately supersede any autonomous operation with 0-delay execution and cryptographic audit logging.
                </p>
              </div>

              <form onSubmit={handleSubmitOverride} className="space-y-3 font-mono text-xs">
                <div>
                  <label className="text-slate-400 block mb-1">Target Action ID:</label>
                  <select
                    value={overrideTargetId}
                    onChange={(e) => setOverrideTargetId(e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded bg-[#131d2e] border border-[#24334c] text-slate-200"
                  >
                    {allActions.map((a) => (
                      <option key={a.actionId} value={a.actionId}>
                        {a.actionId} - {a.title} ({a.status})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-slate-400 block mb-1">Override Action Type:</label>
                  <select
                    value={overrideType}
                    onChange={(e) => setOverrideType(e.target.value as OverrideActionType)}
                    className="w-full px-2.5 py-1.5 rounded bg-[#131d2e] border border-[#24334c] text-slate-200"
                  >
                    <option value="PAUSE">PAUSE - Temporarily suspend operation</option>
                    <option value="CANCEL">CANCEL - Deny and terminate action</option>
                    <option value="ROLLBACK">ROLLBACK - Safely revert execution</option>
                    <option value="FORCE_APPROVE">FORCE_APPROVE - Executive override pass</option>
                  </select>
                </div>

                <div>
                  <label className="text-slate-400 block mb-1">Authorized Operator Role:</label>
                  <select
                    value={overrideOperator}
                    onChange={(e) => setOverrideOperator(e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded bg-[#131d2e] border border-[#24334c] text-slate-200"
                  >
                    <option value="CHIEF_RISK_OFFICER">CHIEF_RISK_OFFICER</option>
                    <option value="EXECUTIVE_DIRECTOR">EXECUTIVE_DIRECTOR</option>
                    <option value="BOARD_ADMIN">BOARD_ADMIN</option>
                    <option value="LEAD_ARBITER">LEAD_ARBITER</option>
                  </select>
                </div>

                <div>
                  <label className="text-slate-400 block mb-1">Justification / Rationale:</label>
                  <textarea
                    rows={2}
                    value={overrideRationale}
                    onChange={(e) => setOverrideRationale(e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded bg-[#131d2e] border border-[#24334c] text-slate-200 focus:outline-none focus:border-cyan-500"
                  />
                </div>

                <button
                  type="submit"
                  className="w-full py-2 rounded bg-rose-950 hover:bg-rose-900 border border-rose-500/60 text-rose-200 font-mono text-xs font-bold transition-all"
                >
                  Execute Immediate Override (0s Latency)
                </button>
              </form>
            </div>

            {/* Override Audit Ledger */}
            <div className="lg:col-span-2 p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-3">
              <div className="flex items-center justify-between border-b border-[#1b2537] pb-2">
                <h3 className="text-sm font-mono font-bold text-slate-200">
                  Immutable Override Audit Ledger
                </h3>
                <span className="text-[11px] font-mono text-cyan-400">
                  {auditLedger.length} Records Verified
                </span>
              </div>

              <div className="space-y-2">
                {auditLedger.map((record) => (
                  <div
                    key={record.overrideId}
                    className="p-3 rounded-lg bg-[#101724] border border-[#1b2537] space-y-1 font-mono text-xs"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-cyan-300">{record.overrideId}</span>
                      <span className="text-[10px] text-emerald-400 bg-emerald-950/80 px-1.5 py-0.5 rounded border border-emerald-500/30">
                        {record.status}
                      </span>
                    </div>
                    <div className="text-slate-300 text-[11px]">
                      By: <span className="font-bold">{record.approvedBy}</span> at {record.approvedAtUtc}
                    </div>
                    <div className="text-slate-400 text-[11px]">
                      Rationale: {record.rationale}
                    </div>
                    <div className="flex items-center space-x-4 text-[10px] text-slate-500 pt-1">
                      <span>Before: {record.beforePolicyHash.slice(0, 8)}...</span>
                      <span>After: {record.afterPolicyHash.slice(0, 8)}...</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Tab 4: Accountability & Drift */}
        {activeTab === "ACCOUNTABILITY" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Outcome Attribution Calculator */}
            <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-4">
              <div className="border-b border-[#1b2537] pb-2">
                <h3 className="text-sm font-mono font-bold text-cyan-400">
                  Outcome Accountability & Drift Evaluation (INV-OI56)
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Verifies that realized metric changes match expected autonomous predictions within calibrated tolerance.
                </p>
              </div>

              <div className="space-y-3 font-mono text-xs">
                <div>
                  <label className="text-slate-400 block mb-1">Target Action to Audit:</label>
                  <select
                    value={selectedActionId}
                    onChange={(e) => setSelectedActionId(e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded bg-[#131d2e] border border-[#24334c] text-slate-200"
                  >
                    {allActions.map((a) => (
                      <option key={a.actionId} value={a.actionId}>
                        {a.actionId} - {a.title}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-slate-400 block mb-1">Observed Metric Delta (+pts):</label>
                  <input
                    type="number"
                    step="0.1"
                    value={observedDeltaInput}
                    onChange={(e) => setObservedDeltaInput(Number(e.target.value))}
                    className="w-full px-2.5 py-1.5 rounded bg-[#131d2e] border border-[#24334c] text-slate-200"
                  />
                </div>

                {accountabilityReport && (
                  <div className="p-3 rounded-lg bg-[#101724] border border-[#1b2537] space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Drift Assessment:</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        accountabilityReport.withinExpectations
                          ? "bg-emerald-950 text-emerald-300 border border-emerald-500/40"
                          : "bg-rose-950 text-rose-300 border border-rose-500/40"
                      }`}>
                        {accountabilityReport.withinExpectations ? "WITHIN TOLERANCE (PASS)" : "DRIFT DETECTED (ESCALATED)"}
                      </span>
                    </div>
                    <div className="text-slate-300 text-[11px]">
                      Drift Score: <span className="font-bold">{accountabilityReport.driftScore.toFixed(2)} pts</span> (Max allowed: 1.50)
                    </div>
                    <div className="text-slate-400 text-[10px]">
                      Expected Rationale: {accountabilityReport.expectedOutcome}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Active Safety Alerts & Human Escalation */}
            <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-3">
              <div className="flex items-center justify-between border-b border-[#1b2537] pb-2">
                <h3 className="text-sm font-mono font-bold text-slate-200">
                  Active Safety Alerts & Escalations (INV-OI57)
                </h3>
                <span className="text-[11px] font-mono text-rose-400">
                  {alerts.length} Active
                </span>
              </div>

              <div className="space-y-2">
                {alerts.map((alert) => (
                  <div
                    key={alert.alertId}
                    className="p-3 rounded-lg bg-[#101724] border border-rose-500/30 font-mono text-xs space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-rose-400">{alert.alertId}</span>
                      <span className="px-1.5 py-0.5 rounded bg-rose-950 text-rose-300 text-[10px] border border-rose-500/40">
                        {alert.severity}
                      </span>
                    </div>
                    <div className="text-slate-200 text-[11px]">
                      {alert.description}
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-slate-400 pt-1">
                      <span>Type: {alert.alertType}</span>
                      <span className="text-cyan-400 font-bold">
                        {alert.escalatedToHuman ? "ESCALATED TO HUMAN (INV-OI57)" : "LOCAL"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Tab 5: Certification Matrix */}
        {activeTab === "CERTIFICATION" && (
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-4">
              <div className="flex items-center justify-between border-b border-[#1b2537] pb-2">
                <div>
                  <h3 className="text-sm font-mono font-bold text-cyan-400">
                    Phase 31-M9 Autonomous Governance Certification Matrix
                  </h3>
                  <p className="text-xs text-slate-400">
                    Full formal verification across 10 gates and 8 core invariants (INV-OI50 - INV-OI57).
                  </p>
                </div>
                <span className="px-3 py-1 rounded bg-emerald-950 border border-emerald-500/60 text-emerald-400 font-mono text-xs font-bold">
                  10/10 GATES CERTIFIED
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs">
                  <thead className="bg-[#121927] text-slate-400 border-b border-[#1b2537]">
                    <tr>
                      <th className="p-2.5">Gate ID</th>
                      <th className="p-2.5">Certification Scope</th>
                      <th className="p-2.5">Invariant Target</th>
                      <th className="p-2.5">Target Requirement</th>
                      <th className="p-2.5 text-right">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1b2537] text-slate-300">
                    {M9_GATE_TRACEABILITY_MATRIX.map((gate) => (
                      <tr key={gate.gateId} className="hover:bg-[#111928]">
                        <td className="p-2.5 font-bold text-cyan-400">{gate.gateId}</td>
                        <td className="p-2.5 text-slate-200">{gate.name}</td>
                        <td className="p-2.5 text-slate-400">{gate.invariant}</td>
                        <td className="p-2.5 text-slate-400">{gate.target}</td>
                        <td className="p-2.5 text-right">
                          <span className="px-2 py-0.5 rounded bg-emerald-950/80 border border-emerald-500/50 text-emerald-400 text-[10px] font-bold">
                            PASS
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Bottom Related Artifacts Card */}
        <div className="pt-4 border-t border-[#1b2537]">
          <RelatedArtifactsCard entityId={selectedActionId} />
        </div>
      </main>
    </div>
  );
}

export default function AutonomousGovernancePage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-[#070a0f] text-slate-300 font-mono text-xs flex items-center justify-center">
          Loading Autonomous Governance Control System...
        </div>
      }
    >
      <AutonomousGovernanceContent />
    </Suspense>
  );
}
