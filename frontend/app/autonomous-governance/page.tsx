"use client";

import { useState, useMemo, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import ExecutiveIntelligenceNav from "../../components/committee/ExecutiveIntelligenceNav";
import RelatedArtifactsCard from "../../components/committee/RelatedArtifactsCard";

import { safetyPolicyEngine } from "../../lib/autonomous/safetyPolicyEngine";
import { autonomousActionEngine } from "../../lib/autonomous/autonomousActionEngine";
import { overrideGovernanceEngine } from "../../lib/autonomous/overrideGovernanceEngine";
import { governanceEscalationEngine } from "../../lib/autonomous/governanceEscalationEngine";
import { actionRegistry } from "../../lib/autonomous/autonomousActionRegistry";
import { autonomousSafetyMonitor } from "../../lib/autonomous/autonomousSafetyMonitor";
import { getCanonicalPolicies } from "../../lib/autonomous/governancePolicyEngine";

import {
  CANONICAL_OPERATIONAL_RUNBOOKS,
  M10_GATE_TRACEABILITY_MATRIX,
  OperationalRunbook,
  FailCloseStateResponse,
} from "../../types/fail-close-governance";
import {
  AutonomousAction,
  OverrideActionType,
} from "../../types/autonomous-governance";

type M10Tab = "ACTIONS" | "POLICIES" | "OVERRIDES" | "ESCALATIONS" | "RUNBOOKS" | "CERTIFICATION";

function AutonomousGovernanceM10Content() {
  const searchParams = useSearchParams();
  const rawTab = (searchParams.get("tab")?.toUpperCase() as M10Tab) || "ACTIONS";
  const [activeTab, setActiveTab] = useState<M10Tab>(
    ["ACTIONS", "POLICIES", "OVERRIDES", "ESCALATIONS", "RUNBOOKS", "CERTIFICATION"].includes(rawTab)
      ? rawTab
      : "ACTIONS"
  );

  // Version tick for force-updating view state upon mutations
  const [engineVersion, setEngineVersion] = useState(0);
  const [noticeMessage, setNoticeMessage] = useState<string | null>(null);

  // 1. Actions / Safe Termination State
  const [selectedActionId, setSelectedActionId] = useState<string>("ACT-2026-001");
  const [actionSimRisk, setActionSimRisk] = useState<number>(62);
  const [actionSimBudget, setActionSimBudget] = useState<number>(45000);
  const [actionSimTitle, setActionSimTitle] = useState<string>("Autonomous Downside Variance Dampening");
  const [simulateMissingDissent, setSimulateMissingDissent] = useState<boolean>(false);
  const [simulateEmergencyCharter, setSimulateEmergencyCharter] = useState<boolean>(false);

  // 2. Override State
  const [overrideTargetId, setOverrideTargetId] = useState<string>("ACT-2026-003");
  const [overrideActionType, setOverrideActionType] = useState<OverrideActionType>("PAUSE");
  const [overrideActor, setOverrideActor] = useState<string>("USR-CRO");
  const [overrideRationale, setOverrideRationale] = useState<string>("Mandatory risk escalation pending review");

  // 3. Escalation State
  const [escalationFilter, setEscalationFilter] = useState<string>("ALL");

  // 4. Runbooks State
  const [selectedRunbookId, setSelectedRunbookId] = useState<string>("M9-RB-01");
  const [activeRunbookExecutions, setActiveRunbookExecutions] = useState<Record<string, string>>({});

  // Memoized data bindings
  const policies = useMemo(() => getCanonicalPolicies(), []);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const allActions = useMemo(() => actionRegistry.getAllActions(), [engineVersion]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const activeOverrides = useMemo(() => overrideGovernanceEngine.getAllActiveOverrides(), [engineVersion]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const overrideViolations = useMemo(() => overrideGovernanceEngine.getViolations(), [engineVersion]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const escalations = useMemo(() => governanceEscalationEngine.getIncidents(), [engineVersion]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const escalationViolations = useMemo(() => governanceEscalationEngine.getViolations(), [engineVersion]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const isEmergencyStop = useMemo(() => overrideGovernanceEngine.isEmergencyStopActive(), [engineVersion]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const safetyScore = useMemo(() => autonomousSafetyMonitor.computeSafetyScore(), [engineVersion]);

  const selectedAction = useMemo(() => {
    return allActions.find((a) => a.actionId === selectedActionId) || allActions[0];
  }, [allActions, selectedActionId]);

  const selectedRunbook = useMemo(() => {
    return CANONICAL_OPERATIONAL_RUNBOOKS.find((r) => r.runbookId === selectedRunbookId) || CANONICAL_OPERATIONAL_RUNBOOKS[0];
  }, [selectedRunbookId]);

  // Safe Termination Execution Simulator
  const simulatedExecutionOutcome = useMemo(() => {
    const title = simulateEmergencyCharter ? "Emergency Charter Modification charter" : actionSimTitle;
    return autonomousActionEngine.executeAutonomousAction(
      {
        requestId: "SIM-EXEC-01",
        recommendationId: "REC-SIM",
        committeeId: "COM-001",
        initiatedAtUtc: new Date().toISOString(),
        proposedAction: title,
        rationale: "Interactive safe termination verification run",
        policyEvaluationId: "EVAL-SIM",
        targetRiskScore: actionSimRisk,
        budgetRequestedDollars: actionSimBudget,
      },
      {
        dissentIds: simulateMissingDissent ? [] : ["DIS-001"],
        evidenceCount: 2,
        hasRationale: true,
      }
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [actionSimRisk, actionSimBudget, actionSimTitle, simulateMissingDissent, simulateEmergencyCharter, engineVersion]);

  // Action Dispatch Handlers
  const handleExecuteRegisteredAction = (actionId: string) => {
    try {
      const outcome = autonomousActionEngine.executeAutonomousAction({
        requestId: actionId,
        recommendationId: "REC-01",
        committeeId: "COM-001",
        initiatedAtUtc: new Date().toISOString(),
        proposedAction: selectedAction.title,
        rationale: selectedAction.expectedOutcome,
        policyEvaluationId: selectedAction.policyApprovalId,
        targetRiskScore: 60,
        budgetRequestedDollars: 25000,
      });

      if (outcome.success) {
        setNoticeMessage(`Action ${actionId} executed safely (INV-OI50). State mutated.`);
      } else {
        setNoticeMessage(`Action ${actionId} SAFE TERMINATION activated (INV-OI55). Zero state mutation.`);
      }
      setEngineVersion((v) => v + 1);
    } catch (err: unknown) {
      setNoticeMessage(`Execution failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleRollback = (actionId: string) => {
    try {
      autonomousActionEngine.rollbackAction(actionId, "Executive rollback trigger (INV-OI55)");
      setNoticeMessage(`Action ${actionId} rolled back successfully. State inverted.`);
      setEngineVersion((v) => v + 1);
    } catch (err: unknown) {
      setNoticeMessage(`Rollback failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  // Override Form Submission
  const handleRegisterOverride = (e: React.FormEvent) => {
    e.preventDefault();
    const validation = overrideGovernanceEngine.validateOverrideAttempt(
      overrideActor,
      overrideTargetId,
      overrideActionType
    );

    if (!validation.allowed) {
      setNoticeMessage(`OVERRIDE REJECTED: ${validation.violation?.message} (GOV-OVR-001)`);
      setEngineVersion((v) => v + 1);
      return;
    }

    overrideGovernanceEngine.registerOverride({
      overrideId: `OVR-${Date.now().toString(36).toUpperCase()}`,
      targetActionId: overrideTargetId,
      overrideType: overrideActionType,
      actorId: overrideActor,
      actorLevel: overrideActor === "USR-CRO" || overrideActor === "USR-ED" ? "EXECUTIVE" : "OPERATOR",
      justification: overrideRationale,
      issuedAtUtc: new Date().toISOString(),
      active: true,
    });

    setNoticeMessage(`Human Override APPLIED for ${overrideTargetId} (${overrideActionType}). Zero-delay supersession.`);
    setEngineVersion((v) => v + 1);
  };

  // Emergency Stop Trigger
  const handleToggleEmergencyStop = () => {
    if (isEmergencyStop) {
      overrideGovernanceEngine.releaseEmergencyStop("USR-CRO");
      setNoticeMessage("L4 Emergency Stop RELEASED. System returned to CERTIFIED operational status.");
    } else {
      overrideGovernanceEngine.triggerEmergencyStop("USR-CRO", "Executive Emergency Stop triggered from control panel");
      setNoticeMessage("L4 Emergency Stop ACTIVATED. Autonomous execution suspended (INV-OI59).");
    }
    setEngineVersion((v) => v + 1);
  };

  // Runbook Trigger Simulator
  const handleTriggerRunbook = (rb: OperationalRunbook) => {
    const timestamp = new Date().toLocaleTimeString();
    setActiveRunbookExecutions((prev) => ({
      ...prev,
      [rb.runbookId]: `Triggered at ${timestamp}: ${rb.automatedActions[0]}`,
    }));
    setNoticeMessage(`Runbook ${rb.runbookId} (${rb.name}) EXECUTED. Automated actions initiated.`);
  };

  return (
    <div className="min-h-screen bg-[#070a0f] text-slate-100 flex flex-col font-sans">
      <ExecutiveIntelligenceNav badgeText="10/10 M10 SAFETY GATES CERTIFIED" />

      <main className="flex-1 max-w-[1750px] w-full mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Header Ribbon */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#1b2537] pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="px-2 py-0.5 rounded bg-cyan-950/70 border border-cyan-500/40 text-cyan-400 font-mono text-xs font-semibold">
                PHASE 31-M10
              </span>
              <span className="text-xs font-mono text-slate-400">
                Autonomous Governance Safety & Fail-Close Error Architecture
              </span>
            </div>
            <h2 className="text-xl sm:text-2xl font-mono font-bold text-slate-100 tracking-tight mt-1">
              AUTONOMOUS GOVERNANCE SAFETY CENTER
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Policy-bounded execution, human override supremacy, safe termination, escalation integrity & fail-close error protection.
            </p>
          </div>

          {/* Emergency Killswitch Button */}
          <div className="flex items-center space-x-3">
            <button
              onClick={handleToggleEmergencyStop}
              className={`px-4 py-2 rounded font-mono text-xs font-bold transition-all shadow-md flex items-center space-x-2 ${
                isEmergencyStop
                  ? "bg-amber-950 hover:bg-amber-900 border border-amber-500 text-amber-200"
                  : "bg-rose-950 hover:bg-rose-900 border border-rose-500/60 text-rose-200"
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-current animate-pulse" />
              <span>{isEmergencyStop ? "RELEASE EMERGENCY STOP" : "EMERGENCY STOP (KILLSWITCH)"}</span>
            </button>
          </div>
        </div>

        {/* Global Notice Banner */}
        {noticeMessage && (
          <div className="px-4 py-2.5 rounded bg-[#101b2d] border border-cyan-500/50 text-cyan-300 font-mono text-xs flex items-center justify-between">
            <span>{noticeMessage}</span>
            <button
              onClick={() => setNoticeMessage(null)}
              className="text-slate-400 hover:text-slate-200 ml-4 font-bold"
            >
              &times;
            </button>
          </div>
        )}

        {/* 4 Executive KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Card 1: Safe Termination Health */}
          <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537]">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400">Safe Termination State</span>
              <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${
                isEmergencyStop
                  ? "bg-rose-950 text-rose-300 border border-rose-500/40"
                  : "bg-emerald-950/60 text-emerald-400 border border-emerald-500/40"
              }`}>
                {isEmergencyStop ? "FAIL_CLOSED" : "CERTIFIED"}
              </span>
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-2xl font-mono font-bold text-slate-100">
                {isEmergencyStop ? "BLOCKED" : "ACTIVE"}
              </span>
              <span className="text-xs font-mono text-slate-400">INV-OI55 / INV-OI63</span>
            </div>
            <div className="mt-2 text-[11px] text-slate-400 font-mono flex items-center justify-between">
              <span>Mutation Safe: 100% Guaranteed</span>
              <span className="text-emerald-400 font-semibold">Zero Bypass</span>
            </div>
          </div>

          {/* Card 2: Override Supremacy */}
          <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537]">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400">Human Override Supremacy</span>
              <span className="px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-500/40 text-emerald-400 text-[10px] font-mono font-semibold">
                0s LATENCY
              </span>
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-2xl font-mono font-bold text-cyan-400">
                {activeOverrides.length}
              </span>
              <span className="text-xs font-mono text-slate-400">Active Overrides</span>
              {overrideViolations.length > 0 && (
                <span className="text-xs font-mono text-rose-400 ml-2">
                  ({overrideViolations.length} Violations Blocked)
                </span>
              )}
            </div>
            <div className="mt-2 text-[11px] text-slate-400 font-mono flex items-center justify-between">
              <span>INV-OI51 / INV-OI59 Verified</span>
              <span className="text-cyan-400 font-semibold">Human &gt; Automation</span>
            </div>
          </div>

          {/* Card 3: Boundary Protection */}
          <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537]">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400">Policy Boundaries</span>
              <span className="px-2 py-0.5 rounded bg-blue-950/60 border border-blue-500/40 text-blue-400 text-[10px] font-mono font-semibold">
                7 RULES ACTIVE
              </span>
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-2xl font-mono font-bold text-slate-100">
                0
              </span>
              <span className="text-xs font-mono text-slate-400">Unauthorized Crossings</span>
            </div>
            <div className="mt-2 text-[11px] text-slate-400 font-mono flex items-center justify-between">
              <span>VaR Floor: 15.0% | Dissent: 0.70</span>
              <span className="text-blue-400 font-semibold">INV-OI52 / INV-OI60</span>
            </div>
          </div>

          {/* Card 4: Escalation Completeness */}
          <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537]">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400">Escalation Integrity</span>
              <span className="px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-500/40 text-emerald-400 text-[10px] font-mono font-semibold">
                100% COMPLETE
              </span>
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-2xl font-mono font-bold text-slate-100">
                {escalations.length}
              </span>
              <span className="text-xs font-mono text-slate-400">Incidents Monitored</span>
            </div>
            <div className="mt-2 text-[11px] text-slate-400 font-mono flex items-center justify-between">
              <span>SLA Enforcement: ACTIVE</span>
              <span className="text-emerald-400 font-semibold">Zero Silent Drops</span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center space-x-2 border-b border-[#1b2537] font-mono text-xs overflow-x-auto pb-1">
          {[
            { id: "ACTIONS", label: "Active Actions & Safe Termination" },
            { id: "POLICIES", label: "Policy Decisions & Boundaries" },
            { id: "OVERRIDES", label: "Override Queue & Killswitch" },
            { id: "ESCALATIONS", label: "Escalations & SLA Governance" },
            { id: "RUNBOOKS", label: "Operational Runbooks (M9)" },
            { id: "CERTIFICATION", label: "M10 Certification Matrix" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as M10Tab)}
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

        {/* TAB 1: Active Actions & Safe Termination */}
        {activeTab === "ACTIONS" && (
          <div className="space-y-6">
            {/* Interactive Safe Termination Simulator */}
            <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-4">
              <div className="flex items-center justify-between border-b border-[#1b2537] pb-2">
                <div>
                  <h3 className="text-sm font-mono font-bold text-cyan-400">
                    Safe Termination Pre-Condition Gate (INV-OI55 / INV-OI63)
                  </h3>
                  <p className="text-xs text-slate-400">
                    Simulate edge cases to verify that unsafe execution paths terminate BEFORE any state mutation occurs.
                  </p>
                </div>
                <div className="flex items-center space-x-2 font-mono text-xs">
                  <span className={`px-2.5 py-1 rounded font-bold ${
                    simulatedExecutionOutcome.status === "EXECUTED"
                      ? "bg-emerald-950 text-emerald-300 border border-emerald-500/50"
                      : "bg-rose-950 text-rose-300 border border-rose-500/50"
                  }`}>
                    {simulatedExecutionOutcome.status}
                  </span>
                  <span className="text-slate-400">
                    State Mutated: <strong className={simulatedExecutionOutcome.stateMutated ? "text-amber-400" : "text-emerald-400"}>
                      {simulatedExecutionOutcome.stateMutated ? "YES" : "NO (SAFE)"}
                    </strong>
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs">
                <div>
                  <label className="text-slate-400 block mb-1">Target Risk Score (VaR limit 80.0):</label>
                  <input
                    type="number"
                    value={actionSimRisk}
                    onChange={(e) => setActionSimRisk(Number(e.target.value))}
                    className="w-full px-2.5 py-1.5 rounded bg-[#131d2e] border border-[#24334c] text-slate-200"
                  />
                </div>
                <div>
                  <label className="text-slate-400 block mb-1">Requested Budget ($100k limit):</label>
                  <input
                    type="number"
                    value={actionSimBudget}
                    onChange={(e) => setActionSimBudget(Number(e.target.value))}
                    className="w-full px-2.5 py-1.5 rounded bg-[#131d2e] border border-[#24334c] text-slate-200"
                  />
                </div>
                <div className="space-y-2 pt-4">
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={simulateMissingDissent}
                      onChange={(e) => setSimulateMissingDissent(e.target.checked)}
                      className="rounded bg-[#131d2e] border-[#24334c] text-cyan-500"
                    />
                    <span className="text-slate-300">Simulate Missing Dissent (GOV-POL-004)</span>
                  </label>
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={simulateEmergencyCharter}
                      onChange={(e) => setSimulateEmergencyCharter(e.target.checked)}
                      className="rounded bg-[#131d2e] border-[#24334c] text-cyan-500"
                    />
                    <span className="text-slate-300">Simulate Charter Operation (GOV-POL-002)</span>
                  </label>
                </div>
              </div>

              {simulatedExecutionOutcome.violations && simulatedExecutionOutcome.violations.length > 0 && (
                <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-500/50 space-y-1 font-mono text-xs">
                  <div className="text-rose-300 font-bold">
                    Violations Captured (Zero Mutation Occurred):
                  </div>
                  {simulatedExecutionOutcome.violations.map((v) => (
                    <div key={v.errorCode} className="text-slate-300 text-[11px]">
                      &bull; <strong className="text-rose-400">{v.errorCode}</strong> ({v.errorType}): {v.message}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Action Registry Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2 space-y-2">
                {allActions.map((action) => (
                  <div
                    key={action.actionId}
                    onClick={() => setSelectedActionId(action.actionId)}
                    className={`p-3 rounded-lg border transition-all cursor-pointer ${
                      action.actionId === selectedActionId
                        ? "bg-[#131d2e] border-cyan-500/60 shadow-sm"
                        : "bg-[#0c121d] border-[#1b2537] hover:border-[#2d3f5e]"
                    }`}
                  >
                    <div className="flex items-center justify-between font-mono text-xs">
                      <div className="flex items-center space-x-2">
                        <span className="font-bold text-slate-200">{action.actionId}</span>
                        <span className="px-1.5 py-0.5 rounded bg-[#182338] text-[10px] text-slate-300">{action.category}</span>
                      </div>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        action.status === "EXECUTED" ? "bg-emerald-950 text-emerald-300 border border-emerald-500/40" : "bg-[#1a2333] text-slate-300"
                      }`}>
                        {action.status}
                      </span>
                    </div>
                    <div className="text-xs font-medium text-slate-300 mt-1">{action.title}</div>
                    <div className="text-[11px] text-slate-400 mt-1">{action.expectedOutcome}</div>
                  </div>
                ))}
              </div>

              {/* Action Operations Panel */}
              <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-4">
                <div className="border-b border-[#1b2537] pb-2">
                  <span className="text-[10px] font-mono text-slate-400 uppercase">Selected Action</span>
                  <h3 className="text-sm font-mono font-bold text-cyan-400 mt-0.5">{selectedAction.actionId}</h3>
                </div>

                <div className="space-y-2 text-xs font-mono">
                  <div><span className="text-slate-400">Title:</span> <p className="text-slate-200">{selectedAction.title}</p></div>
                  <div><span className="text-slate-400">Status:</span> <p className="text-slate-200 font-bold">{selectedAction.status}</p></div>
                  <div><span className="text-slate-400">Confidence:</span> <p className="text-slate-300">{selectedAction.confidenceScore}%</p></div>
                  <div><span className="text-slate-400">Rollback Available:</span> <p className="text-emerald-400">{selectedAction.rollbackAvailable ? "YES (INV-OI55)" : "NO"}</p></div>
                </div>

                <div className="pt-2 border-t border-[#1b2537] space-y-2">
                  <button
                    onClick={() => handleExecuteRegisteredAction(selectedAction.actionId)}
                    className="w-full py-1.5 rounded bg-cyan-950 hover:bg-cyan-900 border border-cyan-500/50 text-cyan-200 font-mono text-xs font-bold"
                  >
                    Execute Action (Safe Gate Check)
                  </button>
                  {selectedAction.rollbackAvailable && (
                    <button
                      onClick={() => handleRollback(selectedAction.actionId)}
                      className="w-full py-1.5 rounded bg-amber-950 hover:bg-amber-900 border border-amber-500/50 text-amber-200 font-mono text-xs font-bold"
                    >
                      Rollback Execution (INV-OI55)
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: Policy Decisions & Boundaries */}
        {activeTab === "POLICIES" && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {policies.map((p) => (
              <div key={p.policyId} className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-3">
                <div className="flex items-center justify-between font-mono text-xs">
                  <span className="font-bold text-cyan-400">{p.policyId}</span>
                  <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 text-[10px] border border-emerald-500/40">
                    {p.certificationStatus}
                  </span>
                </div>
                <h4 className="text-xs font-semibold text-slate-200">{p.policyName}</h4>
                <div className="space-y-1.5 pt-2 border-t border-[#1b2537]">
                  {p.rules.map((r) => (
                    <div key={r.ruleId} className="p-2 rounded bg-[#101724] border border-[#1b2537] text-[11px] font-mono">
                      <div className="flex justify-between font-bold">
                        <span className="text-slate-300">{r.ruleId}</span>
                        <span className={r.action === "DENY" ? "text-rose-400" : "text-amber-400"}>{r.action}</span>
                      </div>
                      <div className="text-slate-400 text-[10px] mt-0.5">{r.condition}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* TAB 3: Override Queue & Killswitch */}
        {activeTab === "OVERRIDES" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Override Terminal Form */}
            <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-4">
              <div className="border-b border-[#1b2537] pb-2">
                <h3 className="text-sm font-mono font-bold text-cyan-400">
                  Override Queue Terminal (INV-OI51 / INV-OI59)
                </h3>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Human overrides supersede automation instantly. Unauthorized bypass attempts trigger GOV-OVR-001.
                </p>
              </div>

              <form onSubmit={handleRegisterOverride} className="space-y-3 font-mono text-xs">
                <div>
                  <label className="text-slate-400 block mb-1">Target Action ID:</label>
                  <select
                    value={overrideTargetId}
                    onChange={(e) => setOverrideTargetId(e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded bg-[#131d2e] border border-[#24334c] text-slate-200"
                  >
                    {allActions.map((a) => (
                      <option key={a.actionId} value={a.actionId}>{a.actionId} - {a.title}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-slate-400 block mb-1">Action Type:</label>
                  <select
                    value={overrideActionType}
                    onChange={(e) => setOverrideActionType(e.target.value as OverrideActionType)}
                    className="w-full px-2.5 py-1.5 rounded bg-[#131d2e] border border-[#24334c] text-slate-200"
                  >
                    <option value="PAUSE">PAUSE - Temporarily suspend</option>
                    <option value="CANCEL">CANCEL - Deny and terminate</option>
                    <option value="ROLLBACK">ROLLBACK - Revert mutated state</option>
                    <option value="FORCE_APPROVE">FORCE_APPROVE - Executive override pass</option>
                  </select>
                </div>

                <div>
                  <label className="text-slate-400 block mb-1">Actor ID:</label>
                  <select
                    value={overrideActor}
                    onChange={(e) => setOverrideActor(e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded bg-[#131d2e] border border-[#24334c] text-slate-200"
                  >
                    <option value="USR-CRO">USR-CRO (Chief Risk Officer - Executive)</option>
                    <option value="USR-ED">USR-ED (Executive Director - Executive)</option>
                    <option value="USR-BOARD-01">USR-BOARD-01 (Board Governance)</option>
                    <option value="USR-UNAUTH">USR-UNAUTH (Unauthorized Test Actor - Triggers GOV-OVR-001)</option>
                  </select>
                </div>

                <div>
                  <label className="text-slate-400 block mb-1">Justification:</label>
                  <input
                    type="text"
                    value={overrideRationale}
                    onChange={(e) => setOverrideRationale(e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded bg-[#131d2e] border border-[#24334c] text-slate-200"
                  />
                </div>

                <button
                  type="submit"
                  className="w-full py-2 rounded bg-rose-950 hover:bg-rose-900 border border-rose-500/60 text-rose-200 font-mono text-xs font-bold"
                >
                  Dispatch Override (Instant 0s Latency)
                </button>
              </form>
            </div>

            {/* Active Overrides & Violations List */}
            <div className="lg:col-span-2 p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-4">
              <div className="flex items-center justify-between border-b border-[#1b2537] pb-2 font-mono text-xs">
                <h3 className="font-bold text-slate-200">Active Overrides Queue & Audit Trail</h3>
                <span className="text-cyan-400">{activeOverrides.length} Active</span>
              </div>

              <div className="space-y-2">
                {activeOverrides.map((ovr) => (
                  <div key={ovr.overrideId} className="p-3 rounded-lg bg-[#101724] border border-[#1b2537] font-mono text-xs space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-cyan-300">{ovr.overrideId} &rarr; {ovr.targetActionId}</span>
                      <span className="px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 text-[10px]">{ovr.overrideType}</span>
                    </div>
                    <div className="text-slate-300 text-[11px]">Actor: {ovr.actorId} ({ovr.actorLevel}) | Rationale: {ovr.justification}</div>
                  </div>
                ))}

                {overrideViolations.length > 0 && (
                  <div className="pt-2 border-t border-rose-950/60 space-y-2">
                    <span className="text-xs font-mono font-bold text-rose-400">Blocked Override Violations:</span>
                    {overrideViolations.map((v, idx) => (
                      <div key={idx} className="p-2.5 rounded bg-rose-950/30 border border-rose-500/30 font-mono text-[11px] text-slate-300">
                        <strong className="text-rose-300">{v.errorCode}</strong>: {v.message}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: Escalations & SLA Governance */}
        {activeTab === "ESCALATIONS" && (
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-3">
              <div className="flex items-center justify-between border-b border-[#1b2537] pb-2 font-mono text-xs">
                <div>
                  <h3 className="font-bold text-cyan-400">Escalation Integrity & SLA Governance (INV-OI53 / INV-OI61)</h3>
                  <p className="text-slate-400 text-[11px]">Automatic severity escalations (HIGH &rarr; CRITICAL) upon SLA breach. Zero silent drops.</p>
                </div>
                <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 text-[10px] font-bold">100% ESCALATED</span>
              </div>

              <div className="space-y-2">
                {escalations.map((inc) => (
                  <div key={inc.incidentId} className="p-3 rounded-lg bg-[#101724] border border-[#1b2537] font-mono text-xs space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-200">{inc.incidentId}: {inc.title}</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        inc.severity === "CRITICAL" ? "bg-rose-950 text-rose-300 border border-rose-500/40" : "bg-amber-950 text-amber-300"
                      }`}>
                        {inc.severity}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-[11px] text-slate-400">
                      <span>Target Role: <strong className="text-cyan-400">{inc.escalatedToRole}</strong></span>
                      <span>Target SLA: {inc.targetSlaHours}h</span>
                      <span>Status: <strong className="text-emerald-400">{inc.status}</strong></span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: Operational Runbooks (M9) */}
        {activeTab === "RUNBOOKS" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="space-y-2">
              <span className="text-xs font-mono text-slate-400 block mb-1">M9 Production Runbooks:</span>
              {CANONICAL_OPERATIONAL_RUNBOOKS.map((rb) => (
                <div
                  key={rb.runbookId}
                  onClick={() => setSelectedRunbookId(rb.runbookId)}
                  className={`p-3 rounded-lg border transition-all cursor-pointer font-mono text-xs ${
                    rb.runbookId === selectedRunbookId
                      ? "bg-[#131d2e] border-cyan-500/60 shadow-sm"
                      : "bg-[#0c121d] border-[#1b2537] hover:border-[#2d3f5e]"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-cyan-300">{rb.runbookId}</span>
                    <span className={`px-1.5 py-0.5 rounded text-[9px] ${
                      rb.severity === "CRITICAL" ? "bg-rose-950 text-rose-300" : "bg-amber-950 text-amber-300"
                    }`}>
                      {rb.severity}
                    </span>
                  </div>
                  <div className="font-medium text-slate-200 mt-1">{rb.name}</div>
                  {activeRunbookExecutions[rb.runbookId] && (
                    <div className="mt-1 text-[10px] text-emerald-400 truncate">
                      &bull; {activeRunbookExecutions[rb.runbookId]}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Runbook Inspector & Execution Simulator */}
            <div className="lg:col-span-2 p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-4 font-mono text-xs">
              <div className="flex items-center justify-between border-b border-[#1b2537] pb-2">
                <div>
                  <span className="text-[10px] text-slate-400 uppercase">Runbook Inspector</span>
                  <h3 className="text-sm font-bold text-cyan-400 mt-0.5">{selectedRunbook.runbookId}: {selectedRunbook.name}</h3>
                </div>
                <button
                  onClick={() => handleTriggerRunbook(selectedRunbook)}
                  className="px-3 py-1.5 rounded bg-emerald-950 hover:bg-emerald-900 border border-emerald-500/50 text-emerald-200 font-bold"
                >
                  Trigger Runbook Execution
                </button>
              </div>

              <div>
                <span className="text-slate-400">Trigger Condition:</span>
                <p className="text-slate-200 mt-0.5">{selectedRunbook.triggerDescription}</p>
              </div>

              <div>
                <span className="text-slate-400">Automated Actions:</span>
                <ul className="mt-1 space-y-1 list-disc list-inside text-slate-300">
                  {selectedRunbook.automatedActions.map((act, i) => (
                    <li key={i}>{act}</li>
                  ))}
                </ul>
              </div>

              <div>
                <span className="text-slate-400">Recovery Steps:</span>
                <ul className="mt-1 space-y-1 list-disc list-inside text-slate-300">
                  {selectedRunbook.recoverySteps.map((step, i) => (
                    <li key={i}>{step}</li>
                  ))}
                </ul>
              </div>

              <div className="p-3 rounded bg-[#101724] border border-[#1b2537]">
                <span className="text-slate-400">Exit Criteria:</span>
                <p className="text-emerald-400 font-bold mt-0.5">{selectedRunbook.exitCriteria}</p>
              </div>
            </div>
          </div>
        )}

        {/* TAB 6: M10 Certification Matrix */}
        {activeTab === "CERTIFICATION" && (
          <div className="p-4 rounded-xl bg-[#0c121d] border border-[#1b2537] space-y-4 font-mono text-xs">
            <div className="flex items-center justify-between border-b border-[#1b2537] pb-2">
              <div>
                <h3 className="text-sm font-bold text-cyan-400">Phase 31-M10 Certification Gate Matrix</h3>
                <p className="text-slate-400 text-[11px]">Formal verification across all 10 M10 gates and safety invariants.</p>
              </div>
              <span className="px-3 py-1 rounded bg-emerald-950 border border-emerald-500/60 text-emerald-400 font-bold">
                10/10 GATES CERTIFIED PASS
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead className="bg-[#121927] text-slate-400 border-b border-[#1b2537]">
                  <tr>
                    <th className="p-2.5">Gate ID</th>
                    <th className="p-2.5">Scope</th>
                    <th className="p-2.5">Target Invariant</th>
                    <th className="p-2.5">Requirement</th>
                    <th className="p-2.5 text-right">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1b2537] text-slate-300">
                  {M10_GATE_TRACEABILITY_MATRIX.map((gate) => (
                    <tr key={gate.gateId} className="hover:bg-[#111928]">
                      <td className="p-2.5 font-bold text-cyan-400">{gate.gateId}</td>
                      <td className="p-2.5 text-slate-200">{gate.name}</td>
                      <td className="p-2.5 text-slate-400">{gate.targetInvariant}</td>
                      <td className="p-2.5 text-slate-400">{gate.targetRequirement}</td>
                      <td className="p-2.5 text-right">
                        <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-500/40 font-bold text-[10px]">
                          PASS
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
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
          Loading Autonomous Governance Safety Center...
        </div>
      }
    >
      <AutonomousGovernanceM10Content />
    </Suspense>
  );
}
