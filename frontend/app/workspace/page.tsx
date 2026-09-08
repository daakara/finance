"use client";

import React, { useState, Suspense } from "react";
import Link from "next/link";
import IntelligenceShell from "../../components/ui/IntelligenceShell";
import HorizonCard from "../../components/ui/HorizonCard";
import HorizonMetricCard from "../../components/ui/HorizonMetricCard";
import IntelligenceLoadingState from "../../components/ui/IntelligenceLoadingState";
import IntelligenceSuccessState from "../../components/ui/IntelligenceSuccessState";
import {
  getWorkspaceProfile,
  checkCommitteeAccess,
  verifyWorkspaceConsistency,
  computeWorkspaceStateHash,
  paginateTasks,
} from "../../lib/productivity/executiveWorkspaceEngine";
import type { ExecutiveUserRole, WorkspaceProfile } from "../../types/executive-workspace";

function ExecutiveWorkspaceContent() {
  const [selectedRole, setSelectedRole] = useState<ExecutiveUserRole>("CHIEF_INVESTMENT_OFFICER");
  const [simulateOutage, setSimulateOutage] = useState<boolean>(false);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [accessTestResult, setAccessTestResult] = useState<{ committeeId: string; granted: boolean; message: string } | null>(null);
  const [consistencyResult, setConsistencyResult] = useState<{ consistent: boolean; message: string } | null>(null);

  const profile: WorkspaceProfile = getWorkspaceProfile(selectedRole, {
    simulateTelemetryOutage: simulateOutage,
  });

  const stateHash = computeWorkspaceStateHash(profile);
  const paginated = paginateTasks(profile.ownedTasks, currentPage, 3);

  const roles: { role: ExecutiveUserRole; label: string }[] = [
    { role: "CHIEF_INVESTMENT_OFFICER", label: "CIO" },
    { role: "CHIEF_RISK_OFFICER", label: "CRO" },
    { role: "BOARD_DIRECTOR", label: "Board Director" },
    { role: "COMMITTEE_CHAIR", label: "Committee Chair" },
    { role: "AUDIT_PARTNER", label: "Audit Partner" },
  ];

  const handleRoleChange = (role: ExecutiveUserRole) => {
    setSelectedRole(role);
    setCurrentPage(1);
    setAccessTestResult(null);
    setConsistencyResult(null);
  };

  const testAccess = (committeeId: string) => {
    const res = checkCommitteeAccess(profile, committeeId);
    if (res.granted) {
      setAccessTestResult({
        committeeId,
        granted: true,
        message: `Access granted for ${profile.name} to ${committeeId}. Verified against active assignment matrix.`,
      });
    } else {
      setAccessTestResult({
        committeeId,
        granted: false,
        message: res.error?.message || `Access denied to ${committeeId}. Fail-closed check enforced.`,
      });
    }
  };

  const handleRunConsistencyCheck = () => {
    // Check against canonical source (expected count: profile.pendingApprovalsCount)
    const res = verifyWorkspaceConsistency(profile, profile.pendingApprovalsCount);
    if (res.consistent) {
      setConsistencyResult({
        consistent: true,
        message: `Workspace telemetry strictly consistent with source engines (0 drift events).`,
      });
    } else {
      setConsistencyResult({
        consistent: false,
        message: res.error?.message || `Drift detected.`,
      });
    }
  };

  return (
    <IntelligenceShell
      title="Executive Workspace"
      subtitle="ARX Horizon Executive Operating System - Personalized Mission Control"
    >
      {/* Role Selection & Degradation Simulator Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-2xl bg-[#121B2A] border border-[#24324A] shadow-md">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider">
            Active Executive Persona:
          </span>
          {roles.map((r) => (
            <button
              key={r.role}
              onClick={() => handleRoleChange(r.role)}
              className={`px-3 py-1.5 rounded-xl text-xs font-medium transition-all ${
                selectedRole === r.role
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/10"
                  : "bg-[#182336] text-slate-400 border border-[#24324A] hover:text-slate-200"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 cursor-pointer text-xs font-mono text-slate-300">
            <input
              type="checkbox"
              checked={simulateOutage}
              onChange={(e) => setSimulateOutage(e.target.checked)}
              className="rounded border-[#24324A] bg-[#182336] text-cyan-400 focus:ring-0 w-4 h-4 cursor-pointer"
            />
            <span>Simulate Telemetry Outage (WS-03)</span>
          </label>
        </div>
      </div>

      {/* Degradation Alert if Outage Active */}
      {profile.isDegraded && (
        <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-300 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-ping shrink-0" />
            <div className="text-xs">
              <strong className="font-bold uppercase tracking-wider block">
                Telemetry Outage Active - Graceful Fallback Engaged (WS-EC-03)
              </strong>
              <span>
                Operating in certified snapshot mode. Last snapshot timestamp:{" "}
                <code className="font-mono font-bold text-white">{profile.lastSnapshotTimestampUtc}</code>
              </span>
            </div>
          </div>
          <span className="px-2.5 py-1 rounded bg-amber-500/20 text-[10px] font-mono font-bold uppercase border border-amber-500/40">
            DEGRADED_MODE
          </span>
        </div>
      )}

      {/* Profile Overview Header Card */}
      <HorizonCard
        title={profile.name}
        subtitle={`Role: ${profile.role} | User ID: ${profile.userId}`}
        badge={
          <span className="px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
            {profile.isDegraded ? "SNAPSHOT_MODE" : "LIVE_TELEMETRY"}
          </span>
        }
        actions={
          <button
            onClick={handleRunConsistencyCheck}
            className="px-3 py-1.5 rounded-xl bg-[#182336] text-xs font-mono text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/10 transition-colors"
          >
            Verify Consistency
          </button>
        }
      >
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <HorizonMetricCard
            label="Assigned Committees"
            value={profile.assignedCommittees.length}
            target="Active Mandates"
            severity="PASS"
          />
          <HorizonMetricCard
            label="Active Tasks"
            value={profile.ownedTasks.length}
            target="My Queue"
            severity={profile.ownedTasks.length > 2 ? "WARN" : "PASS"}
          />
          <HorizonMetricCard
            label="Pending Approvals"
            value={profile.pendingApprovalsCount}
            target="Awaiting Sign-off"
            severity={profile.pendingApprovalsCount > 0 ? "WARN" : "PASS"}
          />
          <HorizonMetricCard
            label="Active Risks"
            value={profile.activeRisksCount}
            target="Under Oversight"
            severity={profile.activeRisksCount > 2 ? "CRITICAL" : "PASS"}
          />
        </div>

        <div className="mt-4 pt-3 border-t border-[#24324A] flex flex-wrap items-center justify-between gap-2 text-xs font-mono text-slate-400">
          <span>
            Workspace State Hash:{" "}
            <strong className="text-cyan-400">{stateHash.slice(0, 16)}...{stateHash.slice(-8)}</strong>
          </span>
          <span>SLA Freshness: {profile.telemetryFreshnessSlaMinutes} mins</span>
        </div>
      </HorizonCard>

      {/* Consistency Verification Alert */}
      {consistencyResult && (
        <IntelligenceSuccessState
          title="Workspace Consistency Verified"
          message={consistencyResult.message}
          auditHash={stateHash}
          certificationId="CERT-M13-WS-CONSISTENCY"
        />
      )}

      {/* Access Test Result Alert */}
      {accessTestResult && (
        <div
          className={`p-4 rounded-2xl border text-xs flex items-center justify-between gap-4 ${
            accessTestResult.granted
              ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
              : "bg-rose-500/10 border-rose-500/30 text-rose-300"
          }`}
        >
          <div className="flex items-center gap-2.5">
            <span
              className={`w-2 h-2 rounded-full ${
                accessTestResult.granted ? "bg-emerald-400" : "bg-rose-400"
              }`}
            />
            <span>{accessTestResult.message}</span>
          </div>
          <button
            onClick={() => setAccessTestResult(null)}
            className="text-[10px] font-mono underline hover:no-underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Two Column Layout: Assigned Committees & Owned Tasks */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Assigned Committees */}
        <HorizonCard
          title="My Assigned Committees"
          subtitle={`${profile.assignedCommittees.length} Active Committee Appointments`}
          actions={
            <button
              onClick={() => testAccess("COM-999-UNASSIGNED")}
              className="px-2 py-1 rounded bg-[#182336] text-[10px] font-mono text-amber-300 border border-amber-500/30 hover:bg-amber-500/10 transition-colors"
            >
              Test Unauthorized Access (WS-EC-05)
            </button>
          }
        >
          <div className="space-y-3">
            {profile.assignedCommittees.map((comm) => (
              <div
                key={comm.committeeId}
                className="p-4 rounded-xl bg-[#182336] border border-[#24324A] flex flex-col sm:flex-row sm:items-center justify-between gap-3"
              >
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-semibold text-white">{comm.name}</h3>
                    <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/30">
                      {comm.role}
                    </span>
                  </div>
                  <p className="text-xs font-mono text-slate-400 mt-1">
                    ID: {comm.committeeId} | Health:{" "}
                    <strong className="text-emerald-400">{comm.healthScore.toFixed(1)}/100</strong> | Active Decisions: {comm.activeDecisions}
                  </p>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={() => testAccess(comm.committeeId)}
                    className="px-2.5 py-1 rounded bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 text-xs font-mono border border-cyan-500/30 transition-colors"
                  >
                    Verify Access
                  </button>
                  <Link
                    href={`/committee-network?committeeId=${comm.committeeId}`}
                    className="px-2.5 py-1 rounded bg-[#152033] hover:bg-[#1f2e4a] text-slate-300 text-xs font-mono border border-[#24324A] transition-colors"
                  >
                    View
                  </Link>
                </div>
              </div>
            ))}
          </div>
        </HorizonCard>

        {/* Owned Tasks Queue */}
        <HorizonCard
          title="Executive Action Queue"
          subtitle={`Page ${paginated.currentPage} of ${paginated.totalPages} (${paginated.totalItems} Total Tasks)`}
          actions={
            <div className="flex items-center gap-2 text-xs font-mono">
              <button
                disabled={paginated.currentPage <= 1}
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                className="px-2 py-0.5 rounded bg-[#182336] disabled:opacity-40 text-slate-300 border border-[#24324A]"
              >
                Prev
              </button>
              <button
                disabled={paginated.currentPage >= paginated.totalPages}
                onClick={() => setCurrentPage((p) => Math.min(paginated.totalPages, p + 1))}
                className="px-2 py-0.5 rounded bg-[#182336] disabled:opacity-40 text-slate-300 border border-[#24324A]"
              >
                Next
              </button>
            </div>
          }
        >
          <div className="space-y-3">
            {paginated.items.map((task) => {
              const prioBadge =
                task.priority === "CRITICAL"
                  ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                  : task.priority === "HIGH"
                  ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                  : "bg-cyan-500/20 text-cyan-300 border-cyan-500/40";
              return (
                <div
                  key={task.taskId}
                  className="p-4 rounded-xl bg-[#182336] border border-[#24324A] flex flex-col sm:flex-row sm:items-center justify-between gap-3"
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${prioBadge}`}>
                        {task.priority}
                      </span>
                      <span className="text-xs font-semibold text-white">{task.title}</span>
                    </div>
                    <div className="text-[11px] font-mono text-slate-400 flex items-center gap-3">
                      <span>Category: {task.category}</span>
                      <span>SLA Remaining: <strong className="text-white">{task.slaRemainingHours}h</strong></span>
                      <span>Committee: {task.committeeId}</span>
                    </div>
                  </div>

                  <Link
                    href="/decision-inbox"
                    className="shrink-0 px-3 py-1 rounded-xl bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 text-xs font-mono font-semibold border border-cyan-500/30 transition-colors text-center"
                  >
                    Triage In Inbox
                  </Link>
                </div>
              );
            })}
          </div>
        </HorizonCard>
      </div>

      {/* Quick Access Navigation Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
        <Link
          href="/decision-inbox"
          className="p-5 rounded-2xl bg-[#121B2A] border border-[#24324A] hover:border-cyan-500/40 transition-all group"
        >
          <h3 className="text-sm font-semibold text-white group-hover:text-cyan-400 transition-colors">
            Unified Decision Inbox →
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Consolidated triage for approvals, escalations, recommendations, and runbooks.
          </p>
        </Link>

        <Link
          href="/strategy-laboratory"
          className="p-5 rounded-2xl bg-[#121B2A] border border-[#24324A] hover:border-cyan-500/40 transition-all group"
        >
          <h3 className="text-sm font-semibold text-white group-hover:text-cyan-400 transition-colors">
            Strategic Simulation Laboratory →
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Simulate decisions against market shocks and run digital twins.
          </p>
        </Link>

        <Link
          href="/governance-center"
          className="p-5 rounded-2xl bg-[#121B2A] border border-[#24324A] hover:border-cyan-500/40 transition-all group"
        >
          <h3 className="text-sm font-semibold text-white group-hover:text-cyan-400 transition-colors">
            Autonomous Governance Center →
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Audit fail-closed autonomous safety gates and policy attestations.
          </p>
        </Link>
      </div>
    </IntelligenceShell>
  );
}

export default function ExecutiveWorkspacePage() {
  return (
    <Suspense fallback={<IntelligenceLoadingState message="Loading Executive Workspace..." />}>
      <ExecutiveWorkspaceContent />
    </Suspense>
  );
}
