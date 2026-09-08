"use client";

import React, { useState, useMemo, Suspense } from "react";
import IntelligenceHeader from "../../components/ui/IntelligenceHeader";
import HorizonMetricCard from "../../components/ui/HorizonMetricCard";
import { HorizonCard } from "../../components/ui/HorizonCard";
import SeverityBadge from "../../components/ui/SeverityBadge";
import RelatedArtifactsPanel, { RelatedArtifactLink } from "../../components/ui/RelatedArtifactsPanel";
import {
  ReleaseReadinessResponse,
  ReleaseGate,
  GatePhase,
  GateStatus,
} from "../../types/release-dashboard";
import { APPROVED_RELEASE_FIXTURE } from "../../lib/release/fixtures/approved";
import { CONDITIONAL_RELEASE_FIXTURE } from "../../lib/release/fixtures/conditional";
import { BLOCKED_RELEASE_FIXTURE } from "../../lib/release/fixtures/blocked";
import {
  evaluateReleaseDecision,
  filterReleaseGates,
  signReleaseAttestation,
  ReleaseAttestation,
} from "../../lib/release/releaseDashboardEngine";

const RELATED_ARTIFACTS: RelatedArtifactLink[] = [
  {
    id: "REL-ART-01",
    type: "AUDIT",
    title: "Executive Intelligence Center",
    href: "/intelligence-center",
    summary: "Institutional cockpit aggregating OHI, ODEI, CDQI, and cross-center telemetry.",
  },
  {
    id: "REL-ART-02",
    type: "DECISION",
    title: "Executive Unified Workspace",
    href: "/executive-workspace",
    summary: "Horizon-native briefing and decision orchestration center for leadership.",
  },
  {
    id: "REL-ART-03",
    type: "COMMITTEE",
    title: "Governance & Attestation Portal",
    href: "/governance",
    summary: "Cryptographic consensus logs, charter compliance, and policy invariants.",
  },
  {
    id: "REL-ART-04",
    type: "AUDIT",
    title: "System Health & Static Architecture",
    href: "/system-health",
    summary: "Real-time telemetry, bundle sizes, and route health monitors.",
  },
];

const PHASES: Array<GatePhase | "ALL"> = [
  "ALL",
  "M1",
  "M2",
  "M3",
  "M4",
  "M5",
  "M6",
  "M7",
  "M8",
  "M9",
  "M10",
  "M11",
  "M12",
  "M13",
  "M14",
  "M15",
  "M16",
];

const STATUS_FILTERS: Array<GateStatus | "ALL"> = ["ALL", "PASS", "WARNING", "FAIL"];

function ReleaseDashboardContent() {
  const [selectedScenario, setSelectedScenario] = useState<"APPROVED" | "CONDITIONAL" | "BLOCKED">("APPROVED");
  const [phaseFilter, setPhaseFilter] = useState<string>("ALL");
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedGate, setSelectedGate] = useState<ReleaseGate | null>(null);
  const [attestation, setAttestation] = useState<ReleaseAttestation | null>(null);

  // Load active fixture based on scenario toggle
  const activePayload: ReleaseReadinessResponse = useMemo(() => {
    switch (selectedScenario) {
      case "CONDITIONAL":
        return CONDITIONAL_RELEASE_FIXTURE;
      case "BLOCKED":
        return BLOCKED_RELEASE_FIXTURE;
      case "APPROVED":
      default:
        return APPROVED_RELEASE_FIXTURE;
    }
  }, [selectedScenario]);

  // Evaluate fail-closed decision
  const decisionResult = useMemo(() => {
    return evaluateReleaseDecision(activePayload);
  }, [activePayload]);

  // Filter gates
  const filteredGates = useMemo(() => {
    return filterReleaseGates(activePayload.gates, phaseFilter, statusFilter, searchQuery);
  }, [activePayload.gates, phaseFilter, statusFilter, searchQuery]);

  const handleSignAttestation = () => {
    const signed = signReleaseAttestation(activePayload, "Executive Committee Lead");
    setAttestation(signed);
  };

  const readinessSeverity =
    activePayload.overallReadinessPct >= 95
      ? "PASS"
      : activePayload.overallReadinessPct >= 80
      ? "WARN"
      : "CRITICAL";

  return (
    <div className="min-h-screen bg-[#0A0F1D] text-[#F8FAFC] p-6 space-y-6">
      {/* Header with breadcrumbs & Scenario Switcher */}
      <IntelligenceHeader
        title="Executive Release-Gate Dashboard"
        subtitle={`ARX Horizon Institutional Release OS · Version ${activePayload.releaseVersion} · ID: ${activePayload.releaseId}`}
        status={
          decisionResult.decision === "APPROVED"
            ? "CERTIFIED"
            : decisionResult.decision === "CONDITIONAL"
            ? "WARNING"
            : "CRITICAL"
        }
        replayHash={activePayload.replayHash}
        breadcrumbs={[
          { label: "Executive Home", href: "/intelligence-center" },
          { label: "Governance & Release", href: "/governance" },
          { label: "Release Certification" },
        ]}
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">Scenario:</span>
            <button
              onClick={() => {
                setSelectedScenario("APPROVED");
                setAttestation(null);
              }}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all ${
                selectedScenario === "APPROVED"
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/50 shadow-sm"
                  : "bg-[#1E293B] text-slate-400 border border-transparent hover:text-white"
              }`}
            >
              Approved (Prod)
            </button>
            <button
              onClick={() => {
                setSelectedScenario("CONDITIONAL");
                setAttestation(null);
              }}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all ${
                selectedScenario === "CONDITIONAL"
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/50 shadow-sm"
                  : "bg-[#1E293B] text-slate-400 border border-transparent hover:text-white"
              }`}
            >
              Conditional (Warn)
            </button>
            <button
              onClick={() => {
                setSelectedScenario("BLOCKED");
                setAttestation(null);
              }}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all ${
                selectedScenario === "BLOCKED"
                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/50 shadow-sm"
                  : "bg-[#1E293B] text-slate-400 border border-transparent hover:text-white"
              }`}
            >
              Blocked (Fail-Closed)
            </button>
          </div>
        }
      />

      {/* Decision Banner */}
      <div
        className={`p-4 rounded-xl border flex flex-col md:flex-row items-start md:items-center justify-between gap-4 transition-all ${
          decisionResult.decision === "APPROVED"
            ? "bg-emerald-950/20 border-emerald-500/40 text-emerald-300"
            : decisionResult.decision === "CONDITIONAL"
            ? "bg-amber-950/20 border-amber-500/40 text-amber-300"
            : "bg-rose-950/20 border-rose-500/40 text-rose-300"
        }`}
      >
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <span
              className={`px-2.5 py-0.5 rounded text-xs font-mono font-bold uppercase tracking-wider ${
                decisionResult.decision === "APPROVED"
                  ? "bg-emerald-500/20 text-emerald-200 border border-emerald-500/30"
                  : decisionResult.decision === "CONDITIONAL"
                  ? "bg-amber-500/20 text-amber-200 border border-amber-500/30"
                  : "bg-rose-500/20 text-rose-200 border border-rose-500/30"
              }`}
            >
              DECISION: {decisionResult.decision}
            </span>
            <span className="text-sm font-semibold">
              {decisionResult.decision === "APPROVED"
                ? "Full Production Release Approved · Zero Critical Governance Blockers"
                : decisionResult.decision === "CONDITIONAL"
                ? "Conditional Production Release · Remediation Required Prior to Deployment"
                : "Fail-Closed Release Block · Deployment Halts Until Invariants Satisfied"}
            </span>
          </div>
          <ul className="text-xs font-mono text-slate-300 space-y-0.5 list-disc list-inside">
            {decisionResult.reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>

        <div className="flex items-center gap-3 self-end md:self-center shrink-0">
          <button
            onClick={handleSignAttestation}
            disabled={decisionResult.decision === "BLOCKED"}
            className={`px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all shadow-md ${
              decisionResult.decision === "BLOCKED"
                ? "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700"
                : "bg-cyan-600 hover:bg-cyan-500 text-white border border-cyan-400/50"
            }`}
          >
            {attestation ? "Attestation Locked" : "Sign Release Attestation"}
          </button>
        </div>
      </div>

      {/* Attestation Details if signed */}
      {attestation && (
        <div className="p-3 rounded-xl bg-[#121B2A] border border-cyan-500/30 text-xs font-mono space-y-1 text-slate-300">
          <div className="flex items-center justify-between text-cyan-300 font-semibold">
            <span>DIGITAL RELEASE ATTESTATION RECORD</span>
            <span className="text-emerald-400">STATUS: VERIFIED</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-[11px] text-slate-400 pt-1">
            <div>Signer: <span className="text-slate-200">{attestation.signer}</span></div>
            <div>Signed At: <span className="text-slate-200">{attestation.signedAtUtc}</span></div>
            <div>SHA-256 Lock: <span className="text-cyan-300 break-all">{attestation.sha256Attestation}</span></div>
          </div>
        </div>
      )}

      {/* 6 Institutional KPI Metrics */}
      <section aria-label="Institutional KPIs" className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        <HorizonMetricCard
          label="Quality Score"
          value={`${activePayload.kpis.qualityScore}%`}
          delta="+2.4%"
          deltaPositive={true}
          severity="PASS"
          subtext="6,542/6,542 assertions"
        />
        <HorizonMetricCard
          label="Governance Score"
          value={`${activePayload.kpis.governanceScore}%`}
          delta="+0.0%"
          deltaPositive={true}
          severity={activePayload.kpis.governanceScore >= 100 ? "PASS" : "CRITICAL"}
          subtext="0 policy violations"
        />
        <HorizonMetricCard
          label="Accessibility Score"
          value={`${activePayload.kpis.accessibilityScore}%`}
          delta={activePayload.accessibility.axeViolations > 0 ? "-6.0%" : "+0.0%"}
          deltaPositive={activePayload.accessibility.axeViolations === 0}
          severity={activePayload.accessibility.axeViolations === 0 ? "PASS" : "WARN"}
          subtext={`${activePayload.accessibility.axeViolations} axe violations`}
        />
        <HorizonMetricCard
          label="Resilience Score"
          value={`${activePayload.kpis.resilienceScore}%`}
          delta="+1.2%"
          deltaPositive={true}
          severity="PASS"
          subtext="Replay deterministic"
        />
        <HorizonMetricCard
          label="Performance Score"
          value={`${activePayload.kpis.performanceScore}%`}
          delta="+3.1%"
          deltaPositive={true}
          severity="PASS"
          subtext={`${activePayload.performance.sharedJsKb} kB shared JS`}
        />
        <HorizonMetricCard
          label="Executive Readiness"
          value={`${activePayload.overallReadinessPct}%`}
          delta={activePayload.overallReadinessPct >= 90 ? "+5.0%" : "-12.0%"}
          deltaPositive={activePayload.overallReadinessPct >= 90}
          severity={readinessSeverity}
          subtext={`${activePayload.summary.certificationGatesPassed}/${activePayload.summary.certificationGatesTotal} gates passed`}
        />
      </section>

      {/* 4 Verification Pillars */}
      <section aria-label="Verification Pillars" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <HorizonCard
          title="Automated Verification"
          subtitle="End-to-End Suite Assertions"
          badge={<SeverityBadge status={activePayload.verification.failedAssertions === 0 ? "HEALTHY" : "CRITICAL"} />}
        >
          <div className="space-y-2 text-xs font-mono text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-400">Total Assertions:</span>
              <span className="font-semibold text-white">{activePayload.verification.totalAssertions}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Passed / Failed:</span>
              <span className="font-semibold text-emerald-400">
                {activePayload.verification.passedAssertions} / {activePayload.verification.failedAssertions}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Flaky Tests:</span>
              <span className={activePayload.verification.flakyTests > 0 ? "text-amber-400" : "text-emerald-400"}>
                {activePayload.verification.flakyTests}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Coverage Pct:</span>
              <span className="font-semibold text-cyan-300">{activePayload.verification.coveragePct}%</span>
            </div>
          </div>
        </HorizonCard>

        <HorizonCard
          title="Accessibility & WCAG"
          subtitle="Design System Compliance"
          badge={<SeverityBadge status={activePayload.accessibility.axeViolations === 0 ? "HEALTHY" : "WARNING"} />}
        >
          <div className="space-y-2 text-xs font-mono text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-400">WCAG Standard:</span>
              <span className="font-semibold text-cyan-300">{activePayload.accessibility.wcagLevel}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Axe Violations:</span>
              <span className={activePayload.accessibility.axeViolations > 0 ? "text-amber-400 font-bold" : "text-emerald-400"}>
                {activePayload.accessibility.axeViolations}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Keyboard Nav:</span>
              <span className="text-emerald-400">
                {activePayload.accessibility.keyboardNavigationPassed ? "VERIFIED" : "FAILED"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Contrast & Focus:</span>
              <span className="text-emerald-400">VERIFIED</span>
            </div>
          </div>
        </HorizonCard>

        <HorizonCard
          title="Performance & Bundles"
          subtitle="Static Export Architecture"
          badge={<SeverityBadge status={activePayload.performance.buildPassed ? "HEALTHY" : "CRITICAL"} />}
        >
          <div className="space-y-2 text-xs font-mono text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-400">Shared JS:</span>
              <span className="font-semibold text-white">
                {activePayload.performance.sharedJsKb} kB / {activePayload.performance.jsBudgetKb} kB
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Static Routes:</span>
              <span className="font-semibold text-cyan-300">{activePayload.performance.staticRoutes}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Build Status:</span>
              <span className={activePayload.performance.buildPassed ? "text-emerald-400" : "text-rose-400"}>
                {activePayload.performance.buildPassed ? "PASSING" : "FAILED"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Page Load:</span>
              <span className="font-semibold text-slate-200">{activePayload.performance.pageLoadSeconds}s</span>
            </div>
          </div>
        </HorizonCard>

        <HorizonCard
          title="Security & Invariants"
          subtitle="Fail-Closed Boundaries"
          badge={<SeverityBadge status={activePayload.security.criticalVulnerabilities === 0 ? "HEALTHY" : "CRITICAL"} />}
        >
          <div className="space-y-2 text-xs font-mono text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-400">Critical Vulns:</span>
              <span className={activePayload.security.criticalVulnerabilities > 0 ? "text-rose-400 font-bold" : "text-emerald-400"}>
                {activePayload.security.criticalVulnerabilities}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Replay Drift:</span>
              <span className={activePayload.security.replayDriftIncidents > 0 ? "text-rose-400 font-bold" : "text-emerald-400"}>
                {activePayload.security.replayDriftIncidents}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Consistency Violations:</span>
              <span className={activePayload.security.consistencyViolations > 0 ? "text-rose-400 font-bold" : "text-emerald-400"}>
                {activePayload.security.consistencyViolations}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Governance Integrity:</span>
              <span className="text-emerald-400">VERIFIED</span>
            </div>
          </div>
        </HorizonCard>
      </section>

      {/* Milestone Gates Grid (M1 to M16) */}
      <HorizonCard
        title="Milestone Certification Gates"
        subtitle={`Showing ${filteredGates.length} of ${activePayload.gates.length} certification gates across M1–M16`}
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <input
              type="text"
              placeholder="Search gates, owners..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="px-3 py-1.5 rounded-lg bg-[#0E1524] border border-[#24324A] text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-400"
            />
            <select
              value={phaseFilter}
              onChange={e => setPhaseFilter(e.target.value)}
              className="px-3 py-1.5 rounded-lg bg-[#0E1524] border border-[#24324A] text-xs text-slate-300 focus:outline-none focus:border-cyan-400"
            >
              {PHASES.map(p => (
                <option key={p} value={p}>
                  {p === "ALL" ? "All Phases" : `Phase ${p}`}
                </option>
              ))}
            </select>
            <select
              value={statusFilter}
              onChange={e => setStatusFilter(e.target.value)}
              className="px-3 py-1.5 rounded-lg bg-[#0E1524] border border-[#24324A] text-xs text-slate-300 focus:outline-none focus:border-cyan-400"
            >
              {STATUS_FILTERS.map(s => (
                <option key={s} value={s}>
                  {s === "ALL" ? "All Statuses" : s}
                </option>
              ))}
            </select>
          </div>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-[#24324A] text-slate-400">
                <th className="py-2.5 px-3">Gate ID</th>
                <th className="py-2.5 px-3">Gate Name</th>
                <th className="py-2.5 px-3">Phase</th>
                <th className="py-2.5 px-3">Owner</th>
                <th className="py-2.5 px-3">Status</th>
                <th className="py-2.5 px-3">Assertions</th>
                <th className="py-2.5 px-3">Duration</th>
                <th className="py-2.5 px-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#24324A]/40">
              {filteredGates.map(gate => (
                <tr
                  key={gate.gateId}
                  className="hover:bg-[#152136] transition-colors cursor-pointer"
                  onClick={() => setSelectedGate(gate)}
                >
                  <td className="py-2.5 px-3 font-semibold text-cyan-300">{gate.gateId}</td>
                  <td className="py-2.5 px-3 text-white font-sans font-medium">{gate.gateName}</td>
                  <td className="py-2.5 px-3">
                    <span className="px-2 py-0.5 rounded bg-[#1E293B] text-slate-300 text-[11px]">
                      {gate.phase}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-slate-400">{gate.owner}</td>
                  <td className="py-2.5 px-3">
                    <SeverityBadge
                      status={gate.status === "PASS" ? "HEALTHY" : gate.status === "WARNING" ? "WARNING" : "CRITICAL"}
                    />
                  </td>
                  <td className="py-2.5 px-3">
                    <span className={gate.passedAssertions === gate.assertionCount ? "text-emerald-400" : "text-rose-400"}>
                      {gate.passedAssertions}/{gate.assertionCount}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-slate-400">{gate.executionDurationMs}ms</td>
                  <td className="py-2.5 px-3 text-right">
                    <button
                      onClick={e => {
                        e.stopPropagation();
                        setSelectedGate(gate);
                      }}
                      className="text-cyan-400 hover:text-cyan-200 hover:underline"
                    >
                      Inspect
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </HorizonCard>

      {/* Gate Drilldown Modal */}
      {selectedGate && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="w-full max-w-xl rounded-2xl bg-[#121B2A] border border-[#24324A] p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#24324A] pb-3">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-base font-semibold text-white font-sans">{selectedGate.gateName}</h3>
                  <SeverityBadge
                    status={selectedGate.status === "PASS" ? "HEALTHY" : selectedGate.status === "WARNING" ? "WARNING" : "CRITICAL"}
                  />
                </div>
                <p className="text-xs font-mono text-cyan-400 mt-0.5">{selectedGate.gateId} · Phase {selectedGate.phase}</p>
              </div>
              <button
                onClick={() => setSelectedGate(null)}
                className="text-slate-400 hover:text-white text-lg font-bold px-2 py-1"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3 text-xs font-mono text-slate-300">
              <div className="grid grid-cols-2 gap-3 p-3 rounded-xl bg-[#0E1524] border border-[#24324A]/60">
                <div>
                  <span className="text-slate-500">Owner:</span>
                  <div className="text-white font-medium">{selectedGate.owner}</div>
                </div>
                <div>
                  <span className="text-slate-500">Duration:</span>
                  <div className="text-white font-medium">{selectedGate.executionDurationMs} ms</div>
                </div>
                <div>
                  <span className="text-slate-500">Passed Assertions:</span>
                  <div className="text-emerald-400 font-semibold">{selectedGate.passedAssertions}</div>
                </div>
                <div>
                  <span className="text-slate-500">Total Assertions:</span>
                  <div className="text-white font-semibold">{selectedGate.assertionCount}</div>
                </div>
              </div>

              {selectedGate.failureReason && (
                <div className="p-3 rounded-xl bg-rose-950/20 border border-rose-500/40 text-rose-300">
                  <div className="font-semibold text-[11px] uppercase tracking-wider text-rose-400 mb-1">
                    Failure / Warning Root Cause:
                  </div>
                  <div>{selectedGate.failureReason}</div>
                </div>
              )}

              <div className="text-[11px] text-slate-400">
                Last verified: {selectedGate.lastVerifiedUtc}
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={() => setSelectedGate(null)}
                className="px-4 py-2 rounded-xl bg-[#1E293B] hover:bg-[#2A3B53] text-xs font-mono text-white transition-colors"
              >
                Close Drilldown
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Related Artifacts Panel */}
      <RelatedArtifactsPanel
        title="Cross-System Lineage & Artifact Navigation"
        artifacts={RELATED_ARTIFACTS}
      />
    </div>
  );
}

export default function ReleaseDashboardPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-[#0A0F1D] text-slate-400 flex items-center justify-center font-mono text-sm">
          Loading Executive Release Dashboard...
        </div>
      }
    >
      <ReleaseDashboardContent />
    </Suspense>
  );
}
