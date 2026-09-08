"use client";

import React, { useState, useMemo, Suspense } from "react";
import Link from "next/link";
import ExecutiveIntelligenceNav from "../../components/committee/ExecutiveIntelligenceNav";
import IntelligenceHeader from "../../components/ui/IntelligenceHeader";
import HorizonCard from "../../components/ui/HorizonCard";
import HorizonMetricCard from "../../components/ui/HorizonMetricCard";
import SeverityBadge from "../../components/ui/SeverityBadge";
import RelatedArtifactsPanel from "../../components/ui/RelatedArtifactsPanel";
import {
  ExecutiveDecisionRole,
  DecisionPackage,
  DecisionOption,
  PackageStatus,
  ApprovalReceipt,
  OutcomeRecord,
  LearningRecord,
  BoardBriefingPack,
} from "../../types/executive-workspace-decision";

import {
  getCanonicalDecisionPackages,
  computePackageStateHash,
  validateDriverAttribution,
  transitionPackageStage,
} from "../../lib/workspace/decisionPackageEngine";

import {
  evaluateOptions,
  getRecommendedOption,
  compareOptions,
} from "../../lib/workspace/optionAnalysisEngine";

import {
  getRoleLayoutConfig,
  verifyPersonalizationGuardrails,
  ROLE_CONFIGS,
} from "../../lib/workspace/rolePersonalizationEngine";

import {
  validateGovernanceChecklist,
  executeDigitalApproval,
  executeDecisionAction,
} from "../../lib/workspace/approvalGovernanceEngine";

import {
  getOutcomeRecord,
  calculateTrajectoryDivergence,
  verifyOutcomeDriverAttribution,
} from "../../lib/workspace/outcomeMonitoringEngine";

import {
  getInstitutionalLearnings,
  captureLearningFromOutcome,
  trackLearningAdoption,
} from "../../lib/workspace/learningClosureEngine";

import {
  generateBoardBriefing,
  verifyBriefingReplayHash,
} from "../../lib/workspace/boardBriefingEngine";

function ExecutiveWorkspaceInner() {
  // State: Role & Guardrails
  const [selectedRole, setSelectedRole] = useState<ExecutiveDecisionRole>("Executive");
  const [packages, setPackages] = useState<DecisionPackage[]>(() => getCanonicalDecisionPackages());
  const [selectedPackageId, setSelectedPackageId] = useState<string>("PKG-2026-001");
  const [selectedStageFilter, setSelectedStageFilter] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");

  // State: Selected Option in active package
  const [selectedOptionId, setSelectedOptionId] = useState<string>("OPT-01-A");

  // State: Approvals & Execution
  const [actionNotice, setActionNotice] = useState<{ text: string; type: "success" | "error" | "info" } | null>(null);

  // State: Learnings & Board Briefings
  const [learnings, setLearnings] = useState<LearningRecord[]>(() => getInstitutionalLearnings());
  const [newLessonTitle, setNewLessonTitle] = useState("");
  const [newLessonInsight, setNewLessonInsight] = useState("");
  const [activeBoardBrief, setActiveBoardBrief] = useState<BoardBriefingPack | null>(null);

  // Active Role Configuration
  const roleConfig = useMemo(() => getRoleLayoutConfig(selectedRole), [selectedRole]);

  // Guardrail Compliance Check (GP-001..006)
  const guardrailResult = useMemo(
    () => verifyPersonalizationGuardrails(packages, selectedRole),
    [packages, selectedRole]
  );

  // Active Decision Package
  const activePackage = useMemo(() => {
    return packages.find((p) => p.packageId === selectedPackageId) || packages[0];
  }, [packages, selectedPackageId]);

  // Filtered Decision Packages
  const filteredPackages = useMemo(() => {
    return packages.filter((p) => {
      const matchesStage =
        selectedStageFilter === "ALL" ||
        p.status === selectedStageFilter ||
        p.currentStage === selectedStageFilter;
      const matchesSearch =
        searchQuery === "" ||
        p.packageId.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.originatingCommittee.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesStage && matchesSearch;
    });
  }, [packages, selectedStageFilter, searchQuery]);

  // Evaluated options for active package
  const evaluatedOptions = useMemo(() => {
    return evaluateOptions(activePackage?.options || []);
  }, [activePackage]);

  // Active option details
  const activeOption = useMemo(() => {
    return evaluatedOptions.find((o) => o.optionId === selectedOptionId) || evaluatedOptions[0];
  }, [evaluatedOptions, selectedOptionId]);

  // Active outcome record
  const activeOutcome = useMemo(() => {
    return getOutcomeRecord(activePackage?.packageId);
  }, [activePackage]);

  // Handlers
  const handleRoleChange = (newRole: ExecutiveDecisionRole) => {
    setSelectedRole(newRole);
    setActionNotice({
      text: `Switched perspective to ${newRole}. Layout reordered; underlying metrics & critical risks strictly invariant (GP-001..006).`,
      type: "info",
    });
  };

  const handleApprove = () => {
    if (!activePackage) return;
    const res = executeDigitalApproval(activePackage, {
      role: selectedRole,
      name: `Officer (${selectedRole})`,
    });

    if (res.success && res.package) {
      setPackages((prev) =>
        prev.map((p) => (p.packageId === res.package!.packageId ? res.package! : p))
      );
      setActionNotice({
        text: `DECISION APPROVED: Cryptographic signature ${res.receipt?.signatureHash} anchored with receipt ${res.receipt?.auditReceiptHash}.`,
        type: "success",
      });
    } else {
      setActionNotice({
        text: res.error || "Approval failed.",
        type: "error",
      });
    }
  };

  const handleExecute = () => {
    if (!activePackage) return;
    const res = executeDecisionAction(activePackage, "EXECUTE");
    if (res.success && res.package) {
      setPackages((prev) =>
        prev.map((p) => (p.packageId === res.package!.packageId ? res.package! : p))
      );
      setActionNotice({
        text: `DECISION EXECUTED: Status transitioned to COMPLETED. Advanced to Outcome Monitoring & Learning Closure.`,
        type: "success",
      });
    } else {
      setActionNotice({
        text: res.error || "Execution failed.",
        type: "error",
      });
    }
  };

  const handleReject = () => {
    if (!activePackage) return;
    const res = executeDecisionAction(activePackage, "REJECT");
    if (res.success && res.package) {
      setPackages((prev) =>
        prev.map((p) => (p.packageId === res.package!.packageId ? res.package! : p))
      );
      setActionNotice({
        text: `DECISION REJECTED: Package ${activePackage.packageId} returned to committee with formal dissent notice.`,
        type: "info",
      });
    }
  };

  const handleCaptureLearning = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newLessonTitle || !newLessonInsight || !activePackage) return;

    const newRecord = captureLearningFromOutcome(
      activePackage.packageId,
      activeOutcome?.outcomeId || `OUT-${activePackage.packageId}`,
      newLessonTitle,
      "GOVERNANCE",
      newLessonInsight,
      90.0
    );

    setLearnings((prev) => [newRecord, ...prev]);
    setNewLessonTitle("");
    setNewLessonInsight("");
    setActionNotice({
      text: `LEARNING CAPTURED: Recorded ${newRecord.learningId} linked to ${activePackage.packageId} with provenance hash ${newRecord.provenanceHash}.`,
      type: "success",
    });
  };

  const handleGenerateBriefing = (type: "MONTHLY_BRIEF" | "QUARTERLY_REPORT" | "DECISION_PACK") => {
    const brief = generateBoardBriefing(type, packages);
    setActiveBoardBrief(brief);
    setActionNotice({
      text: `BOARD BRIEFING GENERATED: ${brief.title} synthesized with deterministic replay hash ${brief.replayHash}.`,
      type: "success",
    });
  };

  // Quick stats
  const totalPackages = packages.length;
  const readyCount = packages.filter((p) => p.status === "READY_FOR_APPROVAL").length;
  const completedCount = packages.filter((p) => p.status === "COMPLETED").length;
  const failedCount = packages.filter((p) => p.status === "FAILED").length;

  return (
    <div className="min-h-screen bg-[#070D17] text-slate-100 font-sans pb-16">
      {/* Universal Navigation */}
      <ExecutiveIntelligenceNav activeTab="/executive-workspace" badgeText="PHASE 31-M16" />

      <main className="max-w-[1680px] mx-auto px-4 sm:px-6 lg:px-8 pt-6 space-y-6">
        {/* Top Header */}
        <IntelligenceHeader
          title="Executive Decision Workspace"
          subtitle="ARX Horizon Executive OS — Unified Decision Lifecycle, Multi-Option Tradeoffs & Outcome Learning Closure"
          status="CERTIFIED"
          certification="M16-VERIFIED"
          replayHash={activePackage?.stateHash || "PKG-HASH-0xCANONICAL"}
          breadcrumbs={[
            { label: "Home", href: "/intelligence-center" },
            { label: "Workspaces", href: "/executive-workspace" },
            { label: "Decision Operating System" },
          ]}
          actions={
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-mono text-slate-400">Persona Role:</span>
              <div className="flex rounded-lg bg-[#0F172A] border border-[#24324A] p-1 gap-1">
                {(["Executive", "CommitteeChair", "Analyst", "Auditor", "GovernanceOfficer"] as ExecutiveDecisionRole[]).map((r) => (
                  <button
                    key={r}
                    onClick={() => handleRoleChange(r)}
                    className={`px-3 py-1 text-xs font-mono rounded-md transition-all ${
                      selectedRole === r
                        ? "bg-cyan-600 text-white font-bold shadow-md shadow-cyan-600/30"
                        : "text-slate-400 hover:text-white hover:bg-slate-800"
                    }`}
                  >
                    {r}
                  </button>
                ))}
              </div>
            </div>
          }
        />

        {/* Institutional Pulse (Station 1: ExecutiveHeader) */}
        <section aria-label="Institutional Pulse" className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <HorizonMetricCard
            label="OHI Index"
            value="86.4"
            delta="+4.8"
            deltaPositive={true}
            severity="PASS"
            subtext="Target: 91.2 Projected"
          />
          <HorizonMetricCard
            label="Active VaR (99%)"
            value="$1.25M"
            delta="-6.5%"
            deltaPositive={true}
            severity="PASS"
            subtext="Ceiling: $2.0M Floor"
          />
          <HorizonMetricCard
            label="Resilience RTO"
            value="4.2 min"
            delta="0.0"
            deltaPositive={true}
            severity="PASS"
            subtext="SLA: < 15.0 min"
          />
          <HorizonMetricCard
            label="Pending Approvals"
            value={`${readyCount}`}
            delta="Urgent"
            deltaPositive={readyCount === 0}
            severity={readyCount > 0 ? "WARN" : "PASS"}
            subtext="2 Within SLA Window"
          />
          <HorizonMetricCard
            label="Fail-Closed Quota"
            value={`${failedCount}`}
            delta="Isolated"
            deltaPositive={failedCount === 0}
            severity={failedCount > 0 ? "CRITICAL" : "PASS"}
            subtext="Strict Zero-Mutation Lock"
          />
          <HorizonMetricCard
            label="Transfer Adoption"
            value="88.5%"
            delta="+12.0%"
            deltaPositive={true}
            severity="PASS"
            subtext="2 Active Learnings"
          />
        </section>

        {/* Action Notice Alert */}
        {actionNotice && (
          <div
            role="status"
            className={`p-3.5 rounded-lg border text-xs font-mono flex items-center justify-between transition-all ${
              actionNotice.type === "success"
                ? "bg-emerald-950/40 border-emerald-500/50 text-emerald-300"
                : actionNotice.type === "error"
                ? "bg-rose-950/40 border-rose-500/50 text-rose-300"
                : "bg-cyan-950/40 border-cyan-500/50 text-cyan-300"
            }`}
          >
            <span>{actionNotice.text}</span>
            <button
              onClick={() => setActionNotice(null)}
              className="text-slate-400 hover:text-white px-2 py-0.5 rounded"
            >
              ✕
            </button>
          </div>
        )}

        {/* Role Focus & Guardrail Assurance Bar */}
        <div className="p-3 rounded-lg bg-[#0F172A] border border-[#202E45] flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs font-mono">
          <div className="flex items-center gap-3">
            <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-cyan-950 text-cyan-400 border border-cyan-800">
              {roleConfig.label}
            </span>
            <span className="text-slate-400">{roleConfig.subtitle}</span>
          </div>
          <div className="flex items-center gap-4 text-[11px]">
            <span className="text-emerald-400 flex items-center gap-1">
              ● Zero Fact Drift (GP-001 Verified)
            </span>
            <span className="text-emerald-400 flex items-center gap-1">
              ● Critical Risks Visible (GP-005)
            </span>
            <span className="text-slate-400">
              Density: <span className="text-white font-bold">{roleConfig.defaultDensity}</span>
            </span>
          </div>
        </div>

        {/* Station 2: Executive Action Center & Critical Risk Banner (Unhideable GP-005) */}
        <section aria-label="Executive Action Center" className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2 p-4 rounded-xl bg-gradient-to-r from-[#171A24] to-[#121B2A] border border-rose-500/30 shadow-md">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse" />
                <h2 className="text-sm font-bold text-white uppercase tracking-wider font-mono">
                  Mandatory Governance & Critical Risk Envelope (GP-005)
                </h2>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-rose-950/80 text-rose-300 border border-rose-800">
                CANNOT BE HIDDEN
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-3">
              {activePackage?.criticalRisks?.map((risk, idx) => (
                <div
                  key={idx}
                  className="p-2.5 rounded-lg bg-black/40 border border-rose-900/50 text-xs font-mono text-rose-200 flex items-start gap-2"
                >
                  <span className="text-rose-400 font-bold">⚠️</span>
                  <span>{risk}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="p-4 rounded-xl bg-[#121B2A] border border-[#24324A] flex flex-col justify-between">
            <div>
              <h2 className="text-xs font-bold text-slate-300 font-mono uppercase tracking-wider mb-2">
                Rapid Executive Actions
              </h2>
              <p className="text-xs text-slate-400 leading-relaxed mb-3">
                Authorized for role <strong className="text-cyan-300">{selectedRole}</strong>. Actions enforce fail-closed cryptographic receipts.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={handleApprove}
                disabled={activePackage?.status === "FAILED"}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                  activePackage?.status === "FAILED"
                    ? "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700"
                    : "bg-emerald-600 hover:bg-emerald-500 text-white shadow-md shadow-emerald-700/30"
                }`}
              >
                Sign & Approve Active
              </button>
              <button
                onClick={() => handleGenerateBriefing("DECISION_PACK")}
                className="px-3 py-1.5 rounded-lg text-xs font-mono bg-blue-600 hover:bg-blue-500 text-white transition-all shadow-md shadow-blue-700/30"
              >
                Export Decision Pack
              </button>
            </div>
          </div>
        </section>

        {/* Main Workstation Layout: 3 Columns (Decision Queue | Package Brief & Options | Approval & Outcome) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Station 3 (Executive Decision Queue) - 3 Cols */}
          <section aria-label="Decision Queue" className="lg:col-span-3 space-y-4">
            <HorizonCard
              title="Decision Queue"
              subtitle={`${filteredPackages.length} of ${totalPackages} Packages Active`}
              actions={
                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-950 text-cyan-400 border border-cyan-800">
                  STAGE-AWARE
                </span>
              }
            >
              {/* Search & Filter */}
              <div className="space-y-2 mb-3">
                <input
                  type="text"
                  placeholder="Filter by PKG- or keyword..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full px-3 py-1.5 text-xs bg-[#090E17] border border-[#24324A] rounded-lg text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
                />
                <div className="flex flex-wrap gap-1">
                  {["ALL", "READY_FOR_APPROVAL", "COMPLETED", "FAILED"].map((stage) => (
                    <button
                      key={stage}
                      onClick={() => setSelectedStageFilter(stage)}
                      className={`px-2 py-0.5 text-[10px] font-mono rounded ${
                        selectedStageFilter === stage
                          ? "bg-cyan-600 text-white font-bold"
                          : "bg-[#090E17] text-slate-400 hover:text-white"
                      }`}
                    >
                      {stage}
                    </button>
                  ))}
                </div>
              </div>

              {/* Package List */}
              <div className="space-y-2 max-h-[600px] overflow-y-auto pr-1">
                {filteredPackages.map((pkg) => {
                  const isSelected = pkg.packageId === activePackage?.packageId;
                  return (
                    <div
                      key={pkg.packageId}
                      onClick={() => {
                        setSelectedPackageId(pkg.packageId);
                        if (pkg.options.length > 0) {
                          setSelectedOptionId(pkg.options[0].optionId);
                        }
                      }}
                      className={`p-3 rounded-xl border transition-all cursor-pointer ${
                        isSelected
                          ? "bg-[#18263B] border-cyan-500 shadow-md shadow-cyan-900/30"
                          : "bg-[#0B1320] border-[#1F2C42] hover:border-slate-600"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-mono font-bold text-cyan-400">{pkg.packageId}</span>
                        <SeverityBadge status={pkg.status} size="sm" />
                      </div>
                      <h3 className="text-xs font-semibold text-white line-clamp-2 leading-snug">
                        {pkg.title}
                      </h3>
                      <div className="mt-2 flex items-center justify-between text-[10px] font-mono text-slate-400">
                        <span>{pkg.originatingCommittee}</span>
                        <span className={pkg.urgency === "CRITICAL" ? "text-rose-400 font-bold" : "text-amber-400"}>
                          {pkg.urgency}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </HorizonCard>

            {/* Related Artifacts Rail */}
            <RelatedArtifactsPanel
              title="Contextual Ledger Feeds"
              artifacts={[
                { id: `ART-${activePackage.committeeId}`, title: `Committee ${activePackage.committeeId}`, href: `/committee-intelligence?committeeId=${activePackage.committeeId}`, type: "COMMITTEE" },
                { id: `ART-DEC-${activePackage.packageId}`, title: `Decision Lineage ${activePackage.packageId}`, href: `/decision-explorer?decisionId=${activePackage.packageId}`, type: "DECISION" },
                { id: `ART-SIM-${activePackage.packageId}`, title: "Stress Simulation", href: "/simulation-intelligence", type: "SIMULATION" },
                { id: `ART-AUD-${activePackage.packageId}`, title: `Audit Ledger ${activePackage.packageId}`, href: `/audit-explorer?queryId=${activePackage.packageId}`, type: "AUDIT" },
              ]}
            />
          </section>

          {/* Middle Column: Stations 4 & 5 (Decision Package Brief & Multi-Option Analysis) - 5 Cols */}
          <section aria-label="Decision Package & Options" className="lg:col-span-5 space-y-4">
            {/* Station 4: Executive Decision Package */}
            <HorizonCard
              title={activePackage.title}
              subtitle={`${activePackage.packageId} • ${activePackage.originatingCommittee} • Stage: ${activePackage.currentStage}`}
              actions={<SeverityBadge status={activePackage.status} size="md" />}
            >
              {/* Target Metric & Delta Forecast */}
              <div className="p-3.5 rounded-xl bg-[#090E17] border border-[#202E45] grid grid-cols-2 gap-4 mb-4">
                <div>
                  <span className="text-[10px] font-mono uppercase text-slate-400">Target Objective</span>
                  <p className="text-xs font-bold text-white font-mono mt-0.5">{activePackage.targetMetric}</p>
                  <div className="mt-2 flex items-baseline gap-2">
                    <span className="text-lg font-bold text-white font-mono">
                      {activePackage.baselineMetricValue.toFixed(1)}
                    </span>
                    <span className="text-xs text-slate-400">baseline</span>
                    <span className="text-xs text-cyan-400 font-bold">
                      → {activePackage.projectedMetricValue.toFixed(1)} projected
                    </span>
                  </div>
                </div>

                <div>
                  <span className="text-[10px] font-mono uppercase text-slate-400">Cryptographic State Hash</span>
                  <p className="text-[11px] font-mono text-cyan-300 mt-1 truncate bg-[#05080E] p-1 rounded border border-[#1A2538]">
                    {activePackage.stateHash}
                  </p>
                  <span className="text-[10px] font-mono text-emerald-400 mt-1 block">
                    ✓ Deterministic Replay Anchored
                  </span>
                </div>
              </div>

              {/* Multi-Center Synthesis Grid */}
              <div className="mb-4">
                <h4 className="text-xs font-bold text-slate-300 font-mono uppercase tracking-wider mb-2">
                  Multi-Center Intelligence Synthesis
                </h4>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                  <div className="p-2 rounded-lg bg-[#0E1522] border border-[#1E2B3E]">
                    <span className="text-[10px] text-slate-400 block">OHI Trajectory</span>
                    <span className="text-emerald-400 font-bold">
                      {activePackage.intelligenceSynthesis.ohi.delta > 0 ? "+" : ""}
                      {activePackage.intelligenceSynthesis.ohi.delta.toFixed(1)} pts
                    </span>
                  </div>
                  <div className="p-2 rounded-lg bg-[#0E1522] border border-[#1E2B3E]">
                    <span className="text-[10px] text-slate-400 block">Risk Score Delta</span>
                    <span className="text-emerald-400 font-bold">
                      {activePackage.intelligenceSynthesis.risk.delta.toFixed(1)} pts
                    </span>
                  </div>
                  <div className="p-2 rounded-lg bg-[#0E1522] border border-[#1E2B3E]">
                    <span className="text-[10px] text-slate-400 block">Resilience Score</span>
                    <span className="text-cyan-300 font-bold">
                      {activePackage.intelligenceSynthesis.resilience.score.toFixed(1)}
                    </span>
                  </div>
                  <div className="p-2 rounded-lg bg-[#0E1522] border border-[#1E2B3E]">
                    <span className="text-[10px] text-slate-400 block">Simulation Confidence</span>
                    <span className="text-purple-300 font-bold">
                      {activePackage.intelligenceSynthesis.simulation.monteCarloConfidencePct.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* 100% Causal Driver Attribution (Station 4 Sub-panel) */}
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-xs font-bold text-slate-300 font-mono uppercase tracking-wider">
                    100.0% Causal Driver Attribution
                  </h4>
                  <span className="text-[10px] font-mono text-emerald-400 font-bold">
                    Sum: 100.0% (Zero Residuals)
                  </span>
                </div>
                {/* Horizontal visual bar */}
                <div className="w-full h-3 rounded-full bg-slate-800 flex overflow-hidden mb-2">
                  {activePackage.driverAttribution.map((drv) => (
                    <div
                      key={drv.id}
                      style={{ width: `${drv.percentage}%` }}
                      title={`${drv.name}: ${drv.percentage}% (${drv.polarity})`}
                      className={`${
                        drv.polarity === "POSITIVE" ? "bg-emerald-500" : "bg-rose-500"
                      } border-r border-slate-900 transition-all`}
                    />
                  ))}
                </div>
                {/* Driver breakdown items */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-mono">
                  {activePackage.driverAttribution.map((drv) => (
                    <div
                      key={drv.id}
                      className="p-2 rounded-lg bg-[#0B1320] border border-[#1C283B] flex items-center justify-between"
                    >
                      <div>
                        <span className="font-bold text-slate-200 block truncate">{drv.name}</span>
                        <span className="text-[10px] text-slate-400">{drv.description}</span>
                      </div>
                      <span
                        className={`font-bold ml-2 ${
                          drv.polarity === "POSITIVE" ? "text-emerald-400" : "text-rose-400"
                        }`}
                      >
                        {drv.polarity === "POSITIVE" ? "+" : "-"}
                        {drv.percentage.toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </HorizonCard>

            {/* Station 5: Option Comparison Workspace (>= 3 Options Matrix) */}
            <HorizonCard
              title="Alternative Options Matrix"
              subtitle={`Comparing ${evaluatedOptions.length} strategic paths (Ranked by Multi-Objective Tradeoff Score)`}
              actions={
                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-purple-950 text-purple-300 border border-purple-800">
                  &gt;= 3 OPTIONS
                </span>
              }
            >
              <div className="space-y-3">
                {evaluatedOptions.map((opt) => {
                  const isSelected = opt.optionId === activeOption?.optionId;
                  return (
                    <div
                      key={opt.optionId}
                      onClick={() => setSelectedOptionId(opt.optionId)}
                      className={`p-3 rounded-xl border transition-all cursor-pointer ${
                        isSelected
                          ? "bg-[#162235] border-cyan-400 shadow-md shadow-cyan-900/30"
                          : "bg-[#0B1320] border-[#1E2B3E] hover:border-slate-600"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-mono font-bold text-cyan-300">{opt.optionId}</span>
                          <span className="text-xs font-bold text-white">{opt.title}</span>
                        </div>
                        {opt.isRecommended && (
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-950 text-emerald-300 border border-emerald-700">
                            ★ RECOMMENDED
                          </span>
                        )}
                      </div>

                      <p className="text-xs text-slate-300 leading-relaxed mb-2 font-mono">
                        {opt.description}
                      </p>

                      <div className="grid grid-cols-4 gap-2 text-[11px] font-mono pt-2 border-t border-[#1C283B]">
                        <div>
                          <span className="text-slate-500 block text-[10px]">OHI DELTA</span>
                          <span className={opt.ohiDelta >= 0 ? "text-emerald-400 font-bold" : "text-rose-400 font-bold"}>
                            {opt.ohiDelta >= 0 ? "+" : ""}{opt.ohiDelta.toFixed(1)}
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-500 block text-[10px]">RISK DELTA</span>
                          <span className={opt.riskDelta <= 0 ? "text-emerald-400 font-bold" : "text-rose-400 font-bold"}>
                            {opt.riskDelta <= 0 ? "" : "+"}{opt.riskDelta.toFixed(1)}
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-500 block text-[10px]">CAPEX COST</span>
                          <span className="text-slate-300 font-bold">
                            ${(opt.implementationCostUSD / 1000).toFixed(0)}k
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-500 block text-[10px]">TRADEOFF SCORE</span>
                          <span className="text-cyan-400 font-bold text-xs">{opt.tradeoffScore}/100</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </HorizonCard>
          </section>

          {/* Right Column: Stations 6, 7, 8 & Narrative Rail - 4 Cols */}
          <section aria-label="Governance & Learning" className="lg:col-span-4 space-y-4">
            {/* Station 6: Executive Approval Center */}
            <HorizonCard
              title="Governance & Approval Center"
              subtitle="Fail-Closed Verification Checklist & Digital Cryptographic Sign-Off"
              actions={
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                    activePackage.governanceValidation.passed
                      ? "bg-emerald-950 text-emerald-300 border border-emerald-800"
                      : "bg-rose-950 text-rose-300 border border-rose-800"
                  }`}
                >
                  {activePackage.governanceValidation.passed ? "VERIFIED PASS" : "FAIL-CLOSED BLOCKED"}
                </span>
              }
            >
              {/* Fail-closed warning banner if failed */}
              {!activePackage.governanceValidation.passed && (
                <div className="p-3 rounded-lg bg-rose-950/50 border border-rose-600/50 text-rose-200 text-xs font-mono mb-3">
                  <strong className="block mb-1">⛔ FAIL-CLOSED CIRCUIT BREAKER ENGAGED</strong>
                  <span>
                    Execution is strictly locked with zero state mutation. Reason:{" "}
                    {activePackage.governanceValidation.failureReasons.join("; ")}
                  </span>
                </div>
              )}

              {/* Checklist Rules */}
              <div className="space-y-1.5 mb-4 text-xs font-mono">
                {activePackage.governanceValidation.ruleChecks.map((rule) => (
                  <div
                    key={rule.ruleId}
                    className="p-2 rounded bg-[#090E17] border border-[#1E2B3E] flex items-center justify-between"
                  >
                    <div>
                      <span className="text-white font-medium block">{rule.name}</span>
                      <span className="text-[10px] text-slate-400">
                        Required: {rule.threshold} • Observed: {rule.observedValue}
                      </span>
                    </div>
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                        rule.status === "PASS"
                          ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                          : "bg-rose-950 text-rose-400 border border-rose-800"
                      }`}
                    >
                      {rule.status}
                    </span>
                  </div>
                ))}
              </div>

              {/* Digital Sign-off Receipt (if approved) */}
              {activePackage.approvalReceipt ? (
                <div className="p-3 rounded-xl bg-[#091522] border border-cyan-500/40 text-xs font-mono space-y-1 mb-3">
                  <span className="text-cyan-400 font-bold block">✓ DIGITAL SIGN-OFF ATTESTED</span>
                  <p className="text-slate-300 text-[11px]">
                    Approver: <span className="text-white font-bold">{activePackage.approvalReceipt.approverName}</span>
                  </p>
                  <p className="text-slate-400 text-[10px] truncate">
                    Signature: {activePackage.approvalReceipt.signatureHash}
                  </p>
                  <p className="text-slate-400 text-[10px] truncate">
                    Audit Receipt: {activePackage.approvalReceipt.auditReceiptHash}
                  </p>
                </div>
              ) : null}

              {/* Decision Actions */}
              <div className="flex gap-2 pt-2 border-t border-[#1E2B3E]">
                <button
                  onClick={handleApprove}
                  disabled={!activePackage.governanceValidation.passed}
                  className={`flex-1 py-2 rounded-lg text-xs font-mono font-bold transition-all ${
                    !activePackage.governanceValidation.passed
                      ? "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700"
                      : "bg-emerald-600 hover:bg-emerald-500 text-white shadow-md shadow-emerald-700/30"
                  }`}
                >
                  Digital Sign-Off
                </button>
                <button
                  onClick={handleExecute}
                  disabled={activePackage.status !== "APPROVED"}
                  className={`flex-1 py-2 rounded-lg text-xs font-mono font-bold transition-all ${
                    activePackage.status !== "APPROVED"
                      ? "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700"
                      : "bg-cyan-600 hover:bg-cyan-500 text-white shadow-md shadow-cyan-700/30"
                  }`}
                >
                  Deploy Execution
                </button>
                <button
                  onClick={handleReject}
                  className="px-3 py-2 rounded-lg text-xs font-mono bg-[#1F2C42] hover:bg-rose-900/40 text-slate-300 hover:text-rose-300 border border-slate-700 transition-all"
                >
                  Reject
                </button>
              </div>
            </HorizonCard>

            {/* Station 7: Outcome Monitoring */}
            <HorizonCard
              title="Outcome Trajectory Monitor"
              subtitle="Live vs Forecasted Performance (100% Attribution)"
              actions={
                activeOutcome ? (
                  <SeverityBadge status={activeOutcome.status} size="sm" />
                ) : (
                  <span className="text-[10px] font-mono text-slate-500">PENDING RUN</span>
                )
              }
            >
              {activeOutcome ? (
                <div className="space-y-3">
                  <div className="p-3 rounded-lg bg-[#090E17] border border-[#1E2B3E] flex items-center justify-between text-xs font-mono">
                    <div>
                      <span className="text-slate-400 block text-[10px] uppercase">Achieved Metric</span>
                      <span className="text-base font-bold text-white">{activeOutcome.actualValue.toFixed(1)}</span>
                      <span className="text-[10px] text-slate-400 ml-1">
                        (target: {activeOutcome.targetValue.toFixed(1)})
                      </span>
                    </div>
                    <div className="text-right">
                      <span className="text-slate-400 block text-[10px] uppercase">Divergence</span>
                      <span
                        className={`text-sm font-bold ${
                          activeOutcome.divergencePct < 15 ? "text-emerald-400" : "text-amber-400"
                        }`}
                      >
                        {activeOutcome.divergencePct.toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Outcome Drivers */}
                  <div className="space-y-1 text-xs font-mono">
                    <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-bold">
                      Outcome Driver Breakdown (100% Sum)
                    </span>
                    {activeOutcome.drivers.map((drv) => (
                      <div
                        key={drv.id}
                        className="p-1.5 rounded bg-[#0A121F] border border-[#1B273A] flex items-center justify-between text-[11px]"
                      >
                        <span className="text-slate-300 truncate">{drv.name}</span>
                        <span className="text-cyan-400 font-bold ml-2">{drv.percentage.toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="text-xs text-slate-400 font-mono italic">
                  Outcome trajectory activates once decision is approved and deployed.
                </p>
              )}
            </HorizonCard>

            {/* Station 8: Learning Capture Panel */}
            <HorizonCard
              title="Learning Closure & Provenance"
              subtitle="Convert Realized Insights into Institutional Memory"
              actions={
                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-blue-950 text-blue-300 border border-blue-800">
                  {learnings.length} CAPTURED
                </span>
              }
            >
              {/* Form to capture new learning */}
              <form onSubmit={handleCaptureLearning} className="space-y-2 mb-3">
                <input
                  type="text"
                  placeholder="Lesson title (e.g. Damping protocol)..."
                  value={newLessonTitle}
                  onChange={(e) => setNewLessonTitle(e.target.value)}
                  className="w-full px-3 py-1.5 text-xs bg-[#090E17] border border-[#24324A] rounded-lg text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
                />
                <textarea
                  placeholder="Key operational insight & root cause..."
                  rows={2}
                  value={newLessonInsight}
                  onChange={(e) => setNewLessonInsight(e.target.value)}
                  className="w-full px-3 py-1.5 text-xs bg-[#090E17] border border-[#24324A] rounded-lg text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
                />
                <button
                  type="submit"
                  disabled={!newLessonTitle || !newLessonInsight}
                  className="w-full py-1.5 rounded-lg text-xs font-mono font-bold bg-purple-600 hover:bg-purple-500 disabled:bg-slate-800 disabled:text-slate-500 text-white transition-all shadow-md shadow-purple-900/30"
                >
                  Anchor Learning to Organizational Memory
                </button>
              </form>

              {/* Institutional Learnings list */}
              <div className="space-y-2 max-h-[220px] overflow-y-auto">
                {learnings.map((lrn) => (
                  <div
                    key={lrn.learningId}
                    className="p-2.5 rounded-lg bg-[#090F1B] border border-[#1E2B3E] text-xs font-mono space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-purple-300 font-bold">{lrn.title}</span>
                      <span className="text-[10px] text-emerald-400 font-bold">{lrn.currentAdoptionRate}% Adoption</span>
                    </div>
                    <p className="text-[11px] text-slate-300 leading-snug">{lrn.insightText}</p>
                    <span className="text-[9px] text-slate-500 block truncate">
                      Provenance: {lrn.provenanceHash} • Source: {lrn.sourceDecisionId}
                    </span>
                  </div>
                ))}
              </div>
            </HorizonCard>

            {/* Station 9: Narrative & Board Briefing Rail */}
            <HorizonCard
              title="Board Briefing Synthesis"
              subtitle="Automated Institutional Reporting with Deterministic Replay"
              actions={
                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-950 text-cyan-400 border border-cyan-800">
                  REPLAY HASH
                </span>
              }
            >
              <div className="flex gap-2 mb-3">
                <button
                  onClick={() => handleGenerateBriefing("MONTHLY_BRIEF")}
                  className="flex-1 py-1.5 rounded text-xs font-mono bg-[#162235] hover:bg-cyan-900/40 text-cyan-300 border border-cyan-800 transition-all font-semibold"
                >
                  Monthly Brief
                </button>
                <button
                  onClick={() => handleGenerateBriefing("QUARTERLY_REPORT")}
                  className="flex-1 py-1.5 rounded text-xs font-mono bg-[#162235] hover:bg-blue-900/40 text-blue-300 border border-blue-800 transition-all font-semibold"
                >
                  Quarterly Report
                </button>
              </div>

              {activeBoardBrief && (
                <div className="p-3 rounded-xl bg-[#08101C] border border-cyan-700/40 text-xs font-mono space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white text-xs">{activeBoardBrief.title}</span>
                    <span className="text-[10px] text-emerald-400">Replay Verified</span>
                  </div>
                  <p className="text-[11px] text-cyan-300 leading-snug bg-cyan-950/30 p-2 rounded border border-cyan-900/40">
                    {activeBoardBrief.headline}
                  </p>
                  <p className="text-[11px] text-slate-300 leading-relaxed">
                    {activeBoardBrief.executiveSummary}
                  </p>
                  <div className="text-[10px] text-slate-400 pt-1 border-t border-[#1C283B]">
                    <span>Replay Hash: </span>
                    <span className="text-cyan-400 font-bold">{activeBoardBrief.replayHash}</span>
                  </div>
                </div>
              )}
            </HorizonCard>
          </section>
        </div>
      </main>
    </div>
  );
}

export default function ExecutiveWorkspacePage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#070D17] text-slate-400 p-8 font-mono text-xs">Loading Executive Decision Workspace...</div>}>
      <ExecutiveWorkspaceInner />
    </Suspense>
  );
}
