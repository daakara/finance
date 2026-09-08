"use client";

import { useState, useMemo, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import ExecutiveIntelligenceNav from "../../components/committee/ExecutiveIntelligenceNav";
import RelatedArtifactsCard from "../../components/committee/RelatedArtifactsCard";
import {
  calculateOHI,
  CANONICAL_OHI_INPUTS,
  CANONICAL_ORGANIZATIONAL_HEALTH_INDEX,
} from "../../lib/oos/organizationalHealthEngine";
import {
  getUnifiedTelemetrySnapshot,
  getSystemHealth,
  getEnterpriseAlerts,
} from "../../lib/oos/organizationalTelemetryHub";
import {
  generateExecutiveReport,
  CANONICAL_EXECUTIVE_REPORT,
} from "../../lib/oos/executiveReportEngine";
import {
  verifyCrossSystemEquality,
  executeCSCRecovery,
  getCSCRecoveryAudit,
  CANONICAL_CROSS_SYSTEM_SOURCES,
} from "../../lib/oos/consistencyVerificationEngine";
import {
  getOrganizationalStateByMode,
  getStateTransitionForecast,
} from "../../lib/oos/organizationalStateEngine";

type OOSTab = "OVERVIEW" | "EXECUTIVE_DASHBOARD" | "BOARD_REPORT" | "CONSISTENCY_CENTER";

function OOSContent() {
  const searchParams = useSearchParams();
  const initialTab = (searchParams.get("view")?.toUpperCase() as OOSTab) || "OVERVIEW";
  const [activeTab, setActiveTab] = useState<OOSTab>(
    ["OVERVIEW", "EXECUTIVE_DASHBOARD", "BOARD_REPORT", "CONSISTENCY_CENTER"].includes(initialTab)
      ? initialTab
      : "OVERVIEW"
  );

  // Core OOS Engine State
  const ohiResult = useMemo(() => calculateOHI(CANONICAL_OHI_INPUTS), []);
  const snapshot = useMemo(() => getUnifiedTelemetrySnapshot(), []);
  const systemHealth = useMemo(() => getSystemHealth(), []);
  const alerts = useMemo(() => getEnterpriseAlerts(), []);
  const report = useMemo(() => CANONICAL_EXECUTIVE_REPORT, []);
  const consistencyResult = useMemo(() => verifyCrossSystemEquality(), []);
  const forecast90D = useMemo(() => getStateTransitionForecast("ST-CURRENT", "90D"), []);
  const forecast365D = useMemo(() => getStateTransitionForecast("ST-CURRENT", "365D"), []);

  // CSC Sandbox State
  const [cscRecoveries, setCscRecoveries] = useState(getCSCRecoveryAudit());
  const [simulatedVariance, setSimulatedVariance] = useState<number | null>(null);

  const handleSimulateVariance = () => {
    setSimulatedVariance(0.4);
  };

  const handleTriggerCSCRecovery = () => {
    executeCSCRecovery({
      recoveryId: `REC-CSC-${Date.now().toString().slice(-4)}`,
      validationErrorCode: "CROSS_SYSTEM_VARIANCE_DETECTED",
      affectedDrivers: ["KT", "LV"],
      initiatedAtUtc: new Date().toISOString(),
      recoveryMode: "AUTO_REPAIR",
      actorId: "EXEC-OOS-USER",
    });
    setSimulatedVariance(null);
    setCscRecoveries([...getCSCRecoveryAudit()]);
  };

  return (
    <div className="min-h-screen bg-[#0c1017] text-slate-100 font-sans">
      <ExecutiveIntelligenceNav badgeText="10/10 OOS GATES CERTIFIED" />

      <main className="max-w-[1750px] mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Header Ribbon */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#1f2c42] pb-5">
          <div>
            <div className="flex items-center space-x-2">
              <span className="px-2 py-0.5 rounded text-[10px] font-mono tracking-wider font-semibold bg-cyan-950/80 text-cyan-400 border border-cyan-800">
                PHASE 31-M6
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono tracking-wider font-semibold bg-emerald-950/80 text-emerald-400 border border-emerald-800">
                OOS CONTROL PLANE
              </span>
              <span className="text-xs text-slate-400 font-mono">
                INV-OI33 — INV-OI38 CERTIFIED
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white mt-1">
              Organizational Operating System (OOS)
            </h1>
            <p className="text-sm text-slate-400 mt-1 max-w-3xl">
              Master control plane unifying Committee, Network, Learning, Risk, and Prescriptive Intelligence into one certified health index and executive operating model.
            </p>
          </div>

          <div className="flex items-center space-x-3">
            <div className="text-right">
              <div className="text-[11px] font-mono text-slate-400 uppercase">Current Status</div>
              <div className="text-sm font-semibold text-emerald-400 flex items-center justify-end space-x-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span>OPERATING SYSTEM ACTIVE</span>
              </div>
            </div>
          </div>
        </div>

        {/* 4 Header KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-[#111724] border border-[#1f2c42] p-4 rounded-xl relative overflow-hidden">
            <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
              Organizational Health Index (OHI)
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-3xl font-bold font-mono text-cyan-400">
                {ohiResult.score.toFixed(1)}
              </span>
              <span className="text-xs font-mono text-emerald-400 font-medium">
                / 100.0 (OPTIMAL)
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Weighted composite of 7 invariant governance drivers
            </p>
            <div className="mt-3 w-full bg-[#182335] rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-cyan-500 h-1.5 rounded-full transition-all"
                style={{ width: `${ohiResult.score}%` }}
              />
            </div>
          </div>

          <div className="bg-[#111724] border border-[#1f2c42] p-4 rounded-xl relative overflow-hidden">
            <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
              90D Projected OHI
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-3xl font-bold font-mono text-emerald-400">
                {forecast90D.projectedOHI.toFixed(1)}
              </span>
              <span className="text-xs font-mono text-emerald-400">
                (+{forecast90D.projectedDelta.toFixed(1)} pts)
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Confidence: {forecast90D.transitionProbabilityPct}% (±1.5 pts band)
            </p>
            <div className="mt-3 w-full bg-[#182335] rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-emerald-500 h-1.5 rounded-full"
                style={{ width: `${forecast90D.projectedOHI}%` }}
              />
            </div>
          </div>

          <div className="bg-[#111724] border border-[#1f2c42] p-4 rounded-xl relative overflow-hidden">
            <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
              Critical Risks Remediation
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-3xl font-bold font-mono text-amber-400">
                {snapshot.openCriticalRisks}
              </span>
              <span className="text-xs font-mono text-emerald-400 font-medium">
                (100% Remediated)
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              {snapshot.activeRecommendations} prescriptive interventions active
            </p>
            <div className="mt-3 w-full bg-[#182335] rounded-full h-1.5 overflow-hidden">
              <div className="bg-amber-500 h-1.5 rounded-full" style={{ width: "100%" }} />
            </div>
          </div>

          <div className="bg-[#111724] border border-[#1f2c42] p-4 rounded-xl relative overflow-hidden">
            <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
              Coach Impact Ratio
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-3xl font-bold font-mono text-purple-400">
                +{snapshot.coachImpactRatio.toFixed(2)}x
              </span>
              <span className="text-xs font-mono text-slate-400">
                Net Deliberative Gain
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Learning Velocity: +{snapshot.learningVelocity.toFixed(1)} pts/sprint
            </p>
            <div className="mt-3 w-full bg-[#182335] rounded-full h-1.5 overflow-hidden">
              <div className="bg-purple-500 h-1.5 rounded-full" style={{ width: "88%" }} />
            </div>
          </div>
        </div>

        {/* Diagnostic Tabs */}
        <div className="border-b border-[#1f2c42]">
          <nav className="flex space-x-2 sm:space-x-4 overflow-x-auto">
            {[
              { id: "OVERVIEW", label: "Health Overview", badge: "7 Drivers" },
              { id: "EXECUTIVE_DASHBOARD", label: "Executive Dashboard", badge: "Forecasts" },
              { id: "BOARD_REPORT", label: "Board Report", badge: "INV-OI36" },
              { id: "CONSISTENCY_CENTER", label: "Consistency Center", badge: "Zero-Variance" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as OOSTab)}
                className={`py-3 px-3 text-xs sm:text-sm font-medium border-b-2 flex items-center space-x-2 whitespace-nowrap transition-colors ${
                  activeTab === tab.id
                    ? "border-cyan-400 text-cyan-400 font-semibold"
                    : "border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700"
                }`}
              >
                <span>{tab.label}</span>
                <span className={`text-[10px] font-mono px-1.5 py-0.2 rounded ${
                  activeTab === tab.id ? "bg-cyan-950 text-cyan-300 border border-cyan-700" : "bg-[#182335] text-slate-400"
                }`}>
                  {tab.badge}
                </span>
              </button>
            ))}
          </nav>
        </div>

        {/* Tab 1: Overview */}
        {activeTab === "OVERVIEW" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* OHI Driver Breakdown */}
              <div className="lg:col-span-2 bg-[#111724] border border-[#1f2c42] rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-[#1f2c42] pb-3">
                  <div>
                    <h3 className="text-base font-bold text-slate-100">
                      Organizational Health Index (OHI) Drivers
                    </h3>
                    <p className="text-xs text-slate-400">
                      Formal formula: OHI = 0.25 ODEI + 0.20 CDQI + 0.15 DIRatio + 0.15 LV + 0.10 KT + 0.10 GTR + 0.05 RH
                    </p>
                  </div>
                  <span className="px-2.5 py-1 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 text-xs font-mono font-semibold">
                    INV-OI33 CERTIFIED
                  </span>
                </div>

                <div className="space-y-3">
                  {ohiResult.drivers.map((driver) => (
                    <div
                      key={driver.driverId}
                      className="p-3.5 bg-[#0c1017] border border-[#1f2c42] rounded-lg hover:border-slate-700 transition-all"
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <div className="flex items-center space-x-2">
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-[#182335] text-cyan-300 border border-cyan-800">
                            {driver.driverId}
                          </span>
                          <span className="text-sm font-semibold text-slate-200">
                            {driver.name}
                          </span>
                          <span className="text-[11px] font-mono text-slate-400">
                            (Weight: {(driver.weight * 100).toFixed(0)}%)
                          </span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <span className="text-sm font-bold font-mono text-cyan-400">
                            {driver.normalizedValue.toFixed(1)} / 100
                          </span>
                          <span className="px-2 py-0.5 text-[10px] font-mono font-semibold rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                            {driver.trend}
                          </span>
                        </div>
                      </div>
                      <p className="text-xs text-slate-400">{driver.explanation}</p>
                      <div className="mt-2 w-full bg-[#182335] rounded-full h-1 overflow-hidden">
                        <div
                          className="bg-cyan-400 h-1 rounded-full"
                          style={{ width: `${driver.normalizedValue}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Subsystems & System Health */}
              <div className="bg-[#111724] border border-[#1f2c42] rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-[#1f2c42] pb-3">
                  <h3 className="text-base font-bold text-slate-100">Subsystem Health Status</h3>
                  <span className="text-xs font-mono text-emerald-400">
                    {systemHealth.healthySubsystems}/{systemHealth.activeSubsystems} ONLINE
                  </span>
                </div>

                <div className="space-y-3">
                  {Object.entries(systemHealth.subsystemStatuses).map(([subsystem, meta]) => (
                    <div
                      key={subsystem}
                      className="p-3 bg-[#0c1017] border border-[#1f2c42] rounded-lg flex items-center justify-between"
                    >
                      <div>
                        <div className="text-xs font-semibold text-slate-200">
                          {subsystem.replace(/_/g, " ")}
                        </div>
                        <div className="text-[10px] font-mono text-slate-500">
                          Ping: {new Date(meta.lastPingUtc).toLocaleTimeString()}
                        </div>
                      </div>
                      <div className="text-right">
                        <span className="px-2 py-0.5 text-[10px] font-mono font-semibold rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                          {meta.status}
                        </span>
                        <div className="text-[11px] font-mono text-cyan-400 mt-1">
                          Score: {meta.score.toFixed(1)}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Enterprise Alert NOC Feed */}
                <div className="pt-2 border-t border-[#1f2c42]">
                  <h4 className="text-xs font-mono uppercase text-slate-400 mb-2 font-semibold">
                    Enterprise Alert NOC Feed ({alerts.length})
                  </h4>
                  <div className="space-y-2">
                    {alerts.slice(0, 3).map((a) => (
                      <div
                        key={a.alertId}
                        className="p-2.5 bg-[#0c1017] border border-[#1f2c42] rounded text-xs space-y-1"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-slate-200">{a.title}</span>
                          <span
                            className={`px-1.5 py-0.2 text-[9px] font-mono rounded ${
                              a.severity === "CRITICAL"
                                ? "bg-rose-950 text-rose-300 border border-rose-800"
                                : a.severity === "HIGH"
                                ? "bg-amber-950 text-amber-300 border border-amber-800"
                                : "bg-cyan-950 text-cyan-300 border border-cyan-800"
                            }`}
                          >
                            {a.severity}
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-400">{a.description}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Universal Cross-Linking */}
            <RelatedArtifactsCard
              entityId="OHI-001"
              title="Master Organizational Health Index"
            />
          </div>
        )}

        {/* Tab 2: Executive Dashboard */}
        {activeTab === "EXECUTIVE_DASHBOARD" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Multi-Horizon Forecast Curves */}
              <div className="bg-[#111724] border border-[#1f2c42] rounded-xl p-5 space-y-4">
                <div className="border-b border-[#1f2c42] pb-3 flex items-center justify-between">
                  <div>
                    <h3 className="text-base font-bold text-slate-100">
                      Multi-Horizon OHI Forecast Trajectories
                    </h3>
                    <p className="text-xs text-slate-400">
                      Predictive state transitions: Current (84.2) → 365D (90.2)
                    </p>
                  </div>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-950 text-emerald-400 border border-emerald-800">
                    INV-OI37 DETERMINISTIC
                  </span>
                </div>

                <div className="space-y-3">
                  {[
                    { horizon: "30D", delta: "+0.9", ohi: 85.1, prob: 92, band: "±0.8" },
                    { horizon: "90D", delta: "+2.6", ohi: 86.8, prob: 88, band: "±1.5" },
                    { horizon: "180D", delta: "+4.3", ohi: 88.5, prob: 80, band: "±2.4" },
                    { horizon: "365D", delta: "+6.0", ohi: 90.2, prob: 72, band: "±3.8" },
                  ].map((f) => (
                    <div
                      key={f.horizon}
                      className="p-3.5 bg-[#0c1017] border border-[#1f2c42] rounded-lg flex items-center justify-between"
                    >
                      <div className="flex items-center space-x-3">
                        <span className="px-2 py-1 rounded bg-[#182335] text-cyan-300 font-mono font-bold text-xs">
                          {f.horizon}
                        </span>
                        <div>
                          <div className="text-sm font-semibold text-slate-200">
                            Projected OHI: {f.ohi.toFixed(1)}
                          </div>
                          <div className="text-xs text-slate-400">
                            Confidence Band: {f.band} pts ({f.prob}% confidence)
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className="text-sm font-mono font-bold text-emerald-400">
                          {f.delta} pts
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Committee Quality & Risk Ranking */}
              <div className="bg-[#111724] border border-[#1f2c42] rounded-xl p-5 space-y-4">
                <div className="border-b border-[#1f2c42] pb-3 flex items-center justify-between">
                  <h3 className="text-base font-bold text-slate-100">
                    Committee Deliberative Health Matrix
                  </h3>
                  <span className="text-xs font-mono text-slate-400">3 Active Committees</span>
                </div>

                <div className="space-y-3">
                  {[
                    { id: "COM-001", name: "Investment Committee", odei: 85.0, cdqi: 84.0, risk: "LOW" },
                    { id: "COM-002", name: "Governance Committee", odei: 83.0, cdqi: 82.0, risk: "LOW" },
                    { id: "COM-003", name: "Risk & Capital Committee", odei: 87.0, cdqi: 83.0, risk: "HIGH" },
                  ].map((c) => (
                    <div
                      key={c.id}
                      className="p-3.5 bg-[#0c1017] border border-[#1f2c42] rounded-lg flex items-center justify-between"
                    >
                      <div>
                        <div className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
                          <span>{c.name}</span>
                          <span className="text-[10px] font-mono text-slate-400">({c.id})</span>
                        </div>
                        <div className="text-xs text-slate-400 mt-0.5">
                          ODEI: {c.odei.toFixed(1)} | CDQI: {c.cdqi.toFixed(1)}
                        </div>
                      </div>
                      <span
                        className={`px-2 py-0.5 text-[10px] font-mono font-semibold rounded ${
                          c.risk === "HIGH"
                            ? "bg-rose-950 text-rose-300 border border-rose-800"
                            : "bg-emerald-950 text-emerald-300 border border-emerald-800"
                        }`}
                      >
                        {c.risk} RISK
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <RelatedArtifactsCard
              entityId="REP-OOS-001"
              title="Board of Directors Governance Report"
            />
          </div>
        )}

        {/* Tab 3: Board Report */}
        {activeTab === "BOARD_REPORT" && (
          <div className="space-y-6">
            <div className="bg-[#111724] border border-[#1f2c42] rounded-xl p-6 space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#1f2c42] pb-4">
                <div>
                  <div className="text-xs font-mono text-cyan-400 uppercase tracking-wider">
                    {report.reportingPeriod} Board-Ready Artifact
                  </div>
                  <h2 className="text-xl font-bold text-white mt-1">{report.title}</h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Source Snapshot: {report.sourceSnapshotId} | OHI Score: {report.ohiScore.toFixed(1)}
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="px-2.5 py-1 text-xs font-mono font-semibold rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                    INV-OI36 EXPLAINABLE
                  </span>
                </div>
              </div>

              {/* Executive Summary Quote */}
              <div className="p-4 bg-[#0c1017] border-l-4 border-cyan-400 rounded-r-lg">
                <div className="text-xs font-mono uppercase text-cyan-400 font-semibold mb-1">
                  Executive Summary
                </div>
                <p className="text-sm text-slate-200 leading-relaxed">
                  {report.executiveSummary}
                </p>
              </div>

              {/* Report Sections */}
              <div className="space-y-6">
                {report.sections.map((sec) => (
                  <div key={sec.sectionId} className="space-y-3">
                    <h4 className="text-base font-bold text-slate-100 border-b border-[#1f2c42] pb-1.5">
                      {sec.title}
                    </h4>
                    <p className="text-xs text-slate-400">{sec.summary}</p>

                    {/* Findings Cards */}
                    <div className="space-y-3">
                      {sec.findings.map((f) => (
                        <div
                          key={f.findingId}
                          className="p-4 bg-[#0c1017] border border-[#1f2c42] rounded-lg space-y-2.5"
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="flex items-center space-x-2">
                              <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-[#182335] text-cyan-300 border border-cyan-800">
                                {f.findingId}
                              </span>
                              <h5 className="text-sm font-semibold text-slate-100">{f.finding}</h5>
                            </div>
                            <span
                              className={`px-2 py-0.5 text-[10px] font-mono rounded font-semibold ${
                                f.severity === "HIGH"
                                  ? "bg-rose-950 text-rose-300 border border-rose-800"
                                  : "bg-emerald-950 text-emerald-300 border border-emerald-800"
                              }`}
                            >
                              {f.severity}
                            </span>
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs pt-1 border-t border-[#182335]">
                            <div>
                              <div className="text-[10px] font-mono text-slate-500 uppercase">Supporting Evidence</div>
                              <ul className="list-disc list-inside text-slate-300 mt-1 space-y-0.5">
                                {f.evidence.map((ev, i) => (
                                  <li key={i}>{ev}</li>
                                ))}
                              </ul>
                            </div>
                            <div>
                              <div className="text-[10px] font-mono text-slate-500 uppercase">Risk Evaluation</div>
                              <p className="text-slate-300 mt-1">{f.risk}</p>
                            </div>
                            <div>
                              <div className="text-[10px] font-mono text-slate-500 uppercase">Prescriptive Recommendation</div>
                              <p className="text-cyan-300 mt-1">{f.recommendation}</p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <RelatedArtifactsCard
              entityId="REP-OOS-001"
              title="Board of Directors Governance Report"
            />
          </div>
        )}

        {/* Tab 4: Consistency Center */}
        {activeTab === "CONSISTENCY_CENTER" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Cross-System Consistency Matrix */}
              <div className="bg-[#111724] border border-[#1f2c42] rounded-xl p-5 space-y-4">
                <div className="border-b border-[#1f2c42] pb-3 flex items-center justify-between">
                  <div>
                    <h3 className="text-base font-bold text-slate-100">
                      Cross-System Consistency Matrix (INV-OI35)
                    </h3>
                    <p className="text-xs text-slate-400">
                      Target: Variance = 0.0000 across all 5 operational mirrors
                    </p>
                  </div>
                  <span
                    className={`px-2.5 py-1 text-xs font-mono font-semibold rounded ${
                      simulatedVariance
                        ? "bg-rose-950 text-rose-400 border border-rose-800"
                        : "bg-emerald-950 text-emerald-400 border border-emerald-800"
                    }`}
                  >
                    {simulatedVariance ? "VARIANCE DETECTED" : "VARIANCE = 0.0000"}
                  </span>
                </div>

                <div className="space-y-3">
                  {[
                    { source: "DASHBOARD", value: (84.2 + (simulatedVariance || 0)).toFixed(1) },
                    { source: "API", value: "84.2" },
                    { source: "REPORT", value: "84.2" },
                    { source: "AUDIT RECONSTRUCTION", value: "84.2" },
                    { source: "FORECAST ENGINE", value: "84.2" },
                  ].map((s) => (
                    <div
                      key={s.source}
                      className="p-3 bg-[#0c1017] border border-[#1f2c42] rounded-lg flex items-center justify-between"
                    >
                      <span className="text-xs font-mono font-semibold text-slate-200">
                        {s.source}
                      </span>
                      <div className="flex items-center space-x-3">
                        <span className="text-sm font-mono font-bold text-cyan-400">
                          {s.value}
                        </span>
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                          SYNCED
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="pt-2 flex items-center space-x-3">
                  <button
                    onClick={handleSimulateVariance}
                    className="px-3 py-1.5 bg-rose-950/60 hover:bg-rose-900 border border-rose-800 text-rose-300 text-xs font-mono rounded transition-all"
                  >
                    Simulate Variance (0.4)
                  </button>
                  {simulatedVariance && (
                    <button
                      onClick={handleTriggerCSCRecovery}
                      className="px-3 py-1.5 bg-emerald-950/80 hover:bg-emerald-900 border border-emerald-700 text-emerald-300 text-xs font-mono rounded transition-all font-semibold"
                    >
                      Trigger CSC Recovery (Auto-Repair)
                    </button>
                  )}
                </div>
              </div>

              {/* CSC Recovery Ledger */}
              <div className="bg-[#111724] border border-[#1f2c42] rounded-xl p-5 space-y-4">
                <div className="border-b border-[#1f2c42] pb-3 flex items-center justify-between">
                  <div>
                    <h3 className="text-base font-bold text-slate-100">
                      Certification Self-Correction (CSC) Audit
                    </h3>
                    <p className="text-xs text-slate-400">
                      Immutable recovery record log with SHA-256 state locks
                    </p>
                  </div>
                  <span className="text-xs font-mono text-cyan-400">
                    {cscRecoveries.length} Records Logged
                  </span>
                </div>

                <div className="space-y-3">
                  {cscRecoveries.length === 0 ? (
                    <div className="p-4 bg-[#0c1017] border border-[#1f2c42] rounded-lg text-xs text-slate-400 text-center">
                      Zero variance incidents recorded. System running at 100% synchronization.
                    </div>
                  ) : (
                    cscRecoveries.map((rec) => (
                      <div
                        key={rec.recoveryId}
                        className="p-3 bg-[#0c1017] border border-[#1f2c42] rounded-lg space-y-1.5"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-mono text-xs font-bold text-cyan-300">
                            {rec.recoveryId}
                          </span>
                          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                            RESTORED
                          </span>
                        </div>
                        <div className="text-xs text-slate-300">
                          Trigger: {rec.validationCode} | Repaired By: {rec.repairedBy}
                        </div>
                        <div className="text-[10px] font-mono text-slate-500 truncate">
                          Hash: {rec.afterHash}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>

            <RelatedArtifactsCard
              entityId="REC-CSC-001"
              title="Certification Self-Correction Engine"
            />
          </div>
        )}
      </main>
    </div>
  );
}

export default function OOSPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#0c1017] text-slate-400 p-8">Loading Operating System...</div>}>
      <OOSContent />
    </Suspense>
  );
}
