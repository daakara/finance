"use client";

import { useState, useMemo, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import ExecutiveIntelligenceNav from "../../components/committee/ExecutiveIntelligenceNav";
import RelatedArtifactsCard from "../../components/committee/RelatedArtifactsCard";

import {
  evaluateGroupthinkAssessment,
  verifyINV_OI19,
  verifyINV_OI20,
  verifyINV_OI21,
  COMMITTEE_METRICS_PROFILE,
} from "../../lib/governance/groupthinkDetectionEngine";

import {
  getRiskRegistry,
  calculatePortfolioExposure,
} from "../../lib/governance/riskRegistryEngine";

import {
  computeGovernanceForecast,
  verifyINV_OI22,
  predictIncidentEscalation,
} from "../../lib/governance/governanceForecastEngine";

import type { ForecastPeriod, RiskCategory, RiskSeverity } from "../../types/groupthink-intelligence";

const COMMITTEES = [
  { id: "COM-001", name: "Investment Committee", focus: "Capital Allocation & Sizing" },
  { id: "COM-002", name: "Risk Committee", focus: "Market Invariants & Drawdown" },
  { id: "COM-003", name: "Governance Committee", focus: "Policy & Network Dependencies" },
];

function RisksAndGroupthinkContent() {
  const searchParams = useSearchParams();
  const initialQueryId = searchParams.get("queryId") || "";

  const [selectedCommitteeId, setSelectedCommitteeId] = useState<string>("COM-001");
  const [activeTab, setActiveTab] = useState<"GROUPTHINK" | "REGISTRY" | "FORECAST" | "CONSENSUS" | "INCIDENTS">("GROUPTHINK");
  const [forecastHorizon, setForecastHorizon] = useState<ForecastPeriod>("90D");
  const [categoryFilter, setCategoryFilter] = useState<RiskCategory | "ALL">("ALL");
  const [severityFilter, setSeverityFilter] = useState<RiskSeverity | "ALL">("ALL");

  // Engines execution
  const assessment = useMemo(() => evaluateGroupthinkAssessment(selectedCommitteeId), [selectedCommitteeId]);
  const invOI19 = useMemo(() => verifyINV_OI19(selectedCommitteeId), [selectedCommitteeId]);
  const invOI20 = useMemo(() => verifyINV_OI20(selectedCommitteeId), [selectedCommitteeId]);
  const invOI21 = useMemo(() => verifyINV_OI21(selectedCommitteeId), [selectedCommitteeId]);

  const riskExposure = useMemo(() => calculatePortfolioExposure(selectedCommitteeId), [selectedCommitteeId]);
  const allRisks = useMemo(() => getRiskRegistry(), []);

  const filteredRisks = useMemo(() => {
    return allRisks.filter((r) => {
      if (r.committeeId && r.committeeId !== selectedCommitteeId) return false;
      if (categoryFilter !== "ALL" && r.category !== categoryFilter) return false;
      if (severityFilter !== "ALL" && r.severity !== severityFilter) return false;
      return true;
    });
  }, [allRisks, selectedCommitteeId, categoryFilter, severityFilter]);

  const forecast = useMemo(() => computeGovernanceForecast(selectedCommitteeId, forecastHorizon), [selectedCommitteeId, forecastHorizon]);
  const invOI22 = useMemo(() => verifyINV_OI22(forecast), [forecast]);

  const incidentForecast = useMemo(() => predictIncidentEscalation("INC-201"), []);

  return (
    <div className="min-h-screen bg-[#0c1017] text-slate-100 font-mono flex flex-col selection:bg-cyan-500/20">
      <ExecutiveIntelligenceNav badgeText="M4 CERTIFIED" />

      <main className="flex-1 max-w-[1750px] w-full mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Header Ribbon & Committee Switcher */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 p-5 rounded-2xl bg-[#111724] border border-[#1f2c42] shadow-xl">
          <div>
            <div className="flex items-center space-x-2 text-xs text-cyan-400 font-bold mb-1">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
              <span>PREDICTIVE GOVERNANCE & GROUPTHINK INTELLIGENCE</span>
              <span className="text-slate-500">|</span>
              <span className="text-emerald-400">INV-OI19, INV-OI20, INV-OI21, INV-OI22 PASS</span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white">
              Institutional Risk & Groupthink Radar
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Anticipate consensus fatigue, suppressed dissent, and leading governance risk vectors before operational escalation.
            </p>
          </div>

          <div className="flex items-center space-x-2 bg-[#0c1017] p-1.5 rounded-xl border border-[#202d44]">
            {COMMITTEES.map((com) => (
              <button
                key={com.id}
                type="button"
                onClick={() => setSelectedCommitteeId(com.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  selectedCommitteeId === com.id
                    ? "bg-cyan-500 text-slate-950 shadow-md shadow-cyan-950"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#162032]"
                }`}
              >
                {com.id}
              </button>
            ))}
          </div>
        </div>

        {/* 4 Summary KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Card 1: Groupthink Score */}
          <div className="p-5 rounded-2xl bg-[#111724] border border-[#1f2c42] flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400">GROUPTHINK SCORE</span>
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  assessment.riskLevel === "LOW"
                    ? "bg-emerald-950/60 text-emerald-400 border border-emerald-500/30"
                    : assessment.riskLevel === "MEDIUM"
                    ? "bg-amber-950/60 text-amber-400 border border-amber-500/30"
                    : "bg-rose-950/60 text-rose-400 border border-rose-500/30"
                }`}
              >
                {assessment.riskLevel}
              </span>
            </div>
            <div className="my-3">
              <div className="text-3xl font-bold text-white tracking-tight">
                {assessment.groupthinkScore.toFixed(1)}
                <span className="text-xs text-slate-500 ml-1">/ 100</span>
              </div>
              <div className="w-full bg-[#1c273a] h-1.5 rounded-full overflow-hidden mt-2">
                <div
                  className={`h-full rounded-full ${
                    assessment.groupthinkScore < 50 ? "bg-emerald-400" : assessment.groupthinkScore < 75 ? "bg-amber-400" : "bg-rose-400"
                  }`}
                  style={{ width: `${assessment.groupthinkScore}%` }}
                />
              </div>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between">
              <span>INV-OI19 Ceiling: &lt;75.0</span>
              <span className={invOI19.valid ? "text-emerald-400" : "text-rose-400 font-bold"}>
                {invOI19.valid ? "✓ COMPLIANT" : "⚠ BREACH"}
              </span>
            </div>
          </div>

          {/* Card 2: Risk Exposure */}
          <div className="p-5 rounded-2xl bg-[#111724] border border-[#1f2c42] flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400">INSTITUTIONAL EXPOSURE</span>
              <span className="px-2 py-0.5 rounded bg-cyan-950/60 text-cyan-400 text-[10px] font-bold border border-cyan-500/30">
                {riskExposure.openRisks} ACTIVE
              </span>
            </div>
            <div className="my-3">
              <div className="text-3xl font-bold text-white tracking-tight">
                {riskExposure.averageExposure.toFixed(1)}
                <span className="text-xs text-slate-500 ml-1">Avg Score</span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                Peak Exposure: <span className="text-rose-400 font-semibold">{riskExposure.maxExposure.toFixed(1)}</span> (Critical: {riskExposure.criticalRisks})
              </p>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between">
              <span>Top Vector: {riskExposure.topRiskCategory}</span>
              <span className="text-cyan-400 font-semibold">VR-R01..06 PASS</span>
            </div>
          </div>

          {/* Card 3: Forecasted ODEI */}
          <div className="p-5 rounded-2xl bg-[#111724] border border-[#1f2c42] flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400">PROJECTED ODEI ({forecastHorizon})</span>
              <span className="px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-400 text-[10px] font-bold border border-emerald-500/30">
                {forecast.confidencePct}% CONF
              </span>
            </div>
            <div className="my-3">
              <div className="text-3xl font-bold text-white tracking-tight">
                {forecast.projectedODEI.toFixed(1)}
                <span className="text-xs text-emerald-400 ml-2">▲ Upward</span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                DIRatio: <span className="text-cyan-400 font-semibold">{forecast.projectedDIRatio}%</span> | Transfer: <span className="text-cyan-400 font-semibold">{forecast.projectedTransferRate}%</span>
              </p>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between">
              <span>Cert Probability</span>
              <span className="text-emerald-400 font-semibold">{forecast.projectedCertificationProbability}%</span>
            </div>
          </div>

          {/* Card 4: Escalation Watchlist */}
          <div className="p-5 rounded-2xl bg-[#111724] border border-[#1f2c42] flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400">ESCALATION WATCHLIST</span>
              <span className="px-2 py-0.5 rounded bg-rose-950/60 text-rose-400 text-[10px] font-bold border border-rose-500/30 animate-pulse">
                HIGH PROB
              </span>
            </div>
            <div className="my-3">
              <div className="text-3xl font-bold text-white tracking-tight">
                {incidentForecast.escalationProbability}%
                <span className="text-xs text-slate-500 ml-1">In 14 Days</span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1 truncate">
                {incidentForecast.likelyRootCauses[0]}
              </p>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between">
              <span>Repeat Risk: {incidentForecast.recurrenceProbability}%</span>
              <span className="text-amber-400 font-semibold">SLA PRESERVED</span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center space-x-2 border-b border-[#1f2c42] pb-2 text-xs overflow-x-auto">
          {[
            { id: "GROUPTHINK", label: "Groupthink Diagnostics" },
            { id: "REGISTRY", label: "Risk Registry Ledger" },
            { id: "FORECAST", label: "Forecast Academy" },
            { id: "CONSENSUS", label: "Consensus Analytics" },
            { id: "INCIDENTS", label: "Incident Escalation Predictor" },
          ].map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setActiveTab(t.id as any)}
              className={`px-4 py-2 rounded-xl transition-all whitespace-nowrap ${
                activeTab === t.id
                  ? "bg-cyan-500/10 text-cyan-400 font-bold border border-cyan-500/30"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#162032]"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab 1: Groupthink Diagnostics */}
        {activeTab === "GROUPTHINK" && (
          <div className="space-y-6 animate-in fade-in duration-150">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 rounded-xl bg-[#111724] border border-[#1f2c42]">
                <span className="text-xs text-slate-400 font-semibold">UNANIMITY RATE</span>
                <div className="text-2xl font-bold text-white mt-1">{assessment.unanimousDecisionRatePct}%</div>
                <p className="text-[11px] text-slate-400 mt-1">
                  Threshold: &lt;90.0% (Excess unanimity triggers `GROUPTHINK_RISK`)
                </p>
              </div>
              <div className="p-4 rounded-xl bg-[#111724] border border-[#1f2c42]">
                <span className="text-xs text-slate-400 font-semibold">DISSENT COVERAGE (INV-OI21)</span>
                <div className="text-2xl font-bold text-emerald-400 mt-1">{assessment.dissentRatePct}%</div>
                <p className="text-[11px] text-slate-400 mt-1">
                  Floor: &ge;10.0% (Utilization: {assessment.dissentUtilizationRatePct}%)
                </p>
              </div>
              <div className="p-4 rounded-xl bg-[#111724] border border-[#1f2c42]">
                <span className="text-xs text-slate-400 font-semibold">DECISION DIVERSITY (INV-OI20)</span>
                <div className="text-2xl font-bold text-cyan-400 mt-1">{assessment.diversityScore}%</div>
                <p className="text-[11px] text-slate-400 mt-1">
                  Floor: &ge;60.0% (Recommendation: {assessment.recommendationDiversityScore}%)
                </p>
              </div>
            </div>

            {/* Signals Feed */}
            <div className="p-5 rounded-2xl bg-[#111724] border border-[#1f2c42]">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-bold text-white">Active Groupthink Signals & Edge-Case Findings</h3>
                <span className="text-xs text-slate-400">{assessment.signals.length} Detected Signals</span>
              </div>
              {assessment.signals.length === 0 ? (
                <div className="p-4 rounded-xl bg-emerald-950/20 border border-emerald-500/30 text-emerald-400 text-xs">
                  ✓ No critical groupthink patterns detected. Deliberative diversity is well-balanced across all consensus and dissent vectors.
                </div>
              ) : (
                <div className="space-y-3">
                  {assessment.signals.map((sig) => (
                    <div key={sig.signalId} className="p-4 rounded-xl bg-[#0c1017] border border-[#202d44] space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-cyan-400">{sig.signalId} — {sig.signalType}</span>
                        <span className="px-2 py-0.5 rounded bg-rose-950/60 text-rose-400 text-[10px] font-bold border border-rose-500/30">
                          {sig.severity}
                        </span>
                      </div>
                      <p className="text-xs text-slate-300">{sig.description}</p>
                      <div className="text-[11px] text-amber-400 bg-amber-950/20 p-2 rounded-lg border border-amber-500/20">
                        Action Required: {sig.actionRequired}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 2: Risk Registry Ledger */}
        {activeTab === "REGISTRY" && (
          <div className="space-y-4 animate-in fade-in duration-150">
            {/* Filters */}
            <div className="flex flex-wrap items-center justify-between gap-3 p-4 rounded-xl bg-[#111724] border border-[#1f2c42]">
              <div className="flex items-center space-x-2 text-xs">
                <span className="text-slate-400">Category:</span>
                {(["ALL", "GROUPTHINK", "LEARNING", "NETWORK", "REPLAY", "GOVERNANCE", "ATTRIBUTION"] as const).map((cat) => (
                  <button
                    key={cat}
                    type="button"
                    onClick={() => setCategoryFilter(cat)}
                    className={`px-2 py-1 rounded text-[11px] transition-colors ${
                      categoryFilter === cat ? "bg-cyan-500 text-slate-950 font-bold" : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>

              <div className="flex items-center space-x-2 text-xs">
                <span className="text-slate-400">Severity:</span>
                {(["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"] as const).map((sev) => (
                  <button
                    key={sev}
                    type="button"
                    onClick={() => setSeverityFilter(sev)}
                    className={`px-2 py-1 rounded text-[11px] transition-colors ${
                      severityFilter === sev ? "bg-cyan-500 text-slate-950 font-bold" : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {sev}
                  </button>
                ))}
              </div>
            </div>

            {/* Risk Ledger Table */}
            <div className="rounded-2xl bg-[#111724] border border-[#1f2c42] overflow-hidden">
              <div className="p-4 border-b border-[#1f2c42] flex items-center justify-between">
                <h3 className="text-sm font-bold text-white">Institutional Governance Risk Ledger ({filteredRisks.length} Records)</h3>
                <span className="text-xs text-slate-400 font-mono">VR-R04: Exposure = (Likelihood × Impact) / 100</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-[#0c1017] text-slate-400 border-b border-[#1f2c42]">
                    <tr>
                      <th className="p-3">Risk ID</th>
                      <th className="p-3">Title</th>
                      <th className="p-3">Category</th>
                      <th className="p-3">Severity</th>
                      <th className="p-3">Likelihood</th>
                      <th className="p-3">Impact</th>
                      <th className="p-3">Exposure</th>
                      <th className="p-3">Incidents</th>
                      <th className="p-3">Owner</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1f2c42]">
                    {filteredRisks.map((r) => (
                      <tr key={r.riskId} className="hover:bg-[#162032] transition-colors">
                        <td className="p-3 font-bold text-cyan-400">{r.riskId}</td>
                        <td className="p-3">
                          <div className="font-semibold text-white">{r.title}</div>
                          <div className="text-[10px] text-slate-400 max-w-md truncate">{r.description}</div>
                        </td>
                        <td className="p-3">
                          <span className="px-2 py-0.5 rounded bg-[#1c273a] text-slate-300 text-[10px]">
                            {r.category}
                          </span>
                        </td>
                        <td className="p-3">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              r.severity === "CRITICAL"
                                ? "bg-rose-950 text-rose-400 border border-rose-500/40"
                                : r.severity === "HIGH"
                                ? "bg-amber-950 text-amber-400 border border-amber-500/40"
                                : "bg-cyan-950 text-cyan-400 border border-cyan-500/40"
                            }`}
                          >
                            {r.severity}
                          </span>
                        </td>
                        <td className="p-3">{r.likelihoodPct}%</td>
                        <td className="p-3">{r.impactScore}</td>
                        <td className="p-3 font-bold text-white">{r.exposureScore}</td>
                        <td className="p-3">
                          {r.incidentIds.length > 0 ? (
                            <span className="text-emerald-400">{r.incidentIds.join(", ")}</span>
                          ) : (
                            <span className="text-slate-600">—</span>
                          )}
                        </td>
                        <td className="p-3 text-slate-400">{r.ownerId ?? "N/A"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: Forecast Academy */}
        {activeTab === "FORECAST" && (
          <div className="space-y-6 animate-in fade-in duration-150">
            {/* Horizon Switcher */}
            <div className="flex items-center justify-between p-4 rounded-xl bg-[#111724] border border-[#1f2c42]">
              <div>
                <h3 className="text-sm font-bold text-white">Multi-Horizon Predictive Governance Academy</h3>
                <p className="text-xs text-slate-400">Forecasting leading indicators and driver attributions (INV-OI22).</p>
              </div>
              <div className="flex items-center space-x-2">
                {(["30D", "90D", "180D", "365D"] as const).map((h) => (
                  <button
                    key={h}
                    type="button"
                    onClick={() => setForecastHorizon(h)}
                    className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                      forecastHorizon === h
                        ? "bg-cyan-500 text-slate-950"
                        : "bg-[#0c1017] text-slate-400 hover:text-white"
                    }`}
                  >
                    {h}
                  </button>
                ))}
              </div>
            </div>

            {/* Drivers Breakdown */}
            <div className="p-5 rounded-2xl bg-[#111724] border border-[#1f2c42] space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold text-slate-400">LEADING FORECAST DRIVERS (INV-OI22 EXPLAINABILITY: 100%)</h4>
                <span className="text-xs text-emerald-400 font-bold">{invOI22.message}</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {forecast.drivers.map((d) => (
                  <div key={d.driverId} className="p-4 rounded-xl bg-[#0c1017] border border-[#202d44] space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-white">{d.driverName}</span>
                      <span className="text-xs font-bold text-cyan-400">{d.contributionPct}% Contribution</span>
                    </div>
                    <div className="w-full bg-[#1c273a] h-2 rounded-full overflow-hidden">
                      <div className="bg-cyan-400 h-full rounded-full" style={{ width: `${d.contributionPct}%` }} />
                    </div>
                    <div className="flex items-center justify-between text-[11px] text-slate-400">
                      <span>Baseline: {d.currentValue}</span>
                      <span className="text-emerald-400">Projected: {d.projectedValue} ({d.trend})</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Tab 4: Consensus Analytics */}
        {activeTab === "CONSENSUS" && (
          <div className="p-5 rounded-2xl bg-[#111724] border border-[#1f2c42] space-y-4 animate-in fade-in duration-150">
            <h3 className="text-sm font-bold text-white">Cross-Committee Consensus & Independence Matrix</h3>
            <p className="text-xs text-slate-400">
              Evaluates whether voting patterns across committees exhibit excessive alignment or artificial conformity.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
              {COMMITTEES.map((com) => {
                const prof = COMMITTEE_METRICS_PROFILE[com.id];
                return (
                  <div key={com.id} className="p-4 rounded-xl bg-[#0c1017] border border-[#202d44] space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-cyan-400">{com.id}</span>
                      <span className="text-[10px] text-slate-400">{com.focus}</span>
                    </div>
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between text-slate-300">
                        <span>Influence Concentration:</span>
                        <span className="font-bold text-white">{prof.influenceConcentration}%</span>
                      </div>
                      <div className="flex justify-between text-slate-300">
                        <span>Convergence Risk:</span>
                        <span className="font-bold text-emerald-400">{prof.convergenceScore}% (LOW)</span>
                      </div>
                      <div className="flex justify-between text-slate-300">
                        <span>Dissent Health:</span>
                        <span className="font-bold text-cyan-400">{prof.dissentRate}% / {prof.dissentUtilization}%</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Tab 5: Incident Escalation Predictor */}
        {activeTab === "INCIDENTS" && (
          <div className="p-5 rounded-2xl bg-[#111724] border border-[#1f2c42] space-y-4 animate-in fade-in duration-150">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white">Probabilistic Incident Escalation Predictor</h3>
              <span className="px-2 py-0.5 rounded bg-rose-950 text-rose-400 text-xs font-bold border border-rose-500/30">
                SLA BREACH THRESHOLD: 75%
              </span>
            </div>
            <div className="p-4 rounded-xl bg-[#0c1017] border border-[#202d44] space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-cyan-400">{incidentForecast.incidentId} Escalation Path</span>
                <span className="text-xs text-rose-400 font-bold">{incidentForecast.escalationProbability}% Escalation Prob</span>
              </div>
              <div className="space-y-2 text-xs">
                <div className="text-slate-300 font-semibold">Probable Root Causes:</div>
                <ul className="list-disc list-inside text-slate-400 space-y-1">
                  {incidentForecast.likelyRootCauses.map((rc, idx) => (
                    <li key={idx}>{rc}</li>
                  ))}
                </ul>
                <div className="text-slate-300 font-semibold mt-2">Recommended Mitigation Actions:</div>
                <ul className="list-disc list-inside text-emerald-400 space-y-1">
                  {incidentForecast.recommendedActions.map((ra, idx) => (
                    <li key={idx}>{ra}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Bottom Drawer: Universal Cross-Linking */}
        <div className="mt-8">
          <RelatedArtifactsCard
            entityId={initialQueryId || selectedCommitteeId}
            title={initialQueryId ? `Artifact Relationships for ${initialQueryId}` : `Committee Risk & Governance Lineage`}
          />
        </div>
      </main>
    </div>
  );
}

export default function RisksAndGroupthinkPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#0c1017] text-slate-100 font-mono p-8">Loading risks and groupthink intelligence...</div>}>
      <RisksAndGroupthinkContent />
    </Suspense>
  );
}
