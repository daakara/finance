"use client";

import React, { useState, useMemo, Suspense } from "react";
import IntelligenceHeader from "../../components/ui/IntelligenceHeader";
import HorizonMetricCard from "../../components/ui/HorizonMetricCard";
import { HorizonCard } from "../../components/ui/HorizonCard";
import SeverityBadge from "../../components/ui/SeverityBadge";
import RelatedArtifactsPanel, { RelatedArtifactLink } from "../../components/ui/RelatedArtifactsPanel";
import { CANONICAL_ADOPTION_BASELINE } from "../../lib/telemetry/fixtures/adoptionFixtures";
import {
  calculateProductivityGains,
  computeAdoptionReplayHash,
} from "../../lib/telemetry/executiveAdoptionEngine";

const RELATED_ARTIFACTS: RelatedArtifactLink[] = [
  {
    id: "ADP-ART-01",
    type: "DECISION",
    title: "Executive Decision Workspace OS",
    href: "/executive-workspace",
    summary: "Single-page decision cockpit integrating signal-to-learning workflows.",
  },
  {
    id: "ADP-ART-02",
    type: "AUDIT",
    title: "Executive Intelligence Center",
    href: "/intelligence-center",
    summary: "Institutional overview of OHI, ODEI, CDQI, and system health.",
  },
  {
    id: "ADP-ART-03",
    type: "COMMITTEE",
    title: "Decision Inbox & Priority Triage",
    href: "/decision-inbox",
    summary: "Stage-aware triage for high-impact decision packages.",
  },
  {
    id: "ADP-ART-04",
    type: "AUDIT",
    title: "Executive Release Certification Dashboard",
    href: "/release-dashboard",
    summary: "M1-M16 certification gates and immutable release attestation locks.",
  },
];

const TIMEFRAMES = ["7D", "30D", "90D", "YTD"] as const;

function AdoptionCenterContent() {
  const [selectedTimeframe, setSelectedTimeframe] = useState<typeof TIMEFRAMES[number]>("30D");
  const [selectedMilestone, setSelectedMilestone] = useState<string | null>(null);

  const snapshot = CANONICAL_ADOPTION_BASELINE;

  // Recalculate dynamic ROI based on active timeframe
  const productivity = useMemo(() => {
    const scale = selectedTimeframe === "7D" ? 0.25 : selectedTimeframe === "90D" ? 3.0 : selectedTimeframe === "YTD" ? 8.0 : 1.0;
    const completed = Math.round(snapshot.workflowCohort.totalWorkflowsCompleted * scale);
    return calculateProductivityGains(
      completed,
      snapshot.metrics.medianTimeToDecisionMinutes,
      snapshot.metrics.baselineTimeToDecisionMinutes
    );
  }, [selectedTimeframe, snapshot]);

  const replayHash = useMemo(() => {
    return computeAdoptionReplayHash(snapshot);
  }, [snapshot]);

  return (
    <div className="min-h-screen bg-[#0A0F1D] text-[#F8FAFC] p-6 space-y-6">
      {/* Header */}
      <IntelligenceHeader
        title="Executive Adoption & Value Realization"
        subtitle="ARX Horizon Operationalization · Executive Usage & Workflow Telemetry"
        status="CERTIFIED"
        replayHash={replayHash}
        breadcrumbs={[
          { label: "Executive Home", href: "/intelligence-center" },
          { label: "Executive Adoption Center" },
        ]}
        actions={
          <div className="flex items-center gap-1.5 p-1 rounded-xl bg-[#0E1524] border border-[#24324A]">
            {TIMEFRAMES.map((tf) => (
              <button
                key={tf}
                onClick={() => setSelectedTimeframe(tf)}
                className={`px-3 py-1 rounded-lg text-xs font-mono font-medium transition-all ${
                  selectedTimeframe === tf
                    ? "bg-cyan-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
        }
      />

      {/* 6 Institutional Adoption Metric Cards */}
      <section aria-label="Adoption KPIs" className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        <HorizonMetricCard
          label="Active Executives"
          value={`${snapshot.metrics.dailyActiveExecutives} DAE`}
          delta="+12.4%"
          deltaPositive={true}
          severity="PASS"
          subtext={`${snapshot.metrics.monthlyActiveExecutives} monthly active`}
        />
        <HorizonMetricCard
          label="Time to Decision"
          value={`${snapshot.metrics.medianTimeToDecisionMinutes} min`}
          delta={`-${snapshot.metrics.timeReductionPct}%`}
          deltaPositive={true}
          severity="PASS"
          subtext="vs 4.2h legacy baseline"
        />
        <HorizonMetricCard
          label="Actions Executed"
          value={snapshot.metrics.totalActionsExecuted}
          delta="+18.3%"
          deltaPositive={true}
          severity="PASS"
          subtext={`${snapshot.metrics.actionSlaAdherencePct}% SLA adherence`}
        />
        <HorizonMetricCard
          label="Briefings Generated"
          value={snapshot.metrics.totalBriefingsGenerated}
          delta="+24.1%"
          deltaPositive={true}
          severity="PASS"
          subtext="100% replay deterministic"
        />
        <HorizonMetricCard
          label="Search Success Rate"
          value={`${snapshot.metrics.searchSuccessRatePct}%`}
          delta="+2.1%"
          deltaPositive={true}
          severity="PASS"
          subtext={`${snapshot.metrics.averageSearchLatencyMs}ms avg latency`}
        />
        <HorizonMetricCard
          label="Overall Adoption"
          value={`${snapshot.metrics.overallFeatureAdoptionPct}%`}
          delta="+5.8%"
          deltaPositive={true}
          severity="PASS"
          subtext="M11–M16 active usage"
        />
      </section>

      {/* Main Grid: Feature Adoption Matrix & Workflow Lifecycle */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Capability Adoption Matrix */}
        <HorizonCard
          title="Capability Adoption Matrix"
          subtitle="Executive engagement across M11–M16 operational milestones"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-[#24324A] text-slate-400">
                  <th className="py-2.5 px-3">Milestone</th>
                  <th className="py-2.5 px-3">Capability</th>
                  <th className="py-2.5 px-3">Active Users</th>
                  <th className="py-2.5 px-3">Usage</th>
                  <th className="py-2.5 px-3">Adoption</th>
                  <th className="py-2.5 px-3 text-right">Trend</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#24324A]/40">
                {snapshot.milestoneBreakdown.map((m) => (
                  <tr
                    key={m.milestoneId}
                    onClick={() => setSelectedMilestone(m.milestoneId)}
                    className="hover:bg-[#152136] transition-colors cursor-pointer"
                  >
                    <td className="py-2.5 px-3 font-semibold text-cyan-300">{m.milestoneId}</td>
                    <td className="py-2.5 px-3 font-sans text-white font-medium">{m.milestoneName}</td>
                    <td className="py-2.5 px-3 text-slate-300">{m.activeUsers}</td>
                    <td className="py-2.5 px-3 text-slate-300">{m.usageCount}</td>
                    <td className="py-2.5 px-3">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 rounded-full bg-[#1E293B] overflow-hidden">
                          <div
                            className="h-full bg-cyan-400 rounded-full"
                            style={{ width: `${m.adoptionPct}%` }}
                          />
                        </div>
                        <span className="text-cyan-300 font-semibold">{m.adoptionPct}%</span>
                      </div>
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <SeverityBadge status={m.trend === "RISING" ? "HEALTHY" : "WARNING"} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </HorizonCard>

        {/* 8-Stage Workflow Completion & Cycle Times */}
        <HorizonCard
          title="Workflow Lifecycle & Bottleneck Analysis"
          subtitle={`Cohort ${snapshot.workflowCohort.cohortId} · ${snapshot.workflowCohort.completionRatePct}% overall completion`}
        >
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs font-mono text-slate-400 pb-2 border-b border-[#24324A]">
              <span>Initiated: <strong className="text-white">{snapshot.workflowCohort.totalWorkflowsInitiated}</strong></span>
              <span>Completed: <strong className="text-emerald-400">{snapshot.workflowCohort.totalWorkflowsCompleted}</strong></span>
              <span>Median Cycle: <strong className="text-cyan-300">{snapshot.workflowCohort.medianCycleTimeMinutes} min</strong></span>
            </div>

            <div className="space-y-2">
              {snapshot.workflowCohort.stages.map((st) => (
                <div
                  key={st.stageNumber}
                  className="p-2.5 rounded-lg bg-[#0E1524] border border-[#24324A]/60 flex items-center justify-between text-xs font-mono"
                >
                  <div className="flex items-center gap-2.5">
                    <span className="w-5 h-5 rounded-full bg-[#1E293B] flex items-center justify-center text-[10px] text-cyan-400 font-bold">
                      {st.stageNumber}
                    </span>
                    <span className="text-white font-medium">{st.stageName}</span>
                  </div>
                  <div className="flex items-center gap-4 text-slate-400">
                    <span>{st.medianMinutes} min</span>
                    <span className="text-emerald-400 font-semibold">{st.completionRatePct}% complete</span>
                    {st.dropOffRatePct > 0 && (
                      <span className="text-amber-400 text-[10px]">-{st.dropOffRatePct}% drop</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </HorizonCard>
      </div>

      {/* Productivity Gain & Economic ROI Ledger */}
      <HorizonCard
        title="Productivity Gain & Economic ROI Summary"
        subtitle={`Empirical savings based on ${selectedTimeframe} active governance operations`}
      >
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs font-mono">
          <div className="p-4 rounded-xl bg-[#0E1524] border border-[#24324A]/80 space-y-1">
            <span className="text-slate-400 uppercase tracking-wider text-[10px]">Monthly Hours Saved / Exec</span>
            <div className="text-2xl font-bold text-white tracking-tight">
              {productivity.hoursSavedPerExecutiveMonthly} hrs
            </div>
            <p className="text-slate-500 text-[11px] pt-1">
              Direct reduction in meeting and report prep time
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[#0E1524] border border-[#24324A]/80 space-y-1">
            <span className="text-slate-400 uppercase tracking-wider text-[10px]">Total Hours Saved</span>
            <div className="text-2xl font-bold text-emerald-400 tracking-tight">
              {productivity.totalHoursSavedMonthly} hrs
            </div>
            <p className="text-slate-500 text-[11px] pt-1">
              Aggregated across all 34 active executive leaders
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[#0E1524] border border-[#24324A]/80 space-y-1">
            <span className="text-slate-400 uppercase tracking-wider text-[10px]">Effective Value Realized</span>
            <div className="text-2xl font-bold text-cyan-300 tracking-tight">
              ${productivity.effectiveCostSavingsUSD.toLocaleString()}
            </div>
            <p className="text-slate-500 text-[11px] pt-1">
              Based on certified institutional executive hourly benchmark ($250/hr)
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[#0E1524] border border-[#24324A]/80 space-y-1">
            <span className="text-slate-400 uppercase tracking-wider text-[10px]">Decision Velocity Multiplier</span>
            <div className="text-2xl font-bold text-purple-300 tracking-tight">
              {productivity.decisionVelocityMultiplier}x
            </div>
            <p className="text-slate-500 text-[11px] pt-1">
              18.4 min vs 252 min legacy committee latency
            </p>
          </div>
        </div>
      </HorizonCard>

      {/* Related Institutional Artifacts */}
      <RelatedArtifactsPanel
        title="Connected Decision Centers & Operational Workflows"
        artifacts={RELATED_ARTIFACTS}
      />
    </div>
  );
}

export default function AdoptionCenterPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-[#0A0F1D] text-slate-400 flex items-center justify-center font-mono text-sm">
          Loading Executive Adoption Center...
        </div>
      }
    >
      <AdoptionCenterContent />
    </Suspense>
  );
}
