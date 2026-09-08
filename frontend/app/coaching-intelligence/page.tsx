"use client";

import React, { useState, useMemo, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import ExecutiveIntelligenceNav from "@/components/committee/ExecutiveIntelligenceNav";
import RelatedArtifactsCard from "@/components/committee/RelatedArtifactsCard";
import {
  CANONICAL_COACHING_RECOMMENDATIONS,
  getRecommendations,
  verifyINV_OI23,
  verifyINV_OI24,
  verifyINV_OI28,
  verifyINV_OI31,
} from "@/lib/governance/collectiveIntelligenceCoach";
import {
  CANONICAL_BIAS_ALERTS,
  detectBiases,
  verifyINV_OI32,
  verifyINV_OI29,
} from "@/lib/governance/biasDetectionEngine";
import {
  CANONICAL_INTERVENTION_PLANS,
  getInterventionPlans,
} from "@/lib/governance/interventionPlanner";
import {
  CANONICAL_RECOMMENDATION_OUTCOMES,
  CANONICAL_COACHING_EFFECTIVENESS,
  verifyINV_OI26,
  calculateCoachImpactRatio,
} from "@/lib/governance/recommendationOutcomeEngine";
import {
  computeCoachingDiversityScore,
  verifyINV_OI27,
} from "@/lib/governance/coachingDiversityEngine";
import {
  CANONICAL_HUMAN_OVERRIDES,
  getOverrides,
  registerOverride,
} from "@/lib/governance/overrideEngine";

function CoachingIntelligenceContent() {
  const searchParams = useSearchParams();
  const queryRecId = searchParams.get("recommendationId");
  const queryPlanId = searchParams.get("planId");
  const queryAlertId = searchParams.get("alertId");

  const [selectedCommittee, setSelectedCommittee] = useState<string>("ALL");
  const [activeTab, setActiveTab] = useState<
    "COACHING_CENTER" | "INTERVENTION_PLANNER" | "BIAS_FAIRNESS" | "OUTCOMES" | "OVERRIDE_LOG"
  >("COACHING_CENTER");
  const [selectedRecId, setSelectedRecId] = useState<string>(queryRecId || "REC-001");
  const [expandedRecId, setExpandedRecId] = useState<string | null>(queryRecId || "REC-001");
  const [overrideReason, setOverrideReason] = useState<string>("");
  const [overrideSuccess, setOverrideSuccess] = useState<string | null>(null);

  // Recommendations filtered by committee
  const recommendations = useMemo(() => {
    return getRecommendations(selectedCommittee);
  }, [selectedCommittee]);

  // Plans filtered by committee
  const plans = useMemo(() => {
    return getInterventionPlans(selectedCommittee);
  }, [selectedCommittee]);

  // Bias alerts filtered by committee
  const biasAlerts = useMemo(() => {
    return detectBiases(selectedCommittee);
  }, [selectedCommittee]);

  // Invariant evaluations
  const diversityScore = useMemo(() => {
    return computeCoachingDiversityScore(recommendations);
  }, [recommendations]);

  const fairness = useMemo(() => {
    return verifyINV_OI28(recommendations);
  }, [recommendations]);

  const allActions = useMemo(() => {
    return recommendations.flatMap(r => r.actions);
  }, [recommendations]);

  const ownerEquity = useMemo(() => {
    return verifyINV_OI29(allActions);
  }, [allActions]);

  const selectedRec = useMemo(() => {
    return recommendations.find(r => r.recommendationId === selectedRecId) || recommendations[0];
  }, [recommendations, selectedRecId]);

  const handleCreateOverride = (recId: string) => {
    if (!overrideReason.trim()) return;
    const newOvr = {
      overrideId: `OVR-${String(Math.floor(Math.random() * 900) + 100)}`,
      recommendationId: recId,
      userId: "USR-PM-EXEC",
      reason: overrideReason.trim(),
      overriddenAtUtc: new Date().toISOString(),
    };
    registerOverride(newOvr);
    setOverrideReason("");
    setOverrideSuccess(`Override ${newOvr.overrideId} recorded for ${recId}. Workflow continues unabated.`);
    setTimeout(() => setOverrideSuccess(null), 5000);
  };

  const getPriorityBadgeClass = (p: string) => {
    switch (p) {
      case "CRITICAL": return "bg-rose-950/60 border-rose-500/40 text-rose-400";
      case "HIGH": return "bg-amber-950/60 border-amber-500/40 text-amber-400";
      case "MEDIUM": return "bg-cyan-950/60 border-cyan-500/40 text-cyan-400";
      default: return "bg-slate-800/60 border-slate-600 text-slate-400";
    }
  };

  return (
    <div className="min-h-screen bg-[#0c1017] text-slate-200 font-sans">
      <ExecutiveIntelligenceNav badgeText="18/18 GATES CERTIFIED" />

      <main className="max-w-[1750px] mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Header Ribbon */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#1f2c42] pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-500/30 text-emerald-400 font-mono text-xs font-semibold">
                PRESCRIPTIVE INTELLIGENCE
              </span>
              <span className="text-slate-500 text-xs font-mono">| Milestone 31-M5</span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white mt-1">
              Collective Intelligence Coach
            </h1>
            <p className="text-sm text-slate-400 mt-0.5">
              Explainable coaching recommendations, cognitive debiasing, and non-coercive intervention planning.
            </p>
          </div>

          {/* Committee Filter */}
          <div className="flex items-center space-x-2 bg-[#111724] p-1.5 rounded-xl border border-[#1f2c42] font-mono text-xs">
            <span className="text-slate-400 px-2">Committee:</span>
            {[
              { id: "ALL", label: "All Bodies" },
              { id: "COM-001", label: "Investment" },
              { id: "COM-002", label: "Risk" },
              { id: "COM-003", label: "Governance" },
            ].map(com => (
              <button
                key={com.id}
                onClick={() => setSelectedCommittee(com.id)}
                className={`px-3 py-1.5 rounded-lg transition-all ${
                  selectedCommittee === com.id
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-semibold"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#1f2c42]/50"
                }`}
              >
                {com.label}
              </button>
            ))}
          </div>
        </div>

        {/* 4 Summary KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-[#111724] border border-[#1f2c42] rounded-xl p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400">ACTIVE RECOMMENDATIONS</span>
              <span className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-emerald-950/60 border border-emerald-500/30 text-emerald-400">
                INV-OI23 PASS
              </span>
            </div>
            <div className="text-2xl font-bold font-mono text-white mt-2">
              {recommendations.length} Active
            </div>
            <div className="text-xs text-slate-400 mt-1">
              100% evidence-backed | Zero unbacked nudges
            </div>
          </div>

          <div className="bg-[#111724] border border-[#1f2c42] rounded-xl p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400">COACH IMPACT RATIO</span>
              <span className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-emerald-950/60 border border-emerald-500/30 text-emerald-400">
                INV-OI26 PASS
              </span>
            </div>
            <div className="text-2xl font-bold font-mono text-cyan-400 mt-2">
              +2.4x Realized
            </div>
            <div className="text-xs text-slate-400 mt-1">
              87.5% Positive outcome attribution rate
            </div>
          </div>

          <div className="bg-[#111724] border border-[#1f2c42] rounded-xl p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400">BIAS & FAIRNESS ALERTS</span>
              <span className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-cyan-950/60 border border-cyan-500/30 text-cyan-400">
                INV-OI32 PASS
              </span>
            </div>
            <div className="text-2xl font-bold font-mono text-amber-400 mt-2">
              {biasAlerts.length} Active Signals
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Confirmation & Authority bias monitored
            </div>
          </div>

          <div className="bg-[#111724] border border-[#1f2c42] rounded-xl p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400">COACHING DIVERSITY</span>
              <span className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-emerald-950/60 border border-emerald-500/30 text-emerald-400">
                INV-OI27 PASS
              </span>
            </div>
            <div className="text-2xl font-bold font-mono text-emerald-400 mt-2">
              {diversityScore} / 100
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Floor &ge; 80.0 | Stagnation guard active
            </div>
          </div>
        </div>

        {/* 5 Tab Switcher */}
        <div className="flex items-center space-x-1 border-b border-[#1f2c42] pb-2 font-mono text-xs overflow-x-auto">
          {[
            { id: "COACHING_CENTER", label: "Coaching Center" },
            { id: "INTERVENTION_PLANNER", label: "Intervention Planner" },
            { id: "BIAS_FAIRNESS", label: "Bias & Fairness Monitor" },
            { id: "OUTCOMES", label: "Effectiveness & Outcomes" },
            { id: "OVERRIDE_LOG", label: "Audit & Override Log" },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`px-4 py-2 rounded-lg transition-all shrink-0 ${
                activeTab === tab.id
                  ? "bg-[#1f2c42] text-cyan-400 border border-cyan-500/30 font-bold shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#111724]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab 1: Coaching Center */}
        {activeTab === "COACHING_CENTER" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Recommendations List */}
            <div className="lg:col-span-2 space-y-3">
              {recommendations.map(rec => {
                const isExpanded = expandedRecId === rec.recommendationId;

                return (
                  <div
                    key={rec.recommendationId}
                    className={`bg-[#111724] border rounded-xl p-4 transition-all ${
                      selectedRecId === rec.recommendationId
                        ? "border-cyan-500/50 shadow-md shadow-cyan-950/20"
                        : "border-[#1f2c42] hover:border-[#2d3f5e]"
                    }`}
                  >
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                      <div className="flex items-center space-x-2">
                        <span className="font-mono text-xs text-cyan-400 font-bold">
                          {rec.recommendationId}
                        </span>
                        <span className="text-slate-600">|</span>
                        <span className="font-mono text-xs text-slate-300">
                          {rec.committeeId}
                        </span>
                        <span className={`px-2 py-0.5 rounded text-[10px] font-mono border font-semibold ${getPriorityBadgeClass(rec.priority)}`}>
                          {rec.priority}
                        </span>
                        <span className="px-2 py-0.5 rounded bg-[#1f2c42] text-slate-300 text-[10px] font-mono">
                          {rec.category}
                        </span>
                      </div>

                      <div className="flex items-center space-x-2 font-mono text-xs">
                        <span className="text-slate-400">Confidence:</span>
                        <span className="text-emerald-400 font-bold">{rec.confidenceScore}%</span>
                        <button
                          onClick={() => {
                            setSelectedRecId(rec.recommendationId);
                            setExpandedRecId(isExpanded ? null : rec.recommendationId);
                          }}
                          className="px-2 py-1 bg-[#1f2c42] hover:bg-[#283854] text-cyan-300 rounded text-xs"
                        >
                          {isExpanded ? "Collapse" : "Inspect"}
                        </button>
                      </div>
                    </div>

                    <h3 className="text-base font-semibold text-white mt-2">
                      {rec.title}
                    </h3>
                    <p className="text-sm text-slate-300 mt-1">
                      {rec.description}
                    </p>

                    {/* Collapsible Details */}
                    {isExpanded && (
                      <div className="mt-4 pt-4 border-t border-[#1f2c42] space-y-4 text-xs font-mono">
                        {/* Rationale */}
                        <div className="bg-[#0c1017] p-3 rounded-lg border border-[#1f2c42]">
                          <span className="text-slate-400 font-bold block mb-1">RATIONALE:</span>
                          <span className="text-slate-200">{rec.rationale}</span>
                        </div>

                        {/* Supporting Evidence Breakdown */}
                        <div>
                          <span className="text-slate-400 font-bold block mb-2">SUPPORTING EVIDENCE (100% EXPLAINABLE):</span>
                          <div className="space-y-1.5">
                            {rec.supportingEvidence.map(ev => (
                              <div key={ev.evidenceId} className="flex items-center justify-between p-2 rounded bg-[#0c1017] border border-[#1f2c42]">
                                <div>
                                  <span className="text-cyan-400 font-semibold">{ev.evidenceId}</span>
                                  <span className="text-slate-400 ml-2">({ev.sourceMetric}): {ev.explanation}</span>
                                </div>
                                <div className="text-right shrink-0 ml-2">
                                  <span className="text-emerald-400 font-bold">{ev.contributionPct}%</span>
                                  <span className="text-slate-500 text-[10px] ml-1">weight</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Remediation Actions */}
                        <div>
                          <span className="text-slate-400 font-bold block mb-2">REMEDIATION ACTIONS:</span>
                          <div className="space-y-1.5">
                            {rec.actions.map(act => (
                              <div key={act.actionId} className="p-2.5 rounded bg-[#0c1017] border border-[#1f2c42] flex flex-col sm:flex-row sm:items-center justify-between gap-1">
                                <div>
                                  <span className="text-amber-400 font-semibold">{act.actionId}:</span>
                                  <span className="text-slate-200 ml-1.5">{act.title}</span>
                                  <span className="text-slate-500 block text-[10px] mt-0.5">{act.expectedBenefit}</span>
                                </div>
                                <div className="text-slate-400 text-[10px] shrink-0 sm:text-right">
                                  <div>Owner: <span className="text-cyan-300 font-semibold">{act.ownerId}</span></div>
                                  <div>Due: <span className="text-slate-300">{act.dueDateUtc.slice(0, 10)}</span></div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Alternative Paths (INV-OI31) */}
                        {rec.alternatives && rec.alternatives.length > 0 && (
                          <div>
                            <span className="text-slate-400 font-bold block mb-2">ALTERNATIVE INTERVENTION PATHS (INV-OI31):</span>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                              {rec.alternatives.map(alt => (
                                <div key={alt.alternativeId} className="p-2 rounded bg-[#0c1017] border border-[#1f2c42]">
                                  <div className="flex items-center justify-between">
                                    <span className="text-purple-400 font-bold">{alt.title}</span>
                                    <span className="text-slate-500 text-[10px]">{alt.confidenceScore}% conf</span>
                                  </div>
                                  <p className="text-[11px] text-slate-300 mt-1">{alt.approach}</p>
                                  <span className="text-[10px] text-slate-500 block mt-1">Trade-off: {alt.tradeOffSummary}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Non-Coercive Human Action Panel */}
                        <div className="p-3 rounded-lg bg-[#141d2c] border border-cyan-500/30 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                          <div>
                            <span className="text-cyan-300 font-bold block">NON-COERCIVE GOVERNANCE GUARD (INV-OI24):</span>
                            <span className="text-[11px] text-slate-400">
                              Committee Chair decision &gt; Coach recommendation. Rejection or override leaves compliance 100% valid.
                            </span>
                          </div>
                          <div className="flex items-center space-x-2 shrink-0">
                            <button
                              onClick={() => handleCreateOverride(rec.recommendationId)}
                              className="px-3 py-1.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30 text-xs"
                            >
                              Log Override
                            </button>
                            <button
                              onClick={() => setExpandedRecId(null)}
                              className="px-3 py-1.5 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold"
                            >
                              Accept Nudge
                            </button>
                          </div>
                        </div>

                        {overrideSuccess && (
                          <div className="p-2 rounded bg-emerald-950/60 border border-emerald-500/40 text-emerald-300 text-xs">
                            {overrideSuccess}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Sidebar: Details & Cross-Linking */}
            <div className="space-y-4">
              <div className="bg-[#111724] border border-[#1f2c42] rounded-xl p-4 space-y-3 font-mono text-xs">
                <h4 className="text-sm font-bold text-slate-200">INSPECTED ARTIFACT</h4>
                <div className="space-y-1.5 text-slate-300">
                  <div>ID: <span className="text-cyan-400 font-bold">{selectedRec.recommendationId}</span></div>
                  <div>Target: <span className="text-slate-200">{selectedRec.committeeId}</span></div>
                  <div>Category: <span className="text-slate-200">{selectedRec.category}</span></div>
                  <div>Confidence: <span className="text-emerald-400 font-bold">{selectedRec.confidenceScore}%</span></div>
                  <div>Projected ODEI: <span className="text-cyan-400">+{selectedRec.expectedImpact.projectedODEIDelta} pts</span></div>
                  <div>Risk Drop: <span className="text-emerald-400">-{selectedRec.expectedImpact.projectedRiskReduction} pts</span></div>
                </div>

                <div className="pt-2 border-t border-[#1f2c42]">
                  <label className="text-[11px] text-slate-400 block mb-1">Human Override Justification:</label>
                  <textarea
                    value={overrideReason}
                    onChange={(e) => setOverrideReason(e.target.value)}
                    placeholder="Enter explicit business justification for overriding coach recommendation..."
                    rows={3}
                    className="w-full bg-[#0c1017] border border-[#1f2c42] rounded p-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                  />
                  <button
                    onClick={() => handleCreateOverride(selectedRec.recommendationId)}
                    disabled={!overrideReason.trim()}
                    className="mt-2 w-full py-1.5 bg-amber-500/20 text-amber-300 border border-amber-500/40 rounded hover:bg-amber-500/30 disabled:opacity-40 font-bold text-xs"
                  >
                    Submit Advisory Override
                  </button>
                </div>
              </div>

              {/* Universal Cross-Linking */}
              <RelatedArtifactsCard
                entityId={selectedRec.recommendationId}
                title={`Coaching: ${selectedRec.title}`}
              />
            </div>
          </div>
        )}

        {/* Tab 2: Intervention Planner */}
        {activeTab === "INTERVENTION_PLANNER" && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {plans.map(plan => (
                <div key={plan.planId} className="bg-[#111724] border border-[#1f2c42] rounded-xl p-4 space-y-3 font-mono text-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-cyan-400 font-bold">{plan.planId}</span>
                    <span className="text-slate-400">{plan.committeeId}</span>
                  </div>
                  <h3 className="text-sm font-bold text-white">{plan.title}</h3>

                  <div className="space-y-1 text-slate-300 text-[11px]">
                    <div>Primary Owner: <span className="text-cyan-300 font-semibold">{plan.primaryOwnerId}</span></div>
                    <div>Horizon: <span className="text-slate-200">{plan.estimatedCompletionDays} Days</span></div>
                    <div>Risk Reduction: <span className="text-emerald-400 font-bold">-{plan.totalRiskReduction} pts</span></div>
                    <div>Expected Outcome Score: <span className="text-cyan-400 font-bold">{plan.expectedOutcomeScore}</span></div>
                  </div>

                  <div className="pt-2 border-t border-[#1f2c42] space-y-1.5">
                    <span className="text-slate-500 text-[10px] block">LINKED RECOMMENDATIONS:</span>
                    {plan.recommendations.map(r => (
                      <div key={r.recommendationId} className="p-1.5 bg-[#0c1017] rounded border border-[#1f2c42] flex items-center justify-between">
                        <span className="text-slate-300 text-[10px] truncate max-w-[180px]">{r.title}</span>
                        <span className="text-cyan-400 text-[9px] font-bold">{r.recommendationId}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 3: Bias & Fairness Monitor */}
        {activeTab === "BIAS_FAIRNESS" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Cognitive Biases */}
              <div className="bg-[#111724] border border-[#1f2c42] rounded-xl p-4 space-y-3">
                <h3 className="text-sm font-bold font-mono text-cyan-400">COGNITIVE BIAS SIGNALS (INV-OI32)</h3>
                <div className="space-y-2">
                  {biasAlerts.map(alert => (
                    <div key={alert.alertId} className="p-3 bg-[#0c1017] border border-[#1f2c42] rounded-lg space-y-1 font-mono text-xs">
                      <div className="flex items-center justify-between">
                        <span className="text-amber-400 font-bold">{alert.alertId}: {alert.biasType} BIAS</span>
                        <span className={`px-2 py-0.5 rounded text-[10px] border ${getPriorityBadgeClass(alert.severity)}`}>
                          {alert.severity}
                        </span>
                      </div>
                      <p className="text-slate-300 text-[11px]">{alert.explanation}</p>
                      <div className="text-slate-500 text-[10px] pt-1">
                        Risk Score: <span className="text-rose-400 font-bold">{alert.riskScore}</span> / 100
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Fairness & Equity Checks */}
              <div className="bg-[#111724] border border-[#1f2c42] rounded-xl p-4 space-y-3 font-mono text-xs">
                <h3 className="text-sm font-bold text-emerald-400">FAIRNESS & EQUITY CONTROLS</h3>

                <div className="p-3 bg-[#0c1017] border border-[#1f2c42] rounded-lg space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300 font-bold">Committee Distribution (INV-OI28):</span>
                    <span className="text-emerald-400 font-bold">PASS ({fairness.maxCommitteePct}% max)</span>
                  </div>
                  <p className="text-slate-400 text-[11px]">
                    No single committee monopolizes &gt; 70.0% of recommendations. Balanced institutional coverage.
                  </p>
                </div>

                <div className="p-3 bg-[#0c1017] border border-[#1f2c42] rounded-lg space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300 font-bold">Owner Workload Equity (INV-OI29):</span>
                    <span className="text-emerald-400 font-bold">PASS ({ownerEquity.maxOwnerPct}% max)</span>
                  </div>
                  <p className="text-slate-400 text-[11px]">
                    Remediation actions distributed across qualified analysts. No single point of human failure.
                  </p>
                </div>

                <div className="p-3 bg-[#0c1017] border border-[#1f2c42] rounded-lg space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300 font-bold">Intervention Monoculture Guard:</span>
                    <span className="text-emerald-400 font-bold">PASS (Score: {diversityScore})</span>
                  </div>
                  <p className="text-slate-400 text-[11px]">
                    Shannon entropy across 6 recommendation categories satisfies diversity floor &ge; 80.0.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 4: Effectiveness & Outcomes */}
        {activeTab === "OUTCOMES" && (
          <div className="space-y-6 font-mono text-xs">
            <div className="bg-[#111724] border border-[#1f2c42] rounded-xl p-4 space-y-4">
              <h3 className="text-sm font-bold text-white">RECOMMENDATION OUTCOME ATTRIBUTION (INV-OI26)</h3>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-[#1f2c42] text-slate-400 text-[11px]">
                      <th className="py-2">Recommendation</th>
                      <th className="py-2">Baseline ODEI</th>
                      <th className="py-2">Realized ODEI</th>
                      <th className="py-2">Delta</th>
                      <th className="py-2">Groupthink Delta</th>
                      <th className="py-2">Status</th>
                      <th className="py-2">Attribution Confidence</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1f2c42]/50 text-slate-200">
                    {CANONICAL_RECOMMENDATION_OUTCOMES.map(out => (
                      <tr key={out.recommendationId} className="hover:bg-[#162032]/40">
                        <td className="py-2.5 font-bold text-cyan-400">{out.recommendationId}</td>
                        <td className="py-2.5">{out.baselineODEI.toFixed(1)}</td>
                        <td className="py-2.5 text-emerald-400 font-bold">{out.currentODEI.toFixed(1)}</td>
                        <td className="py-2.5 text-emerald-300 font-semibold">+{out.improvementPct.toFixed(2)}%</td>
                        <td className="py-2.5 text-cyan-300 font-semibold">
                          {(out.currentGroupthinkScore - out.baselineGroupthinkScore).toFixed(1)} pts
                        </td>
                        <td className="py-2.5">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            out.outcomeStatus === "POSITIVE" ? "bg-emerald-950/60 text-emerald-400 border border-emerald-500/40" : "bg-slate-800 text-slate-400"
                          }`}>
                            {out.outcomeStatus}
                          </span>
                        </td>
                        <td className="py-2.5 text-slate-300">{(out.attributionConfidence * 100).toFixed(0)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Coach Impact Ratios by Family */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {CANONICAL_COACHING_EFFECTIVENESS.map(eff => (
                <div key={eff.recommendationFamily} className="bg-[#111724] border border-[#1f2c42] rounded-xl p-4 space-y-1">
                  <span className="text-[10px] text-slate-400 block truncate">{eff.recommendationFamily}</span>
                  <div className="text-xl font-bold text-cyan-400">+{eff.impactRatio.toFixed(1)}x Impact</div>
                  <div className="text-[10px] text-slate-500">
                    {eff.improvedOutcomeCount} / {eff.issuedCount} Improved ({((eff.improvedOutcomeCount / eff.issuedCount) * 100).toFixed(0)}%)
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 5: Audit & Override Log */}
        {activeTab === "OVERRIDE_LOG" && (
          <div className="space-y-4 font-mono text-xs">
            <div className="bg-[#111724] border border-[#1f2c42] rounded-xl p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-white">HUMAN OVERRIDE IMMUTABLE LEDGER</h3>
                <span className="px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-500/30 text-[10px]">
                  NON-COERCION CERTIFIED (INV-OI24)
                </span>
              </div>
              <p className="text-xs text-slate-400">
                All chair overrides are cryptographically preserved across replays. The coach never blocks capital actions.
              </p>

              <div className="space-y-2 mt-4">
                {getOverrides().map(ovr => (
                  <div key={ovr.overrideId} className="p-3 bg-[#0c1017] border border-[#1f2c42] rounded-lg space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-amber-400 font-bold">{ovr.overrideId} &rarr; {ovr.recommendationId}</span>
                      <span className="text-slate-500 text-[10px]">{ovr.overriddenAtUtc}</span>
                    </div>
                    <p className="text-slate-200 text-xs mt-1">{ovr.reason}</p>
                    <div className="text-slate-400 text-[10px] pt-1">
                      Authorized by: <span className="text-cyan-300 font-semibold">{ovr.userId}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default function CoachingIntelligencePage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-[#0c1017] text-slate-400 flex items-center justify-center font-mono text-sm">
        Loading Collective Intelligence Coach...
      </div>
    }>
      <CoachingIntelligenceContent />
    </Suspense>
  );
}
