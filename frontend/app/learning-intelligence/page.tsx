"use client";

import React, { useState, useMemo, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import ExecutiveIntelligenceNav from "@/components/committee/ExecutiveIntelligenceNav";
import RelatedArtifactsCard from "@/components/committee/RelatedArtifactsCard";
import {
  getAllLearnings,
  getAllAdoptions,
  getLearningsByCommittee,
  getAdoptionsByCommittee,
} from "@/lib/telemetry/learningIntelligenceEngine";
import {
  computeLearningVelocity,
  computeVelocityTrend,
} from "@/lib/telemetry/learningVelocityEngine";
import {
  getKnowledgeTransferNetwork,
  computeKnowledgeTransferEdge,
} from "@/lib/telemetry/knowledgeTransferNetwork";
import {
  computeLearningFriction,
  getNetworkFrictionOverview,
} from "@/lib/telemetry/learningFrictionEngine";
import {
  getActiveIncidents,
  resolveIncident,
} from "@/lib/governance/alertCorrelationEngine";
import type {
  LearningRecord,
  CorrelatedIncident,
  LearningCategory,
} from "@/types/learning-intelligence";

const COMMITTEES = [
  { id: "COM-001", name: "Investment Committee", role: "Strategy & Capital Allocation" },
  { id: "COM-002", name: "Governance Committee", role: "Policy & Invariant Compliance" },
  { id: "COM-003", name: "Risk & Capital Committee", role: "Tail Risk & VaR Calibration" },
];

function LearningDashboardContent() {
  const searchParams = useSearchParams();
  const initialLearningId = searchParams.get("learningId");
  const initialIncidentId = searchParams.get("incidentId");

  const [selectedCommitteeId, setSelectedCommitteeId] = useState<string>("COM-001");
  const [activeTab, setActiveTab] = useState<"VELOCITY" | "NETWORK" | "FRICTION" | "INCIDENTS" | "CATALOG">("VELOCITY");
  const [categoryFilter, setCategoryFilter] = useState<LearningCategory | "ALL">("ALL");
  const [selectedLearningId, setSelectedLearningId] = useState<string | null>(initialLearningId);
  const [incidents, setIncidents] = useState<CorrelatedIncident[]>(getActiveIncidents());
  const [resolvedNotification, setResolvedNotification] = useState<string | null>(null);

  // Velocity calculations
  const velocityResult = useMemo(() => {
    return computeLearningVelocity(selectedCommitteeId);
  }, [selectedCommitteeId]);

  const velocityTrend = useMemo(() => {
    return computeVelocityTrend(selectedCommitteeId);
  }, [selectedCommitteeId]);

  // Transfer Network calculations
  const transferNetwork = useMemo(() => {
    return getKnowledgeTransferNetwork();
  }, []);

  const committeeOutgoingEdges = useMemo(() => {
    return transferNetwork.edges.filter(e => e.sourceCommitteeId === selectedCommitteeId);
  }, [selectedCommitteeId, transferNetwork]);

  // Friction calculations
  const frictionResult = useMemo(() => {
    return computeLearningFriction(selectedCommitteeId);
  }, [selectedCommitteeId]);

  const networkFriction = useMemo(() => {
    return getNetworkFrictionOverview();
  }, []);

  // Learnings catalog
  const allLearnings = useMemo(() => {
    return getAllLearnings();
  }, []);

  const filteredLearnings = useMemo(() => {
    return allLearnings.filter(l => {
      if (categoryFilter === "ALL") return true;
      return l.category === categoryFilter;
    });
  }, [allLearnings, categoryFilter]);

  const handleResolveIncident = (incidentId: string) => {
    const updated = resolveIncident(incidentId, "Manual resolution certified by operator.");
    setIncidents(getActiveIncidents());
    setResolvedNotification(`Incident ${updated.incidentId} marked as RESOLVED.`);
    setTimeout(() => setResolvedNotification(null), 4000);
  };

  const selectedLearning = useMemo(() => {
    if (!selectedLearningId) return null;
    return allLearnings.find(l => l.learningId === selectedLearningId) ?? null;
  }, [allLearnings, selectedLearningId]);

  return (
    <div className="min-h-screen bg-[#0c1017] text-slate-100 font-sans antialiased pb-20">
      <ExecutiveIntelligenceNav badgeText="10/10 M3 GATES CERTIFIED" />

      <main className="max-w-[1750px] mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Header Ribbon & Committee Switcher */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#1c273a] pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-950/80 text-cyan-400 border border-cyan-500/30">
                EPIC AI-003 · PHASE 31-M3
              </span>
              <span className="text-xs font-mono text-slate-400">
                Invariants INV-OI17 & INV-OI18 Certified
              </span>
            </div>
            <h2 className="text-xl font-bold tracking-tight text-white mt-1">
              Organizational Learning Intelligence
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Team Learning Velocity, Cross-Committee Transfer Network, Friction Diagnostics & Correlated NOC Incidents.
            </p>
          </div>

          {/* Committee selector tabs */}
          <div className="flex items-center space-x-1.5 p-1 bg-[#111724] border border-[#1f2c42] rounded-lg">
            {COMMITTEES.map((com) => (
              <button
                key={com.id}
                type="button"
                onClick={() => setSelectedCommitteeId(com.id)}
                className={`px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
                  selectedCommitteeId === com.id
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-semibold shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#182335]"
                }`}
              >
                {com.id} · {com.name.split(" ")[0]}
              </button>
            ))}
          </div>
        </div>

        {/* Resolved Notification Banner */}
        {resolvedNotification && (
          <div className="p-3 bg-emerald-950/70 border border-emerald-500/40 rounded-lg text-xs font-mono text-emerald-300 flex items-center justify-between animate-fadeIn">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>{resolvedNotification}</span>
            </div>
            <button
              type="button"
              onClick={() => setResolvedNotification(null)}
              className="text-emerald-400 hover:text-emerald-200"
            >
              ✕
            </button>
          </div>
        )}

        {/* 4 Key Metric Executive Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Card 1: Team Learning Velocity */}
          <div className="p-4 rounded-xl bg-[#111724] border border-[#1f2c42] flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">
                Learning Velocity (INV-OI17)
              </span>
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                  velocityResult.status === "POSITIVE"
                    ? "bg-emerald-950/80 text-emerald-400 border border-emerald-500/30"
                    : velocityResult.status === "STAGNANT"
                    ? "bg-amber-950/80 text-amber-400 border border-amber-500/30"
                    : "bg-rose-950/80 text-rose-400 border border-rose-500/30"
                }`}
              >
                {velocityResult.status}
              </span>
            </div>
            <div className="mt-2">
              <div className="flex items-baseline space-x-2">
                <span className="text-3xl font-mono font-bold text-white">
                  {velocityResult.velocity > 0 ? `+${velocityResult.velocity}` : velocityResult.velocity}
                </span>
                <span className="text-xs font-mono text-slate-400">ΔODEI / quarter</span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                Annualized: <strong className="text-slate-200 font-mono">+{velocityResult.annualizedVelocity}</strong> pts/yr | Forecast: <strong className="text-cyan-300 font-mono">{velocityResult.forecastNextQuarter}</strong>
              </p>
            </div>
          </div>

          {/* Card 2: Knowledge Transfer Rate */}
          <div className="p-4 rounded-xl bg-[#111724] border border-[#1f2c42] flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">
                Transfer Rate (INV-OI18)
              </span>
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                  transferNetwork.overallNetworkTransferRate >= 80.0
                    ? "bg-emerald-950/80 text-emerald-400 border border-emerald-500/30"
                    : "bg-rose-950/80 text-rose-400 border border-rose-500/30"
                }`}
              >
                {transferNetwork.overallNetworkTransferRate >= 80.0 ? "COMPLIANT" : "BREACH"}
              </span>
            </div>
            <div className="mt-2">
              <div className="flex items-baseline space-x-2">
                <span className="text-3xl font-mono font-bold text-white">
                  {transferNetwork.overallNetworkTransferRate}%
                </span>
                <span className="text-xs font-mono text-slate-400">Adopted / Published</span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                Target: <strong className="text-slate-200 font-mono">≥ 80.0%</strong> across all committee DAG edges
              </p>
            </div>
          </div>

          {/* Card 3: Learning Friction */}
          <div className="p-4 rounded-xl bg-[#111724] border border-[#1f2c42] flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">
                Friction Score (0-100)
              </span>
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                  frictionResult.frictionScore < 25.0
                    ? "bg-emerald-950/80 text-emerald-400 border border-emerald-500/30"
                    : frictionResult.frictionScore < 50.0
                    ? "bg-amber-950/80 text-amber-400 border border-amber-500/30"
                    : "bg-rose-950/80 text-rose-400 border border-rose-500/30"
                }`}
              >
                {frictionResult.frictionScore < 25 ? "LOW FRICTION" : frictionResult.frictionScore < 50 ? "MODERATE" : "HIGH FRICTION"}
              </span>
            </div>
            <div className="mt-2">
              <div className="flex items-baseline space-x-2">
                <span className="text-3xl font-mono font-bold text-white">
                  {frictionResult.frictionScore}
                </span>
                <span className="text-xs font-mono text-slate-400">/ 100</span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                Top Bottleneck: <strong className="text-amber-300 font-mono">{frictionResult.topFrictionCategory}</strong>
              </p>
            </div>
          </div>

          {/* Card 4: Correlated NOC Incidents */}
          <div className="p-4 rounded-xl bg-[#111724] border border-[#1f2c42] flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">
                Correlated NOC Incidents
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-purple-950/80 text-purple-300 border border-purple-500/30">
                FATIGUE CONTROLS ON
              </span>
            </div>
            <div className="mt-2">
              <div className="flex items-baseline space-x-2">
                <span className="text-3xl font-mono font-bold text-white">
                  {incidents.filter(i => i.status !== "RESOLVED" && i.status !== "CLOSED").length}
                </span>
                <span className="text-xs font-mono text-slate-400">Active Incidents</span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                Total Signals Compressed: <strong className="text-slate-200 font-mono">{incidents.reduce((acc, i) => acc + i.occurrenceCount, 0)} alerts</strong>
              </p>
            </div>
          </div>
        </div>

        {/* View Switcher Tabs */}
        <div className="flex items-center space-x-2 border-b border-[#1c273a] pb-2 text-xs font-mono">
          {[
            { id: "VELOCITY", label: "Learning Velocity & Trajectory" },
            { id: "NETWORK", label: "Knowledge Transfer Network" },
            { id: "FRICTION", label: "Friction Diagnostics" },
            { id: "INCIDENTS", label: `Correlated Incidents (${incidents.filter(i => i.status !== "CLOSED").length})` },
            { id: "CATALOG", label: `Learnings Catalog (${allLearnings.length})` },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id as typeof activeTab)}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                activeTab === tab.id
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-semibold"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#141b29]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab 1: Learning Velocity & Trajectory */}
        {activeTab === "VELOCITY" && (
          <div className="space-y-6">
            <div className="p-6 rounded-xl bg-[#111724] border border-[#1f2c42] space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-white">Quarterly ODEI Progression & Velocity Trajectory</h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Demonstrating continuous organizational learning under INV-OI17 non-regression bounds.
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-mono text-slate-400">Momentum:</span>
                  <span className="px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-emerald-950/80 text-emerald-400 border border-emerald-500/30">
                    {velocityTrend.momentum} (+{velocityTrend.acceleration} acc)
                  </span>
                </div>
              </div>

              {/* Quarterly Bar Graph Representation */}
              <div className="grid grid-cols-4 gap-4 pt-4">
                {velocityResult.historicalQuarterlyVelocities.map((item) => (
                  <div key={item.quarter} className="p-3 bg-[#0d1420] border border-[#1d293d] rounded-lg text-center">
                    <span className="text-[11px] font-mono text-slate-400">{item.quarter}</span>
                    <div className="text-2xl font-mono font-bold text-white my-1">{item.odei}</div>
                    <span className="inline-block px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-950/60 text-emerald-300 border border-emerald-500/30">
                      +{item.velocity} pts/qtr
                    </span>
                  </div>
                ))}
              </div>

              {/* Attributable learnings */}
              <div className="pt-4 border-t border-[#1d293d]">
                <h4 className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-2">
                  Attributed Learning Sources Linking to Improvement (AC-OI17-04 / 06)
                </h4>
                <div className="flex flex-wrap gap-2">
                  {velocityResult.attributableLearnings.map((id) => (
                    <button
                      key={id}
                      type="button"
                      onClick={() => {
                        setSelectedLearningId(id);
                        setActiveTab("CATALOG");
                      }}
                      className="px-2.5 py-1 rounded bg-[#182335] hover:bg-[#202f47] border border-[#23334d] text-xs font-mono text-cyan-300 transition-colors"
                    >
                      {id} ↗
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Knowledge Transfer Network */}
        {activeTab === "NETWORK" && (
          <div className="space-y-6">
            <div className="p-6 rounded-xl bg-[#111724] border border-[#1f2c42] space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-white">Cross-Committee Knowledge Transfer Matrix</h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Propagation of institutional learnings across committee decision boundaries. Threshold: ≥ 80.0%.
                  </p>
                </div>
                <span className="px-2.5 py-1 rounded text-xs font-mono bg-emerald-950/80 text-emerald-300 border border-emerald-500/30">
                  {transferNetwork.overallNetworkTransferRate}% Overall Adoption
                </span>
              </div>

              {/* Table of edges */}
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-[#1d293d] text-slate-400">
                      <th className="py-2.5 px-3">Source Body</th>
                      <th className="py-2.5 px-3">Receiving Body</th>
                      <th className="py-2.5 px-3">Published</th>
                      <th className="py-2.5 px-3">Adopted</th>
                      <th className="py-2.5 px-3">Transfer Rate</th>
                      <th className="py-2.5 px-3">Velocity Impact</th>
                      <th className="py-2.5 px-3">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1d293d]">
                    {transferNetwork.edges.map((edge, idx) => (
                      <tr key={idx} className="hover:bg-[#141b29] transition-colors">
                        <td className="py-2.5 px-3 font-semibold text-white">{edge.sourceCommitteeId}</td>
                        <td className="py-2.5 px-3 text-slate-300">{edge.targetCommitteeId}</td>
                        <td className="py-2.5 px-3 text-slate-400">{edge.publishedLearnings}</td>
                        <td className="py-2.5 px-3 text-cyan-300 font-semibold">{edge.adoptedLearnings}</td>
                        <td className="py-2.5 px-3 font-bold text-white">{edge.transferRatePct}%</td>
                        <td className="py-2.5 px-3 text-emerald-400">+{edge.velocityImpact} pts</td>
                        <td className="py-2.5 px-3">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              edge.status === "COMPLIANT"
                                ? "bg-emerald-950/80 text-emerald-400 border border-emerald-500/30"
                                : "bg-rose-950/80 text-rose-400 border border-rose-500/30"
                            }`}
                          >
                            {edge.status}
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

        {/* Tab 3: Learning Friction Diagnostics */}
        {activeTab === "FRICTION" && (
          <div className="space-y-6">
            <div className="p-6 rounded-xl bg-[#111724] border border-[#1f2c42] space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-white">Learning Friction Engine & Bottleneck Breakdown</h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Analyzing barriers preventing rapid institutional learning adoption in {selectedCommitteeId}.
                  </p>
                </div>
                <span className="text-xs font-mono text-slate-400">
                  Friction Score: <strong className="text-white">{frictionResult.frictionScore}/100</strong>
                </span>
              </div>

              {/* 6 Category Breakdown Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                {[
                  { cat: "IGNORED", count: frictionResult.ignoredCount, label: "Ignored" },
                  { cat: "OWNERSHIP_GAP", count: frictionResult.ownershipGapCount, label: "Ownership Gap" },
                  { cat: "GOVERNANCE_GAP", count: frictionResult.governanceGapCount, label: "Governance Gap" },
                  { cat: "EXPIRED", count: frictionResult.expiredCount, label: "SLA Expired" },
                  { cat: "REJECTED", count: frictionResult.rejectedCount, label: "Rejected" },
                  { cat: "UNKNOWN", count: frictionResult.unknownCount, label: "Unknown Barrier" },
                ].map((item) => (
                  <div key={item.cat} className="p-3 bg-[#0d1420] border border-[#1d293d] rounded-lg text-center">
                    <span className="text-[10px] font-mono text-slate-400 uppercase">{item.label}</span>
                    <div className="text-xl font-mono font-bold text-white mt-1">{item.count}</div>
                  </div>
                ))}
              </div>

              {/* Itemized Friction Bottlenecks */}
              <div className="pt-4 border-t border-[#1d293d] space-y-3">
                <h4 className="text-xs font-mono text-slate-400 uppercase tracking-wider">
                  Documented Adoption Bottlenecks ({frictionResult.items.length})
                </h4>
                {frictionResult.items.map((item, idx) => (
                  <div key={idx} className="p-3 bg-[#0e1624] border border-[#202d42] rounded-lg flex flex-col md:flex-row md:items-center justify-between gap-2">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-950/60 text-cyan-400 border border-cyan-500/30">
                          {item.learningId}
                        </span>
                        <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-950/60 text-amber-400 border border-amber-500/30">
                          {item.category}
                        </span>
                        <span className="text-xs text-slate-300 font-mono">Pending: {item.daysPending} days</span>
                      </div>
                      <p className="text-xs text-slate-300 mt-1">{item.explanation}</p>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedLearningId(item.learningId);
                        setActiveTab("CATALOG");
                      }}
                      className="self-start md:self-auto px-3 py-1 bg-[#1a2538] hover:bg-[#23334d] border border-[#263752] rounded text-xs font-mono text-slate-200 transition-colors"
                    >
                      Inspect Learning ↗
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Tab 4: Correlated Incidents NOC */}
        {activeTab === "INCIDENTS" && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-white">Correlated Governance & Learning Incidents</h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Multi-signal correlation merges raw alert floods into root-cause incident streams with 30-min suppression windows.
                </p>
              </div>
              <span className="text-xs font-mono text-slate-400">
                Active: {incidents.filter(i => i.status !== "RESOLVED" && i.status !== "CLOSED").length} incidents
              </span>
            </div>

            <div className="space-y-3">
              {incidents.map((incident) => (
                <div
                  key={incident.incidentId}
                  className="p-4 bg-[#111724] border border-[#1f2c42] rounded-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4"
                >
                  <div className="space-y-1.5 flex-1">
                    <div className="flex items-center space-x-2 flex-wrap gap-y-1">
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-purple-950/80 text-purple-300 border border-purple-500/30">
                        {incident.incidentId}
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                          incident.severity === "CRITICAL"
                            ? "bg-rose-950/80 text-rose-400 border border-rose-500/30"
                            : incident.severity === "HIGH"
                            ? "bg-amber-950/80 text-amber-400 border border-amber-500/30"
                            : "bg-cyan-950/80 text-cyan-400 border border-cyan-500/30"
                        }`}
                      >
                        {incident.severity}
                      </span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#1c273a] text-slate-300 border border-[#2b3a54]">
                        {incident.status}
                      </span>
                      <span className="text-xs font-mono text-slate-400">
                        Occurrences: <strong className="text-white">{incident.occurrenceCount}x</strong>
                      </span>
                    </div>

                    <h4 className="text-sm font-bold text-white">{incident.incidentType}</h4>
                    <p className="text-xs text-slate-300">{incident.rootCauseHypothesis}</p>

                    <div className="flex items-center space-x-4 text-[11px] font-mono text-slate-400 pt-1">
                      <span>Affected: {incident.affectedCommitteeIds.join(", ")}</span>
                      <span>SLA Deadline: {incident.slaDeadlineUtc.slice(0, 16).replace("T", " ")}</span>
                      <span>Merged Alerts: {incident.sourceAlerts.join(", ")}</span>
                    </div>
                  </div>

                  {incident.status !== "RESOLVED" && incident.status !== "CLOSED" ? (
                    <button
                      type="button"
                      onClick={() => handleResolveIncident(incident.incidentId)}
                      className="px-4 py-2 bg-emerald-600/20 hover:bg-emerald-600/30 border border-emerald-500/50 rounded-lg text-xs font-mono text-emerald-300 font-semibold transition-colors"
                    >
                      Resolve Incident
                    </button>
                  ) : (
                    <span className="px-3 py-1 bg-slate-800 text-slate-400 border border-slate-700 rounded text-xs font-mono">
                      Resolved
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 5: Learnings Catalog */}
        {activeTab === "CATALOG" && (
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <h3 className="text-base font-bold text-white">Institutional Learning Catalog</h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Published lessons learned from verified decisions and realized outcomes.
                </p>
              </div>

              {/* Category Filter Pills */}
              <div className="flex items-center space-x-1.5 p-1 bg-[#111724] border border-[#1f2c42] rounded-lg">
                {(["ALL", "STRATEGY", "RISK", "GOVERNANCE", "ALLOCATION", "PROCESS"] as const).map((cat) => (
                  <button
                    key={cat}
                    type="button"
                    onClick={() => setCategoryFilter(cat)}
                    className={`px-2.5 py-1 rounded text-[11px] font-mono transition-colors ${
                      categoryFilter === cat
                        ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-semibold"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredLearnings.map((learning) => (
                <div
                  key={learning.learningId}
                  onClick={() => setSelectedLearningId(learning.learningId)}
                  className={`p-4 rounded-xl border transition-all cursor-pointer ${
                    selectedLearningId === learning.learningId
                      ? "bg-[#141d2d] border-cyan-500/60 shadow-lg shadow-cyan-950/30"
                      : "bg-[#111724] border-[#1f2c42] hover:border-[#2d3f5e]"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-950/80 text-cyan-400 border border-cyan-500/30">
                        {learning.learningId}
                      </span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#1c273a] text-slate-300 border border-[#2b3a54]">
                        {learning.category}
                      </span>
                    </div>
                    <span className="text-[11px] font-mono text-emerald-400 font-semibold">
                      +{learning.expectedOdeiImpact} ODEI pts
                    </span>
                  </div>

                  <h4 className="text-sm font-bold text-white mt-2">{learning.title}</h4>
                  <p className="text-xs text-slate-300 mt-1">{learning.description}</p>

                  <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 pt-3 border-t border-[#1d293d] mt-3">
                    <span>Source: {learning.sourceCommitteeId}</span>
                    <span>Decision: {learning.sourceDecisionId}</span>
                    <span>Ref: {learning.evidenceReference}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Selected Learning Related Artifacts Card */}
            {selectedLearning && (
              <div className="pt-4">
                <RelatedArtifactsCard
                  entityId={selectedLearning.learningId}
                  title={`Traceability Chain: ${selectedLearning.learningId} (${selectedLearning.title})`}
                />
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default function LearningIntelligencePage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-slate-400 font-mono">Loading Learning Intelligence...</div>}>
      <LearningDashboardContent />
    </Suspense>
  );
}
