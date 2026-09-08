"use client";

import React, { useState, Suspense } from "react";
import IntelligenceShell from "../../components/ui/IntelligenceShell";
import IntelligenceHeader from "../../components/ui/IntelligenceHeader";
import SeverityBadge from "../../components/ui/SeverityBadge";
import RelatedArtifactsPanel, { RelatedArtifactLink } from "../../components/ui/RelatedArtifactsPanel";
import IntelligenceEmptyState from "../../components/ui/IntelligenceEmptyState";
import { SeverityLevel } from "../../lib/ui/horizonTokens";

export interface ActionQueueItem {
  id: string;
  title: string;
  category: "ALERT" | "RECOMMENDATION" | "APPROVAL" | "ESCALATION" | "RUNBOOK";
  sourceCenter: "GOVERNANCE" | "LEARNING" | "RISK" | "RESILIENCE" | "SAFETY" | "COACHING";
  severity: SeverityLevel;
  owner: string;
  slaTarget: string;
  status: "OPEN" | "PENDING_APPROVAL" | "IN_PROGRESS" | "COMPLETED";
  description: string;
  recommendedAction: string;
  relatedArtifacts: RelatedArtifactLink[];
}

const INITIAL_ACTIONS: ActionQueueItem[] = [
  {
    id: "ACT-CRIT-001",
    title: "Replay Drift Remediation & Snapshot Chain Verification",
    category: "RUNBOOK",
    sourceCenter: "SAFETY",
    severity: "CRITICAL",
    owner: "Audit & Risk Board",
    slaTarget: "15m 00s",
    status: "OPEN",
    description: "Execute M9-RB-02 runbook lock and verify deterministic hash chain across 100 snapshot nodes.",
    recommendedAction: "Launch Runbook M9-RB-02 (Deterministic Replay Lock)",
    relatedArtifacts: [
      { id: "DEC-001", type: "DECISION", title: "Tech Allocation Tranche", href: "/decision-explorer?decisionId=DEC-001" },
      { id: "OUT-001", type: "AUDIT", title: "Alpha Return Audit Hash", href: "/audit-explorer?queryId=OUT-001" },
      { id: "RB-002", type: "RUNBOOK", title: "Snapshot Lock Protocol", href: "/resilience-intelligence?tab=runbooks" },
    ],
  },
  {
    id: "ACT-CRIT-002",
    title: "Knowledge Transfer Collapse in Credit Committee",
    category: "ALERT",
    sourceCenter: "LEARNING",
    severity: "CRITICAL",
    owner: "Governance Committee",
    slaTarget: "24h 00m",
    status: "OPEN",
    description: "Zero cross-committee knowledge transfer detected for 3 consecutive quarters. High risk of duplicated losses.",
    recommendedAction: "Launch Coaching Plan & Cross-Committee Briefing",
    relatedArtifacts: [
      { id: "COM-002", type: "COMMITTEE", title: "Credit Subcommittee", href: "/committee-intelligence?committeeId=COM-002" },
      { id: "LRN-002", type: "LEARNING", title: "Loss Avoidance Heuristics", href: "/learning-intelligence?tab=knowledge" },
      { id: "REC-002", type: "RECOMMENDATION", title: "Cross-Committee Rotation", href: "/coaching-intelligence?tab=interventions" },
    ],
  },
  {
    id: "ACT-HIGH-001",
    title: "Capital Rebalancing Plan Execution Authorization",
    category: "APPROVAL",
    sourceCenter: "GOVERNANCE",
    severity: "HIGH",
    owner: "Investment Committee",
    slaTarget: "1h 45m",
    status: "PENDING_APPROVAL",
    description: "Approve automated portfolio rebalancing allocation shift (+2.4% yield efficiency under Minervini bounds).",
    recommendedAction: "Approve Rebalancing Tranche #14",
    relatedArtifacts: [
      { id: "DEC-002", type: "DECISION", title: "Portfolio Rebalance Tranche", href: "/decision-explorer?decisionId=DEC-002" },
      { id: "RSK-002", type: "RISK", title: "Concentration Floor Violation", href: "/risks-and-groupthink?tab=matrix" },
    ],
  },
  {
    id: "ACT-HIGH-002",
    title: "Dissent Escalation SLA Breach Warning (DIS-004)",
    category: "ESCALATION",
    sourceCenter: "GOVERNANCE",
    severity: "HIGH",
    owner: "Governance Committee",
    slaTarget: "2h 30m",
    status: "OPEN",
    description: "Automatic SLA warning triggered: DIS-004 credit liquidity minority dissent requires formal committee response.",
    recommendedAction: "Schedule Emergency Dissent Review Session",
    relatedArtifacts: [
      { id: "DIS-004", type: "DECISION", title: "Credit Liquidity Dissent", href: "/dissent-explorer?dissentId=DIS-004" },
      { id: "COM-001", type: "COMMITTEE", title: "Investment Committee", href: "/committee-intelligence?committeeId=COM-001" },
    ],
  },
  {
    id: "ACT-MED-001",
    title: "Intervention Plan #3 Ratification: Groupthink De-biasing",
    category: "RECOMMENDATION",
    sourceCenter: "COACHING",
    severity: "MEDIUM",
    owner: "Executive Coaching Board",
    slaTarget: "6h 00m",
    status: "OPEN",
    description: "Deploy devil's advocate rotation protocol to alleviate emerging groupthink alignment in credit subcommittee.",
    recommendedAction: "Ratify Devil's Advocate Assignment",
    relatedArtifacts: [
      { id: "REC-001", type: "RECOMMENDATION", title: "Devil's Advocate Protocol", href: "/coaching-intelligence?tab=interventions" },
      { id: "COM-002", type: "COMMITTEE", title: "Credit Subcommittee", href: "/committee-intelligence?committeeId=COM-002" },
    ],
  },
  {
    id: "ACT-LOW-001",
    title: "L1 Metric Cache Pruning & Telemetry Indexing",
    category: "ALERT",
    sourceCenter: "RESILIENCE",
    severity: "LOW",
    owner: "Infrastructure Ops",
    slaTarget: "24h 00m",
    status: "OPEN",
    description: "Routine cleanup of transient evaluation logs and snapshot delta caches (RTO impact: 0.0s).",
    recommendedAction: "Execute Background Cache Pruning",
    relatedArtifacts: [
      { id: "RB-001", type: "RUNBOOK", title: "Cache Maintenance Runbook", href: "/resilience-intelligence?tab=runbooks" },
    ],
  },
];

function ActionCenterContent() {
  const [actions, setActions] = useState<ActionQueueItem[]>(INITIAL_ACTIONS);
  const [categoryFilter, setCategoryFilter] = useState<string>("ALL");
  const [sourceFilter, setSourceFilter] = useState<string>("ALL");
  const [severityFilter, setSeverityFilter] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedAction, setSelectedAction] = useState<ActionQueueItem | null>(null);
  const [notification, setNotification] = useState<string | null>(null);

  const handleExecuteAction = (id: string, actionName: string) => {
    setActions((prev) =>
      prev.map((a) => (a.id === id ? { ...a, status: "COMPLETED" } : a))
    );
    setNotification(`Successfully executed "${actionName}" for item ${id}. Recorded in cryptographic audit log.`);
    setTimeout(() => setNotification(null), 5000);
  };

  const filteredActions = actions
    .filter((a) => (categoryFilter === "ALL" ? true : a.category === categoryFilter))
    .filter((a) => (sourceFilter === "ALL" ? true : a.sourceCenter === sourceFilter))
    .filter((a) => (severityFilter === "ALL" ? true : a.severity === severityFilter))
    .filter((a) => {
      if (!searchQuery.trim()) return true;
      const q = searchQuery.toLowerCase();
      return (
        a.id.toLowerCase().includes(q) ||
        a.title.toLowerCase().includes(q) ||
        a.description.toLowerCase().includes(q) ||
        a.owner.toLowerCase().includes(q)
      );
    })
    .sort((a, b) => {
      const order: Record<SeverityLevel, number> = {
        CRITICAL: 0,
        HIGH: 1,
        MEDIUM: 2,
        LOW: 3,
        PASS: 4,
        INFO: 5,
      };
      return (order[a.severity] ?? 99) - (order[b.severity] ?? 99);
    });

  const activeCount = actions.filter((a) => a.status !== "COMPLETED").length;
  const criticalCount = actions.filter((a) => a.severity === "CRITICAL" && a.status !== "COMPLETED").length;
  const pendingApprovalCount = actions.filter((a) => a.status === "PENDING_APPROVAL").length;

  return (
    <IntelligenceShell
      title="Executive Action Center"
      subtitle="Unified Priority Triage Queue: Alerts, Approvals, Runbooks & Escalations"
      badge="PHASE 31-M15 CERTIFIED"
      activeNavTab="/action-center"
    >
      <div className="space-y-6">
        {/* Header with summary stats */}
        <IntelligenceHeader
          title="Executive Action Triage Queue"
          subtitle="Consolidates actionable work across Governance, Learning, Risk, Resilience, Safety, and Coaching."
          status={criticalCount > 0 ? "CRITICAL" : activeCount > 0 ? "WARNING" : "HEALTHY"}
          certification="M15 CERTIFIED"
        />

        {/* Real-time Notification Banner */}
        {notification && (
          <div
            role="alert"
            aria-live="polite"
            className="p-3.5 rounded-xl bg-emerald-950/60 border border-emerald-500/50 text-emerald-300 text-xs font-mono flex items-center justify-between animate-in fade-in"
          >
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
              <span>{notification}</span>
            </div>
            <button
              onClick={() => setNotification(null)}
              className="text-emerald-400 hover:text-white font-bold ml-4"
            >
              ✕
            </button>
          </div>
        )}

        {/* Summary Metric Bar (UH-011-AT-001) */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="p-3 rounded-xl bg-[#121B2A] border border-[#24324A]">
            <span className="text-[11px] font-mono text-slate-400 uppercase">Active Actions</span>
            <div className="text-2xl font-bold font-mono text-white mt-1">{activeCount}</div>
            <span className="text-[10px] font-mono text-slate-400">Across 6 centers</span>
          </div>
          <div className="p-3 rounded-xl bg-[#121B2A] border border-red-500/30">
            <span className="text-[11px] font-mono text-red-400 uppercase">Critical Urgency</span>
            <div className="text-2xl font-bold font-mono text-red-400 mt-1">{criticalCount}</div>
            <span className="text-[10px] font-mono text-red-400/80">SLA &lt; 24h</span>
          </div>
          <div className="p-3 rounded-xl bg-[#121B2A] border border-orange-500/30">
            <span className="text-[11px] font-mono text-orange-400 uppercase">Pending Approvals</span>
            <div className="text-2xl font-bold font-mono text-orange-400 mt-1">{pendingApprovalCount}</div>
            <span className="text-[10px] font-mono text-orange-400/80">Executive Sign-off</span>
          </div>
          <div className="p-3 rounded-xl bg-[#121B2A] border border-emerald-500/30">
            <span className="text-[11px] font-mono text-emerald-400 uppercase">Completed Today</span>
            <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">
              {actions.filter((a) => a.status === "COMPLETED").length}
            </div>
            <span className="text-[10px] font-mono text-emerald-400/80">Cryptographic Proof</span>
          </div>
        </div>

        {/* Filter & Search Bar (UH-011-AT-008) */}
        <div className="p-4 rounded-xl bg-[#121B2A] border border-[#24324A] flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-2">
            {/* Category Filter */}
            <div className="flex items-center gap-1 bg-[#0B1220] p-1 rounded-lg border border-[#24324A] text-xs font-mono">
              {['ALL', 'ALERT', 'RECOMMENDATION', 'APPROVAL', 'ESCALATION', 'RUNBOOK'].map((cat) => (
                <button
                  key={cat}
                  onClick={() => setCategoryFilter(cat)}
                  className={`px-2.5 py-1 rounded transition-colors ${
                    categoryFilter === cat
                      ? 'bg-cyan-600 text-white font-semibold'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-[#182336]'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>

            {/* Severity Filter */}
            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="px-3 py-1.5 rounded-lg bg-[#0B1220] border border-[#24324A] text-xs font-mono text-slate-200 focus:outline-none focus:ring-1 focus:ring-cyan-400"
              aria-label="Filter by Severity"
            >
              <option value="ALL">Severity: All Tiers</option>
              <option value="CRITICAL">Critical Only</option>
              <option value="HIGH">High Only</option>
              <option value="MEDIUM">Medium Only</option>
              <option value="LOW">Low Only</option>
            </select>
          </div>

          {/* Search Input */}
          <div className="relative w-full md:w-64">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search (ACT-, COM-, DEC-)..."
              className="w-full px-3 py-1.5 pl-8 rounded-lg bg-[#0B1220] border border-[#24324A] text-xs font-mono text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-cyan-400"
              aria-label="Search Actions Queue"
            />
            <span className="absolute left-2.5 top-2 text-slate-500 text-xs">🔍</span>
          </div>
        </div>

        {/* Action Items List (UH-011-AT-002, UH-011-AT-003) */}
        {filteredActions.length === 0 ? (
          <IntelligenceEmptyState
            title="No Actions Match Current Filters"
            description="Try broadening your category or severity filters to view remaining queue items."
            actionLabel="Reset All Filters"
            onAction={() => {
              setCategoryFilter("ALL");
              setSeverityFilter("ALL");
              setSearchQuery("");
            }}
          />
        ) : (
          <div className="space-y-3">
            {filteredActions.map((item) => {
              const isCompleted = item.status === "COMPLETED";
              return (
                <div
                  key={item.id}
                  className={`p-4 rounded-xl bg-[#121B2A] border transition-all ${
                    item.severity === 'CRITICAL' && !isCompleted
                      ? 'border-red-500/50 shadow-sm shadow-red-950/20'
                      : item.severity === 'HIGH' && !isCompleted
                      ? 'border-orange-500/40'
                      : 'border-[#24324A]'
                  } ${isCompleted ? 'opacity-60 bg-[#0c121c]' : ''}`}
                >
                  <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                    <div className="space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-xs font-mono font-bold text-cyan-400">{item.id}</span>
                        <SeverityBadge level={item.severity} size="sm" />
                        <span className="px-2 py-0.5 rounded bg-[#182336] text-slate-300 text-[10px] font-mono border border-[#24324A]">
                          {item.category}
                        </span>
                        <span className="px-2 py-0.5 rounded bg-[#182336] text-slate-400 text-[10px] font-mono">
                          Source: {item.sourceCenter}
                        </span>
                        {item.status === 'PENDING_APPROVAL' && (
                          <span className="px-2 py-0.5 rounded bg-orange-500/20 text-orange-300 text-[10px] font-mono font-semibold border border-orange-500/40">
                            PENDING APPROVAL
                          </span>
                        )}
                        {isCompleted && (
                          <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[10px] font-mono font-semibold border border-emerald-500/40">
                            COMPLETED
                          </span>
                        )}
                      </div>
                      <h3 className="text-sm font-semibold text-white">{item.title}</h3>
                      <p className="text-xs text-[#94A3B8] font-mono leading-relaxed">
                        {item.description}
                      </p>
                      <div className="text-xs text-cyan-300 font-mono font-medium pt-1">
                        Recommended: <span className="text-white font-semibold">{item.recommendedAction}</span>
                      </div>
                    </div>

                    {/* Action Execution Controls */}
                    <div className="flex flex-col sm:flex-row items-start sm:items-center gap-2 self-start md:self-center shrink-0">
                      <div className="text-right mr-2 hidden sm:block">
                        <div className="text-[10px] font-mono text-slate-400">SLA Window</div>
                        <div className="text-xs font-mono font-semibold text-slate-200">{item.slaTarget}</div>
                      </div>

                      <button
                        onClick={() => setSelectedAction(selectedAction?.id === item.id ? null : item)}
                        className="px-3 py-1.5 rounded-lg bg-[#182336] hover:bg-[#22324d] text-cyan-300 border border-cyan-500/30 text-xs font-mono font-semibold transition-colors"
                      >
                        {selectedAction?.id === item.id ? 'Hide Lineage' : 'Inspect Links'}
                      </button>

                      {!isCompleted ? (
                        <button
                          onClick={() => handleExecuteAction(item.id, item.recommendedAction)}
                          className="px-3.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-mono text-xs font-semibold transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-cyan-400"
                        >
                          {item.category === 'APPROVAL' ? 'Authorize' : item.category === 'RUNBOOK' ? 'Launch Runbook' : 'Execute'}
                        </button>
                      ) : (
                        <span className="px-3 py-1.5 text-xs font-mono text-emerald-400 font-semibold">
                          ✓ Resolved
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Expanded Related Artifacts Drawer (UH-011-AT-004) */}
                  {selectedAction?.id === item.id && (
                    <div className="mt-4 pt-3 border-t border-[#24324A] animate-in fade-in">
                      <RelatedArtifactsPanel
                        title={`Causal Artifacts for ${item.id}`}
                        artifacts={item.relatedArtifacts}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </IntelligenceShell>
  );
}

export default function ActionCenterPage() {
  return (
    <Suspense fallback={<div className="p-8 font-mono text-cyan-400">Loading Executive Action Center...</div>}>
      <ActionCenterContent />
    </Suspense>
  );
}
