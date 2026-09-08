"use client";

import React, { useState, Suspense } from "react";
import IntelligenceShell from "../../components/ui/IntelligenceShell";
import HorizonCard from "../../components/ui/HorizonCard";
import HorizonActionItem from "../../components/ui/HorizonActionItem";
import IntelligenceEmptyState from "../../components/ui/IntelligenceEmptyState";
import { SeverityLevel } from "../../lib/ui/horizonTokens";

interface ActionQueueItem {
  id: string;
  title: string;
  category: "ALERT" | "RECOMMENDATION" | "APPROVAL" | "ESCALATION" | "RUNBOOK";
  severity: SeverityLevel;
  owner: string;
  slaTarget: string;
  description: string;
}

const INITIAL_ACTIONS: ActionQueueItem[] = [
  {
    id: "ACT-CRIT-001",
    title: "Replay Drift Remediation & Snapshot Verification",
    category: "RUNBOOK",
    severity: "CRITICAL",
    owner: "Audit & Risk Board",
    slaTarget: "15m 00s",
    description: "Execute M9-RB-02 runbook lock and verify deterministic hash chain across 100 snapshot nodes.",
  },
  {
    id: "ACT-HIGH-002",
    title: "Capital Rebalancing Plan Execution Authorization",
    category: "APPROVAL",
    severity: "HIGH",
    owner: "Investment Committee",
    slaTarget: "1h 45m",
    description: "Approve automated portfolio rebalancing allocation shift (+2.4% yield efficiency under Minervini bounds).",
  },
  {
    id: "ACT-HIGH-003",
    title: "Dissent Escalation SLA Breach Warning",
    category: "ESCALATION",
    severity: "HIGH",
    owner: "Governance Committee",
    slaTarget: "2h 30m",
    description: "Automatic SLA warning triggered: DIS-004 credit liquidity minority dissent requires formal committee response.",
  },
  {
    id: "ACT-MED-004",
    title: "Intervention Plan #3 Ratification",
    category: "RECOMMENDATION",
    severity: "MEDIUM",
    owner: "Executive Coaching Board",
    slaTarget: "6h 00m",
    description: "Deploy devil's advocate rotation protocol to alleviate emerging groupthink alignment in credit subcommittee.",
  },
  {
    id: "ACT-LOW-005",
    title: "L1 Metric Cache Pruning & Telemetry Indexing",
    category: "ALERT",
    severity: "LOW",
    owner: "Infrastructure Ops",
    slaTarget: "24h 00m",
    description: "Routine cleanup of transient evaluation logs and snapshot delta caches (RTO impact: 0.0s).",
  },
];

function ActionCenterContent() {
  const [actions, setActions] = useState<ActionQueueItem[]>(INITIAL_ACTIONS);
  const [categoryFilter, setCategoryFilter] = useState<string>("ALL");
  const [severityFilter, setSeverityFilter] = useState<string>("ALL");
  const [notification, setNotification] = useState<string | null>(null);

  const handleExecute = (id: string) => {
    setActions((prev) => prev.filter((a) => a.id !== id));
    setNotification(`Successfully executed action ${id}. State mutation recorded in immutable audit log.`);
    setTimeout(() => setNotification(null), 4000);
  };

  const filteredActions = actions
    .filter((a) => (categoryFilter === "ALL" ? true : a.category === categoryFilter))
    .filter((a) => (severityFilter === "ALL" ? true : a.severity === severityFilter))
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

  return (
    <IntelligenceShell
      title="Executive Action Center"
      subtitle="Unified Priority Triage Queue: Alerts, Approvals, Escalations & Runbooks"
      badge="PHASE 31-M11 CERTIFIED"
      activeNavTab="/action-center"
    >
      {notification && (
        <div
          role="status"
          className="rounded-xl border border-emerald-500/50 bg-emerald-950/60 p-4 text-xs font-mono text-emerald-200 flex items-center justify-between"
        >
          <span>{notification}</span>
          <button onClick={() => setNotification(null)} className="text-emerald-400 font-bold hover:underline">
            Dismiss
          </button>
        </div>
      )}

      {/* Filter and Triage Controls */}
      <HorizonCard>
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          {/* Category Tabs */}
          <div className="flex items-center flex-wrap gap-1 font-mono text-xs">
            {["ALL", "ALERT", "RECOMMENDATION", "APPROVAL", "ESCALATION", "RUNBOOK"].map((cat) => {
              const count = cat === "ALL" ? actions.length : actions.filter((a) => a.category === cat).length;
              return (
                <button
                  key={cat}
                  onClick={() => setCategoryFilter(cat)}
                  className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 ${
                    categoryFilter === cat
                      ? "bg-cyan-600 text-white font-semibold"
                      : "bg-[#182336] text-slate-300 hover:bg-[#20304a] border border-[#24324A]"
                  }`}
                >
                  <span>{cat}</span>
                  <span className="px-1.5 py-0.2 rounded bg-black/40 text-[10px]">{count}</span>
                </button>
              );
            })}
          </div>

          {/* Severity Dropdown */}
          <div className="flex items-center gap-2 font-mono text-xs shrink-0">
            <span className="text-slate-400">Severity:</span>
            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="px-3 py-1.5 rounded-lg bg-[#182336] text-slate-200 border border-[#24324A] focus:outline-none focus:ring-1 focus:ring-cyan-400"
            >
              <option value="ALL">All Severities</option>
              <option value="CRITICAL">Critical Only</option>
              <option value="HIGH">High Only</option>
              <option value="MEDIUM">Medium Only</option>
              <option value="LOW">Low Only</option>
            </select>
          </div>
        </div>
      </HorizonCard>

      {/* Actions List */}
      <div className="space-y-3">
        {filteredActions.length === 0 ? (
          <IntelligenceEmptyState
            title="No Actions Pending in Triage Queue"
            description="All scheduled runbooks, approvals, and alert escalations have been resolved within target SLAs."
            actionLabel="Reset Queue Filters"
            onAction={() => {
              setCategoryFilter("ALL");
              setSeverityFilter("ALL");
            }}
          />
        ) : (
          filteredActions.map((item) => (
            <HorizonActionItem
              key={item.id}
              id={item.id}
              title={item.title}
              category={item.category}
              severity={item.severity}
              owner={item.owner}
              slaTarget={item.slaTarget}
              description={item.description}
              onExecute={handleExecute}
              onReview={(id) => alert(`Reviewing details for action ${id}`)}
            />
          ))
        )}
      </div>
    </IntelligenceShell>
  );
}

export default function ActionCenterPage() {
  return (
    <Suspense fallback={<div className="p-8 font-mono text-cyan-400">Loading Action Center...</div>}>
      <ActionCenterContent />
    </Suspense>
  );
}
