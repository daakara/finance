"use client";

import React, { useState, Suspense } from "react";
import IntelligenceShell from "../../components/ui/IntelligenceShell";
import HorizonCard from "../../components/ui/HorizonCard";
import HorizonMetricCard from "../../components/ui/HorizonMetricCard";
import HorizonNarrativeCard from "../../components/ui/HorizonNarrativeCard";
import IntelligenceLoadingState from "../../components/ui/IntelligenceLoadingState";
import IntelligenceSuccessState from "../../components/ui/IntelligenceSuccessState";
import IntelligenceErrorState from "../../components/ui/IntelligenceErrorState";
import {
  CANONICAL_INBOX_ITEMS,
  deduplicateInboxItems,
  rankInboxItems,
  lockInboxItemForExecution,
  executeInboxAction,
  getInboxMetrics,
} from "../../lib/productivity/decisionInboxEngine";
import {
  generateExecutiveBriefing,
} from "../../lib/productivity/briefingGenerationEngine";
import type {
  DecisionInboxItem,
  DecisionCategory,
  BriefingPackage,
  BriefingAudience,
} from "../../types/executive-workspace";

function DecisionInboxContent() {
  const [items, setItems] = useState<DecisionInboxItem[]>(CANONICAL_INBOX_ITEMS);
  const [selectedCategory, setSelectedCategory] = useState<string>("ALL");
  const [simulateAuditFailure, setSimulateAuditFailure] = useState<boolean>(false);
  const [executionReceipt, setExecutionReceipt] = useState<{ executionId: string; auditHash: string; action: string } | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedAudience, setSelectedAudience] = useState<BriefingAudience>("EXECUTIVE");
  const [generatedBriefing, setGeneratedBriefing] = useState<BriefingPackage | null>(null);

  const metrics = getInboxMetrics(items);

  const categories: { label: string; value: string }[] = [
    { label: "All Items", value: "ALL" },
    { label: "Approvals", value: "APPROVAL" },
    { label: "Escalations", value: "ESCALATION" },
    { label: "Recommendations", value: "RECOMMENDATION" },
    { label: "Runbooks", value: "RUNBOOK" },
    { label: "Optimizations", value: "OPTIMIZATION" },
  ];

  const filteredItems = selectedCategory === "ALL"
    ? items
    : items.filter((i) => i.category === selectedCategory);

  const handleDeduplicate = () => {
    const res = deduplicateInboxItems(items);
    setItems(res.deduplicated);
  };

  const handleRank = () => {
    const ranked = rankInboxItems(items);
    setItems(ranked);
  };

  const handleExecute = (itemId: string, actionType: string = "APPROVE") => {
    setErrorMessage(null);
    setExecutionReceipt(null);

    // 1. Lock item
    const lockRes = lockInboxItemForExecution(items, itemId, "USR-CURRENT-EXEC");
    if (!lockRes.success) {
      setErrorMessage(lockRes.error || "Lock failed.");
      return;
    }

    // 2. Execute with audit fail-close check
    const execRes = executeInboxAction(lockRes.items, itemId, actionType, {
      simulateAuditFailure,
      executorId: "USR-CURRENT-EXEC",
    });

    if (!execRes.success && execRes.error) {
      setErrorMessage(execRes.error.message);
      // Ensure item is not permanently locked if execution errored
      setItems(items);
      return;
    }

    if (execRes.receipt) {
      setItems(execRes.items);
      setExecutionReceipt(execRes.receipt);
    }
  };

  const handleGenerateBriefing = (aud: BriefingAudience = selectedAudience) => {
    setSelectedAudience(aud);
    const result = generateExecutiveBriefing({ audience: aud });
    setGeneratedBriefing(result.briefing);
  };

  return (
    <IntelligenceShell
      title="Unified Decision Inbox"
      subtitle="ARX Horizon Executive OS - Cross-Platform Decision Consolidation & Triage"
    >
      {/* Metrics Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
        <HorizonMetricCard
          label="Total Items"
          value={metrics.total}
          target="All Ingested"
          severity="PASS"
        />
        <HorizonMetricCard
          label="Pending Triage"
          value={metrics.pending}
          target="Action Needed"
          severity={metrics.pending > 3 ? "WARN" : "PASS"}
        />
        <HorizonMetricCard
          label="Critical Items"
          value={metrics.critical}
          target="Sev 1 Alerts"
          severity={metrics.critical > 0 ? "CRITICAL" : "PASS"}
        />
        <HorizonMetricCard
          label="Urgent SLA (<60m)"
          value={metrics.urgentSla}
          target="Breach Risk"
          severity={metrics.urgentSla > 0 ? "CRITICAL" : "PASS"}
        />
        <HorizonMetricCard
          label="Executing"
          value={metrics.executing}
          target="In-Flight Locks"
          severity="PASS"
        />
        <HorizonMetricCard
          label="Resolved"
          value={metrics.resolved}
          target="Audit Certified"
          severity="PASS"
        />
      </div>

      {/* Control Bar: Categories, Deduplication, Priority Sort, Audit Outage Toggle */}
      <div className="p-4 rounded-2xl bg-[#121B2A] border border-[#24324A] flex flex-wrap items-center justify-between gap-4">
        {/* Category Filter Tabs */}
        <div className="flex flex-wrap items-center gap-1.5">
          {categories.map((c) => (
            <button
              key={c.value}
              onClick={() => setSelectedCategory(c.value)}
              className={`px-3 py-1.5 rounded-xl text-xs font-medium transition-all ${
                selectedCategory === c.value
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                  : "bg-[#182336] text-slate-400 border border-[#24324A] hover:text-slate-200"
              }`}
            >
              {c.label}
            </button>
          ))}
        </div>

        {/* Action Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={handleDeduplicate}
            className="px-3 py-1.5 rounded-xl bg-[#182336] text-xs font-mono text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/10 transition-colors"
          >
            Deduplicate Feeds (DI-EC-01)
          </button>
          <button
            onClick={handleRank}
            className="px-3 py-1.5 rounded-xl bg-[#182336] text-xs font-mono text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/10 transition-colors"
          >
            Sort Priority & SLA (DI-EC-02)
          </button>
          <label className="flex items-center gap-2 cursor-pointer text-xs font-mono text-slate-300 ml-2">
            <input
              type="checkbox"
              checked={simulateAuditFailure}
              onChange={(e) => setSimulateAuditFailure(e.target.checked)}
              className="rounded border-[#24324A] bg-[#182336] text-rose-400 focus:ring-0 w-4 h-4 cursor-pointer"
            />
            <span className={simulateAuditFailure ? "text-rose-400 font-bold" : ""}>
              Simulate Audit Outage (DI-EC-06)
            </span>
          </label>
        </div>
      </div>

      {/* Execution Receipt Alert */}
      {executionReceipt && (
        <IntelligenceSuccessState
          title="Decision Action Executed & Certified"
          message={`Action ${executionReceipt.action} executed with zero state drift. Certified audit receipt generated.`}
          auditHash={executionReceipt.auditHash}
          certificationId={executionReceipt.executionId}
        />
      )}

      {/* Audit Failure Fail-Close Error Alert */}
      {errorMessage && (
        <IntelligenceErrorState
          errorCode="DI-ERR-002"
          failureClass="Action Blocked Fail-Closed"
          message={errorMessage}
          recoverySteps={[
            "Verify that audit service is reachable before retrying action",
            "Ensure no concurrent execution is holding an active lock",
            "Contact Platform Governance if failure persists",
          ]}
        />
      )}

      {/* Decision Inbox Table / Card List */}
      <HorizonCard
        title="Active Decision Queue"
        subtitle={`Showing ${filteredItems.length} items`}
      >
        <div className="space-y-3">
          {filteredItems.map((item) => {
            const sevBadge =
              item.severity === "CRITICAL"
                ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                : item.severity === "HIGH"
                ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                : "bg-cyan-500/20 text-cyan-300 border-cyan-500/40";

            const statusBadge =
              item.status === "RESOLVED"
                ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                : item.status === "EXECUTING"
                ? "bg-purple-500/20 text-purple-300 border-purple-500/40"
                : item.status === "QUARANTINED"
                ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                : "bg-slate-700/40 text-slate-300 border-slate-600/40";

            return (
              <div
                key={item.itemId}
                className="p-4 rounded-xl bg-[#182336] border border-[#24324A] hover:border-[#334769] transition-all flex flex-col md:flex-row md:items-center justify-between gap-4"
              >
                <div className="space-y-1.5 max-w-3xl">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${sevBadge}`}>
                      {item.severity}
                    </span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${statusBadge}`}>
                      {item.status}
                    </span>
                    <span className="text-xs font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                      {item.category}
                    </span>
                    <h3 className="text-sm font-semibold text-white">{item.title}</h3>
                  </div>

                  <p className="text-xs text-slate-300 leading-relaxed">{item.description}</p>

                  <div className="flex flex-wrap items-center gap-3 text-[11px] font-mono text-slate-400">
                    <span>Source: <strong className="text-slate-300">{item.sourceCenter}</strong></span>
                    <span>Entity ID: <strong className="text-slate-300">{item.entityId}</strong></span>
                    <span>Owner: {item.owner}</span>
                    <span>SLA: <strong className="text-amber-300">{item.slaTargetMinutes}m</strong></span>
                    {item.duplicateSources && item.duplicateSources.length > 1 && (
                      <span className="text-purple-300 bg-purple-500/10 px-1.5 py-0.5 rounded border border-purple-500/30">
                        Merged from {item.duplicateSources.length} feeds
                      </span>
                    )}
                  </div>
                </div>

                {/* Execution Buttons */}
                <div className="flex items-center gap-2 shrink-0">
                  {item.status === "PENDING" && (
                    <>
                      <button
                        onClick={() => handleExecute(item.itemId, "APPROVE")}
                        className="px-3 py-1.5 rounded-xl bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 text-xs font-mono font-semibold border border-emerald-500/30 transition-colors"
                      >
                        Approve
                      </button>
                      <button
                        onClick={() => handleExecute(item.itemId, "ESCALATE")}
                        className="px-3 py-1.5 rounded-xl bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 text-xs font-mono font-semibold border border-amber-500/30 transition-colors"
                      >
                        Escalate
                      </button>
                    </>
                  )}
                  {item.status === "RESOLVED" && (
                    <span className="text-xs font-mono text-emerald-400">✓ Completed</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </HorizonCard>

      {/* One-Click Executive Briefing Generator Section */}
      <HorizonCard
        title="One-Click Executive Briefing Generator"
        subtitle="Instant narrative synthesis with 100% evidence lineage and deterministic replay"
        actions={
          <div className="flex items-center gap-2">
            {(["EXECUTIVE", "BOARD", "COMMITTEE", "INCIDENT"] as BriefingAudience[]).map((aud) => (
              <button
                key={aud}
                onClick={() => handleGenerateBriefing(aud)}
                className={`px-3 py-1 rounded-xl text-xs font-mono font-medium transition-all ${
                  selectedAudience === aud && generatedBriefing
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                    : "bg-[#182336] text-slate-400 border border-[#24324A] hover:text-slate-200"
                }`}
              >
                {aud}
              </button>
            ))}
          </div>
        }
      >
        {generatedBriefing ? (
          <div className="space-y-4">
            <HorizonNarrativeCard
              headline={generatedBriefing.headline}
              executiveSummary={generatedBriefing.executiveSummary}
              category={`${generatedBriefing.audience} BRIEFING`}
              auditHash={generatedBriefing.replayHash}
              replayDeterministic={true}
              metrics={[
                { label: "OHI Score", value: "88.4 / 100", change: "+1.8%", status: "healthy" },
                { label: "DIR Ratio", value: "89.2%", change: "+3.2%", status: "healthy" },
                { label: "Active Risks", value: "2", status: "warning" },
                { label: "Policy Violations", value: "0", status: "healthy" },
              ]}
              findings={generatedBriefing.findings.map((f) => ({
                id: f.findingId,
                title: f.category,
                detail: f.text,
                severity: "HIGH",
                verified: f.supported,
              }))}
              recommendations={generatedBriefing.recommendations.map((r, idx) => ({
                id: `REC-${idx + 1}`,
                action: r,
                impact: "HIGH",
                confidence: 0.94,
              }))}
            />
          </div>
        ) : (
          <div className="p-8 text-center bg-[#152033] rounded-2xl border border-dashed border-[#24324A] space-y-3">
            <p className="text-sm text-slate-300">
              Select an audience above to synthesize a tailored executive briefing package with full cryptographic lineage.
            </p>
            <button
              onClick={() => handleGenerateBriefing("EXECUTIVE")}
              className="px-4 py-2 rounded-xl bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 text-xs font-mono font-bold border border-cyan-500/30 transition-colors"
            >
              Generate Executive Intelligence Flash
            </button>
          </div>
        )}
      </HorizonCard>
    </IntelligenceShell>
  );
}

export default function DecisionInboxPage() {
  return (
    <Suspense fallback={<IntelligenceLoadingState message="Loading Decision Inbox..." />}>
      <DecisionInboxContent />
    </Suspense>
  );
}
