"use client";

import { useState } from "react";
import Link from "next/link";
import {
  AlertWorkflowItem,
  AlertLifecycleStatus,
} from "../../types/navigation-intelligence";
import { ESCALATION_MATRIX } from "../../lib/governance/alertWorkflowEngine";

export interface AlertWorkflowModalProps {
  alert: AlertWorkflowItem | null;
  isOpen: boolean;
  onClose: () => void;
  onStatusChange?: (alertId: string, newStatus: AlertLifecycleStatus, notes?: string) => void;
}

export default function AlertWorkflowModal({
  alert,
  isOpen,
  onClose,
  onStatusChange,
}: AlertWorkflowModalProps) {
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [resolutionNotes, setResolutionNotes] = useState<string>("");

  if (!isOpen || !alert) return null;

  const escalation = ESCALATION_MATRIX[alert.severity];

  const toggleStep = (stepIdx: number) => {
    setCompletedSteps((prev) =>
      prev.includes(stepIdx) ? prev.filter((i) => i !== stepIdx) : [...prev, stepIdx]
    );
  };

  const handleTransition = (status: AlertLifecycleStatus) => {
    onStatusChange?.(alert.alertId, status, resolutionNotes);
    if (status === "CLOSED" || status === "RESOLVED") {
      onClose();
    }
  };

  const getSeverityBadge = (sev: string) => {
    switch (sev) {
      case "CRITICAL":
        return "bg-rose-950/80 border-rose-500/60 text-rose-400 animate-pulse";
      case "HIGH":
        return "bg-amber-950/80 border-amber-500/60 text-amber-400";
      case "MEDIUM":
        return "bg-purple-950/80 border-purple-500/60 text-purple-400";
      case "LOW":
        return "bg-cyan-950/80 border-cyan-500/60 text-cyan-400";
      default:
        return "bg-slate-900 border-slate-700 text-slate-400";
    }
  };

  const allStepsCompleted = completedSteps.length === alert.playbook.remediationSteps.length;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm font-mono animate-in fade-in duration-150">
      <div
        className="bg-[#0e131d] border border-cyan-500/30 w-full max-w-2xl rounded-2xl shadow-2xl shadow-cyan-950/60 overflow-hidden flex flex-col max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#202d44] px-5 py-4 bg-[#131b29]">
          <div className="flex items-center space-x-2.5">
            <span className={`px-2.5 py-0.5 rounded border text-xs font-bold ${getSeverityBadge(alert.severity)}`}>
              {alert.severity}
            </span>
            <span className="text-sm font-bold text-slate-100">{alert.alertCode}</span>
            <span className="text-xs text-slate-400 font-mono">({alert.alertId})</span>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded-lg bg-[#1c273a] hover:bg-[#283852] text-slate-400 hover:text-slate-200 text-xs transition-colors"
          >
            ✕ ESC
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-5 overflow-y-auto space-y-4 text-xs">
          {/* Title & Description */}
          <div>
            <h3 className="text-base font-bold text-slate-100">{alert.title}</h3>
            <p className="text-xs text-slate-300 mt-1 leading-relaxed">{alert.summary}</p>
          </div>

          {/* Context Ribbon: Target SLA, Escalation, Affected Artifact */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 bg-[#0c1017] p-3 rounded-xl border border-[#202d44]">
            <div>
              <span className="text-[10px] text-slate-500 uppercase block">Target SLA</span>
              <span className="font-bold text-rose-400 mt-0.5 block">{alert.playbook.targetSla}</span>
            </div>

            <div>
              <span className="text-[10px] text-slate-500 uppercase block">Escalation Channel</span>
              <span className="font-bold text-cyan-300 mt-0.5 block truncate">
                {escalation.destination}
              </span>
            </div>

            <div>
              <span className="text-[10px] text-slate-500 uppercase block">Impacted Artifact</span>
              <div className="flex items-center space-x-1.5 mt-0.5">
                <span className="font-bold text-purple-300">{alert.affectedArtifactId}</span>
                <Link
                  href={`/audit-explorer?queryId=${alert.affectedArtifactId}`}
                  className="text-[10px] text-cyan-400 hover:underline"
                >
                  Audit &rarr;
                </Link>
              </div>
            </div>
          </div>

          {/* Remediation Playbook Checklist */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider">
                Prescribed Remediation Playbook (AW-08)
              </span>
              <span className="text-[10px] text-slate-400">
                {completedSteps.length} of {alert.playbook.remediationSteps.length} Steps Executed
              </span>
            </div>

            <div className="space-y-2">
              {alert.playbook.remediationSteps.map((step, idx) => {
                const isDone = completedSteps.includes(idx);
                return (
                  <label
                    key={idx}
                    className={`flex items-start space-x-3 p-3 rounded-lg border transition-colors cursor-pointer ${
                      isDone
                        ? "bg-emerald-950/30 border-emerald-500/40 text-slate-200"
                        : "bg-[#0c1017] border-[#202d44] text-slate-300 hover:border-slate-600"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={isDone}
                      onChange={() => toggleStep(idx)}
                      className="mt-0.5 rounded border-slate-700 bg-slate-800 text-cyan-500 focus:ring-0 cursor-pointer"
                    />
                    <span className={`text-xs leading-relaxed ${isDone ? "line-through text-slate-400" : ""}`}>
                      {step}
                    </span>
                  </label>
                );
              })}
            </div>
          </div>

          {/* Closure Condition Box */}
          <div className="p-3 rounded-xl bg-[#0c1017] border border-[#202d44] space-y-1">
            <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider block">
              Closure Condition Benchmark
            </span>
            <p className="text-xs text-slate-300">{alert.playbook.closureCondition}</p>
          </div>

          {/* Resolution Notes Input */}
          <div className="space-y-1">
            <label className="text-[10px] text-slate-400 uppercase tracking-wider block">
              Governance Resolution Log / Evidence Notes
            </label>
            <textarea
              value={resolutionNotes}
              onChange={(e) => setResolutionNotes(e.target.value)}
              placeholder="Record actions taken, committee minutes, or audit verification snapshot hashes..."
              rows={2}
              className="w-full px-3 py-2 bg-[#0c1017] border border-[#202d44] rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono"
            />
          </div>
        </div>

        {/* Footer with Status Lifecycle Transitions */}
        <div className="px-5 py-3.5 bg-[#131b29] border-t border-[#202d44] flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <span className="text-[10px] text-slate-400">Current Status:</span>
            <span className="px-2 py-0.5 rounded bg-[#1c273a] text-cyan-300 font-bold text-xs">
              {alert.status}
            </span>
          </div>

          <div className="flex items-center space-x-2">
            {alert.status === "OPEN" && (
              <button
                type="button"
                onClick={() => handleTransition("INVESTIGATING")}
                className="px-3 py-1.5 rounded-lg bg-[#1f2c42] hover:bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 font-bold text-xs transition-colors"
              >
                Begin Investigation &rarr;
              </button>
            )}

            {alert.status === "INVESTIGATING" && (
              <button
                type="button"
                onClick={() => handleTransition("MITIGATING")}
                className="px-3 py-1.5 rounded-lg bg-purple-900/60 hover:bg-purple-800/60 border border-purple-500/40 text-purple-200 font-bold text-xs transition-colors"
              >
                Apply Mitigation &rarr;
              </button>
            )}

            {(alert.status === "MITIGATING" || alert.status === "INVESTIGATING") && (
              <button
                type="button"
                onClick={() => handleTransition("RESOLVED")}
                disabled={!allStepsCompleted}
                className={`px-3 py-1.5 rounded-lg font-bold text-xs transition-colors ${
                  allStepsCompleted
                    ? "bg-emerald-600 hover:bg-emerald-500 text-slate-950 cursor-pointer"
                    : "bg-slate-800 text-slate-500 cursor-not-allowed"
                }`}
              >
                Mark Resolved &check;
              </button>
            )}

            {alert.status === "RESOLVED" && (
              <button
                type="button"
                onClick={() => handleTransition("CLOSED")}
                className="px-3 py-1.5 rounded-lg bg-emerald-700 hover:bg-emerald-600 text-slate-950 font-bold text-xs transition-colors"
              >
                Close Alert
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
