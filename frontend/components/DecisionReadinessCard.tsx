"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import {
  DecisionReadinessResult,
  DecisionReadinessGate,
  GateState,
} from "../lib/decisionReadiness";

interface DecisionReadinessCardProps {
  symbol: string;
  readinessResult: DecisionReadinessResult;
  onSizePosition?: () => void;
  onSetAlert?: () => void;
  onOpenPreFlight?: () => void;
  className?: string;
}

function getGateBadge(state: GateState) {
  switch (state) {
    case "PASSED":
      return {
        label: "PASSED",
        icon: "✓",
        badgeClass: "bg-emerald-950/50 text-emerald-300 border-emerald-500/50",
      };
    case "BLOCKING":
      return {
        label: "BLOCKING",
        icon: "🛑",
        badgeClass: "bg-rose-950/60 text-rose-300 border-rose-500/60 animate-pulse",
      };
    case "PENDING_DEPENDENCY":
      return {
        label: "PENDING",
        icon: "⏳",
        badgeClass: "bg-slate-900/60 text-slate-400 border-slate-700/50",
      };
    case "UNAVAILABLE":
    default:
      return {
        label: "UNAVAILABLE",
        icon: "—",
        badgeClass: "bg-slate-900/40 text-slate-500 border-slate-800",
      };
  }
}

export default function DecisionReadinessCard({
  symbol,
  readinessResult,
  onSizePosition,
  onSetAlert,
  onOpenPreFlight,
  className = "",
}: DecisionReadinessCardProps) {
  const router = useRouter();
  const [expandedGate, setExpandedGate] = useState<string | null>(
    readinessResult.activeBlockingGate !== "NONE"
      ? readinessResult.activeBlockingGate
      : null
  );

  const {
    isExecutionReady,
    activeBlockingGate,
    gates,
    nextRequiredCondition,
    negativeGuidance,
    primaryAction,
  } = readinessResult;

  const handlePrimaryCTA = () => {
    switch (primaryAction.actionType) {
      case "SIZE_POSITION":
        if (onSizePosition) onSizePosition();
        break;
      case "SET_PULLBACK_ALERT":
      case "SET_BUY_ZONE_ALERT":
        if (onSetAlert) onSetAlert();
        break;
      case "EXPLORE_RADAR":
        router.push("/radar");
        break;
      default:
        router.push("/radar");
        break;
    }
  };

  return (
    <section
      data-testid="decision-readiness-card"
      aria-labelledby="readiness-heading"
      className={`bg-[#0d131f] border border-[#1e2a3f] rounded-xl p-2 sm:p-5 shadow-xl space-y-1.5 sm:space-y-4 font-sans text-slate-200 ${className}`}
    >
      {/* ── Header: Title & Execution Readiness Status ──────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-1.5 sm:gap-2 border-b border-[#1b263b] pb-1.5 sm:pb-3">
        <div className="flex items-center space-x-1.5 sm:space-x-2">
          <span className="text-sm sm:text-base" aria-hidden="true">🚦</span>
          <h3 id="readiness-heading" className="text-xs sm:text-sm font-bold tracking-tight text-white uppercase font-mono">
            Decision Readiness Progression
          </h3>
        </div>

        {/* Binary State Indicator (Non-color only: explicit text + icon) */}
        <div
          data-testid="readiness-status-badge"
          className={`inline-flex items-center gap-1 sm:gap-1.5 px-2 sm:px-2.5 py-0.5 sm:py-1 rounded-full text-[11px] sm:text-xs font-mono font-bold border ${
            isExecutionReady
              ? "bg-emerald-950/60 text-emerald-400 border-emerald-500/50"
              : "bg-amber-950/40 text-amber-400 border-amber-600/50"
          }`}
        >
          <span aria-hidden="true">{isExecutionReady ? "✓" : "⏳"}</span>
          <span>{isExecutionReady ? "EXECUTION READY" : "SETUP IN PROGRESS"}</span>
        </div>
      </div>

      {/* ── Protective Negative Guidance Banner (If Triggered by Evidence) ─── */}
      {negativeGuidance && (
        <div
          data-testid="negative-guidance-banner"
          role="alert"
          className="p-1.5 sm:p-3 rounded-lg bg-rose-950/40 border border-rose-500/60 text-xs text-rose-200 flex items-start gap-2 sm:gap-2.5 shadow-inner"
        >
          <span className="text-base shrink-0" aria-hidden="true">⚠️</span>
          <div className="space-y-0.5">
            <strong className="font-bold text-rose-300 block font-mono text-[11px] tracking-wide uppercase">
              Protective Guidance
            </strong>
            <p className="leading-snug sm:leading-relaxed text-rose-200 font-sans text-xs">{negativeGuidance}</p>
          </div>
        </div>
      )}

      {/* ── 3-Gate Progression Ladder ────────────────────────────────────────── */}
      <div className="space-y-1 sm:space-y-2" role="list" aria-label="Readiness Gates">
        {gates.map((gate: DecisionReadinessGate, index: number) => {
          const badge = getGateBadge(gate.state);
          const isCurrentBlocker = gate.gateId === activeBlockingGate;
          const isExpanded = expandedGate === gate.gateId;

          return (
            <div
              key={gate.gateId}
              role="listitem"
              data-testid={`gate-item-${index + 1}`}
              className={`rounded-lg border p-1.5 sm:p-3 transition-colors ${
                isCurrentBlocker
                  ? "bg-[#141b2a] border-rose-500/50"
                  : gate.state === "PASSED"
                  ? "bg-[#0b101a] border-emerald-900/30"
                  : "bg-[#090d15] border-[#182234]"
              }`}
            >
              <div
                className="flex items-center justify-between gap-1.5 sm:gap-2 cursor-pointer min-h-[44px]"
                onClick={() => setExpandedGate(isExpanded ? null : gate.gateId)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setExpandedGate(isExpanded ? null : gate.gateId);
                  }
                }}
                aria-expanded={isExpanded}
              >
                <div className="flex items-center space-x-1 sm:space-x-2">
                  <span className="text-xs font-mono font-bold text-slate-400">
                    Gate {index + 1}:
                  </span>
                  <span className="text-xs font-bold text-slate-100 sm:inline hidden">
                    {gate.displayName}
                  </span>
                  <span className="text-xs font-bold text-slate-100 sm:hidden inline">
                    {gate.displayName.replace(/^\d+\.\s*/, "")}
                  </span>
                </div>

                <div className="flex items-center space-x-1.5 sm:space-x-2 shrink-0">
                  <span
                    data-testid={`gate-badge-${index + 1}`}
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono font-bold border ${badge.badgeClass}`}
                  >
                    <span aria-hidden="true">{badge.icon}</span>
                    <span>{badge.label}</span>
                  </span>
                  <span className="text-slate-500 text-xs" aria-hidden="true">
                    {isExpanded ? "▲" : "▼"}
                  </span>
                </div>
              </div>

              {/* Progressive Disclosure: Explanation Text */}
              {isExpanded && (
                <div
                  data-testid={`gate-explanation-${index + 1}`}
                  className="mt-1 pt-1 sm:mt-2 sm:pt-2 border-t border-slate-800/60 text-xs text-slate-300 leading-snug sm:leading-relaxed font-sans"
                >
                  {gate.explanation}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* ── Active Blocker & Next Required Condition ─────────────────────────── */}
      <div
        data-testid="next-condition-box"
        className="p-1.5 sm:p-3 rounded-lg bg-[#080d16] border border-[#1b2537] text-xs space-y-1"
        aria-live="polite"
      >
        <div className="flex items-center justify-between">
          <span className="text-[10px] uppercase font-mono font-bold text-slate-400">
            {activeBlockingGate === "NONE" ? "Action Clearance" : "Active Blocker Requirement"}
          </span>
          {activeBlockingGate !== "NONE" && (
            <span
              data-testid="active-blocker-tag"
              className="text-[10px] font-mono font-bold text-rose-400 bg-rose-950/40 px-1.5 py-0.5 rounded border border-rose-800/40"
            >
              ACTIVE BLOCKER
            </span>
          )}
        </div>
        <p className="text-slate-200 leading-snug sm:leading-relaxed font-sans text-xs">{nextRequiredCondition}</p>
      </div>

      {/* ── Operational Primary CTA & Pre-Flight Integration ─────────────────── */}
      <div className="pt-1.5 sm:pt-2 border-t border-[#1b2537] flex flex-wrap items-center justify-between gap-1.5 sm:gap-2">
        <div className="text-[11px] text-slate-400 font-sans hidden sm:block">
          <span>Action: </span>
          <span className="text-slate-300">{primaryAction.reason}</span>
        </div>

        <div className="flex items-center gap-2">
          {/* Secondary Action: Pre-Flight Checklist */}
          {onOpenPreFlight && (
            <button
              type="button"
              onClick={onOpenPreFlight}
              data-testid="readiness-preflight-btn"
              className="min-h-[44px] px-3 py-2 rounded-lg text-xs font-bold transition-all border border-slate-700 bg-slate-800/60 hover:bg-slate-700 text-slate-200 active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"
            >
              <span>✈️ Pre-Flight</span>
            </button>
          )}

          {/* Primary Action Button */}
          <button
            type="button"
            onClick={handlePrimaryCTA}
            data-testid="readiness-primary-cta"
            className={`min-h-[44px] px-4 py-2 rounded-lg text-xs font-bold transition-all shadow-md active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 ${
              isExecutionReady
                ? "bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-black shadow-emerald-950/40"
                : "bg-cyan-600/30 hover:bg-cyan-600 hover:text-slate-950 text-cyan-300 border border-cyan-500/50"
            }`}
          >
            <span>{primaryAction.label}</span>
          </button>
        </div>
      </div>
    </section>
  );
}
