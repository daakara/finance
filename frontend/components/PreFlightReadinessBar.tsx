"use client";

import React, { useState } from "react";
import {
  evaluatePreFlightChecks,
  PreFlightChecklistInputs,
  PreFlightEvaluationResult,
  PreFlightCheckItem,
} from "../lib/preflightEvaluator";

export interface PreFlightReadinessBarProps extends PreFlightChecklistInputs {
  onOpenModal?: () => void;
  className?: string;
  isDrawerExpanded?: boolean;
  onToggleDrawer?: (expanded: boolean) => void;
}

export default function PreFlightReadinessBar(props: PreFlightReadinessBarProps) {
  const { onOpenModal, className = "", isDrawerExpanded: controlledDrawerExpanded, onToggleDrawer, ...evalInputs } = props;
  const [internalDrawerExpanded, setInternalDrawerExpanded] = useState<boolean>(false);
  const isDrawerExpanded = controlledDrawerExpanded !== undefined ? controlledDrawerExpanded : internalDrawerExpanded;
  const [vernacularMode, setVernacularMode] = useState<"PLAIN_ENGLISH" | "PRO_QUANT">("PLAIN_ENGLISH");

  const handleToggleDrawer = () => {
    if (onToggleDrawer) {
      onToggleDrawer(!isDrawerExpanded);
    } else {
      setInternalDrawerExpanded((prev) => !prev);
    }
  };

  React.useEffect(() => {
    try {
      const saved = localStorage.getItem("ARX_VERNACULAR_MODE") as "PLAIN_ENGLISH" | "PRO_QUANT" | null;
      if (saved) setVernacularMode(saved);
    } catch {}

    const handleVernacular = (e: Event) => {
      const custom = e as CustomEvent<"PLAIN_ENGLISH" | "PRO_QUANT">;
      if (custom.detail) setVernacularMode(custom.detail);
    };
    window.addEventListener("finance:vernacular-change", handleVernacular);
    return () => window.removeEventListener("finance:vernacular-change", handleVernacular);
  }, []);

  const evalResult: PreFlightEvaluationResult = evaluatePreFlightChecks(evalInputs, vernacularMode);

  const {
    symbol,
    checks,
    passedCount,
    convictionPct,
    isActionableGranted,
    isCleared,
    clearanceStatusLabel,
    activeBlocker,
    whatNeedsToChange,
    evidenceMeta,
    isSimulatedClearance,
  } = evalResult;

  const isPlain = vernacularMode === "PLAIN_ENGLISH";

  return (
    <div
      id="preflight-readiness-bar"
      data-testid="preflight-readiness-bar"
      role="region"
      aria-label="Preflight Execution Readiness"
      className={`bg-[var(--surface-1)] border border-[var(--border-subtle)] rounded-xl p-3 sm:p-4 shadow-xl space-y-3 font-sans text-[var(--text-1)] mt-2.5 transition-all ${className}`}
    >
      {/* ── 1. Header: Title, Actionability Badge, Clearance Status Badge ── */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border-subtle)] pb-2.5">
        <div className="flex items-center gap-2">
          <svg
            className="w-4 h-4 text-cyan-400 shrink-0"
            viewBox="0 0 24 24"
            fill="currentColor"
            aria-hidden="true"
          >
            <path d="M17.8 19.2L16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.3c.4-.2.6-.6.5-1.1z" />
          </svg>
          <span className="text-xs font-mono font-bold tracking-wider text-[var(--text-1)] uppercase">
            {isPlain ? "PREFLIGHT CLEARANCE" : "PREFLIGHT EXECUTION READINESS"}
          </span>
          <span
            data-testid="inline-actionability-badge"
            className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${
              isActionableGranted
                ? "bg-emerald-950/80 text-emerald-300 border-emerald-700/80"
                : "bg-amber-950/80 text-amber-300 border-amber-700/80"
            }`}
          >
            {isActionableGranted ? "ACTIONABLE" : "AWAITING TRIGGER"}
          </span>
        </div>

        {/* Prominent Clearance Status Badge */}
        <div
          data-testid="inline-clearance-badge"
          className={`px-2.5 py-1 rounded text-xs font-mono font-bold border flex items-center gap-1.5 shadow-sm ${
            isCleared
              ? "bg-emerald-950/80 text-emerald-300 border-emerald-500/80"
              : "bg-amber-950/80 text-amber-300 border-amber-600/80"
          }`}
        >
          <span aria-hidden="true">{isCleared ? "🟢" : "⚠️"}</span>
          <span>{clearanceStatusLabel}</span>
          {isSimulatedClearance && (
            <span
              data-testid="inline-simulated-tag"
              className="ml-1 text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-cyan-950/90 text-cyan-300 border border-cyan-700/60 font-semibold"
            >
              SIMULATED
            </span>
          )}
        </div>
      </div>

      {/* ── 2. Five Mandatory Clearance-Check Badges Row ───────────────── */}
      <div
        id="preflight-badges-row"
        data-testid="preflight-badges-row"
        role="list"
        aria-label="Pre-Flight Five Checks"
        className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2"
      >
        {checks.map((chk: PreFlightCheckItem, index: number) => {
          const isCheckPassed = chk.passed;
          return (
            <div
              key={chk.id}
              role="listitem"
              data-testid={`inline-chk-pill-${index + 1}`}
              className={`p-2 rounded-lg border flex flex-col justify-between gap-1 transition-colors ${
                isCheckPassed
                  ? "bg-[var(--status-positive-bg)] border-emerald-900/40"
                  : chk.status === "HAZARD" || chk.status === "FAIL" || chk.status === "CAPITAL RISK"
                  ? "bg-[var(--status-negative-bg)] border-rose-900/50"
                  : "bg-[var(--status-caution-bg)] border-amber-900/40"
              }`}
            >
              <div className="flex items-center justify-between text-[11px] font-mono">
                <span className="font-bold text-[var(--text-2)]">{chk.tag}</span>
                <span
                  data-testid={`inline-chk-status-${index + 1}`}
                  className={`px-1.5 py-0.5 rounded text-[9.5px] font-extrabold border ${chk.badgeClass}`}
                >
                  {chk.status}
                </span>
              </div>
              <div className="text-[10px] text-[var(--text-2)] truncate font-mono" title={chk.summary}>
                {chk.summary}
              </div>
            </div>
          );
        })}
      </div>

      {/* ── 3. Active Blocker & What Needs to Change Callout ─────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-2 text-xs">
        {/* Active Blocker */}
        <div
          data-testid="inline-active-blocker-box"
          className="md:col-span-5 p-2.5 rounded-lg bg-[var(--surface-2)] border border-[var(--border-subtle)] flex flex-col justify-center space-y-1"
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] uppercase font-mono font-bold text-[var(--text-2)]">
              Active Blocker:
            </span>
            {!isCleared && (
              <span
                data-testid="inline-blocker-tag"
                className="text-[9px] font-mono font-bold text-rose-400 bg-rose-950/50 px-1 py-0.5 rounded border border-rose-800/40"
              >
                BLOCKING
              </span>
            )}
          </div>
          <span
            data-testid="inline-active-blocker-text"
            className={`font-mono font-bold text-[11px] leading-snug ${
              isCleared ? "text-emerald-300" : "text-amber-300"
            }`}
          >
            {activeBlocker}
          </span>
        </div>

        {/* What Needs to Change */}
        <div
          data-testid="inline-change-callout-box"
          className="md:col-span-7 p-2.5 rounded-lg bg-[var(--surface-2)] border border-cyan-900/40 flex flex-col justify-center space-y-1 text-[var(--text-2)]"
        >
          <span className="text-[10px] uppercase font-mono font-bold text-cyan-300">
            What Needs to Change:
          </span>
          <p
            data-testid="inline-what-needs-to-change"
            className="text-[11px] font-sans leading-snug text-[var(--text-1)]"
          >
            {whatNeedsToChange}
          </p>
        </div>
      </div>

      {/* ── 4. Meta Footer: Evidence Provenance & Deep Inspection Controls ── */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-[var(--border-subtle)] text-xs">
        <span
          data-testid="inline-evidence-meta"
          className="text-[10.5px] font-mono text-[var(--text-2)]"
        >
          {evidenceMeta}
        </span>

        <div className="flex items-center gap-2">
          {/* Toggle In-line Telemetry Drawer */}
          <button
            type="button"
            id="toggle-preflight-details-btn"
            data-testid="inline-toggle-details-btn"
            aria-expanded={isDrawerExpanded}
            aria-controls="preflight-details-drawer"
            onClick={handleToggleDrawer}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-mono text-xs font-semibold bg-[var(--surface-2)] hover:bg-[var(--surface-2)] text-[var(--text-1)] border border-[var(--border-subtle)] transition-colors cursor-pointer min-h-[44px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"
          >
            <span>{isDrawerExpanded ? "Collapse Telemetry ▲" : "Inspect Telemetry ▾"}</span>
          </button>

          {/* Optional Full Modal Inspection */}
          {onOpenModal && (
            <button
              type="button"
              data-testid="inline-open-modal-btn"
              onClick={onOpenModal}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-mono text-xs font-bold bg-cyan-950/80 hover:bg-cyan-900 text-cyan-300 border border-cyan-700/80 transition-colors cursor-pointer min-h-[44px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"
            >
              <span>✈️ Full Checklist & Plan</span>
            </button>
          )}
        </div>
      </div>

      {/* ── 5. Expandable In-Line Telemetry Drawer ────────────────────────── */}
      {isDrawerExpanded && (
        <div
          id="preflight-details-drawer"
          data-testid="preflight-details-drawer"
          aria-label="Detailed Preflight Telemetry"
          className="mt-2.5 pt-3 border-t border-dashed border-[var(--border-subtle)] grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 text-xs font-sans animate-in fade-in duration-150"
        >
          {checks.map((chk: PreFlightCheckItem) => (
            <div
              key={chk.id}
              className="p-2.5 rounded-lg bg-[var(--surface-2)] border border-[var(--border-subtle)] space-y-1.5"
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-[var(--text-1)] text-xs flex items-center gap-1">
                  <span>{chk.icon}</span>
                  <span>{chk.name}</span>
                </span>
                <span
                  className={`text-[9.5px] font-mono font-bold px-1.5 py-0.5 rounded border ${chk.badgeClass}`}
                >
                  {chk.status}
                </span>
              </div>
              <p className="text-[11px] text-[var(--text-2)] leading-snug font-sans">
                {chk.detail}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
