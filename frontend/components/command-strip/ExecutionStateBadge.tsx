"use client";

import React from "react";
import { ExecutionState } from "../../types/workstation";
import { ExperienceMode } from "../../types/insight";

export interface ExecutionStateBadgeProps {
  state: ExecutionState;
  mode?: ExperienceMode;
  className?: string;
}

export default function ExecutionStateBadge({
  state,
  mode = "STANDARD",
  className = "",
}: ExecutionStateBadgeProps) {
  // Config per state with anti-cyan semantic colors & directional icons
  switch (state) {
    case "IN_BUY_ZONE":
      return (
        <div
          data-testid="execution-state-badge"
          data-state="IN_BUY_ZONE"
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold tracking-wide border bg-emerald-500/15 text-emerald-400 border-emerald-500/30 ${className}`}
        >
          {/* Pulsing indicator & target icon */}
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
          </span>
          <svg
            className="w-3.5 h-3.5 shrink-0"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <circle cx="12" cy="12" r="10" />
            <circle cx="12" cy="12" r="6" />
            <circle cx="12" cy="12" r="2" />
          </svg>
          <span>
            {mode === "GUIDED"
              ? "Favorable Entry Zone"
              : mode === "QUANT"
              ? "IN_BUY_ZONE [Stage 2]"
              : "IN_BUY_ZONE"}
          </span>
        </div>
      );

    case "APPROACHING_TARGET":
      return (
        <div
          data-testid="execution-state-badge"
          data-state="APPROACHING_TARGET"
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold tracking-wide border bg-emerald-500/15 text-emerald-300 border-emerald-500/30 ${className}`}
        >
          <svg
            className="w-3.5 h-3.5 shrink-0"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
            <polyline points="17 6 23 6 23 12" />
          </svg>
          <span>
            {mode === "GUIDED"
              ? "Approaching Profit Target"
              : mode === "QUANT"
              ? "APPROACHING_T1 [Active]"
              : "APPROACHING_TARGET"}
          </span>
        </div>
      );

    case "WAITING_PULLBACK":
      return (
        <div
          data-testid="execution-state-badge"
          data-state="WAITING_PULLBACK"
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold tracking-wide border bg-amber-500/15 text-amber-400 border-amber-500/30 ${className}`}
        >
          <svg
            className="w-3.5 h-3.5 shrink-0"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
          <span>
            {mode === "GUIDED"
              ? "Waiting for Pullback"
              : mode === "QUANT"
              ? "WAITING_PULLBACK [Consolidation]"
              : "WAITING_PULLBACK"}
          </span>
        </div>
      );

    case "STOPPED_OUT":
      return (
        <div
          data-testid="execution-state-badge"
          data-state="STOPPED_OUT"
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold tracking-wide border bg-rose-500/15 text-rose-400 border-rose-500/30 ${className}`}
        >
          <svg
            className="w-3.5 h-3.5 shrink-0"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
          <span>
            {mode === "GUIDED"
              ? "Invalidated Setup"
              : mode === "QUANT"
              ? "STOPPED_OUT [Floor Broken]"
              : "STOPPED_OUT"}
          </span>
        </div>
      );

    case "NEUTRAL":
    default:
      return (
        <div
          data-testid="execution-state-badge"
          data-state="NEUTRAL"
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold tracking-wide border bg-slate-800/80 text-slate-300 border-slate-700/60 ${className}`}
        >
          <svg
            className="w-3.5 h-3.5 shrink-0"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="8" y1="12" x2="16" y2="12" />
          </svg>
          <span>
            {mode === "GUIDED"
              ? "Neutral Setup"
              : mode === "QUANT"
              ? "NEUTRAL [Unconfirmed]"
              : "NEUTRAL"}
          </span>
        </div>
      );
  }
}
