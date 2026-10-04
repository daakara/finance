"use client";

import React from "react";
import { OwnershipState } from "../../hooks/usePortfolioContext";

interface RadarPortfolioBadgeProps {
  ownershipState: OwnershipState;
  shares?: number;
  className?: string;
  showWhenNotHeld?: boolean;
}

/**
 * Accessible, institutional portfolio ownership badge for Radar rows and hero cards.
 *
 * Invariants:
 * - INV-RADAR-PORTFOLIO-02: Factual state only; no trade recommendations.
 * - Non-color-only communication: Always renders explicit text label "HELD".
 * - WCAG AA accessible with explicit aria-label and hidden decorative indicator.
 */
export const RadarPortfolioBadge: React.FC<RadarPortfolioBadgeProps> = ({
  ownershipState,
  shares,
  className = "",
  showWhenNotHeld = false,
}) => {
  if (ownershipState === "HELD") {
    const sharesLabel = shares && shares > 0 ? ` (${shares} sh)` : "";
    return (
      <span
        role="status"
        aria-label={`Position status: Held in portfolio${shares && shares > 0 ? `, ${shares} shares` : ""}`}
        className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono font-bold tracking-wider bg-indigo-950/80 text-indigo-300 border border-indigo-700/60 shrink-0 ${className}`}
        title={`Asset currently held in portfolio${shares && shares > 0 ? `: ${shares} shares` : ""}`}
      >
        <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 shrink-0" aria-hidden="true" />
        <span>HELD{sharesLabel}</span>
      </span>
    );
  }

  if (showWhenNotHeld && ownershipState === "NOT_HELD") {
    return (
      <span
        role="status"
        aria-label="Position status: Not currently held in portfolio"
        className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold text-slate-500 border border-slate-800 shrink-0 ${className}`}
      >
        <span>NEW</span>
      </span>
    );
  }

  return null;
};
