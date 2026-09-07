"use client";

import React from "react";
import { LiquidityTier } from "../../types/workstation";
import { ExperienceMode } from "../../types/insight";

export interface LiquidityBadgeProps {
  tier: LiquidityTier;
  amihudScore?: number;
  mode?: ExperienceMode;
  className?: string;
}

export default function LiquidityBadge({
  tier,
  amihudScore,
  mode = "STANDARD",
  className = "",
}: LiquidityBadgeProps) {
  // Map tier to appropriate label and visual treatment
  const config = (() => {
    switch (tier) {
      case "HIGH":
        return {
          label: "ADV: High (<1.0% ADV)",
          guidedLabel: "High Liquidity",
          pillClass: "bg-slate-800/80 text-emerald-400 border-slate-700/80",
          iconColor: "text-emerald-400",
        };
      case "MODERATE":
        return {
          label: "ADV: Moderate",
          guidedLabel: "Moderate Liquidity",
          pillClass: "bg-slate-800/80 text-amber-400 border-slate-700/80",
          iconColor: "text-amber-400",
        };
      case "RISK":
        return {
          label: "ADV: Illiquid / Risk",
          guidedLabel: "Low Liquidity (High Slippage)",
          pillClass: "bg-rose-500/10 text-rose-400 border-rose-500/30",
          iconColor: "text-rose-400",
        };
      case "UNKNOWN":
      default:
        return {
          label: "ADV: Unknown",
          guidedLabel: "Unrated Liquidity",
          pillClass: "bg-slate-800/80 text-slate-400 border-slate-700/60",
          iconColor: "text-slate-400",
        };
    }
  })();

  const textLabel = mode === "GUIDED" ? config.guidedLabel : config.label;

  return (
    <div
      data-testid="liquidity-badge"
      data-tier={tier}
      title={
        amihudScore !== undefined
          ? `Amihud Illiquidity Ratio: ${amihudScore.toFixed(5)}`
          : undefined
      }
      className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-mono border ${config.pillClass} ${className}`}
    >
      {/* Activity / Waves Icon */}
      <svg
        className={`w-3 h-3 shrink-0 ${config.iconColor}`}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <path d="M2 12h2l3-7 4 14 3-7 2 4 2-4h4" />
      </svg>
      <span>{textLabel}</span>

      {/* Quant Mode: Append Amihud Ratio */}
      {mode === "QUANT" && amihudScore !== undefined && (
        <span
          data-testid="amihud-score"
          className="text-[10px] text-slate-400 pl-1 border-l border-slate-700"
        >
          Amihud: {amihudScore.toFixed(4)}
        </span>
      )}
    </div>
  );
}
