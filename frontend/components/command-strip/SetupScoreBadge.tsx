"use client";

import React from "react";
import { DomainConfidence } from "../../types/workstation";
import { ExperienceMode } from "../../types/insight";

export interface SetupScoreBadgeProps {
  score: number; // 0 - 100
  domainConfidence?: DomainConfidence;
  mode?: ExperienceMode;
  size?: "sm" | "md" | "lg";
  className?: string;
}

export default function SetupScoreBadge({
  score,
  domainConfidence = "HIGH",
  mode = "STANDARD",
  size = "md",
  className = "",
}: SetupScoreBadgeProps) {
  // Clamp score between 0 and 100
  const normalizedScore = Math.max(0, Math.min(100, Math.round(score)));

  // Strict institutional color thresholding:
  // >= 70: Emerald (Favorable)
  // 50 - 69: Amber (Warning / Pullback / Consolidation)
  // < 50: Rose (High Risk / Invalidation)
  const isHigh = normalizedScore >= 70;
  const isMid = normalizedScore >= 50 && normalizedScore < 70;

  const colorConfig = isHigh
    ? {
        stroke: "#10b981", // Emerald 500
        text: "text-emerald-400",
        bg: "bg-emerald-500/10",
        border: "border-emerald-500/30",
        label: "Favorable",
        guidedLabel: "Strong Setup",
      }
    : isMid
    ? {
        stroke: "#f59e0b", // Amber 500
        text: "text-amber-400",
        bg: "bg-amber-500/10",
        border: "border-amber-500/30",
        label: "Conditional",
        guidedLabel: "Moderate Setup",
      }
    : {
        stroke: "#f43f5e", // Rose 500
        text: "text-rose-400",
        bg: "bg-rose-500/10",
        border: "border-rose-500/30",
        label: "High Risk",
        guidedLabel: "Sub-optimal",
      };

  // Circular gauge geometry
  const radius = 20;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (normalizedScore / 100) * circumference;

  return (
    <div
      role="meter"
      aria-valuenow={normalizedScore}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={`Setup Score: ${normalizedScore} out of 100 (${colorConfig.label})`}
      data-testid="setup-score-badge"
      className={`inline-flex items-center gap-3 px-3 py-1.5 rounded-lg border bg-[#0e1422] ${colorConfig.border} ${className}`}
    >
      {/* Circular Progress Gauge */}
      <div className="relative flex items-center justify-center w-12 h-12 shrink-0">
        <svg
          className="w-12 h-12 transform -rotate-90"
          viewBox="0 0 48 48"
          aria-hidden="true"
        >
          {/* Track */}
          <circle
            cx="24"
            cy="24"
            r={radius}
            className="stroke-slate-800"
            strokeWidth="4"
            fill="transparent"
          />
          {/* Value Progress */}
          <circle
            cx="24"
            cy="24"
            r={radius}
            stroke={colorConfig.stroke}
            strokeWidth="4"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            fill="transparent"
            className="transition-all duration-500 ease-out"
          />
        </svg>
        <span
          data-testid="setup-score-value"
          className={`absolute font-mono font-bold text-sm sm:text-base ${colorConfig.text}`}
        >
          {normalizedScore}
        </span>
      </div>

      {/* Label and Details Cluster */}
      <div className="flex flex-col justify-center min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 font-semibold">
            {mode === "GUIDED" ? "Setup Quality" : "Setup Score"}
          </span>
          <span className="text-[10px] font-mono text-slate-500">/ 100</span>
        </div>

        <div className="flex items-center gap-1.5 mt-0.5">
          <span
            data-testid="setup-score-status"
            className={`text-xs font-semibold ${colorConfig.text}`}
          >
            {mode === "GUIDED" ? colorConfig.guidedLabel : colorConfig.label}
          </span>

          {/* Domain Confidence Tag */}
          {mode === "QUANT" && (
            <span
              data-testid="domain-confidence-tag"
              className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700"
            >
              CONF: {domainConfidence}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
