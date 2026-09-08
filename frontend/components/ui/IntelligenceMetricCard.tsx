"use client";

import React from "react";
import Link from "next/link";
import SeverityBadge from "./SeverityBadge";
import { HorizonStatus, normalizeHorizonStatus } from "../../lib/ui/horizonTokens";

export interface IntelligenceMetricCardProps {
  label: string;
  value: string | number;
  delta?: string;
  deltaPositive?: boolean;
  target?: string;
  confidence?: string;
  status?: HorizonStatus | string;
  subtext?: string;
  targetRoute?: string;
  className?: string;
  helpText?: string;
}

export default function IntelligenceMetricCard({
  label,
  value,
  delta,
  deltaPositive,
  target,
  confidence,
  status = "HEALTHY",
  subtext,
  targetRoute,
  className = "",
  helpText,
}: IntelligenceMetricCardProps) {
  const normStatus = normalizeHorizonStatus(status);

  const cardContent = (
    <div
      tabIndex={targetRoute ? 0 : undefined}
      aria-label={`${label}: ${value}, Status: ${status}${delta ? `, Delta: ${delta}` : ''}`}
      className={`p-4 rounded-xl bg-[#121B2A] border border-[#24324A] hover:border-cyan-500/40 transition-all flex flex-col justify-between group focus:outline-none focus:ring-2 focus:ring-cyan-400 ${
        targetRoute ? 'cursor-pointer hover:bg-[#162236]' : ''
      } ${className}`}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <span className="text-xs font-mono uppercase tracking-wider text-[#94A3B8] font-semibold truncate group-hover:text-slate-200">
          {label}
        </span>
        <SeverityBadge status={normStatus} size="sm" />
      </div>

      <div className="my-1.5 flex items-baseline justify-between">
        <div className="text-2xl font-bold font-mono text-[#F8FAFC] tracking-tight">
          {value}
        </div>
        {delta && (
          <span
            className={`text-xs font-mono font-semibold flex items-center gap-0.5 ${
              deltaPositive ? 'text-emerald-400' : 'text-rose-400'
            }`}
          >
            {deltaPositive ? '↑' : '↓'} {delta}
          </span>
        )}
      </div>

      <div className="pt-2 border-t border-[#24324A]/60 flex items-center justify-between text-[11px] font-mono text-[#94A3B8]">
        {target && (
          <span className="truncate">
            Target: <span className="text-slate-300 font-semibold">{target}</span>
          </span>
        )}
        {confidence && (
          <span className="text-cyan-400/80 font-medium truncate">
            {confidence}
          </span>
        )}
        {!target && !confidence && subtext && (
          <span className="truncate text-slate-400">{subtext}</span>
        )}
      </div>

      {helpText && (
        <div className="mt-1 text-[10px] text-slate-400 font-mono italic truncate">
          {helpText}
        </div>
      )}
    </div>
  );

  if (targetRoute) {
    return (
      <Link href={targetRoute} className="block focus:outline-none">
        {cardContent}
      </Link>
    );
  }

  return cardContent;
}
