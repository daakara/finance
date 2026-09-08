"use client";

import React from "react";

export interface HorizonMetricCardProps {
  label: string;
  value: string | number;
  delta?: string;
  deltaPositive?: boolean;
  target?: string;
  confidence?: string;
  sparkline?: number[];
  severity?: "PASS" | "WARN" | "CRITICAL" | "INFO";
  subtext?: string;
  onClick?: () => void;
}

export default function HorizonMetricCard({
  label,
  value,
  delta,
  deltaPositive = true,
  target,
  confidence,
  severity = "PASS",
  subtext,
  onClick,
}: HorizonMetricCardProps) {
  const borderColor =
    severity === "CRITICAL"
      ? "border-red-500/40"
      : severity === "WARN"
      ? "border-amber-500/40"
      : severity === "INFO"
      ? "border-cyan-500/40"
      : "border-[#24324A]";

  return (
    <div
      onClick={onClick}
      className={`rounded-xl border ${borderColor} bg-[#121B2A] p-4 flex flex-col justify-between transition-all duration-150 hover:border-[#3b4f73] hover:bg-[#152136] ${
        onClick ? "cursor-pointer" : ""
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-mono font-medium text-[#94A3B8] uppercase tracking-wider">
          {label}
        </span>
        {confidence && (
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#182336] text-slate-300 border border-[#24324A]">
            {confidence}
          </span>
        )}
      </div>

      <div className="flex items-baseline justify-between mt-1">
        <span className="text-2xl font-mono font-bold text-[#F8FAFC] tracking-tight">
          {value}
        </span>
        {delta && (
          <span
            className={`text-xs font-mono font-semibold ${
              deltaPositive ? "text-emerald-400" : "text-rose-400"
            }`}
          >
            {delta}
          </span>
        )}
      </div>

      {(target || subtext) && (
        <div className="mt-3 pt-2 border-t border-[#24324A]/60 flex items-center justify-between text-[11px] font-mono text-[#94A3B8]">
          {target && <span>Floor: {target}</span>}
          {subtext && <span className="truncate">{subtext}</span>}
        </div>
      )}
    </div>
  );
}
