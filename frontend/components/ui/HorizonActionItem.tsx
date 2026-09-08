"use client";

import React from "react";
import { SeverityColors, SeverityLevel } from "../../lib/ui/horizonTokens";

export interface HorizonActionItemProps {
  id: string;
  title: string;
  category: "ALERT" | "RECOMMENDATION" | "APPROVAL" | "ESCALATION" | "RUNBOOK";
  severity: SeverityLevel;
  owner: string;
  slaTarget: string;
  description: string;
  onExecute: (id: string) => void;
  onReview?: (id: string) => void;
  isExecuting?: boolean;
}

export default function HorizonActionItem({
  id,
  title,
  category,
  severity,
  owner,
  slaTarget,
  description,
  onExecute,
  onReview,
  isExecuting = false,
}: HorizonActionItemProps) {
  const sevConf = SeverityColors[severity] || SeverityColors.LOW;

  return (
    <div className="rounded-xl border border-[#24324A] bg-[#121B2A] p-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 hover:border-[#384c70] transition-colors">
      <div className="space-y-1.5 flex-1 min-w-0">
        <div className="flex items-center flex-wrap gap-2">
          <span className="text-xs font-mono font-bold text-cyan-400">{id}</span>
          <span className={`text-[10px] font-mono px-2 py-0.5 rounded border uppercase font-semibold ${sevConf.badge}`}>
            {severity}
          </span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#182336] text-slate-300 border border-[#24324A]">
            {category}
          </span>
          <span className="text-[11px] font-mono text-slate-400">
            Owner: <strong className="text-slate-200">{owner}</strong>
          </span>
        </div>

        <h4 className="text-sm font-semibold text-[#F8FAFC] truncate">
          {title}
        </h4>
        <p className="text-xs text-[#94A3B8] line-clamp-2">
          {description}
        </p>
      </div>

      <div className="flex items-center gap-3 shrink-0 w-full md:w-auto justify-end border-t md:border-t-0 pt-2 md:pt-0 border-[#24324A]">
        <div className="text-right hidden sm:block">
          <div className="text-[10px] font-mono text-slate-400 uppercase">Target SLA</div>
          <div className="text-xs font-mono font-semibold text-amber-300">{slaTarget}</div>
        </div>

        {onReview && (
          <button
            onClick={() => onReview(id)}
            className="px-3 py-1.5 rounded bg-[#182336] hover:bg-[#20304a] text-slate-200 border border-[#24324A] text-xs font-mono transition-colors"
          >
            Review
          </button>
        )}

        <button
          onClick={() => onExecute(id)}
          disabled={isExecuting}
          className={`px-3.5 py-1.5 rounded text-xs font-mono font-semibold transition-colors flex items-center gap-1.5 ${
            severity === "CRITICAL"
              ? "bg-red-600 hover:bg-red-500 text-white"
              : "bg-cyan-600 hover:bg-cyan-500 text-white"
          } disabled:opacity-50`}
        >
          {isExecuting ? "Executing..." : "Execute"}
        </button>
      </div>
    </div>
  );
}
