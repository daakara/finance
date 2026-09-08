"use client";

import React from "react";
import { SeverityColors, SeverityLevel } from "../../lib/ui/horizonTokens";

export interface HorizonAlertCardProps {
  id: string;
  title: string;
  description: string;
  severity: SeverityLevel;
  source: string;
  timestamp: string;
  slaRemaining?: string;
  onAction?: (id: string) => void;
  actionLabel?: string;
}

export default function HorizonAlertCard({
  id,
  title,
  description,
  severity,
  source,
  timestamp,
  slaRemaining,
  onAction,
  actionLabel = "Investigate",
}: HorizonAlertCardProps) {
  const conf = SeverityColors[severity] || SeverityColors.LOW;

  return (
    <article
      className={`rounded-xl border ${conf.border} ${conf.bg} p-4 transition-all hover:brightness-110 flex flex-col justify-between`}
    >
      <div>
        <div className="flex items-center justify-between gap-2 mb-2">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${conf.dot}`} />
            <span className={`text-[10px] font-mono uppercase font-bold tracking-wider px-2 py-0.5 rounded border ${conf.badge}`}>
              {conf.label}
            </span>
            <span className="text-xs font-mono text-slate-400 font-semibold">{id}</span>
          </div>

          {slaRemaining && (
            <span className="text-[11px] font-mono text-amber-300 font-semibold bg-amber-950/60 px-2 py-0.5 rounded border border-amber-500/30">
              SLA: {slaRemaining}
            </span>
          )}
        </div>

        <h3 className="text-sm font-semibold text-[#F8FAFC] mb-1">
          {title}
        </h3>
        <p className="text-xs text-[#94A3B8] leading-relaxed mb-3">
          {description}
        </p>
      </div>

      <footer className="flex items-center justify-between pt-2 border-t border-slate-700/40 text-[11px] font-mono text-slate-400">
        <span>{source} &bull; {timestamp}</span>
        {onAction && (
          <button
            onClick={() => onAction(id)}
            className="px-2.5 py-1 rounded bg-[#182336] hover:bg-[#20304a] text-cyan-300 border border-cyan-500/30 font-semibold text-xs transition-colors focus-visible:ring-1 focus-visible:ring-cyan-400"
          >
            {actionLabel}
          </button>
        )}
      </footer>
    </article>
  );
}
