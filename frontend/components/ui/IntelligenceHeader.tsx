"use client";

import React from "react";
import SeverityBadge from "./SeverityBadge";
import { HorizonStatus } from "../../lib/ui/horizonTokens";

export interface IntelligenceHeaderProps {
  title: string;
  subtitle?: string;
  status?: HorizonStatus | string;
  certification?: string;
  replayHash?: string;
  actions?: React.ReactNode;
  breadcrumbs?: Array<{ label: string; href?: string }>;
}

export default function IntelligenceHeader({
  title,
  subtitle,
  status = "CERTIFIED",
  certification,
  replayHash,
  actions,
  breadcrumbs,
}: IntelligenceHeaderProps) {
  return (
    <header className="mb-6 p-4 rounded-xl bg-[#121B2A] border border-[#24324A] shadow-sm">
      {breadcrumbs && breadcrumbs.length > 0 && (
        <nav aria-label="Breadcrumb" className="mb-2.5 flex items-center gap-1.5 text-xs font-mono text-slate-400">
          {breadcrumbs.map((b, idx) => (
            <React.Fragment key={idx}>
              {idx > 0 && <span className="text-slate-600">/</span>}
              {b.href ? (
                <a href={b.href} className="hover:text-cyan-300 transition-colors">
                  {b.label}
                </a>
              ) : (
                <span className="text-slate-200 font-medium">{b.label}</span>
              )}
            </React.Fragment>
          ))}
        </nav>
      )}

      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2.5">
            <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">
              {title}
            </h1>
            <SeverityBadge status={status} size="md" />
            {certification && (
              <span className="px-2.5 py-0.5 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-300 text-[11px] font-mono font-semibold">
                {certification}
              </span>
            )}
          </div>
          {subtitle && (
            <p className="text-xs md:text-sm text-[#94A3B8] font-mono leading-relaxed">
              {subtitle}
            </p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2.5 self-start md:self-center">
          {replayHash && (
            <div className="px-2 py-1 rounded bg-[#0B1220] border border-[#24324A] text-[10px] font-mono text-slate-400 truncate max-w-[180px]" title={`Replay Audit: ${replayHash}`}>
              SHA: <span className="text-cyan-400">{replayHash.substring(0, 10)}...</span>
            </div>
          )}
          {actions}
        </div>
      </div>
    </header>
  );
}
