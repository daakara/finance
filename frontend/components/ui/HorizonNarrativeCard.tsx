"use client";

import React from "react";

export interface NarrativeMetric {
  label: string;
  value: string | number;
  change?: string;
  trend?: 'up' | 'down' | 'neutral';
  status?: 'healthy' | 'warning' | 'critical';
}

export interface NarrativeFinding {
  id: string;
  title: string;
  detail: string;
  severity?: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | string;
  verified?: boolean;
}

export interface NarrativeRecommendation {
  id: string;
  action: string;
  impact: string;
  confidence?: number;
}

export interface HorizonNarrativeCardProps {
  headline: string;
  executiveSummary: string;
  metrics?: NarrativeMetric[];
  findings?: NarrativeFinding[];
  recommendations?: NarrativeRecommendation[];
  auditHash?: string;
  replayDeterministic?: boolean;
  category?: string;
  status?: 'normal' | 'outage' | 'degraded';
  className?: string;
  onActionClick?: (actionId: string) => void;
}

export function HorizonNarrativeCard({
  headline,
  executiveSummary,
  metrics = [],
  findings = [],
  recommendations = [],
  auditHash,
  replayDeterministic = true,
  category = 'EXECUTIVE BRIEFING',
  status = 'normal',
  className = '',
  onActionClick,
}: HorizonNarrativeCardProps) {
  const statusBorder =
    status === 'outage'
      ? 'border-rose-500/40'
      : status === 'degraded'
      ? 'border-amber-500/40'
      : 'border-[#24324A]';

  return (
    <article
      data-testid="horizon-narrative-card"
      className={`rounded-2xl border ${statusBorder} bg-[#121B2A] p-6 shadow-xl shadow-black/30 transition-all duration-200 hover:border-cyan-500/30 ${className}`}
    >
      {/* Top Meta Bar */}
      <header className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-[#24324A]/70">
        <div className="flex items-center gap-2.5">
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase tracking-wider bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
            {category}
          </span>
          {status !== 'normal' && (
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase tracking-wider ${
                status === 'outage'
                  ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                  : 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
              }`}
            >
              {status.toUpperCase()}
            </span>
          )}
        </div>

        {auditHash && (
          <div className="flex items-center gap-1.5 font-mono text-[11px] text-slate-400 bg-[#182336] px-2.5 py-1 rounded-lg border border-[#24324A]">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-slate-500">HASH:</span>
            <span className="text-slate-300">{auditHash.slice(0, 10)}...{auditHash.slice(-6)}</span>
          </div>
        )}
      </header>

      {/* Headline & Executive Summary */}
      <div className="mt-4 space-y-2">
        <h2 className="text-lg font-bold tracking-tight text-white">
          {headline}
        </h2>
        <p className="text-sm text-slate-300 leading-relaxed font-sans">
          {executiveSummary}
        </p>
      </div>

      {/* Metrics Row */}
      {metrics.length > 0 && (
        <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 gap-3">
          {metrics.map((m, idx) => {
            const statusColor =
              m.status === 'critical'
                ? 'text-rose-400'
                : m.status === 'warning'
                ? 'text-amber-400'
                : 'text-emerald-400';
            return (
              <div
                key={idx}
                className="bg-[#182336]/80 border border-[#24324A] rounded-xl p-3 flex flex-col justify-between"
              >
                <span className="text-[11px] font-medium text-slate-400 tracking-wide">
                  {m.label}
                </span>
                <div className="mt-1 flex items-baseline gap-2">
                  <span className={`text-base font-bold font-mono ${statusColor}`}>
                    {m.value}
                  </span>
                  {m.change && (
                    <span className="text-[10px] font-mono text-slate-400">
                      {m.change}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Findings Section */}
      {findings.length > 0 && (
        <div className="mt-5 pt-4 border-t border-[#24324A]/60 space-y-2.5">
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-400">
            Key Findings ({findings.length})
          </h3>
          <div className="space-y-2">
            {findings.map((f) => {
              const sevBadge =
                f.severity === 'CRITICAL'
                  ? 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                  : f.severity === 'HIGH'
                  ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                  : 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30';
              return (
                <div
                  key={f.id}
                  className="flex items-start justify-between gap-3 bg-[#182336]/50 border border-[#24324A]/80 rounded-xl p-3"
                >
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-white">
                        {f.title}
                      </span>
                      {f.verified && (
                        <span className="text-[10px] font-mono text-emerald-400">✓ Verified</span>
                      )}
                    </div>
                    <p className="text-xs text-slate-300">{f.detail}</p>
                  </div>
                  {f.severity && (
                    <span
                      className={`shrink-0 px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${sevBadge}`}
                    >
                      {f.severity}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Recommendations Section */}
      {recommendations.length > 0 && (
        <div className="mt-5 pt-4 border-t border-[#24324A]/60 space-y-2.5">
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-400">
            Actionable Recommendations ({recommendations.length})
          </h3>
          <div className="space-y-2">
            {recommendations.map((r) => (
              <div
                key={r.id}
                className="flex items-center justify-between gap-3 bg-[#182336] border border-[#24324A] rounded-xl p-3"
              >
                <div className="space-y-0.5">
                  <span className="text-xs font-medium text-slate-200">{r.action}</span>
                  <div className="flex items-center gap-2 text-[11px] text-slate-400">
                    <span>Impact: <strong className="text-cyan-400">{r.impact}</strong></span>
                    {r.confidence !== undefined && (
                      <span>Confidence: <strong className="text-emerald-400">{(r.confidence * 100).toFixed(0)}%</strong></span>
                    )}
                  </div>
                </div>
                {onActionClick && (
                  <button
                    onClick={() => onActionClick(r.id)}
                    className="shrink-0 px-3 py-1 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 text-xs font-mono font-semibold rounded-lg border border-cyan-500/30 transition-colors"
                  >
                    Act
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Card Footer */}
      <footer className="mt-5 pt-3 border-t border-[#24324A]/60 flex items-center justify-between text-[11px] font-mono text-slate-400">
        <span className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
          {replayDeterministic ? 'Replay Deterministic (100 runs = 1 hash)' : 'Dynamic Feed'}
        </span>
        <span className="text-slate-400">ARX Horizon Narrative Engine</span>
      </footer>
    </article>
  );
}

export default HorizonNarrativeCard;
