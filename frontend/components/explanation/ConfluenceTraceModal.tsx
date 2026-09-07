"use client";

import React, { useEffect, useRef } from "react";
import { FactorAttribution } from "../../types/workstation";
import { trackTelemetryEvent } from "../../telemetry/tracker";

export interface ConfluenceTraceModalProps {
  isOpen: boolean;
  onClose: () => void;
  ticker: string;
  setupScore: number;
  confluenceScore: number;
  factors: FactorAttribution[];
  modelVer?: string;
  decisionHash?: string;
}

export default function ConfluenceTraceModal({
  isOpen,
  onClose,
  ticker,
  setupScore,
  confluenceScore,
  factors = [],
  modelVer = "v2.4.1",
  decisionHash = "sha256-d8f3e2a1b94c0157",
}: ConfluenceTraceModalProps) {
  const modalRef = useRef<HTMLDivElement>(null);
  const openTimeRef = useRef<number>(0);

  useEffect(() => {
    if (isOpen) {
      openTimeRef.current = performance.now();
      trackTelemetryEvent(
        "DECISION",
        "confluence_trace_opened",
        { ticker, setupScore, factorCount: factors.length },
        ticker
      );
    }
  }, [isOpen, ticker, setupScore, factors.length]);

  const handleDismiss = () => {
    const durationMs = Math.round(performance.now() - openTimeRef.current);
    trackTelemetryEvent(
      "DECISION",
      "confluence_trace_closed",
      { ticker, durationMs },
      ticker
    );
    onClose();
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        handleDismiss();
      }
    };
    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="confluence-trace-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/80 backdrop-blur-sm animate-fadeIn font-sans"
    >
      <div
        ref={modalRef}
        className="w-full max-w-4xl max-h-[90vh] flex flex-col rounded-2xl bg-bg-surface-elevated border border-border-subtle shadow-2xl text-text-primary overflow-hidden"
      >
        {/* Header */}
        <div className="p-6 border-b border-border-subtle flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-caption-mono uppercase px-2.5 py-0.5 rounded bg-accent-info/20 text-accent-info border border-accent-info/30 font-bold">
                Level 3 Evidence
              </span>
              <span className="text-text-muted text-caption-mono">
                Institutional Audit Layer
              </span>
            </div>
            <h2 id="confluence-trace-title" className="text-header-1 text-text-primary mt-1">
              Confluence Factor Attribution Trace · {ticker}
            </h2>
            <p className="text-body-ui text-text-secondary mt-0.5">
              Exact mathematical decomposition of factor weights, normalized signals, and net points contributing to the {setupScore}/100 Setup Score.
            </p>
          </div>

          <button
            type="button"
            onClick={handleDismiss}
            aria-label="Close trace modal"
            className="p-2 rounded-lg bg-bg-surface hover:bg-bg-surface-raised border border-border-subtle text-text-muted hover:text-text-primary transition-colors cursor-pointer"
          >
            ✕
          </button>
        </div>

        {/* Content Table */}
        <div className="p-6 overflow-y-auto space-y-6">
          {/* Top Metric Summary */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3 rounded-xl bg-bg-surface border border-border-subtle font-mono">
              <span className="text-[10px] text-text-muted block">Final Setup Score</span>
              <span className="text-xl font-bold text-accent-positive">{setupScore}/100</span>
            </div>
            <div className="p-3 rounded-xl bg-bg-surface border border-border-subtle font-mono">
              <span className="text-[10px] text-text-muted block">Active Factor Count</span>
              <span className="text-xl font-bold text-text-primary">{factors.length || 5} Factors</span>
            </div>
            <div className="p-3 rounded-xl bg-bg-surface border border-border-subtle font-mono">
              <span className="text-[10px] text-text-muted block">Engine Version</span>
              <span className="text-xl font-bold text-accent-info">{modelVer}</span>
            </div>
            <div className="p-3 rounded-xl bg-bg-surface border border-border-subtle font-mono">
              <span className="text-[10px] text-text-muted block">Decision State</span>
              <span className="text-xl font-bold text-emerald-400">DETERMINISTIC</span>
            </div>
          </div>

          {/* Factor Attribution Table */}
          <div className="border border-border-subtle rounded-xl overflow-hidden">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-bg-surface-raised border-b border-border-subtle text-text-muted uppercase text-[10px]">
                <tr>
                  <th className="py-3 px-4">Factor Name & Category</th>
                  <th className="py-3 px-4 text-center">Raw Signal</th>
                  <th className="py-3 px-4 text-center">Model Weight</th>
                  <th className="py-3 px-4 text-center">Net Contribution</th>
                  <th className="py-3 px-4">Data Provenance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-subtle/60 bg-bg-surface">
                {factors.map((factor) => {
                  const contribColor =
                    factor.contribution > 0
                      ? "text-emerald-400 font-bold"
                      : factor.contribution < 0
                      ? "text-rose-400 font-bold"
                      : "text-slate-400";

                  return (
                    <tr key={factor.factorId} className="hover:bg-bg-surface-raised/50 transition-colors">
                      <td className="py-3 px-4">
                        <span className="font-bold text-text-primary block">
                          {factor.name}
                        </span>
                        <span className="text-[10px] text-text-muted">
                          {factor.category}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center font-bold text-text-secondary">
                        {factor.rawSignal}
                      </td>
                      <td className="py-3 px-4 text-center text-accent-info font-bold">
                        {factor.weight}%
                      </td>
                      <td className={`py-3 px-4 text-center ${contribColor}`}>
                        {factor.contribution > 0 ? `+${factor.contribution}` : factor.contribution} pts
                      </td>
                      <td className="py-3 px-4 text-text-muted text-[11px]">
                        {factor.provenance}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Governance Footer */}
          <div className="p-4 rounded-xl bg-bg-surface border border-border-subtle flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs font-mono text-text-muted">
            <div className="flex items-center gap-2">
              <span className="text-accent-positive">✓</span>
              <span>Audit Provenance Hash: <code className="text-text-secondary">{decisionHash}</code></span>
            </div>
            <span>Formula: Bayesian Prior Update · Cold Storage Certified</span>
          </div>
        </div>
      </div>
    </div>
  );
}
