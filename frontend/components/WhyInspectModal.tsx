"use client";

import React from "react";
import { TerminalViewState, FactorAttributionItem } from "../types/insight";

interface WhyInspectModalProps {
  isOpen: boolean;
  onClose: () => void;
  symbol: string;
  setupScore: number;
  terminalState?: TerminalViewState;
  items?: FactorAttributionItem[];
  catalystToIncreaseScore?: string;
  whatWouldChangeAssessment?: string;
}

export default function WhyInspectModal({
  isOpen,
  onClose,
  symbol,
  setupScore,
  terminalState,
  items = [],
  catalystToIncreaseScore,
  whatWouldChangeAssessment,
}: WhyInspectModalProps) {
  React.useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const changeCondition =
    terminalState?.whatWouldChangeAssessment ||
    whatWouldChangeAssessment ||
    catalystToIncreaseScore;

  const agreementLabel =
    terminalState?.factorAgreement?.displayLabel ||
    `${items.filter((i) => i.impact > 0).length} of ${items.length} evaluated factors are favorable`;

  const domains = terminalState?.domains;
  const modelProv = terminalState?.modelProvenance;

  return (
    <div
      className="fixed inset-0 z-[1300] flex items-center justify-center p-3 sm:p-4 bg-black/80 backdrop-blur-sm animate-fade-in font-sans"
      role="dialog"
      aria-modal="true"
      aria-labelledby="why-modal-title"
    >
      <div className="bg-[#0b101b] border border-[#223147] rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden text-slate-100 max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#1b2537] bg-[#0e1422] shrink-0">
          <div className="flex items-center space-x-2.5">
            <span className="text-xl">🔍</span>
            <div>
              <h2 id="why-modal-title" className="text-sm sm:text-base font-black text-white tracking-tight">
                Traceable Evidence & Score Breakdown
              </h2>
              <p className="text-[11px] text-slate-400 font-mono">
                Multi-factor evaluation provenance for <span className="text-cyan-400 font-bold">{symbol}</span>
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close Breakdown"
            className="text-slate-400 hover:text-white p-1.5 rounded-lg hover:bg-slate-800 transition-all text-sm cursor-pointer"
          >
            ✕
          </button>
        </div>

        {/* Scrollable Body */}
        <div className="p-4 sm:p-5 space-y-4 overflow-y-auto font-sans text-xs">
          {/* Factor Agreement & Coverage Header Box */}
          <div className="bg-[#070b13] p-3.5 rounded-xl border border-[#1b273b] space-y-2">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-slate-300 font-bold block text-xs">Factor Agreement State:</span>
                <span className="text-[11px] text-cyan-300 font-mono font-bold">{agreementLabel}</span>
              </div>
              <div className="flex items-center gap-1.5 font-mono">
                <span className={`text-xl font-black ${setupScore >= 75 ? "text-emerald-400" : setupScore >= 55 ? "text-amber-400" : "text-rose-400"}`}>
                  {setupScore}
                </span>
                <span className="text-[10px] text-slate-500">/ 100</span>
              </div>
            </div>

            {terminalState?.overallEligibility && (
              <div className="flex items-center justify-between pt-2 border-t border-[#141e2e] text-[10px] font-mono text-slate-400">
                <span>Data Eligibility:</span>
                <span className={`px-2 py-0.5 rounded font-bold ${
                  terminalState.overallEligibility === "ELIGIBLE"
                    ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                    : terminalState.overallEligibility === "LIMITED"
                    ? "bg-amber-950 text-amber-400 border border-amber-800"
                    : "bg-rose-950 text-rose-400 border border-rose-800"
                }`}>
                  {terminalState.overallEligibility}
                </span>
              </div>
            )}
          </div>

          {/* Domain Breakdown Cards */}
          <div className="space-y-3">
            <span className="text-[10px] uppercase font-mono tracking-wider text-slate-400 font-bold block">
              Evaluated Domains (Fact vs. Model Rule)
            </span>

            {domains && domains.length > 0 ? (
              domains.map((dom, idx) => {
                const isPositive = dom.pointImpact > 0;
                return (
                  <div
                    key={idx}
                    className="bg-[#070b13] border border-[#182334] rounded-xl p-3 space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className={`font-mono font-bold text-xs ${isPositive ? "text-emerald-400" : dom.pointImpact < 0 ? "text-rose-400" : "text-slate-400"}`}>
                          {isPositive ? `+${dom.pointImpact}` : dom.pointImpact}
                        </span>
                        <span className="font-bold text-white text-xs">{dom.domainName}</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-[#131d2c] text-slate-400 border border-[#1f2e44]">
                          {dom.availability}
                        </span>
                        <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded font-bold ${
                          dom.status === "FAVORABLE"
                            ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                            : dom.status === "UNFAVORABLE"
                            ? "bg-rose-950 text-rose-400 border border-rose-800"
                            : "bg-amber-950 text-amber-400 border border-amber-800"
                        }`}>
                          {dom.status}
                        </span>
                      </div>
                    </div>

                    {/* Factual Observation */}
                    <div className="space-y-0.5">
                      <span className="text-[10px] text-slate-400 font-mono font-bold block">📊 Observation (Fact):</span>
                      <p className="text-xs text-slate-200 leading-relaxed font-sans pl-1">
                        {dom.observation}
                      </p>
                    </div>

                    {/* Model Rule */}
                    <div className="space-y-0.5 pt-1 border-t border-[#141e2e]">
                      <span className="text-[10px] text-cyan-400 font-mono font-bold block">⚙️ Model Weighting Rule:</span>
                      <p className="text-[11px] text-slate-300 leading-relaxed font-sans pl-1">
                        {dom.modelRule}
                      </p>
                    </div>

                    {/* Evidence & Provenance Proof */}
                    {dom.evidence && dom.evidence.length > 0 && (
                      <div className="bg-[#0c121d] p-2 rounded-lg space-y-1 font-mono text-[10px] border border-[#162030]">
                        {dom.evidence.map((ev, evIdx) => (
                          <div key={evIdx} className="flex flex-wrap items-center justify-between text-slate-400 gap-1">
                            <span className="text-slate-300 font-semibold">{ev.metricName}:</span>
                            <span className="text-cyan-300 font-bold">
                              {ev.currentValue} <span className="text-slate-500 font-normal">vs {ev.benchmarkValue}</span>
                            </span>
                            {ev.source && (
                              <span className="text-[9px] text-slate-500 w-full truncate">
                                Source: {ev.source} {ev.freshness ? `(${ev.freshness})` : ""}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })
            ) : (
              items.map((item, idx) => {
                const name = item.factorName || item.category || "Factor";
                const isPositive = item.impact > 0;
                return (
                  <div
                    key={idx}
                    className="bg-[#070b13] border border-[#182334] rounded-xl p-3 space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className={`font-mono font-bold text-xs ${isPositive ? "text-emerald-400" : "text-rose-400"}`}>
                          {isPositive ? `+${item.impact}` : item.impact}
                        </span>
                        <span className="font-bold text-white text-xs">{name}</span>
                      </div>
                      {item.importanceLevel && (
                        <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-[#131d2c] text-slate-400 border border-[#1f2e44]">
                          {item.importanceLevel} IMPACT
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      {item.plainEnglishReason || item.reason}
                    </p>
                  </div>
                );
              })
            )}
          </div>

          {/* Overall What Would Change ARX's Assessment */}
          {changeCondition && (
            <div className="bg-[#081524] border border-cyan-800/50 p-3.5 rounded-xl space-y-1.5">
              <span className="text-[10px] font-mono uppercase text-cyan-400 font-bold block flex items-center gap-1.5">
                <span>🔄</span> What would change this assessment?
              </span>
              <p className="text-xs text-slate-200 font-sans leading-relaxed">
                {changeCondition}
              </p>
            </div>
          )}
        </div>

        {/* Footer with Model Provenance */}
        <div className="p-3.5 bg-[#0a0f18] border-t border-[#1b2537] flex flex-wrap justify-between items-center gap-2 shrink-0">
          <div className="text-[10px] text-slate-500 font-mono">
            {modelProv ? (
              <span>Engine: <span className="text-slate-400">{modelProv.modelId}</span> (v{modelProv.modelVersion} · ruleset {modelProv.rulesetVersion})</span>
            ) : (
              <span>ARX Confluence Engine</span>
            )}
          </div>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-black rounded-lg text-xs font-mono transition-all cursor-pointer"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
