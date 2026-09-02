"use client";

import React from "react";
import { FactorAttributionItem } from "../types/insight";

interface WhyInspectModalProps {
  isOpen: boolean;
  onClose: () => void;
  symbol: string;
  setupScore: number;
  items: FactorAttributionItem[];
  catalystToIncreaseScore?: string;
  whatWouldChangeAssessment?: string;
}

export default function WhyInspectModal({
  isOpen,
  onClose,
  symbol,
  setupScore,
  items,
  catalystToIncreaseScore,
  whatWouldChangeAssessment,
}: WhyInspectModalProps) {
  if (!isOpen) return null;

  const changeCondition = whatWouldChangeAssessment || catalystToIncreaseScore;

  return (
    <div className="fixed inset-0 z-[1300] flex items-center justify-center p-3 sm:p-4 bg-black/80 backdrop-blur-sm animate-fade-in font-sans">
      <div className="bg-[#0b101b] border border-[#223147] rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden text-slate-100 max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#1b2537] bg-[#0e1422] shrink-0">
          <div className="flex items-center space-x-2.5">
            <span className="text-xl">🔍</span>
            <div>
              <h2 className="text-sm sm:text-base font-black text-white tracking-tight">
                Traceable Evidence & Score Breakdown
              </h2>
              <p className="text-[11px] text-slate-400 font-mono">
                Empirical factor attribution for <span className="text-cyan-400 font-bold">{symbol}</span>
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
          {/* Confluence Header Box */}
          <div className="flex items-center justify-between bg-[#070b13] p-3.5 rounded-xl border border-[#1b273b]">
            <div>
              <span className="text-slate-300 font-bold block text-xs">Overall Confluence Alignment:</span>
              <span className="text-[10px] text-slate-500 font-mono">Based on 4 independent factor models</span>
            </div>
            <div className="flex items-center gap-1.5 font-mono">
              <span className={`text-xl font-black ${setupScore >= 75 ? "text-emerald-400" : setupScore >= 55 ? "text-amber-400" : "text-rose-400"}`}>
                {setupScore}
              </span>
              <span className="text-[10px] text-slate-500">/ 100</span>
            </div>
          </div>

          {/* Factor Evidence Cards */}
          <div className="space-y-3">
            <span className="text-[10px] uppercase font-mono tracking-wider text-slate-400 font-bold block">
              Multi-Factor Evidence & Provenance
            </span>

            {items.map((item, idx) => {
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

                  {/* Concrete Evidence Proof if available */}
                  {item.evidence && item.evidence.length > 0 && (
                    <div className="bg-[#0c121d] p-2 rounded-lg space-y-1 font-mono text-[10px] border border-[#162030]">
                      {item.evidence.map((ev, evIdx) => (
                        <div key={evIdx} className="flex flex-wrap items-center justify-between text-slate-400 gap-1">
                          <span className="text-slate-300 font-semibold">{ev.metricName}:</span>
                          <span className="text-cyan-300 font-bold">{ev.currentValue} <span className="text-slate-500 font-normal">vs {ev.benchmarkValue}</span></span>
                          {ev.source && (
                            <span className="text-[9px] text-slate-500 w-full truncate">Source: {ev.source}</span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Condition to change factor */}
                  {item.whatWouldChangeAssessment && (
                    <div className="text-[10px] text-slate-400 font-sans italic pt-1 border-t border-[#121a28]">
                      <span className="text-cyan-400 font-semibold not-italic">Condition for change:</span> {item.whatWouldChangeAssessment}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Overall What Would Change ARX's Assessment */}
          {changeCondition && (
            <div className="bg-[#081524] border border-cyan-800/50 p-3.5 rounded-xl space-y-1.5">
              <span className="text-[10px] font-mono uppercase text-cyan-400 font-bold block flex items-center gap-1.5">
                <span>🔄</span> What would change ARX&apos;s assessment?
              </span>
              <p className="text-xs text-slate-200 font-sans leading-relaxed">
                {changeCondition}
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-3.5 bg-[#0a0f18] border-t border-[#1b2537] flex justify-between items-center shrink-0">
          <span className="text-[10px] text-slate-500 font-mono">
            Empirical Confluence Engine
          </span>
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
