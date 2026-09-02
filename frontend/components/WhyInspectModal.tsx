"use client";

import React from "react";
import { ScoreAttributionItem } from "../types/insight";

interface WhyInspectModalProps {
  isOpen: boolean;
  onClose: () => void;
  symbol: string;
  setupScore: number;
  items: ScoreAttributionItem[];
  catalystToIncreaseScore: string;
}

export default function WhyInspectModal({
  isOpen,
  onClose,
  symbol,
  setupScore,
  items,
  catalystToIncreaseScore,
}: WhyInspectModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[1300] flex items-center justify-center p-3 sm:p-4 bg-black/80 backdrop-blur-sm animate-fade-in font-mono">
      <div className="bg-[#0b101b] border border-[#223147] rounded-2xl w-full max-w-md shadow-2xl overflow-hidden text-slate-100 font-sans">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#1b2537] bg-[#0e1422]">
          <div className="flex items-center space-x-2">
            <span className="text-xl">🔍</span>
            <div>
              <h2 className="text-sm sm:text-base font-black text-white tracking-tight">
                Score Attribution & Breakdown
              </h2>
              <p className="text-[11px] text-slate-400">
                Transparent quantitative attribution for <span className="text-cyan-400 font-bold font-mono">{symbol}</span>
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

        {/* Body */}
        <div className="p-4 space-y-4 font-mono text-xs">
          <div className="flex items-center justify-between bg-[#070b13] p-3 rounded-xl border border-[#1b273b]">
            <span className="text-slate-400 font-bold">Total Confluence Score:</span>
            <div className="flex items-center gap-1.5">
              <span className={`text-lg font-black ${setupScore >= 75 ? "text-emerald-400" : setupScore >= 55 ? "text-amber-400" : "text-rose-400"}`}>
                {setupScore}
              </span>
              <span className="text-[10px] text-slate-500">/ 100</span>
            </div>
          </div>

          <div className="space-y-2">
            <span className="text-[10px] uppercase tracking-wider text-slate-400 font-bold block">
              Additive Factor Breakdown
            </span>
            <div className="space-y-1.5 divide-y divide-[#151f2e]">
              {items.map((item, idx) => (
                <div key={idx} className="flex items-center justify-between pt-1.5 text-xs">
                  <div className="flex items-center gap-2">
                    <span className={item.impact > 0 ? "text-emerald-400 font-bold" : "text-rose-400 font-bold"}>
                      {item.impact > 0 ? `+${item.impact}` : item.impact}
                    </span>
                    <span className="text-slate-200">{item.category}</span>
                  </div>
                  <span className="text-[10px] text-slate-400 font-sans">{item.reason}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Catalyst to increase score */}
          <div className="bg-[#091522] border border-cyan-800/40 p-3 rounded-xl space-y-1">
            <span className="text-[10px] uppercase text-cyan-400 font-bold block flex items-center gap-1">
              <span>🚀</span> What would increase the score?
            </span>
            <p className="text-[11px] text-slate-300 font-sans leading-relaxed">
              {catalystToIncreaseScore}
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="p-3 bg-[#0a0f18] border-t border-[#1b2537] flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-bold font-mono transition-all cursor-pointer"
          >
            Got It
          </button>
        </div>
      </div>
    </div>
  );
}
