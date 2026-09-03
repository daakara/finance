"use client";

import React, { useState } from "react";
import { QuantitativeInsight } from "../../types/insight";
import FinancialDisclaimer from "../FinancialDisclaimer";

interface GuidedTerminalViewProps {
  insight: QuantitativeInsight;
  onOpenSizer: () => void;
  onOpenWhy: () => void;
}

export default function GuidedTerminalView({
  insight,
  onOpenSizer,
  onOpenWhy,
}: GuidedTerminalViewProps) {
  const [activeStep, setActiveStep] = useState<number | null>(null);

  const steps = [
    { title: "1. What's Happening?", text: `${insight.symbol} is trading at $${insight.price.toFixed(2)}, ${insight.changePct >= 0 ? "+" : ""}${insight.changePct.toFixed(2)}% today. ${insight.human.assessmentDescription}` },
    { title: "2. What's the Setup?", text: `ARX identifies the current structure as ${insight.standard.setupSummary}. ${insight.advanced.relativeStrengthScore !== undefined ? `Relative strength score is ${insight.advanced.relativeStrengthScore}/100.` : "Relative strength score is unverified for this security."}` },
    { title: "3. Why does ARX like/caution it?", text: insight.human.reclaimMilestone },
    { title: "4. What could go wrong?", text: `Every thesis has downside risk. If price breaks below $${insight.standard.keyLevels.stopLoss.toFixed(2)}, the setup is invalidated.` },
    { title: "5. How could I trade it?", text: `Plan: Watch for reclaim of ${insight.standard.keyLevels.sma50 !== undefined ? `$${insight.standard.keyLevels.sma50.toFixed(2)}` : "key technical levels"}. Target 1 is ${insight.standard.keyLevels.target1 !== undefined ? `$${insight.standard.keyLevels.target1.toFixed(2)} (+${insight.standard.keyLevels.target1Pct}%)` : "N/A (< 50 sessions)"}.` },
    { title: "6. What should I monitor?", text: `Volume surges, 50-day moving average crossovers, and broader market regime stability.` },
  ];

  return (
    <div className="space-y-4 font-sans text-slate-100 animate-fade-in">
      {/* Guided Badge Banner */}
      <div className="flex items-center justify-between bg-emerald-950/40 border border-emerald-800/50 p-2.5 rounded-xl text-xs font-mono">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-emerald-300 font-bold">🟢 GUIDED EXPERIENCE</span>
          <span className="text-slate-400 hidden sm:inline">• Plain English, step-by-step contextual intelligence</span>
        </div>
        <button
          onClick={onOpenWhy}
          className="text-[11px] text-emerald-400 hover:text-emerald-300 underline font-bold cursor-pointer"
        >
          Explain Score →
        </button>
      </div>

      {/* ARX Assessment Card */}
      <div className="bg-[#0b101b] border border-[#1d293d] rounded-2xl p-4 sm:p-5 shadow-xl space-y-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <span className="text-[11px] text-slate-400 font-mono font-bold uppercase tracking-wider block mb-1">
              ARX Assessment
            </span>
            <h2 className="text-lg sm:text-2xl font-black text-white tracking-tight">
              {insight.human.assessmentHeadline}
            </h2>
            <p className="text-xs sm:text-sm text-slate-300 mt-1.5 leading-relaxed max-w-2xl font-sans">
              {insight.human.assessmentDescription}
            </p>
          </div>

          <div
            onClick={onOpenWhy}
            className="flex flex-col items-center justify-center p-3 bg-[#06090f] border border-[#24334b] hover:border-cyan-500/60 rounded-xl cursor-pointer transition-all shadow-md group shrink-0"
            title="Click to inspect exact score breakdown"
          >
            <span className="text-[10px] text-slate-400 font-mono">Setup Score</span>
            <span className={`text-2xl font-black font-mono ${insight.setupScore >= 75 ? "text-emerald-400" : insight.setupScore >= 55 ? "text-amber-400" : "text-rose-400"}`}>
              {insight.setupScore}
            </span>
            <span className="text-[9px] text-cyan-400 group-hover:underline font-mono mt-0.5">Why? 🔍</span>
          </div>
        </div>

        {/* Why ARX Thinks This: 4-Pill Grid */}
        <div className="pt-2 border-t border-[#172235]">
          <span className="text-[11px] text-slate-400 font-mono font-bold block mb-2">
            Why ARX thinks this
          </span>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 text-xs">
            {insight.human.whyPills.map((pill, idx) => (
              <div
                key={idx}
                className="bg-[#070b13] p-2.5 rounded-xl border border-[#1b2639] space-y-1"
              >
                <div className="flex items-center justify-between">
                  <span className="text-slate-400 text-[11px] font-bold">{pill.category}</span>
                  <span
                    className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold ${
                      pill.sentiment === "positive"
                        ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                        : pill.sentiment === "negative"
                        ? "bg-rose-950 text-rose-400 border border-rose-800"
                        : "bg-amber-950 text-amber-400 border border-amber-800"
                    }`}
                  >
                    {pill.status}
                  </span>
                </div>
                <p className="text-[11px] text-slate-300 leading-snug font-sans">
                  {pill.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* What would improve the setup? */}
        <div className="bg-[#09111c] border border-cyan-900/40 p-3 rounded-xl flex items-start gap-2.5">
          <span className="text-lg">🎯</span>
          <div>
            <strong className="text-xs text-cyan-300 font-bold block">What would improve the setup?</strong>
            <p className="text-xs text-slate-300 mt-0.5 font-sans leading-relaxed">
              {insight.human.reclaimMilestone}
            </p>
          </div>
        </div>

        {/* Watch These Levels */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 font-mono text-xs">
          <div className="bg-[#080d16] p-3 rounded-xl border border-[#182335]">
            <span className="text-[10px] text-slate-500 uppercase block font-semibold">Watch Level</span>
            <span className="text-sm font-black text-amber-300 mt-0.5 block">{insight.human.watchLevels.watchZone}</span>
            <span className="text-[10px] text-slate-400 mt-0.5 block">Needs base rebound</span>
          </div>

          <div className="bg-[#080d16] p-3 rounded-xl border border-[#182335]">
            <span className="text-[10px] text-slate-500 uppercase block font-semibold">Key Level (50D SMA)</span>
            <span className="text-sm font-black text-cyan-300 mt-0.5 block">{insight.human.watchLevels.keyLevel}</span>
            <span className="text-[10px] text-slate-400 mt-0.5 block">Must reclaim</span>
          </div>

          <div className="bg-[#080d16] p-3 rounded-xl border border-rose-950/60">
            <span className="text-[10px] text-rose-400 uppercase block font-semibold">Risk Level (Stop)</span>
            <span className="text-sm font-black text-rose-300 mt-0.5 block">{insight.human.watchLevels.riskStop}</span>
            <span className="text-[10px] text-rose-500/80 mt-0.5 block">Protect if broken</span>
          </div>
        </div>

        {/* Action Guidance & Position Sizer trigger */}
        <div className="flex flex-wrap items-center justify-between gap-3 bg-[#0c121e] p-3.5 rounded-xl border border-[#1f2c42]">
          <div className="flex items-center gap-2">
            <span className="text-lg">👀</span>
            <div>
              <strong className="text-xs text-white font-bold block">
                ARX View: {insight.terminalState.uiStateLabel}
              </strong>
              <p className="text-xs text-slate-400 font-sans mt-0.5">
                {insight.terminalState.headlineExplanation}
              </p>
            </div>
          </div>

          {insight.terminalState.posture === "ACQUIRE" ? (
            <button
              onClick={onOpenSizer}
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-slate-950 rounded-xl text-xs font-mono font-black shadow-md transition-all active:scale-95 cursor-pointer shrink-0"
            >
              ⚖️ Size Position (Buy Zone)
            </button>
          ) : insight.terminalState.posture === "EXIT_REVIEW" ? (
            <button
              onClick={onOpenWhy}
              className="px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white rounded-xl text-xs font-mono font-black shadow-md transition-all active:scale-95 cursor-pointer shrink-0"
            >
              🚨 Review Invalidation
            </button>
          ) : insight.terminalState.posture === "RESEARCH" ? (
            <button
              onClick={onOpenWhy}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-xl text-xs font-mono font-black shadow-md transition-all active:scale-95 cursor-pointer shrink-0"
            >
              📋 Open Research
            </button>
          ) : insight.terminalState.posture === "AVOID" ? (
            <a
              href="/screener"
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-mono font-black shadow-md transition-all active:scale-95 cursor-pointer shrink-0"
            >
              🔎 Find Setups
            </a>
          ) : (
            <button
              onClick={onOpenWhy}
              className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-slate-950 rounded-xl text-xs font-mono font-black shadow-md transition-all active:scale-95 cursor-pointer shrink-0"
            >
              ⏳ Wait for Trigger
            </button>
          )}
        </div>
      </div>

      {/* 6-Step Novice Guided Breadcrumb Walkthrough */}
      <div className="bg-[#070b13] border border-[#1a2538] rounded-xl p-3.5 space-y-2.5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono font-bold text-slate-300 flex items-center gap-1.5">
            <span>🧭</span> Step-by-Step Stock Analysis Walkthrough
          </span>
          <span className="text-[10px] text-slate-500 font-mono">Click any step to inspect</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-1.5">
          {steps.map((step, idx) => (
            <button
              key={idx}
              id={`walkthrough-step-btn-${idx}`}
              onClick={() => setActiveStep(activeStep === idx ? null : idx)}
              aria-expanded={activeStep === idx}
              aria-controls={`walkthrough-step-panel-${idx}`}
              className={`p-2 rounded-lg text-left text-xs font-mono border transition-all cursor-pointer ${
                activeStep === idx
                  ? "bg-cyan-950/80 border-cyan-500 text-cyan-200"
                  : "bg-[#0b101b] border-[#182335] text-slate-400 hover:text-slate-200"
              }`}
            >
              <span className="font-bold block truncate">{step.title}</span>
            </button>
          ))}
        </div>

        {activeStep !== null && (
          <div
            id={`walkthrough-step-panel-${activeStep}`}
            role="region"
            aria-labelledby={`walkthrough-step-btn-${activeStep}`}
            className="p-3 bg-[#0d1422] border border-cyan-800/40 rounded-lg text-xs font-sans text-slate-200 animate-fade-in leading-relaxed"
          >
            <strong className="text-cyan-400 font-mono block mb-1">{steps[activeStep].title}</strong>
            <span>{steps[activeStep].text}</span>
          </div>
        )}
      </div>

      <FinancialDisclaimer variant="compact" />
    </div>
  );
}
