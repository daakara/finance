"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useExperienceMode } from "../context/ExperienceModeContext";
import { generateQuantitativeInsight } from "../lib/insightGenerator";
import { TimeHorizon, OwnershipState } from "../types/insight";
import GuidedTerminalView from "./terminal/GuidedTerminalView";
import StandardTerminalView from "./terminal/StandardTerminalView";
import AdvancedTerminalView from "./terminal/AdvancedTerminalView";
import WhyInspectModal from "./WhyInspectModal";
import PositionSizerModal from "./PositionSizerModal";

interface AdaptiveTerminalProps {
  symbol: string;
  companyName?: string;
  currentPrice: number;
  changePct: number;
  setupScore?: number;
  isStage4?: boolean;
}

export default function AdaptiveTerminal({
  symbol,
  companyName = "Asset Intelligence",
  currentPrice,
  changePct,
  setupScore = 60,
  isStage4 = false,
}: AdaptiveTerminalProps) {
  const searchParams = useSearchParams();
  const fromGoal = searchParams.get("fromGoal");
  const fromCount = searchParams.get("fromCount");

  const { experienceMode } = useExperienceMode();
  const [isWhyOpen, setIsWhyOpen] = useState(false);
  const [isSizerOpen, setIsSizerOpen] = useState(false);
  const [timeHorizon, setTimeHorizon] = useState<TimeHorizon>("SWING");
  const [ownership, setOwnership] = useState<OwnershipState>("UNKNOWN");

  const insight = generateQuantitativeInsight(
    symbol,
    companyName,
    currentPrice,
    changePct,
    setupScore,
    isStage4 ? 4 : 2,
    timeHorizon,
    ownership
  );

  return (
    <div className="w-full space-y-3 font-sans">
      {/* 🧭 Dead-End Recovery Breadcrumb (if navigated from Screener) */}
      {fromGoal && (
        <div className="flex items-center justify-between bg-[#081322] border border-cyan-800/50 px-3.5 py-2 rounded-xl text-xs font-mono">
          <Link
            href={`/screener?goal=${fromGoal}`}
            className="text-cyan-300 hover:text-white font-bold flex items-center gap-1.5 transition-colors"
          >
            <span>←</span>
            <span>Back to &ldquo;{fromGoal.replace(/_/g, " ").toUpperCase()}&rdquo; Candidates {fromCount ? `(${fromCount} saved)` : ""}</span>
          </Link>
          <span className="text-[10px] text-slate-400 hidden sm:inline">Context & Filters Preserved</span>
        </div>
      )}

      {/* 💼 Ownership Context Prompt (When Ownership is UNKNOWN) */}
      {ownership === "UNKNOWN" && (
        <div className="bg-[#0c121d] border border-[#223147] p-3 rounded-xl flex flex-wrap items-center justify-between gap-2.5 text-xs text-slate-300">
          <span className="font-semibold text-slate-200">
            What is your current relationship with <span className="text-cyan-400 font-mono font-bold">{symbol}</span>?
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setOwnership("NOT_OWNED")}
              className="px-2.5 py-1 rounded-lg bg-[#141d2d] hover:bg-cyan-900/60 border border-[#263750] text-cyan-200 font-mono font-bold text-[11px] transition-all cursor-pointer"
            >
              🔍 Considering buying
            </button>
            <button
              onClick={() => setOwnership("OWNED")}
              className="px-2.5 py-1 rounded-lg bg-[#141d2d] hover:bg-emerald-900/60 border border-[#263750] text-emerald-200 font-mono font-bold text-[11px] transition-all cursor-pointer"
            >
              💼 I already own it
            </button>
            <button
              onClick={() => setOwnership("NOT_OWNED")}
              className="px-2 py-1 text-slate-400 hover:text-slate-200 text-[11px] cursor-pointer"
            >
              📊 Just researching
            </button>
          </div>
        </div>
      )}

      {/* ⏱️ First-Class Time Horizon Selector */}
      <div className="flex items-center justify-between bg-[#080d16] px-3 py-1.5 rounded-xl border border-[#1b2537] text-xs font-mono">
        <span className="text-slate-400 text-[11px] font-bold">Horizon Evaluation:</span>
        <div className="flex items-center gap-1">
          {(["INTRADAY", "SWING", "POSITION", "LONG_TERM"] as TimeHorizon[]).map((hz) => (
            <button
              key={hz}
              onClick={() => setTimeHorizon(hz)}
              className={`px-2 py-0.5 rounded text-[10px] font-bold transition-all cursor-pointer ${
                timeHorizon === hz
                  ? "bg-cyan-600 text-slate-950 font-black"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#131d2c]"
              }`}
            >
              {hz}
            </button>
          ))}
        </div>
      </div>

      {/* 3 Presentation Lenses */}
      {experienceMode === "GUIDED" && (
        <GuidedTerminalView
          insight={insight}
          onOpenSizer={() => setIsSizerOpen(true)}
          onOpenWhy={() => setIsWhyOpen(true)}
        />
      )}

      {experienceMode === "STANDARD" && (
        <StandardTerminalView
          insight={insight}
          onOpenSizer={() => setIsSizerOpen(true)}
          onOpenWhy={() => setIsWhyOpen(true)}
        />
      )}

      {experienceMode === "ADVANCED" && (
        <AdvancedTerminalView
          insight={insight}
          onOpenSizer={() => setIsSizerOpen(true)}
          onOpenWhy={() => setIsWhyOpen(true)}
        />
      )}

      {/* Why Score Attribution Modal */}
      <WhyInspectModal
        isOpen={isWhyOpen}
        onClose={() => setIsWhyOpen(false)}
        symbol={symbol}
        setupScore={insight.setupScore}
        items={insight.scoreAttribution.items}
        catalystToIncreaseScore={insight.scoreAttribution.catalystToIncreaseScore}
        whatWouldChangeAssessment={insight.whatWouldChangeAssessment}
      />

      {/* Institutional Position Sizer Modal */}
      <PositionSizerModal
        isOpen={isSizerOpen}
        onClose={() => setIsSizerOpen(false)}
        symbol={symbol}
        entryPrice={currentPrice}
        stopLoss={insight.standard.keyLevels.stopLoss}
        takeProfit1={insight.standard.keyLevels.target1}
        isStage4={isStage4}
      />
    </div>
  );
}
