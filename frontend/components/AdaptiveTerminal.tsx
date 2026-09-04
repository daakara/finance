"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useExperienceMode } from "../context/ExperienceModeContext";
import { generateQuantitativeInsight } from "../lib/insightGenerator";
import { TimeHorizon, OwnershipState, DecisionTrace, FreshnessInfo } from "../types/insight";
import GuidedTerminalView from "./terminal/GuidedTerminalView";
import StandardTerminalView from "./terminal/StandardTerminalView";
import AdvancedTerminalView from "./terminal/AdvancedTerminalView";
import WhyInspectModal from "./WhyInspectModal";
import PositionSizerModal from "./PositionSizerModal";

import { CandleData, ConfluenceData, OptimalExecutionPlan } from "../lib/api";
import { resolveOverallEvidenceBadge } from "../lib/dataProvenance";

interface AdaptiveTerminalProps {
  symbol: string;
  companyName?: string;
  currentPrice: number;
  changePct: number;
  setupScore?: number;
  confluence?: ConfluenceData;
  isStage4?: boolean;
  candles?: CandleData[];
  dataSource?: "live" | "fallback" | "unavailable";
  decisionTrace?: DecisionTrace;
  optimalExecution?: OptimalExecutionPlan;
  freshness?: FreshnessInfo;
}

export default function AdaptiveTerminal({
  symbol,
  companyName = "Asset Intelligence",
  currentPrice,
  changePct,
  setupScore,
  confluence,
  isStage4,
  candles,
  dataSource,
  decisionTrace,
  optimalExecution,
  freshness,
}: AdaptiveTerminalProps) {
  const searchParams = useSearchParams();
  const fromGoal = searchParams.get("fromGoal");
  const fromCount = searchParams.get("fromCount");
  const urlOwnership = searchParams.get("ownership")?.toUpperCase();

  const { experienceMode } = useExperienceMode();
  const [isWhyOpen, setIsWhyOpen] = useState(false);
  const [isSizerOpen, setIsSizerOpen] = useState(false);
  const [timeHorizon, setTimeHorizon] = useState<TimeHorizon>("SWING");

  const initialOwnership: OwnershipState =
    urlOwnership === "OWNED" || urlOwnership === "NOT_OWNED"
      ? (urlOwnership as OwnershipState)
      : "UNKNOWN";

  const [ownership, setOwnershipState] = useState<OwnershipState>(initialOwnership);

  // Sync ownership if URL parameter updates
  React.useEffect(() => {
    if (urlOwnership === "OWNED" || urlOwnership === "NOT_OWNED") {
      setOwnershipState(urlOwnership as OwnershipState);
    }
  }, [urlOwnership]);

  const handleSetOwnership = (newOwnership: OwnershipState) => {
    setOwnershipState(newOwnership);
    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      if (newOwnership === "UNKNOWN") {
        url.searchParams.delete("ownership");
      } else {
        url.searchParams.set("ownership", newOwnership);
      }
      window.history.replaceState({}, "", url.toString());
    }
  };

  const insight = generateQuantitativeInsight(
    symbol,
    companyName,
    currentPrice,
    changePct,
    setupScore,
    isStage4 !== undefined ? (isStage4 ? 4 : 2) : undefined,
    timeHorizon,
    ownership,
    "USER_DECLARED",
    candles,
    dataSource,
    confluence,
    decisionTrace,
    optimalExecution,
    freshness?.status
  );

  const evidenceBadge = resolveOverallEvidenceBadge({
    hasLiveFeed: dataSource === "live",
    candleCount: candles?.length || 0,
    hasSecFilings: Boolean(confluence?.pillars?.some((p) => p.pillar === "FUNDAMENTAL_SOLVENCY" && p.status === "positive")),
    isCataloged: Boolean(confluence),
    price: currentPrice,
  });

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
              onClick={() => handleSetOwnership("NOT_OWNED")}
              className="px-2.5 py-1 rounded-lg bg-[#141d2d] hover:bg-cyan-900/60 border border-[#263750] text-cyan-200 font-mono font-bold text-[11px] transition-all cursor-pointer"
            >
              🔍 Considering buying
            </button>
            <button
              onClick={() => handleSetOwnership("OWNED")}
              className="px-2.5 py-1 rounded-lg bg-[#141d2d] hover:bg-emerald-900/60 border border-[#263750] text-emerald-200 font-mono font-bold text-[11px] transition-all cursor-pointer"
            >
              💼 I already own it
            </button>
            <button
              onClick={() => handleSetOwnership("NOT_OWNED")}
              className="px-2 py-1 text-slate-400 hover:text-slate-200 text-[11px] cursor-pointer"
            >
              📊 Just researching
            </button>
          </div>
        </div>
      )}

      {/* ⏱️ First-Class Time Horizon Selector & Evidence State Pill */}
      <div className="flex flex-wrap items-center justify-between gap-2 bg-[#080d16] px-3 py-1.5 rounded-xl border border-[#1b2537] text-xs font-mono">
        <div className="flex items-center gap-2">
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

        {/* 🛡️ Evidence State Provenance Badge */}
        <div
          title={evidenceBadge.tooltip}
          className={`flex items-center gap-1.5 px-2.5 py-0.5 rounded-md border text-[10px] font-mono font-semibold cursor-help transition-colors ${evidenceBadge.badgeClass}`}
        >
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-current" />
          <span>{evidenceBadge.label}</span>
        </div>
      </div>

      {/* ⚠️ Ineligible / Limited Evidence Notice */}
      {insight.terminalState.overallEligibility !== "ELIGIBLE" && (
        <div className="bg-[#181106] border border-amber-800/60 p-3 rounded-xl flex items-start gap-2.5 text-xs text-amber-200" role="alert">
          <span className="text-base shrink-0">⚠️</span>
          <div className="space-y-1">
            <strong className="font-mono font-bold block text-amber-300">
              {insight.terminalState.overallEligibility === "INELIGIBLE"
                ? "Insufficient Evidence to Derive Confident Posture"
                : "Partial Evidence: Reduced Domain Confidence"}
            </strong>
            <p className="text-slate-300 text-[11px] font-sans leading-relaxed">
              {insight.terminalState.headlineExplanation} Some model inputs (e.g. quarterly SEC filings or options flow) are unavailable. Missing data is treated as unassessed, not negative.
            </p>
          </div>
        </div>
      )}

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

      {/* Why Score Attribution Modal with Full Provenance */}
      <WhyInspectModal
        isOpen={isWhyOpen}
        onClose={() => setIsWhyOpen(false)}
        symbol={symbol}
        setupScore={insight.setupScore}
        terminalState={insight.terminalState}
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
        riskRewardRatio={insight.standard.keyLevels.profitRiskRatio}
        isStage4={isStage4 || (insight.advanced.vcpStage === undefined && insight.verdict !== "ACTIONABLE_BUY_ZONE")}
      />
    </div>
  );
}
