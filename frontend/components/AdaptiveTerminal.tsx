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
  dataSource?: "live" | "historical" | "fallback" | "unavailable";
  decisionTrace?: DecisionTrace;
  optimalExecution?: OptimalExecutionPlan;
  freshness?: FreshnessInfo;
  userRole?: "DAY_TRADER" | "LONG_TERM";
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
  userRole = "LONG_TERM",
}: AdaptiveTerminalProps) {
  const searchParams = useSearchParams();
  const fromGoal = searchParams.get("fromGoal");
  const fromCount = searchParams.get("fromCount");
  const urlOwnership = searchParams.get("ownership")?.toUpperCase();

  const { experienceMode } = useExperienceMode();
  const [isWhyOpen, setIsWhyOpen] = useState(false);
  const [isSizerOpen, setIsSizerOpen] = useState(false);
  const effectiveHorizon: TimeHorizon = userRole === "DAY_TRADER" ? "INTRADAY" : "SWING";

  const initialOwnership: OwnershipState =
    urlOwnership === "OWNED" || urlOwnership === "NOT_OWNED"
      ? (urlOwnership as OwnershipState)
      : "UNKNOWN";

  const [ownership, setOwnershipState] = useState<OwnershipState>(initialOwnership);

  // Sync / reset ownership when symbol or URL parameter updates
  React.useEffect(() => {
    const freshOwnership: OwnershipState =
      urlOwnership === "OWNED" || urlOwnership === "NOT_OWNED"
        ? (urlOwnership as OwnershipState)
        : "UNKNOWN";
    setOwnershipState(freshOwnership);
  }, [symbol, urlOwnership]);

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
    effectiveHorizon,
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

      {/* ⏱️ Authoritative Trading Horizon & Evidence Provenance Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 bg-[#080d16] px-3 py-1.5 rounded-xl border border-[#1b2537] text-xs font-mono">
        <div className="flex items-center gap-2">
          <span className="text-slate-400 text-[11px] font-bold">Horizon:</span>
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md text-xs font-bold font-mono bg-[#131d2c] border border-cyan-900/60 text-cyan-300">
            <span>{userRole === "DAY_TRADER" ? "⚡ INTRADAY (Day Scalp)" : "🏛️ SWING (Multi-Day)"}</span>
          </span>
          <span className="text-[10px] text-slate-500 hidden md:inline">
            (Governed by Trading Horizon switch)
          </span>
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

      {/* 3 Presentation Lenses — Unblocked Initial Assessment */}
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

      {experienceMode === "QUANT" && (
        <AdvancedTerminalView
          insight={insight}
          onOpenSizer={() => setIsSizerOpen(true)}
          onOpenWhy={() => setIsWhyOpen(true)}
        />
      )}

      {/* 💼 Contextual Portfolio Refinement (Non-blocking Progressive Disclosure) */}
      <div className="bg-[#090e17] border border-[#1b2537] rounded-xl p-2.5 text-xs text-slate-400">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="text-slate-300 font-medium">Portfolio Relationship:</span>
            <span className="font-mono font-semibold text-cyan-300">
              {ownership === "OWNED" ? "💼 Currently Owned" : ownership === "NOT_OWNED" ? "🔍 Researching / Considering" : "Unspecified"}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => handleSetOwnership("NOT_OWNED")}
              className={`min-h-[44px] sm:min-h-[32px] px-2.5 py-1 rounded-lg font-mono text-[11px] font-semibold transition-colors cursor-pointer flex items-center justify-center ${
                ownership === "NOT_OWNED"
                  ? "bg-cyan-950 text-cyan-300 border border-cyan-700"
                  : "bg-[#111722] hover:bg-[#182232] text-slate-400 border border-[#223147]"
              }`}
            >
              Considering
            </button>
            <button
              type="button"
              onClick={() => handleSetOwnership("OWNED")}
              className={`min-h-[44px] sm:min-h-[32px] px-2.5 py-1 rounded-lg font-mono text-[11px] font-semibold transition-colors cursor-pointer flex items-center justify-center ${
                ownership === "OWNED"
                  ? "bg-emerald-950 text-emerald-300 border border-emerald-700"
                  : "bg-[#111722] hover:bg-[#182232] text-slate-400 border border-[#223147]"
              }`}
            >
              I Own It
            </button>
            {ownership !== "UNKNOWN" && (
              <button
                type="button"
                onClick={() => handleSetOwnership("UNKNOWN")}
                className="min-h-[44px] sm:min-h-[32px] px-2 py-1 text-slate-500 hover:text-slate-300 text-[10px] cursor-pointer flex items-center justify-center"
                title="Reset relationship"
              >
                Reset
              </button>
            )}
          </div>
        </div>
      </div>

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
