"use client";

import React, { useState, useEffect, useRef } from "react";
import { MASTER_ASSET_CATALOG } from "../lib/masterCatalog";
import { trackPreFlightOutcome, trackTradePlanCopied } from "../lib/matomo";
import { evaluatePreFlightChecks, PreFlightChecklistInputs, PREFLIGHT_CONFIG } from "../lib/preflightEvaluator";

export interface PreFlightChecklistModalProps extends PreFlightChecklistInputs {
  isOpen: boolean;
  onClose: () => void;
}

export default function PreFlightChecklistModal({
  isOpen,
  onClose,
  symbol,
  currentPrice,
  stopLoss,
  takeProfit1,
  riskRewardRatio,
  setupPattern = "Minervini Volatility Contraction Pattern (VCP 3-Stage)",
  isDayTrader = false,
  isStage4 = false,
  optimalEntryMin,
  optimalEntryMax,
  breakoutPivot,
  isDistributionTrap,
  hasImminentEarnings = null,
  catalystState,
  catalystEvidence,
  vix,
  vixObservationTime,
  macroEvidence,
  isActionable = false,
  canSizeTrade,
  decisionState,
  dataSource,
  isSynthetic,
  isHistorical,
  quoteStatus,
  liveSource,
  liveFreshnessStatus,
  marketSession,
  observationTime,
  shortFloat,
  qualityScore,
  assetClass,
  securityType,
  verdict,
}: PreFlightChecklistModalProps) {
  const [copied, setCopied] = useState<boolean>(false);
  const [vernacularMode, setVernacularMode] = useState<"PLAIN_ENGLISH" | "PRO_QUANT">("PLAIN_ENGLISH");

  const modalRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLElement | null>(null);
  const lastEvaluatedKeyRef = useRef<string | null>(null);

  // Consolidated effect — active only while modal is open.
  // Handles: localStorage read, vernacular listener, Escape key, Tab focus trap.
  useEffect(() => {
    if (!isOpen) return;

    // 1. Restore persisted vernacular preference
    try {
      const saved = localStorage.getItem("ARX_VERNACULAR_MODE") as "PLAIN_ENGLISH" | "PRO_QUANT" | null;
      if (saved) setVernacularMode(saved);
    } catch {}

    // 2. Live vernacular change listener
    const handleVernacular = (e: Event) => {
      const custom = e as CustomEvent<"PLAIN_ENGLISH" | "PRO_QUANT">;
      if (custom.detail) setVernacularMode(custom.detail);
    };
    window.addEventListener("finance:vernacular-change", handleVernacular);

    // 3. Escape closes; Tab cycles within modal (focus trap)
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") { onClose(); return; }
      if (e.key === "Tab" && modalRef.current) {
        const focusable = Array.from(
          modalRef.current.querySelectorAll<HTMLElement>(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
          )
        ).filter((el) => !el.hasAttribute("disabled"));
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey) {
          if (document.activeElement === first) { e.preventDefault(); last.focus(); }
        } else {
          if (document.activeElement === last) { e.preventDefault(); first.focus(); }
        }
      }
    };
    document.addEventListener("keydown", handleKeyDown);

    // 4. Move focus into modal; remember what opened it
    triggerRef.current = document.activeElement as HTMLElement;
    modalRef.current?.querySelector<HTMLElement>("button")?.focus();

    return () => {
      window.removeEventListener("finance:vernacular-change", handleVernacular);
      document.removeEventListener("keydown", handleKeyDown);
      triggerRef.current?.focus();
    };
  }, [isOpen, onClose]);

  const isPlain = vernacularMode === "PLAIN_ENGLISH";
  const evalResult = evaluatePreFlightChecks(
    {
      symbol,
      currentPrice,
      stopLoss,
      takeProfit1,
      riskRewardRatio,
      setupPattern,
      isDayTrader,
      isStage4,
      optimalEntryMin,
      optimalEntryMax,
      breakoutPivot,
      isDistributionTrap,
      hasImminentEarnings,
      catalystState,
      catalystEvidence,
      vix,
      vixObservationTime,
      macroEvidence,
      isActionable,
      canSizeTrade,
      decisionState,
      dataSource,
      isSynthetic,
      isHistorical,
      quoteStatus,
      liveSource,
      liveFreshnessStatus,
      marketSession,
      observationTime,
      shortFloat,
      qualityScore,
      assetClass,
      securityType,
      verdict,
    },
    vernacularMode
  );

  const {
    checks,
    passedCount,
    convictionPct,
    activeBlocker,
    whatNeedsToChange,
    safePrice,
    safeStop,
    safeTarget,
    safeRR,
    safeEntryMin,
    safeEntryMax,
    safePivot,
    currentVix,
    stopLossPct,
    target1Pct,
    target2Pct,
    safeTarget2,
    entryZoneStr,
    stopLossStr,
    target1Str,
    target2Str,
    resolvedCatalystState,
  } = evalResult;

  const isActionableGranted = isActionable === true;
  const isCleared = isActionableGranted && evalResult.isCleared;
  const isCatalystConfirmed = catalystEvidence?.confirmationStatus === "CONFIRMED";

  const cleanSym = (symbol || "ASSET").toUpperCase().replace("-USD", "");
  const catalogItem = MASTER_ASSET_CATALOG[cleanSym];
  const isPriceValid = typeof currentPrice === "number" && !isNaN(currentPrice) && currentPrice > 0;
  const isRRValid = (typeof riskRewardRatio === "number" && !isNaN(riskRewardRatio) && riskRewardRatio > 0) || (typeof safeRR === "number" && !isNaN(safeRR) && safeRR > 0);
  const hasExecutionLevels = safeStop !== null && safeTarget !== null;
  const isLevelDirectionValid = evalResult.isLevelDirectionValid ?? (isPriceValid && hasExecutionLevels && safeStop! < safePrice && safeTarget! > safePrice);
  const isRRPassed = checks[0].passed;
  const isTrendPassed = checks[1].passed && !isStage4;
  const isExtendedAboveZone = Boolean(safeEntryMax && safePrice > safeEntryMax * 1.02);
  const isWithinTolerance = Boolean(safeEntryMax && safePrice > safeEntryMax && safePrice <= safeEntryMax * 1.02);
  const isInCorridor = Boolean(safeEntryMin && safeEntryMax && safePrice >= safeEntryMin && safePrice <= safeEntryMax);
  const isBelowCorridor = Boolean(safeEntryMin && safePrice < safeEntryMin);
  const isCapitalRiskPassed = checks[2].passed;
  const isCapitalRiskFailed = checks[2].status === "CAPITAL RISK";
  const resolvedShortFloat = (shortFloat !== undefined && shortFloat !== null)
    ? Number(shortFloat)
    : (catalogItem?.shortFloat !== undefined && catalogItem.shortFloat !== null)
    ? Number(catalogItem.shortFloat)
    : null;
  const resolvedQualityScore = (qualityScore !== undefined && qualityScore !== null)
    ? Number(qualityScore)
    : (catalogItem?.qualityScore !== undefined && catalogItem.qualityScore !== null)
    ? Number(catalogItem.qualityScore)
    : null;
  const isEtf = assetClass === "ETF" || securityType === "ETF" || catalystEvidence?.coverageStatus === "NOT_APPLICABLE";
  const hasShortFloat = resolvedShortFloat !== null && !isNaN(resolvedShortFloat);
  const hasQualityScore = resolvedQualityScore !== null && !isNaN(resolvedQualityScore);
  const capitalRiskEvidenceState: "COMPLETE" | "PARTIAL" | "UNAVAILABLE" = isEtf
    ? "COMPLETE"
    : hasShortFloat && hasQualityScore
    ? "COMPLETE"
    : hasShortFloat || hasQualityScore
    ? "PARTIAL"
    : "UNAVAILABLE";
  const isCatalystPassed = checks[3].passed;
  const isMacroPassed = checks[4].passed;

  const isVixValid = typeof vix === "number" && !isNaN(vix) && vix > 0;
  const isVixElevated = Boolean(evalResult?.isVixElevated ?? (isMacroPassed && currentVix !== null && currentVix >= 20.0 && currentVix < 26.0));

  const renderCheck2Detail = () => {
    if (isStage4) {
      return isPlain
        ? `⚠️ Watchlist Only: Readiness input price ($${safePrice.toFixed(2)}) is in Stage 4 correction below 50-day average. Await base formation.`
        : `Stage 4 correction structure: Readiness input ($${safePrice.toFixed(2)}) requires 50-day breakout pivot ${safePivot !== null ? `above $${safePivot.toFixed(2)}` : "(pivot uncalculated)"}.`;
    }
    if (safeEntryMin === null || safeEntryMax === null) {
      return "Optimal buy zone levels uncalculated. Awaiting technical structure.";
    }
    if (isExtendedAboveZone) {
      return isPlain
        ? `⚠️ Extended: Price is above ideal buy zone ($${safeEntryMin.toFixed(2)} - $${safeEntryMax.toFixed(2)}). Wait for pullback.`
        : "Extended structure: Readiness input is above value area. Chasing creates negative R:R risk.";
    }
    if (isWithinTolerance) {
      return isPlain
        ? `Readiness input price ($${safePrice.toFixed(2)}) is slightly above preferred buying range ($${safeEntryMin.toFixed(2)} - $${safeEntryMax.toFixed(2)}), within allowable extension tolerance (+2%). Defending key support.`
        : `Readiness input price ($${safePrice.toFixed(2)}) is above corridor ceiling ($${safeEntryMax.toFixed(2)}) within +2% extension tolerance. Defending key moving average support (20 EMA / 50 SMA).`;
    }
    if (isBelowCorridor) {
      return isPlain
        ? `Readiness input price ($${safePrice.toFixed(2)}) is below preferred buying range ($${safeEntryMin.toFixed(2)} - $${safeEntryMax.toFixed(2)}).`
        : `Readiness input price ($${safePrice.toFixed(2)}) is below preferred accumulation corridor ($${safeEntryMin.toFixed(2)}–$${safeEntryMax.toFixed(2)}).`;
    }
    return isPlain
      ? `Readiness input price ($${safePrice.toFixed(2)}) is inside the verified buying range defending key support.`
      : "Defending key moving average support (20 EMA / 50 SMA).";
  };

  const renderCheck4Detail = () => {
    if (isEtf || catalystEvidence?.coverageStatus === "NOT_APPLICABLE") {
      return isPlain
        ? "Exchange-Traded Fund (ETF): Does not report single-stock corporate earnings. Event runway clear."
        : "Asset Class ETF: Pooled vehicle exempt from single-issuer corporate earnings calendar hazard.";
    }
    if (resolvedCatalystState === "PASS") {
      const isConfirmed = catalystEvidence?.confirmationStatus === "CONFIRMED";
      const dteStr = catalystEvidence?.daysToEarnings ? ` (${catalystEvidence.daysToEarnings.toFixed(1)}d DTE)` : "";
      if (isPlain) {
        return isConfirmed
          ? `Verified event runway: Confirmed earnings announcement scheduled beyond the 7-day trade window${dteStr}.`
          : `Unconfirmed event runway: Estimated announcement scheduled beyond the 7-day trade window${dteStr}, but source confirmation remains unverified.`;
      }
      return isConfirmed
        ? `Catalyst Buffer: Clean window verified (>7 days to confirmed binary catalyst${dteStr}).`
        : `Catalyst Buffer: Clean window indicated (${catalystEvidence?.daysToEarnings ? catalystEvidence.daysToEarnings.toFixed(1) + "d DTE" : ">7d"}), but corporate event confirmation is unverified.`;
    }
    if (resolvedCatalystState === "WARN") {
      return isPlain
        ? `⚠️ High Risk: Scheduled earnings announcement ${catalystEvidence?.daysToEarnings !== undefined && catalystEvidence?.daysToEarnings !== null ? (catalystEvidence.daysToEarnings <= 1.0 ? "in <= 24h" : `in ${catalystEvidence.daysToEarnings.toFixed(1)} days`) : "within 7-day window"}${catalystEvidence?.eventDate ? ` (${catalystEvidence.eventDate})` : ""}.`
        : `⚠️ Binary Gap Risk: Earnings announcement scheduled for ${catalystEvidence?.eventDate || "imminent"}${catalystEvidence?.callTimeEt ? ` (${catalystEvidence.callTimeEt} ET Call)` : ""}. Position sizing restricted.`;
    }
    if (resolvedCatalystState === "ERROR") {
      return isPlain
        ? "Technical error retrieving corporate earnings schedule."
        : "Catalyst Evaluation Error: Provider telemetry stream failed to resolve.";
    }
    if (catalystEvidence?.daysToEarnings !== undefined && catalystEvidence?.daysToEarnings !== null && catalystEvidence.daysToEarnings < 0) {
      return isPlain
        ? `Prior earnings reported (${catalystEvidence.eventDate || "historical"}), but forward calendar is unverified. Cannot certify forward runway.`
        : `Historical Event Elapsed: Prior report on ${catalystEvidence.eventDate || "historical record"}. Forward schedule unverified; runway uncertified.`;
    }
    return isPlain
      ? "Earnings calendar coverage unavailable: Event runway cannot be certified."
      : "Catalyst Telemetry Unavailable: Corporate event schedule unverified from primary source.";
  };

  // Analytics — strictly single dispatch per unique evaluation event via useEffect (P1-F04 / PREFLIGHT-001)
  useEffect(() => {
    if (!isOpen) {
      lastEvaluatedKeyRef.current = null;
      return;
    }

    // Evaluation event identity incorporates asset, observation timestamp / price, and outcome
    const evaluationEventKey = `${symbol}:${observationTime ?? safePrice}:${passedCount}:${isCleared}`;
    if (lastEvaluatedKeyRef.current !== evaluationEventKey) {
      try {
        trackPreFlightOutcome(symbol, passedCount, isCleared);
        lastEvaluatedKeyRef.current = evaluationEventKey;
      } catch (err) {
        console.warn("Analytics error in PreFlightChecklistModal:", err);
      }
    }
  }, [isOpen, symbol, observationTime, safePrice, passedCount, isCleared]);

  // Trade plan markdown — differentiated by vernacularMode
  const isNonLiveOrBlocked =
    !isCleared ||
    evalResult.isSimulatedClearance ||
    evalResult.isHistoricalReference ||
    evalResult.provenanceCategory === "UNVERIFIED";

  const exportHeaderNotice = isNonLiveOrBlocked
    ? `> ⚠️ **RESEARCH / JOURNAL ONLY — NOT EXECUTABLE**\n> Epistemic Mode: ${evalResult.provenanceCategory} · ${
        !isCleared
          ? `Clearance Blockers: ${evalResult.allBlockers.join("; ") || activeBlocker}`
          : evalResult.isSimulatedClearance
          ? "Notice: Synthetic fixture — not authorized for live order execution."
          : evalResult.isHistoricalReference
          ? "Notice: Historical EOD observation — requires live market confirmation."
          : "Notice: Unverified market feed — requires verified real-time data before execution."
      }\n\n`
    : isVixElevated
    ? `> ⚠️ **CAUTION: ELEVATED MARKET VOLATILITY (VIX ${currentVix?.toFixed(1)})**\n> Volatility regime elevated (20.0–26.0) — exercise caution on position sizing.\n\n`
    : "";

  const clearanceExportLabelPlain = isCleared
    ? (evalResult.isSimulatedClearance
        ? "SIMULATED CLEARANCE — NOT EXECUTABLE"
        : evalResult.isHistoricalReference
        ? "HISTORICAL REFERENCE — NOT EXECUTABLE"
        : isVixElevated
        ? "⚠️ CLEARED — ELEVATED VOLATILITY CAUTION"
        : "🟢 CLEARED")
    : `⚠️ NOT CLEARED — ${activeBlocker}`;

  const clearanceExportLabelPro = isCleared
    ? (evalResult.isSimulatedClearance
        ? "SIMULATED CLEARANCE — NOT EXECUTABLE"
        : evalResult.isHistoricalReference
        ? "HISTORICAL REFERENCE — NOT EXECUTABLE"
        : isVixElevated
        ? "⚠️ CLEARED — ELEVATED VOLATILITY CAUTION"
        : "🟢 CLEARED FOR EXECUTION")
    : `⚠️ NOT CLEARED (${activeBlocker})`;

  const tradePlanMarkdown = isPlain
    ? `${exportHeaderNotice}### 📋 ARX Terminal Trade Plan: ${symbol}
- **Date**: ${new Date().toISOString().split("T")[0]}
- **Asset**: ${symbol} | **Mode**: ${isDayTrader ? "⚡ Day Trader" : "🏛️ Swing / Long-Term Compounder"}
- **Current Price**: $${safePrice.toFixed(2)}
- **Buy Zone**: ${entryZoneStr}
- **Stop Loss**: ${stopLossStr}
- **Profit Goal 1 (TP1)**: ${target1Str}
- **Profit Goal 2 (TP2 Runner)**: ${target2Str}
- **Tactical Rule (Protected Trailing Transition)**: ${safeTarget !== null ? `When Profit Goal 1 ($${safeTarget.toFixed(2)}) is hit, sell 50% of position and immediately move Stop Loss on remaining 50% to purchase price ($${safePrice.toFixed(2)}) to lock in a protected trade.` : "Establish verified target levels before executing tactical scaling rules."}
- **Setup**: ${setupPattern || "Minervini VCP Pattern"}
- **Pre-Flight Score**: ${passedCount}/5 Checks Passed (${convictionPct}% Checklist) · ${clearanceExportLabelPlain}

> ⚠️ **PLATFORM HOLD NOTICE (HOLD_EXEC_001)**: Simulation and institutional quantitative research only. Zero live order execution or capital commitment authorized. All levels require live session confirmation before execution.
`
    : `${exportHeaderNotice}### 📋 ARX Institutional Execution Brief: ${symbol}
- **Date**: ${new Date().toISOString().split("T")[0]}
- **Asset**: ${symbol} | **Mode**: ${isDayTrader ? "Intraday Momentum Pullback" : "Swing / Compounder Accumulation"}
- **Readiness input**: $${safePrice.toFixed(2)} | **Optimal Entry**: ${entryZoneStr}
- **Hard Stop / Invalidation**: ${stopLossStr}
- **Target 1 (TP1)**: ${target1Str}
- **Target 2 (TP2 Runner)**: ${target2Str}
- **Execution Rule (Protected Trailing Ratchet)**: ${safeTarget !== null ? `Scale 0.50x tranche @ TP1 ($${safeTarget.toFixed(2)}). Ratchet trailing stop to cost basis ($${safePrice.toFixed(2)}) to ensure protected runner convexity.` : "Establish verified target levels before executing tactical scaling rules."}
- **Risk/Reward**: ${isRRValid && hasExecutionLevels && isLevelDirectionValid ? `${safeRR.toFixed(2)}:1 (${PREFLIGHT_CONFIG.REQUIRED_RR_LABEL})` : `Unavailable (${PREFLIGHT_CONFIG.REQUIRED_RR_LABEL})`}
- **Setup Pattern**: ${setupPattern || "Minervini VCP"}
- **Pre-Flight Clearance**: ${passedCount}/5 Checks Passed (${convictionPct}% Checklist) — ${clearanceExportLabelPro}
- **VIX Regime**: ${currentVix !== null ? currentVix.toFixed(1) : "UNAVAILABLE"}${isVixElevated ? " (ELEVATED CAUTION)" : ""} | **Capital Risk**: ${isCapitalRiskPassed ? "PASS" : isCapitalRiskFailed ? "DETECTED" : capitalRiskEvidenceState} | **Earnings Hazard**: ${resolvedCatalystState === "PASS" ? "CLEAR" : resolvedCatalystState === "WARN" ? "HAZARD" : resolvedCatalystState}

> ⚠️ **PLATFORM HOLD NOTICE (HOLD_EXEC_001)**: Simulation and institutional quantitative research only. Zero live order execution or capital commitment authorized. All levels require live session confirmation before execution.
`;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(tradePlanMarkdown);
      setCopied(true);
      trackTradePlanCopied(symbol, setupPattern);
      setTimeout(() => setCopied(false), 2500);
    } catch (err) {
      console.warn("Failed to copy trade plan:", err);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[1200] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-150"
      aria-modal="true"
      role="dialog"
      aria-labelledby="preflight-title"
    >
      <div
        ref={modalRef}
        className="bg-[var(--surface-1)] border border-cyan-800/80 rounded-2xl max-w-xl w-full p-5 sm:p-6 shadow-2xl font-sans text-[var(--text-1)] relative flex flex-col max-h-[90vh]"
      >
        {/* Header — pinned */}
        <div className="flex items-center justify-between border-b border-[var(--border-subtle)] pb-3.5 shrink-0">
          <div className="flex items-center gap-2.5">
            <span className="text-xl">✈️</span>
            <div>
              <h2 id="preflight-title" className="text-base sm:text-lg font-bold text-white tracking-tight flex items-center gap-2">
                <span>{isPlain ? `Pre-Flight Trade Checklist: ${symbol}` : `Institutional Pre-Flight Clearance: ${symbol}`}</span>
              </h2>
              <p className="text-xs text-[var(--text-2)]">
                {isPlain
                  ? "5-point risk and confirmation checklist before entering this position."
                  : "Automated pre-trade validation gate enforcing risk-reward and flow confluence."}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close Pre-Flight Checklist"
            className="text-[var(--text-2)] hover:text-white p-1.5 rounded-lg hover:bg-slate-800 transition"
          >
            ✕
          </button>
        </div>

        {/* Scrollable body — barometer + 5-point checklist */}
        <div className="overflow-y-auto flex-1 space-y-4 py-4 pr-0.5">
          {/* Actionability State Banner (Awaiting Trigger / Non-Actionable Gate) */}
          {isActionable === false && (
            <div className="p-3 rounded-xl bg-amber-950/40 border border-amber-600/60 text-xs text-amber-200 flex items-start gap-2.5">
              <span className="text-base shrink-0">⏸️</span>
              <div className="space-y-1 w-full">
                <strong className="font-bold text-[11px] uppercase tracking-wide block text-amber-300 font-mono">
                  {isPlain
                    ? "TRADE NOT CLEARED: AWAITING CONFIRMATION"
                    : "NON-ACTIONABLE STATE: AWAITING TRIGGER CONFIRMATION"}
                </strong>
                <p className="text-[11px] leading-relaxed text-[var(--text-2)] font-sans">
                  {isPlain
                    ? `A valid setup is forming for ${symbol}, but the entry trigger has not confirmed yet. This checklist reviews setup and risk conditions; it does not bypass the required confirmation trigger.`
                    : `Valid setup structure is present (${decisionState || "VALID_SETUP"}), but execution criteria remain incomplete. Current state: WAIT FOR TRIGGER. Sizing and execution remain locked until confirmation conditions are satisfied.`}
                </p>
                <details className="pt-1 text-[10px] text-[var(--text-2)] font-mono">
                  <summary className="cursor-pointer hover:text-slate-200 select-none">
                    Technical Provenance
                  </summary>
                  <div className="mt-1 pl-2 border-l border-amber-800/40 text-[10px] space-y-0.5 text-[var(--text-2)]">
                    <div>State: {decisionState || "VALID_SETUP"}</div>
                    <div>Actionable: false</div>
                    <div>Authority: Canonical Decision Engine</div>
                  </div>
                </details>
              </div>
            </div>
          )}

          {/* Warning banner when price or execution levels are unverified */}
          {!isPriceValid && (
            <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-800/60 text-xs font-mono text-rose-300 flex items-center gap-2">
              <span className="text-base">⚠️</span>
              <span>Readiness input price or execution levels are unverified for this asset. Trade execution is blocked until verified market data is received.</span>
            </div>
          )}

          {/* Clearance Conviction Barometer */}
          <div className={`p-3.5 rounded-xl border flex items-center justify-between ${
            isCleared
              ? (isVixElevated
                  ? "bg-amber-950/40 border-amber-700/80 text-amber-300"
                  : "bg-emerald-950/40 border-emerald-700/80 text-emerald-300")
              : "bg-amber-950/40 border-amber-700/80 text-amber-300"
          }`}>
            <div>
              <span className="text-[10px] uppercase font-bold tracking-wider font-mono block">
                {isPlain ? "Trade Readiness Score" : "Quantitative Clearance Status"}
              </span>
              <div className="text-base sm:text-lg font-extrabold flex items-center gap-1.5">
                <span>
                  {isCleared
                    ? (evalResult.isSimulatedClearance
                        ? "SIMULATED CLEARANCE — NOT EXECUTABLE"
                        : evalResult.isHistoricalReference
                        ? "HISTORICAL REFERENCE — NOT EXECUTABLE"
                        : isVixElevated
                        ? "⚠️ CLEARED — ELEVATED VOLATILITY CAUTION"
                        : "🟢 CLEARED TO EXECUTE")
                    : `⚠️ NOT CLEARED — ${activeBlocker}`}
                </span>
                <span className="text-xs font-mono font-normal">({convictionPct}% Checklist)</span>
              </div>
            </div>
            <div className="text-right font-mono">
              <span className="text-2xl sm:text-3xl font-black">{passedCount}/5</span>
              <span className="text-[10px] text-[var(--text-2)] block">Checks Passed</span>
            </div>
          </div>

          {/* Active Blockers List when multiple blockers exist (PREFLIGHT-001 Target 6) */}
          {!isCleared && evalResult.allBlockers.length > 1 && (
            <div data-testid="preflight-all-blockers" className="p-3 rounded-xl bg-amber-950/40 border border-amber-700/80 text-xs text-amber-200 space-y-1.5">
              <span className="text-[10px] uppercase font-bold tracking-wider font-mono block text-amber-400">
                Active Execution Blockers ({evalResult.allBlockers.length}):
              </span>
              <ul className="list-disc list-inside space-y-0.5 font-mono text-[11px]">
                {evalResult.allBlockers.map((b, idx) => (
                  <li key={idx} className="text-amber-300">{b}</li>
                ))}
              </ul>
            </div>
          )}

          {/* 5-Point Validation Checklist */}
          <div className="space-y-2.5 text-xs">
            {/* Check 1: Reward vs. Risk (Issue #181, ARX-UX-RR-030) */}
            <div className="p-3 rounded-lg bg-[var(--surface-2)] border border-[var(--border-subtle)] flex items-start justify-between gap-3">
              <div className="space-y-0.5 min-w-0 flex-1">
                <div className="font-bold text-[var(--text-1)] flex items-center gap-1.5 flex-wrap">
                  <span aria-hidden="true">{isRRPassed ? "✅" : "❌"}</span>
                  <span className="break-words">
                    1.{" "}
                    <span data-testid="preflight-chk1-title">
                      {isPlain ? "Reward vs. Risk" : "Reward vs. Risk (Asymmetric Payoff)"}
                    </span>
                  </span>
                </div>
                <p className="text-[11px] text-[var(--text-2)] pl-5 break-words">
                  Current:{" "}
                  <strong className="text-cyan-300 font-mono">
                    {isRRValid && hasExecutionLevels && isLevelDirectionValid
                      ? `${safeRR.toFixed(2)}:1 (${PREFLIGHT_CONFIG.REQUIRED_RR_LABEL})`
                      : `Unavailable (${PREFLIGHT_CONFIG.REQUIRED_RR_LABEL})`}
                  </strong>{" "}
                  <span className="text-[var(--text-2)]">
                    {isRRPassed
                      ? `(${PREFLIGHT_CONFIG.EXPLANATION_ADEQUATE})`
                      : !hasExecutionLevels
                      ? "(Execution levels missing: Stop and target required)"
                      : !isLevelDirectionValid
                      ? "(Invalid execution boundaries: Stop and target levels inverted)"
                      : isRRValid
                      ? `(${PREFLIGHT_CONFIG.EXPLANATION_BELOW_MINIMUM})`
                      : "(Evidence incomplete: risk-reward unavailable)"}
                  </span>
                </p>
              </div>
              <span
                data-testid="preflight-chk1-badge"
                className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
                  isRRPassed
                    ? "bg-emerald-950 text-emerald-300 border-emerald-800"
                    : !hasExecutionLevels || !isLevelDirectionValid || !isRRValid
                    ? "bg-slate-900 text-[var(--text-2)] border-slate-700"
                    : "bg-rose-950 text-rose-300 border-rose-800"
                }`}
                aria-label={`Status: ${
                  isRRPassed
                    ? PREFLIGHT_CONFIG.LABEL_PASS
                    : !hasExecutionLevels || !isLevelDirectionValid || !isRRValid
                    ? PREFLIGHT_CONFIG.LABEL_UNAVAILABLE
                    : PREFLIGHT_CONFIG.LABEL_BELOW_MINIMUM
                }`}
              >
                {isRRPassed
                  ? PREFLIGHT_CONFIG.LABEL_PASS
                  : !hasExecutionLevels || !isLevelDirectionValid || !isRRValid
                  ? PREFLIGHT_CONFIG.LABEL_UNAVAILABLE
                  : PREFLIGHT_CONFIG.LABEL_BELOW_MINIMUM}
              </span>
            </div>

            {/* Check 2 */}
            <div className="p-3 rounded-lg bg-[var(--surface-2)] border border-[var(--border-subtle)] flex items-start justify-between gap-3">
              <div className="space-y-0.5">
                <div className="font-bold text-[var(--text-1)] flex items-center gap-1.5">
                  <span>{isTrendPassed ? "✅" : (isStage4 ? "⏳" : "⚠️")}</span>
                  <span>{isPlain ? "2. Trend & Moving Averages (Stage Discipline & Buying Range)" : "2. Technical Structure (Above 20 EMA / 50 SMA Pivot)"}</span>
                </div>
                <p className="text-[11px] text-[var(--text-2)] pl-5">
                  {renderCheck2Detail()}
                </p>
              </div>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
                isTrendPassed
                  ? "bg-emerald-950 text-emerald-300 border-emerald-800"
                  : (isStage4 ? "bg-amber-950 text-amber-300 border-amber-800" : (safeEntryMin === null ? "bg-rose-950 text-rose-300 border-rose-800" : "bg-rose-950 text-rose-300 border-rose-800"))
              }`}>{isTrendPassed ? "PASS" : (isStage4 ? "STAGE 4 WAIT" : (safeEntryMin === null ? "MISSING LEVELS" : "CHASING"))}</span>
            </div>

            {/* Check 3 */}
            <div className="p-3 rounded-lg bg-[var(--surface-2)] border border-[var(--border-subtle)] flex items-start justify-between gap-3">
              <div className="space-y-0.5">
                <div className="font-bold text-[var(--text-1)] flex items-center gap-1.5">
                  <span>{isCapitalRiskPassed ? "✅" : (isCapitalRiskFailed ? "❌" : "⚠️")}</span>
                  <span>{isPlain ? "3. Capital Health & Squeeze Risk (Manageable short float & solvent balance sheet)" : "3. Structural Capital Risk (Short Interest Floor & Quality Rating)"}</span>
                </div>
                <p className="text-[11px] text-[var(--text-2)] pl-5">
                  {isEtf
                    ? (isPlain
                        ? "Exchange-Traded Fund (ETF): Diversified pooled capital structure. Single-stock solvency and squeeze risk not applicable."
                        : "ETF Capital Structure: Index/basket vehicle exempt from single-issuer solvency and short-squeeze constraints.")
                    : isCapitalRiskPassed
                    ? (hasShortFloat
                        ? (isPlain
                            ? `Short interest is moderate (${resolvedShortFloat?.toFixed(1)}% < 12%) and financial quality score (${resolvedQualityScore}/100) indicates stable balance sheet health.`
                            : `Capital Risk Guard: Short float (${resolvedShortFloat?.toFixed(1)}% < 12%) and fundamental quality score (${resolvedQualityScore} >= 60) confirm balance sheet health.`)
                        : (isPlain
                            ? `Financial quality score (${resolvedQualityScore}/100) indicates stable balance sheet health.`
                            : `Capital Risk Guard: Fundamental quality score (${resolvedQualityScore} >= 60) confirms balance sheet health.`))
                    : isCapitalRiskFailed
                    ? (resolvedQualityScore !== null && resolvedQualityScore < 60
                        ? (isPlain
                            ? `⚠️ Warning: Low financial quality score (${resolvedQualityScore}/100 < 60) indicates fundamental balance sheet distress.`
                            : `Capital Risk Warning: Fundamental quality score (${resolvedQualityScore} < 60 threshold) indicates financial distress.`)
                        : resolvedShortFloat !== null && resolvedShortFloat > 12.0
                        ? (isPlain
                            ? `⚠️ Warning: High short float (${resolvedShortFloat?.toFixed(1)}% > 12.0%) creates squeeze and volatility vulnerability.`
                            : `Capital Risk Warning: Elevated short interest (${resolvedShortFloat?.toFixed(1)}% > 12.0% threshold).`)
                        : (isPlain
                            ? "⚠️ Warning: Capital risk guard tripped. Fundamental distress, turnaround, or distribution trap detected."
                            : "Capital Risk Warning: Structural capital risk constraints violated."))
                    : capitalRiskEvidenceState === "PARTIAL"
                    ? (isPlain
                        ? "Capital risk evidence partial: Short interest or solvency rating unverified. Cannot certify capital safety."
                        : "Evidence Incomplete: Fundamental quality score or short interest input missing. Pre-Flight cannot certify capital risk.")
                    : (isPlain
                        ? "Capital risk evidence unavailable: Financial quality metrics unverified."
                        : "Capital risk evidence unavailable: Financial quality metrics unverified.")}
                </p>
              </div>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
                isCapitalRiskPassed
                  ? "bg-emerald-950 text-emerald-300 border-emerald-800"
                  : isCapitalRiskFailed
                  ? "bg-rose-950 text-rose-300 border-rose-800"
                  : "bg-amber-950 text-amber-300 border-amber-800"
              }`}>{isCapitalRiskPassed ? "PASS" : isCapitalRiskFailed ? "CAPITAL RISK" : capitalRiskEvidenceState === "PARTIAL" ? "PARTIAL" : "UNAVAILABLE"}</span>
            </div>

            {/* Check 4 */}
            <div className="p-3 rounded-lg bg-[var(--surface-2)] border border-[var(--border-subtle)] flex items-start justify-between gap-3">
              <div className="space-y-0.5">
                <div className="font-bold text-[var(--text-1)] flex items-center gap-1.5">
                  <span>
                    {resolvedCatalystState === "PASS"
                      ? "✅"
                      : resolvedCatalystState === "WARN"
                      ? "⚠️"
                      : "⚪"}
                  </span>
                  <span>
                    {isPlain
                      ? "4. News Event Safety (No surprise earnings report tomorrow)"
                      : "4. Catalyst Hazard Buffer (>7 Days to Binary Earnings)"}
                  </span>
                </div>
                <p className="text-[11px] text-[var(--text-2)] pl-5">
                  {renderCheck4Detail()}
                </p>
              </div>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
                resolvedCatalystState === "PASS"
                  ? "bg-emerald-950 text-emerald-300 border-emerald-800"
                  : resolvedCatalystState === "WARN"
                  ? "bg-rose-950 text-rose-300 border-rose-800"
                  : resolvedCatalystState === "ERROR"
                  ? "bg-rose-950/80 text-rose-300 border-rose-800"
                  : "bg-slate-900 text-[var(--text-2)] border-slate-700"
              }`}>
                {resolvedCatalystState === "PASS"
                  ? "PASS"
                  : resolvedCatalystState === "WARN"
                  ? "HAZARD"
                  : resolvedCatalystState === "ERROR"
                  ? "ERROR"
                  : "UNAVAILABLE"}
              </span>
            </div>

            {/* Check 5 */}
            <div className="p-3 rounded-lg bg-[var(--surface-2)] border border-[var(--border-subtle)] flex items-start justify-between gap-3">
              <div className="space-y-0.5">
                <div className="font-bold text-[var(--text-1)] flex items-center gap-1.5">
                  <span>{isMacroPassed ? (isVixElevated ? "⚠️" : "✅") : "⚠️"}</span>
                  <span>
                    {isPlain
                      ? (isVixElevated
                          ? "5. Overall Market Weather (VIX elevated, caution warranted)"
                          : isMacroPassed
                          ? "5. Overall Market Weather (VIX normal, market calm)"
                          : !isVixValid
                          ? "5. Overall Market Weather (VIX unavailable)"
                          : "5. Overall Market Weather (VIX high, risk blocked)")
                      : (isVixElevated
                          ? "5. Macro Regime Guard (VIX Elevated, Caution Warranted)"
                          : isMacroPassed
                          ? "5. Macro Regime Guard (VIX Volatility Guardrails Safe)"
                          : !isVixValid
                          ? "5. Macro Regime Guard (VIX Unavailable)"
                          : "5. Macro Regime Guard (VIX High, Risk Blocked)")}
                  </span>
                </div>
                <p className="text-[11px] text-[var(--text-2)] pl-5">
                  {isMacroPassed
                    ? (isVixElevated
                        ? (isPlain
                            ? `Elevated market volatility (VIX ${currentVix?.toFixed(1)}). Below 26.0 defensive cutoff, but macro caution warranted.`
                            : `Macro Guard: Elevated volatility regime (20.0 <= VIX ${currentVix?.toFixed(1)} < 26.0). Permitted within defensive ceiling, elevated caution required.`)
                        : (isPlain
                            ? `Broad market volatility is calm (VIX ${currentVix?.toFixed(1)} < 20.0). Standard position risk permitted.`
                            : `Macro Guard: Normal volatility regime (VIX ${currentVix?.toFixed(1)} < 20.0). Guardrails cleared.`))
                    : !isVixValid
                    ? (isPlain
                        ? "Market volatility reading (VIX) is currently unavailable. Macro guard uncertified."
                        : "Market Volatility: Volatility evidence unavailable from live exchange feed. Macro guard cannot be certified.")
                    : (isPlain
                        ? `⚠️ High Volatility: Market VIX (${currentVix?.toFixed(1)}) exceeds 26.0 defensive threshold. Risk deployment blocked.`
                        : `⚠️ Elevated Macro Risk: VIX (${currentVix?.toFixed(1)}) exceeds 26.0 threshold.`)}
                </p>
              </div>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
                isMacroPassed
                  ? (isVixElevated ? "bg-amber-950 text-amber-300 border-amber-800" : "bg-emerald-950 text-emerald-300 border-emerald-800")
                  : "bg-rose-950 text-rose-300 border-rose-800"
              }`}>{isMacroPassed ? "PASS" : isVixValid ? "HIGH VIX" : "UNAVAILABLE"}</span>
            </div>
          </div>
        </div>

        {/* Action Buttons — pinned to bottom */}
        <div className="flex flex-wrap items-center justify-between gap-2.5 pt-3 border-t border-[var(--border-subtle)] shrink-0">
          <button
            type="button"
            onClick={handleCopy}
            className="px-4 py-2 rounded-lg text-xs font-bold transition-all active:scale-95 border bg-cyan-600/20 hover:bg-cyan-500 hover:text-slate-950 border-cyan-500/60 text-cyan-300 flex items-center gap-1.5 shadow"
          >
            <span>{copied ? "✅ Plan Copied!" : "📋 Copy Trade Plan for Journal"}</span>
          </button>

          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-xs font-bold transition bg-slate-800 hover:bg-slate-700 text-[var(--text-1)] border border-slate-700"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
