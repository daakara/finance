"use client";

import { useState, useEffect, useRef } from "react";
import { MASTER_ASSET_CATALOG } from "../lib/masterCatalog";
import { trackPreFlightOutcome, trackTradePlanCopied } from "../lib/matomo";
import { evaluatePreFlightChecks, PreFlightChecklistInputs } from "../lib/preflightEvaluator";

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
  isActionable = false,
  decisionState,
}: PreFlightChecklistModalProps) {
  const [copied, setCopied] = useState<boolean>(false);
  const [vernacularMode, setVernacularMode] = useState<"PLAIN_ENGLISH" | "PRO_QUANT">("PLAIN_ENGLISH");

  const modalRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLElement | null>(null);

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

  if (!isOpen) return null;

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
      isActionable,
      decisionState,
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

  const cleanSym = (symbol || "ASSET").toUpperCase().replace("-USD", "");
  const catalogItem = MASTER_ASSET_CATALOG[cleanSym];
  const isPriceValid = typeof currentPrice === "number" && !isNaN(currentPrice) && currentPrice > 0;
  const isRRValid = typeof riskRewardRatio === "number" && !isNaN(riskRewardRatio) && riskRewardRatio > 0;
  const hasExecutionLevels = safeStop !== null && safeTarget !== null;
  const isRRPassed = checks[0].passed;
  const isTrendPassed = checks[1].passed && !isStage4;
  const isExtendedAboveZone = Boolean(safeEntryMax && safePrice > safeEntryMax * 1.02);
  const isWithinTolerance = Boolean(safeEntryMax && safePrice > safeEntryMax && safePrice <= safeEntryMax * 1.02);
  const isInCorridor = Boolean(safeEntryMin && safeEntryMax && safePrice >= safeEntryMin && safePrice <= safeEntryMax);
  const isBelowCorridor = Boolean(safeEntryMin && safePrice < safeEntryMin);
  const isCapitalRiskPassed = checks[2].passed;
  const isCapitalRiskFailed = checks[2].status === "CAPITAL RISK";
  const hasShortFloat = catalogItem !== undefined && catalogItem.shortFloat !== undefined && catalogItem.shortFloat !== null;
  const hasQualityScore = catalogItem !== undefined && catalogItem.qualityScore !== undefined && catalogItem.qualityScore !== null;
  const capitalRiskEvidenceState: "COMPLETE" | "PARTIAL" | "UNAVAILABLE" =
    hasShortFloat && hasQualityScore
      ? "COMPLETE"
      : hasShortFloat || hasQualityScore
      ? "PARTIAL"
      : "UNAVAILABLE";
  const isCatalystPassed = checks[3].passed;
  const isMacroPassed = checks[4].passed;

  const isVixValid = typeof vix === "number" && !isNaN(vix) && vix > 0;

  const renderCheck2Detail = () => {
    if (isStage4) {
      return isPlain
        ? `⚠️ Watchlist Only: Spot price ($${safePrice.toFixed(2)}) is in Stage 4 correction below 50-day average. Await base formation.`
        : `Stage 4 correction structure: Spot ($${safePrice.toFixed(2)}) requires 50-day breakout pivot ${safePivot !== null ? `above $${safePivot.toFixed(2)}` : "(pivot uncalculated)"}.`;
    }
    if (safeEntryMin === null || safeEntryMax === null) {
      return "Optimal buy zone levels uncalculated. Awaiting technical structure.";
    }
    if (isExtendedAboveZone) {
      return isPlain
        ? `⚠️ Extended: Price is above ideal buy zone ($${safeEntryMin.toFixed(2)} - $${safeEntryMax.toFixed(2)}). Wait for pullback.`
        : "Extended structure: Spot is above value area. Chasing creates negative R:R risk.";
    }
    if (isWithinTolerance) {
      return isPlain
        ? `Spot price ($${safePrice.toFixed(2)}) is slightly above preferred buying range ($${safeEntryMin.toFixed(2)} - $${safeEntryMax.toFixed(2)}), within allowable extension tolerance (+2%). Defending key support.`
        : `Spot price ($${safePrice.toFixed(2)}) is above corridor ceiling ($${safeEntryMax.toFixed(2)}) within +2% extension tolerance. Defending key moving average support (20 EMA / 50 SMA).`;
    }
    if (isBelowCorridor) {
      return isPlain
        ? `Spot price ($${safePrice.toFixed(2)}) is below preferred buying range ($${safeEntryMin.toFixed(2)} - $${safeEntryMax.toFixed(2)}).`
        : `Spot price ($${safePrice.toFixed(2)}) is below preferred accumulation corridor ($${safeEntryMin.toFixed(2)}–$${safeEntryMax.toFixed(2)}).`;
    }
    return isPlain
      ? `Spot price ($${safePrice.toFixed(2)}) is inside the verified buying range defending key support.`
      : "Defending key moving average support (20 EMA / 50 SMA).";
  };

  const renderCheck4Detail = () => {
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

  // Analytics — inline try/catch, no useEffect needed (stable values within a single open session)
  try {
    trackPreFlightOutcome(symbol, passedCount, isCleared);
  } catch (err) {
    console.warn("Analytics error in PreFlightChecklistModal:", err);
  }

  // Trade plan markdown — differentiated by vernacularMode
  const tradePlanMarkdown = isPlain
    ? `### 📋 ARX Terminal Trade Plan: ${symbol}
- **Date**: ${new Date().toISOString().split("T")[0]}
- **Asset**: ${symbol} | **Mode**: ${isDayTrader ? "⚡ Day Trader" : "🏛️ Swing / Long-Term Compounder"}
- **Current Price**: $${safePrice.toFixed(2)}
- **Buy Zone**: ${entryZoneStr}
- **Stop Loss**: ${stopLossStr}
- **Profit Goal 1 (TP1)**: ${target1Str}
- **Profit Goal 2 (TP2 Runner)**: ${target2Str}
- **Tactical Rule (Protected Trailing Transition)**: ${safeTarget !== null ? `When Profit Goal 1 ($${safeTarget.toFixed(2)}) is hit, sell 50% of position and immediately move Stop Loss on remaining 50% to purchase price ($${safePrice.toFixed(2)}) to lock in a protected trade.` : "Establish verified target levels before executing tactical scaling rules."}
- **Setup**: ${setupPattern || "Minervini VCP Pattern"}
- **Pre-Flight Score**: ${convictionPct}% Checklist (${isCleared ? "🟢 CLEARED" : `⚠️ NOT CLEARED — ${activeBlocker}`})
`
    : `### 📋 ARX Institutional Execution Brief: ${symbol}
- **Date**: ${new Date().toISOString().split("T")[0]}
- **Asset**: ${symbol} | **Mode**: ${isDayTrader ? "Intraday Momentum Pullback" : "Swing / Compounder Accumulation"}
- **Spot**: $${safePrice.toFixed(2)} | **Optimal Entry**: ${entryZoneStr}
- **Hard Stop / Invalidation**: ${stopLossStr}
- **Target 1 (TP1)**: ${target1Str}
- **Target 2 (TP2 Runner)**: ${target2Str}
- **Execution Rule (Protected Trailing Ratchet)**: ${safeTarget !== null ? `Scale 0.50x tranche @ TP1 ($${safeTarget.toFixed(2)}). Ratchet trailing stop to cost basis ($${safePrice.toFixed(2)}) to ensure protected runner convexity.` : "Establish verified target levels before executing tactical scaling rules."}
- **Risk/Reward**: ${isRRValid ? `${safeRR.toFixed(2)} : 1.0` : "Unverified"}
- **Setup Pattern**: ${setupPattern || "Minervini VCP"}
- **Pre-Flight Clearance**: ${convictionPct}% Checklist — ${isCleared ? "🟢 CLEARED FOR EXECUTION" : `⚠️ NOT CLEARED (${activeBlocker})`}
- **VIX Regime**: ${currentVix !== null ? currentVix.toFixed(1) : "UNAVAILABLE"} | **Capital Risk**: ${isCapitalRiskPassed ? "PASS" : isCapitalRiskFailed ? "DETECTED" : capitalRiskEvidenceState} | **Earnings Hazard**: ${resolvedCatalystState === "PASS" ? "CLEAR" : resolvedCatalystState === "WARN" ? "HAZARD" : resolvedCatalystState}
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

  return (
    <div
      className="fixed inset-0 z-[1200] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-150"
      aria-modal="true"
      role="dialog"
      aria-labelledby="preflight-title"
    >
      <div
        ref={modalRef}
        className="bg-[#0b1019] border border-cyan-800/80 rounded-2xl max-w-xl w-full p-5 sm:p-6 shadow-2xl font-sans text-slate-200 relative flex flex-col max-h-[90vh]"
      >
        {/* Header — pinned */}
        <div className="flex items-center justify-between border-b border-[#1e293b] pb-3.5 shrink-0">
          <div className="flex items-center gap-2.5">
            <span className="text-xl">✈️</span>
            <div>
              <h2 id="preflight-title" className="text-base sm:text-lg font-bold text-white tracking-tight flex items-center gap-2">
                <span>{isPlain ? `Pre-Flight Trade Checklist: ${symbol}` : `Institutional Pre-Flight Clearance: ${symbol}`}</span>
              </h2>
              <p className="text-xs text-slate-400">
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
            className="text-slate-400 hover:text-white p-1.5 rounded-lg hover:bg-slate-800 transition"
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
                <p className="text-[11px] leading-relaxed text-slate-300 font-sans">
                  {isPlain
                    ? `A valid setup is forming for ${symbol}, but the entry trigger has not confirmed yet. This checklist reviews setup and risk conditions; it does not bypass the required confirmation trigger.`
                    : `Valid setup structure is present (${decisionState || "VALID_SETUP"}), but execution criteria remain incomplete. Current state: WAIT FOR TRIGGER. Sizing and execution remain locked until confirmation conditions are satisfied.`}
                </p>
                <details className="pt-1 text-[10px] text-slate-400 font-mono">
                  <summary className="cursor-pointer hover:text-slate-200 select-none">
                    Technical Provenance
                  </summary>
                  <div className="mt-1 pl-2 border-l border-amber-800/40 text-[10px] space-y-0.5 text-slate-400">
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
              <span>Live spot price or execution levels are unverified for this asset. Trade execution is blocked until verified market data is received.</span>
            </div>
          )}

          {/* Clearance Conviction Barometer */}
          <div className={`p-3.5 rounded-xl border flex items-center justify-between ${
            isCleared
              ? "bg-emerald-950/40 border-emerald-700/80 text-emerald-300"
              : "bg-amber-950/40 border-amber-700/80 text-amber-300"
          }`}>
            <div>
              <span className="text-[10px] uppercase font-bold tracking-wider font-mono block">
                {isPlain ? "Trade Readiness Score" : "Quantitative Clearance Status"}
              </span>
              <div className="text-base sm:text-lg font-extrabold flex items-center gap-1.5">
                <span>{isCleared ? "🟢 CLEARED TO EXECUTE" : `⚠️ NOT CLEARED — ${activeBlocker}`}</span>
                <span className="text-xs font-mono font-normal">({convictionPct}% Checklist)</span>
              </div>
            </div>
            <div className="text-right font-mono">
              <span className="text-2xl sm:text-3xl font-black">{passedCount}/5</span>
              <span className="text-[10px] text-slate-400 block">Checks Passed</span>
            </div>
          </div>

          {/* 5-Point Validation Checklist */}
          <div className="space-y-2.5 text-xs">
            {/* Check 1 */}
            <div className="p-3 rounded-lg bg-[#111722] border border-[#1e293b] flex items-start justify-between gap-3">
              <div className="space-y-0.5">
                <div className="font-bold text-slate-100 flex items-center gap-1.5">
                  <span>{isRRPassed ? "✅" : "❌"}</span>
                  <span>{isPlain ? "1. Reward vs Risk Balance (At least 2 to 1)" : "1. Asymmetric Payoff (Reward:Risk >= 2.0:1)"}</span>
                </div>
                <p className="text-[11px] text-slate-400 pl-5">
                  Current: <strong className="text-cyan-300 font-mono">{isRRValid && hasExecutionLevels ? `${safeRR.toFixed(2)} : 1.0` : "N/A (Missing Execution Levels)"}</strong>{" "}
                  {isRRPassed ? "(Adequate upside cushion)" : !hasExecutionLevels ? "(Execution levels missing: Stop and target required)" : isRRValid ? "(Hazard: upside too small for downside risk)" : "(Evidence incomplete: risk-reward unavailable)"}
                </p>
              </div>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
                isRRPassed ? "bg-emerald-950 text-emerald-300 border-emerald-800" : "bg-rose-950 text-rose-300 border-rose-800"
              }`}>{isRRPassed ? "PASS" : (!hasExecutionLevels ? "MISSING LEVELS" : "FAIL")}</span>
            </div>

            {/* Check 2 */}
            <div className="p-3 rounded-lg bg-[#111722] border border-[#1e293b] flex items-start justify-between gap-3">
              <div className="space-y-0.5">
                <div className="font-bold text-slate-100 flex items-center gap-1.5">
                  <span>{isTrendPassed ? "✅" : (isStage4 ? "⏳" : "⚠️")}</span>
                  <span>{isPlain ? "2. Trend & Moving Averages (Price in upward corridor)" : "2. Technical Structure (Above 20 EMA / 50 SMA Pivot)"}</span>
                </div>
                <p className="text-[11px] text-slate-400 pl-5">
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
            <div className="p-3 rounded-lg bg-[#111722] border border-[#1e293b] flex items-start justify-between gap-3">
              <div className="space-y-0.5">
                <div className="font-bold text-slate-100 flex items-center gap-1.5">
                  <span>{isCapitalRiskPassed ? "✅" : (isCapitalRiskFailed ? "❌" : "⚠️")}</span>
                  <span>{isPlain ? "3. Capital Health & Squeeze Risk (Manageable short float & solvent balance sheet)" : "3. Structural Capital Risk (Short Interest Floor & Quality Rating)"}</span>
                </div>
                <p className="text-[11px] text-slate-400 pl-5">
                  {isCapitalRiskPassed
                    ? (isPlain
                        ? "Short interest is moderate (<12%) and financial quality metrics indicate stable balance sheet health."
                        : "Capital Risk Guard: Short float (<12%) and fundamental solvency score confirm absence of acute balance sheet distress.")
                    : isCapitalRiskFailed
                    ? (isPlain
                        ? "⚠️ Warning: Elevated short interest (>12%) or turnaround rating indicates potential capital distress."
                        : "⚠️ Capital Risk Warning: Elevated short interest (>12%), turnaround status, or sub-60 quality score detected.")
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
            <div className="p-3 rounded-lg bg-[#111722] border border-[#1e293b] flex items-start justify-between gap-3">
              <div className="space-y-0.5">
                <div className="font-bold text-slate-100 flex items-center gap-1.5">
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
                <p className="text-[11px] text-slate-400 pl-5">
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
                  : "bg-slate-900 text-slate-400 border-slate-700"
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
            <div className="p-3 rounded-lg bg-[#111722] border border-[#1e293b] flex items-start justify-between gap-3">
              <div className="space-y-0.5">
                <div className="font-bold text-slate-100 flex items-center gap-1.5">
                  <span>{isMacroPassed ? "✅" : "⚠️"}</span>
                  <span>{isPlain ? "5. Overall Market Weather (VIX normal, market calm)" : "5. Macro Regime Guard (VIX Volatility Guardrails Safe)"}</span>
                </div>
                <p className="text-[11px] text-slate-400 pl-5">
                  {isMacroPassed
                    ? (isPlain
                        ? `Market volatility is within standard parameters (VIX ${currentVix?.toFixed(1)}).`
                        : `Macro Guard: Normal volatility regime (VIX ${currentVix?.toFixed(1)} < 26.0).`)
                    : !isVixValid
                    ? (isPlain
                        ? "Market volatility reading (VIX) is currently unavailable. Macro guard uncertified."
                        : "Market Volatility: Volatility evidence unavailable from live exchange feed. Macro guard cannot be certified.")
                    : (isPlain
                        ? `⚠️ High Volatility: Market VIX (${currentVix?.toFixed(1)}) indicates elevated systemic turbulence.`
                        : `⚠️ Elevated Macro Risk: VIX (${currentVix?.toFixed(1)}) exceeds 26.0 threshold.`)}
                </p>
              </div>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
                isMacroPassed ? "bg-emerald-950 text-emerald-300 border-emerald-800" : "bg-amber-950 text-amber-300 border-amber-800"
              }`}>{isMacroPassed ? "PASS" : isVixValid ? "HIGH VIX" : "UNAVAILABLE"}</span>
            </div>
          </div>
        </div>

        {/* Action Buttons — pinned to bottom */}
        <div className="flex flex-wrap items-center justify-between gap-2.5 pt-3 border-t border-[#1e293b] shrink-0">
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
            className="px-4 py-2 rounded-lg text-xs font-bold transition bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
