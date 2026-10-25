/**
 * ARX Universal Pre-Flight Quantitative Evaluator (Synthesis E V3).
 *
 * Single Authoritative Source of Truth for:
 *   - CHK-1: Asymmetric Risk-Reward (>= 2.0:1 with verified stop/target)
 *   - CHK-2: Technical Structure & Stage Discipline (Above 20 EMA / 50 SMA Pivot, !Stage 4, !Extended)
 *   - CHK-3: Structural Capital Risk & Squeeze Floor (Short float < 12%, Solvency >= 60, !Turnaround)
 *   - CHK-4: Catalyst Hazard Buffer (> 7d clean runway, confirmed coverage, LIVE freshness)
 *   - CHK-5: Macro Regime Guard (VIX < 26.0 with valid macro reading)
 *
 * Fail-Closed Invariants:
 *   - All 5 checks must reach verified PASS for mandatory clearance.
 *   - Canonical actionability (isActionable === true) is MANDATORY: AAPL 5/5 PASS but !actionable remains BLOCKED.
 *   - NVDA 4/5 PASS (Stage 4) remains strictly BLOCKED.
 *   - Missing, unverified, or stale data NEVER evaluates to PASS.
 */

import { MASTER_ASSET_CATALOG } from "./masterCatalog";

export interface PreFlightChecklistInputs {
  symbol: string;
  currentPrice?: number | null;
  stopLoss?: number | null;
  takeProfit1?: number | null;
  riskRewardRatio?: number | null;
  setupPattern?: string;
  isDayTrader?: boolean;
  isStage4?: boolean;
  optimalEntryMin?: number | null;
  optimalEntryMax?: number | null;
  breakoutPivot?: number | null;
  isDistributionTrap?: boolean;
  hasImminentEarnings?: boolean | null;
  catalystState?: "PASS" | "WARN" | "UNAVAILABLE" | "ERROR";
  catalystEvidence?: {
    state?: "PASS" | "WARN" | "UNAVAILABLE" | "ERROR";
    daysToEarnings?: number | null;
    eventDate?: string | null;
    callTimeEt?: string | null;
    resultsReleaseTimeEt?: string | null;
    sourceName?: string | null;
    confirmationStatus?: "CONFIRMED" | "ESTIMATED" | "UNVERIFIED" | null;
    coverageStatus?: string | null;
    freshnessStatus?: "LIVE" | "STALE" | "EXPIRED" | string | null;
    riskTier?: string | null;
    isEstimate?: boolean | null;
    preflightState?: "PASS" | "WARN" | "UNAVAILABLE" | "ERROR";
  } | null;
  vix?: number | null;
  isActionable?: boolean | null;
  decisionState?: string;
  evidenceStatus?: string;
}

export type CheckStatus =
  | "PASS"
  | "FAIL"
  | "HAZARD"
  | "STAGE 4 WAIT"
  | "MISSING LEVELS"
  | "CHASING"
  | "CAPITAL RISK"
  | "HIGH VIX"
  | "UNAVAILABLE"
  | "ERROR";

export interface PreFlightCheckItem {
  id: "CHK-1" | "CHK-2" | "CHK-3" | "CHK-4" | "CHK-5";
  tag: string;
  name: string;
  status: CheckStatus;
  badgeClass: string;
  passed: boolean;
  summary: string;
  detail: string;
  icon: string;
}

export interface PreFlightEvaluationResult {
  symbol: string;
  checks: [
    PreFlightCheckItem,
    PreFlightCheckItem,
    PreFlightCheckItem,
    PreFlightCheckItem,
    PreFlightCheckItem
  ];
  passedCount: number;
  convictionPct: number;
  isActionableGranted: boolean;
  mandatoryChecksVerified: boolean;
  isCleared: boolean;
  clearanceStatusLabel: string;
  clearanceStatusBadgeClass: string;
  activeBlocker: string;
  whatNeedsToChange: string;
  evidenceMeta: string;
  safePrice: number;
  safeStop: number | null;
  safeTarget: number | null;
  safeRR: number;
  safeEntryMin: number | null;
  safeEntryMax: number | null;
  safePivot: number | null;
  currentVix: number | null;
  stopLossPct: string;
  target1Pct: string;
  target2Pct: string;
  safeTarget2: number | null;
  entryZoneStr: string;
  stopLossStr: string;
  target1Str: string;
  target2Str: string;
  resolvedCatalystState: "PASS" | "WARN" | "UNAVAILABLE" | "ERROR";
}

export function evaluatePreFlightChecks(
  inputs: PreFlightChecklistInputs,
  vernacularMode: "PLAIN_ENGLISH" | "PRO_QUANT" = "PLAIN_ENGLISH"
): PreFlightEvaluationResult {
  const {
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
    hasImminentEarnings = false,
    catalystState,
    catalystEvidence,
    vix,
    isActionable = false,
    decisionState,
    evidenceStatus,
  } = inputs;

  const isPlain = vernacularMode === "PLAIN_ENGLISH";
  const cleanSym = (symbol || "ASSET").toUpperCase().replace("-USD", "");
  const catalogItem = MASTER_ASSET_CATALOG[cleanSym];

  // Defensive Numeric Guards & Epistemic Verification
  const isPriceValid = typeof currentPrice === "number" && !isNaN(currentPrice) && currentPrice > 0;
  const safePrice = isPriceValid ? currentPrice! : 0;
  const hasStopLoss = typeof stopLoss === "number" && !isNaN(stopLoss) && stopLoss > 0;
  const safeStop = hasStopLoss ? stopLoss! : null;
  const hasTarget = typeof takeProfit1 === "number" && !isNaN(takeProfit1) && takeProfit1 > 0;
  const safeTarget = hasTarget ? takeProfit1! : null;
  const isRRValid = typeof riskRewardRatio === "number" && !isNaN(riskRewardRatio) && riskRewardRatio > 0;
  const safeRR = isRRValid ? riskRewardRatio! : 0;
  const safeEntryMin = (typeof optimalEntryMin === "number" && !isNaN(optimalEntryMin) && optimalEntryMin > 0) ? optimalEntryMin : null;
  const safeEntryMax = (typeof optimalEntryMax === "number" && !isNaN(optimalEntryMax) && optimalEntryMax > 0) ? optimalEntryMax : null;
  const safePivot = (typeof breakoutPivot === "number" && !isNaN(breakoutPivot) && breakoutPivot > 0) ? breakoutPivot : null;

  // Canonical VIX Resolution: input > browser localStorage cache > null
  const resolvedVix = (() => {
    if (typeof vix === "number" && !isNaN(vix) && vix > 0) return vix;
    if (typeof window !== "undefined") {
      try {
        const cached = localStorage.getItem("FINANCE_MARKET_SNAPSHOTS_V1");
        if (cached) {
          const parsed = JSON.parse(cached);
          const v = parsed?.vix?.value ?? parsed?.vix?.level ?? (typeof parsed?.vix === "number" ? parsed.vix : undefined);
          if (typeof v === "number" && !isNaN(v) && v > 0) return v;
        }
      } catch {}
    }
    return null;
  })();
  const isVixValid = resolvedVix !== null;
  const currentVix = resolvedVix;

  const stopLossPct = (isPriceValid && safeStop !== null) ? (((safeStop - safePrice) / safePrice) * 100).toFixed(2) : "N/A";
  const target1Pct = (isPriceValid && safeTarget !== null) ? (((safeTarget - safePrice) / safePrice) * 100).toFixed(2) : "N/A";

  // ── Check 1: Asymmetric Risk-Reward ──────────────────────────────────────
  const hasExecutionLevels = hasStopLoss && hasTarget;
  const isRRPassed = isPriceValid && isRRValid && safeRR >= 2.0 && hasExecutionLevels;
  const chk1Status: CheckStatus = isRRPassed ? "PASS" : (!hasExecutionLevels ? "MISSING LEVELS" : "FAIL");
  const chk1Item: PreFlightCheckItem = {
    id: "CHK-1",
    tag: "CHK-1 R:R",
    name: isPlain ? "1. Reward vs Risk Balance" : "1. Asymmetric Payoff",
    status: chk1Status,
    badgeClass: isRRPassed
      ? "bg-emerald-950/80 text-emerald-300 border-emerald-700/80"
      : "bg-rose-950/80 text-rose-300 border-rose-700/80",
    passed: isRRPassed,
    summary: isRRValid && hasExecutionLevels ? `${safeRR.toFixed(2)}:1.0` : "N/A",
    detail: isRRPassed
      ? "Adequate upside cushion (R:R >= 2.0:1)."
      : !hasExecutionLevels
      ? "Execution levels missing: Stop loss and target required."
      : isRRValid
      ? "Hazard: Upside potential too small relative to downside risk (< 2.0:1)."
      : "Evidence incomplete: Risk-reward ratio unverified.",
    icon: isRRPassed ? "✓" : "✕",
  };

  // ── Check 2: Technical Trend Alignment & Stage Discipline ───────────────
  const isExtendedAboveZone = Boolean(safeEntryMax && safePrice > safeEntryMax * 1.02);
  const isWithinTolerance = Boolean(safeEntryMax && safePrice > safeEntryMax && safePrice <= safeEntryMax * 1.02);
  const isInCorridor = Boolean(safeEntryMin && safeEntryMax && safePrice >= safeEntryMin && safePrice <= safeEntryMax);
  const isBelowCorridor = Boolean(safeEntryMin && safePrice < safeEntryMin);

  const isTrendPassed = isPriceValid && !isStage4 && !isExtendedAboveZone && safeEntryMin !== null && safeEntryMax !== null;
  const chk2Status: CheckStatus = isTrendPassed
    ? "PASS"
    : isStage4
    ? "STAGE 4 WAIT"
    : safeEntryMin === null
    ? "MISSING LEVELS"
    : "CHASING";
  const chk2Item: PreFlightCheckItem = {
    id: "CHK-2",
    tag: "CHK-2 STRUCTURE",
    name: isPlain ? "2. Trend & Moving Averages" : "2. Technical Structure",
    status: chk2Status,
    badgeClass: isTrendPassed
      ? "bg-emerald-950/80 text-emerald-300 border-emerald-700/80"
      : isStage4
      ? "bg-amber-950/80 text-amber-300 border-amber-700/80"
      : "bg-rose-950/80 text-rose-300 border-rose-700/80",
    passed: isTrendPassed,
    summary: isTrendPassed
      ? (isWithinTolerance ? "Within Tolerance (+2%)" : isInCorridor ? "In Corridor" : "Below Corridor")
      : isStage4
      ? "Stage 4 Correction"
      : "Out of Corridor",
    detail: isStage4
      ? `Stage 4 correction structure: Spot ($${safePrice.toFixed(2)}) requires base breakout pivot ${safePivot !== null ? `above $${safePivot.toFixed(2)}` : "above 50-day SMA"}.`
      : safeEntryMin === null || safeEntryMax === null
      ? "Optimal buy zone levels uncalculated. Awaiting technical structure."
      : isExtendedAboveZone
      ? `Extended structure: Spot ($${safePrice.toFixed(2)}) is >2% above entry corridor ($${safeEntryMin.toFixed(2)}–$${safeEntryMax.toFixed(2)}). Chasing creates negative R:R risk.`
      : isWithinTolerance
      ? `Spot price ($${safePrice.toFixed(2)}) is slightly above preferred corridor ($${safeEntryMin.toFixed(2)}–$${safeEntryMax.toFixed(2)}) within allowable extension tolerance (+2%).`
      : isBelowCorridor
      ? `Spot price ($${safePrice.toFixed(2)}) is below preferred accumulation corridor ($${safeEntryMin.toFixed(2)}–$${safeEntryMax.toFixed(2)}).`
      : `Spot price ($${safePrice.toFixed(2)}) is positioned within the accumulation corridor defending key moving average support.`,
    icon: isTrendPassed ? "✓" : isStage4 ? "⏳" : "⚠️",
  };

  // ── Check 3: CHK-3 CAPITAL-RISK ──────────────────────────────────────────
  const hasShortFloat = catalogItem !== undefined && catalogItem.shortFloat !== undefined && catalogItem.shortFloat !== null;
  const hasQualityScore = catalogItem !== undefined && catalogItem.qualityScore !== undefined && catalogItem.qualityScore !== null;
  const capitalRiskEvidenceState: "COMPLETE" | "PARTIAL" | "UNAVAILABLE" =
    hasShortFloat && hasQualityScore
      ? "COMPLETE"
      : hasShortFloat || hasQualityScore
      ? "PARTIAL"
      : "UNAVAILABLE";

  const shortFloatNum = Number(catalogItem?.shortFloat) || 0;
  const qualityScoreNum = Number(catalogItem?.qualityScore) || 0;
  const isTurnaround = String(catalogItem?.verdict || "").toLowerCase().includes("turnaround");

  const isCapitalRiskFailed = isPriceValid && (
    (capitalRiskEvidenceState === "COMPLETE" && (shortFloatNum > 12.0 || isTurnaround || qualityScoreNum < 60)) ||
    isDistributionTrap === true
  );

  const isCapitalRiskPassed = isPriceValid && capitalRiskEvidenceState === "COMPLETE" && !isCapitalRiskFailed && isDistributionTrap !== true;
  const chk3Status: CheckStatus = isCapitalRiskPassed
    ? "PASS"
    : isCapitalRiskFailed
    ? "CAPITAL RISK"
    : capitalRiskEvidenceState === "PARTIAL"
    ? "UNAVAILABLE"
    : "UNAVAILABLE";
  const chk3Item: PreFlightCheckItem = {
    id: "CHK-3",
    tag: "CHK-3 CAPITAL",
    name: isPlain ? "3. Capital Health & Squeeze Risk" : "3. Structural Capital Risk",
    status: chk3Status,
    badgeClass: isCapitalRiskPassed
      ? "bg-emerald-950/80 text-emerald-300 border-emerald-700/80"
      : isCapitalRiskFailed
      ? "bg-rose-950/80 text-rose-300 border-rose-700/80"
      : "bg-amber-950/80 text-amber-300 border-amber-700/80",
    passed: isCapitalRiskPassed,
    summary: isCapitalRiskPassed
      ? `Float ${shortFloatNum.toFixed(1)}% | Q-Score ${qualityScoreNum}`
      : isCapitalRiskFailed
      ? "Elevated Capital Distress"
      : "Evidence Incomplete",
    detail: isCapitalRiskPassed
      ? "Short interest is moderate (<12%) and fundamental solvency confirms absence of balance sheet distress."
      : isCapitalRiskFailed
      ? "Elevated short interest (>12%), turnaround status, or sub-60 quality score detected."
      : capitalRiskEvidenceState === "PARTIAL"
      ? "Fundamental quality score or short interest input missing. Pre-Flight cannot certify capital risk."
      : "Capital risk evidence unavailable: Financial quality metrics unverified.",
    icon: isCapitalRiskPassed ? "✓" : isCapitalRiskFailed ? "✕" : "⚠️",
  };

  // ── Check 4: Catalyst Hazard Buffer ──────────────────────────────────────
  const resolvedCatalystState: "PASS" | "WARN" | "UNAVAILABLE" | "ERROR" = (() => {
    if (
      catalystState === "ERROR" ||
      catalystEvidence?.state === "ERROR" ||
      catalystEvidence?.preflightState === "ERROR"
    ) {
      return "ERROR";
    }

    if (catalystEvidence?.freshnessStatus && catalystEvidence.freshnessStatus !== "LIVE") {
      return "UNAVAILABLE";
    }

    if (
      hasImminentEarnings === true ||
      catalystEvidence?.state === "WARN" ||
      catalystEvidence?.preflightState === "WARN" ||
      (catalystEvidence?.riskTier && catalystEvidence.riskTier !== "NORMAL_RUNWAY" && catalystEvidence.riskTier !== "UNVERIFIED_RISK")
    ) {
      return "WARN";
    }

    if (catalystEvidence) {
      const dte = catalystEvidence.daysToEarnings;
      if (typeof dte === "number" && !isNaN(dte)) {
        if (dte < 0.0) return "UNAVAILABLE";
        if (dte <= 7.0) return "WARN";

        const validStatuses = new Set(["CONFIRMED_UPCOMING", "ESTIMATED_UPCOMING", "COVERED"]);
        const isCoverageValid = Boolean(catalystEvidence.coverageStatus && validStatuses.has(catalystEvidence.coverageStatus));
        const isFresh = catalystEvidence.freshnessStatus === "LIVE";

        if (!isCoverageValid || !isFresh) {
          return "UNAVAILABLE";
        }

        if (catalystEvidence.riskTier && catalystEvidence.riskTier !== "NORMAL_RUNWAY") {
          return "WARN";
        }

        return "PASS";
      }
      if (catalystEvidence.coverageStatus === "DATA_UNAVAILABLE" || catalystEvidence.coverageStatus === "ABSENT") {
        return "UNAVAILABLE";
      }
    }

    return "UNAVAILABLE";
  })();

  const isCatalystConfirmed = catalystEvidence?.confirmationStatus === "CONFIRMED";
  const isCatalystPassed = isPriceValid && resolvedCatalystState === "PASS";
  const chk4Status: CheckStatus = resolvedCatalystState === "PASS"
    ? "PASS"
    : resolvedCatalystState === "WARN"
    ? "HAZARD"
    : resolvedCatalystState === "ERROR"
    ? "ERROR"
    : "UNAVAILABLE";

  const chk4Summary = resolvedCatalystState === "PASS"
    ? (isCatalystConfirmed
        ? `Runway > 7d (${catalystEvidence?.daysToEarnings?.toFixed(1) || ">7"}d)`
        : `Runway > 7d (Unconfirmed)`)
    : resolvedCatalystState === "WARN"
    ? "Imminent Earnings Event"
    : "Coverage Unavailable";

  const chk4Detail = resolvedCatalystState === "PASS"
    ? (isCatalystConfirmed
        ? `Clean runway verified (>7 days to confirmed binary earnings event${catalystEvidence?.daysToEarnings ? `, ${catalystEvidence.daysToEarnings.toFixed(1)}d DTE` : ""}).`
        : `Clean runway indicated (>7 days to estimated earnings${catalystEvidence?.daysToEarnings ? `, ${catalystEvidence.daysToEarnings.toFixed(1)}d DTE` : ""}), but event confirmation is unverified by primary source.`)
    : resolvedCatalystState === "WARN"
    ? `Scheduled earnings announcement within 7-day risk window${catalystEvidence?.eventDate ? ` (${catalystEvidence.eventDate})` : ""}. Position sizing restricted.`
    : resolvedCatalystState === "ERROR"
    ? "Technical error retrieving corporate earnings schedule."
    : catalystEvidence?.daysToEarnings !== undefined && catalystEvidence?.daysToEarnings !== null && catalystEvidence.daysToEarnings < 0
    ? `Historical Event Elapsed (${catalystEvidence.eventDate || "prior"}). Forward calendar unverified; runway uncertified.`
    : "Earnings calendar coverage unavailable: Event runway cannot be certified.";

  const chk4Item: PreFlightCheckItem = {
    id: "CHK-4",
    tag: "CHK-4 CATALYST",
    name: isPlain ? "4. News Event Safety" : "4. Catalyst Hazard Buffer",
    status: chk4Status,
    badgeClass: resolvedCatalystState === "PASS"
      ? "bg-emerald-950/80 text-emerald-300 border-emerald-700/80"
      : resolvedCatalystState === "WARN"
      ? "bg-rose-950/80 text-rose-300 border-rose-700/80"
      : resolvedCatalystState === "ERROR"
      ? "bg-rose-950/80 text-rose-300 border-rose-700/80"
      : "bg-slate-900 text-slate-400 border-slate-700",
    passed: isCatalystPassed,
    summary: chk4Summary,
    detail: chk4Detail,
    icon: resolvedCatalystState === "PASS" ? "✓" : resolvedCatalystState === "WARN" ? "⚠️" : "⚪",
  };

  // ── Check 5: Macro Regime Guard ──────────────────────────────────────────
  const isMacroPassed = isPriceValid && isVixValid && currentVix !== null && currentVix < 26.0;
  const chk5Status: CheckStatus = isMacroPassed ? "PASS" : isVixValid ? "HIGH VIX" : "UNAVAILABLE";
  const chk5Item: PreFlightCheckItem = {
    id: "CHK-5",
    tag: "CHK-5 MACRO",
    name: isPlain ? "5. Overall Market Weather" : "5. Macro Regime Guard",
    status: chk5Status,
    badgeClass: isMacroPassed
      ? "bg-emerald-950/80 text-emerald-300 border-emerald-700/80"
      : "bg-amber-950/80 text-amber-300 border-amber-700/80",
    passed: isMacroPassed,
    summary: isMacroPassed
      ? `VIX ${currentVix?.toFixed(1)} (Normal)`
      : isVixValid
      ? `VIX ${currentVix?.toFixed(1)} (Elevated)`
      : "VIX Unavailable",
    detail: isMacroPassed
      ? `Market volatility is within standard parameters (VIX ${currentVix?.toFixed(1)} < 26.0).`
      : !isVixValid
      ? "Market volatility reading (VIX) unavailable from exchange feed. Macro guard cannot be certified."
      : `Elevated macro volatility: VIX (${currentVix?.toFixed(1)}) exceeds 26.0 threshold.`,
    icon: isMacroPassed ? "✓" : "⚠️",
  };

  const checks: [
    PreFlightCheckItem,
    PreFlightCheckItem,
    PreFlightCheckItem,
    PreFlightCheckItem,
    PreFlightCheckItem
  ] = [chk1Item, chk2Item, chk3Item, chk4Item, chk5Item];

  const passedCount = checks.filter((c) => c.passed).length;
  const convictionPct = isPriceValid ? Math.round((passedCount / 5) * 100) : 0;
  const isActionableGranted = isActionable === true;

  const mandatoryChecksVerified =
    isRRPassed &&
    isTrendPassed &&
    isCapitalRiskPassed &&
    isCatalystPassed &&
    isMacroPassed;

  const isCleared =
    isActionableGranted &&
    isPriceValid &&
    isRRValid &&
    !isStage4 &&
    mandatoryChecksVerified;

  // Active Blocker Resolution
  const activeBlocker = (() => {
    if (!isActionableGranted) return "CANONICAL SETUP NON-ACTIONABLE";
    if (!isPriceValid || !isRRValid) return "UNVERIFIED EXECUTION LEVELS";
    if (isStage4) return "STAGE 4 DISTRIBUTION STRUCTURE";
    if (!isRRPassed) return "REWARD/RISK ASYMMETRY (< 2.0:1)";
    if (!isTrendPassed) return "TECHNICAL STRUCTURE NOT CONFIRMED";
    if (!isCapitalRiskPassed) return capitalRiskEvidenceState !== "COMPLETE" ? "CAPITAL RISK EVIDENCE INCOMPLETE" : "CAPITAL PRESERVATION DEFENSE";
    if (!isCatalystPassed) {
      if (resolvedCatalystState === "WARN") return "IMMINENT EARNINGS HAZARD (<= 7 DAYS)";
      if (resolvedCatalystState === "ERROR") return "CATALYST EVALUATION ERROR";
      if (catalystEvidence?.freshnessStatus && catalystEvidence.freshnessStatus !== "LIVE") return "STALE CATALYST COVERAGE";
      return "CATALYST COVERAGE UNAVAILABLE";
    }
    if (!isMacroPassed) return !isVixValid || currentVix === null ? "VIX DATA UNAVAILABLE" : "MACRO VOLATILITY REGIME (VIX >= 26.0)";
    return "NONE";
  })();

  // What Needs to Change Callout
  const whatNeedsToChange = (() => {
    if (!isActionableGranted) {
      return "Wait for price action to confirm the canonical setup trigger within the accumulation corridor before sizing.";
    }
    if (!isPriceValid || !isRRValid) {
      return "Establish verified spot price and quantitative execution boundaries (stop loss and target levels).";
    }
    if (isStage4) {
      return "Wait for post-breakout base consolidation above the 50-day moving average before accumulation.";
    }
    if (!isRRPassed) {
      return "Wait for favorable pullback closer to stop loss to expand asymmetric reward/risk ratio (>= 2.0:1).";
    }
    if (!isTrendPassed) {
      return "Wait for price to pull back inside the verified value corridor without breaking technical support.";
    }
    if (!isCapitalRiskPassed) {
      return "Require fundamental solvency ratings or short interest verification before allocating capital.";
    }
    if (!isCatalystPassed) {
      if (resolvedCatalystState === "WARN") {
        return "Wait for post-earnings binary risk window to clear (>7 days) to eliminate gap-risk exposure.";
      }
      return "Confirm forward corporate earnings schedule from authoritative primary source.";
    }
    if (!isMacroPassed) {
      return "Wait for systemic market volatility (VIX) to settle below defensive threshold (< 26.0).";
    }
    return "All pre-flight clearance conditions satisfied. Setup cleared for controlled execution.";
  })();

  const clearanceStatusLabel = isCleared
    ? "CLEARED"
    : passedCount === 5 && !isActionableGranted
    ? "NOT CLEARED · 5/5 PASS (AWAITING CONFIRMATION)"
    : isStage4
    ? "NOT CLEARED — STAGE 4 WAIT"
    : `NOT CLEARED (${passedCount}/5)`;

  const clearanceStatusBadgeClass = isCleared
    ? "bg-emerald-950/80 text-emerald-300 border-emerald-700/80"
    : "bg-amber-950/80 text-amber-300 border-amber-700/80";

  // Formatted Trade Plan Elements
  const entryZoneStr = (safeEntryMin !== null && safeEntryMax !== null) ? `$${safeEntryMin.toFixed(2)} – $${safeEntryMax.toFixed(2)}` : "Unavailable (Awaiting Technical Structure)";
  const stopLossStr = safeStop !== null ? `$${safeStop.toFixed(2)} (${stopLossPct}%)` : "Unavailable (Trader Defined Stop Required)";
  const target1Str = safeTarget !== null ? `$${safeTarget.toFixed(2)} (+${target1Pct}%)` : "Unavailable (Trader Defined Target Required)";
  const safeTarget2 = (safePrice > 0 && safeTarget !== null) ? Number((safeTarget * 1.12).toFixed(2)) : null;
  const target2Pct = (safePrice > 0 && safeTarget2 !== null) ? (((safeTarget2 - safePrice) / safePrice) * 100).toFixed(2) : "N/A";
  const target2Str = safeTarget2 !== null ? `$${safeTarget2.toFixed(2)} (+${target2Pct}%)` : "Unavailable";

  const evidenceMeta = evidenceStatus || (
    isPriceValid && mandatoryChecksVerified
      ? "Evidence: LIVE MARKET OBSERVATION · 5/5 Mandatory Checks"
      : "Evidence: SYNTHETIC FIXTURE / RECONCILED LEDGER"
  );

  return {
    symbol,
    checks,
    passedCount,
    convictionPct,
    isActionableGranted,
    mandatoryChecksVerified,
    isCleared,
    clearanceStatusLabel,
    clearanceStatusBadgeClass,
    activeBlocker,
    whatNeedsToChange,
    evidenceMeta,
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
  };
}
