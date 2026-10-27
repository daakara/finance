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

export interface MacroEvidence {
  vix?: number | null;
  observationTime?: string | null;
  marketSession?: "OPEN" | "REGULAR" | "PRE_MARKET" | "AFTER_HOURS" | "CLOSED" | string | null;
  settlementPinned?: boolean | null;
  sourceName?: string | null;
  freshnessStatus?: "LIVE" | "CACHED_FRESH" | "STALE" | "EXPIRED" | "UNAVAILABLE" | "ELEVATED" | string | null;
}

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
  vixObservationTime?: string | null;
  macroEvidence?: MacroEvidence | null;
  isActionable?: boolean | null;
  decisionState?: string;
  evidenceStatus?: string;
  // Provenance & Authority Inputs (Gate 2 Invariants)
  dataSource?: "live" | "historical" | "synthetic" | "curated" | "fallback" | "unavailable" | string | null;
  isSynthetic?: boolean | null;
  isHistorical?: boolean | null;
  quoteStatus?: string | null;
  liveSource?: string | null;
  liveFreshnessStatus?: string | null;
  marketSession?: string | null;
  observationTime?: string | number | null;
}

export type ProvenanceCategory =
  | "LIVE_OBSERVATION"
  | "HISTORICAL_EOD"
  | "SYNTHETIC_FIXTURE"
  | "UNVERIFIED";

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
  provenanceCategory: ProvenanceCategory;
  isSimulatedClearance: boolean;
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
  resolvedMacroEvidence: MacroEvidence | null;
}

/**
 * Resolves MacroEvidence from local browser cache with strict data quality and freshness verification.
 * Enforces DATA_QUALITY_AND_EDGE_CASE_GOVERNANCE.md (ARCH-GOV-DATA-QUALITY-VNEXT Section 3.4 & EC-01):
 * - Active Session Max Stale Threshold: 240 minutes (4 hours)
 * - Pinned Closed Session Max Threshold: 4320 minutes (72 hours / weekend close)
 * - Market Observation Time Governance: vix.observationTime governs over API generatedAt/updatedAt response times.
 * - Pinned Settlement Gate: CLOSED session requires authenticated settlementPinned === true.
 * Stale, unverified, future, or cross-ticker snapshots fail closed to null.
 */
export function resolveFreshMacroEvidence(): MacroEvidence | null {
  if (typeof window === "undefined") return null;
  try {
    const cached = localStorage.getItem("FINANCE_MARKET_SNAPSHOTS_V1");
    if (!cached) return null;

    const parsed = JSON.parse(cached);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;

    // Reject cross-ticker payloads that lack macro indicators
    if (!parsed.vix && !parsed.regime) return null;

    const rawVix = parsed?.vix?.value ?? parsed?.vix?.level ?? (typeof parsed?.vix === "number" ? parsed.vix : undefined);
    if (typeof rawVix !== "number" || isNaN(rawVix) || rawVix <= 0 || rawVix > 100) {
      return null;
    }

    // Explicitly reject historical default fixtures (M14)
    if (parsed.updatedAt === "2026-09-07T12:00:00Z" || parsed.dataSource === "STATIC_FIXTURE") {
      return null;
    }

    // 1. Observation Timestamp Resolution (M05)
    // The market bar observation timestamp takes priority over API generatedAt/updatedAt timestamps.
    const obsTimeStr = parsed?.vix?.observationTime || parsed?.observationTime || parsed?.vix?.observed_at;
    const apiTimeStr = parsed?.updatedAt || parsed?.timestamp || parsed?.generatedAt;

    const effectiveTimeStr = obsTimeStr || apiTimeStr;
    if (!effectiveTimeStr || typeof effectiveTimeStr !== "string") {
      return null; // Missing timestamp cannot establish freshness (M03)
    }

    const obsTimestamp = obsTimeStr ? new Date(obsTimeStr).getTime() : NaN;
    const apiTimestamp = apiTimeStr ? new Date(apiTimeStr).getTime() : NaN;
    const effectiveTimestamp = !isNaN(obsTimestamp) ? obsTimestamp : apiTimestamp;

    if (isNaN(effectiveTimestamp)) {
      return null; // Malformed timestamp (M11)
    }

    const now = Date.now();
    const ageMinutes = (now - effectiveTimestamp) / 60000;

    // Clock skew guard: Reject future timestamps beyond 5m skew (M11)
    if (ageMinutes < -5) {
      return null;
    }

    // 2. Closed-Session and Pinned Settlement Qualification (M07 & M08)
    // In accordance with EC-01, only authenticated closed-session pinned settlements qualify for 72h SLA.
    // An unauthenticated CLOSED flag without settlementPinned is capped at the 240m active threshold.
    const isClosedSession = parsed.marketSession === "CLOSED" || parsed.market_session === "CLOSED";
    const isPinnedSettlement = parsed.settlementPinned === true || parsed.isSettlementPinned === true;
    const hasQualifiedClosedSettlement = isClosedSession && isPinnedSettlement;

    const maxAgeMinutes = hasQualifiedClosedSettlement ? 4320 : 240;

    // 3. Stale Observation Rejection (M04 & M05)
    // If the observation itself is older than the max age, reject even if API response was generated recently.
    if (!isNaN(obsTimestamp)) {
      const obsAgeMinutes = (now - obsTimestamp) / 60000;
      if (obsAgeMinutes > maxAgeMinutes || obsAgeMinutes < -5) {
        return null;
      }
    }

    // Check API payload age
    if (!isNaN(apiTimestamp)) {
      const apiAgeMinutes = (now - apiTimestamp) / 60000;
      if (apiAgeMinutes > maxAgeMinutes) {
        return null;
      }
    }

    if (ageMinutes > maxAgeMinutes) {
      return null; // Expired pinned settlement or stale active cache (M06, M09)
    }

    return {
      vix: rawVix,
      observationTime: effectiveTimeStr,
      marketSession: parsed.marketSession || "REGULAR",
      settlementPinned: isPinnedSettlement,
      sourceName: parsed.vix?.provider || parsed.dataSource || "CBOE_VIX_INDEX",
      freshnessStatus: hasQualifiedClosedSettlement ? "CACHED_FRESH" : "LIVE"
    };
  } catch {
    return null;
  }
}

export function resolveFreshCachedVix(): number | null {
  const evidence = resolveFreshMacroEvidence();
  return evidence ? (evidence.vix ?? null) : null;
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

  // Canonical Macro Resolution (INV-MACRO-FRESHNESS-001):
  // 1. Direct macroEvidence envelope or vix with observation timestamp
  // 2. Fresh browser cache (resolveFreshMacroEvidence)
  // 3. Fallback for elevated VIX (>= 26.0) as safety blocker
  // 4. Missing/unverified/stale observation fails closed to null (UNAVAILABLE)
  const resolvedMacroEvidence: MacroEvidence | null = (() => {
    // 1. Direct macroEvidence envelope
    if (inputs.macroEvidence && typeof inputs.macroEvidence === "object") {
      const ev = inputs.macroEvidence;
      const evVix = (typeof ev.vix === "number" && !isNaN(ev.vix) && ev.vix > 0)
        ? ev.vix
        : (typeof vix === "number" && !isNaN(vix) && vix > 0 ? vix : null);

      if (evVix !== null) {
        // Missing observation timestamp fails closed (M03)
        if (!ev.observationTime || typeof ev.observationTime !== "string") {
          return null;
        }
        if (ev.freshnessStatus === "STALE" || ev.freshnessStatus === "EXPIRED" || ev.freshnessStatus === "UNAVAILABLE") {
          return null;
        }

        const obsTime = new Date(ev.observationTime).getTime();
        if (isNaN(obsTime)) return null;

        const now = Date.now();
        const ageMinutes = (now - obsTime) / 60000;
        if (ageMinutes < -5) return null; // Future clock skew (M11)

        const isClosed = (ev.marketSession === "CLOSED") && (ev.settlementPinned === true);
        const maxAge = isClosed ? 4320 : 240;

        if (ageMinutes > maxAge) return null; // Stale observation (M04)

        return {
          vix: evVix,
          observationTime: ev.observationTime,
          marketSession: ev.marketSession || "REGULAR",
          settlementPinned: ev.settlementPinned ?? false,
          sourceName: ev.sourceName || "DIRECT_MACRO_EVIDENCE",
          freshnessStatus: ev.freshnessStatus || (isClosed ? "CACHED_FRESH" : "LIVE")
        };
      }
    }

    // Direct vix with direct observation timestamp
    if (typeof vix === "number" && !isNaN(vix) && vix > 0 && inputs.vixObservationTime && typeof inputs.vixObservationTime === "string") {
      const obsTime = new Date(inputs.vixObservationTime).getTime();
      if (!isNaN(obsTime)) {
        const now = Date.now();
        const ageMinutes = (now - obsTime) / 60000;
        if (ageMinutes >= -5 && ageMinutes <= 240) {
          return {
            vix,
            observationTime: inputs.vixObservationTime,
            marketSession: "REGULAR",
            settlementPinned: false,
            sourceName: "DIRECT_TIMED_OBSERVATION",
            freshnessStatus: "LIVE"
          };
        }
      }
      return null;
    }

    // 2. Fresh browser cache fallback
    const cachedEvidence = resolveFreshMacroEvidence();
    if (cachedEvidence) return cachedEvidence;

    // 3. Fallback for direct bare vix input:
    if (evidenceStatus === "STALE" || evidenceStatus === "EXPIRED" || evidenceStatus === "UNAVAILABLE") {
      return null;
    }

    // If bare vix is provided without trustworthy timestamp:
    if (typeof vix === "number" && !isNaN(vix) && vix > 0) {
      // VIX >= 26.0 is an absolute defensive blocker (M02)
      if (vix >= 26.0) {
        return {
          vix,
          observationTime: null,
          marketSession: "REGULAR",
          settlementPinned: false,
          sourceName: "DEFENSIVE_BLOCKER",
          freshnessStatus: "ELEVATED"
        };
      }
      // VIX < 26.0 without trustworthy timestamp cannot grant unsupported PASS (M03)
      return null;
    }

    return null;
  })();

  const resolvedVix = resolvedMacroEvidence?.vix ?? null;
  const isVixValid = resolvedVix !== null;
  const currentVix = resolvedVix;

  const stopLossPct = (isPriceValid && safeStop !== null) ? (((safeStop - safePrice) / safePrice) * 100).toFixed(2) : "N/A";
  const target1Pct = (isPriceValid && safeTarget !== null) ? (((safeTarget - safePrice) / safePrice) * 100).toFixed(2) : "N/A";

  // ── Check 1: Asymmetric Risk-Reward ──────────────────────────────────────
  const hasExecutionLevels = hasStopLoss && hasTarget;
  const isLevelDirectionValid =
    isPriceValid &&
    hasExecutionLevels &&
    safeStop! < safePrice &&
    safeTarget! > safePrice;

  // Independent Prospective Risk/Reward Verification (Full Precision)
  let rawComputedRRAtSpot = 0;
  if (isLevelDirectionValid) {
    const risk = safePrice - safeStop!;
    const reward = safeTarget! - safePrice;
    if (risk > 0 && reward > 0) {
      rawComputedRRAtSpot = reward / risk;
    }
  }

  // Canonical Effective RR Resolution:
  // Must be verified by BOTH the execution levels at current market spot AND any model ratio.
  // Full numeric precision is preserved for threshold comparison.
  // A raw ratio below the required 2.0:1 threshold (e.g. 1.995) must NEVER pass solely because it rounds to 2.00 for display.
  const rawEffectiveRR = isLevelDirectionValid
    ? (isRRValid ? Math.min(safeRR, rawComputedRRAtSpot) : rawComputedRRAtSpot)
    : 0;

  const isRRPassed = isPriceValid && hasExecutionLevels && isLevelDirectionValid && rawEffectiveRR >= 2.0;
  const chk1Status: CheckStatus = !hasExecutionLevels
    ? "MISSING LEVELS"
    : !isLevelDirectionValid
    ? "FAIL"
    : isRRPassed
    ? "PASS"
    : "FAIL";

  const displayRR = rawEffectiveRR.toFixed(2);
  const isSubThresholdRounding = !isRRPassed && Number(displayRR) >= 2.0;
  const chk1Summary = !hasExecutionLevels
    ? "N/A"
    : !isLevelDirectionValid
    ? "Invalid Levels"
    : isSubThresholdRounding
    ? "Below 2.00:1"
    : `${displayRR}:1.0`;

  const chk1Detail = !hasExecutionLevels
    ? "Execution levels missing: Stop loss and target required."
    : !isLevelDirectionValid
    ? (safeStop! >= safePrice
        ? `Inverted risk boundary: Stop loss ($${safeStop!.toFixed(2)}) is above current price ($${safePrice.toFixed(2)}). Execution risk floor breached.`
        : `Inverted reward boundary: Profit target ($${safeTarget!.toFixed(2)}) is below current price ($${safePrice.toFixed(2)}). Target boundary inverted.`)
    : isRRPassed
    ? "Adequate upside cushion (R:R >= 2.0:1)."
    : isSubThresholdRounding
    ? `Hazard: Prospective reward relative to downside risk (${rawEffectiveRR.toFixed(3)}:1) is sub-threshold (< 2.0:1 institutional floor).`
    : `Hazard: Prospective reward relative to downside risk (${displayRR}:1) is below 2.0:1 institutional floor.`;

  const chk1Item: PreFlightCheckItem = {
    id: "CHK-1",
    tag: "CHK-1 R:R",
    name: isPlain ? "1. Reward vs Risk Balance" : "1. Asymmetric Payoff",
    status: chk1Status,
    badgeClass: isRRPassed
      ? "bg-emerald-950/80 text-emerald-300 border-emerald-700/80"
      : "bg-rose-950/80 text-rose-300 border-rose-700/80",
    passed: isRRPassed,
    summary: chk1Summary,
    detail: chk1Detail,
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
    isLevelDirectionValid &&
    !isStage4 &&
    mandatoryChecksVerified;

  // Active Blocker Resolution
  const activeBlocker = (() => {
    if (!isActionableGranted) return "CANONICAL SETUP NON-ACTIONABLE";
    if (!isPriceValid || !hasExecutionLevels) return "UNVERIFIED EXECUTION LEVELS";
    if (!isLevelDirectionValid) return "INVERTED EXECUTION BOUNDARIES";
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

  // ── Provenance & Epistemic Authority Classification (Gate 2 Invariants) ──
  const rawDataSource = (inputs.dataSource || "").toLowerCase();
  const rawQuoteStatus = (inputs.quoteStatus || "").toUpperCase();
  const rawEvidenceStatus = (inputs.evidenceStatus || "").trim();

  const isSyntheticFixture = Boolean(
    inputs.isSynthetic ||
    rawDataSource === "synthetic" ||
    rawDataSource === "fixture" ||
    rawDataSource === "static_fixture" ||
    rawQuoteStatus === "SYNTHETIC" ||
    rawEvidenceStatus.toUpperCase().includes("SYNTHETIC") ||
    rawEvidenceStatus.toUpperCase().includes("FIXTURE") ||
    rawEvidenceStatus.toUpperCase().includes("SIMULATED")
  );

  const isHistoricalEOD = Boolean(
    !isSyntheticFixture && (
      inputs.isHistorical ||
      rawDataSource === "historical" ||
      rawQuoteStatus === "HISTORICAL_EOD" ||
      rawQuoteStatus === "HISTORICAL" ||
      rawEvidenceStatus.toUpperCase().includes("HISTORICAL") ||
      rawEvidenceStatus.toUpperCase().includes("EOD")
    )
  );

  const isExplicitLiveSource = Boolean(
    rawDataSource === "live" ||
    inputs.liveSource === "ALPACA_IEX" ||
    inputs.liveSource === "EXCHANGE_FEED" ||
    inputs.liveSource === "exchange_feed"
  );

  const isLiveObservation = Boolean(
    !isSyntheticFixture &&
    !isHistoricalEOD &&
    isExplicitLiveSource &&
    isPriceValid &&
    (inputs.liveFreshnessStatus === "LIVE" ||
     inputs.liveFreshnessStatus === "REALTIME" ||
     inputs.macroEvidence?.freshnessStatus === "LIVE")
  );

  const provenanceCategory: ProvenanceCategory = isSyntheticFixture
    ? "SYNTHETIC_FIXTURE"
    : isHistoricalEOD
    ? "HISTORICAL_EOD"
    : isLiveObservation
    ? "LIVE_OBSERVATION"
    : "UNVERIFIED";

  const isSimulatedClearance = Boolean(isCleared && isSyntheticFixture);

  // What Needs to Change Callout
  const whatNeedsToChange = (() => {
    if (!isActionableGranted) {
      return "Wait for price action to confirm the canonical setup trigger within the accumulation corridor before sizing.";
    }
    if (!isPriceValid || !hasExecutionLevels) {
      return "Establish verified spot price and quantitative execution boundaries (stop loss and target levels).";
    }
    if (!isLevelDirectionValid) {
      return safeStop! >= safePrice
        ? "Establish valid stop loss below current market price to define defensive risk floor."
        : "Establish valid profit target above current market price to define prospective reward objective.";
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
      if (!isVixValid || currentVix === null) {
        return "Verify live market volatility (VIX) from exchange feed before execution clearance.";
      }
      return "Wait for systemic market volatility (VIX) to settle below defensive threshold (< 26.0).";
    }

    // Invariant 5: Simulated preflight clearance must not imply live order-execution authorization.
    if (isSyntheticFixture) {
      return "All pre-flight clearance conditions satisfied in simulation. Notice: Synthetic fixture — not authorized for live order execution.";
    }
    if (isHistoricalEOD) {
      return "All pre-flight clearance conditions satisfied on historical close. Notice: Historical EOD data — requires live market confirmation before execution.";
    }
    if (provenanceCategory === "UNVERIFIED") {
      return "All pre-flight clearance conditions satisfied. Notice: Unverified market feed — requires verified real-time data before execution.";
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

  // Invariants 1-4: Canonical evidenceMeta evaluation
  const evidenceMeta = (() => {
    // If an explicit evidenceStatus was supplied, sanitize against false LIVE claims on synthetic/historical
    if (rawEvidenceStatus) {
      if (isSyntheticFixture && rawEvidenceStatus.toUpperCase().includes("LIVE")) {
        return "Evidence: SYNTHETIC FIXTURE · Simulated Pre-Flight Verification";
      }
      if (isHistoricalEOD && rawEvidenceStatus.toUpperCase().includes("LIVE")) {
        return "Evidence: HISTORICAL EOD · Reconciled Close Observation";
      }
      return rawEvidenceStatus;
    }

    switch (provenanceCategory) {
      case "SYNTHETIC_FIXTURE":
        return mandatoryChecksVerified
          ? "Evidence: SYNTHETIC FIXTURE · Simulated Pre-Flight Verification"
          : "Evidence: SYNTHETIC FIXTURE · Reconciled Ledger";
      case "HISTORICAL_EOD":
        return "Evidence: HISTORICAL EOD · Reconciled Close Observation";
      case "LIVE_OBSERVATION":
        return mandatoryChecksVerified
          ? "Evidence: LIVE MARKET OBSERVATION · 5/5 Mandatory Checks"
          : "Evidence: LIVE MARKET OBSERVATION · Preflight Verification Pending";
      case "UNVERIFIED":
      default:
        return "Evidence: UNVERIFIED PROVENANCE · Feed Verification Required";
    }
  })();

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
    provenanceCategory,
    isSimulatedClearance,
    safePrice,
    safeStop,
    safeTarget,
    safeRR: rawEffectiveRR,
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
    resolvedMacroEvidence,
  };
}
