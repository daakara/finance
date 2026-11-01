/**
 * ARX Pure Assessment & State Engine (docs/ux/07_data_to_ui_contract.md & 08_state_transition_model.md)
 * 
 * INVARIANT: This file is completely pure.
 * Zero dependencies on React, router, localStorage, network calls, or UI rendering.
 * Deterministic mapping: Inputs -> TerminalViewState.
 */

import {
  TimeHorizon,
  Assessment,
  DecisionPosture,
  DecisionState,
  OwnershipState,
  OwnershipSource,
  DomainAssessment,
  FactorAgreement,
  TerminalViewState,
  ModelProvenance,
  ARXAction,
  DecisionTrace,
  FreshnessInfo,
} from "../types/insight";
import { evaluateLevelRelation } from "./reclaimSemantics";

export interface AssessmentEngineInput {
  symbol: string;
  companyName: string;
  currentPrice: number;
  changePct: number;
  horizon: TimeHorizon;
  ownershipState: OwnershipState;
  ownershipSource: OwnershipSource;
  domains: DomainAssessment[];
  isInvalidationBreached?: boolean;
  invalidationPrice?: number;
  reclaimMilestonePrice?: number;
  freshnessStatus?: string;
  freshness?: FreshnessInfo;
  modelProvenance?: ModelProvenance;
  decisionTrace?: DecisionTrace;
  canonicalDecision?: any;
}

/**
 * Pure calculation of factor agreement metrics
 */
export function calculateFactorAgreement(domains: DomainAssessment[]): FactorAgreement {
  let favorable = 0;
  let mixed = 0;
  let unfavorable = 0;
  let unavailable = 0;

  for (const d of domains) {
    if (d.availability === "UNAVAILABLE" || d.status === "UNAVAILABLE") {
      unavailable++;
    } else if (d.status === "FAVORABLE") {
      favorable++;
    } else if (d.status === "MIXED") {
      mixed++;
    } else if (d.status === "UNFAVORABLE") {
      unfavorable++;
    }
  }

  const evaluated = favorable + mixed + unfavorable;

  let displayLabel = `${favorable} of ${evaluated} evaluated factors are favorable`;
  if (evaluated === 0) {
    displayLabel = "No factors currently evaluated (Data Unavailable)";
  } else if (favorable === evaluated && evaluated > 0) {
    displayLabel = `All ${evaluated} evaluated factors are favorable`;
  } else if (favorable === 0 && evaluated > 0) {
    displayLabel = `0 of ${evaluated} evaluated factors are favorable`;
  } else if (favorable === unfavorable && favorable > 0) {
    displayLabel = `Evidence is evenly split across ${evaluated} evaluated factors`;
  }

  return {
    favorable,
    mixed,
    unfavorable,
    unavailable,
    evaluated,
    displayLabel,
  };
}

/**
 * Pure deterministic state derivation
 */
export function deriveAssessmentState(input: AssessmentEngineInput): TerminalViewState {
  const {
    symbol,
    companyName,
    currentPrice,
    changePct,
    horizon,
    ownershipState,
    ownershipSource,
    domains,
    isInvalidationBreached = false,
    invalidationPrice,
    reclaimMilestonePrice,
    freshnessStatus,
    freshness,
    modelProvenance = {
      modelId: "arx-confluence-engine",
      modelVersion: "2.4.0",
      rulesetVersion: "2026.09-v1",
      calculatedAt: new Date().toISOString(),
    },
    decisionTrace,
  } = input;

  const agreement = calculateFactorAgreement(domains);

  // 1. Determine Overall Data Eligibility and Core Evidence Completeness
  // Directive ARX-UX-EVIDENCE-001: Distinguish mandatory core evidence (blocking) from optional enrichment (non-blocking)
  // Required core domains: 'trend' (Technical Trend), 'health' (Company Health / Fund Profile), 'macro' (Macro Regime)
  // Optional enrichment domains: 'smart_money', 'options_flow', 'insider_trades', 'congress_trades'
  const isDomainRequired = (d: DomainAssessment): boolean => {
    if ((d as any).isOptional === true) return false;
    if ((d as any).applicability === "OPTIONAL" || (d as any).applicability === "INAPPLICABLE") return false;
    if (
      d.domainId === "smart_money" ||
      d.domainId === "options_flow" ||
      d.domainId === "insider_trades" ||
      d.domainId === "congress_trades"
    ) {
      return false;
    }
    return true;
  };

  const requiredDomains = domains.filter(isDomainRequired);
  const unavailableRequired = requiredDomains.filter(
    (d) => d.availability === "UNAVAILABLE" || d.status === "UNAVAILABLE"
  );

  const trendDomain = domains.find((d) => d.domainId === "trend");
  const healthDomain = domains.find((d) => d.domainId === "health");

  const isTrendAvailable = trendDomain !== undefined && trendDomain.availability === "AVAILABLE" && trendDomain.status !== "UNAVAILABLE";
  const isHealthAvailable = healthDomain !== undefined && healthDomain.availability === "AVAILABLE" && healthDomain.status !== "UNAVAILABLE";

  let overallEligibility: "ELIGIBLE" | "LIMITED" | "INELIGIBLE" = "ELIGIBLE";
  if (agreement.evaluated === 0) {
    overallEligibility = "INELIGIBLE";
  } else if (unavailableRequired.length > 0) {
    overallEligibility = "LIMITED";
  }

  const isCoreEvidenceAvailable = isTrendAvailable && isHealthAvailable && overallEligibility === "ELIGIBLE";

  // 2. Derive Overall Evidence Assessment (Requires full core evidence for FAVORABLE)
  let assessment: Assessment = "MIXED";
  if (overallEligibility === "INELIGIBLE" || !isCoreEvidenceAvailable) {
    assessment = "INSUFFICIENT_EVIDENCE";
  } else if ((agreement.favorable >= 3 || (agreement.favorable >= 2 && agreement.evaluated <= 2)) && agreement.unfavorable === 0) {
    assessment = "FAVORABLE";
  } else if (agreement.unfavorable >= 2) {
    assessment = "UNFAVORABLE";
  } else {
    assessment = "MIXED";
  }

  // 3. Setup Invalidation Level
  const isPriceValid = typeof currentPrice === "number" && !isNaN(currentPrice) && currentPrice > 0;
  const safePrice = isPriceValid ? currentPrice : 0;
  const hasInvalidationPrice = typeof invalidationPrice === "number" && !isNaN(invalidationPrice) && invalidationPrice > 0;
  const stopLevel = hasInvalidationPrice ? invalidationPrice : 0;
  const distancePct = (isPriceValid && stopLevel > 0)
    ? Number((((stopLevel - safePrice) / safePrice) * 100).toFixed(1))
    : 0;
  const isActuallyBreached = isInvalidationBreached || (invalidationPrice !== undefined && safePrice < invalidationPrice);

  // 4. Resolve Contextual Posture (Precedence: Invalidation Breached -> Missing Evidence -> Assessment + Ownership)
  let posture: DecisionPosture = "RESEARCH";
  let uiStateLabel = "Evidence Incomplete — In-Depth Research Required";
  let headlineExplanation = "Incomplete evidence available to assess setup.";

  const effectiveFreshnessStatus = freshness?.status || freshnessStatus;
  const stalenessDays = freshness?.stalenessDays;
  const isStale =
    effectiveFreshnessStatus === "STALE_HISTORICAL" ||
    effectiveFreshnessStatus === "STALE" ||
    (stalenessDays !== undefined && stalenessDays > 4);

  if (isActuallyBreached) {
    if (ownershipState === "OWNED") {
      posture = "EXIT_REVIEW";
      uiStateLabel = "Thesis Needs Review";
      headlineExplanation = `Price breached the setup invalidation floor at $${stopLevel.toFixed(2)} (${distancePct}%).`;
    } else {
      posture = "AVOID";
      uiStateLabel = "Setup Invalidated";
      headlineExplanation = `Price has fallen below the setup invalidation floor at $${stopLevel.toFixed(2)}. Risk parameters breached.`;
    }
  } else if (!isPriceValid || agreement.evaluated === 0 || effectiveFreshnessStatus === "UNAVAILABLE") {
    posture = "RESEARCH";
    uiStateLabel = "Unverified Asset — Direct Feed Only";
    headlineExplanation = "Asset evidence unverified or market data feed unavailable.";
  } else if (isStale) {
    posture = "RESEARCH";
    uiStateLabel = "Stale Historical Tape (> 4 Days)";
    headlineExplanation = "Market data is historical/stale; live trade triggers are suspended.";
  } else if (!isCoreEvidenceAvailable || overallEligibility !== "ELIGIBLE" || !isTrendAvailable) {
    posture = "RESEARCH";
    uiStateLabel = "Evidence Incomplete — In-Depth Research Required";
    headlineExplanation = !isTrendAvailable
      ? "Technical trend evidence is unavailable (insufficient historical sessions). Active triggers cannot be confirmed."
      : !isHealthAvailable
      ? "Core fundamental financial evidence is unverified. In-depth due diligence required before evaluating setups."
      : "One or more required core evidence domains are unavailable or unverified. In-depth due diligence required before evaluating setups.";
  } else if (ownershipState === "OWNED") {
    if (assessment === "UNFAVORABLE") {
      posture = "TRIM";
      uiStateLabel = "Consider Trimming";
      headlineExplanation = "Fundamental or technical deterioration indicates increased downside risk.";
    } else if (assessment === "FAVORABLE") {
      posture = "HOLD";
      uiStateLabel = "Thesis Intact";
      headlineExplanation = "Multi-factor confluence supports maintaining current position size.";
    } else {
      posture = "HOLD";
      uiStateLabel = "Thesis Neutral";
      headlineExplanation = "Mixed factors present; monitor support levels closely.";
    }
  } else {
    // NOT_OWNED or UNKNOWN (Full core evidence verified)
    // Canonical Decision Authority Enforcement (Phase 2):
    // ACQUIRE is permitted strictly when canonical decision authoritatively approves ACTIONABLE_SETUP.
    // Zero client-side re-derivation of actionable posture from factor counts alone.
    const isCanonicalActionable = Boolean(
      decisionTrace
        ? (decisionTrace.isActionable && decisionTrace.canSizeTrade && decisionTrace.decisionState === "ACTIONABLE_SETUP")
        : false
    );

    const canonicalStateLabel = decisionTrace?.decisionStateLabel || decisionTrace?.stateLabel;
    const isLabelSafeForWatch = canonicalStateLabel && !canonicalStateLabel.toLowerCase().includes("actionable");

    if (isCanonicalActionable) {
      posture = "ACQUIRE";
      uiStateLabel = canonicalStateLabel || "Actionable Setup";
      headlineExplanation = "Multi-factor confluence confirmed in optimal buy zone.";
    } else if (assessment === "UNFAVORABLE") {
      posture = "AVOID";
      uiStateLabel = "Unfavorable Setup";
      headlineExplanation = "Negative trend or poor fundamentals present unfavorable risk/reward.";
    } else {
      posture = "WATCH";
      uiStateLabel = isLabelSafeForWatch
        ? canonicalStateLabel
        : (canonicalStateLabel?.toLowerCase().includes("actionable") ? "Valid Setup — Awaiting Trigger" : (canonicalStateLabel || "Setup Evaluation Pending"));
      const levelRel = evaluateLevelRelation(safePrice, reclaimMilestonePrice, "50-day SMA", symbol);
      headlineExplanation = (decisionTrace && decisionTrace.disqualificationReason)
        ? decisionTrace.disqualificationReason
        : levelRel.status === "UNAVAILABLE"
        ? "Awaiting constructive consolidation and volume confirmation before trigger."
        : levelRel.headlineExplanationWatch;
    }
  }

  // 5. Fail-Closed Execution & Sizing Gating
  // Sizing of a new trade is authorized ONLY for unowned assets in an ACQUIRE posture with full evidence clearance.
  const isEvidenceInsufficient =
    agreement.evaluated === 0 ||
    overallEligibility !== "ELIGIBLE" ||
    !isCoreEvidenceAvailable ||
    !isHealthAvailable ||
    !isTrendAvailable;

  const isExecutionBlocked =
    !isPriceValid ||
    effectiveFreshnessStatus === "UNAVAILABLE" ||
    isStale ||
    isActuallyBreached ||
    isEvidenceInsufficient ||
    posture !== "ACQUIRE" ||
    ownershipState === "OWNED";

  const canSizeTrade = isExecutionBlocked
    ? false
    : Boolean(decisionTrace ? decisionTrace.canSizeTrade : (posture === "ACQUIRE"));

  const isActionable = isExecutionBlocked
    ? false
    : Boolean(decisionTrace ? decisionTrace.isActionable : (posture === "ACQUIRE"));

  // 4a. Resolve Precedence-Enforced DecisionState (Phase 20A / Phase 21 / Phase 2)
  // Preserving genuine upstream model decision state separately from runtime safety restrictions
  const deriveFallbackDecisionState = (): DecisionState => {
    if (!isPriceValid) {
      return "UNVERIFIED";
    }
    if (isStale) {
      return "STALE_DATA";
    }
    if (isActuallyBreached || posture === "AVOID" || posture === "EXIT_REVIEW") {
      return "EVIDENCE_INCOMPLETE";
    }
    if (agreement.evaluated === 0 || effectiveFreshnessStatus === "UNAVAILABLE") {
      return "UNVERIFIED";
    }
    if (!isTrendAvailable) {
      return "INSUFFICIENT_DATA";
    }
    if (!isHealthAvailable || overallEligibility !== "ELIGIBLE" || !isCoreEvidenceAvailable) {
      return "EVIDENCE_INCOMPLETE";
    }
    if (posture === "ACQUIRE") {
      return "ACTIONABLE_SETUP";
    }
    return "VALID_SETUP";
  };

  const authoritativeDecisionState: DecisionState =
    decisionTrace?.decisionState || deriveFallbackDecisionState();

  let decisionState: DecisionState = "VALID_SETUP";
  if (!isPriceValid) {
    decisionState = "UNVERIFIED";
  } else if (isStale) {
    decisionState = "STALE_DATA";
  } else if (isActuallyBreached || posture === "AVOID" || posture === "EXIT_REVIEW") {
    decisionState = "EVIDENCE_INCOMPLETE";
  } else if (agreement.evaluated === 0 || effectiveFreshnessStatus === "UNAVAILABLE") {
    decisionState = "UNVERIFIED";
  } else if (!isTrendAvailable) {
    decisionState = "INSUFFICIENT_DATA";
  } else if (!isHealthAvailable || overallEligibility !== "ELIGIBLE" || !isCoreEvidenceAvailable) {
    decisionState = "EVIDENCE_INCOMPLETE";
  } else if (posture === "ACQUIRE" && canSizeTrade && isActionable) {
    decisionState = "ACTIONABLE_SETUP";
  } else if (decisionTrace && decisionTrace.decisionState !== "ACTIONABLE_SETUP") {
    decisionState = decisionTrace.decisionState;
    const canonicalStateLabel = decisionTrace.decisionStateLabel || decisionTrace.stateLabel;
    if (canonicalStateLabel && ownershipState !== "OWNED" && !canonicalStateLabel.toLowerCase().includes("actionable")) {
      uiStateLabel = canonicalStateLabel;
    }
    if (decisionTrace.disqualificationReason && posture !== "ACQUIRE") {
      headlineExplanation = decisionTrace.disqualificationReason;
    }
  } else {
    decisionState = "VALID_SETUP";
  }

  // Sanitize uiStateLabel: Never allow an "Actionable" claim if execution is blocked
  const canonicalTraceLabel = decisionTrace?.decisionStateLabel || decisionTrace?.stateLabel;
  if (!canSizeTrade && uiStateLabel.toLowerCase().includes("actionable")) {
    if (posture === "WATCH") {
      uiStateLabel = (canonicalTraceLabel && !canonicalTraceLabel.toLowerCase().includes("actionable"))
        ? canonicalTraceLabel
        : "Valid Setup — Awaiting Trigger";
    } else if (posture === "RESEARCH") {
      uiStateLabel = "Evidence Incomplete — In-Depth Research Required";
    } else if (posture === "HOLD") {
      uiStateLabel = "Thesis Intact";
    } else if (posture === "TRIM") {
      uiStateLabel = "Consider Trimming";
    } else if (posture === "EXIT_REVIEW") {
      uiStateLabel = "Thesis Needs Review";
    } else if (posture === "AVOID") {
      uiStateLabel = "Setup Invalidated";
    }
  }

  // 6. Derive Contextual Actions
  const levelRel = evaluateLevelRelation(safePrice, reclaimMilestonePrice, "50-day SMA", symbol);
  const alertActionLabel = reclaimMilestonePrice !== undefined
    ? (levelRel.status === "BELOW"
        ? `Set Alert for Reclaim ($${reclaimMilestonePrice.toFixed(2)})`
        : `Set Alert for $${reclaimMilestonePrice.toFixed(2)}`)
    : "Set Price Alert";

  const canSize = canSizeTrade;
  const availableActions: ARXAction[] = [
    {
      id: "set_alert",
      type: "SET_ALERT",
      label: alertActionLabel,
      enabled: overallEligibility !== "INELIGIBLE",
    },
    {
      id: "size_trade",
      type: "SIZE_POSITION",
      label: "Calculate Position Size",
      enabled: canSize,
      reason: !canSize
        ? (decisionTrace?.disqualificationReason || (posture !== "ACQUIRE" ? uiStateLabel : "Only actionable when in Buy Zone"))
        : undefined,
    },
    {
      id: "review_invalidation",
      type: "REVIEW_THESIS",
      label: "Review Invalidation Criteria",
      enabled: ownershipState === "OWNED",
      reason: ownershipState !== "OWNED" ? "Applies to held positions" : undefined,
    },
    {
      id: "compare_peers",
      type: "COMPARE",
      label: "Compare Peers",
      enabled: true,
    },
  ];

  const terminalResult = {
    symbol: symbol.toUpperCase(),
    companyName,
    currentPrice: safePrice,
    changePct,
    horizon,
    ownership: {
      state: ownershipState,
      source: ownershipSource,
    },
    modelProvenance,
    overallEligibility,
    decisionState,
    authoritativeDecisionState,
    freshness: freshness || (effectiveFreshnessStatus ? { status: effectiveFreshnessStatus, stalenessDays } : undefined),
    freshnessStatus: effectiveFreshnessStatus,
    stalenessDays,
    canSizeTrade,
    isActionable,
    assessment,
    factorAgreement: agreement,
    domains,
    posture,
    uiStateLabel,
    headlineExplanation,
    setupInvalidationLevel: {
      price: stopLevel,
      distancePct,
      description: `Close below $${stopLevel.toFixed(2)} (${distancePct}%) invalidates technical breakout thesis.`,
    },
    whatWouldChangeAssessment: posture === "ACQUIRE"
      ? `A daily close below $${stopLevel.toFixed(2)} (${distancePct}%) would invalidate the setup and downgrade posture to AVOID.`
      : levelRel.status !== "UNAVAILABLE"
      ? levelRel.whatWouldChangeAssessment
      : "Sufficient historical trading data and constructive base formation required to evaluate potential upgrade.",
    primaryAction: {
      label: posture === "ACQUIRE"
        ? "Calculate Position Size"
        : posture === "EXIT_REVIEW"
        ? "Review Exit Criteria"
        : posture === "RESEARCH"
        ? "Conduct Fundamental Research"
        : alertActionLabel,
      actionType: (posture === "ACQUIRE"
        ? "SIZE_TRADE"
        : posture === "EXIT_REVIEW"
        ? "REVIEW_THESIS"
        : posture === "RESEARCH"
        ? "RESEARCH_PROFILE"
        : "SET_ALERT") as "SIZE_TRADE" | "REVIEW_THESIS" | "RESEARCH_PROFILE" | "SET_ALERT",
      enabled: overallEligibility !== "INELIGIBLE",
    },
    availableActions,
  };

  assertDecisionAuthorityConsistency({
    posture: terminalResult.posture,
    decisionState: terminalResult.decisionState,
    authoritativeDecisionState: terminalResult.authoritativeDecisionState,
    canSizeTrade: terminalResult.canSizeTrade,
    isActionable: terminalResult.isActionable,
    uiStateLabel: terminalResult.uiStateLabel,
    availableActions: terminalResult.availableActions,
    ownershipState,
  });

  return terminalResult;
}

export interface DecisionAuthorityCheck {
  isValid: boolean;
  violations: string[];
}

export function validateDecisionAuthorityConsistency(state: {
  posture: DecisionPosture;
  decisionState?: DecisionState;
  authoritativeDecisionState?: DecisionState;
  canSizeTrade?: boolean;
  isActionable?: boolean;
  uiStateLabel: string;
  finalVerdict?: string;
  availableActions?: ARXAction[];
  ownershipState?: OwnershipState;
}): DecisionAuthorityCheck {
  const violations: string[] = [];

  // Invariant 1: canSizeTrade can ONLY be true when posture is ACQUIRE
  if (state.canSizeTrade && state.posture !== "ACQUIRE") {
    violations.push(`canSizeTrade is true while posture is ${state.posture} (must be ACQUIRE)`);
  }

  // Invariant 2: canSizeTrade can NEVER be true on an owned asset
  if (state.canSizeTrade && state.ownershipState === "OWNED") {
    violations.push(`canSizeTrade is true on an OWNED position`);
  }

  // Invariant 3: isActionable can ONLY be true when canSizeTrade is true
  if (state.isActionable && !state.canSizeTrade) {
    violations.push(`isActionable is true while canSizeTrade is false`);
  }

  // Invariant 4: decisionState cannot be ACTIONABLE_SETUP unless canSizeTrade is true
  if (state.decisionState === "ACTIONABLE_SETUP" && !state.canSizeTrade) {
    violations.push(`decisionState is ACTIONABLE_SETUP while canSizeTrade is false`);
  }

  // Invariant 5: finalVerdict cannot be ACTIONABLE_BUY_ZONE unless canSizeTrade is true
  if (state.finalVerdict === "ACTIONABLE_BUY_ZONE" && !state.canSizeTrade) {
    violations.push(`finalVerdict is ACTIONABLE_BUY_ZONE while canSizeTrade is false`);
  }

  // Invariant 6: uiStateLabel cannot claim actionable unless canSizeTrade is true
  if (state.uiStateLabel.toLowerCase().includes("actionable") && !state.canSizeTrade) {
    violations.push(`uiStateLabel claims actionable ('${state.uiStateLabel}') while canSizeTrade is false`);
  }

  // Invariant 7: size_trade action cannot be enabled unless canSizeTrade is true
  if (state.availableActions) {
    const sizeAction = state.availableActions.find((a) => a.id === "size_trade");
    if (sizeAction?.enabled && !state.canSizeTrade) {
      violations.push(`size_trade action is enabled while canSizeTrade is false`);
    }
  }

  return {
    isValid: violations.length === 0,
    violations,
  };
}

export function assertDecisionAuthorityConsistency(state: Parameters<typeof validateDecisionAuthorityConsistency>[0]): void {
  const check = validateDecisionAuthorityConsistency(state);
  if (!check.isValid) {
    throw new Error(`Decision Authority Invariant Violation:\n  - ${check.violations.join("\n  - ")}`);
  }
}
