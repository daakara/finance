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
  OwnershipState,
  OwnershipSource,
  DomainAssessment,
  FactorAgreement,
  TerminalViewState,
  ModelProvenance,
  ARXAction,
} from "../types/insight";

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
  modelProvenance?: ModelProvenance;
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
    modelProvenance = {
      modelId: "arx-confluence-engine",
      modelVersion: "2.4.0",
      rulesetVersion: "2026.09-v1",
      calculatedAt: new Date().toISOString(),
    },
  } = input;

  const agreement = calculateFactorAgreement(domains);

  // 1. Determine Overall Data Eligibility and Core Evidence Completeness
  let overallEligibility: "ELIGIBLE" | "LIMITED" | "INELIGIBLE" = "ELIGIBLE";
  if (agreement.evaluated === 0) {
    overallEligibility = "INELIGIBLE";
  } else if (agreement.unavailable > 0) {
    overallEligibility = "LIMITED";
  }

  const trendDomain = domains.find((d) => d.domainId === "trend");
  const healthDomain = domains.find((d) => d.domainId === "health");

  const isTrendAvailable = trendDomain !== undefined && trendDomain.availability === "AVAILABLE" && trendDomain.status !== "UNAVAILABLE";
  const isHealthAvailable = healthDomain !== undefined && healthDomain.availability === "AVAILABLE" && healthDomain.status !== "UNAVAILABLE";
  const isCoreEvidenceAvailable = isTrendAvailable && isHealthAvailable && overallEligibility === "ELIGIBLE";

  // 2. Derive Overall Evidence Assessment (Requires full core evidence for FAVORABLE)
  let assessment: Assessment = "MIXED";
  if (overallEligibility === "INELIGIBLE" || !isCoreEvidenceAvailable) {
    assessment = "INSUFFICIENT_EVIDENCE";
  } else if (agreement.favorable >= 3 && agreement.unfavorable === 0) {
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
  } else if (!isCoreEvidenceAvailable || overallEligibility !== "ELIGIBLE" || !isTrendAvailable) {
    posture = "RESEARCH";
    uiStateLabel = "Evidence Incomplete — In-Depth Research Required";
    headlineExplanation = !isTrendAvailable
      ? "Technical trend evidence is unavailable (insufficient historical sessions). Active triggers cannot be confirmed."
      : "Core fundamental financial evidence is unverified. In-depth due diligence required before evaluating setups.";
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
    if (assessment === "FAVORABLE") {
      posture = "ACQUIRE";
      uiStateLabel = "Actionable Setup";
      headlineExplanation = "Multi-factor confluence confirmed in optimal buy zone.";
    } else if (assessment === "UNFAVORABLE") {
      posture = "AVOID";
      uiStateLabel = "Unfavorable Setup";
      headlineExplanation = "Negative trend or poor fundamentals present unfavorable risk/reward.";
    } else {
      posture = "WATCH";
      uiStateLabel = "Wait for Trigger";
      headlineExplanation = reclaimMilestonePrice !== undefined
        ? `Awaiting constructive base confirmation and reclaim of $${reclaimMilestonePrice.toFixed(2)}.`
        : "Awaiting constructive consolidation and volume confirmation before trigger.";
    }
  }

  // 5. Derive Contextual Actions
  const availableActions: ARXAction[] = [
    {
      id: "set_alert",
      type: "SET_ALERT",
      label: reclaimMilestonePrice !== undefined ? `Set Alert for $${reclaimMilestonePrice.toFixed(2)}` : "Set Price Alert",
      enabled: overallEligibility !== "INELIGIBLE",
    },
    {
      id: "size_trade",
      type: "SIZE_POSITION",
      label: "Calculate Position Size",
      enabled: posture === "ACQUIRE",
      reason: posture !== "ACQUIRE" ? "Only actionable when in Buy Zone" : undefined,
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

  return {
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
      : reclaimMilestonePrice !== undefined
      ? `Reclaiming and holding above $${reclaimMilestonePrice.toFixed(2)} (50D SMA) with volume expansion would upgrade posture to ACQUIRE.`
      : "Sufficient historical trading data and constructive base formation required to evaluate potential upgrade.",
    primaryAction: {
      label: posture === "ACQUIRE"
        ? "Calculate Position Size"
        : posture === "EXIT_REVIEW"
        ? "Review Exit Criteria"
        : posture === "RESEARCH"
        ? "Conduct Fundamental Research"
        : reclaimMilestonePrice !== undefined
        ? `Set Alert for $${reclaimMilestonePrice.toFixed(2)}`
        : "Set Price Alert",
      actionType: posture === "ACQUIRE"
        ? "SIZE_TRADE"
        : posture === "EXIT_REVIEW"
        ? "REVIEW_THESIS"
        : posture === "RESEARCH"
        ? "RESEARCH_PROFILE"
        : "SET_ALERT",
      enabled: overallEligibility !== "INELIGIBLE",
    },
    availableActions,
  };
}
