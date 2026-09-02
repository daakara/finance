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

  // 1. Determine Overall Data Eligibility
  let overallEligibility: "ELIGIBLE" | "LIMITED" | "INELIGIBLE" = "ELIGIBLE";
  if (agreement.evaluated === 0) {
    overallEligibility = "INELIGIBLE";
  } else if (agreement.unavailable > 0) {
    overallEligibility = "LIMITED";
  }

  // 2. Derive Overall Evidence Assessment
  let assessment: Assessment = "MIXED";
  if (overallEligibility === "INELIGIBLE") {
    assessment = "INSUFFICIENT_EVIDENCE";
  } else if (agreement.favorable >= 3 && agreement.unfavorable === 0) {
    assessment = "FAVORABLE";
  } else if (agreement.unfavorable >= 2) {
    assessment = "UNFAVORABLE";
  } else {
    assessment = "MIXED";
  }

  // 3. Setup Invalidation Level
  const safePrice = currentPrice > 0 ? currentPrice : 100;
  const stopLevel = invalidationPrice ?? Number((safePrice * 0.93).toFixed(2));
  const distancePct = Number((((stopLevel - safePrice) / safePrice) * 100).toFixed(1));
  const reclaimTarget = reclaimMilestonePrice ?? Number((safePrice * 1.115).toFixed(2));

  // 4. Resolve Contextual Posture (Precedence: Data Ineligible -> Invalidation Breached -> Assessment + Ownership)
  let posture: DecisionPosture = "WATCH";
  let uiStateLabel = "Wait for Trigger";
  let headlineExplanation = "Setup not confirmed yet.";

  if (overallEligibility === "INELIGIBLE") {
    posture = "RESEARCH";
    uiStateLabel = "Assessment Unavailable — Data Incomplete";
    headlineExplanation = "ARX cannot reliably assess this asset due to missing required data observations.";
  } else if (ownershipState === "OWNED") {
    if (isInvalidationBreached) {
      posture = "EXIT_REVIEW";
      uiStateLabel = "Thesis Needs Review";
      headlineExplanation = `Price breached the setup invalidation level at $${stopLevel} (${distancePct}%).`;
    } else if (assessment === "UNFAVORABLE") {
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
    // NOT_OWNED or UNKNOWN
    if (assessment === "FAVORABLE" && !isInvalidationBreached) {
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
      headlineExplanation = `Price remains below 50-day average ($${reclaimTarget.toFixed(2)}); awaiting base confirmation.`;
    }
  }

  // 5. Derive Contextual Actions
  const availableActions: ARXAction[] = [
    {
      id: "set_alert",
      type: "SET_ALERT",
      label: `Set Alert for $${reclaimTarget.toFixed(2)}`,
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
      : `Reclaiming and holding above $${reclaimTarget.toFixed(2)} (50D SMA) with volume expansion would upgrade posture to ACQUIRE.`,
    primaryAction: {
      label: posture === "ACQUIRE" ? "Calculate Position Size" : `Set Alert for $${reclaimTarget.toFixed(2)}`,
      actionType: posture === "ACQUIRE" ? "SIZE_TRADE" : "SET_ALERT",
      enabled: overallEligibility !== "INELIGIBLE",
    },
    availableActions,
  };
}
