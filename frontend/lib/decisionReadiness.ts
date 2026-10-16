/**
 * ARX Universal Decision Readiness Resolver (Synthesis E Wave 4).
 *
 * Implements a pure, deterministic 3-gate readiness progression:
 *   GATE 1: LOCATION / GEOMETRY
 *            ↓
 *   GATE 2: DYNAMIC TRIGGER
 *            ↓
 *   GATE 3: RISK / GOVERNANCE CLEARANCE
 *
 * Enforces strict fail-closed dependency cascades and single Active Blocker semantics.
 * Downstream gates NEVER display as failed when upstream prerequisite gates are unmet.
 */

import {
  evaluateCorridorLocation,
  CorridorLocationResult,
  CorridorSpatialRelation,
} from "./corridorLocation";

export { evaluateCorridorLocation };
export type { CorridorLocationResult, CorridorSpatialRelation };

export type GateState = "PASSED" | "BLOCKING" | "PENDING_DEPENDENCY" | "UNAVAILABLE";

export interface DecisionReadinessGate {
  gateId: "GATE_1_LOCATION" | "GATE_2_TRIGGER" | "GATE_3_RISK_CLEARANCE";
  displayName: string;
  state: GateState;
  explanation: string;
}

export type ReadinessActionType =
  | "SIZE_POSITION"
  | "SET_PULLBACK_ALERT"
  | "SET_BUY_ZONE_ALERT"
  | "EXPLORE_RADAR"
  | "OPEN_WATCHLIST"
  | "INSPECT_EVIDENCE"
  | "NONE";

export type ReadinessVerdict =
  | "EXECUTION_READY"
  | "AWAITING_CANONICAL_CLEARANCE"
  | "SETUP_IN_PROGRESS";

export interface DecisionReadinessResult {
  isExecutionReady: boolean;
  technicalGatesPassed: boolean;
  readinessVerdict: ReadinessVerdict;
  readinessVerdictLabel: string;
  activeBlockingGate: "GATE_1_LOCATION" | "GATE_2_TRIGGER" | "GATE_3_RISK_CLEARANCE" | "NONE";
  gates: [DecisionReadinessGate, DecisionReadinessGate, DecisionReadinessGate];
  nextRequiredCondition: string;
  negativeGuidance: string | null;
  primaryAction: {
    label: string;
    actionType: ReadinessActionType;
    reason: string;
  };
}

export interface DecisionReadinessInputs {
  currentPrice?: number | null;
  optimalEntryMin?: number | null;
  optimalEntryMax?: number | null;
  stagePhase?: string | null;
  setupPattern?: string | null;
  candleCount?: number | null;
  isConfirmed?: boolean | null;
  riskRewardRatio?: number | null;
  stopLoss?: number | null;
  takeProfit1?: number | null;
  vix?: number | null;
  isActionable?: boolean | string | null;
}

export function resolveDecisionReadiness(inputs: DecisionReadinessInputs): DecisionReadinessResult {
  const {
    currentPrice,
    optimalEntryMin,
    optimalEntryMax,
    stagePhase,
    setupPattern,
    candleCount,
    isConfirmed,
    riskRewardRatio,
    stopLoss,
    takeProfit1,
    vix,
  } = inputs;

  // ── Helper: Stage 4 Detection ─────────────────────────────────────────────
  const isStage4 = Boolean(
    setupPattern?.toLowerCase().includes("stage 4") ||
    setupPattern?.toLowerCase().includes("correction") ||
    stagePhase?.toLowerCase().includes("stage 4") ||
    stagePhase?.toLowerCase().includes("markdown")
  );

  // ── Pure Geometric Location Evaluation (Workstream D — Slice 1) ───────────
  const corridorLocation = evaluateCorridorLocation(
    currentPrice,
    optimalEntryMin,
    optimalEntryMax,
    0.02
  );

  // ── GATE 1: LOCATION / GEOMETRY ───────────────────────────────────────────
  let gate1State: GateState = "UNAVAILABLE";
  let gate1Explanation = "Optimal entry corridor uncalculated or candle history insufficient (< 50 sessions).";

  const hasSufficientHistory = candleCount == null || candleCount >= 50;

  if (corridorLocation.isValid && hasSufficientHistory) {
    if (isStage4) {
      gate1State = "BLOCKING";
      gate1Explanation = "Asset is in Stage 4 distribution. Structural base required before accumulation corridor applies.";
    } else if (corridorLocation.spatialRelation === "BELOW") {
      gate1State = "BLOCKING";
      gate1Explanation = corridorLocation.description;
    } else if (corridorLocation.spatialRelation === "ABOVE") {
      if (corridorLocation.isWithinTolerance) {
        // Preserves existing 2% implementation tolerance decision behavior while truthfully reporting location
        gate1State = "PASSED";
        gate1Explanation = corridorLocation.description;
      } else {
        gate1State = "BLOCKING";
        gate1Explanation = corridorLocation.description;
      }
    } else if (corridorLocation.spatialRelation === "INSIDE") {
      gate1State = "PASSED";
      gate1Explanation = corridorLocation.description;
    }
  } else if (!corridorLocation.isValid && optimalEntryMin != null && optimalEntryMax != null && optimalEntryMin > optimalEntryMax) {
    gate1State = "UNAVAILABLE";
    gate1Explanation = corridorLocation.description;
  }

  // ── GATE 2: DYNAMIC TRIGGER ───────────────────────────────────────────────
  let gate2State: GateState = "PENDING_DEPENDENCY";
  let gate2Explanation = "Waiting for Gate 1 (Location) to clear before evaluating trigger confirmation.";

  if (gate1State === "PASSED") {
    if (isConfirmed == null) {
      gate2State = "UNAVAILABLE";
      gate2Explanation = "Trigger telemetry unavailable.";
    } else if (isConfirmed === true) {
      gate2State = "PASSED";
      gate2Explanation = "Breakout volume expansion or reversal confirmation candle verified.";
    } else {
      gate2State = "BLOCKING";
      gate2Explanation = corridorLocation.isWithinTolerance
        ? "Price is above the entry corridor (within implementation tolerance), but confirmation trigger is pending. Awaiting volume expansion / pivot reclaim."
        : "Price is inside the accumulation corridor, but confirmation trigger is pending. Awaiting volume expansion / pivot reclaim.";
    }
  }

  // ── GATE 3: RISK / GOVERNANCE CLEARANCE ───────────────────────────────────
  let gate3State: GateState = "PENDING_DEPENDENCY";
  let gate3Explanation = "Waiting for Gate 1 (Location) and Gate 2 (Trigger) to clear before clearing execution risk.";

  if (gate1State === "PASSED" && gate2State === "PASSED") {
    const hasValidStop = typeof stopLoss === "number" && !isNaN(stopLoss) && stopLoss > 0;
    const hasValidTarget = typeof takeProfit1 === "number" && !isNaN(takeProfit1) && takeProfit1 > 0;
    const hasValidRR = typeof riskRewardRatio === "number" && !isNaN(riskRewardRatio) && riskRewardRatio > 0;
    const hasValidVix = typeof vix === "number" && !isNaN(vix) && vix > 0;

    // Strict Fail-Closed Invariant: Missing VIX is strictly UNAVAILABLE, NEVER PASSED
    if (!hasValidVix) {
      gate3State = "UNAVAILABLE";
      gate3Explanation = "Macro volatility reading unavailable. Risk clearance requires live verified VIX tape (fail-closed).";
    } else if (!hasValidStop || !hasValidTarget || !hasValidRR) {
      gate3State = "UNAVAILABLE";
      gate3Explanation = "Execution levels or risk/reward ratio uncalculated.";
    } else if (vix! >= 26.0) {
      gate3State = "BLOCKING";
      gate3Explanation = `Broad market volatility is elevated (VIX ${vix!.toFixed(1)} ≥ 26.0). New equity risk deployment suspended.`;
    } else if (riskRewardRatio! < 2.0) {
      gate3State = "BLOCKING";
      gate3Explanation = `Prospective Risk/Reward (${riskRewardRatio!.toFixed(1)}:1) is below the institutional 2:1 minimum floor. Capital efficiency inadequate.`;
    } else {
      gate3State = "PASSED";
      gate3Explanation = `Setup clears the 2:1 institutional risk/reward floor (${riskRewardRatio!.toFixed(1)}:1) under calm macro volatility (VIX ${vix!.toFixed(1)} < 26.0).`;
    }
  }

  // ── Active Blocker & Execution Readiness ──────────────────────────────────
  let activeBlockingGate: "GATE_1_LOCATION" | "GATE_2_TRIGGER" | "GATE_3_RISK_CLEARANCE" | "NONE" = "NONE";

  if (gate1State !== "PASSED") {
    activeBlockingGate = "GATE_1_LOCATION";
  } else if (gate2State !== "PASSED") {
    activeBlockingGate = "GATE_2_TRIGGER";
  } else if (gate3State !== "PASSED") {
    activeBlockingGate = "GATE_3_RISK_CLEARANCE";
  }

  const technicalGatesPassed =
    gate1State === "PASSED" && gate2State === "PASSED" && gate3State === "PASSED";

  const isActionableProvided = inputs.isActionable !== undefined;
  const canonicalActionable = isActionableProvided
    ? validateCanonicalActionable(inputs.isActionable)
    : true; // Backward compatibility with unit tests when isActionable is omitted

  const isExecutionReady = technicalGatesPassed && canonicalActionable;

  let readinessVerdict: ReadinessVerdict;
  let readinessVerdictLabel: string;

  if (isExecutionReady) {
    readinessVerdict = "EXECUTION_READY";
    readinessVerdictLabel = "EXECUTION READY";
  } else if (technicalGatesPassed) {
    readinessVerdict = "AWAITING_CANONICAL_CLEARANCE";
    readinessVerdictLabel = "TECHNICAL CHECKS PASSED — AWAITING CANONICAL CLEARANCE";
  } else {
    readinessVerdict = "SETUP_IN_PROGRESS";
    readinessVerdictLabel = "SETUP IN PROGRESS";
  }

  // ── Next Required Condition ───────────────────────────────────────────────
  let nextRequiredCondition = "Setup is execution ready. Capital allocation cleared.";
  if (activeBlockingGate === "GATE_1_LOCATION") {
    if (isStage4) {
      nextRequiredCondition = "Wait for structural floor formation and 50-day moving average breakout pivot reclaim.";
    } else if (corridorLocation.spatialRelation === "ABOVE") {
      nextRequiredCondition = "Wait for pullback into optimal accumulation corridor without chasing.";
    } else if (!corridorLocation.isValid) {
      nextRequiredCondition = "Corridor bounds must be verified before accumulation corridor applies.";
    } else {
      nextRequiredCondition = "Price must stabilize and enter the defined accumulation corridor.";
    }
  } else if (activeBlockingGate === "GATE_2_TRIGGER") {
    nextRequiredCondition = "Wait for volume expansion confirmation candle or pivot breakout reclaim.";
  } else if (activeBlockingGate === "GATE_3_RISK_CLEARANCE") {
    if (vix != null && vix >= 26.0) {
      nextRequiredCondition = "Wait for broad market volatility (VIX) to subside below 26.0.";
    } else {
      nextRequiredCondition = "Execution levels must offer at least a 2:1 prospective reward-to-risk asymmetry.";
    }
  } else if (!isExecutionReady) {
    nextRequiredCondition = "Technical checks passed — awaiting canonical clearance. Canonical action posture is Wait; entry remains conditional.";
  }

  // ── Protective Negative Guidance (CLAIM_SET ⊆ EVIDENCE_SET) ───────────────
  let negativeGuidance: string | null = null;
  if (corridorLocation.spatialRelation === "ABOVE" && !corridorLocation.isWithinTolerance) {
    negativeGuidance = "DO NOT CHASE: Price has extended past the accumulation corridor. Wait for a pullback to the buy zone.";
  } else if (isStage4) {
    negativeGuidance = "CAPITAL DEFENSE: Asset is in Stage 4 correction below 50-day SMA. Wait for base stabilization and pivot reclaim.";
  } else if (gate1State === "PASSED" && gate2State === "BLOCKING") {
    negativeGuidance = corridorLocation.isWithinTolerance
      ? "DO NOT PRE-EMPT: Price is above the entry corridor (within extension tolerance), but confirmation trigger is pending. Awaiting volume breakout candle."
      : "DO NOT PRE-EMPT: Price is in buy zone, but confirmation trigger is pending. Awaiting volume breakout candle.";
  } else if (typeof riskRewardRatio === "number" && !isNaN(riskRewardRatio) && riskRewardRatio > 0 && riskRewardRatio < 2.0) {
    negativeGuidance = `INADEQUATE ASYMMETRY: Prospective reward (${riskRewardRatio.toFixed(1)}:1) is below the 2:1 institutional floor. Do not allocate capital without favorable asymmetry.`;
  } else if (typeof vix === "number" && !isNaN(vix) && vix >= 26.0) {
    negativeGuidance = `MACRO CAUTION: Market volatility (VIX ${vix.toFixed(1)} ≥ 26.0) is elevated. Broad equity breakout follow-through suppressed.`;
  } else if (gate1State === "PASSED" && gate2State === "PASSED" && (vix == null || isNaN(vix))) {
    negativeGuidance = "MACRO DATA DEGRADED: Live market volatility unavailable. Risk clearance suspended until authentic VIX tape is acquired.";
  }

  // ── Primary Operational Action Resolution ─────────────────────────────────
  let primaryAction: { label: string; actionType: ReadinessActionType; reason: string } = {
    label: "Explore Radar Setups",
    actionType: "EXPLORE_RADAR",
    reason: "Browse alternative candidate setups on Radar.",
  };

  if (isExecutionReady) {
    primaryAction = {
      label: "Size Position",
      actionType: "SIZE_POSITION",
      reason: "All 3 readiness gates passed. Proceed to position sizing and risk management.",
    };
  } else if (technicalGatesPassed) {
    primaryAction = {
      label: "Size Position",
      actionType: "SIZE_POSITION",
      reason: "Technical checks passed. Review prospective sizing while awaiting canonical clearance.",
    };
  } else if (activeBlockingGate === "GATE_1_LOCATION") {
    if (isStage4) {
      primaryAction = {
        label: "Explore Radar Setups",
        actionType: "EXPLORE_RADAR",
        reason: "Asset in Stage 4 correction. Browse non-distressed alternatives on Radar.",
      };
    } else if (corridorLocation.spatialRelation === "ABOVE") {
      primaryAction = {
        label: "Set Pullback Alert",
        actionType: "SET_PULLBACK_ALERT",
        reason: "Price extended. Set browser alert for pullback into the accumulation corridor.",
      };
    } else if (!corridorLocation.isValid) {
      primaryAction = {
        label: "Explore Radar Setups",
        actionType: "EXPLORE_RADAR",
        reason: "Corridor bounds unavailable or invalid. Browse verified alternatives on Radar.",
      };
    } else {
      primaryAction = {
        label: "Set Buy Zone Alert",
        actionType: "SET_BUY_ZONE_ALERT",
        reason: "Set browser alert to trigger when price enters the accumulation corridor.",
      };
    }
  } else if (activeBlockingGate === "GATE_2_TRIGGER") {
    primaryAction = {
      label: "Set Buy Zone Alert",
      actionType: "SET_BUY_ZONE_ALERT",
      reason: corridorLocation.isWithinTolerance
        ? "Price is above entry corridor (within tolerance); monitor for volume breakout confirmation."
        : "Price is in accumulation corridor; monitor for volume breakout confirmation.",
    };
  } else if (activeBlockingGate === "GATE_3_RISK_CLEARANCE") {
    primaryAction = {
      label: "Explore Radar Setups",
      actionType: "EXPLORE_RADAR",
      reason: "Setup does not satisfy risk floor or macro volatility threshold. Seek higher asymmetry setups.",
    };
  }

  return {
    isExecutionReady,
    technicalGatesPassed,
    readinessVerdict,
    readinessVerdictLabel,
    activeBlockingGate,
    gates: [
      {
        gateId: "GATE_1_LOCATION",
        displayName: "1. Corridor Location & Geometry",
        state: gate1State,
        explanation: gate1Explanation,
      },
      {
        gateId: "GATE_2_TRIGGER",
        displayName: "2. Dynamic Trigger & Volume Confirmation",
        state: gate2State,
        explanation: gate2Explanation,
      },
      {
        gateId: "GATE_3_RISK_CLEARANCE",
        displayName: "3. Risk Floor & Macro Clearance",
        state: gate3State,
        explanation: gate3Explanation,
      },
    ],
    nextRequiredCondition,
    negativeGuidance,
    primaryAction,
  };
}

export interface TacticalExecutionHintInputs {
  evalPrice: number;
  entryMin: number;
  entryMax: number;
  isStage4?: boolean;
  corridorLoc: CorridorLocationResult;
  inZone?: boolean;
  zonePositionPct?: number;
  riskRewardRatio?: number | null;
  isActionable?: boolean | string | null;
  isTriggerConfirmed?: boolean | null;
  readinessResult: DecisionReadinessResult;
}

export interface TacticalExecutionHint {
  label: string;
  advice: string;
  color: string;
}

/**
 * Strict validator for canonical actionability.
 * Fail-closed: Only boolean true or explicit normalized string "true" resolves to true.
 * String "false", "0", null, undefined, objects, numbers, and malformed inputs
 * strictly resolve to false, preventing boolean coercion leakage (e.g. Boolean("false") === true).
 */
export function validateCanonicalActionable(value: unknown): boolean {
  if (value === true) return true;
  if (value === false || value === null || value === undefined) return false;
  if (typeof value === "string") {
    return value.trim().toLowerCase() === "true";
  }
  return false;
}

/**
 * Pure resolver for tactical execution hints in the conditional trade plan card.
 * Enforces Slice 1 consistency:
 * 1. Never describe an above-ceiling price as inside.
 * 2. Suppress actionable scale-in / entry guidance when canonical action or required execution conditions are blocked, pending, or not satisfied.
 * 3. Preserves 2% tolerance and positive hints when fully execution-ready.
 */
export function resolveTacticalExecutionHint(
  inputs: TacticalExecutionHintInputs
): TacticalExecutionHint | null {
  const {
    evalPrice,
    entryMin,
    entryMax,
    isStage4 = false,
    corridorLoc,
    inZone = false,
    zonePositionPct = 50,
    riskRewardRatio,
    isActionable,
    isTriggerConfirmed,
    readinessResult,
  } = inputs;

  const isActionSatisfied = validateCanonicalActionable(isActionable) && readinessResult.isExecutionReady;

  if (isStage4 || !corridorLoc.isValid || (riskRewardRatio != null && riskRewardRatio < 1.0)) {
    return null;
  }

  const min = corridorLoc.entryMin ?? entryMin;
  const max = corridorLoc.entryMax ?? entryMax;

  if (corridorLoc.spatialRelation === "ABOVE" && corridorLoc.isWithinTolerance) {
    if (!isActionSatisfied) {
      // Tolerated extension, but action is blocked / trigger pending / not satisfied
      const blockerReason = isTriggerConfirmed === false
        ? "Wait for required trigger confirmation."
        : readinessResult.activeBlockingGate !== "NONE"
        ? readinessResult.nextRequiredCondition
        : "Canonical action posture is Wait. Entry remains conditional.";
      return {
        label: "⚠️ Above Corridor Ceiling",
        advice: `Spot ($${evalPrice.toFixed(2)}) is above the entry corridor ($${min.toFixed(2)}–$${max.toFixed(2)}). Existing implementation tolerance applies, but entry remains conditional. ${blockerReason}`,
        color: "text-amber-300 border-amber-900/60 bg-amber-950/40",
      };
    } else {
      // Fully satisfied canonical state within tolerance
      return {
        label: "⚠️ Above Corridor Ceiling",
        advice: `Spot ($${evalPrice.toFixed(2)}) is above the entry corridor ($${min.toFixed(2)}–$${max.toFixed(2)}). Existing implementation tolerance applies. Scale in with 30% initial size or place limit orders near $${min.toFixed(2)} to maximize asymmetric R:R.`,
        color: "text-amber-300 border-amber-900/60 bg-amber-950/40",
      };
    }
  }

  if (corridorLoc.spatialRelation === "INSIDE" || (inZone && corridorLoc.spatialRelation !== "ABOVE")) {
    if (!isActionSatisfied) {
      // Inside corridor, but action is blocked / trigger pending / not satisfied
      const blockerReason = isTriggerConfirmed === false
        ? "Wait for required trigger confirmation."
        : readinessResult.activeBlockingGate !== "NONE"
        ? readinessResult.nextRequiredCondition
        : "Canonical action posture is Wait. Entry remains conditional.";
      return {
        label: "⏳ Trigger Pending",
        advice: `Spot ($${evalPrice.toFixed(2)}) is inside the accumulation corridor, but entry remains conditional. ${blockerReason}`,
        color: "text-amber-300 border-amber-900/60 bg-amber-950/40",
      };
    } else {
      // Fully satisfied canonical state inside corridor
      if (zonePositionPct > 65) {
        return {
          label: "⚠️ Near Zone Ceiling",
          advice: `Spot ($${evalPrice.toFixed(2)}) is near the upper bound of the buy zone. Scale in with 30% initial size or place limit orders near $${min.toFixed(2)} to maximize asymmetric R:R.`,
          color: "text-amber-300 border-amber-900/60 bg-amber-950/40",
        };
      } else if (zonePositionPct < 35) {
        return {
          label: "🎯 Near Support Floor",
          advice: `Spot ($${evalPrice.toFixed(2)}) is at the bottom of the accumulation corridor. Favorable asymmetric entry with tight invalidation floor.`,
          color: "text-emerald-300 border-emerald-900/60 bg-emerald-950/40",
        };
      } else {
        return {
          label: "✅ Mid-Zone Value Area",
          advice: `Spot is comfortably centered within the accumulation corridor with favorable ${riskRewardRatio}:1 R:R structure.`,
          color: "text-cyan-300 border-cyan-900/60 bg-cyan-950/40",
        };
      }
    }
  }

  return null;
}

