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

export interface DecisionReadinessResult {
  isExecutionReady: boolean;
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

  // ── GATE 1: LOCATION / GEOMETRY ───────────────────────────────────────────
  let gate1State: GateState = "UNAVAILABLE";
  let gate1Explanation = "Optimal entry corridor uncalculated or candle history insufficient (< 50 sessions).";

  const hasValidPrice = typeof currentPrice === "number" && !isNaN(currentPrice) && currentPrice > 0;
  const hasValidMin = typeof optimalEntryMin === "number" && !isNaN(optimalEntryMin) && optimalEntryMin > 0;
  const hasValidMax = typeof optimalEntryMax === "number" && !isNaN(optimalEntryMax) && optimalEntryMax > 0;
  const hasSufficientHistory = candleCount == null || candleCount >= 50;

  if (hasValidPrice && hasValidMin && hasValidMax && hasSufficientHistory) {
    const entryMin = Math.min(optimalEntryMin!, optimalEntryMax!);
    const entryMax = Math.max(optimalEntryMin!, optimalEntryMax!);

    if (isStage4) {
      gate1State = "BLOCKING";
      gate1Explanation = "Asset is in Stage 4 distribution. Structural base required before accumulation corridor applies.";
    } else if (currentPrice! > entryMax * 1.02) {
      gate1State = "BLOCKING";
      gate1Explanation = `Price ($${currentPrice!.toFixed(2)}) has extended >2% past the accumulation corridor ($${entryMin.toFixed(2)}–$${entryMax.toFixed(2)}). Do not chase above resistance.`;
    } else if (currentPrice! < entryMin) {
      gate1State = "BLOCKING";
      gate1Explanation = `Price ($${currentPrice!.toFixed(2)}) is below the accumulation corridor ($${entryMin.toFixed(2)}–$${entryMax.toFixed(2)}). Awaiting base stabilization.`;
    } else {
      gate1State = "PASSED";
      gate1Explanation = `Price ($${currentPrice!.toFixed(2)}) is positioned inside the institutional accumulation corridor ($${entryMin.toFixed(2)}–$${entryMax.toFixed(2)}).`;
    }
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
      gate2Explanation = "Price is inside the accumulation corridor, but confirmation trigger is pending. Awaiting volume expansion / pivot reclaim.";
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

  const isExecutionReady =
    gate1State === "PASSED" && gate2State === "PASSED" && gate3State === "PASSED";

  // ── Next Required Condition ───────────────────────────────────────────────
  let nextRequiredCondition = "Setup is execution ready. Capital allocation cleared.";
  if (activeBlockingGate === "GATE_1_LOCATION") {
    if (isStage4) {
      nextRequiredCondition = "Wait for structural floor formation and 50-day moving average breakout pivot reclaim.";
    } else if (hasValidPrice && hasValidMax && currentPrice! > optimalEntryMax! * 1.02) {
      nextRequiredCondition = "Wait for pullback into optimal accumulation corridor without chasing.";
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
  }

  // ── Protective Negative Guidance (CLAIM_SET ⊆ EVIDENCE_SET) ───────────────
  let negativeGuidance: string | null = null;
  if (hasValidPrice && hasValidMax && currentPrice! > optimalEntryMax! * 1.02) {
    negativeGuidance = "DO NOT CHASE: Price has extended past the accumulation corridor. Wait for a pullback to the buy zone.";
  } else if (isStage4) {
    negativeGuidance = "CAPITAL DEFENSE: Asset is in Stage 4 correction below 50-day SMA. Wait for base stabilization and pivot reclaim.";
  } else if (gate1State === "PASSED" && gate2State === "BLOCKING") {
    negativeGuidance = "DO NOT PRE-EMPT: Price is in buy zone, but confirmation trigger is pending. Awaiting volume breakout candle.";
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
  } else if (activeBlockingGate === "GATE_1_LOCATION") {
    if (isStage4) {
      primaryAction = {
        label: "Explore Radar Setups",
        actionType: "EXPLORE_RADAR",
        reason: "Asset in Stage 4 correction. Browse non-distressed alternatives on Radar.",
      };
    } else if (hasValidPrice && hasValidMax && currentPrice! > optimalEntryMax! * 1.02) {
      primaryAction = {
        label: "Set Pullback Alert",
        actionType: "SET_PULLBACK_ALERT",
        reason: "Price extended. Set browser alert for pullback into the accumulation corridor.",
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
      reason: "Price is in accumulation corridor; monitor for volume breakout confirmation.",
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
