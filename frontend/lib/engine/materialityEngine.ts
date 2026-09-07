/**
 * ARX Terminal vNext - Materiality Engine
 * Source of Truth: docs/sprints/SPRINT_3_CHANGE_INTELLIGENCE_EXECUTION_PACKAGE.md
 * 
 * Core Invariant (ADR-008):
 * "A change may only appear in Stage 6 if it survives Materiality Evaluation.
 *  Snapshot differences alone are insufficient grounds for user interruption."
 */

import {
  TickerSnapshot,
  DeltaItem,
  DeltaReport,
  MaterialitySeverity,
  AttentionSignal,
} from "../../types/change-intelligence";
import { ExecutionState, LiquidityTier, MarketRegime } from "../../types/workstation";

const SEVERITY_WEIGHTS: Record<MaterialitySeverity, number> = {
  NONE: 0,
  INFO: 1,
  MATERIAL: 2,
  CRITICAL: 3,
};

/**
 * Layer 2: Setup Score Materiality Thresholds
 * 0-2 pts -> NONE (Noise / Decimal variance)
 * 3-5 pts -> INFO (Minor shift)
 * 6-9 pts -> MATERIAL (Noticeable conviction shift)
 * 10+ pts -> CRITICAL (Major breakout / thesis revision)
 */
export function evaluateScoreMateriality(
  prevScore: number,
  currScore: number
): { severity: MaterialitySeverity; delta: number } {
  // Discard microscopic recalculation / float noise
  const roundedPrev = Math.round(prevScore);
  const roundedCurr = Math.round(currScore);
  const delta = roundedCurr - roundedPrev;
  const absDelta = Math.abs(delta);

  if (absDelta <= 2) {
    return { severity: "NONE", delta };
  }
  if (absDelta <= 5) {
    return { severity: "INFO", delta };
  }
  if (absDelta <= 9) {
    return { severity: "MATERIAL", delta };
  }
  return { severity: "CRITICAL", delta };
}

/**
 * Layer 2: Institutional Flow Z-Score Thresholds
 * < 1.0σ   -> NONE
 * 1.0-1.5σ -> INFO
 * 1.5-2.5σ -> MATERIAL
 * >= 2.5σ  -> CRITICAL (Whale Block Accumulation Surge)
 */
export function evaluateFlowMateriality(
  prevZ: number,
  currZ: number
): { severity: MaterialitySeverity; delta: number } {
  const delta = Number((currZ - prevZ).toFixed(2));
  const absDelta = Math.abs(delta);

  if (absDelta < 1.0) {
    return { severity: "NONE", delta };
  }
  if (absDelta < 1.5) {
    return { severity: "INFO", delta };
  }
  if (absDelta < 2.5) {
    return { severity: "MATERIAL", delta };
  }
  return { severity: "CRITICAL", delta };
}

/**
 * Layer 2: Execution State Transitions
 * Unchanged -> NONE
 * Any transition -> CRITICAL (Direct actionability change)
 */
export function evaluateExecutionStateMateriality(
  prevState: ExecutionState,
  currState: ExecutionState
): { severity: MaterialitySeverity; reason: string } {
  if (prevState === currState) {
    return { severity: "NONE", reason: "Execution state unchanged" };
  }

  if (currState === "IN_BUY_ZONE") {
    return {
      severity: "CRITICAL",
      reason: `Entered Actionable Buy Zone (was ${prevState})`,
    };
  }
  if (currState === "STOPPED_OUT") {
    return {
      severity: "CRITICAL",
      reason: "Price violated Stop Loss Floor - Thesis stopped out",
    };
  }
  if (currState === "APPROACHING_TARGET") {
    return {
      severity: "MATERIAL",
      reason: "Price approaching Take Profit Target 1",
    };
  }

  return {
    severity: "CRITICAL",
    reason: `State shifted from ${prevState} to ${currState}`,
  };
}

/**
 * Layer 2: Market Regime Rotation
 * Unchanged -> NONE
 * Any transition -> CRITICAL (Macro posture shift)
 */
export function evaluateRegimeMateriality(
  prevRegime: MarketRegime,
  currRegime: MarketRegime
): { severity: MaterialitySeverity; reason: string } {
  if (prevRegime === currRegime) {
    return { severity: "NONE", reason: "Market regime unchanged" };
  }

  return {
    severity: "CRITICAL",
    reason: `Market macro regime rotated from ${prevRegime} to ${currRegime}`,
  };
}

/**
 * Layer 2: Liquidity Tier Transitions
 * Same tier -> NONE
 * High -> Moderate -> MATERIAL
 * Any -> Risk -> CRITICAL
 */
export function evaluateLiquidityMateriality(
  prevTier: LiquidityTier,
  currTier: LiquidityTier
): { severity: MaterialitySeverity; reason: string } {
  if (prevTier === currTier) {
    return { severity: "NONE", reason: "Liquidity tier unchanged" };
  }

  if (currTier === "RISK") {
    return {
      severity: "CRITICAL",
      reason: "Liquidity degraded to RISK tier (slippage hazard)",
    };
  }
  if (prevTier === "HIGH" && currTier === "MODERATE") {
    return {
      severity: "MATERIAL",
      reason: "Liquidity adjusted from HIGH to MODERATE tier",
    };
  }

  return {
    severity: "INFO",
    reason: `Liquidity tier adjusted from ${prevTier} to ${currTier}`,
  };
}

/**
 * Layer 2: Validation Depth Tier
 */
export function evaluateValidationMateriality(
  prevTier: "THIN" | "DEVELOPING" | "ESTABLISHED",
  currTier: "THIN" | "DEVELOPING" | "ESTABLISHED"
): { severity: MaterialitySeverity; reason: string } {
  if (prevTier === currTier) {
    return { severity: "NONE", reason: "Validation tier unchanged" };
  }
  return {
    severity: "MATERIAL",
    reason: `Validation sample depth evolved from ${prevTier} to ${currTier}`,
  };
}

/**
 * Master Evaluation Function:
 * Compares baseline snapshot against latest state and produces a deterministic DeltaReport.
 * Discards L0 (NONE) noise. Generates AttentionSignal only if severity >= MATERIAL.
 */
export function evaluateMateriality(
  baseline: TickerSnapshot,
  latest: TickerSnapshot
): DeltaReport {
  const items: DeltaItem[] = [];

  // 1. Setup Score
  const scoreResult = evaluateScoreMateriality(baseline.setupScore, latest.setupScore);
  if (scoreResult.severity !== "NONE") {
    items.push({
      field: "setupScore",
      category: "SCORE",
      previousValue: baseline.setupScore,
      currentValue: latest.setupScore,
      deltaDisplay: scoreResult.delta > 0 ? `+${scoreResult.delta} pts` : `${scoreResult.delta} pts`,
      severity: scoreResult.severity,
      reason: `Setup score moved by ${Math.abs(scoreResult.delta)} points`,
    });
  }

  // 2. Execution State
  const stateResult = evaluateExecutionStateMateriality(
    baseline.executionState,
    latest.executionState
  );
  if (stateResult.severity !== "NONE") {
    items.push({
      field: "executionState",
      category: "EXECUTION",
      previousValue: baseline.executionState,
      currentValue: latest.executionState,
      deltaDisplay: `${baseline.executionState} → ${latest.executionState}`,
      severity: stateResult.severity,
      reason: stateResult.reason,
    });
  }

  // 3. Institutional Flow Z-Score
  const flowResult = evaluateFlowMateriality(baseline.flowZScore, latest.flowZScore);
  if (flowResult.severity !== "NONE") {
    items.push({
      field: "flowZScore",
      category: "FLOW",
      previousValue: `${baseline.flowZScore.toFixed(2)}σ`,
      currentValue: `${latest.flowZScore.toFixed(2)}σ`,
      deltaDisplay: flowResult.delta > 0 ? `+${flowResult.delta}σ` : `${flowResult.delta}σ`,
      severity: flowResult.severity,
      reason: `Institutional block flow shifted by ${Math.abs(flowResult.delta)}σ`,
    });
  }

  // 4. Market Regime
  const regimeResult = evaluateRegimeMateriality(baseline.marketRegime, latest.marketRegime);
  if (regimeResult.severity !== "NONE") {
    items.push({
      field: "marketRegime",
      category: "REGIME",
      previousValue: baseline.marketRegime,
      currentValue: latest.marketRegime,
      deltaDisplay: `${baseline.marketRegime} → ${latest.marketRegime}`,
      severity: regimeResult.severity,
      reason: regimeResult.reason,
    });
  }

  // 5. Liquidity Tier
  const liqResult = evaluateLiquidityMateriality(baseline.liquidityTier, latest.liquidityTier);
  if (liqResult.severity !== "NONE") {
    items.push({
      field: "liquidityTier",
      category: "LIQUIDITY",
      previousValue: baseline.liquidityTier,
      currentValue: latest.liquidityTier,
      deltaDisplay: `${baseline.liquidityTier} → ${latest.liquidityTier}`,
      severity: liqResult.severity,
      reason: liqResult.reason,
    });
  }

  // 6. Validation Depth
  const valResult = evaluateValidationMateriality(
    baseline.validationTier,
    latest.validationTier
  );
  if (valResult.severity !== "NONE") {
    items.push({
      field: "validationTier",
      category: "VALIDATION",
      previousValue: baseline.validationTier,
      currentValue: latest.validationTier,
      deltaDisplay: `${baseline.validationTier} → ${latest.validationTier}`,
      severity: valResult.severity,
      reason: valResult.reason,
    });
  }

  // Calculate Max Severity
  let maxSeverity: MaterialitySeverity = "NONE";
  let maxWeight = 0;

  for (const item of items) {
    const weight = SEVERITY_WEIGHTS[item.severity];
    if (weight > maxWeight) {
      maxWeight = weight;
      maxSeverity = item.severity;
    }
  }

  // Calculate Elapsed Days
  const baselineDate = new Date(baseline.timestamp);
  const latestDate = new Date(latest.timestamp);
  const diffMs = Math.max(0, latestDate.getTime() - baselineDate.getTime());
  const daysSinceBaseline = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  const isMaterial = maxSeverity === "MATERIAL" || maxSeverity === "CRITICAL";

  // Build Synthesis Headline
  let headline = "No material thesis changes detected since your last acknowledged baseline.";
  if (items.length > 0) {
    const topItem = items.reduce((prev, curr) =>
      SEVERITY_WEIGHTS[curr.severity] > SEVERITY_WEIGHTS[prev.severity] ? curr : prev
    );
    headline = `Since your review ${daysSinceBaseline === 0 ? "earlier today" : `${daysSinceBaseline} day${daysSinceBaseline > 1 ? "s" : ""} ago`}: ${topItem.reason}.`;
  }

  // Generate Attention Signal for L3/L4 Critical Items
  let attentionSignal: AttentionSignal | undefined;
  if (isMaterial) {
    const criticalItem = items.find((i) => i.severity === "CRITICAL") || items[0];
    attentionSignal = {
      id: `signal-${latest.ticker}-${Date.now()}`,
      ticker: latest.ticker,
      severity: criticalItem.severity === "CRITICAL" ? "CRITICAL" : "MATERIAL",
      category: criticalItem.category,
      headline: criticalItem.reason,
      rationale: `Baseline established on ${baseline.timestamp.slice(0, 10)}. Current spot price is $${latest.spotPrice.toFixed(2)}.`,
      timestamp: latest.timestamp,
    };
  }

  return {
    ticker: latest.ticker,
    baselineSnapshotId: baseline.snapshotId,
    baselineTimestamp: baseline.timestamp,
    latestSnapshotId: latest.snapshotId,
    latestTimestamp: latest.timestamp,
    daysSinceBaseline,
    items,
    maxSeverity,
    isMaterial,
    headline,
    attentionSignal,
  };
}
