/**
 * ARX Terminal vNext - Outcome Intelligence & Learning Loop Contracts
 * Source: docs/architecture/OUTCOME_INTELLIGENCE_AND_LEARNING_LOOP.md
 */

export type OutcomeClass =
  | "SUCCESS"
  | "PARTIAL_SUCCESS"
  | "FAILURE"
  | "EXPIRED"
  | "INVALIDATED";

export const OutcomeClass = {
  SUCCESS: "SUCCESS" as const,
  PARTIAL_SUCCESS: "PARTIAL_SUCCESS" as const,
  FAILURE: "FAILURE" as const,
  EXPIRED: "EXPIRED" as const,
  INVALIDATED: "INVALIDATED" as const,
};

export type AttributionCategory =
  | "EXECUTION_SUCCESS"
  | "EXECUTION_FAILURE"
  | "REGIME_CHANGE"
  | "FLOW_DECAY"
  | "VALIDATION_BREAKDOWN"
  | "STOP_TRIGGERED"
  | "TARGET_REACHED"
  | "THESIS_EXPIRED";

export const AttributionCategory = {
  EXECUTION_SUCCESS: "EXECUTION_SUCCESS" as const,
  EXECUTION_FAILURE: "EXECUTION_FAILURE" as const,
  REGIME_CHANGE: "REGIME_CHANGE" as const,
  FLOW_DECAY: "FLOW_DECAY" as const,
  VALIDATION_BREAKDOWN: "VALIDATION_BREAKDOWN" as const,
  STOP_TRIGGERED: "STOP_TRIGGERED" as const,
  TARGET_REACHED: "TARGET_REACHED" as const,
  THESIS_EXPIRED: "THESIS_EXPIRED" as const,
};

export interface OutcomeRecord {
  outcomeId: string;
  predictionId: string;
  ticker: string;
  predictedAt: string;
  resolvedAt: string;
  outcomeClass: OutcomeClass;
  attributionCategory: AttributionCategory;
  explanation: string;
  outcomeReturnPct?: number;
  outcomeConfidence: number;
  snapshotHash?: string;
}

export interface AttributionResult {
  attributionId: string;
  predictionId: string;
  primaryDriver: string;
  secondaryDrivers: string[];
  confidence: number;
  createdAt: string;
}

export interface LearningMetric {
  metricId: string;
  period: string;
  signalType: string;
  totalPredictions: number;
  successes: number;
  failures: number;
  winRate: number;
  averageOutcomePct: number;
}

export interface ActionabilityMetric {
  reviewedPredictions: number;
  displayedPredictions: number;
  par: number; // reviewed / displayed
  target: number; // 0.50 (50%)
  status: "PASS" | "FAIL";
}

export interface OutcomeSummary {
  totalReviewed: number;
  successCount: number;
  partialSuccessCount: number;
  failureCount: number;
  expiredCount: number;
  successRate: number;
  partialSuccessRate: number;
  failureRate: number;
  expiredRate: number;
  topDrivers: Array<{ driver: string; winRate: number }>;
  failureDrivers: Array<{ driver: string; percentage: number }>;
  actionability: ActionabilityMetric;
}

/**
 * Validates outcome record before storage (INV-O2).
 * Strictly requires populated attributionCategory and non-empty explanation.
 */
export function validateOutcomeRecord(o: Partial<OutcomeRecord>): OutcomeRecord {
  if (!o.predictionId || o.predictionId.trim() === "") {
    throw new Error("PREDICTION_ID_REQUIRED");
  }
  if (!o.outcomeClass || !Object.values(OutcomeClass).includes(o.outcomeClass)) {
    throw new Error("INVALID_OUTCOME_CLASS");
  }
  if (!o.attributionCategory || !Object.values(AttributionCategory).includes(o.attributionCategory)) {
    throw new Error("ATTRIBUTION_CATEGORY_REQUIRED");
  }
  if (!o.explanation || o.explanation.trim() === "") {
    throw new Error("EXPLANATION_REQUIRED");
  }

  return {
    outcomeId: o.outcomeId || `out-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
    predictionId: o.predictionId,
    ticker: o.ticker || "UNKNOWN",
    predictedAt: o.predictedAt || new Date().toISOString(),
    resolvedAt: o.resolvedAt || new Date().toISOString(),
    outcomeClass: o.outcomeClass,
    attributionCategory: o.attributionCategory,
    explanation: o.explanation,
    outcomeReturnPct: o.outcomeReturnPct ?? 0,
    outcomeConfidence: o.outcomeConfidence ?? 0.8,
    snapshotHash: o.snapshotHash,
  };
}
