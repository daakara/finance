/**
 * ARX Terminal vNext - Predictive Intelligence & Calibration Governance Contracts
 * Source: docs/architecture/PREDICTIVE_INTELLIGENCE_AND_CALIBRATION_GOVERNANCE.md
 */

export type PredictionConfidence = "LOW" | "MEDIUM" | "HIGH" | "VERY_HIGH";
export const PredictionConfidence = {
  LOW: "LOW" as const,
  MEDIUM: "MEDIUM" as const,
  HIGH: "HIGH" as const,
  VERY_HIGH: "VERY_HIGH" as const,
};

export type PredictionSeverity = "INFORMATIONAL" | "MATERIAL" | "CRITICAL" | "STRATEGIC";
export const PredictionSeverity = {
  INFORMATIONAL: "INFORMATIONAL" as const,
  MATERIAL: "MATERIAL" as const,
  CRITICAL: "CRITICAL" as const,
  STRATEGIC: "STRATEGIC" as const,
};

export type PredictionType =
  | "BUY_ZONE_ENTRY"
  | "REGIME_TRANSITION"
  | "TARGET_REACH"
  | "STOP_TRIGGER"
  | "VALIDATION_PROMOTION"
  | "FLOW_ACCELERATION"
  | "ATTENTION_REQUIRED";

export const PredictionType = {
  BUY_ZONE_ENTRY: "BUY_ZONE_ENTRY" as const,
  REGIME_TRANSITION: "REGIME_TRANSITION" as const,
  TARGET_REACH: "TARGET_REACH" as const,
  STOP_TRIGGER: "STOP_TRIGGER" as const,
  VALIDATION_PROMOTION: "VALIDATION_PROMOTION" as const,
  FLOW_ACCELERATION: "FLOW_ACCELERATION" as const,
  ATTENTION_REQUIRED: "ATTENTION_REQUIRED" as const,
};

export type PredictionStatus = "ACTIVE" | "CONFIRMED" | "EXPIRED" | "INVALIDATED";
export const PredictionStatus = {
  ACTIVE: "ACTIVE" as const,
  CONFIRMED: "CONFIRMED" as const,
  EXPIRED: "EXPIRED" as const,
  INVALIDATED: "INVALIDATED" as const,
};

export interface PredictionRecord {
  predictionId: string;
  ticker: string;
  predictionType: PredictionType;
  confidence: PredictionConfidence;
  severity: PredictionSeverity;
  generatedAt: string;
  expirationAt: string;
  modelVersion: string;
  rationale: string[];
  predictedState: Record<string, unknown>;
  currentStateHash: string;
  probability: number;
  status: PredictionStatus;
}

export type OutcomeResult = "CORRECT" | "INCORRECT" | "PARTIAL";
export const OutcomeResult = {
  CORRECT: "CORRECT" as const,
  INCORRECT: "INCORRECT" as const,
  PARTIAL: "PARTIAL" as const,
};

export interface PredictionOutcome {
  predictionId: string;
  evaluatedAt: string;
  result: OutcomeResult;
  actualState: Record<string, unknown>;
  predictionError?: string;
}

export interface PortfolioForecast {
  forecastDate: string;
  criticalPredictions: PredictionRecord[];
  materialPredictions: PredictionRecord[];
  riskScore: number;
  confidenceScore: number;
}

export interface CalibrationBucket {
  rangeMin: number;
  rangeMax: number;
  predictionCount: number;
  actualSuccessRate: number;
}

export interface CalibrationReport {
  ece: number;
  brierScore: number;
  samples: number;
  buckets: CalibrationBucket[];
  status: "PASS" | "FAIL";
}

export interface DriftMetric {
  metricName: string;
  warningThreshold: number;
  criticalThreshold: number;
  currentShiftPct: number;
  status: "STABLE" | "WARNING" | "CRITICAL";
}

export interface DriftReport {
  generatedAt: string;
  overallStatus: "STABLE" | "WARNING" | "CRITICAL";
  metrics: DriftMetric[];
}

export type ModelStatus = "REGISTERED" | "ACTIVE" | "ROLLED_BACK" | "DEPRECATED";
export const ModelStatus = {
  REGISTERED: "REGISTERED" as const,
  ACTIVE: "ACTIVE" as const,
  ROLLED_BACK: "ROLLED_BACK" as const,
  DEPRECATED: "DEPRECATED" as const,
};

export interface ModelMetadata {
  modelId: string;
  modelName: string;
  version: string;
  checksum: string;
  registeredAt: string;
  activatedAt?: string;
  status: ModelStatus;
  calibration?: CalibrationReport;
}

/**
 * Validates prediction against mandatory Sprint 6 invariants:
 * INV-P1: expirationAt required
 * INV-P2: 0.0 <= probability <= 1.0
 * INV-P3: status != CONFIRMED at creation
 * INV-P5: currentStateHash required
 * INV-P6: rationale.length > 0
 */
export function validatePrediction(p: Partial<PredictionRecord>): PredictionRecord {
  if (p.probability === undefined || p.probability < 0.0 || p.probability > 1.0) {
    throw new Error("INVALID_PROBABILITY");
  }
  if (!p.expirationAt || p.expirationAt.trim() === "") {
    throw new Error("EXPIRATION_REQUIRED");
  }
  if (!p.rationale || p.rationale.length === 0) {
    throw new Error("RATIONALE_REQUIRED");
  }
  if (!p.currentStateHash || p.currentStateHash.trim() === "") {
    throw new Error("SNAPSHOT_HASH_REQUIRED");
  }
  if (p.status === "CONFIRMED") {
    throw new Error("PREDICTION_CANNOT_BE_CONFIRMED_AT_CREATION");
  }

  return {
    predictionId: p.predictionId || `pred-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
    ticker: p.ticker || "UNKNOWN",
    predictionType: p.predictionType || "ATTENTION_REQUIRED",
    confidence: p.confidence || "MEDIUM",
    severity: p.severity || "MATERIAL",
    generatedAt: p.generatedAt || new Date().toISOString(),
    expirationAt: p.expirationAt,
    modelVersion: p.modelVersion || "v1.0.0",
    rationale: p.rationale,
    predictedState: p.predictedState || {},
    currentStateHash: p.currentStateHash,
    probability: p.probability,
    status: p.status || "ACTIVE",
  };
}

/**
 * Validates outcome record before storage (INV-P4).
 */
export function validateOutcome(o: Partial<PredictionOutcome>): PredictionOutcome {
  if (!o.predictionId) {
    throw new Error("PREDICTION_ID_REQUIRED");
  }
  if (!o.result || !["CORRECT", "INCORRECT", "PARTIAL"].includes(o.result)) {
    throw new Error("INVALID_OUTCOME_RESULT");
  }

  return {
    predictionId: o.predictionId,
    evaluatedAt: o.evaluatedAt || new Date().toISOString(),
    result: o.result,
    actualState: o.actualState || {},
    predictionError: o.predictionError,
  };
}
