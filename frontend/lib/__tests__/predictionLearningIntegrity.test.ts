import { describe, it, expect, beforeEach } from "vitest";
import { CalibrationEngine } from "../prediction/calibrationEngine";
import { OutcomeResolver, MarketObservation } from "../outcome/outcomeResolver";
import { LearningLoopMetrics } from "../outcome/learningLoopMetrics";
import { ModelRegistry } from "../prediction/modelRegistry";
import { RollbackController } from "../prediction/rollbackController";
import { PredictionRepository } from "../prediction/predictionRepository";
import {
  PredictionRecord,
  PredictionOutcome,
  ModelStatus,
} from "../../types/predictive-intelligence";
import {
  OutcomeRecord,
  OutcomeClass,
  AttributionCategory,
} from "../../types/outcome-intelligence";

describe("ARX-LEARNING-INTEGRITY-001 Adversarial Suite (Frontend)", () => {
  const dummyPrediction: PredictionRecord = {
    predictionId: "pred-adv-001",
    ticker: "NVDA",
    predictionType: "BUY_ZONE_ENTRY",
    confidence: "HIGH",
    severity: "CRITICAL",
    generatedAt: "2026-11-01T10:00:00Z",
    expirationAt: "2026-11-04T16:00:00Z",
    modelVersion: "v1.0.0",
    rationale: ["Corridor within 2%", "Surging institutional flow"],
    predictedState: { expectedExecutionState: "IN_BUY_ZONE" },
    currentStateHash: "sha256:advtestdummyhash123",
    probability: 0.85,
    status: "ACTIVE",
  };

  // 1. Empty calibration (N=0) returns FAIL, never PASS
  it("FAIL-CLOSED: Empty calibration (N=0) returns FAIL, never PASS", () => {
    const report = CalibrationEngine.calculateCalibration([]);
    expect(report.samples).toBe(0);
    expect(report.status).toBe("FAIL");
    expect(report.status).not.toBe("PASS");
  });

  // 1b. Small calibration sample (N < 30) returns FAIL even if mathematical error is low
  it("FAIL-CLOSED: Under-sampled calibration (N < 30) returns FAIL due to lack of statistical power", () => {
    const report = CalibrationEngine.calculateCalibration([
      { probability: 0.8, actualOutcome: true },
      { probability: 0.8, actualOutcome: true },
      { probability: 0.2, actualOutcome: false },
    ]);
    expect(report.samples).toBe(3);
    expect(report.status).toBe("FAIL");
  });

  // 2. Missing returns in OutcomeResolver never generate synthetic returns
  it("FAIL-CLOSED: Missing returnPct in MarketObservation does NOT fabricate return percentages", () => {
    const obs: MarketObservation = {
      targetReached: true,
      returnPct: undefined, // Explicitly missing
    };
    const outcome = OutcomeResolver.resolveOutcome(dummyPrediction, obs);
    expect(outcome.outcomeReturnPct).toBeUndefined();
    expect(outcome.outcomeReturnPct).not.toBe(6.8);
    expect(outcome.outcomeReturnPct).not.toBe(0.5);
  });

  // 2b. Missing confidence in OutcomeResolver never generates synthetic 0.85 confidence
  it("FAIL-CLOSED: Missing observedConfidence does NOT fabricate 0.85 confidence", () => {
    const obs: MarketObservation = {
      targetReached: true,
      observedConfidence: undefined,
    };
    const outcome = OutcomeResolver.resolveOutcome(dummyPrediction, obs);
    expect(outcome.outcomeConfidence).toBe(0.0);
    expect(outcome.outcomeConfidence).not.toBe(0.85);
  });

  // 3. Unobserved outcomes never become PARTIAL_SUCCESS
  it("FAIL-CLOSED: Unobserved observation defaults to EXPIRED, never PARTIAL_SUCCESS with fabricated gains", () => {
    const obs: MarketObservation = {
      // No events observed
    };
    const outcome = OutcomeResolver.resolveOutcome(dummyPrediction, obs);
    expect(outcome.outcomeClass).toBe(OutcomeClass.EXPIRED);
    expect(outcome.attributionCategory).toBe(AttributionCategory.THESIS_EXPIRED);
    expect(outcome.outcomeClass).not.toBe(OutcomeClass.PARTIAL_SUCCESS);
    expect(outcome.explanation).toContain("Outcome unobserved");
  });

  // 4. Subthreshold PAR (< 0.5) never gets substituted with 0.62
  it("FAIL-CLOSED: Subthreshold PAR (< 0.5) reports authentic rate and FAIL status, never substituted with 0.62", () => {
    const singleOutcome: OutcomeRecord = {
      outcomeId: "out-par-test-1",
      predictionId: "pred-par-test-1",
      ticker: "AAPL",
      predictedAt: "2026-11-01T10:00:00Z",
      resolvedAt: "2026-11-02T10:00:00Z",
      outcomeClass: "SUCCESS",
      attributionCategory: "TARGET_REACHED",
      explanation: "Target 1 reached",
      outcomeConfidence: 0.9,
    };

    // 1 reviewed out of 100 displayed -> PAR = 0.010 (< 0.50 target)
    const summary = LearningLoopMetrics.computeSummary([singleOutcome], 100);
    expect(summary.actionability.par).toBe(0.01);
    expect(summary.actionability.par).not.toBe(0.62);
    expect(summary.actionability.status).toBe("FAIL");
    expect(summary.actionability.status).not.toBe("PASS");
  });

  // 5. Driver stats dynamically derived, not hardcoded
  it("TRUTHFULNESS: Driver win rates and failure percentages derived dynamically from actual data", () => {
    const outcomes: OutcomeRecord[] = [
      {
        outcomeId: "out-1",
        predictionId: "pred-1",
        ticker: "AAPL",
        predictedAt: "2026-11-01T10:00:00Z",
        resolvedAt: "2026-11-02T10:00:00Z",
        outcomeClass: "SUCCESS",
        attributionCategory: "TARGET_REACHED",
        explanation: "Accumulation",
        outcomeConfidence: 0.9,
      },
      {
        outcomeId: "out-2",
        predictionId: "pred-2",
        ticker: "MSFT",
        predictedAt: "2026-11-01T10:00:00Z",
        resolvedAt: "2026-11-02T10:00:00Z",
        outcomeClass: "FAILURE",
        attributionCategory: "STOP_TRIGGERED",
        explanation: "Stop hit",
        outcomeConfidence: 0.9,
      },
    ];

    const summary = LearningLoopMetrics.computeSummary(outcomes, 10);
    expect(summary.topDrivers.length).toBeGreaterThan(0);
    // Institutional Accumulation: 1 total, 1 win = 1.0 (not hardcoded 0.72)
    expect(summary.topDrivers[0].winRate).toBe(1.0);
    expect(summary.topDrivers[0].winRate).not.toBe(0.72);

    // Stop Triggered: 1 out of 1 failure = 1.0 (not hardcoded 0.21)
    expect(summary.failureDrivers[0].percentage).toBe(1.0);
    expect(summary.failureDrivers[0].percentage).not.toBe(0.21);
  });

  // 6. Inadequate evidence blocks model promotion
  it("GOVERNANCE: Inadequate calibration evidence blocks model promotion", () => {
    const registry = ModelRegistry.getInstance();
    registry.reset();

    const uncalibrated = registry.registerModel({
      modelName: "test-challenger",
      version: "v2.0.0",
      checksum: "sha256:dummychecksumforv2",
      calibration: {
        ece: 0.08, // Exceeds 0.05 limit
        brierScore: 0.20, // Exceeds 0.15 limit
        samples: 50,
        buckets: [],
        status: "FAIL",
      },
    });

    expect(() => registry.promoteModel(uncalibrated.modelId)).toThrow("VALIDATION_REQUIRED");
  });

  // 7. Model rollback is not reported as complete unless the model actually changes
  it("FAIL-CLOSED: Model rollback fails closed if active model cannot change or has no prior version", () => {
    const registry = ModelRegistry.getInstance();
    registry.reset(); // Has only default mdl-v1.0.0

    const controller = RollbackController.getInstance();
    controller.reset();

    // Trying to roll back when already at baseline v1.0.0 without prior version
    const result = controller.executeRollback("HARD");
    expect(result.success).toBe(false);
    expect(result.state).toBe("ROLLBACK_FAILED");
    expect(result.reason).toContain("ROLLBACK_TARGET_NOT_FOUND");
  });

  // 8. Published historical predictions and outcomes remain immutable
  it("IMMUTABILITY: Modifying published historical outcomes is rejected (OUTCOME_IMMUTABLE)", () => {
    const repo = PredictionRepository.getInstance();
    repo.reset();

    repo.savePrediction(dummyPrediction);

    const outcome: PredictionOutcome = {
      predictionId: dummyPrediction.predictionId,
      evaluatedAt: new Date().toISOString(),
      result: "CORRECT",
      actualState: { executionState: "IN_BUY_ZONE" },
    };

    repo.storeOutcome(outcome);

    expect(() => repo.updateOutcome(dummyPrediction.predictionId)).toThrow("OUTCOME_IMMUTABLE");
  });

  // 9. Duplicate outcome evaluations are rejected
  it("INTEGRITY: Duplicate outcome evaluation for same predictionId is rejected", () => {
    const repo = PredictionRepository.getInstance();
    repo.reset();

    repo.savePrediction(dummyPrediction);

    const outcome: PredictionOutcome = {
      predictionId: dummyPrediction.predictionId,
      evaluatedAt: new Date().toISOString(),
      result: "CORRECT",
      actualState: { executionState: "IN_BUY_ZONE" },
    };

    repo.storeOutcome(outcome);
    expect(() => repo.storeOutcome(outcome)).toThrow("OUTCOME_ALREADY_EXISTS");
  });

  // 10. Outcome evaluation requires existing prediction identity
  it("IDENTITY: Outcome evaluation for non-existent predictionId is rejected", () => {
    const repo = PredictionRepository.getInstance();
    repo.reset();

    const orphanOutcome: PredictionOutcome = {
      predictionId: "pred-nonexistent-999",
      evaluatedAt: new Date().toISOString(),
      result: "CORRECT",
      actualState: {},
    };

    expect(() => repo.storeOutcome(orphanOutcome)).toThrow("PREDICTION_NOT_FOUND");
  });
});
