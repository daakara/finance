import { PredictionRecord } from "../../types/predictive-intelligence";
import {
  AttributionCategory,
  OutcomeClass,
  OutcomeRecord,
  validateOutcomeRecord,
} from "../../types/outcome-intelligence";

export interface MarketObservation {
  targetReached?: boolean;
  stopBreached?: boolean;
  isExpired?: boolean;
  regimeShifted?: boolean;
  isPartialSuccess?: boolean;
  newRegime?: string;
  returnPct?: number;
  observedConfidence?: number;
}

export class OutcomeResolver {
  /**
   * Resolves a prediction hypothesis into an immutable OutcomeRecord.
   * Enforces AC-OI-01, AC-OI-02, AC-OI-03, AC-OI-04.
   */
  public static resolveOutcome(
    prediction: PredictionRecord,
    observation: MarketObservation
  ): OutcomeRecord {
    let outcomeClass: OutcomeClass = OutcomeClass.SUCCESS;
    let attributionCategory: AttributionCategory = AttributionCategory.TARGET_REACHED;
    let explanation = "";

    if (observation.stopBreached) {
      outcomeClass = OutcomeClass.FAILURE;
      attributionCategory = AttributionCategory.STOP_TRIGGERED;
      explanation = "Stop loss corridor breached due to adverse price volatility";
    } else if (observation.regimeShifted) {
      outcomeClass = OutcomeClass.INVALIDATED;
      attributionCategory = AttributionCategory.REGIME_CHANGE;
      explanation = `Market regime transitioned to ${observation.newRegime || "DEFENSIVE"}, invalidating core thesis assumptions`;
    } else if (observation.targetReached) {
      outcomeClass = OutcomeClass.SUCCESS;
      attributionCategory = AttributionCategory.TARGET_REACHED;
      explanation = "Target 1 price objective reached with technical confluence and institutional flow support";
    } else if (observation.isExpired) {
      outcomeClass = OutcomeClass.EXPIRED;
      attributionCategory = AttributionCategory.THESIS_EXPIRED;
      explanation = "Observation window elapsed without target objective or stop loss being triggered";
    } else if (observation.isPartialSuccess) {
      outcomeClass = OutcomeClass.PARTIAL_SUCCESS;
      attributionCategory = AttributionCategory.EXECUTION_SUCCESS;
      explanation = "Partial target objective reached within execution corridor";
    } else {
      outcomeClass = OutcomeClass.EXPIRED;
      attributionCategory = AttributionCategory.THESIS_EXPIRED;
      explanation = "Outcome unobserved: neither target, stop, nor expiration conditions were captured";
    }

    return validateOutcomeRecord({
      predictionId: prediction.predictionId,
      ticker: prediction.ticker,
      predictedAt: prediction.generatedAt,
      resolvedAt: new Date().toISOString(),
      outcomeClass,
      attributionCategory,
      explanation,
      outcomeReturnPct: observation.returnPct,
      outcomeConfidence: observation.observedConfidence ?? 0.0,
      snapshotHash: prediction.currentStateHash,
    });
  }
}
