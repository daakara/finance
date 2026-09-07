import {
  OutcomeResult,
  PredictionOutcome,
  PredictionRecord,
  validateOutcome,
} from "../../types/predictive-intelligence";

export class PredictionRepository {
  private static instance: PredictionRepository;
  private predictions: Map<string, PredictionRecord> = new Map();
  private outcomes: Map<string, PredictionOutcome> = new Map();

  private constructor() {}

  public static getInstance(): PredictionRepository {
    if (!PredictionRepository.instance) {
      PredictionRepository.instance = new PredictionRepository();
    }
    return PredictionRepository.instance;
  }

  public reset(): void {
    this.predictions.clear();
    this.outcomes.clear();
  }

  public savePrediction(prediction: PredictionRecord): void {
    this.predictions.set(prediction.predictionId, prediction);
  }

  public getPrediction(predictionId: string): PredictionRecord | null {
    return this.predictions.get(predictionId) || null;
  }

  public getPredictions(ticker?: string): PredictionRecord[] {
    const all = Array.from(this.predictions.values());
    if (!ticker) return all;
    return all.filter((p) => p.ticker.toUpperCase() === ticker.toUpperCase());
  }

  public getActivePredictions(): PredictionRecord[] {
    const now = new Date().toISOString();
    return Array.from(this.predictions.values()).filter(
      (p) => p.status === "ACTIVE" && p.expirationAt > now
    );
  }

  public acknowledgePrediction(predictionId: string): PredictionRecord {
    const p = this.predictions.get(predictionId);
    if (!p) throw new Error("PREDICTION_NOT_FOUND");
    return p;
  }

  public confirmPrediction(prediction: PredictionRecord): PredictionRecord {
    if (prediction.status === "EXPIRED") {
      throw new Error("INVALID_PREDICTION_STATE");
    }
    prediction.status = "CONFIRMED";
    this.predictions.set(prediction.predictionId, prediction);
    return prediction;
  }

  /**
   * Stores evaluated outcome. Enforces:
   * - Outcomes are unique per prediction (cannot be evaluated twice)
   * - Outcomes are append-only and immutable (INV-P4)
   */
  public storeOutcome(outcome: PredictionOutcome): void {
    const validated = validateOutcome(outcome);
    if (this.outcomes.has(validated.predictionId)) {
      throw new Error("OUTCOME_ALREADY_EXISTS");
    }
    this.outcomes.set(validated.predictionId, validated);
  }

  public updateOutcome(predictionId: string): void {
    throw new Error("OUTCOME_IMMUTABLE");
  }

  public getOutcome(predictionId: string): PredictionOutcome | null {
    return this.outcomes.get(predictionId) || null;
  }

  public evaluatePrediction(
    prediction: PredictionRecord,
    actualState: Record<string, unknown>
  ): PredictionOutcome {
    const isCorrect =
      JSON.stringify(prediction.predictedState) === JSON.stringify(actualState) ||
      (actualState.executionState && actualState.executionState === prediction.predictedState.expectedExecutionState) ||
      (actualState.marketRegime && actualState.marketRegime === prediction.predictedState.expectedRegime);

    const result: OutcomeResult = isCorrect ? OutcomeResult.CORRECT : OutcomeResult.INCORRECT;

    const outcome: PredictionOutcome = {
      predictionId: prediction.predictionId,
      evaluatedAt: new Date().toISOString(),
      result,
      actualState,
      predictionError: isCorrect ? undefined : "Observed state deviated from projected values",
    };

    return outcome;
  }
}
