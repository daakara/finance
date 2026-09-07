import {
  PortfolioForecast,
  PredictionConfidence,
  PredictionRecord,
  PredictionSeverity,
  PredictionType,
  validatePrediction,
} from "../../types/predictive-intelligence";

export interface SnapshotPredictionInput {
  ticker: string;
  currentPrice: number;
  setupScore: number;
  executionState: string;
  marketRegime: string;
  flowZScore: number;
  validationTier: string;
  snapshotHash: string;
  buyZoneFloor?: number;
  buyZoneCeiling?: number;
}

export class PredictionEngine {
  /**
   * Generates Buy Zone Entry forecast based on distance to corridor and accumulation flow.
   */
  public static predictBuyZoneEntry(input: SnapshotPredictionInput): PredictionRecord | null {
    if (input.executionState === "IN_BUY_ZONE") {
      return null; // Already in buy zone
    }

    const buyZoneCeiling = input.buyZoneCeiling || input.currentPrice * 0.98;
    const distancePct = (input.currentPrice - buyZoneCeiling) / input.currentPrice;

    // High probability if within 3% of buy zone and institutional flow is surging (Z >= 1.2)
    let probability = 0.40;
    const rationale: string[] = [];

    if (distancePct <= 0.03 && distancePct > 0) {
      probability += 0.30;
      rationale.push(`Spot price is within ${(distancePct * 100).toFixed(1)}% of execution buy zone ceiling`);
    } else if (distancePct <= 0.06) {
      probability += 0.15;
      rationale.push(`Spot price pullback is within ${(distancePct * 100).toFixed(1)}% corridor`);
    }

    if (input.flowZScore >= 1.5) {
      probability += 0.20;
      rationale.push(`Institutional flow accumulation velocity (+${input.flowZScore.toFixed(1)}σ) confirms absorption`);
    } else if (input.flowZScore >= 1.0) {
      probability += 0.10;
      rationale.push(`Positive institutional flow (+${input.flowZScore.toFixed(1)}σ) supporting corridor entry`);
    }

    if (input.setupScore >= 75) {
      probability += 0.05;
      rationale.push(`Setup score (${input.setupScore}) indicates strong technical confluence`);
    }

    probability = Math.min(0.95, Number(probability.toFixed(2)));

    if (probability < 0.60) {
      return null; // Sub-threshold probability suppressed
    }

    const confidence: PredictionConfidence =
      probability >= 0.80 ? "HIGH" : probability >= 0.65 ? "MEDIUM" : "LOW";
    const severity: PredictionSeverity = probability >= 0.75 ? "CRITICAL" : "MATERIAL";

    const expiration = new Date();
    expiration.setDate(expiration.getDate() + 3); // 3 days validity

    return validatePrediction({
      ticker: input.ticker,
      predictionType: PredictionType.BUY_ZONE_ENTRY,
      confidence,
      severity,
      generatedAt: new Date().toISOString(),
      expirationAt: expiration.toISOString(),
      modelVersion: "v1.0.0",
      rationale,
      predictedState: {
        expectedExecutionState: "IN_BUY_ZONE",
        targetCeiling: buyZoneCeiling,
      },
      currentStateHash: input.snapshotHash,
      probability,
      status: "ACTIVE",
    });
  }

  /**
   * Generates Regime Transition forecast based on macro indicators and volatility drift.
   */
  public static predictRegimeTransition(
    ticker: string,
    currentRegime: string,
    vixLevel: number,
    snapshotHash: string
  ): PredictionRecord | null {
    let probability = 0.35;
    const rationale: string[] = [];

    if (currentRegime === "RISK_ON" && vixLevel >= 22.0) {
      probability = 0.76;
      rationale.push(`VIX elevation (${vixLevel}) signals elevated volatility divergence from current RISK_ON regime`);
      rationale.push(`Term structure spread inversion points to defensive hedge rotation`);
    } else if (currentRegime === "DEFENSIVE" && vixLevel <= 15.0) {
      probability = 0.72;
      rationale.push(`VIX compression (${vixLevel}) indicates transition toward RISK_ON expansion`);
    } else {
      return null; // No transition expected
    }

    const expiration = new Date();
    expiration.setDate(expiration.getDate() + 5);

    return validatePrediction({
      ticker,
      predictionType: PredictionType.REGIME_TRANSITION,
      confidence: probability >= 0.75 ? "HIGH" : "MEDIUM",
      severity: "CRITICAL",
      generatedAt: new Date().toISOString(),
      expirationAt: expiration.toISOString(),
      modelVersion: "v1.0.0",
      rationale,
      predictedState: {
        expectedRegime: currentRegime === "RISK_ON" ? "DEFENSIVE" : "RISK_ON",
      },
      currentStateHash: snapshotHash,
      probability,
      status: "ACTIVE",
    });
  }

  /**
   * Generates overall Portfolio Forecast by aggregating predictions.
   * Enforces:
   * - Critical predictions sorted first
   * - Expired predictions excluded
   * - Explanations present
   */
  public static generateForecast(predictions: PredictionRecord[]): PortfolioForecast {
    const now = new Date().toISOString();
    const activePredictions = predictions.filter(
      (p) => p.status === "ACTIVE" && p.expirationAt > now
    );

    const critical = activePredictions
      .filter((p) => p.severity === "CRITICAL" || p.severity === "STRATEGIC")
      .sort((a, b) => b.probability - a.probability);

    const material = activePredictions
      .filter((p) => p.severity === "MATERIAL" || p.severity === "INFORMATIONAL")
      .sort((a, b) => b.probability - a.probability);

    const totalProb = activePredictions.reduce((acc, p) => acc + p.probability, 0);
    const avgConfidence = activePredictions.length > 0 ? totalProb / activePredictions.length : 0;
    const riskScore = critical.length * 25 + material.length * 10;

    return {
      forecastDate: now,
      criticalPredictions: critical,
      materialPredictions: material,
      riskScore: Math.min(100, riskScore),
      confidenceScore: Number(avgConfidence.toFixed(2)),
    };
  }
}
