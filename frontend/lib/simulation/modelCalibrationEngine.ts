/**
 * Horizon 4: Model Calibration Engine (M17)
 * 
 * Enforces Invariants:
 * - INV-OI71: Model Calibration Accuracy (Historical backtesting calibrates edge weights & confidence with bounded MAE)
 */

import {
  CalibrationResult,
  BacktestObservation,
  INV_OI71,
} from '../../types/simulation-digital-twin';

// -------------------------------------------------------------
// HISTORICAL BACKTESTING DATASETS
// -------------------------------------------------------------

export const CANONICAL_BACKTEST_OBSERVATIONS: BacktestObservation[] = [
  { decisionId: 'DEC-2025-Q1-01', interventionType: 'TRAINING_BUDGET', predictedDelta: 4.0, actualDelta: 3.5, error: -0.5, observedAtUtc: '2025-03-31T00:00:00Z' },
  { decisionId: 'DEC-2025-Q2-02', interventionType: 'TRAINING_BUDGET', predictedDelta: 3.8, actualDelta: 3.6, error: -0.2, observedAtUtc: '2025-06-30T00:00:00Z' },
  { decisionId: 'DEC-2025-Q3-03', interventionType: 'TRAINING_BUDGET', predictedDelta: 5.2, actualDelta: 4.4, error: -0.8, observedAtUtc: '2025-09-30T00:00:00Z' },
  { decisionId: 'DEC-2025-Q4-04', interventionType: 'TRAINING_BUDGET', predictedDelta: 4.5, actualDelta: 4.2, error: -0.3, observedAtUtc: '2025-12-31T00:00:00Z' },
  { decisionId: 'DEC-2026-Q1-05', interventionType: 'GOVERNANCE_AUTOMATION', predictedDelta: 3.0, actualDelta: 3.2, error: 0.2, observedAtUtc: '2026-03-31T00:00:00Z' },
  { decisionId: 'DEC-2026-Q2-06', interventionType: 'GOVERNANCE_AUTOMATION', predictedDelta: 2.8, actualDelta: 2.9, error: 0.1, observedAtUtc: '2026-06-30T00:00:00Z' },
  { decisionId: 'DEC-2026-Q1-07', interventionType: 'INFRASTRUCTURE_RESILIENCE', predictedDelta: 5.0, actualDelta: 4.8, error: -0.2, observedAtUtc: '2026-03-31T00:00:00Z' },
  { decisionId: 'DEC-2026-Q2-08', interventionType: 'CAPITAL_FREEZE', predictedDelta: -2.0, actualDelta: -1.9, error: 0.1, observedAtUtc: '2026-06-30T00:00:00Z' },
];

/**
 * Calculates Mean Absolute Error (MAE) across backtest observations:
 * MAE = (1 / N) * sum(|Actual - Predicted|)
 */
export function calculateMAE(observations: BacktestObservation[]): number {
  if (observations.length === 0) return 0;
  const sum = observations.reduce((acc, obs) => acc + Math.abs(obs.actualDelta - obs.predictedDelta), 0);
  return Number((sum / observations.length).toFixed(3));
}

/**
 * Calculates Root Mean Squared Error (RMSE):
 * RMSE = sqrt((1 / N) * sum((Actual - Predicted)^2))
 */
export function calculateRMSE(observations: BacktestObservation[]): number {
  if (observations.length === 0) return 0;
  const sumSquares = observations.reduce((acc, obs) => acc + Math.pow(obs.actualDelta - obs.predictedDelta, 2), 0);
  return Number(Math.sqrt(sumSquares / observations.length).toFixed(3));
}

/**
 * Calculates Prediction Bias:
 * Bias = (1 / N) * sum(Actual - Predicted)
 * Negative indicates model overestimation; positive indicates underestimation.
 */
export function calculatePredictionBias(observations: BacktestObservation[]): number {
  if (observations.length === 0) return 0;
  const sum = observations.reduce((acc, obs) => acc + (obs.actualDelta - obs.predictedDelta), 0);
  return Number((sum / observations.length).toFixed(3));
}

/**
 * Calibrates edge weights based on historical bias and empirical transmission.
 */
export function recalibrateEdgeWeight(
  priorWeight: number,
  bias: number,
  learningRate = 0.2
): number {
  // If model is overestimating (bias < 0), adjust weight downward
  const adjustment = bias * learningRate;
  const updated = priorWeight + adjustment;
  return Number(Math.max(0.05, Math.min(2.0, updated)).toFixed(3));
}

/**
 * Calibrates edge confidence based on historical empirical accuracy hit rate.
 */
export function recalibrateEdgeConfidence(
  observations: BacktestObservation[],
  tolerance = 0.5
): number {
  if (observations.length === 0) return 90.0;
  const hits = observations.filter((obs) => Math.abs(obs.actualDelta - obs.predictedDelta) <= tolerance).length;
  const rate = (hits / observations.length) * 100;
  return Number(Math.max(50, Math.min(99, rate)).toFixed(1));
}

/**
 * Computes full calibration results for canonical model metrics.
 */
export function performModelCalibration(
  customObservations?: BacktestObservation[]
): CalibrationResult[] {
  const dataset = customObservations || CANONICAL_BACKTEST_OBSERVATIONS;

  const trainingObs = dataset.filter((o) => o.interventionType === 'TRAINING_BUDGET');
  const govObs = dataset.filter((o) => o.interventionType === 'GOVERNANCE_AUTOMATION');
  const overallObs = dataset;

  const results: CalibrationResult[] = [
    {
      metricId: 'TRAINING_BUDGET->LEARNING_VELOCITY',
      previousWeight: 0.80,
      calibratedWeight: recalibrateEdgeWeight(0.80, calculatePredictionBias(trainingObs)),
      weightDelta: Number((recalibrateEdgeWeight(0.80, calculatePredictionBias(trainingObs)) - 0.80).toFixed(3)),
      predictionError: calculateMAE(trainingObs),
      mae: calculateMAE(trainingObs),
      rmse: calculateRMSE(trainingObs),
      predictionBias: calculatePredictionBias(trainingObs),
      confidencePct: recalibrateEdgeConfidence(trainingObs, 0.6),
      sampleSize: trainingObs.length,
      calibratedAtUtc: new Date().toISOString(),
    },
    {
      metricId: 'GOVERNANCE_ADHERENCE->DECISION_QUALITY',
      previousWeight: 0.75,
      calibratedWeight: recalibrateEdgeWeight(0.75, calculatePredictionBias(govObs)),
      weightDelta: Number((recalibrateEdgeWeight(0.75, calculatePredictionBias(govObs)) - 0.75).toFixed(3)),
      predictionError: calculateMAE(govObs),
      mae: calculateMAE(govObs),
      rmse: calculateRMSE(govObs),
      predictionBias: calculatePredictionBias(govObs),
      confidencePct: recalibrateEdgeConfidence(govObs, 0.4),
      sampleSize: govObs.length,
      calibratedAtUtc: new Date().toISOString(),
    },
    {
      metricId: 'OVERALL_ORGANIZATIONAL_TWIN',
      previousWeight: 1.0,
      calibratedWeight: recalibrateEdgeWeight(1.0, calculatePredictionBias(overallObs)),
      weightDelta: Number((recalibrateEdgeWeight(1.0, calculatePredictionBias(overallObs)) - 1.0).toFixed(3)),
      predictionError: calculateMAE(overallObs),
      mae: calculateMAE(overallObs),
      rmse: calculateRMSE(overallObs),
      predictionBias: calculatePredictionBias(overallObs),
      confidencePct: recalibrateEdgeConfidence(overallObs, 0.5),
      sampleSize: overallObs.length,
      calibratedAtUtc: new Date().toISOString(),
    },
  ];

  return results;
}

/**
 * Validates Invariant INV-OI71: Model Calibration Accuracy.
 */
export function verifyCalibrationInvariant(result: CalibrationResult): boolean {
  // MAE must be bounded below certification threshold (< 1.5 OHI points)
  return result.mae < 1.5 && Math.abs(result.predictionBias) < 1.0;
}
