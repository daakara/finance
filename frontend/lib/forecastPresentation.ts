/**
 * ARCH-001: Presentation boundary for legacy forecast-like arithmetic.
 *
 * The API currently supplies deterministic heuristics, not validated
 * Monte Carlo/GARCH percentile forecasts. Preserve the original values
 * for transparent research display without inventing quantile semantics.
 * Do NOT use this view model for execution eligibility or position sizing.
 */
export interface LegacyReturnHeuristicInput {
  p10Pessimistic: number;
  p50Expected: number;
  p90Optimistic: number;
  annualizedVolatility: number;
  forecastHorizonDays: number;
}

export interface UncalibratedReturnHeuristics {
  kind: "UNCALIBRATED_HEURISTICS";
  usableForExecution: false;
  estimatedDownside: number;
  recentMoveEstimate: number;
  volatilityUpside: number;
  annualizedVolatility: number;
}

export function presentLegacyReturnHeuristics(
  value: LegacyReturnHeuristicInput | null | undefined,
): UncalibratedReturnHeuristics | null {
  if (!value) return null;
  const values = [
    value.p10Pessimistic,
    value.p50Expected,
    value.p90Optimistic,
    value.annualizedVolatility,
  ];
  if (values.some((v) => typeof v !== "number" || !Number.isFinite(v))) return null;
  if (value.annualizedVolatility < 0) return null;
  // The legacy bands have no calibrated confidence interpretation and do
  // not explicitly scale volatility to forecastHorizonDays.
  return {
    kind: "UNCALIBRATED_HEURISTICS",
    usableForExecution: false,
    estimatedDownside: value.p10Pessimistic,
    recentMoveEstimate: value.p50Expected,
    volatilityUpside: value.p90Optimistic,
    annualizedVolatility: value.annualizedVolatility,
  };
}
