import { describe, expect, it } from "vitest";
import { presentLegacyReturnHeuristics } from "../forecastPresentation";

describe("ARCH-001 legacy return display truth", () => {
  const scenario = {
    p10Pessimistic: -6.4,
    p50Expected: 18.5,
    p90Optimistic: 11.2,
    annualizedVolatility: 10,
    forecastHorizonDays: 90,
  };
  it("exposes nonmonotonic legacy scenarios without false probability claims", () => {
    const output = presentLegacyReturnHeuristics(scenario);
    expect(output).toEqual({
      kind: "UNCALIBRATED_HEURISTICS",
      usableForExecution: false,
      estimatedDownside: -6.4,
      recentMoveEstimate: 18.5,
      volatilityUpside: 11.2,
      annualizedVolatility: 10,
    });
    expect(scenario.p50Expected).toBeGreaterThan(scenario.p90Optimistic);
  });
  it("fails closed for invalid or nonfinite scenario inputs", () => {
    expect(presentLegacyReturnHeuristics(undefined)).toBeNull();
    expect(presentLegacyReturnHeuristics({ ...scenario, p50Expected: NaN })).toBeNull();
    expect(presentLegacyReturnHeuristics({ ...scenario, p90Optimistic: Infinity })).toBeNull();
    expect(presentLegacyReturnHeuristics({ ...scenario, annualizedVolatility: -1 })).toBeNull();
  });
  it("never exposes an execution-authority flag", () => {
    const result = presentLegacyReturnHeuristics(scenario);
    expect(result?.usableForExecution).toBe(false);
    expect(result?.kind).toBe("UNCALIBRATED_HEURISTICS");
  });
});
