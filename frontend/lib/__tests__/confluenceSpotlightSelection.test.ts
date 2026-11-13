/** Production Spotlight's exact qualification and ranking functions. */
import { describe, it, expect } from "vitest";
import {
  isQualifiedSpotlightSetup,
  rankSpotlightCandidates,
} from "../../components/WeeklyConfluenceSpotlight";
import type { TradeSetupSpec } from "../simulation/governorSizingEngine";

function fixture(
  ticker: string,
  score: number,
  pillars: number | undefined,
  actionable = true,
): TradeSetupSpec {
  return {
    ticker,
    setupName: "Verified analytic fixture",
    confluenceScore: score,
    confluenceVerifiedPillarCount: pillars,
    decisionState: actionable ? "ACTIONABLE_SETUP" : "EVIDENCE_INCOMPLETE",
    isActionable: actionable,
    entryPivot: 100,
    stopLoss: 95,
    target1: 110,
    analysisReferencePrice: 100,
  };
}

function actualQualifiedCardSymbols(setups: TradeSetupSpec[]): string[] {
  return rankSpotlightCandidates(
    setups.filter(isQualifiedSpotlightSetup).map(setup => ({
      setup,
      confScore: setup.confluenceScore!,
    })),
  ).map(x => x.setup.ticker);
}

describe("WeeklyConfluenceSpotlight — authoritative selection", () => {
  const a = fixture("AAPL", 90, 4);
  const n = fixture("NVDA", 88, 4);
  const m = fixture("MSFT", 88, 4);
  const t = fixture("TSLA", 84, 3);
  const missing = fixture("UNVERIFIED", 99, undefined);
  const insufficient = fixture("INCOMPLETE", 99, 2);
  const rejected = fixture("REJECTED", 99, 4, false);

  it("shows zero cards for missing or suppressed candidates", () => {
    expect(actualQualifiedCardSymbols([missing, insufficient, rejected])).toEqual([]);
  });
  it("shows exactly one legitimate candidate", () => {
    expect(actualQualifiedCardSymbols([n, missing])).toEqual(["NVDA"]);
  });
  it("shows two valid cards, no synthetic third", () => {
    expect(actualQualifiedCardSymbols([m, n, rejected])).toEqual(["MSFT", "NVDA"]);
  });
  it("caps N actual qualifying candidates to three ranked cards", () => {
    expect(actualQualifiedCardSymbols([t, m, a, n, missing])).toEqual(["AAPL", "MSFT", "NVDA"]);
  });
  it("rejects absent, fractional, over-four, nonfinite and negative pillar evidence", () => {
    for (const pillars of [undefined, -1, 2, 3.5, 5, Number.NaN]) {
      expect(isQualifiedSpotlightSetup(fixture("TEST", 90, pillars))).toBe(false);
    }
  });
  it("rejects nonfinite or zero confidence and nonactionable decision state", () => {
    for (const value of [0, Number.NaN, Infinity]) {
      expect(isQualifiedSpotlightSetup(fixture("TEST", value, 4))).toBe(false);
    }
    expect(isQualifiedSpotlightSetup(rejected)).toBe(false);
  });
});
