/**
 * Analytical Invariant for Technical Reference Levels and Reclaim Language.
 *
 * Invariant:
 * For any reference level L (e.g. 50D SMA, breakout pivot, EMA20):
 * - price < L: may justify language such as "needs to reclaim", "below L", "awaiting reclaim"
 * - price == L: boundary testing language such as "testing L", "at L"
 * - price >= L: must NOT produce language implying the level remains unreclaimed.
 *   Contextually accurate language: "holding constructively above L", "reclaimed L", "above L".
 * - missing/undefined L or price: honest "unavailable/unassessed", no fabricated certainty.
 */

export interface LevelRelation {
  status: "BELOW" | "AT_LEVEL" | "ABOVE" | "UNAVAILABLE";
  reclaimMilestone: string;
  headlineExplanationWatch: string;
  whatWouldChangeAssessment: string;
  uiBadgeLabel: string;
}

export function evaluateLevelRelation(
  price: number | undefined | null,
  referenceLevel: number | undefined | null,
  levelName: string = "50-Day SMA",
  symbol: string = "Asset"
): LevelRelation {
  if (
    price === undefined ||
    price === null ||
    !Number.isFinite(price) ||
    price <= 0 ||
    referenceLevel === undefined ||
    referenceLevel === null ||
    !Number.isFinite(referenceLevel) ||
    referenceLevel <= 0
  ) {
    return {
      status: "UNAVAILABLE",
      reclaimMilestone: `Reference level data unavailable for ${symbol} (${levelName} not established).`,
      headlineExplanationWatch: `${levelName} reference level unavailable for ${symbol}.`,
      whatWouldChangeAssessment: `Valid ${levelName} level required to evaluate reclaim criteria.`,
      uiBadgeLabel: "Unassessed",
    };
  }

  const delta = price - referenceLevel;
  // Use a small epsilon (0.005) for float equality comparison
  if (Math.abs(delta) < 0.005) {
    return {
      status: "AT_LEVEL",
      reclaimMilestone: `Price ($${price.toFixed(2)}) is testing the ${levelName} ($${referenceLevel.toFixed(2)}). Awaiting decisive volume expansion and breakout confirmation above this pivot.`,
      headlineExplanationWatch: `Testing ${levelName} at $${referenceLevel.toFixed(2)}; awaiting decisive volume confirmation.`,
      whatWouldChangeAssessment: `Decisive expansion and volume-backed hold above $${referenceLevel.toFixed(2)} (${levelName}) would upgrade posture to ACQUIRE.`,
      uiBadgeLabel: "Testing Level",
    };
  }

  if (price < referenceLevel) {
    return {
      status: "BELOW",
      reclaimMilestone: `Price ($${price.toFixed(2)}) is below the ${levelName} ($${referenceLevel.toFixed(2)}). ${symbol} needs to reclaim $${referenceLevel.toFixed(2)} and show strong base formation on higher volume.`,
      headlineExplanationWatch: `Awaiting constructive base confirmation and reclaim of $${referenceLevel.toFixed(2)}.`,
      whatWouldChangeAssessment: `Reclaiming and holding above $${referenceLevel.toFixed(2)} (${levelName}) with volume expansion would upgrade posture to ACQUIRE.`,
      uiBadgeLabel: "Must reclaim",
    };
  }

  // price > referenceLevel
  return {
    status: "ABOVE",
    reclaimMilestone: `Price ($${price.toFixed(2)}) is holding constructively above the ${levelName} ($${referenceLevel.toFixed(2)}). Awaiting upper base consolidation and volume expansion for entry trigger.`,
    headlineExplanationWatch: `Holding constructively above ${levelName} ($${referenceLevel.toFixed(2)}); awaiting consolidation before trigger.`,
    whatWouldChangeAssessment: `Sustained consolidation above $${referenceLevel.toFixed(2)} (${levelName}) with actionable volume trigger would upgrade posture to ACQUIRE.`,
    uiBadgeLabel: "Holding above",
  };
}
