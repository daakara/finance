/**
 * ARX Pure Corridor Location & Geometry Helper (Workstream D — Slice 1).
 *
 * Implements strict geometric location evaluation decoupled from tolerated gate policies:
 * 1. Geometric position on the real line:
 *    - price < lower bound: BELOW
 *    - lower bound <= price <= upper bound: INSIDE (inclusive boundary convention)
 *    - price > upper bound: ABOVE (never described as inside)
 * 2. Extension tolerance qualification:
 *    - If price > upper bound and price <= upper bound * (1 + toleranceRatio),
 *      tolerance applies, but geometric position remains strictly ABOVE.
 * 3. Fail-closed invariants:
 *    - Missing, non-finite, non-positive, or inverted (min > max) bounds strictly yield UNAVAILABLE.
 *    - Never fabricates a location or passes inverted bounds.
 */

export type CorridorSpatialRelation = "BELOW" | "INSIDE" | "ABOVE" | "UNAVAILABLE";

export interface CorridorLocationResult {
  /** Whether price and bounds are valid, finite, positive, and non-inverted */
  isValid: boolean;
  /** Spatial relation of price relative to bounds */
  spatialRelation: CorridorSpatialRelation;
  /** True strictly when price is at or between bounds (inclusive) */
  isInside: boolean;
  /** True strictly when price is above upper bound */
  isAboveCeiling: boolean;
  /** True strictly when price is below lower bound */
  isBelowFloor: boolean;
  /** True strictly when price is above upper bound but within the extension tolerance */
  isWithinTolerance: boolean;
  /** The tolerance ratio evaluated (e.g. 0.02 for 2%) */
  toleranceRatio: number;
  /** The evaluated price, or null if invalid */
  evaluatedPrice: number | null;
  /** The evaluated lower bound, or null if invalid */
  entryMin: number | null;
  /** The evaluated upper bound, or null if invalid */
  entryMax: number | null;
  /** Truthful descriptive explanation of location */
  description: string;
}

/**
 * Pure evaluator for price location relative to an accumulation corridor.
 * Strictly respects inclusive boundary conventions:
 * - price === rawMin: INSIDE
 * - price === rawMax: INSIDE
 * - rawMin > rawMax: UNAVAILABLE (inverted bounds fail closed)
 */
export function evaluateCorridorLocation(
  price?: number | null,
  rawMin?: number | null,
  rawMax?: number | null,
  toleranceRatio: number = 0.02
): CorridorLocationResult {
  const hasValidPrice = typeof price === "number" && Number.isFinite(price) && price > 0;
  const hasValidMin = typeof rawMin === "number" && Number.isFinite(rawMin) && rawMin > 0;
  const hasValidMax = typeof rawMax === "number" && Number.isFinite(rawMax) && rawMax > 0;

  // Inverted bounds invariant: rawMin must not exceed rawMax
  const isInverted = hasValidMin && hasValidMax && rawMin! > rawMax!;

  if (!hasValidPrice || !hasValidMin || !hasValidMax || isInverted) {
    return {
      isValid: false,
      spatialRelation: "UNAVAILABLE",
      isInside: false,
      isAboveCeiling: false,
      isBelowFloor: false,
      isWithinTolerance: false,
      toleranceRatio,
      evaluatedPrice: hasValidPrice ? price! : null,
      entryMin: hasValidMin ? rawMin! : null,
      entryMax: hasValidMax ? rawMax! : null,
      description: isInverted
        ? "Corridor bounds are invalid (inverted bounds: min > max)."
        : "Optimal entry corridor bounds or price are unavailable or non-finite.",
    };
  }

  const p = price!;
  const min = rawMin!;
  const max = rawMax!;
  const toleranceLimit = max * (1 + toleranceRatio);

  if (p < min) {
    return {
      isValid: true,
      spatialRelation: "BELOW",
      isInside: false,
      isAboveCeiling: false,
      isBelowFloor: true,
      isWithinTolerance: false,
      toleranceRatio,
      evaluatedPrice: p,
      entryMin: min,
      entryMax: max,
      description: `Price ($${p.toFixed(2)}) is below the accumulation corridor ($${min.toFixed(2)}–$${max.toFixed(2)}). Awaiting base stabilization.`,
    };
  }

  if (p <= max) {
    return {
      isValid: true,
      spatialRelation: "INSIDE",
      isInside: true,
      isAboveCeiling: false,
      isBelowFloor: false,
      isWithinTolerance: false,
      toleranceRatio,
      evaluatedPrice: p,
      entryMin: min,
      entryMax: max,
      description: `Price ($${p.toFixed(2)}) is positioned inside the institutional accumulation corridor ($${min.toFixed(2)}–$${max.toFixed(2)}).`,
    };
  }

  // Price is above upper bound: never describe as inside
  const isWithinTolerance = p <= toleranceLimit;
  return {
    isValid: true,
    spatialRelation: "ABOVE",
    isInside: false,
    isAboveCeiling: true,
    isBelowFloor: false,
    isWithinTolerance,
    toleranceRatio,
    evaluatedPrice: p,
    entryMin: min,
    entryMax: max,
    description: isWithinTolerance
      ? `Price ($${p.toFixed(2)}) is above the entry corridor ($${min.toFixed(2)}–$${max.toFixed(2)}). Existing implementation tolerance applies.`
      : `Price ($${p.toFixed(2)}) has extended >${Math.round(toleranceRatio * 100)}% past the accumulation corridor ($${min.toFixed(2)}–$${max.toFixed(2)}). Do not chase above resistance.`,
  };
}
