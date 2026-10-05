import { QuantitativeInsight } from "../types/insight";

export interface UnmetConditionItem {
  id: string;
  category: "CORRIDOR" | "TRIGGER" | "CONFLUENCE" | "STRUCTURE" | "EVIDENCE";
  title: string;
  description: string;
  status: "MET" | "UNMET" | "PENDING";
}

/**
 * Deterministically derives the conditions required to reach or maintain
 * an actionable state without mutating canonical backend assessment data.
 */
export function deriveUnmetConditions(insight: QuantitativeInsight): UnmetConditionItem[] {
  const conditions: UnmetConditionItem[] = [];
  const kl = insight.standard?.keyLevels;
  const isActionable = Boolean(insight.terminalState?.isActionable);
  const eligibility = insight.terminalState?.overallEligibility;
  const price = insight.price;

  // 1. Spatial Corridor (Location)
  if (kl) {
    let lower: number | undefined = kl.entryMin;
    let upper: number | undefined = kl.entryMax;

    // Fallback: parse canonical watchZone string ("$182.50 – $185.00") if entryMin/Max not explicitly provided
    if ((lower == null || upper == null) && kl.watchZone && kl.watchZone !== "Unavailable") {
      const match = kl.watchZone.match(/\$?([\d.]+)\s*[–-]\s*\$?([\d.]+)/);
      if (match) {
        lower = parseFloat(match[1]);
        upper = parseFloat(match[2]);
      }
    }

    if (lower != null && upper != null && !isNaN(lower) && !isNaN(upper) && lower > 0 && upper > 0) {
      const canonicalLower = Math.min(lower, upper);
      const canonicalUpper = Math.max(lower, upper);
      const isInCorridor = price >= canonicalLower && price <= canonicalUpper;

      if (isInCorridor) {
        conditions.push({
          id: "corridor",
          category: "CORRIDOR",
          title: "Price Inside Accumulation Corridor",
          description: `Current spot ($${price.toFixed(2)}) is positioned within the preferred risk corridor (${kl.watchZone || `$${canonicalLower.toFixed(2)} – $${canonicalUpper.toFixed(2)}`}).`,
          status: "MET",
        });
      } else if (price > canonicalUpper) {
        conditions.push({
          id: "corridor",
          category: "CORRIDOR",
          title: "Pullback into Accumulation Corridor",
          description: `Price ($${price.toFixed(2)}) is extended above preferred entry corridor (${kl.watchZone || `$${canonicalLower.toFixed(2)} – $${canonicalUpper.toFixed(2)}`}). Wait for low-volume pullback.`,
          status: "UNMET",
        });
      } else {
        conditions.push({
          id: "corridor",
          category: "CORRIDOR",
          title: "Reclaim Base Floor",
          description: `Price ($${price.toFixed(2)}) is testing lower bounds below corridor (${kl.watchZone || `$${canonicalLower.toFixed(2)}`}). Requires stabilization before entry.`,
          status: "UNMET",
        });
      }
    } else {
      conditions.push({
        id: "corridor",
        category: "CORRIDOR",
        title: "Accumulation Corridor Unavailable",
        description: "Canonical execution corridor bounds are unavailable for this asset (insufficient trading history or unverified market data).",
        status: "UNMET",
      });
    }
  }

  // 2. Market Event Trigger (Dynamic Event Confirmation)
  if (isActionable) {
    conditions.push({
      id: "event_trigger",
      category: "TRIGGER",
      title: "Market Event Trigger Confirmed",
      description: "Active volume expansion and technical breakout confirmation observed.",
      status: "MET",
    });
  } else {
    conditions.push({
      id: "event_trigger",
      category: "TRIGGER",
      title: "Market Event Trigger Pending",
      description: insight.human?.reclaimMilestone || "Awaiting volume breakout or decisive reversal candle before capital commitment.",
      status: "UNMET",
    });
  }

  // 3. Confluence Score & Risk/Reward Thresholds
  const score = insight.setupScore;
  if (score >= 70) {
    conditions.push({
      id: "confluence_floor",
      category: "CONFLUENCE",
      title: "Confluence Floor Satisfied",
      description: `Setup score ${score}/100 meets institutional confluence threshold (≥ 70).`,
      status: "MET",
    });
  } else {
    conditions.push({
      id: "confluence_floor",
      category: "CONFLUENCE",
      title: "Elevate Multi-Model Confluence",
      description: `Current score ${score}/100 is below the 70/100 threshold required for high-conviction execution.`,
      status: "UNMET",
    });
  }

  // 4. Moving Average Structural Requirement
  if (kl?.sma50 !== undefined) {
    if (price >= kl.sma50) {
      conditions.push({
        id: "ma_reclaim",
        category: "STRUCTURE",
        title: "50-Day Moving Average Support",
        description: `Holding constructively above 50-day moving average ($${kl.sma50.toFixed(2)}).`,
        status: "MET",
      });
    } else {
      conditions.push({
        id: "ma_reclaim",
        category: "STRUCTURE",
        title: "Reclaim 50-Day Moving Average",
        description: `Needs decisive close above 50-day SMA ($${kl.sma50.toFixed(2)}) to neutralize intermediate downtrend.`,
        status: "UNMET",
      });
    }
  }

  // 5. Evidence Availability & Integrity
  if (eligibility === "ELIGIBLE") {
    conditions.push({
      id: "evidence_completeness",
      category: "EVIDENCE",
      title: "Evidence Data Complete",
      description: "All independent quant models received verified inputs.",
      status: "MET",
    });
  } else {
    conditions.push({
      id: "evidence_completeness",
      category: "EVIDENCE",
      title: "Evidence Data Completeness",
      description: "Missing or partial model inputs (e.g. SEC filings or options flow). Unassessed factors do not penalize but reduce confidence.",
      status: eligibility === "LIMITED" ? "PENDING" : "UNMET",
    });
  }

  return conditions;
}

/**
 * Returns primary decision reason deterministically.
 */
export function deriveDecisionReason(insight: QuantitativeInsight): string {
  if (insight.terminalState?.headlineExplanation) {
    return insight.terminalState.headlineExplanation;
  }
  if (insight.standard?.bottomLine) {
    return insight.standard.bottomLine;
  }
  return insight.human?.assessmentDescription || "Analytical assessment derived from multi-pillar quantitative evaluation.";
}

/**
 * Returns primary decision verdict deterministically.
 */
export function deriveDecisionVerdict(insight: QuantitativeInsight): string {
  return insight.verdictLabel || insight.human?.assessmentHeadline || insight.terminalState?.uiStateLabel || "ANALYSIS VERDICT";
}
