import { ExecutionStatus } from "../types/decisionContract";
import { CanonicalRadarCategory } from "./api";
import { MASTER_ASSET_CATALOG, MasterAssetEntry } from "./masterCatalog";

export interface RadarAsset {
  ticker: string;
  name: string;
  price: number | null;
  priceBasis?: "DAILY_CLOSE" | null;
  priceDate?: string | null;
  rvol?: string | null;
  riskRewardRatio?: number | null;
  vcpStage: string;
  volumeDryUpPct?: number | null;
  confluenceScore: number | null;
  confluenceBadgeColor?: string | null;
  gemScore?: number | null;
  verifiedPillarCount?: number | null;
  smartMoneyEvidenceState?: string | null;
  catalyst: string;
  categories: CanonicalRadarCategory[];
  sector: string | null;
  executionStatus: ExecutionStatus;
  screeningStatus?: string;
  screeningGeometry?: string;
  decisionState?: string;
  decisionStateLabel?: string;
  isActionable?: boolean;
  canSizeTrade?: boolean;
  disqualificationReason?: string | null;
  decisionContextId?: string;
  allowedActions: string[];
  screeningEligible?: boolean;
  opportunityExplanation?: string | null;
  stageClassification?: string;
}

export function parseCanonicalExecutionStatus(raw: string): ExecutionStatus {
  const s = (raw || "").trim().toUpperCase();
  if (s === "IN_BUY_ZONE" || s === "READY_TO_BUY") return s;
  if (s === "IN_BUY_ZONE_AWAITING_TRIGGER") return s;
  if (s === "WAITING_PULLBACK") return s;
  if (s === "APPROACHING_TARGET") return s;
  if (s === "STOPPED_OUT") return s;
  if (s === "INSUFFICIENT_HISTORY") return s;
  if (s === "DELISTED_SECURITY" || s.includes("DELISTED")) return "DELISTED_SECURITY";
  if (s === "UNVERIFIED_ASSET") return s;
  if (s === "STALE_MARKET_DATA") return s;
  if (s.includes("BUY_ZONE") && !s.includes("AWAITING")) return "IN_BUY_ZONE";
  if (s.includes("AWAITING") || s.includes("TRIGGER")) return "IN_BUY_ZONE_AWAITING_TRIGGER";
  if (s.includes("APPROACHING") || s.includes("TARGET")) return "APPROACHING_TARGET";
  if (s.includes("PULLBACK") || s.includes("WAITING")) return "WAITING_PULLBACK";
  return "UNKNOWN";
}

/**
 * Maps raw backend candidate JSON into the canonical typed RadarAsset.
 *
 * Invariants (Issue #166):
 * 1. Confluence score is preserved as an exact unrounded float (or null).
 * 2. Secondary ranking factor gemScore is extracted and preserved.
 * 3. Verified pillar count is validated as a finite integer [0, 4] (never derived from positive/warning/negative signal status).
 * 4. Missing sector is null; never defaults to "Broad Market".
 * 5. Missing stage/pattern never becomes a Stage 2 assertion (defaults to null or unassigned).
 * 6. Backend opportunityExplanation is preserved only when explicitly supplied; unrelated catalyst/thesis text is never substituted.
 */
export function mapRadarCandidate(gem: any, catalogEntry?: MasterAssetEntry | null): RadarAsset {
  const cat: CanonicalRadarCategory[] = Array.isArray(gem.categories)
    ? (gem.categories.filter((c: any) => ["VALUE_GARP", "VCP", "SMART_MONEY"].includes(c)) as CanonicalRadarCategory[])
    : [];

  const rawStatus = (gem.executionStatus || gem.execution_status || gem.factor_verdict || "").toUpperCase();
  const executionStatus = parseCanonicalExecutionStatus(rawStatus);

  const dryUp = typeof gem.volume_dry_up === "number"
    ? gem.volume_dry_up
    : (typeof gem.volumeDryUpPct === "number" ? gem.volumeDryUpPct : null);

  const rvolVal = typeof gem.rvol === "string" && gem.rvol !== "N/A" && gem.rvol !== "—" ? gem.rvol : null;
  const rrVal = typeof gem.riskRewardRatio === "number" && Number.isFinite(gem.riskRewardRatio) ? gem.riskRewardRatio : null;

  // Pattern / Stage string: never assert Stage 2 if not present in backend
  const stageStr = gem.setup_pattern ||
    gem.vcpStage ||
    gem.vcp_stage ||
    (gem.stage_phase ? `Stage ${gem.stage_phase} Base` : (gem.stageClassification ? String(gem.stageClassification) : (executionStatus === "IN_BUY_ZONE" ? "Breakout Pattern" : "Base Under Evaluation")));

  const cleanTicker = (gem.ticker || gem.symbol || "").toUpperCase();
  const catEntry = catalogEntry ?? MASTER_ASSET_CATALOG[cleanTicker];
  const resolvedName = gem.companyName || gem.name || catEntry?.name || cleanTicker;

  const rawPrice = gem.currentPrice ?? gem.current_price;
  const resolvedPrice = typeof rawPrice === "number" && Number.isFinite(rawPrice) && rawPrice > 0
    ? Number(rawPrice.toFixed(2))
    : null;

  // Unrounded confluence score preserved
  const rawConfluence = typeof gem.confluenceScore === "number"
    ? gem.confluenceScore
    : (typeof gem.composite_score === "number" ? gem.composite_score : null);
  const resolvedConfluence = rawConfluence !== null && Number.isFinite(rawConfluence)
    ? rawConfluence
    : null;

  // Secondary factor gemScore
  const rawGem = typeof gem.gemScore === "number"
    ? gem.gemScore
    : (typeof gem.gem_score === "number" ? gem.gem_score : null);
  const resolvedGem = rawGem !== null && Number.isFinite(rawGem) ? rawGem : null;

  // Verified pillar count: finite integer 0..4, independent of signal direction
  const rawPillars = typeof gem.verifiedPillarCount === "number"
    ? gem.verifiedPillarCount
    : (typeof gem.verified_pillar_count === "number" ? gem.verified_pillar_count : null);
  const validPillars = (rawPillars !== null && Number.isInteger(rawPillars) && rawPillars >= 0 && rawPillars <= 4)
    ? rawPillars
    : null;

  const rawSmEvidence = gem.smartMoneyEvidenceState || gem.smart_money_evidence_state || null;

  // Opportunity explanation: only accept genuine backend explanation; do NOT fall back to unrelated catalyst/thesis
  const validOppExplanation = (typeof gem.opportunityExplanation === "string" && gem.opportunityExplanation.trim().length > 0)
    ? gem.opportunityExplanation.trim()
    : null;

  const resolvedCatalyst = gem.primary_catalyst ||
    gem.investment_thesis ||
    (executionStatus === "DELISTED_SECURITY" ? "Trading suspended for delisted security." : "Catalyst disclosures unavailable.");

  return {
    ticker: cleanTicker,
    name: resolvedName,
    price: resolvedPrice,
    priceBasis: gem.priceBasis ?? (gem.price_basis as any) ?? null,
    priceDate: gem.priceDate ?? gem.price_date ?? null,
    rvol: rvolVal,
    riskRewardRatio: rrVal,
    vcpStage: stageStr,
    volumeDryUpPct: dryUp,
    confluenceScore: resolvedConfluence,
    confluenceBadgeColor: gem.confluenceBadgeColor || gem.confluence_badge_color || gem.badgeColor || null,
    gemScore: resolvedGem,
    verifiedPillarCount: validPillars,
    smartMoneyEvidenceState: rawSmEvidence,
    catalyst: resolvedCatalyst,
    categories: cat,
    sector: gem.sector || catEntry?.sector || null,
    executionStatus,
    screeningStatus: gem.screeningStatus || (executionStatus === "DELISTED_SECURITY" ? "DELISTED" : (executionStatus === "UNVERIFIED_ASSET" ? "UNVERIFIED" : "UNKNOWN")),
    screeningGeometry: gem.screeningGeometry || (executionStatus === "DELISTED_SECURITY" ? "INACTIVE" : "UNKNOWN"),
    decisionState: gem.decisionState || "UNVERIFIED_SETUP",
    decisionStateLabel: gem.decisionStateLabel || (gem.decisionState === "ACTIONABLE_SETUP" ? "Discovery Candidate — Analyze for Trigger" : "Discovery Candidate"),
    isActionable: false, // Invariant: Radar is a discovery scanner; canonical actionability reserved for DecisionTrace
    canSizeTrade: false,
    disqualificationReason: gem.disqualificationReason || null,
    decisionContextId: gem.decisionContextId,
    allowedActions: Array.isArray(gem.allowedActions) ? gem.allowedActions : [],
    screeningEligible: executionStatus === "IN_BUY_ZONE" && gem.screeningEligible === true,
    opportunityExplanation: validOppExplanation,
    stageClassification: gem.stageClassification,
  };
}

/**
 * Canonical Radar deterministic asset comparator.
 *
 * Sorting priority:
 * 1. Non-tradables (DELISTED_SECURITY, UNVERIFIED_ASSET) sink to the bottom.
 * 2. Unrounded confluenceScore descending (Discovery Score ranking). Non-finite or missing scores sink.
 * 3. gemScore descending (secondary factor tie-breaker).
 * 4. symbol ascending (lexicographical deterministic tie-breaker).
 */
export function compareRadarAssets(
  a: RadarAsset,
  b: RadarAsset,
  sortBy: "SCORE" | "RVOL" | "PRICE" = "SCORE"
): number {
  // 1. Delisted and unverified securities sink to bottom
  const aNonTradable = (a.executionStatus === "DELISTED_SECURITY" || a.executionStatus === "UNVERIFIED_ASSET") ? 1 : 0;
  const bNonTradable = (b.executionStatus === "DELISTED_SECURITY" || b.executionStatus === "UNVERIFIED_ASSET") ? 1 : 0;
  if (aNonTradable !== bNonTradable) {
    return aNonTradable - bNonTradable;
  }

  if (sortBy === "SCORE") {
    // 2. Unrounded confluence score descending
    const aScore = (typeof a.confluenceScore === "number" && Number.isFinite(a.confluenceScore)) ? a.confluenceScore : -1;
    const bScore = (typeof b.confluenceScore === "number" && Number.isFinite(b.confluenceScore)) ? b.confluenceScore : -1;
    const diff = bScore - aScore;
    if (diff !== 0) return diff;

    // 3. gemScore descending (secondary factor tie-breaker)
    const aGem = (typeof a.gemScore === "number" && Number.isFinite(a.gemScore)) ? a.gemScore : -1;
    const bGem = (typeof b.gemScore === "number" && Number.isFinite(b.gemScore)) ? b.gemScore : -1;
    const gemDiff = bGem - aGem;
    if (gemDiff !== 0) return gemDiff;

    // 4. symbol ascending (lexicographical deterministic tie-breaker)
    return a.ticker.localeCompare(b.ticker);
  }

  if (sortBy === "RVOL") {
    const aRvol = a.rvol ? parseFloat(a.rvol.replace("x", "")) : 0;
    const bRvol = b.rvol ? parseFloat(b.rvol.replace("x", "")) : 0;
    const diff = (Number.isFinite(bRvol) ? bRvol : 0) - (Number.isFinite(aRvol) ? aRvol : 0);
    if (diff !== 0) return diff;
    return a.ticker.localeCompare(b.ticker);
  }

  if (sortBy === "PRICE") {
    const aPrice = (typeof a.price === "number" && Number.isFinite(a.price)) ? a.price : -1;
    const bPrice = (typeof b.price === "number" && Number.isFinite(b.price)) ? b.price : -1;
    const diff = bPrice - aPrice;
    if (diff !== 0) return diff;
    return a.ticker.localeCompare(b.ticker);
  }

  return a.ticker.localeCompare(b.ticker);
}
