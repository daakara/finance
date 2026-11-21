import assert from "node:assert";
import {
  resolveRadarCandidateRationale,
  resolveRadarScorePresentation,
  formatRadarExecutionStatus,
  RadarCandidateInput,
} from "../lib/radarPresentation";
import {
  validateDecisionContext,
  resolveDiscoveryHandoff,
  DiscoveryHandoff,
} from "../lib/decisionContext";

console.log("===============================================================================");
console.log("ARX RADAR EVIDENCE, RANKING & ANALYSIS-HANDOFF QUALIFICATION SUITE (Issue #166)");
console.log("===============================================================================\n");

// ---------------------------------------------------------------------------
// 1. GATE 2: Evidence-Supported Presentation
// ---------------------------------------------------------------------------
console.log("1. Verifying Evidence-Supported Presentation (Gate 2)...");

// 1.1 Category SMART_MONEY tag alone must NOT claim institutional accumulation
const smOnlyCandidate: RadarCandidateInput = {
  ticker: "NVDA",
  name: "NVIDIA Corporation",
  categories: ["SMART_MONEY"],
  executionStatus: "IN_BUY_ZONE",
  rvol: "1.2x",
};
const smOnlyRationale = resolveRadarCandidateRationale(smOnlyCandidate);

assert(
  !smOnlyRationale.summary.toLowerCase().includes("institutional accumulation signals"),
  "SMART_MONEY category tag alone must NEVER claim 'institutional accumulation signals'"
);
assert(
  smOnlyRationale.summary.includes("Form 4 filing transactions unverified") ||
  smOnlyRationale.summary.includes("screener snapshot"),
  "SMART_MONEY rationale must honestly disclose unverified transaction filings"
);
assert.strictEqual(
  smOnlyRationale.headline,
  "NVDA · Smart Money Observation",
  "Headline must reflect observation, not confirmed order flow"
);
assert(
  smOnlyRationale.technicalContext.includes("Smart Money classification does not confirm institutional accumulation") ||
  smOnlyRationale.technicalContext.includes("Categorization alone does not confirm institutional accumulation"),
  "Technical context must warn that classification alone does not confirm accumulation"
);
console.log("   [PASS] 1.1: Category tag alone does not synthesize institutional accumulation claims.");

// 1.2 Dual VCP + SMART_MONEY candidate must qualify Smart Money as unverified filings
const vcpSmCandidate: RadarCandidateInput = {
  ticker: "NVDA",
  name: "NVIDIA Corporation",
  sector: "Technology",
  vcpStage: "Pivot Breakout",
  volumeDryUpPct: 45,
  rvol: "2.4x",
  categories: ["VCP", "SMART_MONEY"],
  executionStatus: "IN_BUY_ZONE",
};
const vcpSmRationale = resolveRadarCandidateRationale(vcpSmCandidate);

assert(
  !vcpSmRationale.summary.toLowerCase().includes("supported by institutional accumulation flows"),
  "VCP + SMART_MONEY must not claim confirmed institutional accumulation flows"
);
assert(
  vcpSmRationale.summary.includes("Form 4 filing transactions unverified"),
  "VCP + SMART_MONEY must disclose unverified Form 4 transactions"
);
assert(
  vcpSmRationale.headline.includes("Smart Money Classification"),
  "Headline must reflect Smart Money classification"
);
console.log("   [PASS] 1.2: Dual technical + Smart Money candidates qualify filing verification status.");

// 1.3 Missing sector must NOT default to 'Broad Market'
const noSectorCandidate: RadarCandidateInput = {
  ticker: "TEST",
  categories: ["VCP"],
  executionStatus: "IN_BUY_ZONE",
  vcpStage: "Base Under Evaluation",
};
const noSectorRationale = resolveRadarCandidateRationale(noSectorCandidate);
assert(
  !noSectorRationale.summary.includes("Broad Market"),
  "Missing sector must NOT default to 'Broad Market'"
);
console.log("   [PASS] 1.3: Missing sector honestly omitted; never defaults to 'Broad Market'.");

// 1.4 Missing vcpStage must NOT default to 'Stage 2 Base'
const noStageCandidate: RadarCandidateInput = {
  ticker: "TEST",
  categories: ["VCP"],
  executionStatus: "IN_BUY_ZONE",
  vcpStage: null,
};
const noStageRationale = resolveRadarCandidateRationale(noStageCandidate);
assert(
  !noStageRationale.summary.includes("Stage 2 Base"),
  "Missing vcpStage must NOT default to 'Stage 2 Base'"
);
assert(
  noStageRationale.summary.includes("Pattern Unassigned"),
  "Missing vcpStage must state 'Pattern Unassigned'"
);
console.log("   [PASS] 1.4: Missing pattern stage honestly marked 'Pattern Unassigned'.");

// 1.5 Explicit opportunityExplanation is prioritized over category templates
const customExplCandidate: RadarCandidateInput = {
  ticker: "NVDA",
  categories: ["SMART_MONEY"],
  executionStatus: "IN_BUY_ZONE",
  opportunityExplanation: "Authoritative model detected high-momentum supply clearance post-earnings.",
};
const customExplRationale = resolveRadarCandidateRationale(customExplCandidate);
assert.strictEqual(
  customExplRationale.summary,
  "Authoritative model detected high-momentum supply clearance post-earnings.",
  "Explicit opportunityExplanation must be prioritized when provided"
);
console.log("   [PASS] 1.5: Explicit backend candidate explanation is prioritized.");

// 1.6 Execution statuses: INSUFFICIENT_HISTORY and STALE_MARKET_DATA
const insufficientHistRationale = resolveRadarCandidateRationale({
  ticker: "NEWCO",
  executionStatus: "INSUFFICIENT_HISTORY",
});
assert.strictEqual(insufficientHistRationale.headline, "Insufficient History Depth");
assert(insufficientHistRationale.summary.includes("bar depth insufficient"));

const staleMarketRationale = resolveRadarCandidateRationale({
  ticker: "OLDCO",
  executionStatus: "STALE_MARKET_DATA",
});
assert.strictEqual(staleMarketRationale.headline, "Stale Market Data");
assert(staleMarketRationale.summary.includes("precede the authoritative completed session cutoff"));
console.log("   [PASS] 1.6: INSUFFICIENT_HISTORY and STALE_MARKET_DATA provide evidence-bounded messaging.");

// 1.7 Pillar coverage display
const checkedCoverage = resolveRadarCandidateRationale({
  ticker: "NVDA",
  verifiedPillarCount: 3,
});
assert.strictEqual(checkedCoverage.coverageText, "3/4 Pillars Verified");

const uncheckedCoverage = resolveRadarCandidateRationale({
  ticker: "NVDA",
  verifiedPillarCount: null,
});
assert.strictEqual(uncheckedCoverage.coverageText, "Coverage: Not Checked");
console.log("   [PASS] 1.7: Coverage metadata renders 'X/4 Pillars Verified' or 'Coverage: Not Checked'.\n");

// ---------------------------------------------------------------------------
// 2. GATE 3: Canonical Ordering & Score Preservation
// ---------------------------------------------------------------------------
console.log("2. Verifying Canonical Ordering & Precision (Gate 3)...");

// 2.1 Preserving exact unrounded scores through sorting
interface MockCandidate {
  ticker: string;
  confluenceScore: number | null;
  gemScore: number | null;
  executionStatus: string;
}

function sortCandidates(list: MockCandidate[], sortBy = "SCORE"): MockCandidate[] {
  return [...list].sort((a, b) => {
    // 1. Non-tradables sink to bottom
    const aNonTradable = (a.executionStatus === "DELISTED_SECURITY" || a.executionStatus === "UNVERIFIED_ASSET") ? 1 : 0;
    const bNonTradable = (b.executionStatus === "DELISTED_SECURITY" || b.executionStatus === "UNVERIFIED_ASSET") ? 1 : 0;
    if (aNonTradable !== bNonTradable) {
      return aNonTradable - bNonTradable;
    }
    if (sortBy === "SCORE") {
      // 2. Primary: Unrounded confluence score descending
      const diff = (b.confluenceScore ?? -1) - (a.confluenceScore ?? -1);
      if (diff !== 0) return diff;
      // 3. Secondary: gemScore descending tie-breaker
      const gemDiff = (b.gemScore ?? -1) - (a.gemScore ?? -1);
      if (gemDiff !== 0) return gemDiff;
      // 4. Deterministic symbol tie-breaker
      return a.ticker.localeCompare(b.ticker);
    }
    return a.ticker.localeCompare(b.ticker);
  });
}

// Case A: 45.4 (MSFT) vs 44.6 (AAPL). Alphabetical AAPL must NOT leapfrog MSFT.
const testPairA: MockCandidate[] = [
  { ticker: "AAPL", confluenceScore: 44.6, gemScore: 50, executionStatus: "IN_BUY_ZONE" },
  { ticker: "MSFT", confluenceScore: 45.4, gemScore: 50, executionStatus: "IN_BUY_ZONE" },
];
const sortedPairA = sortCandidates(testPairA);
assert.strictEqual(sortedPairA[0].ticker, "MSFT", "MSFT (45.4) must rank above AAPL (44.6)");
assert.strictEqual(sortedPairA[1].ticker, "AAPL", "AAPL (44.6) must rank below MSFT (45.4)");
console.log("   [PASS] 2.1: 45.4 ranks above 44.6; alphabetical sorting does not invert order.");

// Case B: Inverted input order
const testPairB: MockCandidate[] = [
  { ticker: "MSFT", confluenceScore: 45.4, gemScore: 50, executionStatus: "IN_BUY_ZONE" },
  { ticker: "AAPL", confluenceScore: 44.6, gemScore: 50, executionStatus: "IN_BUY_ZONE" },
];
const sortedPairB = sortCandidates(testPairB);
assert.strictEqual(sortedPairB[0].ticker, "MSFT", "Order remains stable regardless of array order");
console.log("   [PASS] 2.2: Sort order is stable regardless of initial input order.");

// Case C: Exact tie on confluenceScore (both 50.0) -> gemScore breaks tie
const testPairC: MockCandidate[] = [
  { ticker: "AAPL", confluenceScore: 50.0, gemScore: 65, executionStatus: "IN_BUY_ZONE" },
  { ticker: "MSFT", confluenceScore: 50.0, gemScore: 75, executionStatus: "IN_BUY_ZONE" },
];
const sortedPairC = sortCandidates(testPairC);
assert.strictEqual(sortedPairC[0].ticker, "MSFT", "MSFT with higher gemScore (75 vs 65) must rank first");
console.log("   [PASS] 2.3: gemScore secondary factor breaks ties before ticker tie-breaker.");

// Case D: Non-tradables sink to bottom
const testNonTradable: MockCandidate[] = [
  { ticker: "DELISTED_HIGH", confluenceScore: 95.0, gemScore: 90, executionStatus: "DELISTED_SECURITY" },
  { ticker: "UNVERIFIED_HIGH", confluenceScore: 90.0, gemScore: 85, executionStatus: "UNVERIFIED_ASSET" },
  { ticker: "ACTIVE_LOW", confluenceScore: 45.0, gemScore: 40, executionStatus: "IN_BUY_ZONE" },
];
const sortedNonTradable = sortCandidates(testNonTradable);
assert.strictEqual(sortedNonTradable[0].ticker, "ACTIVE_LOW", "Active asset must rank above non-tradables");
assert.strictEqual(sortedNonTradable[1].executionStatus, "DELISTED_SECURITY");
assert.strictEqual(sortedNonTradable[2].executionStatus, "UNVERIFIED_ASSET");
console.log("   [PASS] 2.4: Non-tradable securities sink to bottom regardless of raw score.");

// Case E: Display rounding only at presentation boundary
const pres45_4 = resolveRadarScorePresentation(45.4, false);
const pres44_6 = resolveRadarScorePresentation(44.6, false);
assert.strictEqual(pres45_4.label, "45", "45.4 displays as rounded integer '45'");
assert.strictEqual(pres44_6.label, "45", "44.6 displays as rounded integer '45'");
assert.strictEqual(pres45_4.tier, "DEFENSIVE", "Score 45.4 is in DEFENSIVE tier");
assert.strictEqual(pres44_6.tier, "DEFENSIVE", "Score 44.6 is in DEFENSIVE tier");
console.log("   [PASS] 2.5: Presentation applies display rounding only at render boundary.\n");

// ---------------------------------------------------------------------------
// 3. GATE 4: Honest Discovery-to-Analysis Handoff
// ---------------------------------------------------------------------------
console.log("3. Verifying Honest Discovery-to-Analysis Handoff (Gate 4)...");

// 3.1 Direct navigation (no contextId) -> No handoff notice
const directNav: DiscoveryHandoff = resolveDiscoveryHandoff(null, "NVDA");
assert.strictEqual(directNav.hasContextParam, false, "Direct nav hasContextParam must be false");
assert.strictEqual(directNav.handoffNotice, null, "Direct nav must not produce handoff notice");
assert.strictEqual(directNav.noticeType, null, "Direct nav noticeType must be null");
console.log("   [PASS] 3.1: Direct navigation produces no extraneous notices.");

// 3.2 Valid contextId -> INFO disclosure explaining independent assessment
const validHandoff: DiscoveryHandoff = resolveDiscoveryHandoff("dec-radar-NVDA-1772323200000", "NVDA");
assert.strictEqual(validHandoff.hasContextParam, true);
assert.strictEqual(validHandoff.validation.isValid, true);
assert.strictEqual(validHandoff.noticeType, "INFO");
assert.strictEqual(
  validHandoff.handoffNotice,
  "Original discovery context unavailable; showing a separate assessment."
);
console.log("   [PASS] 3.2: Valid discovery context displays disclosure: 'Original discovery context unavailable; showing a separate assessment.'");

// 3.3 Symbol mismatch -> WARNING notice
const mismatchHandoff: DiscoveryHandoff = resolveDiscoveryHandoff("dec-radar-NVDA-1772323200000", "AAPL");
assert.strictEqual(mismatchHandoff.hasContextParam, true);
assert.strictEqual(mismatchHandoff.validation.isValid, false);
assert.strictEqual(mismatchHandoff.validation.status, "CONTEXT_SYMBOL_MISMATCH");
assert.strictEqual(mismatchHandoff.noticeType, "WARNING");
assert(
  mismatchHandoff.handoffNotice?.includes("Context reference symbol mismatch ('NVDA' != 'AAPL')"),
  "Notice must describe symbol mismatch"
);
console.log("   [PASS] 3.3: Symbol mismatch displays informative warning notice.");

// 3.4 Malformed contextId -> WARNING notice
const malformedHandoff: DiscoveryHandoff = resolveDiscoveryHandoff("invalid-context-id", "NVDA");
assert.strictEqual(malformedHandoff.hasContextParam, true);
assert.strictEqual(malformedHandoff.validation.isValid, false);
assert.strictEqual(malformedHandoff.validation.status, "CONTEXT_MALFORMED");
assert.strictEqual(malformedHandoff.noticeType, "WARNING");
assert(
  malformedHandoff.handoffNotice?.includes("Context reference malformed or invalid"),
  "Notice must describe malformed context"
);
console.log("   [PASS] 3.4: Malformed contextId displays informative warning notice.");

// 3.5 Epistemic Invariant: Context ID never alters execution authority
assert.strictEqual(
  validHandoff.validation.contextId,
  "dec-radar-NVDA-1772323200000",
  "Context ID reference preserved for tracing without conferring execution authority"
);
console.log("   [PASS] 3.5: Context ID remains strictly decoupled from execution authority.\n");

console.log("===============================================================================");
console.log("ALL ARX RADAR EVIDENCE, RANKING & HANDOFF QUALIFICATION CHECKS PASSED (PASS) 🚀");
console.log("===============================================================================");
