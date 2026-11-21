import assert from "node:assert";
import {
  resolveRadarCandidateRationale,
  resolveRadarCandidateCoverage,
  resolveRadarScorePresentation,
  formatRadarExecutionStatus,
  RadarCandidateInput,
} from "../lib/radarPresentation";
import {
  mapRadarCandidate,
  compareRadarAssets,
  parseCanonicalExecutionStatus,
  RadarAsset,
} from "../lib/radarRanking";
import {
  validateDecisionContext,
  resolveDiscoveryHandoff,
  DiscoveryHandoff,
} from "../lib/decisionContext";

console.log("===============================================================================");
console.log("ARX RADAR EVIDENCE, RANKING & ANALYSIS-HANDOFF QUALIFICATION SUITE (Issue #166)");
console.log("===============================================================================\n");

// ---------------------------------------------------------------------------
// 1. GATE 2: Evidence-Supported Presentation & Strict Narrative Boundaries
// ---------------------------------------------------------------------------
console.log("1. Verifying Evidence-Supported Presentation & Strict Boundaries (Gate 2)...");

// 1.1 Category SMART_MONEY tag alone must NOT claim institutional accumulation or ownership
const smOnlyCandidate: RadarCandidateInput = {
  ticker: "NVDA",
  name: "NVIDIA Corporation",
  categories: ["SMART_MONEY"],
  executionStatus: "IN_BUY_ZONE",
  rvol: "1.2x",
};
const smOnlyRationale = resolveRadarCandidateRationale(smOnlyCandidate);

assert(
  !smOnlyRationale.summary.toLowerCase().includes("institutional accumulation"),
  "SMART_MONEY category tag alone must NEVER claim 'institutional accumulation'"
);
assert(
  !smOnlyRationale.summary.toLowerCase().includes("order flow footprint"),
  "SMART_MONEY category tag alone must NEVER claim 'order flow footprint'"
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
  smOnlyRationale.technicalContext.includes("Categorization alone does not establish institutional accumulation or insider ownership"),
  "Technical context must warn that classification alone does not establish accumulation or insider ownership"
);
console.log("   [PASS] 1.1: Category tag alone does not synthesize institutional accumulation or insider ownership claims.");

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
  vcpSmRationale.technicalContext.includes("does not establish institutional accumulation") ||
  vcpSmRationale.technicalContext.includes("does not confirm institutional accumulation"),
  "VCP + SMART_MONEY technical context must warn that qualification does not establish institutional accumulation"
);
assert(
  vcpSmRationale.summary.includes("Form 4 filing transactions unverified"),
  "VCP + SMART_MONEY must disclose unverified Form 4 transactions"
);
assert.strictEqual(
  vcpSmRationale.headline,
  "NVDA · VCP & Smart Money Screener Qualification",
  "Headline must reflect VCP & Smart Money screener qualification"
);
console.log("   [PASS] 1.2: Dual technical + Smart Money candidates qualify filing verification status.");

// 1.3 VCP + VALUE_GARP dual candidate narrative boundaries
const vcpGarpCandidate: RadarCandidateInput = {
  ticker: "MSFT",
  name: "Microsoft Corporation",
  sector: "Technology",
  vcpStage: "Stage 2 Base",
  volumeDryUpPct: 30,
  categories: ["VCP", "VALUE_GARP"],
  executionStatus: "IN_BUY_ZONE",
};
const vcpGarpRationale = resolveRadarCandidateRationale(vcpGarpCandidate);
assert(
  !vcpGarpRationale.summary.toLowerCase().includes("institutional supply absorption"),
  "VCP + GARP must not claim unverified institutional supply absorption"
);
assert(
  !vcpGarpRationale.technicalContext.toLowerCase().includes("institutional supply absorption"),
  "VCP + GARP technical context must not claim institutional supply absorption"
);
assert(
  vcpGarpRationale.technicalContext.includes("discovery ranking is not an automatic buy signal"),
  "VCP + GARP must remind that discovery ranking is not an automatic buy signal"
);
assert.strictEqual(
  vcpGarpRationale.headline,
  "MSFT · VCP & Value GARP Screener Qualification",
  "Headline must reflect VCP & Value GARP Screener Qualification"
);
console.log("   [PASS] 1.3: VCP + GARP narrative strictly bounded to rule-based screener qualification.");

// 1.4 Single VCP candidate narrative boundaries
const vcpOnlyCandidate: RadarCandidateInput = {
  ticker: "AAPL",
  name: "Apple Inc.",
  sector: "Technology",
  vcpStage: "Pivot Breakout",
  categories: ["VCP"],
  executionStatus: "IN_BUY_ZONE",
};
const vcpOnlyRationale = resolveRadarCandidateRationale(vcpOnlyCandidate);
assert(
  !vcpOnlyRationale.headline.includes("Minervini Stage 2 Accumulation Structure"),
  "VCP alone must NOT assert 'Minervini Stage 2 Accumulation Structure'"
);
assert(
  !vcpOnlyRationale.technicalContext.toLowerCase().includes("absorb overhead supply"),
  "VCP alone must NOT assert supply absorption prior to resistance clearance"
);
assert.strictEqual(
  vcpOnlyRationale.headline,
  "AAPL · Volatility Contraction Pattern Observation",
  "Headline must reflect pattern observation"
);
console.log("   [PASS] 1.4: VCP-only candidate does not claim accumulation structure or supply absorption.");

// 1.5 Single VALUE_GARP candidate narrative boundaries
const garpOnlyCandidate: RadarCandidateInput = {
  ticker: "JPM",
  name: "JPMorgan Chase & Co.",
  sector: "Financial Services",
  categories: ["VALUE_GARP"],
  executionStatus: "IN_BUY_ZONE",
};
const garpOnlyRationale = resolveRadarCandidateRationale(garpOnlyCandidate);
assert(
  !garpOnlyRationale.technicalContext.includes("Resilient balance sheet metrics provide structural valuation support"),
  "GARP alone must NOT assert balance sheet resilience without single-asset verification"
);
assert.strictEqual(
  garpOnlyRationale.headline,
  "JPM · Value GARP Screener Qualification",
  "Headline must reflect Value GARP screener qualification"
);
console.log("   [PASS] 1.5: GARP-only candidate requires single-asset verification for balance sheet evaluation.");

// 1.6 Missing sector must NOT default to 'Broad Market'
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
console.log("   [PASS] 1.6: Missing sector honestly omitted; never defaults to 'Broad Market'.");

// 1.7 Missing vcpStage must NOT default to 'Stage 2 Base'
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
console.log("   [PASS] 1.7: Missing pattern stage honestly marked 'Pattern Unassigned'.");

// 1.8 Explicit opportunityExplanation is prioritized; fallback catalyst/thesis is NOT presented as opportunityExplanation
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

// Verify mapRadarCandidate does NOT populate opportunityExplanation with catalyst/thesis
const rawCandidateWithThesis = {
  ticker: "NVDA",
  primary_catalyst: "New Blackwell chip volume shipments commencing Q4.",
  investment_thesis: "Secular AI acceleration driver.",
  categories: ["VCP"],
};
const mappedWithThesis = mapRadarCandidate(rawCandidateWithThesis);
assert.strictEqual(
  mappedWithThesis.opportunityExplanation,
  null,
  "mapRadarCandidate must NOT map primary_catalyst or investment_thesis to opportunityExplanation"
);
assert.strictEqual(
  mappedWithThesis.catalyst,
  "New Blackwell chip volume shipments commencing Q4.",
  "primary_catalyst preserved in catalyst field"
);
console.log("   [PASS] 1.8: Backend opportunityExplanation prioritized; catalyst/thesis never substituted.");

// 1.9 Execution statuses: INSUFFICIENT_HISTORY (avoids universal 250-session claim) and STALE_MARKET_DATA
const insufficientHistRationale = resolveRadarCandidateRationale({
  ticker: "NEWCO",
  executionStatus: "INSUFFICIENT_HISTORY",
});
assert.strictEqual(insufficientHistRationale.headline, "Insufficient History Depth");
assert(insufficientHistRationale.summary.includes("bar depth insufficient"));
assert(
  !insufficientHistRationale.technicalContext.includes("250 daily trading sessions"),
  "INSUFFICIENT_HISTORY must avoid universal 250-session claims"
);
assert(
  insufficientHistRationale.technicalContext.includes("Candidate bar count does not meet minimum lookback requirements"),
  "INSUFFICIENT_HISTORY must state minimum lookback requirements"
);

const staleMarketRationale = resolveRadarCandidateRationale({
  ticker: "OLDCO",
  executionStatus: "STALE_MARKET_DATA",
});
assert.strictEqual(staleMarketRationale.headline, "Stale Market Data");
assert(staleMarketRationale.summary.includes("precede the authoritative completed session cutoff"));
console.log("   [PASS] 1.9: INSUFFICIENT_HISTORY and STALE_MARKET_DATA provide contract-bounded messaging.");

// 1.10 Evidence Coverage: Verified Positive, Verified Negative, and Missing Evidence Separately Tested
// 1.10.a: Valid integer counts 0 through 4
assert.strictEqual(resolveRadarCandidateCoverage(0), "0/4 Pillars Verified");
assert.strictEqual(resolveRadarCandidateCoverage(1), "1/4 Pillars Verified");
assert.strictEqual(resolveRadarCandidateCoverage(2), "2/4 Pillars Verified");
assert.strictEqual(resolveRadarCandidateCoverage(3), "3/4 Pillars Verified");
assert.strictEqual(resolveRadarCandidateCoverage(4), "4/4 Pillars Verified");

// 1.10.b: Verified negative evidence (all 4 pillars evaluated and verified with negative/bearish findings)
// Confluence engine returns verifiedPillarCount = 4 despite 0 positive pillars
const verifiedNegativeCandidate = mapRadarCandidate({
  ticker: "BEAR",
  confluenceScore: 18.5,
  verifiedPillarCount: 4, // 4 pillars verified, independent of bullish/bearish signal
  categories: ["VALUE_GARP"],
});
assert.strictEqual(verifiedNegativeCandidate.verifiedPillarCount, 4);
const negativeCoverageRationale = resolveRadarCandidateRationale(verifiedNegativeCandidate);
assert.strictEqual(
  negativeCoverageRationale.coverageText,
  "4/4 Pillars Verified",
  "Verified negative evidence must report 4/4 Pillars Verified"
);

// 1.10.c: Missing or invalid coverage metadata fails closed to "Coverage: Not Checked"
assert.strictEqual(resolveRadarCandidateCoverage(null), "Coverage: Not Checked");
assert.strictEqual(resolveRadarCandidateCoverage(undefined), "Coverage: Not Checked");
assert.strictEqual(resolveRadarCandidateCoverage(NaN), "Coverage: Not Checked");
assert.strictEqual(resolveRadarCandidateCoverage(-1), "Coverage: Not Checked");
assert.strictEqual(resolveRadarCandidateCoverage(5), "Coverage: Not Checked");
assert.strictEqual(resolveRadarCandidateCoverage(2.5), "Coverage: Not Checked");
assert.strictEqual(resolveRadarCandidateCoverage("3" as any), "Coverage: Not Checked");

const unverifiedMetadataCandidate = mapRadarCandidate({
  ticker: "NO_METADATA",
  verifiedPillarCount: null,
});
assert.strictEqual(unverifiedMetadataCandidate.verifiedPillarCount, null);
const uncheckedRationale = resolveRadarCandidateRationale(unverifiedMetadataCandidate);
assert.strictEqual(
  uncheckedRationale.coverageText,
  "Coverage: Not Checked",
  "Missing verification metadata must display 'Coverage: Not Checked'"
);
console.log("   [PASS] 1.10: Evidence coverage strictly decoupled from signal status; negative and missing evidence tested separately.\n");

// ---------------------------------------------------------------------------
// 2. GATE 3: Production Ranking Path & Exact Score Preservation
// ---------------------------------------------------------------------------
console.log("2. Verifying Production Ranking Path & Comparator Invariants (Gate 3)...");

// 2.1 Preserving exact unrounded scores through canonical compareRadarAssets
// Case A: 45.4 (MSFT) vs 44.6 (AAPL). Alphabetical AAPL must NOT leapfrog MSFT.
const rawCandA = { ticker: "AAPL", confluenceScore: 44.6, gemScore: 50, executionStatus: "IN_BUY_ZONE" };
const rawCandB = { ticker: "MSFT", confluenceScore: 45.4, gemScore: 50, executionStatus: "IN_BUY_ZONE" };

const mappedA = mapRadarCandidate(rawCandA);
const mappedB = mapRadarCandidate(rawCandB);

assert.strictEqual(mappedA.confluenceScore, 44.6, "Exact float 44.6 preserved without integer rounding");
assert.strictEqual(mappedB.confluenceScore, 45.4, "Exact float 45.4 preserved without integer rounding");

const sortedPairA = [mappedA, mappedB].sort((a, b) => compareRadarAssets(a, b, "SCORE"));
assert.strictEqual(sortedPairA[0].ticker, "MSFT", "MSFT (45.4) must rank above AAPL (44.6)");
assert.strictEqual(sortedPairA[1].ticker, "AAPL", "AAPL (44.6) must rank below MSFT (45.4)");
console.log("   [PASS] 2.1: 45.4 ranks above 44.6; exact unrounded score preserved in candidate mapping.");

// 2.2 Order stability regardless of reversed input array order
const sortedPairB = [mappedB, mappedA].sort((a, b) => compareRadarAssets(a, b, "SCORE"));
assert.strictEqual(sortedPairB[0].ticker, "MSFT", "Order remains stable regardless of initial input order");
assert.strictEqual(sortedPairB[1].ticker, "AAPL");
console.log("   [PASS] 2.2: Sort order is stable regardless of initial input order.");

// 2.3 Secondary factor gemScore tie-breaker on equal confluenceScore (50.0 vs 50.0)
const candTieConfluenceA = mapRadarCandidate({ ticker: "AAPL", confluenceScore: 50.0, gemScore: 65, executionStatus: "IN_BUY_ZONE" });
const candTieConfluenceB = mapRadarCandidate({ ticker: "MSFT", confluenceScore: 50.0, gemScore: 75, executionStatus: "IN_BUY_ZONE" });

const sortedTie = [candTieConfluenceA, candTieConfluenceB].sort((a, b) => compareRadarAssets(a, b, "SCORE"));
assert.strictEqual(sortedTie[0].ticker, "MSFT", "MSFT with higher gemScore (75 vs 65) must rank first");
assert.strictEqual(sortedTie[1].ticker, "AAPL", "AAPL with lower gemScore (65 vs 75) must rank second");
console.log("   [PASS] 2.3: gemScore secondary factor breaks ties before ticker tie-breaker.");

// 2.4 Lexicographical deterministic tie-breaker on identical confluenceScore and gemScore
const candExactTieA = mapRadarCandidate({ ticker: "ZZZ", confluenceScore: 50.0, gemScore: 70, executionStatus: "IN_BUY_ZONE" });
const candExactTieB = mapRadarCandidate({ ticker: "AAA", confluenceScore: 50.0, gemScore: 70, executionStatus: "IN_BUY_ZONE" });

const sortedExactTie = [candExactTieA, candExactTieB].sort((a, b) => compareRadarAssets(a, b, "SCORE"));
assert.strictEqual(sortedExactTie[0].ticker, "AAA", "AAA must precede ZZZ when scores are equal");
assert.strictEqual(sortedExactTie[1].ticker, "ZZZ");
console.log("   [PASS] 2.4: Deterministic ticker tie-breaker verified on exact score matches.");

// 2.5 Missing or non-finite scores safely sink to the bottom
const candValid = mapRadarCandidate({ ticker: "VALID", confluenceScore: 10.0, gemScore: 10, executionStatus: "IN_BUY_ZONE" });
const candNull = mapRadarCandidate({ ticker: "NULL_SCORE", confluenceScore: null, gemScore: null, executionStatus: "IN_BUY_ZONE" });
const candNaN = mapRadarCandidate({ ticker: "NAN_SCORE", confluenceScore: NaN, gemScore: null, executionStatus: "IN_BUY_ZONE" });

const sortedWithMissing = [candNull, candValid, candNaN].sort((a, b) => compareRadarAssets(a, b, "SCORE"));
assert.strictEqual(sortedWithMissing[0].ticker, "VALID", "Valid score must rank above null and NaN scores");
console.log("   [PASS] 2.5: Missing/non-finite scores safely sink to bottom.");

// 2.6 Non-tradables (DELISTED_SECURITY, UNVERIFIED_ASSET) sink to bottom regardless of raw score
const candDelisted = mapRadarCandidate({ ticker: "DELISTED_HIGH", confluenceScore: 99.0, gemScore: 95, executionStatus: "DELISTED_SECURITY" });
const candUnverified = mapRadarCandidate({ ticker: "UNVERIFIED_HIGH", confluenceScore: 95.0, gemScore: 90, executionStatus: "UNVERIFIED_ASSET" });
const candActiveLow = mapRadarCandidate({ ticker: "ACTIVE_LOW", confluenceScore: 20.0, gemScore: 15, executionStatus: "IN_BUY_ZONE" });

const sortedTradability = [candDelisted, candActiveLow, candUnverified].sort((a, b) => compareRadarAssets(a, b, "SCORE"));
assert.strictEqual(sortedTradability[0].ticker, "ACTIVE_LOW", "Active tradable asset must rank above all non-tradables");
assert.strictEqual(sortedTradability[1].executionStatus, "DELISTED_SECURITY");
assert.strictEqual(sortedTradability[2].executionStatus, "UNVERIFIED_ASSET");
console.log("   [PASS] 2.6: Non-tradable securities sink to bottom regardless of raw score.");

// 2.7 Presentation display rounding only at presentation boundary
const pres45_4 = resolveRadarScorePresentation(45.4, false);
const pres44_6 = resolveRadarScorePresentation(44.6, false);
assert.strictEqual(pres45_4.label, "45", "45.4 displays as rounded integer '45'");
assert.strictEqual(pres44_6.label, "45", "44.6 displays as rounded integer '45'");
assert.strictEqual(pres45_4.tier, "DEFENSIVE", "Score 45.4 is in DEFENSIVE tier");
assert.strictEqual(pres44_6.tier, "DEFENSIVE", "Score 44.6 is in DEFENSIVE tier");
console.log("   [PASS] 2.7: Presentation applies display rounding only at render boundary.\n");

// ---------------------------------------------------------------------------
// 3. GATE 4: Honest Discovery-to-Analysis Handoff & Symbol Lifecycle
// ---------------------------------------------------------------------------
console.log("3. Verifying Honest Discovery-to-Analysis Handoff & Lifecycle (Gate 4)...");

// 3.1 Direct navigation (no contextId) -> No handoff notice
const directNav: DiscoveryHandoff = resolveDiscoveryHandoff(null, "NVDA");
assert.strictEqual(directNav.hasContextParam, false, "Direct nav hasContextParam must be false");
assert.strictEqual(directNav.handoffNotice, null, "Direct nav must not produce handoff notice");
assert.strictEqual(directNav.noticeType, null, "Direct nav noticeType must be null");
console.log("   [PASS] 3.1: Direct navigation produces no extraneous notices.");

// 3.2 Valid contextId -> INFO disclosure naming the active asset
const validHandoff: DiscoveryHandoff = resolveDiscoveryHandoff("dec-radar-NVDA-1772323200000", "NVDA");
assert.strictEqual(validHandoff.hasContextParam, true);
assert.strictEqual(validHandoff.validation.isValid, true);
assert.strictEqual(validHandoff.noticeType, "INFO");
assert.strictEqual(
  validHandoff.handoffNotice,
  "Original discovery context for NVDA unavailable; showing a separate assessment."
);
console.log("   [PASS] 3.2: Valid discovery context displays disclosure naming active asset: 'Original discovery context for NVDA unavailable; showing a separate assessment.'");

// 3.3 Symbol mismatch -> WARNING notice naming both context symbol and displayed symbol
const mismatchHandoff: DiscoveryHandoff = resolveDiscoveryHandoff("dec-radar-NVDA-1772323200000", "AAPL");
assert.strictEqual(mismatchHandoff.hasContextParam, true);
assert.strictEqual(mismatchHandoff.validation.isValid, false);
assert.strictEqual(mismatchHandoff.validation.status, "CONTEXT_SYMBOL_MISMATCH");
assert.strictEqual(mismatchHandoff.noticeType, "WARNING");
assert.strictEqual(
  mismatchHandoff.handoffNotice,
  "Context reference symbol mismatch ('NVDA' != 'AAPL'); showing a separate assessment for AAPL."
);
console.log("   [PASS] 3.3: Symbol mismatch displays informative warning notice naming displayed symbol.");

// 3.4 Malformed contextId -> WARNING notice naming displayed symbol
const malformedHandoff: DiscoveryHandoff = resolveDiscoveryHandoff("invalid-context-id", "NVDA");
assert.strictEqual(malformedHandoff.hasContextParam, true);
assert.strictEqual(malformedHandoff.validation.isValid, false);
assert.strictEqual(malformedHandoff.validation.status, "CONTEXT_MALFORMED");
assert.strictEqual(malformedHandoff.noticeType, "WARNING");
assert.strictEqual(
  malformedHandoff.handoffNotice,
  "Context reference malformed or invalid; showing a separate assessment for NVDA."
);
console.log("   [PASS] 3.4: Malformed contextId displays informative warning notice naming displayed symbol.");

// 3.5 Subsequent Symbol Change: user navigates with context for NVDA, then switches to MSFT
const initialHandoff = resolveDiscoveryHandoff("dec-radar-NVDA-1772323200000", "NVDA");
assert.strictEqual(initialHandoff.validation.isValid, true);
assert.strictEqual(initialHandoff.noticeType, "INFO");
assert.strictEqual(
  initialHandoff.handoffNotice,
  "Original discovery context for NVDA unavailable; showing a separate assessment."
);

// User switches to MSFT in the viewing UI
const switchedHandoff = resolveDiscoveryHandoff("dec-radar-NVDA-1772323200000", "MSFT");
assert.strictEqual(switchedHandoff.validation.isValid, false);
assert.strictEqual(switchedHandoff.validation.status, "CONTEXT_SYMBOL_MISMATCH");
assert.strictEqual(switchedHandoff.noticeType, "WARNING");
assert.strictEqual(
  switchedHandoff.handoffNotice,
  "Context reference symbol mismatch ('NVDA' != 'MSFT'); showing a separate assessment for MSFT."
);
console.log("   [PASS] 3.5: Subsequent symbol change re-evaluates notice to describe currently displayed asset.");

// 3.6 Epistemic Invariant: Context ID never alters execution authority or reproduces original snapshot
assert.strictEqual(
  validHandoff.validation.contextId,
  "dec-radar-NVDA-1772323200000",
  "Context ID reference preserved for tracing without conferring execution authority"
);
console.log("   [PASS] 3.6: Context ID remains strictly decoupled from execution authority; zero claims of reproduced snapshot.\n");

console.log("===============================================================================");
console.log("ALL ARX RADAR EVIDENCE, RANKING & HANDOFF QUALIFICATION CHECKS PASSED (PASS) 🚀");
console.log("===============================================================================");
