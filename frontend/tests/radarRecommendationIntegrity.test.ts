import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import {
  resolveRadarScorePresentation,
  formatRadarExecutionStatus,
  RadarScoreTier,
} from "../lib/radarPresentation";

console.log("==================================================================");
console.log("ARX RADAR RECOMMENDATION INTEGRITY TEST SUITE");
console.log("Directive: ARX-RADAR-RECOMMENDATION-INTEGRITY-018");
console.log("==================================================================\n");

const RADAR_PAGE_PATH = path.join(__dirname, "..", "app", "radar", "page.tsx");
const radarSrc = fs.readFileSync(RADAR_PAGE_PATH, "utf8");

// ---------------------------------------------------------------------------
// 1. Highest-Ranked Candidate Scoring 45/100
// ---------------------------------------------------------------------------
console.log("1. Verifying Candidate Scoring 45/100 Presentation Invariants...");

const pres45 = resolveRadarScorePresentation(45, false, null);
assert.strictEqual(pres45.tier, "DEFENSIVE", "Score 45 must resolve to DEFENSIVE tier");
assert.strictEqual(pres45.colorClass, "text-rose-400", "Score 45 text color must be text-rose-400");
assert.ok(
  pres45.badgeClass.includes("text-rose-400") && pres45.badgeClass.includes("bg-rose-950/60"),
  "Score 45 badgeClass must use rose styling, never emerald"
);
assert.ok(
  !pres45.badgeClass.includes("text-emerald-400"),
  "Score 45 badgeClass must NOT contain text-emerald-400"
);
assert.ok(
  pres45.tooltip.includes("Relative discovery ranking — not an actionable buy recommendation"),
  "Score 45 tooltip must explicitly state: 'Relative discovery ranking — not an actionable buy recommendation'"
);
console.log("   [PASS] 1.1 Candidate scoring 45/100 strictly resolves to DEFENSIVE rose presentation with explicit warning tooltip");

// ---------------------------------------------------------------------------
// 2. Defensive-Tier Styling on Desktop and Mobile
// ---------------------------------------------------------------------------
console.log("\n2. Verifying Desktop Table and Mobile Card Markup Invariants...");

assert.ok(
  radarSrc.includes("resolveRadarScorePresentation"),
  "Radar page must invoke resolveRadarScorePresentation"
);
assert.ok(
  radarSrc.includes("asset.confluenceScore,\n                              asset.executionStatus === 'DELISTED_SECURITY',") ||
  radarSrc.includes("asset.confluenceScore,") && radarSrc.includes("asset.executionStatus === 'DELISTED_SECURITY'"),
  "Radar desktop & mobile must pass confluenceScore and delisted status into resolveRadarScorePresentation"
);
// Confirm neither desktop table nor mobile card hardcodes emerald styling for non-delisted assets via ternary
assert.ok(
  !radarSrc.includes("? 'text-rose-400 bg-rose-950/60 border border-rose-800/60'"),
  "Radar desktop and mobile must NOT use old hardcoded delisted-vs-emerald ternary for score styling"
);
console.log("   [PASS] 2.1 Both desktop table and mobile card dynamically consume resolveRadarScorePresentation");

// ---------------------------------------------------------------------------
// 3. Exact Canonical Tier Boundaries (48/49, 64/65, 79/80)
// ---------------------------------------------------------------------------
console.log("\n3. Verifying Canonical Confluence Tier Boundaries...");

// 0-48: Defensive / Capital Preservation (Rose)
const pres0 = resolveRadarScorePresentation(0, false, null);
assert.strictEqual(pres0.tier, "DEFENSIVE");
assert.strictEqual(pres0.colorClass, "text-rose-400");

const pres48 = resolveRadarScorePresentation(48, false, null);
assert.strictEqual(pres48.tier, "DEFENSIVE", "Boundary 48 must be DEFENSIVE");
assert.strictEqual(pres48.colorClass, "text-rose-400");

// 49-64: Selective / Range-Bound (Amber)
const pres49 = resolveRadarScorePresentation(49, false, null);
assert.strictEqual(pres49.tier, "SELECTIVE", "Boundary 49 must be SELECTIVE");
assert.strictEqual(pres49.colorClass, "text-amber-400");

const pres64 = resolveRadarScorePresentation(64, false, null);
assert.strictEqual(pres64.tier, "SELECTIVE", "Boundary 64 must be SELECTIVE");
assert.strictEqual(pres64.colorClass, "text-amber-400");

// 65-79: Constructive (Cyan)
const pres65 = resolveRadarScorePresentation(65, false, null);
assert.strictEqual(pres65.tier, "CONSTRUCTIVE", "Boundary 65 must be CONSTRUCTIVE");
assert.strictEqual(pres65.colorClass, "text-cyan-400");

const pres79 = resolveRadarScorePresentation(79, false, null);
assert.strictEqual(pres79.tier, "CONSTRUCTIVE", "Boundary 79 must be CONSTRUCTIVE");
assert.strictEqual(pres79.colorClass, "text-cyan-400");

// 80-100: High Conviction (Emerald)
const pres80 = resolveRadarScorePresentation(80, false, null);
assert.strictEqual(pres80.tier, "HIGH_CONVICTION", "Boundary 80 must be HIGH_CONVICTION");
assert.strictEqual(pres80.colorClass, "text-emerald-400");

const pres100 = resolveRadarScorePresentation(100, false, null);
assert.strictEqual(pres100.tier, "HIGH_CONVICTION", "Boundary 100 must be HIGH_CONVICTION");
assert.strictEqual(pres100.colorClass, "text-emerald-400");

// Null, negative, NaN -> UNAVAILABLE
const presNull = resolveRadarScorePresentation(null, false, null);
assert.strictEqual(presNull.tier, "UNAVAILABLE");
assert.strictEqual(presNull.label, "—");

const presNeg = resolveRadarScorePresentation(-1, false, null);
assert.strictEqual(presNeg.tier, "UNAVAILABLE");

// Delisted -> DEFENSIVE
const presDelisted = resolveRadarScorePresentation(95, true, null);
assert.strictEqual(presDelisted.tier, "DEFENSIVE", "Delisted security must always resolve to DEFENSIVE");
assert.strictEqual(presDelisted.colorClass, "text-rose-400");
assert.strictEqual(presDelisted.tooltip, "Delisted Security · Trading Suspended");

console.log("   [PASS] 3.1 All boundaries (48/49, 64/65, 79/80) and edge cases strictly adhere to canonical ConfluenceEngine tiers");

// ---------------------------------------------------------------------------
// 4. READY_TO_BUY Screening Status Label Invariant
// ---------------------------------------------------------------------------
console.log("\n4. Verifying Screening Status Copy Invariants...");

const rtbLabel = formatRadarExecutionStatus("READY_TO_BUY" as any);
assert.strictEqual(
  rtbLabel,
  "NEAR SCREENING ZONE",
  "formatRadarExecutionStatus('READY_TO_BUY') must return 'NEAR SCREENING ZONE', never 'READY TO BUY'"
);
assert.notStrictEqual(
  rtbLabel,
  "READY TO BUY",
  "Radar candidate copy must NEVER be 'READY TO BUY'"
);

const buyZoneLabel = formatRadarExecutionStatus("IN_BUY_ZONE" as any);
assert.strictEqual(buyZoneLabel, "NEAR SCREENING ZONE");

console.log("   [PASS] 4.1 'READY_TO_BUY' execution directive cleanly demoted to 'NEAR SCREENING ZONE'");

// ---------------------------------------------------------------------------
// 5. Non-Actionable Discovery Candidate Invariants
// ---------------------------------------------------------------------------
console.log("\n5. Verifying Discovery Candidate Actionability Safeguards...");

assert.ok(
  radarSrc.includes("isActionable: false, // Invariant: Radar is a discovery scanner; canonical actionability reserved for DecisionTrace"),
  "Radar candidates must have isActionable: false strictly hardcoded"
);
assert.ok(
  radarSrc.includes("canSizeTrade: false,"),
  "Radar candidates must have canSizeTrade: false strictly hardcoded"
);
assert.ok(
  radarSrc.includes("Discovery Candidate · Relative discovery ranking — not an actionable buy recommendation"),
  "Radar hero card must explicitly state: 'Discovery Candidate · Relative discovery ranking — not an actionable buy recommendation'"
);

console.log("   [PASS] 5.1 Actionability gates and explicit non-recommendation copy verified");

// ---------------------------------------------------------------------------
// 6. Zero Qualifying Actionable Opportunities Disambiguation
// ---------------------------------------------------------------------------
console.log("\n6. Verifying Sorting and Zero-Actionable Relative Ranking Reminders...");

assert.ok(
  radarSrc.includes("relative discovery ranking — not an actionable buy recommendation"),
  "Radar sorting explanation must declare: 'relative discovery ranking — not an actionable buy recommendation'"
);

console.log("   [PASS] 6.1 Relative discovery ranking explicitly distinguished from actionable trade setups");

// ---------------------------------------------------------------------------
// 7. Synthetic, Stale or Unverified Evidence Fails Closed
// ---------------------------------------------------------------------------
console.log("\n7. Verifying Epistemic Fail-Closed Handling...");

assert.strictEqual(
  formatRadarExecutionStatus("UNVERIFIED_ASSET" as any),
  "UNVERIFIED ASSET"
);
assert.strictEqual(
  formatRadarExecutionStatus("STALE_MARKET_DATA" as any),
  "STALE MARKET DATA"
);
assert.strictEqual(
  formatRadarExecutionStatus("DELISTED_SECURITY" as any),
  "DELISTED / INACTIVE"
);

console.log("   [PASS] 7.1 Unverified, stale, and delisted statuses fail-closed cleanly");

// ---------------------------------------------------------------------------
// 8. Cross-Component Scoring and Provenance Parity
// ---------------------------------------------------------------------------
console.log("\n8. Verifying Backend ConfluenceEngine Parity...");

// Verify backend badgeColor override respect
const presBackendAmber = resolveRadarScorePresentation(85, false, "amber");
assert.strictEqual(
  presBackendAmber.tier,
  "SELECTIVE",
  "When backend flags incomplete pillars with amber, frontend honors selective amber tier"
);

console.log("   [PASS] 8.1 Backend badgeColor overrides and parity cleanly preserved");

console.log("\n==================================================================");
console.log("ALL ARX RADAR RECOMMENDATION INTEGRITY TESTS PASSED (PASS)");
console.log("==================================================================\n");
