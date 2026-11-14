import assert from "node:assert";
import {
  resolveRadarCandidateRationale,
  formatRadarExecutionStatus,
  resolveRadarScorePresentation,
} from "../lib/radarPresentation";

console.log("Starting Radar Candidate Rationale & Contextual Learn Hub Deep-Links Test Suite (Issue #230)...\n");

// ---------------------------------------------------------------------------
// 1. Candidate Rationale Deduplication & Specificity
// ---------------------------------------------------------------------------
console.log("1. Testing candidate rationale deduplication across diverse candidate profiles...");

const nvdaRationale = resolveRadarCandidateRationale({
  ticker: "NVDA",
  name: "NVIDIA Corporation",
  sector: "Technology",
  vcpStage: "Pivot Breakout",
  volumeDryUpPct: 45,
  rvol: "2.4x",
  confluenceScore: 78,
  categories: ["VCP", "SMART_MONEY"],
  executionStatus: "IN_BUY_ZONE",
  riskRewardRatio: 3.2,
});

const msftRationale = resolveRadarCandidateRationale({
  ticker: "MSFT",
  name: "Microsoft Corporation",
  sector: "Technology",
  vcpStage: "Stage 2 Base",
  volumeDryUpPct: 30,
  rvol: "1.2x",
  confluenceScore: 72,
  categories: ["VCP", "VALUE_GARP"],
  executionStatus: "IN_BUY_ZONE",
  riskRewardRatio: 2.5,
});

const jpmRationale = resolveRadarCandidateRationale({
  ticker: "JPM",
  name: "JPMorgan Chase & Co.",
  sector: "Financial Services",
  vcpStage: "Base Under Evaluation",
  confluenceScore: 68,
  categories: ["VALUE_GARP"],
  executionStatus: "IN_BUY_ZONE",
  riskRewardRatio: 2.1,
});

// 1.1 Rationales must not be identical strings
assert.notStrictEqual(
  nvdaRationale.summary,
  msftRationale.summary,
  "Candidate rationales must be deduplicated and asset-specific"
);
assert.notStrictEqual(
  msftRationale.summary,
  jpmRationale.summary,
  "Candidate rationales must differ across technical and fundamental categories"
);

// 1.2 Rationales must contain asset-specific indicators
assert(
  nvdaRationale.summary.includes("NVDA"),
  "NVDA rationale must reference its own ticker"
);
assert(
  nvdaRationale.summary.includes("Pivot Breakout"),
  "NVDA rationale must reflect its specific VCP pattern"
);
assert(
  nvdaRationale.summary.includes("45% volume dry-up"),
  "NVDA rationale must disclose specific volume dry-up metric"
);
assert(
  nvdaRationale.summary.includes("2.4x relative volume"),
  "NVDA rationale must disclose relative volume momentum"
);

console.log("  [PASS] Candidate rationales dynamically adapt with asset-specific technical and fundamental context.\n");

// ---------------------------------------------------------------------------
// 2. Contextual Learn Hub & Glossary Deep-Links
// ---------------------------------------------------------------------------
console.log("2. Testing contextual Learn Hub deep-links wiring...");

// 2.1 VCP + Smart Money candidate must link to Technical and Smart Money modules
const nvdaHrefs = nvdaRationale.learnLinks.map((l) => l.href);
assert(
  nvdaHrefs.includes("/guide#module-technical"),
  "VCP candidate must link to Field Manual Technical Module"
);
assert(
  nvdaHrefs.includes("/glossary#minervini-vcp"),
  "VCP candidate must link to Minervini VCP Glossary Term"
);
assert(
  nvdaHrefs.includes("/guide#module-smart-money"),
  "Smart Money candidate must link to Field Manual Smart Money Module"
);
assert(
  nvdaHrefs.includes("/glossary#stock-act"),
  "Smart Money candidate must link to STOCK Act Glossary Term"
);
assert(
  nvdaHrefs.includes("/glossary#arx-model"),
  "All candidates must link to Confluence Ranking / ARX Model Guide"
);
assert(
  nvdaHrefs.includes("/glossary#cornish-fisher-var"),
  "Candidate with >= 2:1 RR must link to Cornish-Fisher M-VaR Risk Fencing"
);

// 2.2 Value GARP candidate must link to Fundamental module and Piotroski F-Score
const jpmHrefs = jpmRationale.learnLinks.map((l) => l.href);
assert(
  jpmHrefs.includes("/guide#module-fundamental"),
  "Value GARP candidate must link to Fundamental Module"
);
assert(
  jpmHrefs.includes("/glossary#piotroski-f-score"),
  "Value GARP candidate must link to Piotroski F-Score Glossary Term"
);
assert(
  !jpmHrefs.includes("/guide#module-smart-money"),
  "Pure Value GARP candidate must not link to Smart Money module"
);

console.log("  [PASS] Contextual deep-links accurately target published Learn Hub and Glossary anchors.\n");

// ---------------------------------------------------------------------------
// 3. Delisted & Unverified Security Edge Cases
// ---------------------------------------------------------------------------
console.log("3. Testing delisted and unverified security edge cases...");

const delistedRationale = resolveRadarCandidateRationale({
  ticker: "CEG_OLD",
  name: "Constellation Acquisition",
  executionStatus: "DELISTED_SECURITY",
  confluenceScore: null,
});

assert.strictEqual(
  delistedRationale.headline,
  "Trading Suspended",
  "Delisted security must have 'Trading Suspended' headline"
);
assert(
  delistedRationale.summary.includes("Trading suspended for delisted security"),
  "Delisted security summary must explain trade suspension"
);
assert(
  delistedRationale.learnLinks.some((l) => l.href === "/guide#module-foundations"),
  "Delisted security must link to audit provenance in foundations module"
);

const unverifiedRationale = resolveRadarCandidateRationale({
  ticker: "XYZ",
  executionStatus: "UNVERIFIED_ASSET",
  confluenceScore: null,
});

assert.strictEqual(
  unverifiedRationale.headline,
  "Market Feed Unverified",
  "Unverified security must have 'Market Feed Unverified' headline"
);
assert(
  unverifiedRationale.summary.includes("feed and trade tape unverified"),
  "Unverified security summary must explain market data suppression"
);

console.log("  [PASS] Delisted and unverified assets fail closed with truthful audit and provenance disclosures.\n");

// ---------------------------------------------------------------------------
// 4. Recommendation Integrity & Safeguard Invariants
// ---------------------------------------------------------------------------
console.log("4. Testing recommendation integrity and safeguard invariants...");

// 4.1 formatRadarExecutionStatus must never output 'READY TO BUY'
assert.strictEqual(
  formatRadarExecutionStatus("READY_TO_BUY" as any),
  "NEAR SCREENING ZONE",
  "formatRadarExecutionStatus must demote READY_TO_BUY to NEAR SCREENING ZONE"
);
assert.strictEqual(
  formatRadarExecutionStatus("IN_BUY_ZONE" as any),
  "NEAR SCREENING ZONE",
  "formatRadarExecutionStatus must format IN_BUY_ZONE as NEAR SCREENING ZONE"
);

// 4.2 Score 45/100 must resolve to DEFENSIVE rose presentation
const score45 = resolveRadarScorePresentation(45, false);
assert.strictEqual(
  score45.tier,
  "DEFENSIVE",
  "Score 45 must resolve to DEFENSIVE tier"
);
assert(
  score45.colorClass.includes("rose"),
  "Score 45 must have rose colorClass"
);
assert(
  score45.tooltip.includes("Relative discovery ranking — not an actionable buy recommendation"),
  "Score 45 must warn that discovery ranking is not an actionable buy recommendation"
);

// 4.3 Technical context must emphasize discovery ranking != actionable buy
assert(
  msftRationale.technicalContext.includes("discovery ranking is not an automatic buy signal"),
  "Technical context must remind user that discovery ranking is not an automatic buy signal"
);

console.log("  [PASS] Recommendation safety and non-actionability barriers strictly enforced.\n");

console.log("All Radar candidate rationale & contextual Learn Hub tests PASSED successfully!");
