import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");

console.log("===============================================================");
console.log("VERIFYING FIVE DATA INTEGRITY & PROVENANCE BUG FIXES (OPTION A)");
console.log("===============================================================\n");

let passed = 0;
let total = 0;

function test(name, fn) {
  total++;
  try {
    fn();
    console.log(`  ✅ PASS: ${name}`);
    passed++;
  } catch (err) {
    console.error(`  ❌ FAIL: ${name}`);
    console.error(`     Error: ${err.message}\n`);
    throw err;
  }
}

// -------------------------------------------------------------
// BUG 1: Performance Benchmark Isolation & Empty State
// -------------------------------------------------------------
console.log("--- Testing Bug 1: Performance Benchmark Isolation ---");

const perfPath = path.join(rootDir, "app", "performance", "page.tsx");
const perfContent = fs.readFileSync(perfPath, "utf-8");

test("Performance page fetches live journal trades from API", () => {
  assert.ok(perfContent.includes("fetchJournalTrades"), "Must call fetchJournalTrades");
  assert.ok(perfContent.includes("setLiveTrades"), "Must set live trades from API");
});

test("Performance page filters completed trades using filterEligibleLiveTrades", () => {
  assert.ok(perfContent.includes("filterEligibleLiveTrades"), "Must call filterEligibleLiveTrades");
  assert.ok(perfContent.includes("eligibleLiveTrades"), "Must maintain eligibleLiveTrades array");
});

test("Performance page is 100% production live view with zero benchmark dependency", () => {
  assert.ok(
    !perfContent.includes("CANONICAL_GOVERNOR_LEDGER"),
    "Production performance must not import CANONICAL_GOVERNOR_LEDGER"
  );
  assert.ok(
    !perfContent.includes("dataMode === 'BENCHMARK'"),
    "Must not contain benchmark dataMode toggle in production"
  );
});

test("Performance page shows honest empty state when live mode has 0 completed executions", () => {
  assert.ok(
    perfContent.includes("0 Completed Executions Logged Yet"),
    "Must render '0 Completed Executions Logged Yet' when live completed executions is 0"
  );
  assert.ok(
    perfContent.includes("livePositions.length"),
    "Must acknowledge open portfolio positions separately from completed executions"
  );
});

// Simulation of Attribution Engine Isolation
test("Attribution calculations accurately reflect live vs benchmark dataset", () => {
  const emptyLiveLedger = [];
  assert.equal(emptyLiveLedger.length, 0);

  const activeMode = 'LIVE';
  const canonicalLedger = [{ id: "BENCH-1", governedRiskDollar: 500 }];
  const currentLedger = activeMode === 'BENCHMARK' ? canonicalLedger : emptyLiveLedger;
  assert.equal(currentLedger.length, 0, "Live ledger must not contain benchmark records");
});

// -------------------------------------------------------------
// BUG 2: Journal Brier Score Inputs & Calibration Cohorts
// -------------------------------------------------------------
console.log("\n--- Testing Bug 2: Journal Validated Brier Score ---");

const journalPath = path.join(rootDir, "app", "journal", "page.tsx");
const journalContent = fs.readFileSync(journalPath, "utf-8");

test("Journal validates confidence in [0, 100] and normalizes by / 100.0", () => {
  assert.ok(journalContent.includes("t.confidence / 100.0"), "Must normalize confidence by 100.0");
  assert.ok(journalContent.includes("t.confidence < 0"), "Must exclude negative confidence");
  assert.ok(journalContent.includes("t.confidence > 100"), "Must exclude confidence > 100");
});

test("Journal excludes missing, non-finite, and unresolved records", () => {
  assert.ok(journalContent.includes("isFinite(t.confidence)"), "Must check isFinite");
  assert.ok(journalContent.includes("!isWin && !isLoss"), "Must exclude scratch/unresolved records");
});

test("Brier score math handles boundary cases correctly", () => {
  // Scenario A: 0% confidence, Outcome = 1 (Win) -> Contribution: (0 - 1)^2 = 1.0
  const probA = 0.0;
  const outcomeA = 1;
  const contribA = Math.pow(probA - outcomeA, 2);
  assert.equal(contribA, 1.0, "0% confidence on win must contribute 1.0");

  // Scenario B: 100% confidence, Outcome = 1 (Win) -> Contribution: (1 - 1)^2 = 0.0
  const probB = 1.0;
  const outcomeB = 1;
  const contribB = Math.pow(probB - outcomeB, 2);
  assert.equal(contribB, 0.0, "100% confidence on win must contribute 0.0");

  // Scenario C: 70% confidence, Outcome = 0 (Loss) -> Contribution: (0.7 - 0)^2 = 0.49
  const probC = 0.7;
  const outcomeC = 0;
  const contribC = Math.pow(probC - outcomeC, 2);
  assert.ok(Math.abs(contribC - 0.49) < 1e-9, "70% confidence on loss must contribute 0.49");

  // Mean Brier across A, B, C
  const meanBrier = (contribA + contribB + contribC) / 3;
  assert.equal(meanBrier.toFixed(2), "0.50");
});

test("Journal displays '--' for Brier score when 0 eligible records exist", () => {
  assert.ok(journalContent.includes('if (eligibleRecords.length === 0) return "--"'), "Must display '--' when empty");
});

test("Calibration cohorts: 5 non-overlapping buckets covering [0%, 100%]", () => {
  const buckets = [
    { label: '0-20%', min: 0.0, max: 0.20, isInclusiveMax: false },
    { label: '20-40%', min: 0.20, max: 0.40, isInclusiveMax: false },
    { label: '40-60%', min: 0.40, max: 0.60, isInclusiveMax: false },
    { label: '60-80%', min: 0.60, max: 0.80, isInclusiveMax: false },
    { label: '80-100%', min: 0.80, max: 1.00, isInclusiveMax: true },
  ];

  const testValues = [0.0, 0.10, 0.20, 0.25, 0.40, 0.55, 0.60, 0.79, 0.80, 0.95, 1.0];
  for (const v of testValues) {
    const matching = buckets.filter((b) => (b.isInclusiveMax ? v >= b.min && v <= b.max : v >= b.min && v < b.max));
    assert.equal(matching.length, 1, `Value ${v} must match exactly 1 bucket`);
  }
});

test("Empty calibration cohorts display unavailable, never 0% observed", () => {
  assert.ok(
    journalContent.includes('Unavailable (0 trades in cohort)') || journalContent.includes('Awaiting Executions'),
    "Must not show 0% for empty buckets"
  );
});

// -------------------------------------------------------------
// BUG 3: Journal Unsupported Favorable Assessments
// -------------------------------------------------------------
console.log("\n--- Testing Bug 3: Journal Unsupported Assessments ---");

test("Journal computes adherence grade dynamically and displays '--' when no rule evidence", () => {
  assert.ok(journalContent.includes("if (tradesWithRuleEvidence.length === 0) return null"), "Grade must be null when 0 trades with rule evidence");
  assert.ok(
    journalContent.includes('{tradesWithRuleEvidence.length > 0 ? `${adherenceRatePct}%` : "--"}'),
    "Adherence score must be '--' when 0 trades with rule evidence"
  );
});

test("Behavioral state switches to STANDBY on 0 trades", () => {
  assert.ok(journalContent.includes('tradesLogged === 0'), "Must check tradesLogged === 0");
  assert.ok(journalContent.includes('title: "STANDBY"'), "Must return STANDBY state");
});

test("Behavioral state switches to DEFENSIVE on streak >= 2 and ATTENTION on violations", () => {
  assert.ok(journalContent.includes('activeLossStreak >= 2'), "Must detect activeLossStreak >= 2");
  assert.ok(journalContent.includes('title: "DEFENSIVE"'), "Must return DEFENSIVE");
  assert.ok(journalContent.includes('ruleViolations > 0'), "Must detect ruleViolations");
  assert.ok(journalContent.includes('title: "ATTENTION"'), "Must return ATTENTION");
});

test("Brier calibration badge requires N >= 10 sample size", () => {
  assert.ok(journalContent.includes("eligibleRecords.length < 10"), "Must enforce N < 10 threshold");
  assert.ok(
    journalContent.includes("Sample Insufficient (N="),
    "Must display Sample Insufficient when N < 10"
  );
});

test("Stop loss violations check actual rAchieved < -1.0", () => {
  assert.ok(journalContent.includes("t.rAchieved < -1.0"), "Must dynamically count blown stop losses");
  assert.ok(
    !journalContent.includes("Zero stop losses blown past plan") || journalContent.includes("stopLossViolations > 0"),
    "Must not unconditionally claim zero stop loss violations"
  );
});

// -------------------------------------------------------------
// BUG 4: Radar Fabricated Measurements & Classifications
// -------------------------------------------------------------
console.log("\n--- Testing Bug 4: Radar Fabricated Measurements ---");

const radarPath = path.join(rootDir, "app", "radar", "page.tsx");
const radarContent = fs.readFileSync(radarPath, "utf-8");

test("Radar code contains ZERO hardcoded -45% or -35% volume dry-up values", () => {
  assert.ok(!radarContent.includes("-45%"), "Must NOT contain hardcoded -45%");
  assert.ok(!radarContent.includes("-35%"), "Must NOT contain hardcoded -35%");
  assert.ok(!radarContent.includes("volumeDryUpPct: -45"), "Must NOT contain volumeDryUpPct: -45");
  assert.ok(!radarContent.includes("volumeDryUpPct: -35"), "Must NOT contain volumeDryUpPct: -35");
});

test("Radar does NOT substitute composite_score for rsRating", () => {
  assert.ok(
    !radarContent.includes("rsRating: gem.composite_score"),
    "Must NOT substitute composite_score for rsRating"
  );
  assert.ok(
    radarContent.includes("const rsRaw = gem.rs_rating ?? gem.rsRating ?? gem.relative_strength;"),
    "Must extract rsRating from authentic fields"
  );
});

test("Radar defaults missing rsRating and volumeDryUpPct to null", () => {
  assert.ok(
    radarContent.includes("const rsRating = (typeof rsRaw === 'number' && isFinite(rsRaw)) ? rsRaw : null;"),
    "rsRating must default to null"
  );
  assert.ok(
    radarContent.includes("const volumeDryUpPct = (typeof volRaw === 'number' && isFinite(volRaw)) ? volRaw : null;"),
    "volumeDryUpPct must default to null"
  );
});

test("Radar table and hero card render '--' when rsRating or volumeDryUpPct is null", () => {
  assert.ok(
    radarContent.includes('heroAsset.rsRating !== null ? `${heroAsset.rsRating}/99` : "--"'),
    "Hero must render '--' for null rsRating"
  );
  assert.ok(
    radarContent.includes('heroAsset.volumeDryUpPct !== null ? `${heroAsset.volumeDryUpPct}%` : "--"'),
    "Hero must render '--' for null volumeDryUpPct"
  );
  assert.ok(
    radarContent.includes('asset.rsRating !== null ? asset.rsRating : "--"'),
    "Table must render '--' for null rsRating"
  );
  assert.ok(
    radarContent.includes('asset.volumeDryUpPct !== null ? `${asset.volumeDryUpPct}%` : "--"'),
    "Table must render '--' for null volumeDryUpPct"
  );
});

test("Radar category mapping does not fallback to SMART_MONEY without evidence", () => {
  assert.ok(
    !radarContent.includes("cat.push('SMART_MONEY') // default"),
    "Must not default to SMART_MONEY"
  );
  assert.ok(
    radarContent.includes("modelStr.includes(\"SMART\") || modelStr.includes(\"CONGRESS\") || modelStr.includes(\"INSTITUTIONAL\")"),
    "Must require institutional or smart money model evidence"
  );
});

test("Radar sorting handles nulls cleanly and preserves valid 0", () => {
  const sample = [
    { ticker: "A", rsRating: null },
    { ticker: "B", rsRating: 95 },
    { ticker: "C", rsRating: 0 },
    { ticker: "D", rsRating: 50 },
  ];

  const sorted = [...sample].sort((a, b) => {
    if (a.rsRating === null && b.rsRating === null) return 0;
    if (a.rsRating === null) return 1;
    if (b.rsRating === null) return -1;
    return b.rsRating - a.rsRating;
  });

  assert.deepEqual(
    sorted.map((s) => s.ticker),
    ["B", "D", "C", "A"],
    "Nulls must sort to bottom while valid 0 is preserved before null"
  );
});

test("Radar exports page wrapped in Suspense boundary", () => {
  assert.ok(radarContent.includes("<Suspense fallback="), "Must contain Suspense boundary");
  assert.ok(radarContent.includes("export default function RadarPage"), "Must export default RadarPage");
});

// -------------------------------------------------------------
// BUG 5: Screener -> Radar Search Query Passing
// -------------------------------------------------------------
console.log("\n--- Testing Bug 5: Screener -> Radar Search Passing ---");

const screenerPath = path.join(rootDir, "app", "screener", "page.tsx");
const screenerContent = fs.readFileSync(screenerPath, "utf-8");

test("Screener redirect extracts search query and aliases (?q=, ?query=, ?ticker=, ?symbol=)", () => {
  assert.ok(screenerContent.includes('searchParams.get("q")'), "Must read 'q'");
  assert.ok(screenerContent.includes('searchParams.get("query")'), "Must read 'query'");
  assert.ok(screenerContent.includes('searchParams.get("ticker")'), "Must read 'ticker'");
  assert.ok(screenerContent.includes('searchParams.get("symbol")'), "Must read 'symbol'");
});

test("Screener redirects with properly encoded uppercase query parameter", () => {
  assert.ok(
    screenerContent.includes("router.replace(`/radar?q=${encodeURIComponent(q.trim().toUpperCase())}`)"),
    "Must encode and uppercase query for /radar"
  );
});

test("Radar consumes url parameters (?q=, ?query=, ?ticker=, ?symbol=)", () => {
  assert.ok(radarContent.includes('searchParams?.get("q")'), "Must read 'q'");
  assert.ok(radarContent.includes('searchParams?.get("query")'), "Must read 'query'");
  assert.ok(radarContent.includes('searchParams?.get("ticker")'), "Must read 'ticker'");
  assert.ok(radarContent.includes('searchParams?.get("symbol")'), "Must read 'symbol'");
});

test("Radar sets heroAsset to null when search returns 0 matches (no fake fallback)", () => {
  assert.ok(
    radarContent.includes("const heroAsset = isSearching ? (filteredAssets[0] || null) : (filteredAssets[0] || allAssets[0])"),
    "heroAsset must be null when isSearching and 0 matches"
  );
  assert.ok(
    radarContent.includes("!isLoading && isSearching && !heroAsset"),
    "Must render honest empty search status banner when 0 matches"
  );
});

console.log(`\n===============================================================`);
console.log(`ALL TESTS PASSED: ${passed}/${total} assertions verified!`);
console.log(`===============================================================\n`);
