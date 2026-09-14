import assert from "node:assert";
import { evaluateLevelRelation } from "../lib/reclaimSemantics";
import { generateQuantitativeInsight } from "../lib/insightGenerator";
import { deriveAssessmentState } from "../lib/assessmentEngine";

console.log("================================================================");
console.log("   RECLAIM SEMANTICS & CROSS-MODULE CONSISTENCY VERIFICATION   ");
console.log("================================================================\n");

// 1. Pure Helper Unit Tests
console.log("[Suite 1] Pure Level Relation Evaluator");

// Test 1a: Price clearly below reference level
{
  const res = evaluateLevelRelation(140.0, 150.0, "50-Day SMA", "AAPL");
  assert.strictEqual(res.status, "BELOW");
  assert(res.reclaimMilestone.includes("needs to reclaim $150.00"), "Should mention reclaim when below");
  assert(res.headlineExplanationWatch.includes("reclaim of $150.00"), "Headline should mention reclaim when below");
  assert.strictEqual(res.uiBadgeLabel, "Must reclaim");
  console.log("  ✔ PASS: Price clearly below reference level permits reclaim language");
}

// Test 1b: Price exactly at reference level (boundary condition)
{
  const res = evaluateLevelRelation(150.0, 150.0, "50-Day SMA", "AAPL");
  assert.strictEqual(res.status, "AT_LEVEL");
  assert(!res.reclaimMilestone.toLowerCase().includes("needs to reclaim"), "Should NOT say needs to reclaim at level");
  assert(res.reclaimMilestone.includes("testing the 50-Day SMA"), "Should describe testing the level");
  assert.strictEqual(res.uiBadgeLabel, "Testing Level");
  console.log("  ✔ PASS: Price at boundary outputs testing language without reclaim-needed claim");
}

// Test 1c: Price slightly above reference level (e.g. +0.5%)
{
  const res = evaluateLevelRelation(150.75, 150.0, "50-Day SMA", "AAPL");
  assert.strictEqual(res.status, "ABOVE");
  assert(!res.reclaimMilestone.toLowerCase().includes("needs to reclaim"), "Must NOT say needs to reclaim when above");
  assert(!res.reclaimMilestone.toLowerCase().includes("must reclaim"), "Must NOT say must reclaim when above");
  assert(!res.reclaimMilestone.toLowerCase().includes("awaiting reclaim"), "Must NOT say awaiting reclaim when above");
  assert(res.reclaimMilestone.includes("holding constructively above"), "Should say holding constructively above");
  assert.strictEqual(res.uiBadgeLabel, "Holding above");
  console.log("  ✔ PASS: Price slightly above level outputs holding-above without reclaim language");
}

// Test 1d: Price well above reference level (e.g. +15%)
{
  const res = evaluateLevelRelation(172.50, 150.0, "50-Day SMA", "AAPL");
  assert.strictEqual(res.status, "ABOVE");
  assert(!res.reclaimMilestone.toLowerCase().includes("needs to reclaim"), "Must NOT say needs to reclaim when well above");
  assert(!res.headlineExplanationWatch.toLowerCase().includes("reclaim"), "Headline must NOT say reclaim when well above");
  assert(res.reclaimMilestone.includes("holding constructively above"), "Should say holding constructively above");
  assert.strictEqual(res.uiBadgeLabel, "Holding above");
  console.log("  ✔ PASS: Price well above level outputs holding-above without reclaim language");
}

// Test 1e: Reference level or price unavailable / NaN / null / zero / Infinity
{
  const res1 = evaluateLevelRelation(undefined, 150.0, "50-Day SMA", "AAPL");
  assert.strictEqual(res1.status, "UNAVAILABLE");
  assert(res1.reclaimMilestone.includes("unavailable"), "Should report unavailable when price undefined");
  assert.strictEqual(res1.uiBadgeLabel, "Unassessed");
  assert(!res1.reclaimMilestone.includes("insufficient trading history"), "Must not invent causes for missing data");
  assert(!res1.headlineExplanationWatch.includes("awaiting constructive consolidation"), "Must not invent market conditions");

  const res2 = evaluateLevelRelation(150.0, undefined, "50-Day SMA", "AAPL");
  assert.strictEqual(res2.status, "UNAVAILABLE");
  assert(res2.reclaimMilestone.includes("unavailable"), "Should report unavailable when level undefined");

  const res3 = evaluateLevelRelation(0, 150.0, "50-Day SMA", "AAPL");
  assert.strictEqual(res3.status, "UNAVAILABLE");

  const res4 = evaluateLevelRelation(150.0, null, "50-Day SMA", "AAPL");
  assert.strictEqual(res4.status, "UNAVAILABLE");

  const resInf1 = evaluateLevelRelation(Infinity, 150.0, "50-Day SMA", "AAPL");
  assert.strictEqual(resInf1.status, "UNAVAILABLE", "Infinity price must yield UNAVAILABLE");

  const resInf2 = evaluateLevelRelation(150.0, Infinity, "50-Day SMA", "AAPL");
  assert.strictEqual(resInf2.status, "UNAVAILABLE", "Infinity reference level must yield UNAVAILABLE");

  const resNegInf = evaluateLevelRelation(-Infinity, 150.0, "50-Day SMA", "AAPL");
  assert.strictEqual(resNegInf.status, "UNAVAILABLE", "-Infinity price must yield UNAVAILABLE");
  console.log("  ✔ PASS: Missing/NaN/null/Infinity data honestly reports unavailable with zero fabricated certainty");
}

// Test 1f: Exact boundary equality check
{
  const res = evaluateLevelRelation(100.0, 100.0, "50-Day SMA", "NVDA");
  assert.strictEqual(res.status, "AT_LEVEL");
  assert.strictEqual(res.uiBadgeLabel, "Testing Level", "Boundary equality must yield 'Testing Level'");
  console.log("  ✔ PASS: Exact level boundary produces 'Testing Level' badge");
}

// 2. Integration with insightGenerator
console.log("\n[Suite 2] insightGenerator Behavioral Semantics");

// Test 2a: Asset above 50D SMA (AAPL at $325 with SMA50 $317.07 from S03)
{
  const mockCandles = Array.from({ length: 55 }, (_, i) => ({
    date: `2026-08-${String(i + 1).padStart(2, '0')}`,
    open: 315.0,
    high: 326.0,
    low: 314.0,
    close: 317.07,
    volume: 1000000,
  }));

  const insight = generateQuantitativeInsight(
    "AAPL",
    "Apple Inc.",
    325.0, // Current price safely ABOVE $317.07
    1.25,
    73,
    undefined,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    mockCandles,
    "live"
  );

  assert.strictEqual(insight.standard.keyLevels.sma50, 317.07);
  assert(insight.price > (insight.standard.keyLevels.sma50 as number), "Precondition: price > sma50");
  
  // INVARIANT: When price > sma50, reclaimMilestone must NEVER tell the user it needs to reclaim that level!
  const reclaimText = insight.human.reclaimMilestone;
  assert(!reclaimText.toLowerCase().includes("needs to reclaim"), `Violated: ${reclaimText}`);
  assert(!reclaimText.toLowerCase().includes("must reclaim"), `Violated: ${reclaimText}`);
  assert(reclaimText.includes("holding constructively above"), `Expected holding above, got: ${reclaimText}`);
  console.log("  ✔ PASS: AAPL S03 defect resolved: price > sma50 never outputs 'needs to reclaim'");
}

// Test 2b: Asset below 50D SMA (Bearish / Stage 4 asset)
{
  const mockCandles = Array.from({ length: 55 }, (_, i) => ({
    date: `2026-08-${String(i + 1).padStart(2, '0')}`,
    open: 330.0,
    high: 335.0,
    low: 325.0,
    close: 330.0,
    volume: 1000000,
  }));

  const insight = generateQuantitativeInsight(
    "DOWN",
    "Down Corp.",
    300.0, // Current price well BELOW $330.00
    -2.5,
    35,
    undefined,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    mockCandles,
    "live"
  );

  assert(insight.price < (insight.standard.keyLevels.sma50 as number), "Precondition: price < sma50");
  const reclaimText = insight.human.reclaimMilestone;
  assert(reclaimText.includes("needs to reclaim"), `Expected reclaim language, got: ${reclaimText}`);
  console.log("  ✔ PASS: Asset below sma50 correctly preserves 'needs to reclaim' guidance");
}

// Test 2c: Price Trend whyPill decouples from stage (Stage 1 with price below SMA50 outputs Weak, not Healthy)
{
  const mockCandles = Array.from({ length: 55 }, (_, i) => ({
    date: `2026-08-${String(i + 1).padStart(2, '0')}`,
    open: 100.0,
    high: 105.0,
    low: 95.0,
    close: 100.0,
    volume: 500000,
  }));

  const insight = generateQuantitativeInsight(
    "BASE",
    "Base Asset Corp.",
    92.0, // price BELOW $100.00 SMA50
    -0.5,
    50,
    1, // STAGE 1 (NOT Stage 4)
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    mockCandles,
    "live"
  );

  const priceTrendPill = insight.human.whyPills.find((p) => p.category === "Price Trend");
  assert(!!priceTrendPill, "Price Trend whyPill must exist");
  assert.strictEqual(priceTrendPill.status, "Weak", "Price Trend must be Weak when price < SMA50 even if stage is not 4");
  assert(priceTrendPill.description.includes("below the 50-day moving average"), "Must describe price below 50-day moving average");
  assert(!priceTrendPill.description.includes("holding firmly above"), "Must NOT claim holding firmly above");
  console.log("  ✔ PASS: Price Trend whyPill decouples from stage: evaluates actual level relation directly");
}

// Test 2d: Price Trend whyPill at exact boundary produces Neutral and testing language
{
  const mockCandles = Array.from({ length: 55 }, (_, i) => ({
    date: `2026-08-${String(i + 1).padStart(2, '0')}`,
    open: 100.0,
    high: 105.0,
    low: 95.0,
    close: 100.0,
    volume: 500000,
  }));

  const insight = generateQuantitativeInsight(
    "FLAT",
    "Flat Asset Corp.",
    100.0, // price EXACTLY at $100.00 SMA50
    0.0,
    50,
    2,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    mockCandles,
    "live"
  );

  const priceTrendPill = insight.human.whyPills.find((p) => p.category === "Price Trend");
  assert(!!priceTrendPill, "Price Trend whyPill must exist");
  assert.strictEqual(priceTrendPill.status, "Neutral", "Price Trend must be Neutral at level boundary");
  assert(priceTrendPill.description.includes("testing the 50-day moving average"), "Must describe testing 50-day moving average");
  console.log("  ✔ PASS: Price Trend whyPill at exact boundary outputs Neutral and testing language");
}

// 3. Integration with assessmentEngine
console.log("\n[Suite 3] assessmentEngine Behavioral Semantics");

// Test 3a: assessmentEngine with price > reclaimMilestonePrice in WATCH state
{
  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corporation",
    currentPrice: 125.0,
    changePct: 0.8,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: [
      {
        domainId: "trend",
        domainName: "Price Trend",
        availability: "AVAILABLE",
        status: "MIXED",
        pointImpact: 0,
        importanceLevel: "HIGH",
        observation: "Trend consolidating.",
        modelRule: "Consolidation rule.",
        evidence: [],
      },
      {
        domainId: "health",
        domainName: "Company Health",
        availability: "AVAILABLE",
        status: "FAVORABLE",
        pointImpact: 20,
        importanceLevel: "HIGH",
        observation: "Health solid.",
        modelRule: "Health rule.",
        evidence: [],
      },
    ],
    reclaimMilestonePrice: 120.0, // price 125 > reclaimMilestone 120
  });

  assert.strictEqual(state.posture, "WATCH");
  // INVARIANT: When price > reclaimMilestonePrice, headlineExplanation must NOT say "reclaim of $120.00"
  assert(!state.headlineExplanation.toLowerCase().includes("reclaim of $120.00"), `Violated: ${state.headlineExplanation}`);
  assert(state.headlineExplanation.includes("Holding constructively above"), `Expected holding above, got: ${state.headlineExplanation}`);
  assert(!state.whatWouldChangeAssessment.toLowerCase().includes("reclaiming and holding"), `Violated: ${state.whatWouldChangeAssessment}`);
  console.log("  ✔ PASS: assessmentEngine WATCH state does not emit reclaim requirement when price > level");
}

// Test 3b: assessmentEngine with price < reclaimMilestonePrice in WATCH state
{
  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corporation",
    currentPrice: 110.0,
    changePct: -1.2,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: [
      {
        domainId: "trend",
        domainName: "Price Trend",
        availability: "AVAILABLE",
        status: "MIXED",
        pointImpact: 0,
        importanceLevel: "HIGH",
        observation: "Trend consolidating.",
        modelRule: "Consolidation rule.",
        evidence: [],
      },
      {
        domainId: "health",
        domainName: "Company Health",
        availability: "AVAILABLE",
        status: "FAVORABLE",
        pointImpact: 20,
        importanceLevel: "HIGH",
        observation: "Health solid.",
        modelRule: "Health rule.",
        evidence: [],
      },
    ],
    reclaimMilestonePrice: 120.0, // price 110 < reclaimMilestone 120
  });

  assert.strictEqual(state.posture, "WATCH");
  assert(state.headlineExplanation.includes("reclaim of $120.00"), `Expected reclaim requirement, got: ${state.headlineExplanation}`);
  assert(state.whatWouldChangeAssessment.includes("Reclaiming and holding above $120.00"), `Expected reclaim condition, got: ${state.whatWouldChangeAssessment}`);
  console.log("  ✔ PASS: assessmentEngine WATCH state properly emits reclaim requirement when price < level");
}

// 4. Cross-Module Consistency
console.log("\n[Suite 4] Cross-Module Consistency (Zero Contradictions)");
{
  const price = 325.0;
  const sma50 = 317.07;
  const mockCandles = Array.from({ length: 55 }, (_, i) => ({
    date: `2026-08-${String(i + 1).padStart(2, '0')}`,
    open: 315.0,
    high: 326.0,
    low: 314.0,
    close: sma50,
    volume: 1000000,
  }));

  const insight = generateQuantitativeInsight(
    "AAPL",
    "Apple Inc.",
    price,
    1.5,
    70,
    undefined,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    mockCandles,
    "live"
  );

  const assessmentState = deriveAssessmentState({
    symbol: "AAPL",
    companyName: "Apple Inc.",
    currentPrice: price,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: insight.terminalState.domains,
    reclaimMilestonePrice: sma50,
  });

  // Cross-check: Neither module may claim that AAPL needs to reclaim $317.07
  assert(!insight.human.reclaimMilestone.includes("needs to reclaim $317.07"), "insightGenerator contradiction detected");
  assert(!assessmentState.headlineExplanation.includes("reclaim of $317.07"), "assessmentEngine contradiction detected");
  console.log("  ✔ PASS: insightGenerator and assessmentEngine are 100% consistent: zero contradictory outputs");
}

console.log("\n================================================================");
console.log("All 10 Reclaim Semantics & Cross-Module Tests PASSED!");
console.log("================================================================");
