import assert from "node:assert";
import {
  generateFallbackAnalytics,
  fetchDirectYahooFinanceChart,
  AnalyticsResponse,
} from "../lib/api";
import { generateQuantitativeInsight } from "../lib/insightGenerator";
import {
  DecisionState,
  isDecisionActionable,
  isStatusActionable,
  createDegradedDecisionContext,
} from "../types/decisionContract";

console.log("Starting Client Fail-Closed Fallback & Canonical Context Test Suite...\n");

// ---------------------------------------------------------------------------
// 1. Verify generateFallbackAnalytics Fail-Closed Governance
// ---------------------------------------------------------------------------
console.log("1. Testing generateFallbackAnalytics fail-closed contract...");
const fallback = generateFallbackAnalytics("AAPL");

assert.strictEqual(fallback.degradedMode, true, "Fallback must declare degradedMode: true");
assert.strictEqual(fallback.decisionUnavailable, true, "Fallback must declare decisionUnavailable: true");
assert.strictEqual(fallback.optimalExecution?.execution_status, "UNAVAILABLE");
assert.strictEqual(fallback.optimalExecution?.optimal_entry_min, null);
assert.strictEqual(fallback.optimalExecution?.optimal_entry_max, null);
assert.strictEqual(fallback.optimalExecution?.stop_loss, null);
assert.strictEqual(fallback.optimalExecution?.take_profit_1, null);
assert.strictEqual(fallback.optimalExecution?.take_profit_2, null);
assert.strictEqual(fallback.decisionTrace?.decisionState, "UNVERIFIED");
assert.strictEqual(fallback.decisionTrace?.isActionable, false);
console.log("   [OK] generateFallbackAnalytics enforces fail-closed state");

// ---------------------------------------------------------------------------
// 2. Verify fetchDirectYahooFinanceChart Direct Client Fallback Authority
// ---------------------------------------------------------------------------
console.log("2. Testing fetchDirectYahooFinanceChart demotion to DISPLAY_ONLY_MARKET_DATA...");

// Mock global fetch to return 60 days of authentic mock Yahoo chart data
const originalFetch = global.fetch;

const mockTimestamps = Array.from({ length: 60 }, (_, i) => 1720000000 + i * 86400);
const mockCloses = Array.from({ length: 60 }, (_, i) => 150 + i * 0.5);
const mockOpens = mockCloses.map((c) => c - 0.2);
const mockHighs = mockCloses.map((c) => c + 1.0);
const mockLows = mockCloses.map((c) => c - 1.0);
const mockVolumes = Array.from({ length: 60 }, () => 1000000);

(global as any).fetch = async (url: string | URL | Request) => {
  const urlStr = url.toString();
  if (urlStr.includes("yahoo.com") || urlStr.includes("query1.finance.yahoo.com") || urlStr.includes("query2.finance.yahoo.com")) {
    return {
      ok: true,
      status: 200,
      json: async () => ({
        chart: {
          result: [
            {
              meta: {
                currency: "USD",
                symbol: "TSLA",
                regularMarketPrice: 220.5,
                regularMarketTime: 1725000000,
                previousClose: 218.0,
                shortName: "Tesla, Inc.",
              },
              timestamp: mockTimestamps,
              indicators: {
                quote: [
                  {
                    open: mockOpens,
                    high: mockHighs,
                    low: mockLows,
                    close: mockCloses,
                    volume: mockVolumes,
                  },
                ],
              },
            },
          ],
          error: null,
        },
      }),
    } as any;
  }
  return { ok: false, status: 500 } as any;
};

async function testDirectYahooFetcher() {
  try {
    const yfResult = await fetchDirectYahooFinanceChart("TSLA", "1y", "1d");
    assert(yfResult !== null, "Yahoo direct fetcher returned null with valid mock");

    // Must be marked as degraded display-only
    assert.strictEqual(yfResult.degradedMode, true, "Direct Yahoo fetch must have degradedMode: true");
    assert.strictEqual(yfResult.decisionUnavailable, true, "Direct Yahoo fetch must have decisionUnavailable: true");

    // ZERO synthetic trade corridor invariants
    const exec = yfResult.optimalExecution;
    assert(exec !== undefined, "optimalExecution must exist");
    assert.strictEqual(
      exec?.execution_status,
      "UNVERIFIED_ASSET",
      "Direct Yahoo fallback execution_status must be strictly UNVERIFIED_ASSET"
    );
    assert.notStrictEqual(
      exec?.execution_status,
      "IN_BUY_ZONE",
      "Direct Yahoo fallback must NEVER declare IN_BUY_ZONE"
    );
    assert.strictEqual(exec?.optimal_entry_min, null, "optimal_entry_min must be null");
    assert.strictEqual(exec?.optimal_entry_max, null, "optimal_entry_max must be null");
    assert.strictEqual(exec?.stop_loss, null, "stop_loss must be null");
    assert.strictEqual(exec?.take_profit_1, null, "take_profit_1 must be null");
    assert.strictEqual(exec?.take_profit_2, null, "take_profit_2 must be null");
    assert.strictEqual(exec?.risk_reward_ratio, null, "risk_reward_ratio must be null");

    // Decision trace must be UNVERIFIED and non-actionable
    assert.strictEqual(
      yfResult.decisionTrace?.decisionState,
      "UNVERIFIED",
      "decisionTrace.decisionState must be UNVERIFIED"
    );
    assert.strictEqual(
      yfResult.decisionTrace?.isActionable,
      false,
      "decisionTrace.isActionable must be false"
    );

    // Candles must still be present for DISPLAY_ONLY_MARKET_DATA
    assert(yfResult.candles.length > 0, "Candles must be present for display purposes");
    assert(yfResult.currentPrice !== null && yfResult.currentPrice > 0, "Current price must be present for display purposes");

    console.log("   [OK] fetchDirectYahooFinanceChart strictly adheres to DISPLAY_ONLY_MARKET_DATA (no synthetic trade plan)");
  } finally {
    global.fetch = originalFetch;
  }
}

// ---------------------------------------------------------------------------
// 3. Verify Phase 1 Canonical Decision Context Contract & Factory
// ---------------------------------------------------------------------------
console.log("3. Testing createDegradedDecisionContext and Phase 1 contract...");
const context = createDegradedDecisionContext("MSFT", "SWING", "LONG_TERM", {
  candlesCount: 120,
  lastClose: 410.5,
  provider: "yahoo_finance_direct",
  asOf: "2026-09-22T06:00:00Z",
});

assert.strictEqual(context.symbol, "MSFT");
assert.strictEqual(context.isDegraded, true);
assert.strictEqual(context.evidenceCompleteness, "DEGRADED");
assert.strictEqual(context.marketEvidence.quality, "FALLBACK");
assert.strictEqual(context.marketEvidence.payload.lastClose, 410.5);
assert.strictEqual(context.fundamentalEvidence.quality, "UNAVAILABLE");
assert.strictEqual(context.fundamentalEvidence.isStale, true);
assert.strictEqual(context.macroEvidence.quality, "UNAVAILABLE");
assert.strictEqual(context.liquidityEvidence.quality, "UNAVAILABLE");
assert.strictEqual(context.liquidityEvidence.payload.liquidityGatePassed, false);

// Actionability check
assert.strictEqual(
  isDecisionActionable(DecisionState.UNVERIFIED, "UNVERIFIED_ASSET"),
  false,
  "UNVERIFIED context must NOT be actionable"
);
assert.strictEqual(
  isDecisionActionable(DecisionState.ACTIONABLE_SETUP, "UNVERIFIED_ASSET"),
  false,
  "UNVERIFIED_ASSET status must NOT be actionable"
);

console.log("   [OK] Canonical decision context and evidence items correctly formatted and fail-closed");

// ---------------------------------------------------------------------------
// 4. Verify Gate 11: Insight Generator Fail-Closed Under Degraded Mode
// ---------------------------------------------------------------------------
console.log("4. Testing generateQuantitativeInsight under degraded mode...");
const degradedInsight = generateQuantitativeInsight(
  "TSLA",
  "Tesla, Inc.",
  220.5,
  1.2,
  undefined,
  undefined,
  "SWING",
  "NOT_OWNED",
  "USER_DECLARED",
  Array.from({ length: 60 }, (_, i) => ({
    time: `2026-08-${String((i % 28) + 1).padStart(2, "0")}`,
    open: 219,
    high: 222,
    low: 218,
    close: 220.5,
    volume: 1000000,
  })),
  "fallback",
  undefined,
  {
    symbol: "TSLA",
    decisionState: "UNVERIFIED",
    stateLabel: "Degraded Market Tape — Decision Engine Unreachable",
    isActionable: false,
    canSizeTrade: false,
    allowedActions: ["RESEARCH_PROFILE"],
    disqualificationReason: "Direct client Yahoo fallback. Analytical backend decision authority unavailable.",
  },
  {
    current_price: 220.5,
    optimal_entry_min: null as any,
    optimal_entry_max: null as any,
    stop_loss: null as any,
    stop_loss_pct: 0,
    take_profit_1: null as any,
    take_profit_1_pct: 0,
    take_profit_2: null as any,
    take_profit_2_pct: 0,
    risk_reward_ratio: null as any,
    execution_status: "UNVERIFIED_ASSET",
    setup_pattern: "Degraded Market Tape (Display Only)",
    entry_thesis: "Direct client market data tape. Decision engine unreachable.",
    invalidation_condition: "Awaiting authoritative backend decision engine.",
    stage_phase: "Unverified Asset",
    vcp_contraction_status: "Unverified",
    atr_14: 4.2,
  },
  "UNAVAILABLE"
);

// Gate 11 invariants:
// 1. BULLISH_POSTURE = ABSENT
assert.strictEqual(
  degradedInsight.posture,
  "RESEARCH",
  "Degraded mode must yield RESEARCH posture, never ACQUIRE, BUY, or ENTER"
);
assert.notStrictEqual(degradedInsight.posture, "ACQUIRE");
assert.notStrictEqual(degradedInsight.human.actionCallout.action, "ENTER");
assert.strictEqual(degradedInsight.human.actionCallout.action, "RESEARCH");

// 2. ACTIONABLE_PLAN = ABSENT
assert.strictEqual(
  degradedInsight.standard.keyLevels.stopLoss,
  0,
  "Key level stop loss must be 0 (suppressed) in degraded mode"
);
assert.strictEqual(
  degradedInsight.standard.keyLevels.target1,
  undefined,
  "Key level target1 must be undefined in degraded mode"
);
assert.strictEqual(
  degradedInsight.standard.keyLevels.profitRiskRatio,
  undefined,
  "Key level profitRiskRatio must be undefined in degraded mode"
);
const sizeAction = degradedInsight.terminalState.availableActions.find(a => a.id === "size_trade");
assert.strictEqual(sizeAction?.enabled, false, "Trade sizing must be strictly disabled");

// 3. CANONICAL_CONFIDENCE = ABSENT
assert.strictEqual(
  degradedInsight.setupScore,
  0,
  "Degraded mode must yield 0 setup score, never synthetic confidence"
);
for (const pillar of degradedInsight.standard.confluenceBreakdown) {
  assert.strictEqual(
    pillar.score,
    0,
    `Confluence pillar ${pillar.dimension} must have score 0 in degraded mode`
  );
}

// 4. VERDICT = UNVERIFIED
assert.strictEqual(degradedInsight.verdict, "UNVERIFIED");
console.log("   [OK] Insight generator strictly enforces BULLISH_POSTURE=ABSENT, ACTIONABLE_PLAN=ABSENT, CANONICAL_CONFIDENCE=ABSENT");

testDirectYahooFetcher().then(() => {
  console.log("\nALL CLIENT FAIL-CLOSED TESTS PASSED SUCCESSFULLY!");
});
