import { evaluatePreFlightChecks } from "../lib/preflightEvaluator";

// Helper function implementing the freshness validation
function resolveFreshCachedVix(mockStorage: Record<string, string>): number | null {
  try {
    const cached = mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"];
    if (!cached) return null;

    const parsed = JSON.parse(cached);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;

    if (!parsed.vix && !parsed.regime) return null;

    const rawVix = parsed?.vix?.value ?? parsed?.vix?.level ?? (typeof parsed?.vix === "number" ? parsed.vix : undefined);
    if (typeof rawVix !== "number" || isNaN(rawVix) || rawVix <= 0 || rawVix > 100) {
      return null;
    }

    const timestampStr = parsed.updatedAt || parsed.timestamp;
    if (!timestampStr || typeof timestampStr !== "string") {
      return null;
    }

    const timestamp = new Date(timestampStr).getTime();
    if (isNaN(timestamp)) {
      return null;
    }

    const now = Date.now();
    const ageMinutes = (now - timestamp) / 60000;

    if (ageMinutes < -5) {
      return null;
    }

    const isClosedSession = parsed.marketSession === "CLOSED" || parsed.settlementPinned === true || parsed.isSettlementPinned === true;
    const maxAgeMinutes = isClosedSession ? 4320 : 240;

    if (ageMinutes > maxAgeMinutes) {
      return null;
    }

    return rawVix;
  } catch {
    return null;
  }
}

console.log("=== TESTING MACRO FRESHNESS SCENARIOS ===");

const scenarios = [
  {
    name: "Scenario 1: Fresh live feed (VIX 18.2)",
    inputVix: 18.2,
    storage: {},
    expectedStatus: "PASS"
  },
  {
    name: "Scenario 2: Fresh cache (10 mins old, VIX 17.5)",
    inputVix: undefined,
    storage: {
      "FINANCE_MARKET_SNAPSHOTS_V1": JSON.stringify({
        vix: { value: 17.5 },
        marketSession: "OPEN",
        updatedAt: new Date(Date.now() - 10 * 60 * 1000).toISOString()
      })
    },
    expectedStatus: "PASS"
  },
  {
    name: "Scenario 3: Stale cache (30 days old, VIX 15.2)",
    inputVix: undefined,
    storage: {
      "FINANCE_MARKET_SNAPSHOTS_V1": JSON.stringify({
        vix: { value: 15.2 },
        marketSession: "OPEN",
        updatedAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString()
      })
    },
    expectedStatus: "UNAVAILABLE"
  },
  {
    name: "Scenario 4: Stale open-session cache (5 hours old, VIX 16.0)",
    inputVix: undefined,
    storage: {
      "FINANCE_MARKET_SNAPSHOTS_V1": JSON.stringify({
        vix: { value: 16.0 },
        marketSession: "OPEN",
        updatedAt: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString()
      })
    },
    expectedStatus: "UNAVAILABLE"
  },
  {
    name: "Scenario 5: Missing cache entirely",
    inputVix: undefined,
    storage: {},
    expectedStatus: "UNAVAILABLE"
  },
  {
    name: "Scenario 6: Corrupted cache JSON",
    inputVix: undefined,
    storage: {
      "FINANCE_MARKET_SNAPSHOTS_V1": "{ corrupted json syntax"
    },
    expectedStatus: "UNAVAILABLE"
  },
  {
    name: "Scenario 7: Cross-ticker cache leak (AAPL ticker snapshot instead of macro)",
    inputVix: undefined,
    storage: {
      "FINANCE_MARKET_SNAPSHOTS_V1": JSON.stringify({
        ticker: "AAPL",
        spotPrice: 228.5,
        setupScore: 82,
        updatedAt: new Date().toISOString()
      })
    },
    expectedStatus: "UNAVAILABLE"
  },
  {
    name: "Scenario 8: Cache missing updatedAt timestamp",
    inputVix: undefined,
    storage: {
      "FINANCE_MARKET_SNAPSHOTS_V1": JSON.stringify({
        vix: { value: 15.5 },
        marketSession: "OPEN"
      })
    },
    expectedStatus: "UNAVAILABLE"
  },
  {
    name: "Scenario 9: Future timestamp (+30 mins clock skew)",
    inputVix: undefined,
    storage: {
      "FINANCE_MARKET_SNAPSHOTS_V1": JSON.stringify({
        vix: { value: 15.5 },
        marketSession: "OPEN",
        updatedAt: new Date(Date.now() + 30 * 60 * 1000).toISOString()
      })
    },
    expectedStatus: "UNAVAILABLE"
  },
  {
    name: "Scenario 10: Valid weekend pinned closed session (36 hours old, VIX 16.8)",
    inputVix: undefined,
    storage: {
      "FINANCE_MARKET_SNAPSHOTS_V1": JSON.stringify({
        vix: { value: 16.8 },
        marketSession: "CLOSED",
        settlementPinned: true,
        updatedAt: new Date(Date.now() - 36 * 60 * 60 * 1000).toISOString()
      })
    },
    expectedStatus: "PASS"
  },
  {
    name: "Scenario 11: Stale weekend session exceeding 72h (80 hours old, VIX 16.8)",
    inputVix: undefined,
    storage: {
      "FINANCE_MARKET_SNAPSHOTS_V1": JSON.stringify({
        vix: { value: 16.8 },
        marketSession: "CLOSED",
        settlementPinned: true,
        updatedAt: new Date(Date.now() - 80 * 60 * 60 * 1000).toISOString()
      })
    },
    expectedStatus: "UNAVAILABLE"
  }
];

let allPassed = true;

for (const sc of scenarios) {
  const resolved = sc.inputVix !== undefined ? sc.inputVix : resolveFreshCachedVix(sc.storage);
  const result = evaluatePreFlightChecks({
    symbol: "AAPL",
    currentPrice: 220.0,
    stopLoss: 210.0,
    takeProfit1: 245.0,
    riskRewardRatio: 2.5,
    optimalEntryMin: 215.0,
    optimalEntryMax: 225.0,
    breakoutPivot: 228.0,
    isActionable: true,
    vix: resolved,
    catalystState: "PASS",
    catalystEvidence: {
      state: "PASS",
      daysToEarnings: 14.0,
      freshnessStatus: "LIVE",
      confirmationStatus: "CONFIRMED"
    }
  });

  const actualStatus = result.checks[4].status;
  const passed = actualStatus === sc.expectedStatus;
  if (!passed) allPassed = false;
  console.log(`[${passed ? "PASS" : "FAIL"}] ${sc.name} -> Actual: ${actualStatus}, Expected: ${sc.expectedStatus}`);
}

console.log(`\nOverall Test Result: ${allPassed ? "ALL 11 SCENARIOS PASSED" : "FAILURES DETECTED"}`);
