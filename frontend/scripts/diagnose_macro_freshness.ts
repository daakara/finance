import { evaluatePreFlightChecks } from "../lib/preflightEvaluator";

// Simulate browser environment with localStorage
const mockStorage: Record<string, string> = {};
(global as any).window = {};
(global as any).localStorage = {
  getItem: (key: string) => mockStorage[key] || null,
  setItem: (key: string, val: string) => { mockStorage[key] = val; },
  removeItem: (key: string) => { delete mockStorage[key]; }
};

console.log("=== DIAGNOSTIC: MACRO FRESHNESS SAFETY CHECK ===");

// Scenario 1: Authoritative live macro feed is UNAVAILABLE (vix is undefined in inputs)
// AND localStorage contains an old cached snapshot with VIX = 15.2 (from 30 days ago)
const staleTimestamp = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
  spx: { symbol: "SPY", price: 540.0, changePct: 0.5 },
  vix: { level: 15.2, value: 15.2, change: -0.5, changePct: -3.0, tier: "NORMAL" },
  regime: "RISK_ON",
  marketSession: "CLOSED",
  settlementPinned: true,
  dataSource: "CACHED_STORE",
  updatedAt: staleTimestamp
});

console.log(`\n1. Injected stale cache into localStorage:`);
console.log(`   Timestamp: ${staleTimestamp} (30 days old)`);
console.log(`   Cached VIX: 15.2`);

// Run Pre-Flight evaluation where input vix is undefined (live feed down / timeout)
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
  vix: undefined, // LIVE MACRO FEED UNAVAILABLE
  catalystState: "PASS",
  catalystEvidence: {
    state: "PASS",
    daysToEarnings: 14.0,
    freshnessStatus: "LIVE",
    confirmationStatus: "CONFIRMED"
  }
});

console.log("\n2. Pre-Flight Check 5 (Macro Regime Guard) Result:");
console.log(`   Status: ${result.checks[4].status}`);
console.log(`   Passed: ${result.checks[4].passed}`);
console.log(`   Summary: ${result.checks[4].summary}`);
console.log(`   Detail: ${result.checks[4].detail}`);
console.log(`\nOverall Pre-Flight Clearance:`);
console.log(`   isCleared: ${result.isCleared}`);
console.log(`   passedCount: ${result.checks.filter(c => c.passed).length}/5`);
console.log(`   Clearance Status: ${result.clearanceStatusLabel}`);
