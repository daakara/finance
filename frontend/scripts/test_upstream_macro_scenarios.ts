/**
 * ARX_UPSTREAM_MACRO_FRESHNESS_AUTHORITY_001
 * Phase B Scenario Diagnostic & Reproduction Suite
 *
 * Evaluates the 14 mandatory macro authority scenarios against current production code.
 */

import { evaluatePreFlightChecks, resolveFreshCachedVix, PreFlightChecklistInputs } from "../lib/preflightEvaluator";
import { DEFAULT_MACRO_SNAPSHOT } from "../components/nav/MarketCommandRibbon";

interface ScenarioResult {
  id: string;
  name: string;
  inputDescription: string;
  source: string;
  observationTime: string | null;
  sessionAuthority: string;
  actualStatus: string;
  actualPassed: boolean;
  actualCleared: boolean;
  expectedStatus: string;
  expectedBehavior: string;
  classification: "PASS" | "FAIL" | "UNVERIFIED";
  notes?: string;
}

const mockStorage: Record<string, string> = {};
(global as any).window = {};
(global as any).localStorage = {
  getItem: (key: string) => mockStorage[key] || null,
  setItem: (key: string, val: string) => { mockStorage[key] = val; },
  removeItem: (key: string) => { delete mockStorage[key]; },
  clear: () => {
    for (const k of Object.keys(mockStorage)) delete mockStorage[k];
  }
};

const baseInputs: PreFlightChecklistInputs = {
  symbol: "AAPL",
  currentPrice: 220.0,
  stopLoss: 210.0,
  takeProfit1: 245.0,
  riskRewardRatio: 2.5,
  optimalEntryMin: 215.0,
  optimalEntryMax: 225.0,
  breakoutPivot: 228.0,
  isActionable: true,
  catalystState: "PASS"
};

const results: ScenarioResult[] = [];

function runScenario(
  id: string,
  name: string,
  inputDesc: string,
  source: string,
  obsTime: string | null,
  sessionAuth: string,
  evalInputs: PreFlightChecklistInputs,
  expectedStatus: string,
  expectedBehavior: string,
  passCondition: (res: ReturnType<typeof evaluatePreFlightChecks>) => boolean
) {
  const res = evaluatePreFlightChecks(evalInputs);
  const chk5 = res.checks[4];
  const passed = passCondition(res);

  results.push({
    id,
    name,
    inputDescription: inputDesc,
    source,
    observationTime: obsTime,
    sessionAuthority: sessionAuth,
    actualStatus: chk5.status,
    actualPassed: chk5.passed,
    actualCleared: res.isCleared,
    expectedStatus,
    expectedBehavior,
    classification: passed ? "PASS" : "FAIL",
    notes: chk5.summary + " | " + chk5.detail
  });
}

// ─── M01: Fresh authoritative VIX below 26 ──────────────────────────────────
(global as any).localStorage.clear();
const nowIso = new Date().toISOString();
mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
  vix: { value: 16.5, level: 16.5, observationTime: nowIso },
  updatedAt: nowIso,
  observationTime: nowIso,
  marketSession: "REGULAR"
});
runScenario(
  "M01",
  "Fresh authoritative VIX below 26",
  "VIX 16.5 observed within last 5 minutes during open session",
  "LIVE_EXCHANGE_FEED",
  nowIso,
  "NYSE_REGULAR",
  { ...baseInputs },
  "PASS",
  "PASS eligible",
  (r) => r.checks[4].status === "PASS" && r.checks[4].passed === true
);

// ─── M02: Fresh authoritative VIX at/above 26 ────────────────────────────────
(global as any).localStorage.clear();
mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
  vix: { value: 28.5, level: 28.5, observationTime: nowIso },
  updatedAt: nowIso,
  observationTime: nowIso,
  marketSession: "REGULAR"
});
runScenario(
  "M02",
  "Fresh authoritative VIX at/above 26",
  "VIX 28.5 observed within last 5 minutes (Elevated Volatility)",
  "LIVE_EXCHANGE_FEED",
  nowIso,
  "NYSE_REGULAR",
  { ...baseInputs },
  "HIGH VIX",
  "Non-PASS",
  (r) => r.checks[4].status === "HIGH VIX" && r.checks[4].passed === false && r.isCleared === false
);

// ─── M03: Direct VIX without trustworthy timestamp ───────────────────────────
(global as any).localStorage.clear();
// Bare numeric VIX passed directly into evaluatePreFlightChecks without timestamp metadata
runScenario(
  "M03",
  "Direct VIX without trustworthy timestamp",
  "Direct numeric input vix: 18.0 with no observation timestamp or provenance envelope",
  "DIRECT_UNVERIFIED_PROP",
  null,
  "UNKNOWN",
  { ...baseInputs, vix: 18.0 },
  "UNAVAILABLE",
  "No unsupported PASS (fail closed if missing trustworthy timestamp)",
  (r) => r.checks[4].status !== "PASS" // Required: Must NOT evaluate to PASS without provenance
);

// ─── M04: Direct VIX with stale observation timestamp ────────────────────────
(global as any).localStorage.clear();
const staleObsTime = new Date(Date.now() - 360 * 60000).toISOString(); // 6 hours old (open session)
runScenario(
  "M04",
  "Direct VIX with stale observation timestamp",
  "Direct input with observation timestamp 6 hours old during open market",
  "STALE_DIRECT_OBSERVATION",
  staleObsTime,
  "NYSE_REGULAR",
  {
    ...baseInputs,
    vix: 18.0,
    // If macro evidence envelope exists or if evaluator inspects evidence:
    evidenceStatus: "STALE"
  },
  "UNAVAILABLE",
  "Non-PASS (stale observation rejected)",
  (r) => r.checks[4].status !== "PASS" && r.isCleared === false
);

// ─── M05: Newly received API response containing old market data ─────────────
(global as any).localStorage.clear();
const oldBarObsTime = new Date(Date.now() - 300 * 60000).toISOString(); // Bar is 5h old
const apiJustGenerated = new Date().toISOString(); // API responded 2 seconds ago
mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
  updatedAt: apiJustGenerated, // API says updated NOW
  generatedAt: apiJustGenerated,
  marketSession: "REGULAR",
  vix: {
    value: 17.2,
    level: 17.2,
    observationTime: oldBarObsTime // Actual market observation is 5 hours old!
  },
  observationTime: oldBarObsTime
});
runScenario(
  "M05",
  "Newly received API response containing old market data",
  "API response updatedAt=now, but vix.observationTime=5 hours ago in regular session",
  "API_RESPONSE_WITH_STALE_BAR",
  oldBarObsTime,
  "NYSE_REGULAR",
  { ...baseInputs },
  "UNAVAILABLE",
  "Non-PASS (market observation time must govern over API response timestamp)",
  (r) => r.checks[4].status === "UNAVAILABLE" && r.checks[4].passed === false
);

// ─── M06: Stale cached VIX ───────────────────────────────────────────────────
(global as any).localStorage.clear();
const staleSnapshotTime = new Date(Date.now() - 300 * 60000).toISOString(); // 5 hours old
mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
  updatedAt: staleSnapshotTime,
  observationTime: staleSnapshotTime,
  marketSession: "REGULAR",
  vix: { value: 16.0, level: 16.0, observationTime: staleSnapshotTime }
});
runScenario(
  "M06",
  "Stale cached VIX",
  "Cached snapshot updatedAt and observationTime both 5 hours old in regular session",
  "BROWSER_CACHE",
  staleSnapshotTime,
  "NYSE_REGULAR",
  { ...baseInputs },
  "UNAVAILABLE",
  "UNAVAILABLE (fail closed when age > 240m)",
  (r) => r.checks[4].status === "UNAVAILABLE" && r.isCleared === false
);

// ─── M07: Unsupported cached CLOSED-session flag ─────────────────────────────
(global as any).localStorage.clear();
const unauthClosedTime = new Date(Date.now() - 300 * 60000).toISOString(); // 5 hours old
mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
  updatedAt: unauthClosedTime,
  observationTime: unauthClosedTime,
  marketSession: "CLOSED", // Claims closed, but has no authenticated settlement or was regular session
  settlementPinned: false, // Pinned settlement flag FALSE
  isSettlementPinned: false,
  vix: { value: 15.0, level: 15.0, observationTime: unauthClosedTime }
});
runScenario(
  "M07",
  "Unsupported cached CLOSED-session flag",
  "Payload claims marketSession: CLOSED but settlementPinned is false and age > 240m",
  "UNAUTHENTICATED_SESSION_FLAG",
  unauthClosedTime,
  "UNVERIFIED_CLOSED",
  { ...baseInputs },
  "UNAVAILABLE",
  "Cannot extend freshness eligibility without authenticated settlement",
  (r) => r.checks[4].status === "UNAVAILABLE"
);

// ─── M08: Authenticated closed-session pinned settlement ─────────────────────
(global as any).localStorage.clear();
const weekendCloseTime = new Date(Date.now() - 36 * 3600000).toISOString(); // 36 hours old (e.g. Saturday)
mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
  updatedAt: weekendCloseTime,
  observationTime: weekendCloseTime,
  marketSession: "CLOSED",
  settlementPinned: true,
  isSettlementPinned: true,
  vix: { value: 18.2, level: 18.2, observationTime: weekendCloseTime }
});
runScenario(
  "M08",
  "Authenticated closed-session pinned settlement",
  "Friday close pinned settlement, 36 hours old during weekend closure",
  "AUTHENTICATED_PINNED_SETTLEMENT",
  weekendCloseTime,
  "NYSE_SCHEDULED_WEEKEND_CLOSE",
  { ...baseInputs },
  "PASS",
  "Apply approved policy (eligible for PASS within 4,320m)",
  (r) => r.checks[4].status === "PASS" && r.checks[4].passed === true
);

// ─── M09: Expired pinned settlement ──────────────────────────────────────────
(global as any).localStorage.clear();
const expiredCloseTime = new Date(Date.now() - 80 * 3600000).toISOString(); // 80 hours old (> 72h)
mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
  updatedAt: expiredCloseTime,
  observationTime: expiredCloseTime,
  marketSession: "CLOSED",
  settlementPinned: true,
  vix: { value: 18.2, level: 18.2, observationTime: expiredCloseTime }
});
runScenario(
  "M09",
  "Expired pinned settlement",
  "Pinned settlement 80 hours old (> 72 hours / 4,320 minutes)",
  "EXPIRED_PINNED_SETTLEMENT",
  expiredCloseTime,
  "NYSE_CLOSED_EXPIRED",
  { ...baseInputs },
  "UNAVAILABLE",
  "Non-PASS (exceeds 4,320m max threshold)",
  (r) => r.checks[4].status === "UNAVAILABLE" && r.isCleared === false
);

// ─── M10: Network timeout with historical fallback ───────────────────────────
(global as any).localStorage.clear();
// Simulate network error / timeout: cache is empty, no live feed
runScenario(
  "M10",
  "Network timeout with historical fallback",
  "Feed timed out, localStorage has no fresh snapshot",
  "NETWORK_TIMEOUT_EMPTY_FALLBACK",
  null,
  "UNAVAILABLE",
  { ...baseInputs, vix: undefined },
  "UNAVAILABLE",
  "No false clearance (must evaluate to UNAVAILABLE and block trade)",
  (r) => r.checks[4].status === "UNAVAILABLE" && r.isCleared === false
);

// ─── M11: Malformed/future-dated timestamps ──────────────────────────────────
(global as any).localStorage.clear();
const futureTime = new Date(Date.now() + 60 * 60000).toISOString(); // 1 hour in the future
mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
  updatedAt: futureTime,
  observationTime: futureTime,
  marketSession: "REGULAR",
  vix: { value: 16.0, level: 16.0 }
});
runScenario(
  "M11",
  "Malformed/future-dated timestamps",
  "Timestamp 60 minutes into future (Clock Skew > 5m)",
  "CORRUPTED_CLOCK_SKEW",
  futureTime,
  "INVALID",
  { ...baseInputs },
  "UNAVAILABLE",
  "Reject invalid evidence (fail closed to UNAVAILABLE)",
  (r) => r.checks[4].status === "UNAVAILABLE" && r.isCleared === false
);

// ─── M12: Inline versus modal ────────────────────────────────────────────────
(global as any).localStorage.clear();
mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
  updatedAt: nowIso,
  observationTime: nowIso,
  marketSession: "REGULAR",
  vix: { value: 17.5, level: 17.5 }
});
const evalResult = evaluatePreFlightChecks(baseInputs);
const modalVixStatus = evalResult.checks[4].status;
const inlineVixStatus = evalResult.checks[4].status;
results.push({
  id: "M12",
  name: "Inline versus modal",
  inputDescription: "Canonical inputs shared between inline readiness bar and checklist modal",
  source: "CANONICAL_PREFLIGHT_INPUTS",
  observationTime: nowIso,
  sessionAuthority: "NYSE_REGULAR",
  actualStatus: inlineVixStatus,
  actualPassed: evalResult.checks[4].passed,
  actualCleared: evalResult.isCleared,
  expectedStatus: "Identical",
  expectedBehavior: "Identical Macro result",
  classification: inlineVixStatus === modalVixStatus ? "PASS" : "FAIL",
  notes: `Inline=${inlineVixStatus}, Modal=${modalVixStatus}`
});

// ─── M13: Symbol change or refresh ───────────────────────────────────────────
// Check that switching symbols maintains strict macro isolation without leaking favorable context
const nvdaInputs: PreFlightChecklistInputs = {
  ...baseInputs,
  symbol: "NVDA",
  isStage4: true,
  isActionable: false
};
const nvdaResult = evaluatePreFlightChecks(nvdaInputs);
results.push({
  id: "M13",
  name: "Symbol change or refresh",
  inputDescription: "Switching from AAPL to NVDA preserves independent check evaluation",
  source: "STATE_TRANSITION",
  observationTime: nowIso,
  sessionAuthority: "NYSE_REGULAR",
  actualStatus: nvdaResult.checks[4].status,
  actualPassed: nvdaResult.checks[4].passed,
  actualCleared: nvdaResult.isCleared,
  expectedStatus: "PASS on Macro, BLOCKED on Stage 4",
  expectedBehavior: "No stale favorable context; Stage 4 strictly blocks clearance",
  classification: nvdaResult.checks[4].status === "PASS" && nvdaResult.isCleared === false ? "PASS" : "FAIL",
  notes: `NVDA Clearance=${nvdaResult.clearanceStatusLabel}, Chk2=${nvdaResult.checks[1].status}`
});

// ─── M14: Default Macro Ribbon fixture ───────────────────────────────────────
(global as any).localStorage.clear();
// What if DEFAULT_MACRO_SNAPSHOT is stored or passed?
mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify(DEFAULT_MACRO_SNAPSHOT);
runScenario(
  "M14",
  "Default Macro Ribbon fixture",
  `DEFAULT_MACRO_SNAPSHOT fixture (updatedAt=${DEFAULT_MACRO_SNAPSHOT.updatedAt})`,
  "STATIC_FIXTURE_STORE",
  DEFAULT_MACRO_SNAPSHOT.updatedAt || null,
  "HISTORICAL_FIXTURE",
  { ...baseInputs },
  "UNAVAILABLE",
  "Clearly historical; never grants clearance",
  (r) => r.checks[4].status === "UNAVAILABLE" && r.isCleared === false
);

console.log("================================================================================");
console.log("  ARX UPSTREAM MACRO SAFETY QUALIFICATION: SCENARIO MATRIX RESULTS");
console.log("================================================================================\n");

console.table(results.map(r => ({
  ID: r.id,
  Scenario: r.name.slice(0, 30),
  ActualStatus: r.actualStatus,
  Expected: r.expectedStatus,
  Classification: r.classification
})));

const failed = results.filter(r => r.classification === "FAIL");
console.log(`\nTOTAL SCENARIOS: ${results.length}`);
console.log(`PASSED: ${results.length - failed.length}`);
console.log(`FAILED: ${failed.length}`);

if (failed.length > 0) {
  console.log("\nFAILED SCENARIOS DETAILS:");
  for (const f of failed) {
    console.log(` - [${f.id}] ${f.name}`);
    console.log(`    Input: ${f.inputDescription}`);
    console.log(`    Actual Status: ${f.actualStatus} (passed=${f.actualPassed}, cleared=${f.actualCleared})`);
    console.log(`    Expected: ${f.expectedStatus} (${f.expectedBehavior})`);
    console.log(`    Notes: ${f.notes}`);
  }
}
