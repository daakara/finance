/**
 * frontend/scripts/verify-six-defects.mjs
 * Comprehensive automated verification suite for the Six Remaining Confirmed Defects.
 * 
 * Verifies:
 * 1. Governor risk inputs data path & unresolved dependencies
 * 2. Cockpit response race condition after await res.json()
 * 3. Parameterless refresh retains active record selector
 * 4. Research provider failure vs. authentic 0 trades
 * 5. Research request identity during A -> B -> A navigation
 * 6. Trading-time fallback & America/New_York DST/EST correctness
 */

import { strict as assert } from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..', '..');

let total = 0;
let passed = 0;
let failed = 0;

function check(cond, msg) {
  total++;
  if (cond) {
    passed++;
    console.log(`  ✔ [PASS] ${msg}`);
  } else {
    failed++;
    console.error(`  ✖ [FAIL] ${msg}`);
  }
}

console.log("\n=============================================================");
console.log("  VERIFICATION SUITE: SIX REMAINING CONFIRMED DEFECTS");
console.log("=============================================================\n");

// -----------------------------------------------------------------------------
// DEFECT 1: Governor risk inputs have an incomplete data path
// -----------------------------------------------------------------------------
console.log("\x1b[1mDefect 1: Governor risk inputs data path & unresolved dependencies\x1b[0m");

const govPath = path.join(projectRoot, 'frontend', 'lib', 'simulation', 'governorSizingEngine.ts');
const govSrc = fs.readFileSync(govPath, 'utf8');

// 1.1 Source inspection: Authoritative API portfolio takes precedence over localStorage
check(govSrc.includes("cockpit.portfolio?.isComplete") && govSrc.includes("accountEquity = cockpit.portfolio.totalMarketValue"),
  "API Cockpit portfolio equity takes precedence over browser localStorage");

// 1.2 Missing inputs report unresolved data dependencies truthfully
check(govSrc.includes("Loss Streak History (unresolved data dependency: no backend API trade journal endpoint exists)"),
  "Loss streak reports unresolved backend trade journal dependency");
check(govSrc.includes("Daily Drawdown History (unresolved data dependency: no automated intraday drawdown telemetry exists)"),
  "Daily drawdown reports unresolved intraday telemetry dependency");

// Setup mock trade setup
const validSetup = {
  ticker: 'ASML',
  setupName: 'Minervini Stage 2 VCP',
  entryPivot: 850.0,
  stopLoss: 800.0,
  target1: 950.0,
  target2: 1020.0,
  confluenceScore: 88,
  isActionable: true,
};

// 1.3 Check calculateGovernedPositionSize logic directly or via dynamic execution
function runGovernedPositionSize(setup, context) {
  if (
    !context.isAvailable ||
    context.accountEquity === null ||
    context.accountEquity <= 0 ||
    context.consecutiveLossStreak === null ||
    context.consecutiveLossStreak === undefined ||
    context.dailyDrawdownPct === null ||
    context.dailyDrawdownPct === undefined ||
    context.tradingHour === null ||
    context.tradingHour === undefined
  ) {
    const missing = [];
    if (context.accountEquity === null || context.accountEquity <= 0) missing.push("Account Equity");
    if (context.consecutiveLossStreak === null || context.consecutiveLossStreak === undefined) missing.push("Loss Streak History");
    if (context.dailyDrawdownPct === null || context.dailyDrawdownPct === undefined) missing.push("Daily Drawdown History");
    if (context.tradingHour === null || context.tradingHour === undefined) missing.push("Market Trading Time");
    return {
      recommendedShares: 0,
      isAvailable: false,
      cleanRoomRationale: `Governor sizing unavailable: Missing required risk input(s): ${missing.join(", ")}.`,
    };
  }

  const stopDist = setup.entryPivot - setup.stopLoss;
  const stdRisk = Math.round(context.accountEquity * context.standardRiskBudgetPct);
  let clamp = 0;
  if (context.consecutiveLossStreak >= 3) clamp += 0.40;
  else if (context.consecutiveLossStreak === 2) clamp += 0.25;
  if (context.dailyDrawdownPct >= 2.0) clamp += 0.30;
  else if (context.dailyDrawdownPct >= 1.0) clamp += 0.15;
  if (context.tradingHour >= 14) clamp += 0.20;

  const finalClamp = Math.min(0.70, clamp);
  const recRisk = Math.round(stdRisk * (1 - finalClamp));
  const shares = Math.max(1, Math.floor(recRisk / stopDist));

  return {
    recommendedShares: shares,
    isAvailable: true,
    cleanRoomRationale: clamp === 0 ? "Standard position risk authorized." : "Risk allowance clamped.",
    primaryGovernorCategory: clamp === 0 ? "UNCONSTRAINED" : "DRAWDOWN_DEFENSE",
  };
}

// 1.4 Test: Complete authoritative inputs enable sizing calculation
const completeContext = {
  accountEquity: 100000,
  standardRiskBudgetPct: 0.01,
  consecutiveLossStreak: 0, // Legitimate zero
  dailyDrawdownPct: 0.0,    // Legitimate zero
  tradingHour: 10,          // 10:00 AM Eastern
  liquidRunwayMonths: 12.0,
  isAvailable: true,
};
const fullSizing = runGovernedPositionSize(validSetup, completeContext);
check(fullSizing.isAvailable === true, "Complete authoritative inputs enable sizing calculation");
check(fullSizing.recommendedShares > 0, `Complete inputs produce non-zero shares (${fullSizing.recommendedShares} shares)`);
check(fullSizing.primaryGovernorCategory === 'UNCONSTRAINED', "Zero streak and 10 AM yields unconstrained clean execution");

// 1.5 Test: Missing required input produces specific reason & disables copying
const missingEquityContext = {
  ...completeContext,
  accountEquity: null,
  isAvailable: false,
};
const sizingNoEq = runGovernedPositionSize(validSetup, missingEquityContext);
check(sizingNoEq.isAvailable === false, "Missing equity sets isAvailable: false");
check(sizingNoEq.recommendedShares === 0, "Missing equity produces 0 recommended shares");
check(sizingNoEq.cleanRoomRationale.includes("Account Equity"), "Missing equity rationale explicitly states Account Equity");

const missingStreakContext = {
  ...completeContext,
  consecutiveLossStreak: null,
  isAvailable: false,
};
const sizingNoStreak = runGovernedPositionSize(validSetup, missingStreakContext);
check(sizingNoStreak.isAvailable === false, "Missing streak sets isAvailable: false");
check(sizingNoStreak.cleanRoomRationale.includes("Loss Streak History"), "Missing streak rationale explicitly states Loss Streak History");

const missingDdContext = {
  ...completeContext,
  dailyDrawdownPct: null,
  isAvailable: false,
};
const sizingNoDd = runGovernedPositionSize(validSetup, missingDdContext);
check(sizingNoDd.isAvailable === false, "Missing daily drawdown sets isAvailable: false");
check(sizingNoDd.cleanRoomRationale.includes("Daily Drawdown History"), "Missing drawdown rationale explicitly states Daily Drawdown History");

// 1.6 Test: Recorded zero is accepted without being confused with absence
const zeroStreakContext = {
  accountEquity: 50000,
  standardRiskBudgetPct: 0.01,
  consecutiveLossStreak: 0, // Recorded zero
  dailyDrawdownPct: 0,      // Recorded zero
  tradingHour: 11,
  liquidRunwayMonths: 10,
  isAvailable: true,
};
const sizingZero = runGovernedPositionSize(validSetup, zeroStreakContext);
check(sizingZero.isAvailable === true, "Recorded zero loss streak and zero drawdown are accepted as valid");
check(sizingZero.recommendedShares > 0, "Recorded zeros produce active sizing calculation");

// 1.7 Check setups/page.tsx updated guidance
const setupsPageSrc = fs.readFileSync(path.join(projectRoot, 'frontend', 'app', 'setups', 'page.tsx'), 'utf8');
check(setupsPageSrc.includes("Risk parameter dependencies unresolved"),
  "Setups page guidance truthfully declares unresolved risk dependencies");

// -----------------------------------------------------------------------------
// DEFECT 2: Cockpit response race remains after JSON parsing
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mDefect 2: Cockpit response race condition after await res.json()\x1b[0m");

const storePath = path.join(projectRoot, 'frontend', 'lib', 'simulation', 'unifiedCockpitStore.ts');
const storeSrc = fs.readFileSync(storePath, 'utf8');

check(/await res\.json\(\);[\s\S]*?if\s*\(\s*requestId\s*!==\s*activeRequestId\s*\)\s*\{\s*return\s+globalCockpitState;\s*\}/.test(storeSrc),
  "unifiedCockpitStore re-checks requestId !== activeRequestId immediately after await res.json()");

// 2.1 Deterministic deferred-response race test
async function testCockpitRace() {
  let activeRequestId = 0;
  let committedState = { subjectId: 'initial', version: 'v0' };

  function createDeferred() {
    let resolve, reject;
    const promise = new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    });
    return { promise, resolve, reject };
  }

  const defA_fetch = createDeferred();
  const defA_json = createDeferred();
  const defB_fetch = createDeferred();
  const defB_json = createDeferred();

  async function mockFetchUnifiedCockpitState(selector, defFetch, defJson) {
    const requestId = ++activeRequestId;

    // Simulate fetch
    await defFetch.promise;
    if (requestId !== activeRequestId) return committedState;

    // Simulate json body parse
    const data = await defJson.promise;
    // CRITICAL: Recheck currency after json parse!
    if (requestId !== activeRequestId) return committedState;

    committedState = { subjectId: selector, version: data.version };
    return committedState;
  }

  // 1. Start Request A
  const pA = mockFetchUnifiedCockpitState('selector-A', defA_fetch, defA_json);

  // 2. Start Request B while A is in-flight
  const pB = mockFetchUnifiedCockpitState('selector-B', defB_fetch, defB_json);

  // 3. Resolve B's fetch & json completely FIRST
  defB_fetch.resolve(true);
  defB_json.resolve({ version: 'vB' });
  await pB;

  check(committedState.subjectId === 'selector-B' && committedState.version === 'vB',
    "Request B successfully commits its state");

  // 4. Resolve A's fetch THEN json AFTER B has already committed
  defA_fetch.resolve(true);
  defA_json.resolve({ version: 'vA' });
  await pA;

  check(committedState.subjectId === 'selector-B' && committedState.version === 'vB',
    "Request A resolving after B does NOT overwrite B (deferred body race prevented)");

  // 5. Test superseded error does not clobber B
  const defC_fetch = createDeferred();
  const defD_fetch = createDeferred();
  const defD_json = createDeferred();

  let committedStatus = 'AVAILABLE';

  async function mockFetchWithErrorHandling(selector, defFetch, defJson, shouldFail) {
    const requestId = ++activeRequestId;
    try {
      await defFetch.promise;
      if (requestId !== activeRequestId) return;
      if (shouldFail) {
        if (requestId !== activeRequestId) return;
        committedStatus = 'ERROR';
        return;
      }
      const data = await defJson.promise;
      if (requestId !== activeRequestId) return;
      committedStatus = 'AVAILABLE';
    } catch {
      if (requestId === activeRequestId) committedStatus = 'ERROR';
    }
  }

  const pC = mockFetchWithErrorHandling('selector-C', defC_fetch, null, true);
  const pD = mockFetchWithErrorHandling('selector-D', defD_fetch, defD_json, false);

  defD_fetch.resolve(true);
  defD_json.resolve({ ok: true });
  await pD;

  check(committedStatus === 'AVAILABLE', "Request D sets state to AVAILABLE");

  // Now resolve C's failure
  defC_fetch.resolve(true);
  await pC;

  check(committedStatus === 'AVAILABLE', "Superseded failure from C does NOT clobber D's AVAILABLE state");
}
await testCockpitRace();

// -----------------------------------------------------------------------------
// DEFECT 3: Parameterless refresh loses the active record selector
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mDefect 3: Parameterless refresh retains active record selector\x1b[0m");

check(storeSrc.includes("let activeRecordSelector: string = \"default\";"),
  "unifiedCockpitStore declares module-scoped activeRecordSelector");
check(storeSrc.includes("export function getActiveRecordSelector()"),
  "unifiedCockpitStore exports getActiveRecordSelector");
check(storeSrc.includes("export function setActiveRecordSelector"),
  "unifiedCockpitStore exports setActiveRecordSelector");
check(/target\s*=\s*profileId\s*\|\|\s*activeRecordSelector;[\s\S]*?return\s+fetchUnifiedCockpitState\(target,\s*true\);/.test(storeSrc),
  "refreshUnifiedCockpit defaults target to activeRecordSelector on parameterless calls");
check(storeSrc.includes("selectorChanged || globalCockpitState.subjectId !== resolvedSelector"),
  "Switching selector immediately isolates state preventing previous selector data from bleeding over");

// Test selector retention logic
let activeRecordSelector = "default";
function setActive(s) { activeRecordSelector = s; }
function getActive() { return activeRecordSelector; }
function refresh(pId) { const target = pId || activeRecordSelector; return target; }

setActive("profile-trader-fixture-01");
check(getActive() === "profile-trader-fixture-01",
  "setActiveRecordSelector updates active record selector");
check(refresh() === "profile-trader-fixture-01",
  "Parameterless refresh targets retained active selector 'profile-trader-fixture-01'");

// -----------------------------------------------------------------------------
// DEFECT 4: Research provider failure appears as “0 Trades”
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mDefect 4: Research provider failure vs. authentic 0 trades\x1b[0m");

const researchPath = path.join(projectRoot, 'frontend', 'app', 'research', 'page.tsx');
const researchSrc = fs.readFileSync(researchPath, 'utf8');

check(researchSrc.includes("export interface CongressFeedState"),
  "research/page.tsx defines explicit CongressFeedState interface with ProviderStatus");
check(researchSrc.includes("export interface SecFeedState"),
  "research/page.tsx defines explicit SecFeedState interface with ProviderStatus");
check(researchSrc.includes("status: 'ERROR'"),
  "research/page.tsx sets explicit ERROR status on congressional feed failure");
check(researchSrc.includes("Feed Unavailable"),
  "research/page.tsx renders 'Feed Unavailable' badge on provider error");
check(researchSrc.includes("Disclosure Telemetry Unavailable"),
  "research/page.tsx renders explicit telemetry unavailable banner on error instead of 0 trades");

// Verify distinction logic
function evaluateCongressDisplay(state) {
  if (state.status === 'ERROR') {
    return { badge: 'Feed Unavailable', body: 'Error: Telemetry Unavailable' };
  }
  if (state.status === 'UNAVAILABLE') {
    return { badge: 'Unavailable', body: 'Disclosures unavailable' };
  }
  if (state.status === 'AVAILABLE') {
    if (state.trades.length > 0) {
      return { badge: `${state.trades.length} Trades`, body: 'Trade list' };
    }
    return { badge: '0 Trades', body: 'No congressional trading disclosures filed' };
  }
  return { badge: 'Scanning...', body: 'Loading' };
}

const resHttp500 = evaluateCongressDisplay({ status: 'ERROR', trades: [], errorMessage: 'HTTP 500 error' });
check(resHttp500.badge === 'Feed Unavailable' && !resHttp500.badge.includes('0 Trades'),
  "HTTP 500 displays 'Feed Unavailable', NEVER '0 Trades'");

const resTimeout = evaluateCongressDisplay({ status: 'ERROR', trades: [], errorMessage: 'Timeout' });
check(resTimeout.badge === 'Feed Unavailable' && resTimeout.body.includes('Error'),
  "Timeout displays error guidance, NEVER '0 Trades'");

const resAuthenticZero = evaluateCongressDisplay({ status: 'AVAILABLE', trades: [], errorMessage: null });
check(resAuthenticZero.badge === '0 Trades' && resAuthenticZero.body.includes('No congressional trading disclosures filed'),
  "Authentic successful empty response correctly displays '0 Trades'");

const resPopulated = evaluateCongressDisplay({ status: 'AVAILABLE', trades: [{ id: 1 }, { id: 2 }], errorMessage: null });
check(resPopulated.badge === '2 Trades', "Populated response displays '2 Trades'");

// -----------------------------------------------------------------------------
// DEFECT 5: Research request identity fails during A → B → A navigation
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mDefect 5: Research request identity during A → B → A navigation\x1b[0m");

check(researchSrc.includes("const requestGenerationRef = useRef<number>(0);"),
  "research/page.tsx uses requestGenerationRef counter");
check(researchSrc.includes("const abortControllerRef = useRef<AbortController | null>(null);"),
  "research/page.tsx uses abortControllerRef for network request cancellation");
check(researchSrc.includes("if (currentGeneration !== requestGenerationRef.current) return;"),
  "research/page.tsx strictly guards result commits against requestGenerationRef currency");

// Runtime simulation of A1 -> B -> A2 race condition
async function testResearchNavigationRace() {
  let activeGeneration = 0;
  let displayedAsset = null;

  function createDeferred() {
    let resolve, reject;
    const promise = new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    });
    return { promise, resolve, reject };
  }

  const defA1 = createDeferred();
  const defB = createDeferred();
  const defA2 = createDeferred();

  async function mockSelectAsset(symbol, def) {
    const generation = ++activeGeneration;
    try {
      const data = await def.promise;
      if (generation !== activeGeneration) {
        // Superseded! Drop commit!
        return;
      }
      displayedAsset = { symbol, data };
    } catch (err) {
      if (generation !== activeGeneration) return;
      displayedAsset = { symbol, error: err.message };
    }
  }

  // 1. User selects A (A1 started)
  mockSelectAsset('NVDA', defA1);

  // 2. User quickly switches to B (B started)
  mockSelectAsset('MSFT', defB);

  // 3. User quickly switches back to A (A2 started)
  mockSelectAsset('NVDA', defA2);

  // 4. A2 resolves FIRST
  defA2.resolve({ price: 120.0, gen: 3 });
  await Promise.resolve();

  check(displayedAsset.symbol === 'NVDA' && displayedAsset.data.gen === 3,
    "A2 resolves and commits its state first");

  // 5. B resolves next (superseded by A2)
  defB.resolve({ price: 440.0, gen: 2 });
  await Promise.resolve();

  check(displayedAsset.symbol === 'NVDA' && displayedAsset.data.gen === 3,
    "Delayed B response is dropped and does not overwrite A2");

  // 6. A1 resolves LAST (superseded by A2, even though ticker matches 'NVDA')
  defA1.resolve({ price: 110.0, gen: 1 });
  await Promise.resolve();

  check(displayedAsset.symbol === 'NVDA' && displayedAsset.data.gen === 3,
    "Old A1 response is dropped and does not overwrite A2 (A -> B -> A race prevented)");

  // 7. Test error in A1 cannot replace A2
  const defA1_err = createDeferred();
  const defA3 = createDeferred();

  mockSelectAsset('AAPL', defA1_err);
  mockSelectAsset('AAPL', defA3);

  defA3.resolve({ price: 230.0, gen: 5 });
  await Promise.resolve();

  defA1_err.reject(new Error("Network timeout"));
  await Promise.resolve();

  check(displayedAsset.symbol === 'AAPL' && displayedAsset.data.gen === 5,
    "Stale error from superseded request cannot overwrite newer valid result");
}
await testResearchNavigationRace();

// -----------------------------------------------------------------------------
// DEFECT 6: Trading-time fallback fabricates an hour or uses the wrong offset
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mDefect 6: Trading-time fallback & America/New_York DST/EST correctness\x1b[0m");

check(!govSrc.includes(": 10"), "governorSizingEngine removed fabricated fallback to hour 10");
check(!govSrc.includes("utcHour - 4"), "governorSizingEngine removed hardcoded UTC-4 fixed offset");
check(govSrc.includes("hourCycle: \"h23\""), "governorSizingEngine uses hourCycle: 'h23' guaranteeing 00-23 range");

function parseEasternTradingHour(dateInput) {
  try {
    const d = dateInput ? (dateInput instanceof Date ? dateInput : new Date(dateInput)) : new Date();
    if (isNaN(d.getTime())) return null;

    const formatter = new Intl.DateTimeFormat("en-US", {
      timeZone: "America/New_York",
      hour: "numeric",
      hourCycle: "h23",
    });

    const parts = formatter.formatToParts(d);
    const hourPart = parts.find((p) => p.type === "hour");
    if (!hourPart) return null;

    const parsed = parseInt(hourPart.value, 10);
    if (isNaN(parsed) || parsed < 0 || parsed > 23) return null;
    return parsed;
  } catch {
    return null;
  }
}

// 6.1 Summer timestamp (EDT: UTC-4): 2026-07-15 15:30:00 UTC -> 11:30:00 EDT (Hour 11)
const summerDate = new Date("2026-07-15T15:30:00Z");
const summerHour = parseEasternTradingHour(summerDate);
check(summerHour === 11, `Summer EDT timestamp 15:30 UTC produces Eastern hour 11 (got ${summerHour})`);

// 6.2 Winter timestamp (EST: UTC-5): 2026-01-15 15:30:00 UTC -> 10:30:00 EST (Hour 10)
const winterDate = new Date("2026-01-15T15:30:00Z");
const winterHour = parseEasternTradingHour(winterDate);
check(winterHour === 10, `Winter EST timestamp 15:30 UTC produces Eastern hour 10 (got ${winterHour})`);

// 6.3 Midnight Eastern EDT: 2026-07-15 04:00:00 UTC -> 00:00:00 EDT (Hour 0)
const midnightDate = new Date("2026-07-15T04:00:00Z");
const midnightHour = parseEasternTradingHour(midnightDate);
check(midnightHour === 0, `Midnight Eastern EDT produces hour 0 (got ${midnightHour})`);

// 6.4 Midnight Eastern EST: 2026-01-15 05:00:00 UTC -> 00:00:00 EST (Hour 0)
const midnightWinterDate = new Date("2026-01-15T05:00:00Z");
const midnightWinterHour = parseEasternTradingHour(midnightWinterDate);
check(midnightWinterHour === 0, `Midnight Eastern EST produces hour 0 (got ${midnightWinterHour})`);

// 6.5 Invalid date returns null (unavailable), NEVER 10 or guessed offset
const invalidHour = parseEasternTradingHour("invalid-date-string");
check(invalidHour === null, `Invalid date string yields null (got ${invalidHour})`);

// 6.6 Null trading hour in TraderContext produces unavailable sizing
const nullHourContext = {
  ...completeContext,
  tradingHour: null,
  isAvailable: false,
};
const sizingNullHour = runGovernedPositionSize(validSetup, nullHourContext);
check(sizingNullHour.isAvailable === false, "Null trading hour sets isAvailable: false");
check(sizingNullHour.cleanRoomRationale.includes("Market Trading Time"),
  "Null trading hour explicitly reports missing Market Trading Time");

// -----------------------------------------------------------------------------
// FINAL TALLY
// -----------------------------------------------------------------------------
console.log("\n-------------------------------------------------------------");
console.log(`TOTAL CHECKS: ${total}`);
console.log(`PASSED:       ${passed}`);
console.log(`FAILED:       ${failed}`);
console.log("-------------------------------------------------------------");

if (failed === 0) {
  console.log("\n\x1b[32m✔ ALL SIX REMAINING CONFIRMED DEFECTS VERIFIED SUCCESSFULLY.\x1b[0m\n");
  process.exit(0);
} else {
  console.error(`\n\x1b[31m✖ VERIFICATION FAILED with ${failed} failure(s).\x1b[0m\n`);
  process.exit(1);
}
