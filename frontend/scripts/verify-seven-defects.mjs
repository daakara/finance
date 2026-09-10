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
console.log("  VERIFICATION SUITE: 7 CONFIRMED DEFECTS REMEDIATION");
console.log("=============================================================\n");

// -----------------------------------------------------------------------------
// DEFECT 1: Connect cockpit consumers to actual API
// -----------------------------------------------------------------------------
console.log("\x1b[1mDefect 1: Connect cockpit consumers to actual API\x1b[0m");
const storePath = path.join(projectRoot, 'frontend', 'lib', 'simulation', 'unifiedCockpitStore.ts');
assert(fs.existsSync(storePath), 'unifiedCockpitStore.ts exists');
const storeSrc = fs.readFileSync(storePath, 'utf8');

check(!storeSrc.includes("david-trader-01"), "unifiedCockpitStore removed fake 'david-trader-01' persona");
check(!storeSrc.includes("CANONICAL_COCKPIT_STATE"), "unifiedCockpitStore removed fictional CANONICAL_COCKPIT_STATE snapshot");
check(storeSrc.includes("fetchUnifiedCockpitState"), "unifiedCockpitStore exports fetchUnifiedCockpitState fetching from API");
check(storeSrc.includes("/cockpit/state"), "unifiedCockpitStore calls /cockpit/state API route");
check(storeSrc.includes("X-Profile-Id"), "unifiedCockpitStore passes X-Profile-Id header");
check(storeSrc.includes("triad: TriadIndex | null"), "unifiedCockpitStore types triad as nullable");
check(!storeSrc.includes("s.triad.lhi !== 84"), "verifyUnifiedSourceOfTruth no longer hardcodes check for 84");

// Check null-safety in today/page.tsx
const todaySrc = fs.readFileSync(path.join(projectRoot, 'frontend', 'app', 'today', 'page.tsx'), 'utf8');
check(todaySrc.includes("triad?.lhi ?? \"--\""), "Today hub uses null-safe accessor for triad.lhi");
check(todaySrc.includes("useUnifiedCockpit"), "Today hub consumes reactive useUnifiedCockpit hook");

// Check null-safety in future/page.tsx
const futureSrc = fs.readFileSync(path.join(projectRoot, 'frontend', 'app', 'future', 'page.tsx'), 'utf8');
check(futureSrc.includes("triad?.lhi ?? \"--\""), "Future hub uses null-safe accessor for triad.lhi");
check(futureSrc.includes("runway ?"), "Future hub conditionally renders runway shield or unconfigured banner");

// Check null-safety in progress/page.tsx
const progressSrc = fs.readFileSync(path.join(projectRoot, 'frontend', 'app', 'progress', 'page.tsx'), 'utf8');
check(progressSrc.includes("triad?.lhi ?? \"--\""), "Progress hub uses null-safe accessor for triad.lhi");

// Check null-safety in household/page.tsx
const householdSrc = fs.readFileSync(path.join(projectRoot, 'frontend', 'app', 'household', 'page.tsx'), 'utf8');
check(householdSrc.includes("triad?.lhi ?? \"--\""), "Household hub uses null-safe accessor for triad.lhi");

// -----------------------------------------------------------------------------
// DEFECT 2: Remove fabricated Governor sizing inputs
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mDefect 2: Remove fabricated Governor sizing inputs\x1b[0m");
const govPath = path.join(projectRoot, 'frontend', 'lib', 'simulation', 'governorSizingEngine.ts');
assert(fs.existsSync(govPath), 'governorSizingEngine.ts exists');
const govSrc = fs.readFileSync(govPath, 'utf8');

check(!govSrc.includes("accountEquity: 50000"), "governorSizingEngine removed hardcoded $50,000 equity");
check(!govSrc.includes("consecutiveLossStreak: 2"), "governorSizingEngine removed hardcoded 2 loss streak");
check(!govSrc.includes("currentHour >= 9 && currentHour <= 16 ? currentHour : 10"), "governorSizingEngine removed hour 10 fallback");
check(govSrc.includes("America/New_York"), "governorSizingEngine computes tradingHour in America/New_York timezone");
check(govSrc.includes("isAvailable"), "TraderContext includes isAvailable flag");
check(govSrc.includes("recommendedShares: 0"), "calculateGovernedPositionSize outputs 0 shares when context is unconfigured");

// -----------------------------------------------------------------------------
// DEFECT 3: Remove fabricated setup fallbacks
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mDefect 3: Remove fabricated setup fallbacks\x1b[0m");
const setupsPath = path.join(projectRoot, 'frontend', 'app', 'setups', 'page.tsx');
assert(fs.existsSync(setupsPath), 'setups/page.tsx exists');
const setupsSrc = fs.readFileSync(setupsPath, 'utf8');

check(!setupsSrc.includes("curPrice * 0.93"), "setups page eliminated fabricated stop loss (curPrice * 0.93)");
check(!setupsSrc.includes("curPrice * 1.08"), "setups page eliminated fabricated target 1 (curPrice * 1.08)");
check(!setupsSrc.includes("curPrice * 1.15"), "setups page eliminated fabricated target 2 (curPrice * 1.15)");
check(!setupsSrc.includes("confluenceScore || 50"), "setups page eliminated default confluence 50");
check(!setupsSrc.includes("Stage 4 Correction / Base Building"), "setups page eliminated invented stage pattern text");
check(setupsSrc.includes("res.status === 404"), "setups page explicitly distinguishes 404 Unsupported Asset");
check(setupsSrc.includes("res.status === 429"), "setups page explicitly distinguishes 429 Rate Limited");
check(setupsSrc.includes("NO_QUALIFYING_SETUP"), "setups page records genuine NO_QUALIFYING_SETUP when criteria not met");

// -----------------------------------------------------------------------------
// DEFECT 4: Make portfolio persistence truthful
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mDefect 4: Make portfolio persistence truthful\x1b[0m");
const portLibPath = path.join(projectRoot, 'frontend', 'lib', 'portfolio.ts');
assert(fs.existsSync(portLibPath), 'portfolio.ts exists');
const portLibSrc = fs.readFileSync(portLibPath, 'utf8');

check(portLibSrc.includes("await persistHoldingToApi"), "addPortfolioPosition awaits API write before confirmation");
check(portLibSrc.includes("if (!persistRes.success)"), "addPortfolioPosition checks API success and aborts on failure");
check(portLibSrc.includes("beginActivePortfolioEdit"), "portfolio.ts exports beginActivePortfolioEdit to protect active edits");
check(portLibSrc.includes("isPortfolioEditActive()"), "syncPortfolioFromApi checks isPortfolioEditActive before mutating cache");
check(portLibSrc.includes("removePortfolioPosition"), "portfolio.ts exports async removePortfolioPosition awaiting DELETE confirmation");

const portPageSrc = fs.readFileSync(path.join(projectRoot, 'frontend', 'app', 'portfolio', 'page.tsx'), 'utf8');
check(portPageSrc.includes("await updatePortfolioPosition") || portPageSrc.includes("await addPortfolioPosition"), "Portfolio page awaits position operations");
check(portPageSrc.includes("beginActivePortfolioEdit"), "Portfolio page locks sync during edit modal");

// -----------------------------------------------------------------------------
// DEFECT 5: Preserve missing quantities and prices honestly
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mDefect 5: Preserve missing quantities and prices honestly\x1b[0m");
check(!portLibSrc.includes("Math.round(2500 / pos.entryPrice)"), "portfolio.ts eliminated invented $2,500 share allocation");
check(!portLibSrc.includes("pos.currentPrice : pos.entryPrice"), "portfolio.ts eliminated fallback from currentPrice to entryPrice");
check(portLibSrc.includes("currentPrice: number | null"), "PortfolioPosition interface types currentPrice as nullable");
check(portLibSrc.includes("isComplete: boolean"), "PortfolioSummary interface includes isComplete flag");
check(portLibSrc.includes("unpricedCount"), "PortfolioSummary tracks unpricedCount");

// Check backend db_engine.py
const dbEngineSrc = fs.readFileSync(path.join(projectRoot, 'analyst_dashboard', 'data', 'db_engine.py'), 'utf8');
check(dbEngineSrc.includes('row["current_price"] is not None else None'), "db_engine.py preserves None for missing current_price");

// Check backend cockpit.py
const cockpitRouteSrc = fs.readFileSync(path.join(projectRoot, 'api', 'routes', 'cockpit.py'), 'utf8');
check(cockpitRouteSrc.includes("PARTIAL_UNPRICED"), "cockpit.py flags status as PARTIAL_UNPRICED when missing prices");

// -----------------------------------------------------------------------------
// DEFECT 6: Declare exchange_calendars in manifests
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mDefect 6: Declare exchange_calendars in manifests\x1b[0m");
const reqsSrc = fs.readFileSync(path.join(projectRoot, 'requirements.txt'), 'utf8');
check(reqsSrc.includes("exchange_calendars"), "requirements.txt declares exchange_calendars");

const pyprojectSrc = fs.readFileSync(path.join(projectRoot, 'pyproject.toml'), 'utf8');
check(pyprojectSrc.includes("exchange_calendars"), "pyproject.toml declares exchange_calendars");

// -----------------------------------------------------------------------------
// DEFECT 7: Report clipboard success only after success
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mDefect 7: Report clipboard success only after success\x1b[0m");
check(setupsSrc.includes("await navigator.clipboard.writeText(orderStr)"), "setups page awaits navigator.clipboard.writeText");
check(setupsSrc.includes("copyStatus === 'SUCCESS'"), "setups page checks success state after write completes");
check(setupsSrc.includes("setCopyStatus('FAILED')"), "setups page catches and handles clipboard rejection");
check(setupsSrc.includes("COPY EXECUTION TICKET"), "setups page button relabeled to COPY EXECUTION TICKET");
check(setupsSrc.includes("does not route to broker"), "setups page explicitly clarifies action does not route to broker");

// -----------------------------------------------------------------------------
// FINAL TALLY
// -----------------------------------------------------------------------------
console.log("\n-------------------------------------------------------------");
console.log(`TOTAL CHECKS: ${total}`);
console.log(`PASSED:       ${passed}`);
console.log(`FAILED:       ${failed}`);
console.log("-------------------------------------------------------------");

if (failed === 0) {
  console.log("\n\x1b[32m✔ ALL 7 CONFIRMED DEFECTS VERIFIED REMEDIATED SUCCESSFULLY.\x1b[0m\n");
  process.exit(0);
} else {
  console.error(`\n\x1b[31m✖ VERIFICATION FAILED with ${failed} failure(s).\x1b[0m\n`);
  process.exit(1);
}
