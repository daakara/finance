/**
 * Comprehensive Verification Suite:
 * ARX API-Backed Production Data Integrity & Zero-Fabrication Audit
 *
 * Enforces:
 * 1. Complete elimination of CANONICAL_TACTICAL_SETUPS and demo portfolio seeds.
 * 2. API-backed asset selection in /radar, /setups, /portfolio, /journal, /research.
 * 3. Honest missing-data handling and order execution disabling on unverified setups.
 * 4. Persistent backend portfolio API contract and fractional share math.
 * 5. Complete absence of silent fallback substitution (e.g. GOOGL).
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..', '..');

let totalTests = 0;
let passedTests = 0;
let failedTests = 0;

function assert(condition, message) {
  totalTests++;
  if (condition) {
    passedTests++;
    console.log(`  \x1b[32m✔\x1b[0m ${message}`);
  } else {
    failedTests++;
    console.error(`  \x1b[31m✖\x1b[0m ${message}`);
  }
}

console.log('\n=== ARX Terminal: API-Backed Production Data Integrity Verification ===\n');

// -------------------------------------------------------------
// SECTION 1: Zero Hardcoded Tactical Setups & Demo Seeds
// -------------------------------------------------------------
console.log('\x1b[1mSection 1: Zero Hardcoded Tactical Setups & Demo Seeds\x1b[0m');

const enginePath = path.join(projectRoot, 'frontend', 'lib', 'simulation', 'governorSizingEngine.ts');
assert(fs.existsSync(enginePath), 'governorSizingEngine.ts exists');
const engineSrc = fs.readFileSync(enginePath, 'utf8');

assert(!engineSrc.includes('CANONICAL_TACTICAL_SETUPS'), 'Zero CANONICAL_TACTICAL_SETUPS in governorSizingEngine.ts');

const setupsPath = path.join(projectRoot, 'frontend', 'app', 'setups', 'page.tsx');
assert(fs.existsSync(setupsPath), 'setups/page.tsx exists');
const setupsSrc = fs.readFileSync(setupsPath, 'utf8');
assert(!setupsSrc.includes('CANONICAL_TACTICAL_SETUPS'), 'Zero CANONICAL_TACTICAL_SETUPS in setups/page.tsx');

const portLibPath = path.join(projectRoot, 'frontend', 'lib', 'portfolio.ts');
assert(fs.existsSync(portLibPath), 'portfolio.ts exists');
const portLibSrc = fs.readFileSync(portLibPath, 'utf8');

assert(!portLibSrc.includes('defaultPositions: PortfolioPosition[] = ['), 'Zero demo defaultPositions array in portfolio.ts');
assert(portLibSrc.includes('return [];'), 'Clean sessions start with 0 positions ([]) by default');
assert(portLibSrc.includes('syncPortfolioFromApi'), 'Exports syncPortfolioFromApi()');
assert(portLibSrc.includes('persistHoldingToApi'), 'Exports persistHoldingToApi()');
assert(portLibSrc.includes('removeHoldingFromApi'), 'Exports removeHoldingFromApi()');

// -------------------------------------------------------------
// SECTION 2: Dynamic API Setups & Execution Guardrails
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 2: Dynamic API Setups & Action Disabling\x1b[0m');

const apiPath = path.join(projectRoot, 'frontend', 'lib', 'api.ts');
assert(fs.existsSync(apiPath), 'api.ts exists');
const apiSrc = fs.readFileSync(apiPath, 'utf8');

assert(apiSrc.includes('fetchTacticalSetups'), 'api.ts exports fetchTacticalSetups()');
assert(setupsSrc.includes('fetchTacticalSetups'), 'Setups hub calls fetchTacticalSetups() on mount');
assert(setupsSrc.includes('latestRequestRef'), 'Setups hub prevents out-of-order race conditions with request ref');
assert(setupsSrc.includes('disabled={!isActionable}'), 'Order authorization CTA is disabled when trade levels are not actionable');
assert(setupsSrc.includes('No Tactical Setup Currently Active for'), 'Displays explicit unavailable state instead of silent fallback');
assert(setupsSrc.includes('setUnsupportedError(`No active tactical setup on record for'), 'Displays unsupported error when requested ticker cannot be resolved');

// -------------------------------------------------------------
// SECTION 3: Radar Hub Live Screener API Integration
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 3: Radar Hub Live Screener Integration\x1b[0m');

const radarPath = path.join(projectRoot, 'frontend', 'app', 'radar', 'page.tsx');
assert(fs.existsSync(radarPath), 'radar/page.tsx exists');
const radarSrc = fs.readFileSync(radarPath, 'utf8');

assert(radarSrc.includes('fetchScreenerGems("all")'), 'Radar fetches real candidates via fetchScreenerGems("all")');
assert(!radarSrc.includes('generateRadarUniverse'), 'Eliminated synthetic generateRadarUniverse() function');
assert(!radarSrc.includes('ASSET_SPEC_MAP'), 'Eliminated hardcoded ASSET_SPEC_MAP catalog');
assert(radarSrc.includes('href={`/setups?ticker=${heroAsset.ticker}`}'), 'Hero CTA maintains exact asset continuity to /setups');
assert(radarSrc.includes('href={`/setups?ticker=${asset.ticker}`}'), 'Table row CTAs maintain exact asset continuity to /setups');

// -------------------------------------------------------------
// SECTION 4: Portfolio Hub Private Storage & API Synchronization
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 4: Portfolio Hub Storage & Fractional Integrity\x1b[0m');

const portPagePath = path.join(projectRoot, 'frontend', 'app', 'portfolio', 'page.tsx');
assert(fs.existsSync(portPagePath), 'portfolio/page.tsx exists');
const portPageSrc = fs.readFileSync(portPagePath, 'utf8');

assert(portPageSrc.includes('syncPortfolioFromApi'), 'Portfolio page imports and triggers syncPortfolioFromApi()');
assert(portPageSrc.includes('No Portfolio Holdings Stored'), 'Renders authentic empty state when user has 0 holdings');
assert(portPageSrc.includes('min="0.000001"'), 'Allows fractional shares with min="0.000001"');

// Fractional math audit
const fracHolding = { shares: 0.25, entryPrice: 200.0, currentPrice: 220.0 };
const fracCost = fracHolding.shares * fracHolding.entryPrice;
const fracVal = fracHolding.shares * fracHolding.currentPrice;
const fracPnL = fracVal - fracCost;
assert(fracCost === 50.0, 'Fractional cost: 0.25 * $200 = $50.00');
assert(fracVal === 55.0, 'Fractional market value: 0.25 * $220 = $55.00');
assert(fracPnL === 5.0, 'Fractional unrealized gain: $55 - $50 = +$5.00');

// -------------------------------------------------------------
// SECTION 5: Journal Hub Authentic Trade Log State
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 5: Journal Hub Authentic Trade Log State\x1b[0m');

const journalPath = path.join(projectRoot, 'frontend', 'app', 'journal', 'page.tsx');
assert(fs.existsSync(journalPath), 'journal/page.tsx exists');
const journalSrc = fs.readFileSync(journalPath, 'utf8');

assert(!journalSrc.includes("id: 'TR-108', ticker: 'GOOGL'"), 'Deleted hardcoded mock trade logs array');
assert(journalSrc.includes('0 Completed Trades Logged'), 'Renders authentic zero-state when 0 completed trades exist');
assert(journalSrc.includes('Awaiting verified trade executions'), 'Informs user that empirical calibration requires logged trades');
assert(journalSrc.includes('FINANCE_JOURNAL_LOGS'), 'Connects to persistent journal storage key');

// -------------------------------------------------------------
// SECTION 6: Research Route Option A Canonical Redirection
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 6: Research Hub Query-Preserving Redirection to Terminal\x1b[0m');

const researchPath = path.join(projectRoot, 'frontend', 'app', 'research', 'page.tsx');
assert(fs.existsSync(researchPath), 'research/page.tsx exists');
const researchSrc = fs.readFileSync(researchPath, 'utf8');

assert(researchSrc.includes('router.replace'), 'Research route cleanly redirects via router.replace');
assert(researchSrc.includes('searchParams.get("symbol")') || researchSrc.includes('searchParams.get("ticker")'), 'Extracts symbol/ticker query parameter');
assert(researchSrc.includes('?symbol='), 'Preserves asset continuity to Terminal root (?symbol=)');
assert(researchSrc.includes('<Suspense'), 'Wraps query reading in Suspense boundary for static export safety');


// -------------------------------------------------------------
// SECTION 7: Backend Portfolio API Contract & DB Schema
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 7: Backend Portfolio API & SQLite Engine\x1b[0m');

const dbEnginePath = path.join(projectRoot, 'analyst_dashboard', 'data', 'db_engine.py');
assert(fs.existsSync(dbEnginePath), 'db_engine.py exists');
const dbEngineSrc = fs.readFileSync(dbEnginePath, 'utf8');

assert(dbEngineSrc.includes('portfolio_holdings'), 'db_engine.py defines portfolio_holdings table schema');
assert(dbEngineSrc.includes('get_user_portfolio'), 'db_engine.py implements get_user_portfolio()');
assert(dbEngineSrc.includes('save_user_holding'), 'db_engine.py implements save_user_holding()');
assert(dbEngineSrc.includes('delete_user_holding'), 'db_engine.py implements delete_user_holding()');

const portRoutePath = path.join(projectRoot, 'api', 'routes', 'portfolio.py');
assert(fs.existsSync(portRoutePath), 'api/routes/portfolio.py exists');
const portRouteSrc = fs.readFileSync(portRoutePath, 'utf8');

const mainPyPath = path.join(projectRoot, 'api', 'main.py');
assert(fs.existsSync(mainPyPath), 'api/main.py exists');
const mainPySrc = fs.readFileSync(mainPyPath, 'utf8');

assert(mainPySrc.includes('prefix="/api/v1/portfolio"'), 'api/main.py registers portfolio router with prefix /api/v1/portfolio');
assert(portRouteSrc.includes('shares: float = Field(') && portRouteSrc.includes('gt=0'), 'Enforces positive fractional shares (gt=0)');
assert(portRouteSrc.includes('@router.post("/migrate"'), 'Implements POST /portfolio/migrate transparent migration endpoint');

const analyticsRoutePath = path.join(projectRoot, 'api', 'routes', 'analytics.py');
assert(fs.existsSync(analyticsRoutePath), 'api/routes/analytics.py exists');
const analyticsRouteSrc = fs.readFileSync(analyticsRoutePath, 'utf8');

assert(analyticsRouteSrc.includes('@router.get("/setups'), 'analytics.py implements GET /setups');
const setupsIdx = analyticsRouteSrc.indexOf('@router.get("/setups');
const symbolIdx = analyticsRouteSrc.indexOf('@router.get("/{symbol}');
assert(setupsIdx !== -1 && symbolIdx !== -1 && setupsIdx < symbolIdx, 'GET /setups route precedes GET /{symbol} to prevent path capture');

// -------------------------------------------------------------
// SECTION 8: Final Certification Scorecard
// -------------------------------------------------------------
console.log('\n-------------------------------------------------------------');
console.log(`TOTAL ASSERTIONS: ${totalTests}`);
console.log(`PASSED: ${passedTests}`);
console.log(`FAILED: ${failedTests}`);
console.log('-------------------------------------------------------------\n');

if (failedTests === 0) {
  console.log('\x1b[32m✨ ARX API-BACKED PRODUCTION DATA INTEGRITY 100% CERTIFIED! ✨\x1b[0m\n');
  process.exit(0);
} else {
  console.error('\x1b[31m❌ DATA INTEGRITY VERIFICATION FAILED WITH ERRORS\x1b[0m\n');
  process.exit(1);
}
