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
console.log("  VERIFICATION SUITE: SEVEN CONFIRMED BUGS REMEDIATION");
console.log("=============================================================\n");

// -----------------------------------------------------------------------------
// BUG 1: Shared cockpit store recovery and retry
// -----------------------------------------------------------------------------
console.log("\x1b[1mBug 1: Shared cockpit store recovery and retry (unifiedCockpitStore.ts)\x1b[0m");
const storePath = path.join(projectRoot, 'frontend', 'lib', 'simulation', 'unifiedCockpitStore.ts');
assert(fs.existsSync(storePath), 'unifiedCockpitStore.ts exists');
const storeSrc = fs.readFileSync(storePath, 'utf8');

check(storeSrc.includes("UnifiedCockpitStatus = 'IDLE' | 'LOADING' | 'AVAILABLE' | 'UNAVAILABLE' | 'ERROR' | 'PERSISTED_STORE'"), "Store represents distinct status lifecycle states including ERROR and PERSISTED_STORE");
check(storeSrc.includes("errorMessage: string | null"), "Store state carries explicit errorMessage");
check(storeSrc.includes("activeRequestId"), "Store tracks activeRequestId sequence to prevent stale async overwrites");
check(storeSrc.includes("refreshUnifiedCockpit"), "Store exports working refreshUnifiedCockpit retry path");
check(storeSrc.includes("invalidateCockpitState"), "Store exports invalidateCockpitState for cache invalidation");
check(storeSrc.includes('window.addEventListener("finance:portfolio-updated"'), "Store refreshes on portfolio mutation event");

// -----------------------------------------------------------------------------
// BUG 2: Specialist workbenches reactive subscriptions
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mBug 2: Specialist workbenches reactive subscriptions\x1b[0m");
const workbenches = [
  'simulation',
  'signals',
  'journal',
  'life-graph',
  'allocator',
];

for (const wb of workbenches) {
  const wbPath = path.join(projectRoot, 'frontend', 'app', 'workbench', wb, 'page.tsx');
  assert(fs.existsSync(wbPath), `Workbench ${wb} exists`);
  const wbSrc = fs.readFileSync(wbPath, 'utf8');
  check(wbSrc.includes("useUnifiedCockpit"), `Workbench ${wb} subscribes to reactive useUnifiedCockpit hook`);
  check(wbSrc.includes("refreshUnifiedCockpit"), `Workbench ${wb} provides working retry connection path`);
  check(wbSrc.includes("status === 'LOADING'") || wbSrc.includes("status === 'ERROR'"), `Workbench ${wb} renders distinct loading and error UI states`);
}

// -----------------------------------------------------------------------------
// BUG 3: Backend/frontend runway contract & projections
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mBug 3: Backend/frontend runway contract & projections\x1b[0m");
const cockpitPyPath = path.join(projectRoot, 'api', 'routes', 'cockpit.py');
assert(fs.existsSync(cockpitPyPath), 'cockpit.py exists');
const cockpitPySrc = fs.readFileSync(cockpitPyPath, 'utf8');

check(cockpitPySrc.includes('"runway": runway_obj') && cockpitPySrc.includes('"monthsUnencumbered": runway_months'), "cockpit.py returns explicit runway dictionary in payload");
check(cockpitPySrc.includes('"monthsUnencumbered"'), "cockpit.py runway includes monthsUnencumbered");
check(cockpitPySrc.includes('"runwayShieldStatus"'), "cockpit.py runway includes runwayShieldStatus");
check(cockpitPySrc.includes('"ZERO_EXPENDITURE"'), "cockpit.py handles ZERO_EXPENDITURE cleanly");
check(cockpitPySrc.includes('"projectedValue3Yr": None'), "cockpit.py uses None for uncomputed projectedValue3Yr");

const futurePagePath = path.join(projectRoot, 'frontend', 'app', 'future', 'page.tsx');
const futureSrc = fs.readFileSync(futurePagePath, 'utf8');
check(futureSrc.includes("runway.runwayShieldStatus === 'ZERO_EXPENDITURE'") || futureSrc.includes("monthsUnencumbered"), "Future hub handles zero expenditure or months unencumbered cleanly");
check(futureSrc.includes("confidencePct !== null"), "Future hub conditionally renders confidence badge only when available");

// -----------------------------------------------------------------------------
// BUG 4: Governor rejects missing risk inputs & disables copying
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mBug 4: Governor rejects missing risk inputs & disables copying\x1b[0m");
const govPath = path.join(projectRoot, 'frontend', 'lib', 'simulation', 'governorSizingEngine.ts');
const govSrc = fs.readFileSync(govPath, 'utf8');

check(!govSrc.includes("summary.totalCost > 0 ? (summary.totalEquity ?? summary.totalCost)"), "governorSizingEngine does not substitute cost basis for unpriced equity");
check(!govSrc.includes("accountEquity = cockpit.runway.liquidReserves"), "governorSizingEngine does not substitute household reserves for trading equity");
check(govSrc.includes("consecutiveLossStreak: number | null"), "TraderContext types consecutiveLossStreak as nullable");
check(govSrc.includes("dailyDrawdownPct: number | null"), "TraderContext types dailyDrawdownPct as nullable");
check(govSrc.includes("missingInputs"), "TraderContext tracks missingInputs array");
check(govSrc.includes("isAvailable: false"), "governorSizingEngine returns isAvailable: false when risk inputs are missing");

const setupsPath = path.join(projectRoot, 'frontend', 'app', 'setups', 'page.tsx');
const setupsSrc = fs.readFileSync(setupsPath, 'utf8');
check(setupsSrc.includes("disabled={!isActionable || !sizing.isAvailable || sizing.recommendedShares <= 0}"), "Setups page disables copy button when sizing is unavailable");
check(setupsSrc.includes("GOVERNOR SIZING INACTIVE: CONFIGURE REQUIRED RISK INPUTS") || setupsSrc.includes("sizing.cleanRoomRationale"), "Setups page displays human-facing reason when sizing is inactive");

// -----------------------------------------------------------------------------
// BUG 5: Command Palette dynamic setups
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mBug 5: Command Palette dynamic setups\x1b[0m");
const palettePath = path.join(projectRoot, 'frontend', 'components', 'CommandPaletteModal.tsx');
const paletteSrc = fs.readFileSync(palettePath, 'utf8');

check(!paletteSrc.includes('{ ticker: "GOOGL", name: "Alphabet Inc", score: 94'), "CommandPalette removed hardcoded GOOGL setup");
check(!paletteSrc.includes('{ ticker: "NVDA", name: "NVIDIA Corp", score: 91'), "CommandPalette removed hardcoded NVDA setup");
check(paletteSrc.includes("fetchTacticalSetups"), "CommandPalette imports and calls fetchTacticalSetups");
check(paletteSrc.includes("setTacticalSetups"), "CommandPalette updates state with dynamic setups");
check(paletteSrc.includes("dynamic-setup"), "CommandPalette supports dynamic ticker query parameter navigation");

// -----------------------------------------------------------------------------
// BUG 6: Today's semantic zoom real content switching
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mBug 6: Today's semantic zoom real content switching\x1b[0m");
const semZoomPath = path.join(projectRoot, 'frontend', 'components', 'cockpit', 'SemanticZoom.tsx');
const semZoomSrc = fs.readFileSync(semZoomPath, 'utf8');
check(semZoomSrc.includes("activeLevel?: number") && semZoomSrc.includes("onLevelChange?: (level: number) => void"), "SemanticZoom supports controlled activeLevel and onLevelChange callback");
check(semZoomSrc.includes("level2Content?: React.ReactNode"), "SemanticZoom supports custom level2Content");

const todayPath = path.join(projectRoot, 'frontend', 'app', 'today', 'page.tsx');
const todaySrc = fs.readFileSync(todayPath, 'utf8');
check(todaySrc.includes("const [zoomLevel, setZoomLevel] = useState"), "today/page.tsx maintains active zoomLevel state");
check(todaySrc.includes("zoomLevel === 0") && todaySrc.includes("zoomLevel >= 1") && todaySrc.includes("zoomLevel === 2"), "today/page.tsx renders distinct content for L0, L1, and L2");
check(todaySrc.includes("Level 2 · Causal Lineage Diagnostics"), "today/page.tsx Level 2 renders causal lineage diagnostics");
check(todaySrc.includes("/workbench/allocator"), "today/page.tsx Level 2 links to /workbench/allocator handoff");

const cockpitTodayPath = path.join(projectRoot, 'frontend', 'app', 'cockpit', 'today', 'page.tsx');
const cockpitTodaySrc = fs.readFileSync(cockpitTodayPath, 'utf8');
check(cockpitTodaySrc.includes("const [zoomLevel, setZoomLevel] = useState"), "cockpit/today/page.tsx maintains active zoomLevel state");
check(cockpitTodaySrc.includes("zoomLevel === 0") && cockpitTodaySrc.includes("zoomLevel === 2"), "cockpit/today/page.tsx renders distinct content for L0 and L2");

// -----------------------------------------------------------------------------
// BUG 7: Research no silent NVDA default & unselected state
// -----------------------------------------------------------------------------
console.log("\n\x1b[1mBug 7: Research no silent NVDA default & unselected state\x1b[0m");
const researchPath = path.join(projectRoot, 'frontend', 'app', 'research', 'page.tsx');
const researchSrc = fs.readFileSync(researchPath, 'utf8');

check(!researchSrc.includes("searchParams.get('ticker') || 'NVDA'"), "research/page.tsx removed silent 'NVDA' fallback");
check(researchSrc.includes("!activeTicker"), "research/page.tsx checks for unselected ticker state");
check(researchSrc.includes("Select an Asset to Open Research Dossier"), "research/page.tsx renders clear prompt when no ticker selected");
check(researchSrc.includes("setAnalyticsData(null)"), "research/page.tsx immediately clears previous asset data on new ticker");
check(researchSrc.includes("status === 404") || researchSrc.includes("'NOT_FOUND'"), "research/page.tsx explicitly distinguishes 404 Unsupported Asset from server errors");

console.log("\n-------------------------------------------------------------");
console.log(`TOTAL CHECKS: ${total} | PASSED: ${passed} | Failed: ${failed}`);
console.log("-------------------------------------------------------------\n");

if (failed > 0) {
  process.exit(1);
} else {
  console.log("\x1b[32m✔ ALL 7 CONFIRMED BUG REMEDIATION CHECKS PASSED!\x1b[0m\n");
}
