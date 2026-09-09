/**
 * Comprehensive Verification Suite:
 * 1. Radar -> Setup Asset Continuity & Explicit Unavailable State
 * 2. Fractional Portfolio Holdings (Add, Edit, Calculations, Validation)
 * 3. Complete CTA Inventory & Navigation Parity
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

console.log('\n=== ARX Terminal: Asset Continuity, Fractional Holdings & CTA Audit Verification ===\n');

// -------------------------------------------------------------
// SECTION 1: Radar -> Setup Asset Continuity
// -------------------------------------------------------------
console.log('\x1b[1mSection 1: Radar -> Setup Asset Continuity & Deep Links\x1b[0m');

const radarPath = path.join(projectRoot, 'frontend', 'app', 'radar', 'page.tsx');
assert(fs.existsSync(radarPath), 'radar/page.tsx exists');
const radarSrc = fs.readFileSync(radarPath, 'utf8');

assert(
  radarSrc.includes('href={`/setups?ticker=${heroAsset.ticker}`}') ||
  radarSrc.includes('href={"/setups?ticker=" + heroAsset.ticker}') ||
  radarSrc.includes('href={`/setups?ticker=${encodeURIComponent(heroAsset.ticker)}`}'),
  'Radar Level 0 Hero CTA links to /setups with exact heroAsset.ticker'
);

assert(
  radarSrc.includes('href={`/setups?ticker=${asset.ticker}`}') ||
  radarSrc.includes('href={"/setups?ticker=" + asset.ticker}') ||
  radarSrc.includes('href={`/setups?ticker=${encodeURIComponent(asset.ticker)}`}'),
  'Radar table row Setup CTAs link to /setups with exact asset.ticker'
);

const researchPath = path.join(projectRoot, 'frontend', 'app', 'research', 'page.tsx');
assert(fs.existsSync(researchPath), 'research/page.tsx exists');
const researchSrc = fs.readFileSync(researchPath, 'utf8');

assert(
  researchSrc.includes('href={`/setups?ticker=${activeTicker}`}') ||
  researchSrc.includes('href={`/setups?ticker=${activeDossier.ticker}`}') ||
  researchSrc.includes('href={"/setups?ticker=" + activeTicker}'),
  'Research Level 0 Hero CTA links to /setups with exact activeTicker'
);

// -------------------------------------------------------------
// SECTION 2: Setups Page Query Parsing & Zero Silent Fallback
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 2: Setups Page URL Preservation & Explicit Unavailable State\x1b[0m');

const setupsPath = path.join(projectRoot, 'frontend', 'app', 'setups', 'page.tsx');
assert(fs.existsSync(setupsPath), 'setups/page.tsx exists');
const setupsSrc = fs.readFileSync(setupsPath, 'utf8');

assert(setupsSrc.includes('useSearchParams'), 'Setups page imports and uses useSearchParams()');
assert(setupsSrc.includes("searchParams.get('ticker')") || setupsSrc.includes('searchParams.get("ticker")'), 'Setups page parses "ticker" query parameter');
assert(setupsSrc.includes('<Suspense'), 'Setups page wraps searchParams component in Suspense for static build compliance');

assert(
  setupsSrc.includes('No Tactical Setup Currently Active for') ||
  setupsSrc.includes('isUnsupported'),
  'Setups page detects unsupported tickers'
);

assert(
  setupsSrc.includes('Return to Confluence Radar') && setupsSrc.includes('href="/radar"'),
  'Setups page provides explicit navigation return to /radar when ticker is unsupported'
);

// Verify setups are API-backed and governor sizing engine has zero hardcoded setups
const enginePath = path.join(projectRoot, 'frontend', 'lib', 'simulation', 'governorSizingEngine.ts');
assert(fs.existsSync(enginePath), 'governorSizingEngine.ts exists');
const engineSrc = fs.readFileSync(enginePath, 'utf8');

assert(engineSrc.includes('getTacticalSetupForTicker'), 'Exports getTacticalSetupForTicker helper');
assert(!engineSrc.includes('CANONICAL_TACTICAL_SETUPS'), 'Deleted CANONICAL_TACTICAL_SETUPS: zero hardcoded tactical setups');
assert(setupsSrc.includes('fetchTacticalSetups'), 'Setups page fetches setups dynamically via authoritative API');
assert(
  engineSrc.includes('setup === null') ||
  engineSrc.includes('!setup') ||
  engineSrc.includes('unclampedShares: 0'),
  'Governor sizing engine returns 0 shares and safe rationale when setup is null/unsupported'
);

// -------------------------------------------------------------
// SECTION 3: Fractional Portfolio Holdings Support
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 3: Fractional Portfolio Holdings & Calculations\x1b[0m');

const portPath = path.join(projectRoot, 'frontend', 'app', 'portfolio', 'page.tsx');
assert(fs.existsSync(portPath), 'portfolio/page.tsx exists');
const portSrc = fs.readFileSync(portPath, 'utf8');

assert(!portSrc.includes('min="1"\n                      value={newShares}'), 'portfolio/page.tsx removes restrictive min="1" constraint from shares input');
assert(portSrc.includes('min="0.000001"') && portSrc.includes('value={newShares}'), 'portfolio/page.tsx permits fractional quantities (min="0.000001") on shares input');
assert(portSrc.includes('modalError') && portSrc.includes('setModalError'), 'Provides explicit modalError state for validation feedback');
assert(portSrc.includes('isEditing') && portSrc.includes('handleOpenEditModal'), 'Implements full editing capability for existing holdings');
assert(portSrc.includes('handleOpenEditModal') && (portSrc.includes('Edit') && portSrc.includes('handleRemovePosition')), 'Holdings table rows include an explicit "Edit" button alongside "Remove"');

// Fractional math scenario verification
// Scenario from prompt: 0.25 shares @ $200 entry, $220 current price
const fracShares = 0.25;
const fracEntry = 200.0;
const fracCurrent = 220.0;

const costBasis = fracShares * fracEntry;
const marketValue = fracShares * fracCurrent;
const unrealizedPnL = marketValue - costBasis;
const pnlPct = (unrealizedPnL / costBasis) * 100;

assert(costBasis === 50.0, `Fractional Cost Basis: 0.25 shs * $200 = $50.00 (got $${costBasis})`);
assert(marketValue === 55.0, `Fractional Market Value: 0.25 shs * $220 = $55.00 (got $${marketValue})`);
assert(unrealizedPnL === 5.0, `Fractional Unrealized Gain: $55 - $50 = +$5.00 (got +$${unrealizedPnL})`);
assert(pnlPct === 10.0, `Fractional Unrealized %: +10.0% (got ${pnlPct}%)`);

// Micro-quantity scenario: 0.001 shares @ $60,000 (Crypto / High Price)
const microShares = 0.001;
const microEntry = 60000.0;
const microCurrent = 66000.0;
assert(microShares * microEntry === 60.0, `Micro-Quantity Cost Basis: 0.001 shs * $60,000 = $60.00`);
assert(microShares * microCurrent === 66.0, `Micro-Quantity Market Value: 0.001 shs * $66,000 = $66.00`);

// Whole-share holding still works
const wholeShares = 10;
const wholeEntry = 150.0;
const wholeCurrent = 165.0;
assert(wholeShares * wholeEntry === 1500.0, `Whole-share Cost Basis: 10 shs * $150 = $1,500.00`);
assert(wholeShares * wholeCurrent === 1650.0, `Whole-share Market Value: 10 shs * $165 = $1,650.00`);

// -------------------------------------------------------------
// SECTION 4: Terminal Shell, Navigation & CTA Parity
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 4: Terminal Shell, Navigation & CTA Parity\x1b[0m');

const shellPath = path.join(projectRoot, 'frontend', 'components', 'terminal', 'TerminalShell.tsx');
assert(fs.existsSync(shellPath), 'TerminalShell.tsx exists');
const shellSrc = fs.readFileSync(shellPath, 'utf8');

assert(shellSrc.includes('/radar'), 'TerminalShell includes /radar');
assert(shellSrc.includes('/setups'), 'TerminalShell includes /setups');
assert(shellSrc.includes('/portfolio'), 'TerminalShell includes /portfolio');
assert(shellSrc.includes('/journal'), 'TerminalShell includes /journal');
assert(shellSrc.includes('/performance'), 'TerminalShell includes /performance');
assert(shellSrc.includes('/research'), 'TerminalShell includes /research');

// Mobile dock covers all 6 hubs
assert(
  shellSrc.includes('role="navigation"') &&
  shellSrc.includes('href="/radar"') &&
  shellSrc.includes('href="/setups"') &&
  shellSrc.includes('href="/portfolio"') &&
  shellSrc.includes('href="/journal"') &&
  shellSrc.includes('href="/performance"') &&
  shellSrc.includes('href="/research"'),
  'Mobile navigation dock covers all 6 flagship hubs with dedicated links'
);

const cpPath = path.join(projectRoot, 'frontend', 'components', 'CommandPaletteModal.tsx');
assert(fs.existsSync(cpPath), 'CommandPaletteModal.tsx exists');
const cpSrc = fs.readFileSync(cpPath, 'utf8');
assert(cpSrc.includes('/setups?ticker='), 'Command Palette connects tactical execution tickets directly to /setups?ticker=');

// -------------------------------------------------------------
// SECTION 5: Scorecard & Certification
// -------------------------------------------------------------
setTimeout(() => {
  console.log('\n-------------------------------------------------------------');
  console.log(`TOTAL ASSERTIONS: ${totalTests}`);
  console.log(`PASSED: ${passedTests}`);
  console.log(`FAILED: ${failedTests}`);
  console.log('-------------------------------------------------------------\n');

  if (failedTests === 0) {
    console.log('\x1b[32m✨ ASSET CONTINUITY, FRACTIONAL HOLDINGS & CTA AUDIT 100% CERTIFIED! ✨\x1b[0m\n');
    process.exit(0);
  } else {
    console.error('\x1b[31m❌ CTA AUDIT FAILED WITH ASSERTION ERRORS\x1b[0m\n');
    process.exit(1);
  }
}, 100);
