// frontend/scripts/verify-horizon14-navigation.mjs
// Verification Suite for Horizon 14.1: Terminal Shell Consolidation & Navigation Integrity
// Validates INV-OI115-P (Persistent Terminal Navigation) and H14-Gate-11 Certification

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..', '..');

let passedAssertions = 0;
let failedAssertions = 0;

function assert(condition, message) {
  if (condition) {
    passedAssertions++;
    console.log(`  \x1b[32m✔\x1b[0m ${message}`);
  } else {
    failedAssertions++;
    console.error(`  \x1b[31m✖ FAIL:\x1b[0m ${message}`);
  }
}

console.log('\x1b[1m\x1b[36m=== Horizon 14.1 Terminal Shell & Navigation Verification (INV-OI115-P) ===\x1b[0m\n');

// -------------------------------------------------------------
// SECTION 1: Persistent Terminal Shell (TerminalShell.tsx)
// -------------------------------------------------------------
console.log('\x1b[1mSection 1: Shared Terminal Shell (TerminalShell.tsx)\x1b[0m');
const shellPath = path.join(projectRoot, 'frontend', 'components', 'terminal', 'TerminalShell.tsx');
assert(fs.existsSync(shellPath), 'TerminalShell.tsx component exists');

if (fs.existsSync(shellPath)) {
  const shellSrc = fs.readFileSync(shellPath, 'utf8');

  assert(shellSrc.includes('import Navbar from "../Navbar"'), 'TerminalShell imports shared Navbar');
  assert(shellSrc.includes('export type TerminalHub ='), 'Defines TerminalHub type alias');
  assert(shellSrc.includes('export type TerminalHubId = "radar" | "setups" | "portfolio" | "journal" | "performance" | "research"'), 'TerminalHubId union covers all 6 flagship hubs');
  assert(shellSrc.includes('TERMINAL_HUBS: TerminalHubMeta[]'), 'Defines metadata record for all flagship hubs');
  
  // Single question focus for all 6 hubs
  assert(shellSrc.includes('"What deserves attention today?"'), 'Radar answers: "What deserves attention today?"');
  assert(shellSrc.includes('"What is actionable right now?"'), 'Setups answers: "What is actionable right now?"');
  assert(shellSrc.includes('"What risk am I carrying?"'), 'Portfolio answers: "What risk am I carrying?"');
  assert(shellSrc.includes('"Did I follow my rules?"'), 'Journal answers: "Did I follow my rules?"');
  assert(shellSrc.includes('"Is ARX actually improving my results?"'), 'Performance answers: "Is ARX actually improving my results?"');
  assert(shellSrc.includes('"Why does this opportunity exist?"'), 'Research answers: "Why does this opportunity exist?"');

  // UI Invariants inside the Shell
  assert(shellSrc.includes('INV-OI115-P'), 'References INV-OI115-P Persistent Terminal Navigation invariant');
  assert(shellSrc.includes('border-t border-[#1e293b]'), 'Persistent terminal sub-header divider');
  assert(shellSrc.includes('Link') && shellSrc.includes('hub.route'), 'Sub-header renders navigational links for fast switching');
  assert(shellSrc.includes('activeHub === hub.id'), 'Sub-header provides visual active tab distinction');
  assert(shellSrc.includes('REGIME: Confirmed Uptrend'), 'Persistent Market Regime status badge displayed');
  assert(shellSrc.includes('🛡️ Governor'), 'Persistent Behavioral Governor status link to /cockpit');
  assert(shellSrc.includes('⌘K') || shellSrc.includes('Cmd+K'), 'Command Palette shortcut indicator present');
  assert(shellSrc.includes('{children}'), 'Shell wraps page content children cleanly without re-mounting root shell');

  // Mobile navigation dock inside TerminalShell
  assert(shellSrc.includes('role="navigation"') && shellSrc.includes('Mobile Terminal Navigation'), 'TerminalShell contains dedicated Mobile Terminal Navigation dock');
  assert(shellSrc.includes('href="/radar"') && shellSrc.includes('Radar'), 'Mobile dock contains /radar');
  assert(shellSrc.includes('href="/setups"') && shellSrc.includes('Setups'), 'Mobile dock contains /setups');
  assert(shellSrc.includes('href="/portfolio"') && shellSrc.includes('Portfolio'), 'Mobile dock contains /portfolio');
  assert(shellSrc.includes('href="/journal"') && shellSrc.includes('Journal'), 'Mobile dock contains /journal');
  assert(shellSrc.includes('href="/performance"') && shellSrc.includes('Alpha'), 'Mobile dock contains /performance (Alpha)');
}

// -------------------------------------------------------------
// SECTION 2: Navbar Integrity & Governor Integration
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 2: Primary Navbar & Top Navigation\x1b[0m');
const navbarPath = path.join(projectRoot, 'frontend', 'components', 'Navbar.tsx');
assert(fs.existsSync(navbarPath), 'Navbar.tsx component exists');

if (fs.existsSync(navbarPath)) {
  const navSrc = fs.readFileSync(navbarPath, 'utf8');

  // Desktop navigation links
  assert(navSrc.includes('href="/radar"'), 'Navbar includes /radar link');
  assert(navSrc.includes('href="/setups"'), 'Navbar includes /setups link');
  assert(navSrc.includes('href="/portfolio"'), 'Navbar includes /portfolio link');
  assert(navSrc.includes('href="/journal"'), 'Navbar includes /journal link');
  assert(navSrc.includes('href="/performance"'), 'Navbar includes /performance link');
  assert(navSrc.includes('href="/research"'), 'Navbar includes /research link');

  // Governor integration
  assert(navSrc.includes('href="/cockpit"'), 'Navbar links directly to Governor Cockpit (/cockpit)');
  assert(navSrc.includes('Governor'), 'Navbar renders Behavioral Governor badge');

  // Command palette button
  assert(navSrc.includes('isCommandPaletteOpen'), 'Navbar manages Command Palette open/close state');
  assert(navSrc.includes('CommandPaletteModal'), 'Navbar renders CommandPaletteModal');
}

// -------------------------------------------------------------
// SECTION 3: Flagship Pages Shell Adoption
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 3: Flagship Terminal Hubs Shell Wrapping\x1b[0m');
const hubs = [
  { name: 'Radar', file: 'radar/page.tsx', hubKey: 'radar' },
  { name: 'Setups', file: 'setups/page.tsx', hubKey: 'setups' },
  { name: 'Portfolio', file: 'portfolio/page.tsx', hubKey: 'portfolio' },
  { name: 'Journal', file: 'journal/page.tsx', hubKey: 'journal' },
  { name: 'Performance', file: 'performance/page.tsx', hubKey: 'performance' },
  { name: 'Research', file: 'research/page.tsx', hubKey: 'research' },
];

for (const hub of hubs) {
  const pagePath = path.join(projectRoot, 'frontend', 'app', hub.file);
  assert(fs.existsSync(pagePath), `${hub.name} page exists (${hub.file})`);

  if (fs.existsSync(pagePath)) {
    const src = fs.readFileSync(pagePath, 'utf8');
    assert(src.includes('TerminalShell'), `${hub.name} imports TerminalShell`);
    assert(
      src.includes(`<TerminalShell activeHub="${hub.hubKey}">`),
      `${hub.name} properly mounts TerminalShell with activeHub="${hub.hubKey}"`
    );
    assert(src.includes('</TerminalShell>'), `${hub.name} properly closes </TerminalShell>`);
    assert(!src.includes('<Navbar />'), `${hub.name} does not have duplicate standalone <Navbar /> outside shell`);
  }
}

// -------------------------------------------------------------
// SECTION 4: Governor Cockpit Portal & Workbench Routing
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 4: Governor Cockpit Portal & Workbench Routing\x1b[0m');
const cockpitPath = path.join(projectRoot, 'frontend', 'app', 'cockpit', 'page.tsx');
assert(fs.existsSync(cockpitPath), 'Cockpit portal page exists (/cockpit)');

if (fs.existsSync(cockpitPath)) {
  const cockpitSrc = fs.readFileSync(cockpitPath, 'utf8');
  assert(cockpitSrc.includes('useUnifiedCockpit') || cockpitSrc.includes('getUnifiedCockpitState'), 'Cockpit consumes centralized unifiedCockpitStore');
  assert(cockpitSrc.includes('href="/radar"'), 'Cockpit provides Return to Terminal link (/radar)');
  
  // 4 Core Human Hubs
  assert(cockpitSrc.includes('href="/today"'), 'Cockpit links to Core Hub: Today');
  assert(cockpitSrc.includes('href="/future"'), 'Cockpit links to Core Hub: Future');
  assert(cockpitSrc.includes('href="/progress"'), 'Cockpit links to Core Hub: Progress');
  assert(cockpitSrc.includes('href="/household"'), 'Cockpit links to Core Hub: Household');

  // Specialist Workbenches mapped from store
  assert(cockpitSrc.includes('workbenches.map'), 'Cockpit iterates through specialist workbenches dynamically');
  assert(cockpitSrc.includes('wb.route'), 'Cockpit maps workbench routes from state store');
}

// -------------------------------------------------------------
// SECTION 5: Governor Sizing Engine Integration
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 5: Governor Sizing Engine & CQRS Bridge\x1b[0m');
const govEnginePath = path.join(projectRoot, 'frontend', 'lib', 'simulation', 'governorSizingEngine.ts');
assert(fs.existsSync(govEnginePath), 'governorSizingEngine.ts exists');

if (fs.existsSync(govEnginePath)) {
  const govSrc = fs.readFileSync(govEnginePath, 'utf8');
  assert(govSrc.includes('getTraderContextFromUnifiedCockpit'), 'Exports getTraderContextFromUnifiedCockpit helper');
  assert(govSrc.includes('calculateGovernedPositionSize'), 'Implements calculateGovernedPositionSize');
  assert(govSrc.includes('clampPenalty') && govSrc.includes('consecutiveLossStreak'), 'Applies sizing clamp penalties on loss streak');
  assert(govSrc.includes('INV-OI112-P') && govSrc.includes('INV-OI114-P'), 'Enforces Horizon 14 Invariants in governor sizing');
}

// -------------------------------------------------------------
// SECTION 6: Invariant Ledger & H14-Gate-11 Certification
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 6: Invariant Ledger & H14-Gate-11 Certification\x1b[0m');
const invPath = path.join(projectRoot, 'frontend', 'lib', 'simulation', 'horizon14Invariants.ts');
assert(fs.existsSync(invPath), 'horizon14Invariants.ts exists');

if (fs.existsSync(invPath)) {
  const invSrc = fs.readFileSync(invPath, 'utf8');
  assert(invSrc.includes('INV-OI115-P'), 'Ledger contains INV-OI115-P Persistent Terminal Navigation');
  assert(invSrc.includes('INV-OI112-P'), 'Ledger contains INV-OI112-P Experience Boundary Integrity');
  assert(invSrc.includes('INV-OI113-P'), 'Ledger contains INV-OI113-P Counterfactual Proof Determinism');
  assert(invSrc.includes('INV-OI114-P'), 'Ledger contains INV-OI114-P Human Agency Sizing Bounds');
  assert(invSrc.includes('verifyPersistentTerminalNavigation'), 'Exports verifyPersistentTerminalNavigation verifier');
}

// -------------------------------------------------------------
// SUMMARY
// -------------------------------------------------------------
console.log('\n-------------------------------------------------------------');
console.log(`Total Assertions: ${passedAssertions + failedAssertions}`);
console.log(`\x1b[32mPassed: ${passedAssertions}\x1b[0m`);
console.log(`\x1b[31mFailed: ${failedAssertions}\x1b[0m`);

if (failedAssertions === 0) {
  console.log('\x1b[1m\x1b[32m\n✔ H14-Gate-11 (Persistent Terminal Navigation) CERTIFIED PASSED.\x1b[0m\n');
  process.exit(0);
} else {
  console.error('\x1b[1m\x1b[31m\n✖ H14-Gate-11 VERIFICATION FAILED with ' + failedAssertions + ' error(s).\x1b[0m\n');
  process.exit(1);
}
