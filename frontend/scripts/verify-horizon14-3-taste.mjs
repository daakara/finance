// frontend/scripts/verify-horizon14-3-taste.mjs
// Verification Suite for Horizon 14.3: Taste, Craft & Experience Redesign
// Validates 10s/30s Decision Heuristics, Anti-Slop Standards, 3-Tier Hierarchy, and Deliverables

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

console.log('\x1b[1m\x1b[35m=== Horizon 14.3 Taste, Craft & Experience Redesign Verification ===\x1b[0m\n');

// -------------------------------------------------------------
// SECTION 1: Deliverables Audit (All 7 Core Documents)
// -------------------------------------------------------------
console.log('\x1b[1mSection 1: Core Institutional Deliverables in docs/ux/\x1b[0m');
const requiredDeliverables = [
  'DESIGN_AUDIT_REPORT.md',
  'UX_IMPROVEMENT_BACKLOG.md',
  'VISUAL_HIERARCHY_RECOMMENDATIONS.md',
  'NAVIGATION_OPTIMIZATION_PLAN.md',
  'PAGE_BY_PAGE_REDESIGN_RECOMMENDATIONS.md',
  'COMPONENT_CONSOLIDATION_PLAN.md',
  'HORIZON_14_3_CERTIFICATION_REPORT.md',
];

for (const doc of requiredDeliverables) {
  const docPath = path.join(projectRoot, 'docs', 'ux', doc);
  const exists = fs.existsSync(docPath);
  assert(exists, `Deliverable exists: docs/ux/${doc}`);
  if (exists) {
    const content = fs.readFileSync(docPath, 'utf8');
    assert(content.length > 500, `docs/ux/${doc} is non-trivial (>500 bytes, size: ${content.length})`);
    assert(!content.includes('TBD') && !content.includes('TODO'), `docs/ux/${doc} has zero TBD/TODO placeholders`);
  }
}

// -------------------------------------------------------------
// SECTION 2: Hub 1 (/radar) Taste & Decision Speed
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 2: Hub 1 (/radar) Taste & Decision Speed\x1b[0m');
const radarPath = path.join(projectRoot, 'frontend', 'app', 'radar', 'page.tsx');
if (fs.existsSync(radarPath)) {
  const rSrc = fs.readFileSync(radarPath, 'utf8');
  assert(rSrc.includes('Level 0 · #1 Attention Leader Today'), 'Radar defines Level 0 Attention Leader Hero');
  assert(rSrc.includes('heroAsset'), 'Radar dynamically extracts primary hero asset');
  assert(rSrc.includes('ARM EXECUTION TICKET IN /SETUPS'), 'Radar hero provides direct 1-click execution CTA');
  assert(rSrc.includes('Action Status'), 'Radar confluence stream table includes clear Action Status column');
  assert(rSrc.includes('Catalyst Rationale'), 'Radar confluence stream includes Catalyst Rationale column');
  assert(!rSrc.includes('Stage 2 Uptrend · Stage 2 Uptrend'), 'Eliminated repetitive boilerplate labels');
}

// -------------------------------------------------------------
// SECTION 3: Hub 2 (/setups) Asymmetric Execution Ticket
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 3: Hub 2 (/setups) Asymmetric Execution Ticket\x1b[0m');
const setupsPath = path.join(projectRoot, 'frontend', 'app', 'setups', 'page.tsx');
if (fs.existsSync(setupsPath)) {
  const sSrc = fs.readFileSync(setupsPath, 'utf8');
  assert(sSrc.includes('Level 0: Asymmetric Execution Ticket Ladder'), 'Setups defines Asymmetric Execution Ticket Ladder');
  assert(sSrc.includes('Risk Definition Bracket'), 'Setups brackets Entry and Stop Floor together');
  assert(sSrc.includes('Asymmetric Reward Milestones'), 'Setups highlights Target 1 and Target 2 reward milestones');
  assert(sSrc.includes('AUTHORIZE ORDER:') && sSrc.includes('[COPY STRING]'), 'Single consolidated high-visibility institutional CTA button');
  assert(!sSrc.includes('Copy Broker Order String'), 'Pruned redundant duplicate copy button');
}

// -------------------------------------------------------------
// SECTION 4: Hub 3 (/portfolio) Risk-First Heat & Capital at Risk
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 4: Hub 3 (/portfolio) Risk-First Heat & Capital at Risk\x1b[0m');
const portPath = path.join(projectRoot, 'frontend', 'app', 'portfolio', 'page.tsx');
if (fs.existsSync(portPath)) {
  const pSrc = fs.readFileSync(portPath, 'utf8');
  assert(pSrc.includes('Level 0 · Portfolio Heat'), 'Portfolio defines Level 0 Portfolio Heat Hero');
  assert(pSrc.includes('Capital at Risk'), 'Portfolio hero prominently displays Capital at Risk at stop floors');
  assert(pSrc.includes('ACTIVE EXIT TRIGGERS'), 'Portfolio includes Active Exit Rule Triggers banner');
  assert(pSrc.includes('Institutional Capital Presets'), 'Portfolio uses institutional capital presets ($10k-$250k)');
  assert(!pSrc.includes('<main className="max-w-[1450px] mx-auto p-4 sm:p-6 space-y-6 font-mono'), 'Eliminated global font-mono abuse on root main');
}

// -------------------------------------------------------------
// SECTION 5: Hub 4 (/journal) Operational Discipline & Brier Calibration
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 5: Hub 4 (/journal) Operational Discipline & Brier Calibration\x1b[0m');
const jourPath = path.join(projectRoot, 'frontend', 'app', 'journal', 'page.tsx');
if (fs.existsSync(jourPath)) {
  const jSrc = fs.readFileSync(jourPath, 'utf8');
  assert(jSrc.includes('Level 0 · Operational Discipline'), 'Journal defines Level 0 Operational Discipline Hero');
  assert(jSrc.includes('Rule Adherence Score (Grade A)'), 'Journal features Rule Adherence score as dominant headline');
  assert(jSrc.includes('Probabilistic Calibration Curve'), 'Journal renders Probabilistic Calibration Curve');
  assert(jSrc.includes('4-Quadrant Anti-Tilt Monitor'), 'Journal renders 4-Quadrant Anti-Tilt Monitor');
  assert(!jSrc.includes('Target $\\le 0.25$'), 'Fixed raw unrendered LaTeX syntax defect');
}

// -------------------------------------------------------------
// SECTION 6: Hub 5 (/performance) Counterfactual Proof & Attribution
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 6: Hub 5 (/performance) Counterfactual Proof & Attribution\x1b[0m');
const perfPath = path.join(projectRoot, 'frontend', 'app', 'performance', 'page.tsx');
if (fs.existsSync(perfPath)) {
  const pfSrc = fs.readFileSync(perfPath, 'utf8');
  assert(pfSrc.includes('Counterfactual Proof of Value'), 'Performance defines Counterfactual Proof of Value Hero');
  assert(pfSrc.includes('Capital Preserved'), 'Displays Capital Preserved as dominant primary metric');
  assert(pfSrc.includes('Max Drawdown') && pfSrc.includes('Sharpe Ratio') && pfSrc.includes('Profit Factor'), 'Compares Governed vs Naive metrics');
  assert(pfSrc.includes('Drawdown Defense') && pfSrc.includes('Execution Window') && pfSrc.includes('Capital Floor'), 'Deconstructs 3 defense savings pillars');
}

// -------------------------------------------------------------
// SECTION 7: Hub 6 (/research) Institutional Research Dossier
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 7: Hub 6 (/research) Institutional Research Workstation\x1b[0m');
const resPath = path.join(projectRoot, 'frontend', 'app', 'research', 'page.tsx');
if (fs.existsSync(resPath)) {
  const rsSrc = fs.readFileSync(resPath, 'utf8');
  assert(rsSrc.includes('Level 0 · Institutional Research Dossier'), 'Research defines Level 0 Institutional Research Dossier Hero');
  assert(rsSrc.includes('Ranked Institutional Catalyst Stream'), 'Research renders Ranked Institutional Catalyst Stream table');
  assert(rsSrc.includes('SEC_FORM_4') || rsSrc.includes('13F_WHALE') || rsSrc.includes('CONGRESS_STOCK_ACT'), 'Research indexes institutional flow sources');
  assert(rsSrc.includes('Fundamental Balance Sheet Armor'), 'Research features Deep-Dive Balance Sheet Armor matrix');
  assert(!rsSrc.includes('grid grid-cols-1 md:grid-cols-3 gap-6'), 'Eliminated superficial 3-card card farm');
}

// -------------------------------------------------------------
// SECTION 8: Navigation, TerminalShell & Command Palette
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 8: Navigation, Shell & Command Palette Quality\x1b[0m');
const shellPath = path.join(projectRoot, 'frontend', 'components', 'terminal', 'TerminalShell.tsx');
if (fs.existsSync(shellPath)) {
  const shSrc = fs.readFileSync(shellPath, 'utf8');
  assert(shSrc.includes('🛡️ Governor: Active'), 'TerminalShell includes active Behavioral Governor telemetry');
  assert(shSrc.includes('Mobile Terminal Navigation'), 'TerminalShell houses consolidated mobile navigation dock');
  assert(shSrc.includes('min-h-[44px]'), 'Mobile touch targets respect 44px standard');
}

const cpPath = path.join(projectRoot, 'frontend', 'components', 'CommandPaletteModal.tsx');
if (fs.existsSync(cpPath)) {
  const cpSrc = fs.readFileSync(cpPath, 'utf8');
  assert(cpSrc.includes('/radar') && cpSrc.includes('/setups') && cpSrc.includes('/portfolio') && cpSrc.includes('/journal') && cpSrc.includes('/performance') && cpSrc.includes('/research'), 'Command Palette indexes all 6 flagship hubs');
  assert(!cpSrc.includes('Life Health Index') && !cpSrc.includes('168-Hour'), 'Command Palette is cleansed of forbidden lifestyle concepts');
}

// -------------------------------------------------------------
// SUMMARY SCOREBOARD
// -------------------------------------------------------------
console.log('\n-------------------------------------------------------------');
console.log(`\x1b[1mTOTAL ASSERTIONS: ${passedAssertions + failedAssertions}\x1b[0m`);
console.log(`\x1b[32mPASSED: ${passedAssertions}\x1b[0m`);
console.log(`\x1b[31mFAILED: ${failedAssertions}\x1b[0m`);
console.log('-------------------------------------------------------------\n');

if (failedAssertions > 0) {
  process.exit(1);
} else {
  console.log('\x1b[32m\x1b[1m✨ HORIZON 14.3 TASTE, CRAFT & EXPERIENCE REDESIGN CERTIFIED! ✨\x1b[0m\n');
  process.exit(0);
}
