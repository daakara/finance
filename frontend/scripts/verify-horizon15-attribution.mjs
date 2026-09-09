// frontend/scripts/verify-horizon15-attribution.mjs
// Comprehensive Verification Suite for Horizon 15: Proof of Edge & Attribution
// Enforces INV-OI116-P, INV-OI117-P, INV-OI118-P and Radar Universe Multi-Factor Filtering

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

console.log('\x1b[1m\x1b[36m=== Horizon 15: Proof of Edge, Attribution & Radar Universe Verification ===\x1b[0m\n');

// -------------------------------------------------------------
// SECTION 1: Counterfactual Attribution Engine (INV-OI116-P)
// -------------------------------------------------------------
console.log('\x1b[1mSection 1: Counterfactual Attribution Math Determinism (INV-OI116-P)\x1b[0m');
const enginePath = path.join(projectRoot, 'frontend', 'lib', 'attribution', 'counterfactualEngine.ts');
assert(fs.existsSync(enginePath), 'counterfactualEngine.ts exists');

if (fs.existsSync(enginePath)) {
  const engineSrc = fs.readFileSync(enginePath, 'utf8');

  assert(engineSrc.includes('CANONICAL_GOVERNOR_LEDGER'), 'Exports canonical governor ledger');
  assert(engineSrc.includes('computeCounterfactualAttribution'), 'Exports computeCounterfactualAttribution function');
  assert(engineSrc.includes('discoverPersonalEdge'), 'Exports discoverPersonalEdge function');
  assert(engineSrc.includes('INV-OI116-P'), 'References INV-OI116-P Counterfactual Math Determinism');
  assert(engineSrc.includes('INV-OI117-P'), 'References INV-OI117-P Audit Ledger Completeness');
  assert(engineSrc.includes('INV-OI118-P'), 'References INV-OI118-P Personal Edge Statistical Significance');

  const ledgerMatches = [...engineSrc.matchAll(/id:\s*'([^']+)',[\s\S]*?ticker:\s*'([^']+)',[\s\S]*?setupArchetype:\s*'([^']+)',[\s\S]*?clampFactorPct:\s*(\d+),[\s\S]*?tradeOutcome:\s*'([^']+)',[\s\S]*?unclampedPnLDollar:\s*(-?\d+),[\s\S]*?governedPnLDollar:\s*(-?\d+),[\s\S]*?capitalPreservedDollar:\s*(\d+)/g)];
  
  assert(ledgerMatches.length >= 30, `Canonical ledger contains at least 30 audited trades (found ${ledgerMatches.length})`);

  let calculatedPreservedTotal = 0;
  let mathViolations = 0;

  ledgerMatches.forEach((m) => {
    const id = m[1];
    const ticker = m[2];
    const clampPct = parseInt(m[4], 10);
    const outcome = m[5];
    const unclampedPnL = parseInt(m[6], 10);
    const governedPnL = parseInt(m[7], 10);
    const preserved = parseInt(m[8], 10);

    if (clampPct > 0 && outcome === 'LOSS') {
      const expectedPreserved = Math.abs(unclampedPnL) - Math.abs(governedPnL);
      if (preserved !== expectedPreserved) {
        mathViolations++;
        console.error(`    Mismatch in ${id} (${ticker}): preserved=${preserved}, expected=${expectedPreserved}`);
      }
    }
    calculatedPreservedTotal += preserved;
  });

  assert(mathViolations === 0, 'Every individual clamped loss satisfies preserved == |unclampedLoss| - |governedLoss|');
  assert(calculatedPreservedTotal > 5000, `Cumulative capital preserved exceeds $5,000 (calculated: $${calculatedPreservedTotal})`);
}

// -------------------------------------------------------------
// SECTION 2: Invariant Ledger & Mathematical Certification
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 2: Invariant Ledger & Proof Rules (horizon15Invariants.ts)\x1b[0m');
const invPath = path.join(projectRoot, 'frontend', 'lib', 'simulation', 'horizon15Invariants.ts');
assert(fs.existsSync(invPath), 'horizon15Invariants.ts exists');

if (fs.existsSync(invPath)) {
  const invSrc = fs.readFileSync(invPath, 'utf8');
  assert(invSrc.includes('verifyCounterfactualMathDeterminism'), 'Exports verifyCounterfactualMathDeterminism');
  assert(invSrc.includes('verifyAuditLedgerCompleteness'), 'Exports verifyAuditLedgerCompleteness');
  assert(invSrc.includes('verifyPersonalEdgeSignificance'), 'Exports verifyPersonalEdgeSignificance');
  assert(invSrc.includes('auditHorizon15Master'), 'Exports auditHorizon15Master');
  assert(invSrc.includes('INV-OI116-P'), 'Ledger codifies INV-OI116-P');
  assert(invSrc.includes('INV-OI117-P'), 'Ledger codifies INV-OI117-P');
  assert(invSrc.includes('INV-OI118-P'), 'Ledger codifies INV-OI118-P');
}

const reexportPath = path.join(projectRoot, 'frontend', 'lib', 'invariants', 'horizon15Invariants.ts');
assert(fs.existsSync(reexportPath), 'invariants/horizon15Invariants.ts re-export exists');

// -------------------------------------------------------------
// SECTION 3: Radar Multi-Factor Universe & Dynamic Filtering
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 3: Radar Dynamic Universe & Multi-Factor Filtering (/radar)\x1b[0m');
const radarPath = path.join(projectRoot, 'frontend', 'app', 'radar', 'page.tsx');
assert(fs.existsSync(radarPath), 'radar/page.tsx exists');

if (fs.existsSync(radarPath)) {
  const radarSrc = fs.readFileSync(radarPath, 'utf8');

  assert(radarSrc.includes('fetchScreenerGems'), 'Radar fetches live screener candidates via fetchScreenerGems API');
  assert(!radarSrc.includes('generateRadarUniverse'), 'Eliminated hardcoded generateRadarUniverse() generator');
  assert(radarSrc.includes('TerminalShell'), 'Radar renders inside TerminalShell');
  assert(radarSrc.includes('activeHub="radar"'), 'Radar sets activeHub="radar"');

  assert(radarSrc.includes("c.category === 'VCP'") || radarSrc.includes("'VCP'"), 'Radar classifies VCP confluence setups');
  assert(radarSrc.includes("c.category === 'SMART_MONEY'") || radarSrc.includes("'SMART_MONEY'"), 'Radar classifies Smart Money institutional flows');
  assert(radarSrc.includes("c.category === 'VALUE'") || radarSrc.includes("'VALUE'"), 'Radar classifies Value & GARP compounders');

  assert(radarSrc.includes("setActiveFilter('ALL')"), 'Radar has All Confluences filter tab');
  assert(radarSrc.includes("setActiveFilter('VCP')"), 'Radar has VCP filter tab');
  assert(radarSrc.includes("setActiveFilter('SMART_MONEY')"), 'Radar has Smart Money filter tab');
  assert(radarSrc.includes("setActiveFilter('VALUE')"), 'Radar has Value & GARP filter tab');

  assert(radarSrc.includes('searchQuery') && radarSrc.includes('setSearchQuery'), 'Radar includes interactive search query state');
  assert(radarSrc.includes('sortBy') && radarSrc.includes('setSortBy'), 'Radar includes sorting controls (Score, RS, Price)');
  assert(!radarSrc.includes('CANONICAL_RADAR_ASSETS'), 'Eliminated old 4-item static mock array');

  assert(radarSrc.includes('counts.ALL'), 'Radar tab displays dynamic All count badge');
  assert(radarSrc.includes('counts.VCP'), 'Radar tab displays dynamic VCP count badge');
  assert(radarSrc.includes('counts.SMART_MONEY'), 'Radar tab displays dynamic Smart Money count badge');
  assert(radarSrc.includes('counts.VALUE'), 'Radar tab displays dynamic Value count badge');
}

// -------------------------------------------------------------
// SECTION 4: Performance Hub Expansion (/performance)
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 4: Performance Hub Expansion (/performance)\x1b[0m');
const perfPath = path.join(projectRoot, 'frontend', 'app', 'performance', 'page.tsx');
assert(fs.existsSync(perfPath), 'performance/page.tsx exists');

if (fs.existsSync(perfPath)) {
  const perfSrc = fs.readFileSync(perfPath, 'utf8');

  assert(perfSrc.includes('TerminalShell'), 'Performance page renders inside TerminalShell');
  assert(perfSrc.includes('activeHub="performance"'), 'Performance page sets activeHub="performance"');
  assert(perfSrc.includes('computeCounterfactualAttribution'), 'Performance page imports computeCounterfactualAttribution');
  assert(perfSrc.includes('discoverPersonalEdge'), 'Performance page imports discoverPersonalEdge');

  assert(perfSrc.includes('Capital Preserved'), 'Displays Capital Preserved metric');
  assert(perfSrc.includes('summary.capitalPreservedTotal'), 'Derives Preserved Capital dynamically from summary');
  assert(perfSrc.includes('Max Drawdown'), 'Displays Max Drawdown comparison');
  assert(perfSrc.includes('Sharpe Ratio'), 'Displays Sharpe Ratio comparison');
  assert(perfSrc.includes('Profit Factor'), 'Displays Profit Factor comparison');
  assert(perfSrc.includes('Risk of Ruin'), 'Displays Risk of Ruin reduction');

  assert(perfSrc.includes("'OVERVIEW'"), 'Includes Attribution Overview tab');
  assert(perfSrc.includes("'EDGE'"), 'Includes Personal Edge Discovery tab');
  assert(perfSrc.includes("'LEDGER'"), 'Includes Governor Audit Ledger tab');

  assert(perfSrc.includes('Drawdown Defense') && perfSrc.includes('summary.preservedByCategory.drawdownDefense'), 'Overview breaks down Drawdown Defense savings');
  assert(perfSrc.includes('Execution Window') && perfSrc.includes('summary.preservedByCategory.executionWindow'), 'Overview breaks down Execution Window savings');
  assert(perfSrc.includes('Capital Floor') && perfSrc.includes('summary.preservedByCategory.capitalFloor'), 'Overview breaks down Capital Floor savings');
  assert(perfSrc.includes('Counterfactual Equity Trajectory'), 'Displays Counterfactual Equity Trajectory');

  assert(perfSrc.includes('Highest Expectancy Setup'), 'Edge Discovery highlights Highest Expectancy Setup');
  assert(perfSrc.includes('Primary Capital Leak'), 'Edge Discovery highlights Primary Capital Leak / Tilt Warning');
  assert(perfSrc.includes('Setup Archetype Performance'), 'Edge Discovery audits Setup Archetypes');
  assert(perfSrc.includes('Performance by Execution Window'), 'Edge Discovery breaks down Execution Windows');
  assert(perfSrc.includes('Performance by Market Regime'), 'Edge Discovery breaks down Market Regimes');

  assert(perfSrc.includes('<table') && perfSrc.includes('filteredLedger.map'), 'Audit Ledger renders detailed inspection table');
  assert(perfSrc.includes('Unclamped Risk') && perfSrc.includes('Governed Risk'), 'Ledger compares unclamped vs governed risk');
  assert(perfSrc.includes('Clamp Factor'), 'Ledger displays exact clamp percentage');
  assert(perfSrc.includes('Reason / Rationale'), 'Ledger displays clean-room reason detail');
  assert(perfSrc.includes('Capital Preserved'), 'Ledger displays dollars preserved per trade');
}

// -------------------------------------------------------------
// SECTION 5: Dynamic Runtime Invariant Audit
// -------------------------------------------------------------
console.log('\n\x1b[1mSection 5: Dynamic Runtime Invariant Audit\x1b[0m');

async function runDynamicAudit() {
  const engine = await import('../../frontend/lib/attribution/counterfactualEngine.ts');
  const invariants = await import('../../frontend/lib/simulation/horizon15Invariants.ts');

  const summary = engine.computeCounterfactualAttribution(engine.CANONICAL_GOVERNOR_LEDGER);
  const edge = engine.discoverPersonalEdge(engine.CANONICAL_GOVERNOR_LEDGER);

  assert(summary.currentGovernedEquity > summary.currentUnclampedEquity, `Governed equity ($${summary.currentGovernedEquity}) exceeds unclamped equity ($${summary.currentUnclampedEquity})`);
  assert(summary.governedMaxDrawdown > summary.unclampedMaxDrawdown, `Governed drawdown (${summary.governedMaxDrawdown}%) is significantly milder than unclamped (${summary.unclampedMaxDrawdown}%)`);
  assert(summary.governedSharpe > summary.unclampedSharpe, `Governed Sharpe (${summary.governedSharpe}) exceeds unclamped (${summary.unclampedSharpe})`);
  assert(summary.governedProfitFactor > summary.unclampedProfitFactor, `Governed Profit Factor (${summary.governedProfitFactor}) exceeds unclamped (${summary.unclampedProfitFactor})`);

  const vcpEdge = edge.archetypes.find((a) => a.archetype === 'VCP_BREAKOUT');
  assert(vcpEdge && vcpEdge.edgeTier === 'STRONG_EDGE' && vcpEdge.sampleCount >= 5, 'Minervini VCP classified as STRONG_EDGE with >= 5 samples');

  const dipBuy = edge.archetypes.find((a) => a.archetype === 'DIP_BUY');
  assert(dipBuy && dipBuy.edgeTier === 'NEGATIVE_TILT' && dipBuy.profitFactor < 1.0, 'Late session Dip Buy classified as NEGATIVE_TILT with PF < 1.0');

  const masterAudit = invariants.auditHorizon15Master({
    ledger: engine.CANONICAL_GOVERNOR_LEDGER,
    summary,
    edgeReport: edge,
  });

  assert(masterAudit.certified === true, 'Horizon 15 Master Audit returned certified === true');
  assert(masterAudit.totalViolations === 0, `Total invariant violations is exactly 0 (got ${masterAudit.totalViolations})`);

  console.log('\n-------------------------------------------------------------');
  console.log(`Total Assertions: ${passedAssertions + failedAssertions}`);
  console.log(`\x1b[32mPassed: ${passedAssertions}\x1b[0m`);
  console.log(`\x1b[31mFailed: ${failedAssertions}\x1b[0m`);

  if (failedAssertions === 0) {
    console.log('\x1b[1m\x1b[32m\n✔ HORIZON 15 (Proof of Edge & Attribution) CERTIFIED PASSED.\x1b[0m\n');
  } else {
    console.error('\x1b[1m\x1b[31m\n✖ HORIZON 15 VERIFICATION FAILED with ' + failedAssertions + ' error(s).\x1b[0m\n');
    process.exitCode = 1;
  }
}

runDynamicAudit().catch((err) => {
  console.error('Audit execution error:', err);
  process.exitCode = 1;
});
