import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// 1. Direct imports of REAL production modules
import {
  filterEligibleLiveTrades,
  computeRealizedMetrics,
  computeChronologicalTrajectory,
  groupSetupsByPattern,
} from '../lib/performanceMetrics';
import {
  formatOrderPlanString,
  copyOrderPlanToClipboard,
} from '../lib/orderClipboard';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '../..');

console.log('Running True Behavioral Unit & Integration Regression Suite for A1a...\n');

let totalTests = 0;
let passedTests = 0;
let failedTests = 0;

async function runTest(name: string, fn: () => Promise<void> | void) {
  totalTests++;
  try {
    await fn();
    console.log(`  [PASS] ${name}`);
    passedTests++;
  } catch (err: any) {
    failedTests++;
    console.error(`  [FAIL] ${name}`);
    console.error(`     Error: ${err.message}\n${err.stack}`);
  }
}

async function main() {
  // -------------------------------------------------------------------------
  // SUITE 1: Copy Plan Isolation & Zero Side-Effect Clipboard Operations
  // -------------------------------------------------------------------------
  console.log('--- Suite 1: Order Plan Formatting & Clipboard Isolation ---');

  await runTest('formatOrderPlanString formats order deterministically with Target 1', () => {
    const formatted = formatOrderPlanString({
      recommendedShares: 100,
      ticker: 'AAPL',
      entryPivot: 225.5,
      stopLoss: 215.0,
      target1: 245.0,
    });
    assert.equal(formatted, 'BUY 100 AAPL LMT $225.50 | STP $215.00 | TGT $245.00');
  });

  await runTest('formatOrderPlanString formats order cleanly when Target 1 is missing or 0', () => {
    const formatted = formatOrderPlanString({
      recommendedShares: 50,
      ticker: 'TSLA',
      entryPivot: 210.0,
      stopLoss: 195.0,
      target1: null,
    });
    assert.equal(formatted, 'BUY 50 TSLA LMT $210.00 | STP $195.00 | TGT --');
  });

  await runTest('copyOrderPlanToClipboard invokes clipboard.writeText with exact string and returns success', async () => {
    let written = null;
    let writeCalls = 0;
    const mockClipboard = {
      writeText: async (text: string) => {
        writeCalls++;
        written = text;
      },
    };

    const res = await copyOrderPlanToClipboard('BUY 10 NVDA LMT $120.00 | STP $110.00 | TGT $140.00', mockClipboard);
    assert.equal(res.success, true);
    assert.equal(writeCalls, 1);
    assert.equal(written, 'BUY 10 NVDA LMT $120.00 | STP $110.00 | TGT $140.00');
  });

  await runTest('copyOrderPlanToClipboard catches permission errors gracefully without throwing', async () => {
    const rejectingClipboard = {
      writeText: async () => {
        throw new Error('NotAllowedError: Document is not focused.');
      },
    };

    const res = await copyOrderPlanToClipboard('BUY 10 NVDA LMT $120.00', rejectingClipboard);
    assert.equal(res.success, false);
    assert.match(res.error || '', /Document is not focused/);
  });

  await runTest('copyOrderPlanToClipboard handles missing clipboard API gracefully', async () => {
    const res = await copyOrderPlanToClipboard('BUY 10 NVDA LMT $120.00', undefined);
    assert.equal(res.success, false);
    assert.match(res.error || '', /Clipboard API unavailable/);
  });

  await runTest('Source AST: setups/page.tsx has zero saveJournalTrade import or call sites', () => {
    const setupsSrc = fs.readFileSync(path.join(rootDir, 'frontend/app/setups/page.tsx'), 'utf-8');
    assert.equal(setupsSrc.includes('saveJournalTrade'), false, 'saveJournalTrade must not be present in setups/page.tsx');
    assert.equal(setupsSrc.includes('COPY TRADE PLAN'), true, 'Must use COPY TRADE PLAN button text');
    assert.equal(setupsSrc.includes('copyOrderPlanToClipboard'), true, 'Must use copyOrderPlanToClipboard helper');
  });

  // -------------------------------------------------------------------------
  // SUITE 2: Completed Trade Eligibility Invariants (Production filterEligibleLiveTrades)
  // -------------------------------------------------------------------------
  console.log('\n--- Suite 2: Strict Completed Trade Eligibility Contract ---');

  await runTest('filterEligibleLiveTrades excludes OPEN trades even when exitPrice is positive (BUG 1)', () => {
    const input: any[] = [
      {
        id: '1',
        ticker: 'NVDA',
        status: 'OPEN', // OPEN trade with positive exit price (e.g. from premature plan logging)
        entryPrice: 120.0,
        exitPrice: 135.0,
        shares: 50,
        pnl: 750,
      },
    ];
    const result = filterEligibleLiveTrades(input);
    assert.equal(result.length, 0, 'OPEN trades with positive exitPrice must be strictly excluded');
  });

  await runTest('filterEligibleLiveTrades excludes records with missing or non-finite realized outcomes', () => {
    const input: any[] = [
      {
        id: '1',
        ticker: 'AAPL',
        status: 'CLOSED',
        entryPrice: 200.0,
        exitPrice: 210.0,
        shares: 20,
        pnl: null, // missing P&L
        pnlRaw: null,
      },
      {
        id: '2',
        ticker: 'MSFT',
        status: 'CLOSED',
        entryPrice: 400.0,
        exitPrice: 420.0,
        shares: 10,
        pnl: 'N/A', // non-numeric string
      },
    ];
    const result = filterEligibleLiveTrades(input);
    assert.equal(result.length, 0, 'Records without explicit, valid realized outcome data must not be inferred');
  });

  await runTest('filterEligibleLiveTrades preserves valid zero outcomes (pnl === 0) as SCRATCH trades', () => {
    const input: any[] = [
      {
        id: '1',
        ticker: 'GOOGL',
        status: 'CLOSED',
        entryPrice: 175.0,
        exitPrice: 175.0,
        shares: 100,
        pnl: 0,
        pnlRaw: 0,
      },
    ];
    const result = filterEligibleLiveTrades(input);
    assert.equal(result.length, 1);
    assert.equal(result[0].outcome, 'SCRATCH');
    assert.equal(result[0].pnl, 0);
  });

  await runTest('filterEligibleLiveTrades excludes non-positive price or shares levels', () => {
    const input: any[] = [
      { id: '1', status: 'CLOSED', entryPrice: 0, exitPrice: 10, shares: 10, pnl: 10 },
      { id: '2', status: 'CLOSED', entryPrice: 10, exitPrice: 0, shares: 10, pnl: 10 },
      { id: '3', status: 'CLOSED', entryPrice: 10, exitPrice: 10, shares: 0, pnl: 10 },
      { id: '4', status: 'CLOSED', entryPrice: -5, exitPrice: 10, shares: 10, pnl: 10 },
    ];
    const result = filterEligibleLiveTrades(input);
    assert.equal(result.length, 0, 'Zero or negative entry, exit, or shares must be excluded');
  });

  await runTest('filterEligibleLiveTrades does not invent fallback ticker ASSET or Recent dates', () => {
    const input: any[] = [
      {
        id: '1',
        ticker: '',
        symbol: '',
        status: 'CLOSED',
        entryPrice: 50.0,
        exitPrice: 55.0,
        shares: 10,
        pnl: 50,
        date: '',
        entryDate: '',
      },
    ];
    const result = filterEligibleLiveTrades(input);
    assert.equal(result.length, 1);
    assert.equal(result[0].ticker, null, 'Must preserve null rather than injecting ASSET');
    assert.equal(result[0].entryDate, null, 'Must preserve null rather than injecting Recent');
  });

  // -------------------------------------------------------------------------
  // SUITE 3: Empirical Realized Metrics & Average R Denominator (Production computeRealizedMetrics)
  // -------------------------------------------------------------------------
  console.log('\n--- Suite 3: Empirical Realized Metric Integrity ---');

  await runTest('computeRealizedMetrics excludes trades missing R from average-R denominator', () => {
    const trades: any[] = [
      {
        id: '1',
        ticker: 'AMD',
        setupName: 'VCP',
        entryPrice: 150,
        exitPrice: 165,
        shares: 10,
        pnl: 150,
        rAchieved: 2.5, // Trade with R
        followedRules: true,
        entryDate: '2026-08-01',
        exitDate: '2026-08-05',
        outcome: 'WIN',
      },
      {
        id: '2',
        ticker: 'INTC',
        setupName: 'Breakout',
        entryPrice: 30,
        exitPrice: 33,
        shares: 50,
        pnl: 150,
        rAchieved: null, // Missing R
        followedRules: null,
        entryDate: '2026-08-02',
        exitDate: '2026-08-06',
        outcome: 'WIN',
      },
    ];

    const metrics = computeRealizedMetrics(trades, 2);
    // Average R should be 2.50 (2.5 / 1), NOT 1.25 (2.5 / 2)!
    assert.equal(metrics.avgR, '2.50', 'Average R must exclude missing R trades from the denominator');
    assert.equal(metrics.eligibleRTradesCount, 1);
    assert.equal(metrics.winRatePct, '100.0');
    assert.equal(metrics.totalRealizedPnL, 300);
  });

  await runTest('groupSetupsByPattern groups by exact recorded name without keyword guessing', () => {
    const trades: any[] = [
      { id: '1', setupName: 'Minervini VCP Breakout', pnl: 200, outcome: 'WIN' },
      { id: '2', setupName: 'Minervini VCP Breakout', pnl: -50, outcome: 'LOSS' },
      { id: '3', setupName: 'Pocket Pivot', pnl: 100, outcome: 'WIN' },
      { id: '4', setupName: null, pnl: 50, outcome: 'WIN' },
    ];
    const groups = groupSetupsByPattern(trades);
    assert.equal(groups.length, 3);
    const vcp = groups.find((g) => g.name === 'Minervini VCP Breakout');
    assert.ok(vcp);
    assert.equal(vcp.count, 2);
    assert.equal(vcp.winRatePct, '50.0');
    assert.equal(vcp.totalPnL, 150);

    const unspecified = groups.find((g) => g.name === 'Unspecified Setup');
    assert.ok(unspecified);
    assert.equal(unspecified.count, 1);
  });

  // -------------------------------------------------------------------------
  // SUITE 4: Chronological Trajectory Ordering & Missing Timestamp Handling
  // -------------------------------------------------------------------------
  console.log('\n--- Suite 4: Chronological Trajectory & Closing Timestamp Integrity ---');

  await runTest('computeChronologicalTrajectory sorts strictly ascending by closing timestamp (exitDate)', () => {
    const trades: any[] = [
      {
        id: 'trade-late-entry-early-exit',
        ticker: 'META',
        entryDate: '2026-08-10',
        exitDate: '2026-08-12', // Closes FIRST
        pnl: 300,
      },
      {
        id: 'trade-early-entry-late-exit',
        ticker: 'AMZN',
        entryDate: '2026-08-01', // Entered earlier, but closes LATER
        exitDate: '2026-08-20', // Closes SECOND
        pnl: -100,
      },
    ];

    const res = computeChronologicalTrajectory(trades);
    assert.equal(res.isAvailable, true);
    assert.equal(res.points.length, 2);

    // Trade 1 in trajectory must be META (exitDate 2026-08-12)
    assert.equal(res.points[0].id, 'trade-late-entry-early-exit');
    assert.equal(res.points[0].exitDate, '2026-08-12');
    assert.equal(res.points[0].cumulativePnL, 300);

    // Trade 2 in trajectory must be AMZN (exitDate 2026-08-20)
    assert.equal(res.points[1].id, 'trade-early-entry-late-exit');
    assert.equal(res.points[1].exitDate, '2026-08-20');
    assert.equal(res.points[1].cumulativePnL, 200);
  });

  await runTest('computeChronologicalTrajectory returns unavailable if any trade lacks exitDate', () => {
    const trades: any[] = [
      { id: '1', ticker: 'MSFT', entryDate: '2026-08-01', exitDate: '2026-08-05', pnl: 100 },
      { id: '2', ticker: 'GOOG', entryDate: '2026-08-02', exitDate: null, pnl: 50 }, // Missing exitDate!
    ];
    const res = computeChronologicalTrajectory(trades);
    assert.equal(res.isAvailable, false);
    assert.match(res.unavailableReason || '', /lack a verified closing timestamp/);
    assert.equal(res.points.length, 0);
  });

  // -------------------------------------------------------------------------
  // SUITE 5: Zero Synthetic Benchmark Dependency in Production Performance
  // -------------------------------------------------------------------------
  console.log('\n--- Suite 5: Production Performance Route Purity ---');

  await runTest('Performance page has zero CANONICAL_GOVERNOR_LEDGER or synthetic benchmark imports', () => {
    const perfSrc = fs.readFileSync(path.join(rootDir, 'frontend/app/performance/page.tsx'), 'utf-8');
    assert.equal(
      perfSrc.includes('CANONICAL_GOVERNOR_LEDGER'),
      false,
      'CANONICAL_GOVERNOR_LEDGER must not be imported in performance/page.tsx'
    );
    assert.equal(
      perfSrc.includes('computeCounterfactualAttribution'),
      false,
      'computeCounterfactualAttribution must not be imported in performance/page.tsx'
    );
    assert.equal(
      perfSrc.includes('discoverPersonalEdge'),
      false,
      'discoverPersonalEdge must not be imported in performance/page.tsx'
    );
    assert.equal(
      perfSrc.includes('Audited Benchmark (31)'),
      false,
      'Benchmark mode button must not exist in production performance page'
    );
    assert.equal(
      perfSrc.includes('filterEligibleLiveTrades'),
      true,
      'Must import and use filterEligibleLiveTrades'
    );
    assert.equal(
      perfSrc.includes('computeRealizedMetrics'),
      true,
      'Must import and use computeRealizedMetrics'
    );
  });

  // -------------------------------------------------------------------------
  // SUITE 6: Portfolio Storage Wording & Claims
  // -------------------------------------------------------------------------
  console.log('\n--- Suite 6: Truthful Storage Claims in UI ---');

  await runTest('portfolio/page.tsx uses AUTHORITATIVE API PERSISTENCE and has zero private-storage claims', () => {
    const portSrc = fs.readFileSync(path.join(rootDir, 'frontend/app/portfolio/page.tsx'), 'utf-8');
    assert.equal(portSrc.includes('ZERO-LOGIN PRIVATE STORAGE'), false);
    assert.equal(portSrc.includes('100% private to your browser'), false);
    assert.equal(portSrc.includes('AUTHORITATIVE API PERSISTENCE'), true);
  });

  await runTest('OnboardingTourModal.tsx has zero client-only encrypted private storage claims', () => {
    const tourSrc = fs.readFileSync(path.join(rootDir, 'frontend/components/OnboardingTourModal.tsx'), 'utf-8');
    assert.equal(tourSrc.includes('100% private to your browser'), false);
    assert.equal(tourSrc.includes('encrypted on your device'), false);
  });

  // -------------------------------------------------------------------------
  // SUMMARY
  // -------------------------------------------------------------------------
  console.log('\n========================================');
  console.log(`Results: ${passedTests}/${totalTests} Passed (${failedTests} Failed)`);
  console.log('========================================');

  if (failedTests > 0) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('Fatal test error:', err);
  process.exit(1);
});
