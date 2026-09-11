import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// 1. Direct imports of REAL production modules
import {
  RequestProgress,
  DataFreshness,
  TradeLifecycleState,
  SetupEligibility,
  validateLifecycleTransition,
  validateFillParams,
  validateExitParams,
  calculateRealizedPnL,
  calculateRealizedR,
  generateIdempotencyKey,
} from '../lib/tradeLifecycle';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '../..');

console.log('Running Behavioral Unit & Integration Regression Suite for A1b Lifecycle...\n');

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
  // SUITE 1: Four Orthogonal Status Dimensions Non-Collapsing
  // -------------------------------------------------------------------------
  console.log('--- Suite 1: Four Orthogonal Status Dimensions ---');

  await runTest('Orthogonal dimensions have distinct non-overlapping value sets', () => {
    const reqProgressValues = Object.values(RequestProgress);
    const dataFreshnessValues = Object.values(DataFreshness);
    const lifecycleValues = Object.values(TradeLifecycleState);
    const setupEligValues = Object.values(SetupEligibility);

    // Verify TradeLifecycleState contains exactly PLANNED, OPEN, CLOSED
    assert.deepEqual(lifecycleValues.sort(), ['CLOSED', 'OPEN', 'PLANNED'].sort());

    // Verify DataFreshness does not conflate with Lifecycle
    assert.ok(dataFreshnessValues.includes(DataFreshness.STALE));
    assert.ok(!lifecycleValues.includes('STALE' as any));

    // Verify RequestProgress does not conflate with Lifecycle
    assert.ok(reqProgressValues.includes(RequestProgress.SUBMITTING));
    assert.ok(!lifecycleValues.includes('SUBMITTING' as any));

    // Verify SetupEligibility does not conflate with Lifecycle
    assert.ok(setupEligValues.includes(SetupEligibility.DISQUALIFIED));
    assert.ok(!lifecycleValues.includes('DISQUALIFIED' as any));
  });

  // -------------------------------------------------------------------------
  // SUITE 2: Strict Finite State Machine Transition Validation
  // -------------------------------------------------------------------------
  console.log('\n--- Suite 2: Lifecycle State Transition Validation ---');

  await runTest('Allows legal PLANNED -> OPEN transition', () => {
    const res = validateLifecycleTransition(TradeLifecycleState.PLANNED, TradeLifecycleState.OPEN);
    assert.equal(res.valid, true);
    assert.equal(res.error, undefined);
  });

  await runTest('Allows legal OPEN -> CLOSED transition', () => {
    const res = validateLifecycleTransition(TradeLifecycleState.OPEN, TradeLifecycleState.CLOSED);
    assert.equal(res.valid, true);
    assert.equal(res.error, undefined);
  });

  await runTest('Rejects illegal direct PLANNED -> CLOSED transition without broker fill', () => {
    const res = validateLifecycleTransition(TradeLifecycleState.PLANNED, TradeLifecycleState.CLOSED);
    assert.equal(res.valid, false);
    assert.ok(res.error?.includes('Illegal trade lifecycle transition'));
  });

  await runTest('Rejects reopening CLOSED trades (terminal state)', () => {
    const res = validateLifecycleTransition(TradeLifecycleState.CLOSED, TradeLifecycleState.OPEN);
    assert.equal(res.valid, false);
    assert.ok(res.error?.includes('terminal'));
  });

  await runTest('Rejects reversing OPEN back to PLANNED', () => {
    const res = validateLifecycleTransition(TradeLifecycleState.OPEN, TradeLifecycleState.PLANNED);
    assert.equal(res.valid, false);
  });

  await runTest('Rejects invalid lifecycle states', () => {
    const res = validateLifecycleTransition('UNKNOWN_STATE' as any, TradeLifecycleState.OPEN);
    assert.equal(res.valid, false);
    assert.ok(res.error?.includes('Invalid'));
  });

  // -------------------------------------------------------------------------
  // SUITE 3: Broker Execution Fill Validation
  // -------------------------------------------------------------------------
  console.log('\n--- Suite 3: Broker Fill Parameter Validation ---');

  await runTest('Accepts valid broker fill parameters', () => {
    const res = validateFillParams({
      symbol: 'NVDA',
      entryPrice: 128.5,
      shares: 100,
      stopLoss: 119.0,
      target1: 155.0,
      confidence: 85,
    });
    assert.equal(res.valid, true);
  });

  await runTest('Rejects zero or negative entry price', () => {
    const resZero = validateFillParams({ symbol: 'NVDA', entryPrice: 0, shares: 100 });
    assert.equal(resZero.valid, false);
    assert.ok(resZero.error?.includes('Entry price'));

    const resNeg = validateFillParams({ symbol: 'NVDA', entryPrice: -10, shares: 100 });
    assert.equal(resNeg.valid, false);
  });

  await runTest('Rejects zero or negative shares count', () => {
    const resZero = validateFillParams({ symbol: 'NVDA', entryPrice: 100, shares: 0 });
    assert.equal(resZero.valid, false);
    assert.ok(resZero.error?.includes('Share quantity'));

    const resNeg = validateFillParams({ symbol: 'NVDA', entryPrice: 100, shares: -5 });
    assert.equal(resNeg.valid, false);
  });

  await runTest('Rejects invalid confidence scores out of [0, 100]', () => {
    const res = validateFillParams({ symbol: 'NVDA', entryPrice: 100, shares: 10, confidence: 150 });
    assert.equal(res.valid, false);
    assert.ok(res.error?.includes('Confidence'));
  });

  await runTest('Rejects empty or missing ticker symbol', () => {
    const res = validateFillParams({ symbol: '   ', entryPrice: 100, shares: 10 });
    assert.equal(res.valid, false);
    assert.ok(res.error?.includes('Ticker symbol'));
  });

  // -------------------------------------------------------------------------
  // SUITE 4: Trade Exit & Scale-Out Accounting Validation
  // -------------------------------------------------------------------------
  console.log('\n--- Suite 4: Trade Exit & Scale-Out Accounting Validation ---');

  await runTest('Accepts valid partial exit within open share limits', () => {
    const res = validateExitParams(100, 40, 135.0);
    assert.equal(res.valid, true);
  });

  await runTest('Accepts valid full exit of 100% remaining shares', () => {
    const res = validateExitParams(100, 100, 135.0);
    assert.equal(res.valid, true);
  });

  await runTest('Rejects exit shares exceeding remaining open shares', () => {
    const res = validateExitParams(100, 105, 135.0);
    assert.equal(res.valid, false);
    assert.ok(res.error?.includes('exceeds remaining open shares'));
  });

  await runTest('Rejects zero or negative exit shares', () => {
    const res = validateExitParams(100, 0, 135.0);
    assert.equal(res.valid, false);
    assert.ok(res.error?.includes('Exit shares count'));
  });

  await runTest('Rejects zero or negative exit price', () => {
    const res = validateExitParams(100, 50, 0);
    assert.equal(res.valid, false);
    assert.ok(res.error?.includes('Exit price'));
  });

  // -------------------------------------------------------------------------
  // SUITE 5: Realized Accounting Math & Missing Evidence Preservation
  // -------------------------------------------------------------------------
  console.log('\n--- Suite 5: Realized Accounting Math & Missing Evidence ---');

  await runTest('Computes realized PnL correctly for profit and loss legs', () => {
    const profit = calculateRealizedPnL(100.0, 125.0, 40);
    assert.equal(profit, 1000.0);

    const loss = calculateRealizedPnL(100.0, 92.0, 40);
    assert.equal(loss, -320.0);
  });

  await runTest('Computes realized R-multiple accurately', () => {
    // Risk: 100 - 90 = 10. Exit at 120 -> gain 20 -> 2.0R
    const rProf = calculateRealizedR(100.0, 120.0, 90.0);
    assert.equal(rProf, 2.0);

    // Risk: 100 - 90 = 10. Exit at 85 -> loss 15 -> -1.5R
    const rLoss = calculateRealizedR(100.0, 85.0, 90.0);
    assert.equal(rLoss, -1.5);
  });

  await runTest('Preserves missing evidence: returns null R when stop loss is absent or invalid', () => {
    const rNull = calculateRealizedR(100.0, 120.0, undefined);
    assert.equal(rNull, null, 'Must return null without fabricating default stop');

    const rSame = calculateRealizedR(100.0, 120.0, 100.0);
    assert.equal(rSame, null, 'Must return null when entry equals stop');
  });

  await runTest('Generates distinct client-side idempotency keys', () => {
    const key1 = generateIdempotencyKey('fill', 'AAPL', 'testuser');
    const key2 = generateIdempotencyKey('fill', 'AAPL', 'testuser');
    assert.notEqual(key1, key2);
    assert.ok(key1.includes('AAPL'));
    assert.ok(key1.includes('fill'));
    assert.ok(key1.includes('testuser'));
  });

  // -------------------------------------------------------------------------
  // SUITE 6: Frontend Route & AST Invariant Checks
  // -------------------------------------------------------------------------
  console.log('\n--- Suite 6: Frontend Route AST & Invariant Checks ---');

  await runTest('Setups page has ZERO saveJournalTrade calls (clipboard isolation intact)', () => {
    const setupsSrc = fs.readFileSync(path.join(rootDir, 'frontend/app/setups/page.tsx'), 'utf-8');
    assert.equal(
      setupsSrc.includes('saveJournalTrade'),
      false,
      'setups/page.tsx must NOT call or import saveJournalTrade'
    );
    assert.ok(
      setupsSrc.includes('recordBrokerFill'),
      'setups/page.tsx must offer recordBrokerFill for explicit actual execution'
    );
    assert.ok(
      setupsSrc.includes('COPY TRADE PLAN'),
      'setups/page.tsx must maintain COPY TRADE PLAN pure clipboard button'
    );
  });

  await runTest('Portfolio page imports and uses exit recording API and validators', () => {
    const portSrc = fs.readFileSync(path.join(rootDir, 'frontend/app/portfolio/page.tsx'), 'utf-8');
    assert.ok(
      portSrc.includes('recordTradeExit'),
      'portfolio/page.tsx must import recordTradeExit'
    );
    assert.ok(
      portSrc.includes('recordTradeClose'),
      'portfolio/page.tsx must import recordTradeClose'
    );
    assert.ok(
      portSrc.includes('validateExitParams'),
      'portfolio/page.tsx must import validateExitParams'
    );
    assert.ok(
      portSrc.includes('Record Exit'),
      'portfolio/page.tsx must render Record Exit button'
    );
    assert.ok(
      portSrc.includes('Record Trade Exit / Scale-Out'),
      'portfolio/page.tsx must render exit modal header'
    );
  });

  await runTest('Journal page renders Lifecycle State column and displays OPEN vs CLOSED', () => {
    const journalSrc = fs.readFileSync(path.join(rootDir, 'frontend/app/journal/page.tsx'), 'utf-8');
    assert.ok(
      journalSrc.includes('Lifecycle State'),
      'journal/page.tsx must have Lifecycle State column'
    );
    assert.ok(
      journalSrc.includes('OPEN HOLDING'),
      'journal/page.tsx must have OPEN HOLDING badge'
    );
    assert.ok(
      journalSrc.includes('executionRole'),
      'journal/page.tsx must support executionRole'
    );
    assert.ok(
      journalSrc.includes('remainingShares'),
      'journal/page.tsx must display remainingShares'
    );
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
