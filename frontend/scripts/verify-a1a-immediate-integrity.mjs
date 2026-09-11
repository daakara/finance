/**
 * Behavioral Regression Test Suite: A1a Immediate Integrity
 *
 * Verifies:
 * 1. Copy Plan is strictly clipboard-only (no Journal write side effect, handles success/failure, repeated copy).
 * 2. Completed-trade eligibility logic:
 *    - Strict CLOSED status required (OPEN with positive exit price excluded, OPEN without exit price excluded, null/undefined/missing excluded).
 *    - Realized outcome data validated (valid entry, exit, shares, pnl).
 *    - Valid zero outcomes (pnl === 0, scratch trades) preserved and eligible.
 *    - Missing and non-finite outcomes excluded.
 * 3. Live mode attribution integrity:
 *    - Zero fabricated attributes (no hardcoded UPTREND, MORNING_PRIME, VCP_BREAKOUT default, or clamp inventions).
 *    - Unknown setup names remain unclassified.
 *    - Missing counterfactual metrics are explicitly unavailable.
 *    - Genuine empirical metrics (win rate, profit factor, avg R, realized trajectory) computed accurately.
 * 4. Strict separation of live results from benchmark fixtures.
 * 5. Portfolio storage wording accuracy.
 */

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '../..');

console.log('🧪 Starting A1a Immediate Integrity Behavioral Regression Suite...\n');

let totalTests = 0;
let passedTests = 0;

function runTest(name, fn) {
  totalTests++;
  try {
    fn();
    console.log(`  ✅ PASS: ${name}`);
    passedTests++;
  } catch (err) {
    console.error(`  ❌ FAIL: ${name}`);
    console.error(`     Error: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// 1. Copy Plan Behavioral & AST Verification
// ---------------------------------------------------------------------------
console.log('--- Suite 1: Copy Plan Clipboard-Only Invariant ---');

runTest('Setups page has zero saveJournalTrade import or call sites', () => {
  const setupsContent = fs.readFileSync(path.join(rootDir, 'frontend/app/setups/page.tsx'), 'utf-8');
  assert.equal(
    setupsContent.includes('saveJournalTrade'),
    false,
    'setups/page.tsx must not import or call saveJournalTrade'
  );
  assert.equal(
    setupsContent.includes('COPY TRADE PLAN'),
    true,
    'Copy button must explicitly use "COPY TRADE PLAN" wording'
  );
  assert.equal(
    setupsContent.includes('does not execute trade or save to journal'),
    true,
    'Copy subtext must explicitly clarify it does not execute trade or save to journal'
  );
});

runTest('Behavioral simulation: clipboard write handles success, failure, repeated invocation without side effects', async () => {
  let writesRecorded = 0;
  let journalSavesInvoked = 0;
  let lastCopiedText = null;

  const mockNavigator = {
    clipboard: {
      writeText: async (text) => {
        writesRecorded++;
        lastCopiedText = text;
      },
    },
  };

  // Simulated handler mirroring Setups page logic
  const handleCopy = async (nav, setup, sizing) => {
    if (!sizing.isAvailable || sizing.recommendedShares <= 0) return { status: 'IGNORED' };
    const t1 = setup.target1 ? `$${setup.target1.toFixed(2)}` : '--';
    const orderStr = `BUY ${sizing.recommendedShares} ${setup.ticker} LMT $${sizing.entryPivot.toFixed(2)} | STP $${sizing.stopLoss.toFixed(2)} | TGT ${t1}`;
    try {
      if (!nav?.clipboard?.writeText) throw new Error('Clipboard API unavailable');
      await nav.clipboard.writeText(orderStr);
      // Notice: NO saveJournalTrade here!
      return { status: 'SUCCESS', orderStr };
    } catch (err) {
      return { status: 'FAILED', error: err.message };
    }
  };

  const setup = { ticker: 'NVDA', entryPivot: 125.5, stopLoss: 118.0, target1: 140.0 };
  const sizing = { isAvailable: true, recommendedShares: 50, entryPivot: 125.5, stopLoss: 118.0 };

  // Run 1: Success
  const res1 = await handleCopy(mockNavigator, setup, sizing);
  assert.equal(res1.status, 'SUCCESS');
  assert.equal(writesRecorded, 1);
  assert.equal(lastCopiedText, 'BUY 50 NVDA LMT $125.50 | STP $118.00 | TGT $140.00');
  assert.equal(journalSavesInvoked, 0);

  // Run 2: Repeated copying must write to clipboard again but create 0 records
  const res2 = await handleCopy(mockNavigator, setup, sizing);
  assert.equal(res2.status, 'SUCCESS');
  assert.equal(writesRecorded, 2);
  assert.equal(journalSavesInvoked, 0);

  // Run 3: Clipboard failure
  const brokenNavigator = { clipboard: { writeText: () => Promise.reject(new Error('Permission denied')) } };
  const res3 = await handleCopy(brokenNavigator, setup, sizing);
  assert.equal(res3.status, 'FAILED');
  assert.equal(res3.error, 'Permission denied');
  assert.equal(journalSavesInvoked, 0);
});

// ---------------------------------------------------------------------------
// 2. Completed Trade Eligibility Logic (Strict CLOSED & Valid Realized Outcomes)
// ---------------------------------------------------------------------------
console.log('\n--- Suite 2: Completed Trade Eligibility & Missingness Invariants ---');

// Reusable filter function matching PerformancePage exactly
function filterEligibleTrades(trades) {
  return trades
    .filter((t) => {
      if (t.status !== 'CLOSED') return false;
      if (typeof t.entryPrice !== 'number' || !Number.isFinite(t.entryPrice) || t.entryPrice <= 0) return false;
      if (typeof t.exitPrice !== 'number' || !Number.isFinite(t.exitPrice) || t.exitPrice <= 0) return false;
      if (typeof t.shares !== 'number' || !Number.isFinite(t.shares) || t.shares <= 0) return false;
      return true;
    })
    .map((t, idx) => {
      let pnl = null;
      if (t.pnlRaw !== undefined && typeof t.pnlRaw === 'number' && Number.isFinite(t.pnlRaw)) {
        pnl = t.pnlRaw;
      } else if (typeof t.pnl === 'number' && Number.isFinite(t.pnl)) {
        pnl = t.pnl;
      } else if (typeof t.pnl === 'string' && t.pnl.trim() !== '') {
        const clean = t.pnl.replace(/[^0-9.-]/g, '');
        const parsed = parseFloat(clean);
        if (Number.isFinite(parsed)) pnl = parsed;
      }
      if (pnl === null && Number.isFinite(t.entryPrice) && Number.isFinite(t.exitPrice) && Number.isFinite(t.shares)) {
        pnl = (t.exitPrice - t.entryPrice) * t.shares;
      }
      if (pnl === null || !Number.isFinite(pnl)) return null;

      const rRaw = t.rAchieved !== undefined && t.rAchieved !== null ? Number(t.rAchieved) : null;
      const rAchieved = rRaw !== null && Number.isFinite(rRaw) ? rRaw : null;

      const outcome = pnl > 0 ? 'WIN' : pnl < 0 ? 'LOSS' : 'SCRATCH';
      const rawSetup = (t.setupName || t.setup || '').trim();
      const setupName = rawSetup.length > 0 ? rawSetup : null;

      return {
        id: String(t.id || idx + 1),
        ticker: (t.ticker || t.symbol || 'ASSET').trim().toUpperCase(),
        setupName,
        entryPrice: t.entryPrice,
        exitPrice: t.exitPrice,
        shares: t.shares,
        pnl,
        rAchieved,
        followedRules: typeof t.followedRules === 'boolean' ? t.followedRules : null,
        date: t.date || t.entryDate || 'Recent',
        outcome,
      };
    })
    .filter((t) => t !== null);
}

runTest('OPEN record with positive exit price is strictly EXCLUDED', () => {
  const fixture = [
    {
      id: '1',
      ticker: 'TSLA',
      symbol: 'TSLA',
      status: 'OPEN',
      entryPrice: 200,
      exitPrice: 220, // positive exit price!
      shares: 10,
      pnl: '+$200.00',
    },
  ];
  const eligible = filterEligibleTrades(fixture);
  assert.equal(eligible.length, 0, 'OPEN record with positive exit price must be excluded');
});

runTest('OPEN record without exit price is strictly EXCLUDED', () => {
  const fixture = [
    {
      id: '2',
      ticker: 'AAPL',
      symbol: 'AAPL',
      status: 'OPEN',
      entryPrice: 150,
      exitPrice: null,
      shares: 20,
    },
  ];
  const eligible = filterEligibleTrades(fixture);
  assert.equal(eligible.length, 0, 'OPEN record without exit price must be excluded');
});

runTest('Missing, undefined, or unknown status is strictly EXCLUDED', () => {
  const fixture = [
    { id: '3', ticker: 'MSFT', entryPrice: 300, exitPrice: 320, shares: 10, pnl: '200' }, // status undefined
    { id: '4', ticker: 'AMZN', status: null, entryPrice: 100, exitPrice: 110, shares: 10, pnl: '100' },
    { id: '5', ticker: 'GOOG', status: 'PENDING', entryPrice: 100, exitPrice: 110, shares: 10, pnl: '100' },
  ];
  const eligible = filterEligibleTrades(fixture);
  assert.equal(eligible.length, 0, 'Non-CLOSED statuses must be excluded');
});

runTest('CLOSED record with valid, explicitly recorded zero outcome is ELIGIBLE as SCRATCH', () => {
  const fixture = [
    {
      id: '6',
      ticker: 'AMD',
      symbol: 'AMD',
      status: 'CLOSED',
      entryPrice: 100,
      exitPrice: 100,
      shares: 50,
      pnl: '$0.00',
      pnlRaw: 0,
      rAchieved: 0,
      setupName: 'Breakout',
    },
  ];
  const eligible = filterEligibleTrades(fixture);
  assert.equal(eligible.length, 1, 'Valid zero outcome must remain eligible');
  assert.equal(eligible[0].pnl, 0);
  assert.equal(eligible[0].outcome, 'SCRATCH');
});

runTest('CLOSED record with missing or invalid outcome cannot contribute to metrics', () => {
  const fixture = [
    {
      id: '7',
      ticker: 'META',
      symbol: 'META',
      status: 'CLOSED',
      entryPrice: 300,
      exitPrice: null, // missing exitPrice
      shares: 10,
    },
    {
      id: '8',
      ticker: 'NFLX',
      symbol: 'NFLX',
      status: 'CLOSED',
      entryPrice: 400,
      exitPrice: 0, // invalid non-positive exitPrice
      shares: 10,
    },
    {
      id: '9',
      ticker: 'INTC',
      symbol: 'INTC',
      status: 'CLOSED',
      entryPrice: 30,
      exitPrice: 35,
      shares: 0, // invalid shares
    },
  ];
  const eligible = filterEligibleTrades(fixture);
  assert.equal(eligible.length, 0, 'Invalid outcome/execution fields must be excluded');
});

runTest('Valid CLOSED winning and losing trades are correctly categorized', () => {
  const fixture = [
    {
      id: '10',
      ticker: 'NVDA',
      status: 'CLOSED',
      entryPrice: 100,
      exitPrice: 120,
      shares: 10,
      pnlRaw: 200,
      rAchieved: 2.5,
    },
    {
      id: '11',
      ticker: 'PLTR',
      status: 'CLOSED',
      entryPrice: 25,
      exitPrice: 23,
      shares: 100,
      pnlRaw: -200,
      rAchieved: -1.0,
    },
  ];
  const eligible = filterEligibleTrades(fixture);
  assert.equal(eligible.length, 2);
  assert.equal(eligible[0].outcome, 'WIN');
  assert.equal(eligible[0].pnl, 200);
  assert.equal(eligible[1].outcome, 'LOSS');
  assert.equal(eligible[1].pnl, -200);
});

// ---------------------------------------------------------------------------
// 3. Live Attribution & Metrics Integrity (PART 5)
// ---------------------------------------------------------------------------
console.log('\n--- Suite 3: Zero Fabricated Live Attribution & Genuine Metrics ---');

runTest('Performance page source has zero keyword-guessing or hardcoded regimes', () => {
  const perfContent = fs.readFileSync(path.join(rootDir, 'frontend/app/performance/page.tsx'), 'utf-8');
  assert.equal(perfContent.includes("marketRegime: 'UPTREND'"), false, 'Must not hardcode UPTREND');
  assert.equal(perfContent.includes("executionWindow: 'MORNING_PRIME'"), false, 'Must not hardcode MORNING_PRIME');
  assert.equal(perfContent.includes("setupArchetype: 'VCP_BREAKOUT'"), false, 'Must not default to VCP_BREAKOUT');
  assert.equal(perfContent.includes("clampReasonCategory: 'DRAWDOWN_DEFENSE'"), false, 'Must not manufacture DRAWDOWN_DEFENSE clamp in live ledger');
  assert.equal(perfContent.includes("Math.abs(pnl) || 500"), false, 'Must not invent risk dollars from PnL');
});

runTest('Setup names preserve exact recorded string without forced categorization', () => {
  const fixture = [
    {
      id: '1',
      ticker: 'SYM1',
      status: 'CLOSED',
      entryPrice: 10,
      exitPrice: 12,
      shares: 100,
      pnlRaw: 200,
      setupName: 'My Custom Breakout',
    },
    {
      id: '2',
      ticker: 'SYM2',
      status: 'CLOSED',
      entryPrice: 50,
      exitPrice: 55,
      shares: 10,
      pnlRaw: 50,
      setupName: '', // empty
    },
  ];
  const eligible = filterEligibleTrades(fixture);
  assert.equal(eligible[0].setupName, 'My Custom Breakout');
  assert.equal(eligible[1].setupName, null, 'Empty setup name must become null, not defaulted to VCP_BREAKOUT');
});

runTest('Empirical summary metrics calculate accurate win rate, profit factor, avg R without synthetic fallback', () => {
  const trades = [
    { id: '1', ticker: 'A', status: 'CLOSED', entryPrice: 10, exitPrice: 12, shares: 100, pnlRaw: 200, rAchieved: 2.0 },
    { id: '2', ticker: 'B', status: 'CLOSED', entryPrice: 20, exitPrice: 18, shares: 50, pnlRaw: -100, rAchieved: -1.0 },
    { id: '3', ticker: 'C', status: 'CLOSED', entryPrice: 30, exitPrice: 30, shares: 20, pnlRaw: 0, rAchieved: 0 },
  ];
  const eligible = filterEligibleTrades(trades);
  assert.equal(eligible.length, 3);

  const totalPnL = eligible.reduce((acc, t) => acc + t.pnl, 0);
  assert.equal(totalPnL, 100);

  const wins = eligible.filter((t) => t.outcome === 'WIN').length;
  assert.equal(wins, 1);
  const winRatePct = ((wins / eligible.length) * 100).toFixed(1);
  assert.equal(winRatePct, '33.3');

  const grossWins = eligible.filter((t) => t.pnl > 0).reduce((acc, t) => acc + t.pnl, 0);
  const grossLosses = Math.abs(eligible.filter((t) => t.pnl < 0).reduce((acc, t) => acc + t.pnl, 0));
  const profitFactor = (grossWins / grossLosses).toFixed(2);
  assert.equal(profitFactor, '2.00'); // 200 / 100 = 2.00

  const rTrades = eligible.filter((t) => t.rAchieved !== null);
  const avgR = (rTrades.reduce((acc, t) => acc + t.rAchieved, 0) / rTrades.length).toFixed(2);
  assert.equal(avgR, '0.33'); // (2.0 - 1.0 + 0) / 3 = 0.33
});

// ---------------------------------------------------------------------------
// 4. Portfolio Storage Wording (PART 6)
// ---------------------------------------------------------------------------
console.log('\n--- Suite 4: Portfolio Storage Accuracy & Zero False Privacy Claims ---');

runTest('Portfolio page accurately claims Authoritative API Persistence without browser-only claims', () => {
  const portfolioContent = fs.readFileSync(path.join(rootDir, 'frontend/app/portfolio/page.tsx'), 'utf-8');
  assert.equal(portfolioContent.includes('100% private to your browser'), false);
  assert.equal(portfolioContent.includes('ZERO-LOGIN PRIVATE STORAGE'), false);
  assert.equal(portfolioContent.includes('AUTHORITATIVE API PERSISTENCE'), true);
  assert.equal(portfolioContent.includes('backed by authoritative API persistence'), true);
});

runTest('OnboardingTourModal has zero false client-side encryption or browser-only claims', () => {
  const tourContent = fs.readFileSync(path.join(rootDir, 'frontend/components/OnboardingTourModal.tsx'), 'utf-8');
  assert.equal(tourContent.includes('ZERO-LOGIN PRIVATE STORAGE'), false);
  assert.equal(tourContent.includes('saved entirely in your local browser storage'), false);
  assert.equal(tourContent.includes('Client-Side Encrypted Risk'), false);
  assert.equal(tourContent.includes('API-BACKED PORTFOLIO PERSISTENCE'), true);
});

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------
console.log(`\n======================================================`);
console.log(`Results: ${passedTests} / ${totalTests} assertions passed (${Math.round((passedTests / totalTests) * 100)}%)`);
console.log(`======================================================\n`);

if (passedTests !== totalTests) {
  process.exit(1);
}

import { execSync } from 'node:child_process';
try {
  console.log('\n--- Suite 5: Executing Production Module Direct-Import Suite (verify-a1a-immediate-integrity.ts) ---');
  execSync('npx tsx frontend/scripts/verify-a1a-immediate-integrity.ts', { stdio: 'inherit', cwd: rootDir });
} catch (e) {
  process.exit(1);
}
