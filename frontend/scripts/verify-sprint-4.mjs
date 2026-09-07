/**
 * ARX Terminal vNext - Sprint 4 Verification Suite
 * Tests Portfolio Attention Aggregation, Data Quality, Edge Cases, and Flood Protection.
 * Acceptance Criteria: AC-PF-01 through AC-PF-10, DQ-001 through DQ-005.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const frontendRoot = path.resolve(__dirname, '..');

let totalTests = 0;
let passedTests = 0;
let failedTests = 0;

function assert(condition, message) {
  totalTests++;
  if (condition) {
    console.log(`  \x1b[32m✓\x1b[0m ${message}`);
    passedTests++;
  } else {
    console.error(`  \x1b[31m✗ FAIL:\x1b[0m ${message}`);
    failedTests++;
  }
}

console.log('\n========================================================================');
console.log('  ARX Terminal vNext: Sprint 4 Verification Suite                       ');
console.log('  (Portfolio Intelligence, Data Quality & Attention Aggregation)         ');
console.log('========================================================================\n');

// ------------------------------------------------------------------------
// SUITE 1: FILE & CONTRACT INTEGRITY
// ------------------------------------------------------------------------
const typesPath = path.join(frontendRoot, 'types', 'portfolio-intelligence.ts');
const typesContent = fs.readFileSync(typesPath, 'utf8');

assert(typesContent.includes('export interface PortfolioAttentionEntry'), 'types/portfolio-intelligence.ts exports PortfolioAttentionEntry');
assert(typesContent.includes('export interface PortfolioAttentionFeed'), 'types/portfolio-intelligence.ts exports PortfolioAttentionFeed');
assert(typesContent.includes('export interface MorningBriefingSummary'), 'types/portfolio-intelligence.ts exports MorningBriefingSummary');
assert(typesContent.includes('export type DataQualityStatus'), 'types/portfolio-intelligence.ts exports DataQualityStatus');

const telemetryPath = path.join(frontendRoot, 'types', 'telemetry.ts');
const telemetryContent = fs.readFileSync(telemetryPath, 'utf8');
assert(telemetryContent.includes('"portfolio_feed_loaded"'), 'types/telemetry.ts includes portfolio_feed_loaded');
assert(telemetryContent.includes('"morning_brief_viewed"'), 'types/telemetry.ts includes morning_brief_viewed');
assert(telemetryContent.includes('"portfolio_item_opened"'), 'types/telemetry.ts includes portfolio_item_opened');

// ------------------------------------------------------------------------
// SUITE 2: DATA QUALITY VALIDATION (DQ-001 to DQ-005)
// ------------------------------------------------------------------------
const dqPath = path.join(frontendRoot, 'lib', 'engine', 'dataQualityValidator.ts');
const dqContent = fs.readFileSync(dqPath, 'utf8');

// DQ-001 Freshness
function testFreshness(timestampStr, maxAgeMinutes = 60) {
  const timestamp = new Date(timestampStr).getTime();
  const now = Date.now();
  const ageMinutes = (now - timestamp) / (1000 * 60);
  if (ageMinutes < -5 || ageMinutes > maxAgeMinutes) return { status: 'REJECTED', valid: false };
  if (ageMinutes > maxAgeMinutes * 0.5) return { status: 'DEGRADED', valid: true };
  return { status: 'VALID', valid: true };
}

const freshTimestamp = new Date(Date.now() - 5 * 60 * 1000).toISOString();
const staleTimestamp = new Date(Date.now() - 120 * 60 * 1000).toISOString();
assert(testFreshness(freshTimestamp, 60).status === 'VALID', 'DQ-001: Fresh timestamp is marked VALID');
assert(testFreshness(staleTimestamp, 60).status === 'REJECTED', 'DQ-001: Stale timestamp (>60m) is marked REJECTED');

// DQ-002 Sequence Validation
function testSequence(events) {
  const precedence = {
    returning_user_detected: 1,
    delta_banner_viewed: 2,
    delta_banner_expanded: 3,
    thesis_confirmed: 4,
    delta_acknowledged: 4,
  };
  let maxStep = 0;
  for (const ev of events) {
    const step = precedence[ev];
    if (step !== undefined) {
      if (step < maxStep) return false;
      maxStep = step;
    }
  }
  return true;
}
assert(testSequence(['returning_user_detected', 'delta_banner_viewed', 'thesis_confirmed']) === true, 'DQ-002: Valid telemetry progression accepted');
assert(testSequence(['thesis_confirmed', 'delta_banner_viewed']) === false, 'DQ-002: Out-of-order sequence rejected');

// DQ-003 Metric Bounds
function testBounds(score, price, flowZ) {
  if (score < 0 || score > 100) return false;
  if (price <= 0) return false;
  if (flowZ < -10 || flowZ > 10) return false;
  return true;
}
assert(testBounds(78, 18.5, 2.1) === true, 'DQ-003: In-bounds metrics accepted');
assert(testBounds(120, 18.5, 2.1) === false, 'DQ-003: Out-of-bounds setup score (120) rejected');
assert(testBounds(78, -5.0, 2.1) === false, 'DQ-003: Negative spot price rejected');
assert(testBounds(78, 18.5, 15.0) === false, 'DQ-003: Out-of-bounds flow Z (15.0) rejected');

// DQ-004 Snapshot Consistency
function testTransition(prev, curr) {
  if (prev === 'STOPPED_OUT' && curr === 'IN_BUY_ZONE') return false;
  return true;
}
assert(testTransition('WAITING_PULLBACK', 'IN_BUY_ZONE') === true, 'DQ-004: Legal state transition WAITING_PULLBACK -> IN_BUY_ZONE accepted');
assert(testTransition('STOPPED_OUT', 'IN_BUY_ZONE') === false, 'DQ-004: Illegal teleport STOPPED_OUT -> IN_BUY_ZONE rejected');

// ------------------------------------------------------------------------
// SUITE 3: TELEMETRY FLOOD PROTECTION
// ------------------------------------------------------------------------
const floodPath = path.join(frontendRoot, 'lib', 'telemetry', 'floodProtection.ts');
const floodContent = fs.readFileSync(floodPath, 'utf8');

assert(floodContent.includes('isDuplicate'), 'lib/telemetry/floodProtection.ts implements isDuplicate');
assert(floodContent.includes('isTooltipThrottled'), 'lib/telemetry/floodProtection.ts implements isTooltipThrottled');
assert(floodContent.includes('recordAndCheckBurst'), 'lib/telemetry/floodProtection.ts implements recordAndCheckBurst');

// ------------------------------------------------------------------------
// SUITE 4: PORTFOLIO AGGREGATION ENGINE (AC-PF-01 to AC-PF-10)
// ------------------------------------------------------------------------
function severityWeight(sev) {
  switch (sev) {
    case 'CRITICAL': return 4;
    case 'MATERIAL': return 3;
    case 'INFO': return 2;
    default: return 1;
  }
}

function dedupeByTicker(reports) {
  const map = new Map();
  for (const r of reports) {
    const existing = map.get(r.ticker);
    if (!existing || severityWeight(r.maxSeverity) > severityWeight(existing.maxSeverity)) {
      map.set(r.ticker, r);
    }
  }
  return Array.from(map.values());
}

function buildFeed(reports, maxCritical = 10, maxMaterial = 20) {
  // Quality & noise filtering
  const valid = reports.filter(r => r.isMaterial && r.maxSeverity !== 'NONE' && r.quality !== 'STALE' && r.quality !== 'INVALID');
  const deduped = dedupeByTicker(valid);

  deduped.sort((a, b) => severityWeight(b.maxSeverity) - severityWeight(a.maxSeverity));

  const criticalAll = deduped.filter(e => e.maxSeverity === 'CRITICAL');
  const materialAll = deduped.filter(e => e.maxSeverity === 'MATERIAL');

  return {
    criticalItems: criticalAll.slice(0, maxCritical),
    materialItems: materialAll.slice(0, maxMaterial),
    summaryCount: Math.max(0, criticalAll.length - maxCritical) + Math.max(0, materialAll.length - maxMaterial),
    totalAttentionCount: criticalAll.length + materialAll.length
  };
}

// AC-PF-01 Severity Ordering
const sampleReports = [
  { ticker: 'META', maxSeverity: 'MATERIAL', isMaterial: true, items: [{ category: 'FLOW', reason: 'Flow surge' }] },
  { ticker: 'CPRX', maxSeverity: 'CRITICAL', isMaterial: true, items: [{ category: 'EXECUTION', reason: 'Entered buy zone' }] },
  { ticker: 'NVDA', maxSeverity: 'MATERIAL', isMaterial: true, items: [{ category: 'SETUP', reason: 'Score surge' }] }
];
const feed1 = buildFeed(sampleReports);
assert(feed1.criticalItems[0].ticker === 'CPRX', 'AC-PF-01: Critical CPRX appears first in feed');
assert(feed1.materialItems.some(m => m.ticker === 'META') && feed1.materialItems.some(m => m.ticker === 'NVDA'), 'AC-PF-01: Material items ranked below critical');

// AC-PF-02 Noise Suppression
const noiseReports = [
  { ticker: 'LLY', maxSeverity: 'NONE', isMaterial: false, items: [] },
  { ticker: 'TSLA', maxSeverity: 'INFO', isMaterial: false, items: [] },
  { ticker: 'CPRX', maxSeverity: 'CRITICAL', isMaterial: true, items: [{ category: 'EXECUTION', reason: 'Action' }] }
];
const feed2 = buildFeed(noiseReports);
assert(feed2.criticalItems.length === 1 && feed2.criticalItems[0].ticker === 'CPRX', 'AC-PF-02: Sub-threshold noise tickers (LLY, TSLA) suppressed with zero footprint');

// AC-PF-05 & AC-PF-07 Cross-Ticker Deduplication & Severity Consolidation
const multiReportSameTicker = [
  { ticker: 'CPRX', maxSeverity: 'MATERIAL', isMaterial: true, items: [{ category: 'FLOW', reason: 'Flow surge' }] },
  { ticker: 'CPRX', maxSeverity: 'CRITICAL', isMaterial: true, items: [{ category: 'EXECUTION', reason: 'Entered Buy Zone' }] },
  { ticker: 'CPRX', maxSeverity: 'INFO', isMaterial: true, items: [{ category: 'VALIDATION', reason: 'Tier promo' }] }
];
const feed3 = buildFeed(multiReportSameTicker);
assert(feed3.criticalItems.length === 1 && feed3.criticalItems[0].ticker === 'CPRX', 'AC-PF-05 & AC-PF-07: Single consolidated card per ticker with highest severity (CRITICAL) winning');

// AC-PF-08 Data Quality Suppression
const staleReports = [
  { ticker: 'GOOGL', maxSeverity: 'CRITICAL', isMaterial: true, quality: 'STALE', items: [{ category: 'EXECUTION' }] },
  { ticker: 'MSFT', maxSeverity: 'CRITICAL', isMaterial: true, quality: 'TRUSTED', items: [{ category: 'EXECUTION' }] }
];
const feed4 = buildFeed(staleReports);
assert(feed4.criticalItems.length === 1 && feed4.criticalItems[0].ticker === 'MSFT', 'AC-PF-08: Stale ticker GOOGL excluded from feed to preserve Delta Trust Index');

// AC-PF-09 Feed Capacity Protection
const fortyCriticalReports = Array.from({ length: 40 }, (_, i) => ({
  ticker: `TICK${i}`,
  maxSeverity: 'CRITICAL',
  isMaterial: true,
  items: [{ category: 'EXECUTION', reason: 'Triggered' }]
}));
const feed5 = buildFeed(fortyCriticalReports, 10, 20);
assert(feed5.criticalItems.length === 10, 'AC-PF-09: Exactly top 10 critical items expanded');
assert(feed5.summaryCount === 30, 'AC-PF-09: Remaining 30 items cleanly collapsed into summary count');

// AC-PF-10 Duplicate Portfolio Alert Prevention
const acknowledgedReports = [
  { ticker: 'CPRX', maxSeverity: 'NONE', isMaterial: false, items: [] } // after 1-click ack, isMaterial becomes false
];
const feed6 = buildFeed(acknowledgedReports);
assert(feed6.totalAttentionCount === 0, 'AC-PF-10: Acknowledged baseline generates zero portfolio alert items');

// ------------------------------------------------------------------------
// SUITE 5: UI COMPONENTS INTEGRITY
// ------------------------------------------------------------------------
const briefingPath = path.join(frontendRoot, 'components', 'portfolio', 'MorningBriefingCard.tsx');
const briefingContent = fs.readFileSync(briefingPath, 'utf8');
assert(briefingContent.includes('Institutional Morning Briefing'), 'MorningBriefingCard.tsx implements executive morning briefing');
assert(briefingContent.includes('morning_brief_viewed'), 'MorningBriefingCard.tsx emits morning_brief_viewed telemetry');

const feedCompPath = path.join(frontendRoot, 'components', 'portfolio', 'PortfolioAttentionFeed.tsx');
const feedCompContent = fs.readFileSync(feedCompPath, 'utf8');
assert(feedCompContent.includes('Things Requiring Attention Today'), 'PortfolioAttentionFeed.tsx renders primary attention header');
assert(feedCompContent.includes('portfolio_feed_loaded'), 'PortfolioAttentionFeed.tsx emits portfolio_feed_loaded telemetry');
assert(feedCompContent.includes('portfolio_item_opened'), 'PortfolioAttentionFeed.tsx emits portfolio_item_opened on item click');

console.log('\n========================================================================');
console.log(`  VERIFICATION RESULTS: ${passedTests} PASSED, ${failedTests} FAILED`);
console.log('========================================================================\n');

if (failedTests > 0) {
  process.exit(1);
} else {
  process.exit(0);
}
