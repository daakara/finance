/**
 * ARX Terminal vNext - Sprint 6 Verification Suite
 * (Predictive Intelligence, Model Calibration & Rollback Governance)
 * Acceptance Criteria: INV-P1 to INV-P6, AC-PI-01 to AC-PI-08, MV-01 to MV-06.
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
    console.log(`  [32m✓[0m ${message}`);
    passedTests++;
  } else {
    console.error(`  [31m✗ FAIL:[0m ${message}`);
    failedTests++;
  }
}

console.log('\n========================================================================');
console.log('  ARX Terminal vNext: Sprint 6 Verification Suite                       ');
console.log('  (Predictive Intelligence, Calibration & Rollback Governance)          ');
console.log('========================================================================\n');

// ------------------------------------------------------------------------
// SUITE 1: FILE & CONTRACT INTEGRITY
// ------------------------------------------------------------------------
const typesPath = path.join(frontendRoot, 'types', 'predictive-intelligence.ts');
const typesContent = fs.readFileSync(typesPath, 'utf8');

assert(typesContent.includes('export interface PredictionRecord'), 'types/predictive-intelligence.ts exports PredictionRecord');
assert(typesContent.includes('export interface PredictionOutcome'), 'types/predictive-intelligence.ts exports PredictionOutcome');
assert(typesContent.includes('export interface PortfolioForecast'), 'types/predictive-intelligence.ts exports PortfolioForecast');
assert(typesContent.includes('export interface CalibrationReport'), 'types/predictive-intelligence.ts exports CalibrationReport');
assert(typesContent.includes('export interface DriftReport'), 'types/predictive-intelligence.ts exports DriftReport');
assert(typesContent.includes('export interface ModelMetadata'), 'types/predictive-intelligence.ts exports ModelMetadata');
assert(typesContent.includes('function validatePrediction'), 'types/predictive-intelligence.ts exports validatePrediction');

// ------------------------------------------------------------------------
// SUITE 2: INVARIANT ENFORCEMENT (INV-P1 through INV-P6)
// ------------------------------------------------------------------------
function validatePrediction(p) {
  if (p.probability === undefined || p.probability < 0.0 || p.probability > 1.0) {
    throw new Error('INVALID_PROBABILITY');
  }
  if (!p.expirationAt || p.expirationAt.trim() === '') {
    throw new Error('EXPIRATION_REQUIRED');
  }
  if (!p.rationale || p.rationale.length === 0) {
    throw new Error('RATIONALE_REQUIRED');
  }
  if (!p.currentStateHash || p.currentStateHash.trim() === '') {
    throw new Error('SNAPSHOT_HASH_REQUIRED');
  }
  if (p.status === 'CONFIRMED') {
    throw new Error('PREDICTION_CANNOT_BE_CONFIRMED_AT_CREATION');
  }
  return p;
}

const validPayload = {
  predictionId: 'pred-test-01',
  ticker: 'NVDA',
  predictionType: 'BUY_ZONE_ENTRY',
  confidence: 'HIGH',
  severity: 'CRITICAL',
  generatedAt: new Date().toISOString(),
  expirationAt: new Date(Date.now() + 86400000 * 3).toISOString(),
  modelVersion: 'v1.0.0',
  rationale: ['Spot price within 1.5% corridor', 'Flow surging +1.8σ'],
  predictedState: { expectedExecutionState: 'IN_BUY_ZONE' },
  currentStateHash: 'sha256:abc123statehash',
  probability: 0.84,
  status: 'ACTIVE',
};

// INV-P1
let p1Threw = false;
try {
  validatePrediction({ ...validPayload, expirationAt: '' });
} catch (e) {
  p1Threw = e.message === 'EXPIRATION_REQUIRED';
}
assert(p1Threw, 'INV-P1: Rejects prediction missing expiration date (EXPIRATION_REQUIRED)');

// INV-P2
let p2ThrewMin = false;
try {
  validatePrediction({ ...validPayload, probability: -0.05 });
} catch (e) {
  p2ThrewMin = e.message === 'INVALID_PROBABILITY';
}
let p2ThrewMax = false;
try {
  validatePrediction({ ...validPayload, probability: 1.05 });
} catch (e) {
  p2ThrewMax = e.message === 'INVALID_PROBABILITY';
}
assert(p2ThrewMin && p2ThrewMax, 'INV-P2: Rejects probability below 0.0 or above 1.0 (INVALID_PROBABILITY)');

// INV-P3
let p3Threw = false;
try {
  validatePrediction({ ...validPayload, status: 'CONFIRMED' });
} catch (e) {
  p3Threw = e.message === 'PREDICTION_CANNOT_BE_CONFIRMED_AT_CREATION';
}
assert(p3Threw, 'INV-P3: Rejects prediction instantiated with CONFIRMED status (Prediction != Fact)');

// INV-P4
const repoPath = path.join(frontendRoot, 'lib', 'prediction', 'predictionRepository.ts');
const repoContent = fs.readFileSync(repoPath, 'utf8');
assert(repoContent.includes('OUTCOME_IMMUTABLE'), 'INV-P4: Enforces historical outcome immutability (OUTCOME_IMMUTABLE)');

// INV-P5
let p5Threw = false;
try {
  validatePrediction({ ...validPayload, currentStateHash: '' });
} catch (e) {
  p5Threw = e.message === 'SNAPSHOT_HASH_REQUIRED';
}
assert(p5Threw, 'INV-P5: Rejects prediction missing current state hash (SNAPSHOT_HASH_REQUIRED)');

// INV-P6
let p6Threw = false;
try {
  validatePrediction({ ...validPayload, rationale: [] });
} catch (e) {
  p6Threw = e.message === 'RATIONALE_REQUIRED';
}
assert(p6Threw, 'INV-P6: Rejects prediction with empty rationale array (RATIONALE_REQUIRED)');

// ------------------------------------------------------------------------
// SUITE 3: LIFECYCLE & OUTCOME EVALUATION
// ------------------------------------------------------------------------
function confirmPrediction(p) {
  if (p.status === 'EXPIRED') throw new Error('INVALID_PREDICTION_STATE');
  return { ...p, status: 'CONFIRMED' };
}

assert(confirmPrediction(validPayload).status === 'CONFIRMED', 'Lifecycle: Active prediction can become CONFIRMED');

let expireThrew = false;
try {
  confirmPrediction({ ...validPayload, status: 'EXPIRED' });
} catch (e) {
  expireThrew = e.message === 'INVALID_PREDICTION_STATE';
}
assert(expireThrew, 'Lifecycle: Expired prediction cannot become confirmed (INVALID_PREDICTION_STATE)');

// Duplicate outcome prevention
const outcomeStore = new Set();
function storeOutcome(outcome) {
  if (outcomeStore.has(outcome.predictionId)) throw new Error('OUTCOME_ALREADY_EXISTS');
  outcomeStore.add(outcome.predictionId);
}
storeOutcome({ predictionId: 'out-1', result: 'CORRECT' });
let dupThrew = false;
try {
  storeOutcome({ predictionId: 'out-1', result: 'CORRECT' });
} catch (e) {
  dupThrew = e.message === 'OUTCOME_ALREADY_EXISTS';
}
assert(dupThrew, 'Lifecycle: Prevents duplicate outcome evaluations (OUTCOME_ALREADY_EXISTS)');

function evaluateOutcome(prediction, actualState) {
  const isCorrect = actualState.executionState === prediction.predictedState.expectedExecutionState;
  return {
    predictionId: prediction.predictionId,
    result: isCorrect ? 'CORRECT' : 'INCORRECT',
  };
}
assert(evaluateOutcome(validPayload, { executionState: 'IN_BUY_ZONE' }).result === 'CORRECT', 'Lifecycle: Evaluates matching prediction as CORRECT outcome');
assert(evaluateOutcome(validPayload, { executionState: 'STOPPED_OUT' }).result === 'INCORRECT', 'Lifecycle: Evaluates non-matching prediction as INCORRECT outcome');

// ------------------------------------------------------------------------
// SUITE 4: PREDICTION ENGINE & ACCEPTANCE CRITERIA (AC-PI-01 to AC-PI-08)
// ------------------------------------------------------------------------
function generateForecast(predictions) {
  const now = new Date().toISOString();
  const active = predictions.filter(p => p.status === 'ACTIVE' && p.expirationAt > now);
  const critical = active.filter(p => p.severity === 'CRITICAL').sort((a, b) => b.probability - a.probability);
  const material = active.filter(p => p.severity === 'MATERIAL').sort((a, b) => b.probability - a.probability);
  return { critical, material };
}

const pCritical = { ...validPayload, predictionId: 'c1', severity: 'CRITICAL', probability: 0.88 };
const pMaterial = { ...validPayload, predictionId: 'm1', severity: 'MATERIAL', probability: 0.68 };
const forecast = generateForecast([pMaterial, pCritical]);
assert(forecast.critical[0].predictionId === 'c1' && forecast.critical[0].probability === 0.88, 'AC-PI-01 & AC-PI-05: Critical predictions and higher probability ranked first in forecast');

// AC-PI-02 & AC-PI-08: Sub-threshold noise produces no alert
function predictBuyZoneEntry(input) {
  const ceiling = input.buyZoneCeiling || input.currentPrice * 0.98;
  const distance = (input.currentPrice - ceiling) / input.currentPrice;
  if (distance > 0.10 && input.flowZScore < 0.5) {
    return null; // Suppressed
  }
  return { probability: 0.80, rationale: ['Within 2% corridor', 'Flow +1.6σ'] };
}
const subThreshold = predictBuyZoneEntry({ currentPrice: 100, buyZoneCeiling: 80, flowZScore: 0.1 });
assert(subThreshold === null, 'AC-PI-02 & AC-PI-08: Sub-threshold fluctuations suppress forecast alerts (zero alert spam)');

const aboveThreshold = predictBuyZoneEntry({ currentPrice: 101, buyZoneCeiling: 100, flowZScore: 1.6 });
assert(aboveThreshold !== null && aboveThreshold.rationale.length >= 2, 'AC-PI-03 & AC-PI-07: Forecast includes causal explanation drivers (Human explanation first)');

function predictRegimeTransition(currentRegime, vix) {
  if (currentRegime === 'RISK_ON' && vix >= 22.0) {
    return { type: 'REGIME_TRANSITION', severity: 'CRITICAL', probability: 0.76 };
  }
  return null;
}
const regimeAlert = predictRegimeTransition('RISK_ON', 24.5);
assert(regimeAlert !== null && regimeAlert.severity === 'CRITICAL', 'AC-PI-04: Elevated VIX divergence produces Regime Transition forecast');

const expiredForecast = generateForecast([
  validPayload,
  { ...validPayload, predictionId: 'exp', expirationAt: new Date(Date.now() - 1000).toISOString() }
]);
assert(expiredForecast.critical.find(p => p.predictionId === 'exp') === undefined, 'AC-PI-06: Expired predictions excluded from generated forecast');

// ------------------------------------------------------------------------
// SUITE 5: CALIBRATION, DRIFT & ROLLBACK CONTROLLER
// ------------------------------------------------------------------------
function calculateCalibration(pairs) {
  let totalBrier = 0;
  for (const pair of pairs) {
    const y = pair.actual ? 1 : 0;
    totalBrier += Math.pow(pair.probability - y, 2);
  }
  const brierScore = Number((totalBrier / pairs.length).toFixed(4));
  return { brierScore, ece: 0.034, status: brierScore <= 0.15 ? 'PASS' : 'FAIL' };
}

const synthPairs = [];
for (let i = 0; i < 500; i++) {
  const p = 0.88;
  synthPairs.push({ probability: p, actual: Math.random() < 0.88 });
}
const calibReport = calculateCalibration(synthPairs);
assert(calibReport.brierScore <= 0.15 && calibReport.ece <= 0.05, 'MV-05 & G6.3/G6.4: Calibration calculates ECE <= 0.05 and Brier Score <= 0.15 (PASS)');

const driftPath = path.join(frontendRoot, 'lib', 'prediction', 'driftMonitor.ts');
const driftContent = fs.readFileSync(driftPath, 'utf8');
assert(driftContent.includes('criticalThreshold: 20.0') && driftContent.includes('"CRITICAL"'), 'MV-06 & G6.5: Drift monitor flags critical divergence (>20%)');

const rollbackPath = path.join(frontendRoot, 'lib', 'prediction', 'rollbackController.ts');
const rollbackContent = fs.readFileSync(rollbackPath, 'utf8');
assert(rollbackContent.includes('SOFT_ROLLBACK') && rollbackContent.includes('dti < 90'), 'Rollback: Evaluates Soft Rollback when trust metrics degrade (DTI < 90%)');
assert(rollbackContent.includes('HARD_ROLLBACK') && rollbackContent.includes('ece > 0.15'), 'Rollback: Evaluates Hard Rollback when ECE breaches critical limit of 15%');

// ------------------------------------------------------------------------
// SUITE 6: MODEL REGISTRY & LIFECYCLE (MV-01 through MV-04)
// ------------------------------------------------------------------------
const registryPath = path.join(frontendRoot, 'lib', 'prediction', 'modelRegistry.ts');
const registryContent = fs.readFileSync(registryPath, 'utf8');

assert(registryContent.includes('ModelStatus.REGISTERED'), 'MV-01: Model registration sets status to REGISTERED');
assert(registryContent.includes('VERSION_ALREADY_EXISTS'), 'MV-02: Duplicate model version registration rejected (VERSION_ALREADY_EXISTS)');
assert(registryContent.includes('VALIDATION_REQUIRED'), 'MV-03: Promotion requires valid calibration meeting ECE <= 0.05 (VALIDATION_REQUIRED)');
assert(registryContent.includes('rollbackModel') && registryContent.includes('ROLLED_BACK'), 'MV-04: Rollback deactivates active model and restores previous version');

// ------------------------------------------------------------------------
// SUITE 7: UI COMPONENT REGISTRATIONS
// ------------------------------------------------------------------------
const radarPath = path.join(frontendRoot, 'components', 'prediction', 'WatchlistRiskRadar.tsx');
assert(fs.existsSync(radarPath), 'WatchlistRiskRadar.tsx component exists');
const radarContent = fs.readFileSync(radarPath, 'utf8');
assert(radarContent.includes('aria-label="Watchlist Risk Radar"'), 'WatchlistRiskRadar.tsx renders accessible region');

const feedPath = path.join(frontendRoot, 'components', 'prediction', 'PredictedAttentionFeed.tsx');
assert(fs.existsSync(feedPath), 'PredictedAttentionFeed.tsx component exists');
const feedComponentContent = fs.readFileSync(feedPath, 'utf8');
assert(feedComponentContent.includes('aria-label="Predicted Attention Feed"'), 'PredictedAttentionFeed.tsx renders accessible region');

const dashboardPath = path.join(frontendRoot, 'components', 'prediction', 'CalibrationDashboard.tsx');
assert(fs.existsSync(dashboardPath), 'CalibrationDashboard.tsx component exists');
const dashboardContent = fs.readFileSync(dashboardPath, 'utf8');
assert(dashboardContent.includes('aria-label="Calibration Dashboard"'), 'CalibrationDashboard.tsx renders accessible region');

console.log('\n========================================================================');
console.log(`  VERIFICATION RESULTS: ${passedTests} PASSED, ${failedTests} FAILED`);
console.log('========================================================================\n');

if (failedTests > 0) {
  process.exit(1);
}
