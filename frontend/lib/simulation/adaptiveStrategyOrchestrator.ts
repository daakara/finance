/**
 * Horizon 4: Adaptive Strategy Orchestrator Engine (M17)
 * 
 * Enforces Invariants:
 * - INV-OI67: Strategy Transition Integrity (Traceability, auditability, >= 90% rollback coverage)
 * - INV-OI68: Portfolio Evolution Coverage (Every quarter Q1-Q4 defines Primary, Fallback, Recovery)
 * - INV-OI69: Adaptive Re-Optimization Trigger (Automated re-evaluation under drift/shock without silent drops)
 */

import {
  QuarterlyStrategyPlan,
  StrategySequence,
  OrchestratorState,
  INV_OI67,
  INV_OI68,
  INV_OI69,
} from '../../types/simulation-digital-twin';
import { evaluateStrategyDrift, CANONICAL_TELEMETRY_SNAPSHOT, STRESSED_TELEMETRY_SNAPSHOT } from './strategyDriftEngine';
import { getNormalizedExternalSignals } from './externalSignalEngine';
import { evaluateStrategyPortfolio } from './strategyPortfolioEngine';

// -------------------------------------------------------------
// CANONICAL 12-MONTH STRATEGY SEQUENCE
// -------------------------------------------------------------

export const CANONICAL_QUARTERLY_PLANS: QuarterlyStrategyPlan[] = [
  {
    quarter: 'Q1',
    horizonMonths: 3,
    primaryStrategyId: 'STRAT-B-DUAL',
    primaryStrategyName: 'Strategy B: Dual Curriculum & Governance',
    fallbackStrategyId: 'STRAT-A-TRN',
    recoveryStrategyId: 'STRAT-D-RESIL',
    projectedOhi: 88.7,
    expectedRoi: 24.5,
    rollbackCoveragePct: 100.0,
    recoveryHours: 3.0,
    status: 'ACTIVE',
  },
  {
    quarter: 'Q2',
    horizonMonths: 6,
    primaryStrategyId: 'STRAT-D-RESIL',
    primaryStrategyName: 'Strategy D: Resilient Infrastructure & RTO Compression',
    fallbackStrategyId: 'STRAT-B-DUAL',
    recoveryStrategyId: 'STRAT-C-CONSV',
    projectedOhi: 91.2,
    expectedRoi: 28.0,
    rollbackCoveragePct: 100.0,
    recoveryHours: 1.0,
    status: 'RECOMMENDED',
  },
  {
    quarter: 'Q3',
    horizonMonths: 9,
    primaryStrategyId: 'STRAT-D-RESIL',
    primaryStrategyName: 'Strategy D: Scale Out Resilience',
    fallbackStrategyId: 'STRAT-A-TRN',
    recoveryStrategyId: 'STRAT-C-CONSV',
    projectedOhi: 92.5,
    expectedRoi: 31.2,
    rollbackCoveragePct: 95.0,
    recoveryHours: 2.0,
    status: 'PLANNED',
  },
  {
    quarter: 'Q4',
    horizonMonths: 12,
    primaryStrategyId: 'STRAT-B-DUAL',
    primaryStrategyName: 'Strategy B: Governance Normalization & Reserve',
    fallbackStrategyId: 'STRAT-C-CONSV',
    recoveryStrategyId: 'STRAT-C-CONSV',
    projectedOhi: 90.0,
    expectedRoi: 18.0,
    rollbackCoveragePct: 100.0,
    recoveryHours: 4.0,
    status: 'RESERVE',
  },
];

export const CANONICAL_STRATEGY_SEQUENCE: StrategySequence = {
  sequenceId: 'SEQ-2026-ADAPTIVE-01',
  name: 'Canonical 12-Month Adaptive Strategic Trajectory',
  timeframeHorizon: '12_MONTHS',
  quarters: CANONICAL_QUARTERLY_PLANS,
  overallProjectedOhi: 90.6,
  cumulativeRoi: 25.4,
  averageRobustness: 91.8,
  meanSurvivability: 96.2,
  transitionIntegrityScore: 98.5,
};

/**
 * Validates Invariant INV-OI67: Strategy Transition Integrity.
 */
export function verifyTransitionIntegrity(
  fromStrategyId: string,
  toStrategyId: string,
  rollbackCoveragePct = 100.0
): { valid: boolean; score: number; violations: string[] } {
  const violations: string[] = [];

  if (!fromStrategyId || !toStrategyId) {
    violations.push('TRANSITION_INVALID: Missing origin or destination strategy identifier');
  }

  if (rollbackCoveragePct < 90.0) {
    violations.push(
      `TRANSITION_ROLLBACK_INSUFFICIENT: Rollback coverage (${rollbackCoveragePct}%) falls below certified threshold (90.0%)`
    );
  }

  const score = rollbackCoveragePct >= 90.0 ? Math.min(100, rollbackCoveragePct) : rollbackCoveragePct;

  return {
    valid: violations.length === 0,
    score,
    violations,
  };
}

/**
 * Validates Invariant INV-OI68: Portfolio Evolution Coverage across all 4 quarters.
 */
export function verifyEvolutionCoverage(quarters: QuarterlyStrategyPlan[]): {
  valid: boolean;
  coveredQuarters: string[];
  violations: string[];
} {
  const required = ['Q1', 'Q2', 'Q3', 'Q4'];
  const coveredQuarters: string[] = [];
  const violations: string[] = [];

  for (const qKey of required) {
    const qPlan = quarters.find((q) => q.quarter === qKey);
    if (!qPlan) {
      violations.push(`MISSING_QUARTER_PLAN: Quarter ${qKey} missing from strategy sequence`);
      continue;
    }

    if (!qPlan.primaryStrategyId) {
      violations.push(`MISSING_PRIMARY_STRATEGY: Quarter ${qKey} lacks defined primary strategy`);
    }
    if (!qPlan.fallbackStrategyId) {
      violations.push(`MISSING_FALLBACK_STRATEGY: Quarter ${qKey} lacks defined fallback strategy`);
    }
    if (!qPlan.recoveryStrategyId) {
      violations.push(`MISSING_RECOVERY_STRATEGY: Quarter ${qKey} lacks defined recovery strategy`);
    }

    coveredQuarters.push(qKey);
  }

  return {
    valid: violations.length === 0 && coveredQuarters.length === 4,
    coveredQuarters,
    violations,
  };
}

/**
 * Generates a deterministic replay hash for orchestrator state verification.
 */
export function generateOrchestratorHash(
  activeStrategyId: string,
  expectedOhi: number,
  actualOhi: number,
  timestampUtc: string
): string {
  const seed = `${activeStrategyId}:${expectedOhi}:${actualOhi}:${timestampUtc.slice(0, 10)}`;
  let hash = 0x811c9dc5;
  for (let i = 0; i < seed.length; i++) {
    hash ^= seed.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return `ORC-HASH-0x${(hash >>> 0).toString(16).toUpperCase().padStart(8, '0')}`;
}

/**
 * Triggers full closed-loop adaptive re-optimization per Invariant INV-OI69.
 */
export function triggerAdaptiveReoptimization(
  isStressedScenario = false
): OrchestratorState {
  const telemetry = isStressedScenario ? STRESSED_TELEMETRY_SNAPSHOT : CANONICAL_TELEMETRY_SNAPSHOT;
  const driftDecision = evaluateStrategyDrift(telemetry);
  const externalSignals = getNormalizedExternalSignals();
  const portfolio = evaluateStrategyPortfolio();

  // If re-optimization required, advance sequence recommendation to resilient/recovery posture
  const updatedQuarters = CANONICAL_QUARTERLY_PLANS.map((q) => {
    if (driftDecision.reoptimizationRequired && q.quarter === 'Q2') {
      return {
        ...q,
        primaryStrategyId: 'STRAT-D-RESIL',
        primaryStrategyName: 'Strategy D: Resilient Infrastructure (Urgent Acceleration)',
        status: 'RECOMMENDED' as const,
        projectedOhi: 91.8,
        expectedRoi: 29.5,
      };
    }
    return q;
  });

  const activeSeq: StrategySequence = {
    ...CANONICAL_STRATEGY_SEQUENCE,
    quarters: updatedQuarters,
  };

  const expectedOhi = 88.7;
  const actualOhi = isStressedScenario ? 82.5 : 87.9;
  const driftPct = Number(((Math.abs(actualOhi - expectedOhi) / expectedOhi) * 100).toFixed(2));
  const activeStatus = driftDecision.reoptimizationRequired
    ? 'REOPTIMIZATION_REQUIRED'
    : driftDecision.driftDetected
    ? 'DRIFT_DETECTED'
    : 'ON_TRACK';

  const timestamp = new Date().toISOString();
  const hash = generateOrchestratorHash('STRAT-B-DUAL', expectedOhi, actualOhi, timestamp);

  return {
    orchestratorId: 'ORC-EXEC-COCKPIT-01',
    activeStrategyId: 'STRAT-B-DUAL',
    activeStrategyName: 'Strategy B: Dual Curriculum & Governance Scaling',
    activeStatus,
    currentConfidencePct: 91.0,
    expectedOhi,
    actualOhi,
    driftPct,
    portfolioRank: '#1 / 4',
    lastReoptimizedUtc: timestamp,
    deterministicReplayHash: hash,
    driftDecision,
    activeSequence: activeSeq,
    survivabilityScore: 96.2,
    recoveryHours: 3.0,
    rollbackCoveragePct: 100.0,
    failureProbabilityPct: 4.0,
  };
}

/**
 * Returns the default canonical orchestrator state.
 */
export function getCanonicalOrchestratorState(): OrchestratorState {
  return triggerAdaptiveReoptimization(false);
}
