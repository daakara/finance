/**
 * Phase 31-M3: Learning Velocity Engine (Epic M3-102 / INV-OI17)
 *
 * Implements:
 * - Quarterly & Annualized Learning Velocity: LV = delta ODEI / delta t
 * - INV-OI17 Certification: LearningVelocity > 0.0 (Strictly Positive)
 * - Stagnation (LV = 0) & Degradation (LV < 0) Detection with Alerts
 * - Learning Attribution Linkage (100% Traceability, AC-OI17-04/06)
 * - Predictive Learning Velocity Forecasting
 * - Deterministic SHA-256 Replay Hash (AC-OI17-05)
 */

import type {
  LearningVelocityResult,
  VelocityStatus,
} from '../../types/learning-intelligence';
import type { LearningVelocityMetrics, LearningVelocity } from '../../types/behavioral-intelligence';

import { sha256 } from '../governance/sha256';
import { getAllLearnings, getAllAdoptions } from './learningIntelligenceEngine';

export const COMMITTEE_HISTORICAL_ODEI: Record<string, { baseline: number; current: number; quarters: number }> = {
  'COM-001': { baseline: 81.2, current: 85.0, quarters: 1.0 }, // +3.8 / 1 = +3.8
  'COM-002': { baseline: 80.0, current: 83.0, quarters: 1.0 }, // +3.0 / 1 = +3.0
  'COM-003': { baseline: 82.5, current: 87.0, quarters: 1.0 }, // +4.5 / 1 = +4.5
};

export const QUARTERLY_ODEI_SERIES: Record<string, { quarter: string; odei: number }[]> = {
  'COM-001': [
    { quarter: '2025-Q3', odei: 78.4 },
    { quarter: '2025-Q4', odei: 80.0 },
    { quarter: '2026-Q1', odei: 82.5 },
    { quarter: '2026-Q2', odei: 85.0 },
  ],
  'COM-002': [
    { quarter: '2025-Q3', odei: 77.0 },
    { quarter: '2025-Q4', odei: 79.5 },
    { quarter: '2026-Q1', odei: 81.2 },
    { quarter: '2026-Q2', odei: 83.0 },
  ],
  'COM-003': [
    { quarter: '2025-Q3', odei: 79.0 },
    { quarter: '2025-Q4', odei: 81.5 },
    { quarter: '2026-Q1', odei: 84.2 },
    { quarter: '2026-Q2', odei: 87.0 },
  ],
};

/**
 * Calculates Team Learning Velocity (INV-OI17):
 * LV = (currentODEI - baselineODEI) / elapsedQuarters
 */
export function computeLearningVelocity(
  committeeId: string = 'COM-001',
  baselineODEI?: number,
  currentODEI?: number,
  elapsedQuarters: number = 1.0
): LearningVelocityResult {
  const defaultHistory = COMMITTEE_HISTORICAL_ODEI[committeeId] ?? { baseline: 80.0, current: 84.0, quarters: 1.0 };
  const base = baselineODEI ?? defaultHistory.baseline;
  const curr = currentODEI ?? defaultHistory.current;
  const quarters = elapsedQuarters > 0 ? elapsedQuarters : 1.0;

  const delta = curr - base;
  const rawVelocity = elapsedQuarters <= 0 ? 0 : delta / quarters;
  const velocity = Math.round((rawVelocity + (rawVelocity >= 0 ? 1e-9 : -1e-9)) * 10) / 10;

  // Determine status
  const status: VelocityStatus =
    velocity > 0.0 ? 'POSITIVE' : velocity === 0.0 ? 'STAGNANT' : 'DEGRADING';

  // Invariant INV-OI17: LearningVelocity strictly > 0.0
  const invariantSatisfied = velocity > 0.0;

  // Annualized velocity (quarters * 4)
  const annualizedVelocity = Math.round(velocity * 4 * 10) / 10;

  // Forecast next quarter (dampened momentum)
  const forecastNextQuarter = Math.round((curr + velocity * 0.85) * 10) / 10;

  // Attributable learnings from this committee
  const adoptions = getAllAdoptions().filter(
    a => a.targetCommitteeId === committeeId && a.adoptionStatus === 'ADOPTED'
  );
  const attributableLearnings = adoptions.map(a => a.learningId);

  // Historical quarterly breakdown
  const series = QUARTERLY_ODEI_SERIES[committeeId] ?? QUARTERLY_ODEI_SERIES['COM-001'];
  const historicalQuarterlyVelocities = [];
  for (let i = 1; i < series.length; i++) {
    const qVel = Math.round((series[i].odei - series[i - 1].odei) * 10) / 10;
    historicalQuarterlyVelocities.push({
      quarter: series[i].quarter,
      velocity: qVel,
      odei: series[i].odei,
    });
  }

  return {
    committeeId,
    baselineODEI: base,
    currentODEI: curr,
    elapsedQuarters: quarters,
    velocity,
    status,
    annualizedVelocity,
    forecastNextQuarter,
    invariantSatisfied,
    attributableLearnings,
    historicalQuarterlyVelocities,
  };
}

/**
 * Validates INV-OI17 certification.
 */
export function verifyINV_OI17(
  committeeId: string,
  baseline?: number,
  current?: number,
  quarters: number = 1.0
): {
  valid: boolean;
  committeeId: string;
  velocity: number;
  status: VelocityStatus;
  alertCode?: 'LEARNING_VELOCITY_NON_POSITIVE';
  message: string;
} {
  const res = computeLearningVelocity(committeeId, baseline, current, quarters);
  if (!res.invariantSatisfied) {
    return {
      valid: false,
      committeeId,
      velocity: res.velocity,
      status: res.status,
      alertCode: 'LEARNING_VELOCITY_NON_POSITIVE',
      message: `INV-OI17 VIOLATION: Committee ${committeeId} learning velocity is ${res.velocity} (status: ${res.status}). Positive quarterly velocity required.`,
    };
  }

  return {
    valid: true,
    committeeId,
    velocity: res.velocity,
    status: res.status,
    message: `INV-OI17 PASSED: Committee ${committeeId} learning velocity is +${res.velocity} (status: POSITIVE).`,
  };
}

/**
 * Computes velocity trend across multiple quarters.
 */
export function computeVelocityTrend(committeeId: string = 'COM-001'): {
  committeeId: string;
  averageVelocity: number;
  momentum: 'ACCELERATING' | 'DECELERATING' | 'STEADY';
  acceleration: number;
} {
  const res = computeLearningVelocity(committeeId);
  const vels = res.historicalQuarterlyVelocities.map(h => h.velocity);
  const avg = vels.length > 0 ? Math.round((vels.reduce((a, b) => a + b, 0) / vels.length) * 10) / 10 : res.velocity;

  let acceleration = 0;
  if (vels.length >= 2) {
    acceleration = Math.round((vels[vels.length - 1] - vels[vels.length - 2]) * 10) / 10;
  }

  const momentum = acceleration > 0.2 ? 'ACCELERATING' : acceleration < -0.2 ? 'DECELERATING' : 'STEADY';

  return {
    committeeId,
    averageVelocity: avg,
    momentum,
    acceleration,
  };
}

/**
 * Computes deterministic replay hash of velocity output.
 * Satisfies AC-OI17-05 (identical hash across 100 consecutive replays).
 */
export function hashVelocityResult(res: LearningVelocityResult): string {
  const payload = {
    committeeId: res.committeeId,
    baselineODEI: res.baselineODEI,
    currentODEI: res.currentODEI,
    elapsedQuarters: res.elapsedQuarters,
    velocity: res.velocity,
    status: res.status,
    invariantSatisfied: res.invariantSatisfied,
    attributableLearnings: [...res.attributableLearnings].sort(),
  };
  return sha256(JSON.stringify(payload));
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 28 Behavioral Learning Velocity & Maturity Index Exports
// ─────────────────────────────────────────────────────────────────────────────

export interface LearningVelocityInputs {
  dirTrend?: number;
  outcomeReviews: number;
  learningCoachUsage: number;
  decisionJournalActivity: number;
  recommendationAdoption: number;
  historicalGrowthPeriods?: number[];
  confidence?: number;
}

export interface GrowthPeriod {
  period: string;
  delta: number;
  score: number;
}

export function computeRawLVI(
  qualityGrowth: number,
  adoption: number,
  adherence: number
): number {
  return Math.round(((qualityGrowth * 0.5) + (adoption * 0.3) + (adherence * 0.2)) * 100) / 100;
}

export function computeLearningVelocityIndex(
  qualityGrowth: number = 12,
  adoption: number = 70.5,
  adherence: number = 87.0
): number {
  const raw = computeRawLVI(qualityGrowth, adoption, adherence);
  const normalized = Math.min(100, Math.round((raw / 53.0) * 100));
  return normalized;
}

export function computeBehavioralMaturityIndex(
  decisionQuality: number = 74,
  bar: number = 70.5,
  ruleAdherence: number = 87.0,
  lvi: number = 84,
  driftScore: number = 21.0
): number {
  const driftControl = Math.max(0, 100 - driftScore);
  const bmi = (0.25 * decisionQuality) +
    (0.25 * bar) +
    (0.20 * ruleAdherence) +
    (0.15 * lvi) +
    (0.15 * driftControl);
  return Math.round(bmi * 10) / 10;
}

export function computeImprovementMomentum(
  currentQuarterDelta: number = 6,
  prevQuarterDelta: number = 4
): { multiplier: number; status: 'ACCELERATING' | 'STABLE' | 'SLOWING' } {
  if (prevQuarterDelta <= 0) {
    return { multiplier: 1.0, status: 'STABLE' };
  }
  const multiplier = Math.round((currentQuarterDelta / prevQuarterDelta) * 100) / 100;
  let status: 'ACCELERATING' | 'STABLE' | 'SLOWING' = 'STABLE';
  if (multiplier > 1.05) status = 'ACCELERATING';
  else if (multiplier < 0.95) status = 'SLOWING';

  return { multiplier, status };
}

export function classifyLVI(score: number): 'LOW' | 'MODERATE' | 'HIGH' | 'ELITE' {
  if (score >= 86) return 'ELITE';
  if (score >= 71) return 'HIGH';
  if (score >= 41) return 'MODERATE';
  return 'LOW';
}

export interface EvaluatedLearningVelocity {
  score: number;
  direction: 'ACCELERATING' | 'IMPROVING' | 'STABLE' | 'PLATEAU' | 'REGRESSING';
  acceleration: number;
  confidence: number;
  category: 'LOW' | 'MEDIUM' | 'HIGH' | 'ELITE';
  confidenceInterval: {
    lower: number;
    upper: number;
  };
}

export function evaluateLearningVelocity(inputs: LearningVelocityInputs): EvaluatedLearningVelocity {
  const dirTrend = inputs.dirTrend ?? 3.2;
  const outcomeReviews = Math.max(0, Math.min(100, inputs.outcomeReviews));
  const learningCoachUsage = Math.max(0, Math.min(100, inputs.learningCoachUsage));
  const decisionJournalActivity = Math.max(0, Math.min(100, inputs.decisionJournalActivity));
  const recommendationAdoption = Math.max(0, Math.min(100, inputs.recommendationAdoption));

  const compositeScore = Math.round(
    (0.30 * recommendationAdoption +
      0.25 * learningCoachUsage +
      0.25 * outcomeReviews +
      0.20 * decisionJournalActivity) * 10
  ) / 10;

  const periods = inputs.historicalGrowthPeriods ?? [2, 4, 6];
  let acceleration = 1.0;
  if (periods.length >= 2) {
    const latest = periods[periods.length - 1];
    const previous = periods[periods.length - 2];
    acceleration = previous !== 0 ? Math.round((latest / previous) * 100) / 100 : 1.0;
  }

  let direction: 'ACCELERATING' | 'IMPROVING' | 'STABLE' | 'PLATEAU' | 'REGRESSING' = 'STABLE';

  if (dirTrend < -1.0 || compositeScore < 45) {
    direction = 'REGRESSING';
  } else if (Math.abs(dirTrend) <= 0.5 && acceleration <= 1.02 && acceleration >= 0.98) {
    direction = 'PLATEAU';
  } else if (acceleration > 1.25 && dirTrend > 2.0) {
    direction = 'ACCELERATING';
  } else if (dirTrend > 0.5) {
    direction = 'IMPROVING';
  } else {
    direction = 'STABLE';
  }

  const confidence = inputs.confidence ?? 89;
  const margin = Math.round((1.96 * Math.sqrt((compositeScore * (100 - compositeScore)) / 100) / 10) * 10) / 10;
  const lower = Math.max(0, Math.round((compositeScore - margin) * 10) / 10);
  const upper = Math.min(100, Math.round((compositeScore + margin) * 10) / 10);

  let category: 'LOW' | 'MEDIUM' | 'HIGH' | 'ELITE' = 'MEDIUM';
  if (compositeScore >= 86) category = 'ELITE';
  else if (compositeScore >= 71) category = 'HIGH';
  else if (compositeScore >= 45) category = 'MEDIUM';
  else category = 'LOW';

  return {
    score: compositeScore,
    direction,
    acceleration,
    confidence,
    category,
    confidenceInterval: {
      lower,
      upper,
    },
  };
}

export function sortGrowthPeriods(periods: GrowthPeriod[]): GrowthPeriod[] {
  return [...periods].sort((a, b) => a.period.localeCompare(b.period));
}

export function computeQoQProgression(qCurrent: number, qPrevious: number): {
  delta: number;
  percentageGain: number;
  isImproving: boolean;
} {
  const delta = Math.round((qCurrent - qPrevious) * 10) / 10;
  const percentageGain = qPrevious !== 0 ? Math.round((delta / qPrevious) * 1000) / 10 : 0;
  return {
    delta,
    percentageGain,
    isImproving: delta > 0,
  };
}

export function computeAnnualGrowth(initialScore: number, finalScore: number): {
  annualDelta: number;
  compoundedAnnualRate: number;
} {
  const annualDelta = Math.round((finalScore - initialScore) * 10) / 10;
  const compoundedAnnualRate = initialScore !== 0 ? Math.round((annualDelta / initialScore) * 1000) / 10 : 0;
  return {
    annualDelta,
    compoundedAnnualRate,
  };
}

export const CANONICAL_LEARNING_VELOCITY: LearningVelocityMetrics = {
  velocityIndex: computeLearningVelocityIndex(12, 70.5, 87.0),
  qualityImprovementRate: 12.0,
  recommendationAdoptionRate: 70.5,
  playbookAdherenceRate: 87.0,
  projectedMonthsToGoal: 4.0,
  momentumMultiplier: computeImprovementMomentum(6, 4).multiplier,
  velocityTier: 'HIGH',
};
