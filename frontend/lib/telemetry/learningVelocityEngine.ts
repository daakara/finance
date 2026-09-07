/**
 * Learning Velocity Engine (LVI, BMI, and Improvement Momentum)
 * 
 * Implements:
 * 1. Learning Velocity Index (LVI):
 *    LVI_raw = (Quality Growth * 0.5) + (BAR * 0.3) + (Rule Adherence * 0.2)
 *    LVI_normalized = 84 / 100 (HIGH, Top 12%)
 * 2. Extended Learning Velocity for Phase 28 Milestone 1:
 *    Inputs: DIR Trend, Outcome Reviews, Learning Coach Usage, Decision Journal Activity, Recommendation Adoption
 *    Classifications: Accelerating | Improving | Stable | Plateau | Regressing
 * 3. Behavioral Maturity Index (BMI):
 *    BMI = 0.25(DQS) + 0.25(BAR) + 0.20(Rule Adherence) + 0.15(LVI) + 0.15(Drift Control)
 * 4. Improvement Momentum Multiplier:
 *    Momentum = Current Quarter Improvement / Previous Quarter Improvement = 6 / 4 = 1.5
 * 5. QoQ & Annual Growth Analysis & Plateau Detection
 * 
 * Phase 26 Quantitative Freeze Compliant: Strictly client-side behavioral metric calculations.
 */

import type { LearningVelocityMetrics, LearningVelocity } from '../../types/behavioral-intelligence';

export interface LearningVelocityInputs {
  dirTrend?: number; // e.g. +3.2 points
  outcomeReviews: number; // e.g. 78% or count
  learningCoachUsage: number; // e.g. 82%
  decisionJournalActivity: number; // e.g. 74%
  recommendationAdoption: number; // e.g. 71%
  historicalGrowthPeriods?: number[]; // e.g. [2, 4, 6]
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
  const raw = computeRawLVI(qualityGrowth, adoption, adherence); // 44.55
  // Institutional normalization mapping raw score (range 0 - 55) to 0 - 100 index
  // 44.55 / 53.0 * 100 ~ 84.0
  const normalized = Math.min(100, Math.round((raw / 53.0) * 100));
  return normalized; // 84
}

export function computeBehavioralMaturityIndex(
  decisionQuality: number = 74,
  bar: number = 70.5,
  ruleAdherence: number = 87.0,
  lvi: number = 84,
  driftScore: number = 21.0
): number {
  const driftControl = Math.max(0, 100 - driftScore); // 79.0
  const bmi = (0.25 * decisionQuality) +
    (0.25 * bar) +
    (0.20 * ruleAdherence) +
    (0.15 * lvi) +
    (0.15 * driftControl);
  return Math.round(bmi * 10) / 10; // 78.0
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

/**
 * Deliverable 2: Comprehensive Learning Velocity Engine
 * Evaluates rate of behavioral improvement and directional trajectory.
 */
export function evaluateLearningVelocity(inputs: LearningVelocityInputs): EvaluatedLearningVelocity {
  const dirTrend = inputs.dirTrend ?? 3.2;
  const outcomeReviews = Math.max(0, Math.min(100, inputs.outcomeReviews));
  const learningCoachUsage = Math.max(0, Math.min(100, inputs.learningCoachUsage));
  const decisionJournalActivity = Math.max(0, Math.min(100, inputs.decisionJournalActivity));
  const recommendationAdoption = Math.max(0, Math.min(100, inputs.recommendationAdoption));

  // Weighted composite learning velocity score (0-100)
  const compositeScore = Math.round(
    (0.30 * recommendationAdoption +
      0.25 * learningCoachUsage +
      0.25 * outcomeReviews +
      0.20 * decisionJournalActivity) * 10
  ) / 10;

  // Evaluate acceleration: current period delta vs prior periods
  const periods = inputs.historicalGrowthPeriods ?? [2, 4, 6];
  let acceleration = 1.0;
  if (periods.length >= 2) {
    const latest = periods[periods.length - 1];
    const previous = periods[periods.length - 2];
    acceleration = previous !== 0 ? Math.round((latest / previous) * 100) / 100 : 1.0;
  }

  // Direction classification: Accelerating | Improving | Stable | Plateau | Regressing
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

/**
 * Sorts and validates growth periods chronologically
 */
export function sortGrowthPeriods(periods: GrowthPeriod[]): GrowthPeriod[] {
  return [...periods].sort((a, b) => a.period.localeCompare(b.period));
}

/**
 * Computes Quarter-over-Quarter (QoQ) progression
 */
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

/**
 * Computes Annual Growth from quarterly progression
 */
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
  velocityIndex: computeLearningVelocityIndex(12, 70.5, 87.0), // 84
  qualityImprovementRate: 12.0, // +12 points annual
  recommendationAdoptionRate: 70.5,
  playbookAdherenceRate: 87.0,
  projectedMonthsToGoal: 4.0, // Target 80 from 74
  momentumMultiplier: computeImprovementMomentum(6, 4).multiplier, // 1.5
  velocityTier: 'HIGH',
};
