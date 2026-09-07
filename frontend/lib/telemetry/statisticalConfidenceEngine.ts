/**
 * Statistical Confidence & Cohort Analytics Engine
 * 
 * Implements:
 * 1. 95% Confidence Intervals (Wilson score & Normal SE approximations)
 * 2. 5-Tier Trend Velocity Classification (Rapid, Improving, Stable, Declining, Critical)
 * 3. Behavioral Maturity Tiers (Consumers, Investigators, Practitioners, Learners, Optimizers)
 * 4. Role-based Cohort Analytics (Executives, PMs, Analysts, New Users)
 * 
 * Phase 26 Quantitative Freeze Compliant: Strictly frontend presentation statistics.
 */

import {
  ConfidenceInterval,
  TrendVelocity,
  MetricWithConfidence,
  MaturityTierDistribution,
  RoleCohortMetric,
} from '@/types/behavioral-intelligence';

export function computeWilsonConfidenceInterval(
  successes: number,
  trials: number,
  confidenceLevel: number = 0.95
): ConfidenceInterval {
  if (trials <= 0) {
    return {
      pointEstimate: 0,
      lowerBound: 0,
      upperBound: 0,
      marginOfError: 0,
      confidenceLevel,
      displayString: '0.0% (0.0% - 0.0%)',
    };
  }

  const z = 1.96; // 95% confidence z-score
  const p = successes / trials;
  const z2 = z * z;
  const denominator = 1 + z2 / trials;
  const center = (p + z2 / (2 * trials)) / denominator;
  const spread = (z * Math.sqrt((p * (1 - p)) / trials + z2 / (4 * trials * trials))) / denominator;

  const lower = Math.max(0, Math.round((center - spread) * 1000) / 10);
  const upper = Math.min(100, Math.round((center + spread) * 1000) / 10);
  const point = Math.round(p * 1000) / 10;
  const margin = Math.round(((upper - lower) / 2) * 10) / 10;

  return {
    pointEstimate: point,
    lowerBound: lower,
    upperBound: upper,
    marginOfError: margin,
    confidenceLevel,
    displayString: `${point.toFixed(1)}% (${lower.toFixed(1)}% - ${upper.toFixed(1)}%)`,
  };
}

export function classifyTrendVelocity(deltaPercent: number): TrendVelocity {
  if (deltaPercent >= 10.0) return 'RAPID_IMPROVEMENT';
  if (deltaPercent >= 3.0) return 'IMPROVING';
  if (deltaPercent <= -10.0) return 'CRITICAL_DECLINE';
  if (deltaPercent <= -3.0) return 'DECLINING';
  return 'STABLE';
}

export const CANONICAL_CONFIDENCE_METRICS: Record<string, MetricWithConfidence> = {
  mentorEngagement: {
    name: 'Mentor Engagement',
    value: 67.4,
    unit: '%',
    ci: {
      pointEstimate: 67.4,
      lowerBound: 64.2,
      upperBound: 70.6,
      marginOfError: 3.2,
      confidenceLevel: 0.95,
      displayString: '67.4% (64.2% - 70.6%)',
    },
    trend: 'IMPROVING',
    delta7d: 3.4,
    delta30d: 7.2,
    target: 60.0,
    isPassing: true,
  },
  behavioralAdoption: {
    name: 'Behavioral Adoption Rate (BAR)',
    value: 70.5,
    unit: '%',
    ci: {
      pointEstimate: 70.5,
      lowerBound: 68.1,
      upperBound: 72.7,
      marginOfError: 2.3,
      confidenceLevel: 0.95,
      displayString: '70.5% (68.1% - 72.7%)',
    },
    trend: 'IMPROVING',
    delta7d: 4.2,
    delta30d: 8.5,
    target: 70.0,
    isPassing: true,
  },
  ruleAdherence: {
    name: 'Rule Adherence',
    value: 87.0,
    unit: '%',
    ci: {
      pointEstimate: 87.0,
      lowerBound: 84.8,
      upperBound: 89.0,
      marginOfError: 2.1,
      confidenceLevel: 0.95,
      displayString: '87.0% (84.8% - 89.0%)',
    },
    trend: 'RAPID_IMPROVEMENT',
    delta7d: 5.1,
    delta30d: 11.2,
    target: 80.0,
    isPassing: true,
  },
};

export const MATURITY_TIERS: MaturityTierDistribution[] = [
  {
    tier: 'CONSUMER',
    label: 'Consumers (Level 1)',
    scoreRange: '0 - 20',
    userPercentage: 18,
    description: 'Read recommendations passively; rarely record actions or post-mortems.',
    primaryAction: 'Read & Monitor',
  },
  {
    tier: 'INVESTIGATOR',
    label: 'Investigators (Level 2)',
    scoreRange: '21 - 40',
    userPercentage: 22,
    description: 'Inspect supporting evidence and attribution traces before executing.',
    primaryAction: 'Explore Evidence',
  },
  {
    tier: 'PRACTITIONER',
    label: 'Practitioners (Level 3)',
    scoreRange: '41 - 60',
    userPercentage: 29,
    description: 'Regularly follow AI recommendations with consistent trade journaling.',
    primaryAction: 'Execute Playbook',
  },
  {
    tier: 'LEARNER',
    label: 'Learners (Level 4)',
    scoreRange: '61 - 80',
    userPercentage: 20,
    description: 'Continuously consult AI Coach and refine personal decision boundaries.',
    primaryAction: 'Coach & Adapt',
  },
  {
    tier: 'OPTIMIZER',
    label: 'Optimizers (Level 5)',
    scoreRange: '81 - 100',
    userPercentage: 11,
    description: 'Achieve high adherence, minimal drift (<15%), and institutional consistency.',
    primaryAction: 'Govern & Scale',
  },
];

export const ROLE_COHORTS: RoleCohortMetric[] = [
  {
    role: 'PORTFOLIO_MANAGERS',
    label: 'Portfolio Managers',
    adoptionRate: 82.0,
    decisionQuality: 78,
    userCount: 850,
  },
  {
    role: 'EXECUTIVES',
    label: 'Executives / CIO Desk',
    adoptionRate: 76.0,
    decisionQuality: 75,
    userCount: 420,
  },
  {
    role: 'ANALYSTS',
    label: 'Quantitative Analysts',
    adoptionRate: 61.0,
    decisionQuality: 68,
    userCount: 1420,
  },
  {
    role: 'NEW_USERS',
    label: 'New Users (<30d)',
    adoptionRate: 49.0,
    decisionQuality: 62,
    userCount: 1528,
  },
];
