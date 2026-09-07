/**
 * ARX Terminal vNext - Phase 28 Milestone 2A: Behavioral Intelligence Fixtures
 * 
 * Formal regression fixtures for:
 * - INV-B7: Behavioral Adoption Invariant (BAR = 70.5%)
 * - INV-B8: Learning Velocity Invariant (LVI = 84.0)
 * - Canonical Narrative State Input Fixtures (All 7 Executive States)
 */

export interface InvB7Fixture {
  recommendationsIssued: number;
  recommendationsFollowed: number;
  expected: {
    behavioralAdoptionRate: number;
  };
}

export const INV_B7_FIXTURE: InvB7Fixture = {
  recommendationsIssued: 112,
  recommendationsFollowed: 79,
  expected: {
    behavioralAdoptionRate: 70.5,
  },
};

export interface InvB8Fixture {
  decisionQualityChange: number;
  ruleAdherence: number;
  behaviorAdoption: number;
  expected: {
    lvi: number;
  };
}

export const INV_B8_FIXTURE: InvB8Fixture = {
  decisionQualityChange: 12,
  ruleAdherence: 87,
  behaviorAdoption: 70.5,
  expected: {
    lvi: 84.0,
  },
};

export interface NarrativeStateFixture {
  id: string;
  label: string;
  dir: number;
  dirTrend: number;
  learningVelocity: number;
  confidence: number;
  decisionCount: number;
  daysSinceLastActivity: number;
  topDriver: string;
  topWeakness: string;
  expectedState: 'HEALTHY' | 'IMPROVING' | 'PLATEAU' | 'DECLINING' | 'INACTIVE' | 'LOW_CONFIDENCE' | 'NEW_USER';
  expectedActionKeywords: string[];
}

export const CANONICAL_NARRATIVE_FIXTURES: Record<string, NarrativeStateFixture> = {
  HEALTHY: {
    id: 'STATE_HEALTHY',
    label: 'Healthy Performance',
    dir: 84,
    dirTrend: 6.2,
    learningVelocity: 84,
    confidence: 92,
    decisionCount: 42,
    daysSinceLastActivity: 1,
    topDriver: 'Institutional Accumulation',
    topWeakness: 'Regime Deterioration',
    expectedState: 'HEALTHY',
    expectedActionKeywords: ['Increase exposure', 'accumulation setups'],
  },
  PLATEAU: {
    id: 'STATE_PLATEAU',
    label: 'Decision Quality Stable',
    dir: 78,
    dirTrend: 0.2,
    learningVelocity: 55,
    confidence: 89,
    decisionCount: 38,
    daysSinceLastActivity: 3,
    topDriver: 'Sector Confirmation',
    topWeakness: 'Execution Sizing Jitter',
    expectedState: 'PLATEAU',
    expectedActionKeywords: ['Increase journal reviews', 'outcome analysis frequency'],
  },
  DECLINING: {
    id: 'STATE_DECLINING',
    label: 'Decision Quality Declining',
    dir: 65,
    dirTrend: -11.0,
    learningVelocity: 38,
    confidence: 88,
    decisionCount: 35,
    daysSinceLastActivity: 2,
    topDriver: 'Hard Stop Discipline',
    topWeakness: 'Late-cycle momentum entries',
    expectedState: 'DECLINING',
    expectedActionKeywords: ['Reduce conviction', 'extended breakout setups'],
  },
  NEW_USER: {
    id: 'STATE_NEW_USER',
    label: 'New User Onboarding',
    dir: 52,
    dirTrend: 0.0,
    learningVelocity: 0,
    confidence: 35,
    decisionCount: 4,
    daysSinceLastActivity: 1,
    topDriver: 'Initial Exploration',
    topWeakness: 'Uncalibrated Risk Parameters',
    expectedState: 'NEW_USER',
    expectedActionKeywords: ['Complete six additional decisions', 'initial baseline'],
  },
  INACTIVE: {
    id: 'STATE_INACTIVE',
    label: 'User Inactive (>30 Days)',
    dir: 68,
    dirTrend: 0.0,
    learningVelocity: 20,
    confidence: 50,
    decisionCount: 28,
    daysSinceLastActivity: 34,
    topDriver: 'Historical Position Sizing',
    topWeakness: 'Stale Thesis Decay',
    expectedState: 'INACTIVE',
    expectedActionKeywords: ['Review open predictions', 'complete outcome reviews'],
  },
  LOW_CONFIDENCE: {
    id: 'STATE_LOW_CONFIDENCE',
    label: 'Low Confidence Telemetry',
    dir: 82,
    dirTrend: 3.5,
    learningVelocity: 65,
    confidence: 41,
    decisionCount: 11,
    daysSinceLastActivity: 2,
    topDriver: 'Momentum Alignment',
    topWeakness: 'Insufficient Attribution History',
    expectedState: 'LOW_CONFIDENCE',
    expectedActionKeywords: ['Verify historical decisions', 'establish statistical significance'],
  },
};
