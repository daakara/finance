/**
 * Sprint 8.5: UX Foundations Program Type Definitions
 * 
 * Defines standard contracts for:
 * 1. 7-Stage Decision Lifecycle State Model
 * 2. Personal Decision Identity Profile
 * 3. Unified Intelligence Cards (Insight, Recommendation, Learning, Evidence)
 * 4. Contextual ARX Mentor Framework (5-Stage Cognitive Format)
 * 5. Production UX Audit & Readiness Scorecard
 * 
 * Phase 26 Quantitative Freeze Compliant: Strictly frontend presentation & interaction types.
 */

export type DecisionLifecycleState =
  | 'OBSERVED'
  | 'PREDICTED'
  | 'APPROVED'
  | 'EXECUTING'
  | 'RESOLVED'
  | 'LEARNED'
  | 'PLAYBOOK_UPDATED';

export interface LifecycleStep {
  id: string;
  state: DecisionLifecycleState;
  label: string;
  timestamp: string;
  status: 'COMPLETED' | 'ACTIVE' | 'PENDING' | 'SKIPPED';
  actor: string;
  details: string;
  durationMs?: number;
}

export interface DecisionProfile {
  userId: string;
  userName: string;
  role: string;
  qualityScore: number; // e.g. 74
  qualityScoreDelta: number; // +12 pts from baseline 62
  decileRank: string; // e.g. "Top 18%"
  primaryEdge: string; // e.g. "Institutional Accumulation"
  primaryWeakness: string; // e.g. "Late-Day Momentum Chases"
  activeRulesCount: number; // e.g. 12
  completedDecisionsCount: number; // e.g. 142
  scoreHistory: Array<{
    date: string;
    score: number;
    milestone?: string;
  }>;
}

export type IntelligenceCardVariant = 'INSIGHT' | 'RECOMMENDATION' | 'LEARNING' | 'EVIDENCE';

export type RecommendationCategory = 'DO_MORE' | 'STOP_DOING' | 'CALIBRATE';

export interface StandardInsightData {
  id: string;
  ticker?: string;
  headline: string;
  observation: string;
  category: 'FLOW' | 'REGIME' | 'MOMENTUM' | 'VOLATILITY' | 'VALUATION' | 'BEHAVIOR';
  confidence: number; // 0 - 100
  metricDelta?: {
    label: string;
    value: string;
    isPositive: boolean;
  };
  timestamp: string;
  source: string;
}

export interface StandardRecommendationData {
  id: string;
  ticker?: string;
  action: string;
  category: RecommendationCategory;
  rationale: string;
  projectedImpact: string; // e.g. "+3.4 pts"
  urgency: 'HIGH' | 'MEDIUM' | 'LOW';
  confidence: number;
  primaryActionLabel: string;
  secondaryActionLabel?: string;
}

export interface StandardLearningData {
  id: string;
  ticker?: string;
  title: string;
  takeaway: string;
  causalFactor: string;
  winRateImpact: string; // e.g. "+14.2%"
  errorElimination: string; // e.g. "-38% drawdown"
  ruleRefinementProposal: string;
  statisticalStrength: 'VERY_HIGH' | 'HIGH' | 'MEDIUM';
}

export interface StandardEvidenceData {
  id: string;
  claim: string;
  sampleSize: number; // N = 124
  pValue: number; // p < 0.001
  sharpeImpact: string; // "+0.42"
  lookbackPeriod: string; // "180 Days"
  ledgerHash: string; // SHA-256
  verifiedAt: string;
}

export type MentorContext =
  | 'ATTENTION'
  | 'DECISION'
  | 'ATTRIBUTION'
  | 'LEARNING'
  | 'PLAYBOOK'
  | 'GOVERNANCE';

export interface MentorInsight {
  context: MentorContext;
  roleName: string; // e.g. "Attention Coach", "Decision Coach"
  observation: string; // What happened / What is seen
  understanding: string; // Systemic context / Why it matters
  recommendation: string; // Action to take
  confidence: number; // 0 - 100
  justification: string; // Why this recommendation was derived
  projectedImpact: string; // e.g. "+3.4 pts"
  evidenceHash: string;
  sampleSize: number;
  pValue: number;
  tags: string[];
}

export interface ScorecardCriterion {
  id: string;
  name: string;
  description: string;
  score: number; // 0 - 100
  weight: number; // decimal fraction within category
  passed: boolean;
  benchmark: string;
}

export interface ScorecardCategory {
  id: string;
  name: string;
  weight: number; // Category weight, e.g. 0.20 for 20%
  score: number; // Calculated weighted score for category 0-100
  criteria: ScorecardCriterion[];
}

export interface ReadinessScorecard {
  id: string;
  title: string;
  overallScore: number; // 0 - 100
  targetScore: number; // 90.0
  verdict: 'PRODUCTION_READY' | 'CONDITIONAL_APPROVAL' | 'BLOCKED';
  evaluatedAt: string;
  evaluatorRole: string;
  categories: ScorecardCategory[];
  blockingIssuesCount: number;
  certifiedGatesCount: number;
  totalGatesCount: number;
}
