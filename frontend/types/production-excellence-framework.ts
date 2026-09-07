/**
 * Production Excellence Framework Type Definitions
 * 
 * Defines contracts for:
 * 1. Telemetry Data Quality & Completeness Framework (Invariants TQ-1 to TQ-5)
 * 2. Behavioral Cohort Analysis & Learning Maturity Index (LMI)
 * 3. 30-Day Executive Production Excellence Review
 * 
 * Phase 26 Quantitative Freeze Compliant: Strictly frontend presentation & telemetry contracts.
 */

export interface TelemetryQualityInvariant {
  id: 'TQ-1' | 'TQ-2' | 'TQ-3' | 'TQ-4' | 'TQ-5';
  name: string;
  description: string;
  target: string;
  actual: string;
  compliancePct: number;
  status: 'PASS' | 'AT_RISK' | 'FAIL';
  evaluatedEventsCount: number;
  unbrokenAuditChain: boolean;
}

export interface TelemetryDataQualityKPIs {
  telemetryHealth: number; // 98.9%
  eventCompleteness: number; // 99.7% (Target >= 99.5%)
  attributionCoverage: number; // 100.0% (Target 100%)
  eventQuality: number; // 99.8%
  journeyReconstructability: number; // 97.2% (Target >= 95%)
  evaluatedAt: string;
}

export type DataQualityAlertSeverity = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';

export interface DataQualityAlert {
  id: string;
  severity: DataQualityAlertSeverity;
  threshold: string;
  description: string;
  escalationPolicy: string;
  active: boolean;
  timestampUtc: string;
}

export interface TimeCohort {
  id: 'COHORT_A' | 'COHORT_B' | 'COHORT_C' | 'COHORT_D';
  label: string;
  tenureRange: string; // e.g. "0-30 days"
  userCount: number;
  avgLmiScore: number;
  avgDecisionScore: number;
  churnRisk: 'LOW' | 'MEDIUM' | 'HIGH';
}

export interface DecisionMaturityCohort {
  level: 1 | 2 | 3 | 4 | 5;
  title: string;
  behaviorProfile: string;
  userPercentage: number;
  avgWeeklySessions: number;
  recommendationAdoptionRate: number;
  dominantAction: string;
}

export interface LearningMaturityIndexInputs {
  outcomeReviews: number; // 0 - 100
  aiCoachingEngagement: number; // 0 - 100
  decisionJournalUsage: number; // 0 - 100
  recommendationAcceptance: number; // 0 - 100
}

export interface LearningMaturityIndexResult {
  compositeScore: number; // 0 - 100
  weights: {
    outcomeReviews: 0.30;
    aiCoachingEngagement: 0.25;
    decisionJournalUsage: 0.25;
    recommendationAcceptance: 0.20;
  };
  classification: 'Novice' | 'Developing' | 'Competent' | 'Advanced' | 'Institutional Master';
}

export interface BehavioralImprovementBreakdown {
  risingUsersPct: number; // 64%
  plateauUsersPct: number; // 28%
  regressingUsersPct: number; // 8%
  avgDecisionQualityScore: number; // 74
  decisionQualityDeltaQoQ: number; // +6 points
  topPerformingDriver: {
    name: string;
    winRate: number; // 72%
  };
  topFailureDriver: {
    name: string;
    failureAttribution: number; // 42%
  };
}

export interface Day30ExecutiveReviewData {
  reviewPeriod: string;
  evaluatedBuild: string;
  productionStatus: 'Institutional Production Ready' | 'Production Excellence' | 'Release Candidate';
  productionExcellenceScore: number; // 97/100
  totalActiveUsers: number; // 4,218
  dailyActiveUsers: number; // 2,041
  weeklyRetentionPct: number; // 87%
  aiCoachAdoptionPct: number; // 68%
  predictionActionabilityRate: number; // 62.4% (Target >= 50%)
  outcomeResolutionCoverage: number; // 100% (Target 100%)
  learningVelocityPct: number; // +8.2% QoQ
  committeeDecisionsCount: number; // 1,142
  auditVerifiabilityPct: number; // 100%
  certifiedBy: {
    cio: string;
    productSteeringCommittee: string;
    governanceBoard: string;
    chiefSystemsArchitect: string;
  };
}
