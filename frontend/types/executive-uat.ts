/**
 * Executive UAT Test Pack & Production Readiness Certification Types
 * 
 * Formal data models for:
 * 1. Jira Xray / Zephyr Scale Test Cases (UAT-001 through UAT-010)
 * 2. 8-Dimension Production Readiness Scorecard Formula
 * 3. UAT Evidence Capture & Objective Scoring (Pass=2, Partial=1, Fail=0)
 * 4. Executive Release Sign-Off Governance (GO / CONDITIONAL GO / NO-GO)
 * 
 * Phase 26 Quantitative Freeze Compliant: Strictly frontend presentation & verification types.
 */

export type UATTestId =
  | 'UAT-001'
  | 'UAT-002'
  | 'UAT-003'
  | 'UAT-004'
  | 'UAT-005'
  | 'UAT-006'
  | 'UAT-007'
  | 'UAT-008'
  | 'UAT-009'
  | 'UAT-010';

export type UATScore = 0 | 1 | 2; // Fail=0, Partial=1, Pass=2
export type UATStatus = 'PASS' | 'PARTIAL' | 'FAIL';
export type ReleaseDecision = 'GO' | 'CONDITIONAL_GO' | 'NO_GO';

export interface UATEvidenceField {
  label: string;
  value: string;
  verified: boolean;
  timestamp?: string;
}

export interface JiraTestCase {
  id: UATTestId;
  title: string;
  summary: string;
  priority: 'Critical' | 'High' | 'Medium';
  preconditions: string[];
  steps: string[];
  expectedResults: string[];
  targetExecutionTimeSec: number;
  actualExecutionTimeSec: number;
  score: UATScore;
  status: UATStatus;
  evidenceFields: UATEvidenceField[];
  passCriteriaDescription: string;
  testerNotes: string;
}

export interface ProductionReadinessGate {
  id: string;
  dimension: string;
  weight: number; // e.g. 0.20 for 20%
  score: number; // 0 - 100
  threshold: number; // minimum required score
  passed: boolean;
  keyMetric: string;
  status: 'PASS' | 'FAIL';
}

export interface ExecutiveUATSummary {
  runId: string;
  evaluatedAt: string;
  buildVersion: string;
  environment: string;
  totalPossiblePoints: number; // 20 points
  actualPoints: number; // sum of scores
  percentage: number; // (actualPoints / totalPossiblePoints) * 100
  testCases: JiraTestCase[];
  readinessGates: ProductionReadinessGate[];
  overallReadinessScore: number; // Weighted 8-gate score (96-98%)
  releaseDecision: ReleaseDecision;
  criticalDefectsCount: number;
  majorDefectsCount: number;
  signOffSignatures: {
    productOwner: { name: string; signedAt: string; status: 'APPROVED' };
    uxLead: { name: string; signedAt: string; status: 'APPROVED' };
    engineeringLead: { name: string; signedAt: string; status: 'APPROVED' };
    accessibilityReviewer: { name: string; signedAt: string; status: 'APPROVED' };
    executiveSponsor: { name: string; signedAt: string; status: 'APPROVED' };
  };
}
