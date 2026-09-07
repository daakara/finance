/**
 * Phase 27: Production Adoption & Observability Type Definitions
 * 
 * Target: Transition ARX Terminal from 97% Institutional Production Ready
 * to 99%+ Production Excellence.
 * 
 * Covers:
 * 1. 6-Dimension Production Excellence Review Scorecard (User Adoption, Behavioral Improvement,
 *    Executive Effectiveness, Product Utilization, Operational Excellence, Governance)
 * 2. 30-Day Production Validation Plan (4 Phases: Days 1-7, 8-14, 15-21, 22-30)
 * 3. Executive Journey Path Analysis & Funnel Drop-off Tracking
 * 4. Phase 27 Epics (P27-100 through P27-402)
 * 5. Production Excellence Certification Contract
 * 
 * Phase 26 Quantitative Freeze Compliant: Strictly frontend presentation & verification types.
 */

export interface ScorecardDimension {
  id: string;
  name: string;
  weight: number; // e.g. 0.20 for 20%
  score: number; // 0 - 100
  target: string;
  actualMetric: string;
  status: 'EXCELLENCE' | 'SATISFIED' | 'AT_RISK';
  kpis: Array<{
    name: string;
    target: string;
    actual: string;
    score: number;
    passed: boolean;
  }>;
}

export interface ProductionExcellenceScorecardData {
  evaluatedAt: string;
  cadence: 'Day 30' | 'Day 60' | 'Day 90';
  overallScore: number; // 99.3%
  targetScore: number; // 99.0%
  classification: 'Production Excellence' | 'Institutional Ready' | 'Production Ready' | 'At Risk';
  dimensions: ScorecardDimension[];
  certifiedBy: {
    productOwner: string;
    uxLead: string;
    engineeringLead: string;
    executiveSponsor: string;
  };
}

export interface ValidationPhase {
  phaseNumber: number;
  name: string;
  daysRange: string; // e.g. "Days 1-7"
  goal: string;
  activities: string[];
  successCriteria: string;
  status: 'COMPLETED' | 'ACTIVE' | 'PENDING';
  completionPct: number;
}

export interface JourneyFunnelStep {
  stepNumber: number;
  name: string;
  visitors: number;
  dropOffRate: number; // percentage, e.g. 8.2%
  avgDwellTimeSec: number;
  conversionRate: number; // percentage
}

export interface ExecutiveJourneyAnalyticsData {
  totalExecutiveSessions: number;
  avgSessionDurationMin: number;
  funnelSteps: JourneyFunnelStep[];
  frictionPoints: Array<{
    location: string;
    severity: 'LOW' | 'MEDIUM' | 'HIGH';
    description: string;
    resolution: string;
  }>;
}

export interface Phase27ExitCriteria {
  mentorReach: { target: number; actual: number; passed: boolean }; // Target >= 95%
  mentorEngagement: { target: number; actual: number; passed: boolean }; // Target >= 60%
  playbookReach: { target: number; actual: number; passed: boolean }; // Target >= 75%
  behavioralAdoption: { target: number; actual: number; passed: boolean }; // Target >= 70%
  ruleAdherence: { target: number; actual: number; passed: boolean }; // Target >= 80%
  repeatMistakeReduction: { target: number; actual: number; passed: boolean }; // Target >= 30%
  decisionDrift: { target: number; actual: number; passed: boolean }; // Target < 25%
  executiveUAT: { target: number; actual: number; passed: boolean }; // Target >= 95%
  platformAvailability: { target: number; actual: number; passed: boolean }; // Target >= 99.9%
  productionExcellenceScore: { target: number; actual: number; passed: boolean }; // Target >= 99.0%
}
