/**
 * Phase 28: Decision Improvement Rating (DIR) & Behavioral Cohort Migration Contracts
 * 
 * Formalizes the primary North Star metric of ARX Terminal Phase 28:
 * DIR = 0.30(DQG) + 0.20(BAS) + 0.20(RAS) + 0.15(DRS) + 0.15(LVI)
 * 
 * And the Behavioral Cohort Migration Framework:
 * Consumer -> Investigator -> Practitioner -> Learner -> Optimizer -> Operator
 * 
 * Phase 26 Quantitative Freeze Compliant: Strictly frontend presentation & telemetry contracts.
 */

export type DIRComponentKey = 'dqg' | 'bas' | 'ras' | 'drs' | 'lvi';

export type DIRClassification =
  | 'CRITICAL_REGRESSION' // DIR < 40
  | 'REGRESSING'          // DIR 40-49
  | 'STAGNANT'            // DIR 50-59
  | 'IMPROVING'           // DIR 60-74
  | 'HIGH_PERFORMER'      // DIR 75-84
  | 'EXEMPLARY';          // DIR >= 85

export interface UserDIRInputs {
  currentDecisionScore: number;        // e.g. 74
  baselineDecisionScore: number;       // e.g. 62
  recommendationsFollowed: number;     // e.g. 79
  recommendationsIssued: number;       // e.g. 112
  stopLossAdherence: number;           // e.g. 91 (%)
  macroInvalidationAdherence: number;  // e.g. 85 (%)
  positionSizingLimitAdherence: number;// e.g. 88 (%)
  riskControlsAdherence: number;       // e.g. 84 (%)
  driftScore: number;                  // e.g. 21.0 (%)
  learningVelocityIndex: number;       // e.g. 68.0 (0-100)
  recordedDecisionsCount: number;      // e.g. 42 (Rule 1: >= 30)
  outcomeReviewsCount: number;         // e.g. 28 (Rule 3: >= 10)
  statisticalConfidence: number;       // e.g. 88.0 (%) (Rule 6: >= 70%)
}

export interface DIRComponentDetail {
  key: DIRComponentKey;
  label: string;
  score: number;              // raw value (0-100)
  weight: number;             // e.g. 0.30
  weightedContribution: number; // score * weight
  formula: string;            // readable mathematical formula
  description: string;
}

export interface DIRComponents {
  dqg: DIRComponentDetail; // Decision Quality Growth (30%)
  bas: DIRComponentDetail; // Behavioral Adoption Score (20%)
  ras: DIRComponentDetail; // Rule Adherence Score (20%)
  drs: DIRComponentDetail; // Drift Resistance Score (15%)
  lvi: DIRComponentDetail; // Learning Velocity Index (15%)
}

export interface DIRValidationRule {
  id: string;
  name: string;
  threshold: string;
  passed: boolean;
  actualValue: number | string;
  impactMessage: string;
  penaltyApplied?: number;
  capApplied?: number;
}

export interface DIRProjection {
  currentDIR: number;
  projectedScore90d: number;       // e.g. 69
  projectedConfidence: number;     // e.g. 88 (%)
  targetScore: number;             // e.g. 75
  targetLabel: string;             // "High Performer"
  projectedQuarterlyGain: number;  // e.g. +6.0
  strongestDriver: {
    name: string;
    impact: number;                // e.g. +6.2
    description: string;
  };
  largestObstacle: {
    name: string;
    impact: number;                // e.g. -4.1
    description: string;
  };
}

export interface DIRResult {
  rawDIR: number;                  // Uncapped float e.g. 63.05
  finalDIR: number;                // Display integer e.g. 63
  exactDIR: number;                // Rounded to 1 decimal e.g. 63.1
  classification: DIRClassification;
  percentileCohort: number;        // e.g. 72 (faster than 72% / top 28%)
  components: DIRComponents;
  validationRules: DIRValidationRule[];
  allRulesPassed: boolean;
  isDataSufficient: boolean;       // False if observations < 30
  isLowConfidenceDataset: boolean; // True if confidence < 70%
  appliedCap?: number;             // e.g. 70 if drift > 60%
  appliedPenalty?: number;         // e.g. 15 if RAS < 50
  projection: DIRProjection;
}

// -------------------------------------------------------------------------
// Behavioral Cohort Migration Types
// -------------------------------------------------------------------------

export type BehavioralCohortId =
  | 'consumer'
  | 'investigator'
  | 'practitioner'
  | 'learner'
  | 'optimizer'
  | 'operator';

export interface BehavioralCohortDefinition {
  id: BehavioralCohortId;
  level: number;                   // 1 to 6
  name: string;                    // e.g. "Consumer"
  description: string;
  userSharePercent: number;        // e.g. 18 (%)
  trendDelta: number;              // e.g. -4.0 (%)
  dominantAction: string;
  keyMetricBenchmark: string;
  targetAdvancementDays: number;
  isExpanding: boolean;            // true if trendDelta > 0
}

export interface CohortTransitionFlow {
  from: BehavioralCohortId;
  to: BehavioralCohortId;
  transitionRate: number;          // e.g. 31.0 (%)
  flowLabel: string;
  isHealthy: boolean;
}

export interface CohortMigrationSummary {
  cohortAdvancementRate: number;   // CAR = 31.0 (%) (Target > 25%)
  cohortAdvancementTarget: number; // 25.0 (%)
  cohortRegressionRate: number;    // CRR = 4.0 (%) (Target < 10%)
  cohortRegressionTarget: number;  // 10.0 (%)
  timeToMaturityDays: number;      // TTM = 142 days (Target < 180 days)
  timeToMaturityTargetDays: number;// 180 days
  cohortVelocityScore: number;     // CVS = 0.033 / day
  cohorts: BehavioralCohortDefinition[];
  transitionFlows: CohortTransitionFlow[];
}

export interface DIRQuarterlyHistory {
  quarter: string;
  dirScore: number;
  delta: number;
  dqs: number;
  dominantBehavior: string;
  keyMilestone: string;
}

export interface DIRPeerBenchmark {
  cohortName: string;
  avgDIR: number;
  userDelta: number;               // user DIR - cohort avg
  sampleSize: number;
  colorHex: string;
}
