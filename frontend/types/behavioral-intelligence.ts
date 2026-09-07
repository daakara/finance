/**
 * Phase 28: Behavioral Intelligence & Story-First Decision Operating System Type Contracts
 * 
 * Defines contracts for:
 * 1. Story-First Executive Home & AI Chief of Staff
 * 2. Morning Briefing 2.0 (Market Changed -> Why It Matters -> What Is Affected -> What To Do)
 * 3. Behavioral Intelligence Center & Behavioral Evolution Timeline
 * 4. Learning Velocity Engine (LVI, BMI, Momentum)
 * 5. AI Behavioral Coach & 4-Month Forecasting
 * 6. Statistical Confidence Bands & Behavioral Cohorts
 * 
 * Phase 26 Quantitative Freeze Compliant: Strictly frontend presentation & behavioral contracts.
 */

export type TrendDirection = 'IMPROVING' | 'STABLE' | 'DECLINING';

export type TrendVelocity =
  | 'RAPID_IMPROVEMENT' // >= +10%
  | 'IMPROVING'         // +3% to +10%
  | 'STABLE'            // -3% to +3%
  | 'DECLINING'         // -10% to -3%
  | 'CRITICAL_DECLINE'; // < -10%

export interface ConfidenceInterval {
  pointEstimate: number; // e.g. 70.5
  lowerBound: number;    // e.g. 68.1
  upperBound: number;    // e.g. 72.7
  marginOfError: number; // e.g. 2.3
  confidenceLevel: number; // e.g. 0.95 for 95%
  displayString: string; // e.g. "70.5% (68.1% - 72.7%)"
}

export interface MetricWithConfidence {
  name: string;
  value: number;
  unit: string;
  ci: ConfidenceInterval;
  trend: TrendVelocity;
  delta7d: number;
  delta30d: number;
  target: number;
  isPassing: boolean;
}

export interface DecisionQualityMetrics {
  currentScore: number; // 74
  previousScore: number; // 62
  quarterlyChange: number; // +6
  annualChange: number; // +12
  percentileRank: number; // 18 (Top 18%)
  targetScore: number; // 80
  trend: TrendDirection;
}

export interface LearningVelocityMetrics {
  velocityIndex: number; // 84 (0-100)
  qualityImprovementRate: number; // +12 / year
  recommendationAdoptionRate: number; // 70.5%
  playbookAdherenceRate: number; // 87.0%
  projectedMonthsToGoal: number; // 4.0 months
  momentumMultiplier: number; // 1.5 (>1.0 = accelerating)
  velocityTier: 'LOW' | 'MEDIUM' | 'HIGH' | 'ELITE';
}

export interface DriftMetrics {
  driftScore: number; // 21.0%
  targetThreshold: number; // < 20.0%
  driftCategory: 'LOW' | 'MEDIUM' | 'HIGH';
  majorDeviationDrivers: string[];
}

export interface BehavioralStrength {
  id: string;
  title: string;
  qualityPointContribution: number; // e.g. +6.2
  confidence: number; // e.g. 91%
  description: string;
  evidenceHash: string;
}

export interface BehavioralRisk {
  id: string;
  title: string;
  exposurePercentage: number; // e.g. 21%
  confidence: number; // e.g. 87%
  description: string;
  mitigation: string;
}

export interface ImprovementForecast {
  currentScore: number; // 74
  projectedScoreFourMonths: number; // 80
  projectedScoreSixMonths: number; // 82
  confidence: number; // 87%
  isLowConfidenceWarning: boolean; // true if confidence < 60
  expectedGain: number; // +6.0
  keyCatalysts: string[];
}

export interface BehavioralTimelineEvent {
  quarter: string; // e.g. "Q1 2026"
  problemIdentified: string;
  governanceImprovement: string;
  qualityDelta: number; // e.g. +2
  supportingEvidence: string;
  category: 'STOP_LOSS' | 'MACRO_GATING' | 'FLOW_ACCUMULATION' | 'SIZING';
}

export interface BehavioralStory {
  userName: string; // "David"
  dateString: string; // "Monday, September 7, 2026"
  weeklyAdoptionRate: number; // 84%
  weeklyScoreDelta: number; // +2
  repeatMistakeDelta: number; // -12%
  macroExposureTrend: string;
  recommendedActionToday: string;
  monthlyStats: {
    totalDecisions: number; // 42
    successCount: number; // 30
    failureCount: number; // 12
    largestSuccessDriver: string;
    largestFailureDriver: string;
    netLearningTakeaway: string;
  };
  chiefOfStaffHighlight: {
    actionTitle: string;
    drawdownReductionPct: number; // 4.2%
    confidence: number; // 92%
    evidenceDetail: string;
  };
}

export interface ImpactedPositionRisk {
  symbol: string;
  shares: number;
  currentPrice: number;
  capitalAtRisk: number; // e.g. 78,000
  violatedCondition: string;
  suggestedAction: 'TRIM_50' | 'TIGHTEN_STOP' | 'EXIT';
}

export interface MorningBriefingV2Story {
  marketRiskScorePrev: number; // 42
  marketRiskScoreCurrent: number; // 56
  narrativeSummary: string;
  marketShifts: Array<{
    dimension: string;
    direction: 'STRENGTHENED' | 'WEAKENED' | 'NARROWED';
    detail: string;
  }>;
  affectedPositions: ImpactedPositionRisk[];
  totalCapitalAtRisk: number; // 184,000
  recommendationConfidence: number; // 91%
  isFeedStale: boolean;
  stalenessWarning?: string;
}

export type BehavioralMaturityTier =
  | 'CONSUMER'     // Level 1: 0-20
  | 'INVESTIGATOR' // Level 2: 21-40
  | 'PRACTITIONER' // Level 3: 41-60
  | 'LEARNER'      // Level 4: 61-80
  | 'OPTIMIZER';   // Level 5: 81-100

export interface MaturityTierDistribution {
  tier: BehavioralMaturityTier;
  label: string;
  scoreRange: string;
  userPercentage: number;
  description: string;
  primaryAction: string;
}

export interface RoleCohortMetric {
  role: 'EXECUTIVES' | 'PORTFOLIO_MANAGERS' | 'ANALYSTS' | 'NEW_USERS';
  label: string;
  adoptionRate: number; // e.g. 76%, 82%, 61%, 49%
  decisionQuality: number;
  userCount: number;
}

export interface BehavioralIntelligenceProfile {
  profileId: string;
  userId: string;
  generatedAt: string;
  story: BehavioralStory;
  decisionQuality: DecisionQualityMetrics;
  learningVelocity: LearningVelocityMetrics;
  behaviorAdoption: MetricWithConfidence;
  ruleAdherence: MetricWithConfidence;
  decisionDrift: DriftMetrics;
  strengths: BehavioralStrength[];
  risks: BehavioralRisk[];
  projectedImprovement: ImprovementForecast;
  timeline: BehavioralTimelineEvent[];
}

// Telemetry Event Contracts
export interface ExecutiveHomeViewedEvent {
  event: 'executive_home_viewed';
  screen: 'executive_home';
  user_id: string;
  session_id: string;
  decision_quality: number;
  cohort_rank: number;
  timestamp: string;
}

export interface StoryModuleViewedEvent {
  event: 'story_module_viewed';
  story_id: string;
  story_type: 'behavioral' | 'outcome';
  timestamp: string;
}

export interface BriefingNarrativeViewedEvent {
  event: 'briefing_narrative_viewed';
  market_risk_score: number;
  portfolio_risk_score: number;
  user_id: string;
  timestamp: string;
}

export interface ImpactedPositionsViewedEvent {
  event: 'impacted_positions_viewed';
  position_count: number;
  estimated_capital_at_risk: number;
  timestamp: string;
}

export interface BehavioralCenterViewedEvent {
  event: 'behavioral_center_viewed';
  decision_quality: number;
  behavioral_adoption_rate: number;
  decision_drift: number;
  timestamp: string;
}

export interface BehaviorChangeViewedEvent {
  event: 'behavior_change_viewed';
  change_type: string;
  quality_delta: number;
  timestamp: string;
}

export interface LviViewedEvent {
  event: 'lvi_viewed';
  lvi_score: number;
  cohort_rank: number;
  timestamp: string;
}

export interface BehavioralForecastViewedEvent {
  event: 'behavioral_forecast_viewed';
  predicted_quality_score: number;
  confidence: number;
  timestamp: string;
}

export interface CohortComparisonViewedEvent {
  event: 'cohort_comparison_viewed';
  cohort: string;
  comparison_type: 'decision_quality' | 'adoption' | 'drift';
  timestamp: string;
}

// -------------------------------------------------------------------------
// Phase 28 Final Technical Domain Models & Aggregate Contracts
// -------------------------------------------------------------------------

export type BehavioralStage =
  | 'CONSUMER'
  | 'INVESTIGATOR'
  | 'PRACTITIONER'
  | 'LEARNER'
  | 'OPTIMIZER';

export interface BehaviorInsight {
  id: string;
  title: string;
  category:
    | 'POSITION_SIZING'
    | 'RISK_DISCIPLINE'
    | 'ENTRY_TIMING'
    | 'EXIT_TIMING'
    | 'MACRO_ALIGNMENT'
    | 'PLAYBOOK_ADHERENCE';
  impactScore: number;
  contributionToQuality: number;
  confidence: number;
  trend: 'IMPROVING' | 'STABLE' | 'DECLINING';
}

export interface LearningVelocity {
  score: number;
  percentile: number;
  trend: 'ACCELERATING' | 'STABLE' | 'SLOWING';
  qualityGainLast90Days: number;
  recommendationAdoption: number;
  ruleAdherence: number;
}

export interface BehavioralCoachSummary {
  summary: string;
  keyDrivers: string[];
  nextRecommendation: string;
  projectedImpact: number;
  projectedMonthsToGoal: number;
  confidence: number;
}

export interface BehavioralMilestone {
  milestoneId: string;
  quarter: string;
  problemIdentified: string;
  improvementImplemented: string;
  impactPoints: number;
  newQualityScore: number;
}

export interface CohortMetrics {
  cohortType: 'EXECUTIVE' | 'PORTFOLIO_MANAGER' | 'ANALYST' | 'NEW_USER';
  adoptionRate: number;
  decisionQuality: number;
  driftScore: number;
  learningVelocity: number;
}

export interface ConfidenceBand {
  metricName: string;
  observedValue: number;
  lowerBound: number;
  upperBound: number;
  confidenceLevel: number;
}

export interface CanonicalBehavioralIntelligenceProfile {
  userId: string;
  generatedAt: string;
  version: number;
  decisionQualityScore: number;
  learningVelocityIndex: number;
  behavioralAdoptionRate: number;
  decisionDriftScore: number;
  repeatMistakeReduction: number;
  currentBehavioralStage: BehavioralStage;
  projectedQualityScore: number;
  projectedDateToGoal?: string;
  strongestBehavior: BehaviorInsight;
  weakestBehavior: BehaviorInsight;
  coachSummary: BehavioralCoachSummary;
  cohortMetrics: CohortMetrics;
}

// -------------------------------------------------------------------------
// Phase 28 Milestone 1: Behavioral Intelligence Foundations Interfaces
// -------------------------------------------------------------------------

export interface DIRProfile {
  userId: string;
  currentDecisionQuality: number;
  previousDecisionQuality: number;
  decisionQualityDelta: number;
  learningVelocityIndex: number;
  behavioralAdoptionRate: number;
  ruleAdherenceRate: number;
  repeatMistakeReduction: number;
  decisionDriftScore: number;
  decisionImprovementScore: number;
  percentileRank: number;
  confidence: number;
  generatedAt: string;
}

export interface DecisionImprovementScore {
  overallScore: number;
  qualityContribution: number;
  adoptionContribution: number;
  adherenceContribution: number;
  mistakeReductionContribution: number;
  driftContribution: number;
  trend: 'IMPROVING' | 'STABLE' | 'DECLINING';
  confidence: number;
}

export interface LearningVelocityIndex {
  currentValue: number;
  quarterlyGrowth: number;
  annualGrowth: number;
  acceleration: number;
  percentile: number;
  confidenceLower: number;
  confidenceUpper: number;
  category: 'LOW' | 'MEDIUM' | 'HIGH' | 'ELITE';
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

export interface BehavioralCohort {
  cohortId: string;
  category: 'CONSUMER' | 'INVESTIGATOR' | 'PRACTITIONER' | 'LEARNER' | 'OPTIMIZER';
  confidence: number;
  assignedAt: string;
  characteristics: string[];
  nextTargetCohort?: string;
}

export interface CohortDistribution {
  consumers: number;
  investigators: number;
  practitioners: number;
  learners: number;
  optimizers: number;
  totalUsers: number;
  generatedAt: string;
}

export interface BehavioralRecommendation {
  recommendationId: string;
  title: string;
  category: 'DO_MORE' | 'STOP_DOING' | 'CALIBRATE';
  projectedQualityImpact: number;
  projectedRiskReduction: number;
  projectedTimeToBenefitDays: number;
  confidence: number;
  evidenceCount: number;
  rationale: string;
}

export interface BehavioralIntelligenceDashboard {
  profile: DIRProfile;
  learningVelocity: LearningVelocityIndex;
  cohort: BehavioralCohort;
  distribution: CohortDistribution;
  timeline: BehavioralTimelineEvent[];
  recommendations: BehavioralRecommendation[];
  lastUpdated: string;
}

export interface ExecutiveStory {
  headline: string;
  summary: string;
  topImprovement: string;
  biggestRisk: string;
  recommendedAction: string;
  projectedBenefit: number;
  confidence: number;
  generatedAt: string;
}

export interface DecisionIntelligenceResult {
  dirScore: number;
  confidenceScore: number;
  trendDirection: 'IMPROVING' | 'STABLE' | 'DECLINING';
  percentile: number;
  benchmark: number;
  confidenceBand: {
    lower: number;
    upper: number;
    confidenceLevel: number;
  };
  sampleSize?: number;
  edgeCases?: string[];
  isProvisional?: boolean;
  components?: {
    dqsContribution: number;
    outcomeContribution: number;
    learningContribution: number;
    governanceContribution: number;
  };
}

export interface BehavioralCohortResult {
  cohortName: string;
  tenureCohort: '0-30 Days' | '31-90 Days' | '91-365 Days' | '365+ Days';
  behavioralCohort: 'Observer' | 'Reviewer' | 'Predictor' | 'Learner' | 'Institutional Operator' | 'Consumer' | 'Investigator' | 'Practitioner' | 'Optimizer';
  dir: number;
  par: number; // Participation/Adoption Rate
  learningVelocity: number;
  engagement: number;
  retention: number;
  confidence: number;
}

export interface ExecutiveBenchmarkResult {
  percentileRank: number;
  improvementDelta: number;
  expectedProgression: {
    targetScore: number;
    targetHorizonMonths: number;
    projectedGrowthRate: number;
  };
  benchmarks: {
    personalHistorical: number;
    teamAverage: number;
    institutionAverage: number;
    eliteQuartile: number;
  };
  layerDeltas: {
    vsPersonalHistorical: number;
    vsTeamAverage: number;
    vsInstitutionAverage: number;
    vsEliteQuartile: number;
  };
}

// ---------------------------------------------------------------------------
// Phase 28 Milestone 2A: Executive Narrative & Capability Attribution Contracts
// ---------------------------------------------------------------------------

export type ExecutiveNarrativeState =
  | 'HEALTHY'
  | 'IMPROVING'
  | 'PLATEAU'
  | 'DECLINING'
  | 'INACTIVE'
  | 'LOW_CONFIDENCE'
  | 'NEW_USER';

export interface ExecutiveNarrativeInputs {
  dir: number;
  dirTrend?: number;
  learningVelocity?: number;
  confidence?: number;
  decisionCount?: number;
  daysSinceLastActivity?: number;
  topDriver?: string;
  topWeakness?: string;
  userName?: string;
  dateString?: string;
  portfolioAtRisk?: number;
}

export interface ExecutiveNarrativeResult {
  state: ExecutiveNarrativeState;
  stateLabel: string;
  dirScore: number;
  dirTrend: number;
  trendDirection: 'IMPROVING' | 'STABLE' | 'DECLINING';
  confidence: number;
  headline: string;
  executiveSummary: {
    observation: string;
    learning: string;
    recommendedAction: string;
    actionConfidence: number;
    evidenceTrace: string;
  };
  metrics: {
    learningVelocity: number;
    attentionCount: number;
    portfolioAtRisk: number;
  };
  topOpportunity: ExecutiveTopOpportunity;
  topRisk: ExecutiveTopRisk;
  generatedAt: string;
}

export interface ExecutiveTopOpportunity {
  title: string;
  driverPattern: string;
  historicalWinRate: number;
  estimatedContributionPoints: number;
  actionableDirective: string;
  confidence: number;
  evidenceSample: number;
}

export interface ExecutiveTopRisk {
  title: string;
  threatPattern: string;
  lossContributionPct: number;
  mitigationDirective: string;
  confidence: number;
  evidenceSample: number;
}

export interface CapabilityImpactItem {
  capabilityId: 'outcome_reviews' | 'ai_coach' | 'decision_journal' | 'committee_governance' | string;
  capabilityName: string;
  usageRate: number; // percentage, e.g. 78%
  estimatedContribution: number; // DQ points, e.g. +4.7
  contributionRange: {
    lower: number;
    upper: number;
  };
  confidence: number; // percentage, e.g. 92%
  interactionsCount: number; // e.g. 843
  capabilityRoiIndex: number; // CRI = contribution / (usageRate / 10) or contribution / usage
  executiveExplanation: string;
}

export interface CapabilityImpactAttribution {
  totalImprovementPoints: number; // e.g. 12.0
  explainedImprovementPoints: number; // e.g. 11.4
  residualDriftPoints: number; // e.g. 0.6
  isConservationSatisfied: boolean; // within +/- 0.5 points
  capabilities: CapabilityImpactItem[];
  highestRoiCapability: string;
}

// ---------------------------------------------------------------------------
// Phase 28 Milestone 2B: My Evolution Workspace Contracts
// ---------------------------------------------------------------------------

export type EvolutionInteractionMode = 'SUMMARY' | 'EXPLORATION' | 'ANALYSIS' | 'PROJECTION';

export interface EvolutionMilestone {
  id?: string;
  quarter: string; // e.g. "2025 Q4", "2026 Q1", "2026 Q2", "CURRENT (2026 Q3)", "TARGET (Q1 2027)"
  dirScore: number;
  scoreDelta: number;
  cohort: 'CONSUMER' | 'INVESTIGATOR' | 'PRACTITIONER' | 'LEARNER' | 'OPTIMIZER';
  cohortLabel: string;
  status?: 'COMPLETED' | 'CURRENT' | 'PROJECTED';
  isCurrent: boolean;
  isTarget: boolean;
  problem: string;
  problemStatement?: string;
  actionTaken: string;
  behaviorAdopted: string;
  adoptedHabits?: string[];
  behaviorStopped: string;
  stoppedHabits?: string[];
  primaryCapability: string;
  capabilityContribution: number;
  outcomeImpact: {
    winRate: string;
    drawdown: string;
    profitFactor: string;
  };
  evidenceTrace: string;
  confidence: number;
  confidenceInterval?: {
    lower: number;
    upper: number;
  };
}

export interface BehaviorLedgerItem {
  id: string;
  name: string;
  habitName?: string;
  type: 'ADOPTED' | 'REMOVED';
  quarter: string;
  impactPoints: number;
  dqImpactPoints?: number;
  metricCorrelation: string;
  category?: string;
  frequency?: string;
  confidence: number;
}

export interface CapabilityRoiLeaderboardItem {
  rank: number;
  capabilityId: string;
  name: string;
  capabilityName?: string;
  badge: 'BEST_CAPABILITY' | 'FASTEST_GROWING' | 'MOST_UNDERUSED' | 'GOVERNANCE_ANCHOR';
  efficiencyBadge?: string;
  cri: number;
  capabilityRoiIndex?: number;
  impactPoints: number;
  marginalDIRPoints?: number;
  usageRate: number;
  confidence: number;
  strategicNote: string;
}

export interface EvolutionJourneyProfile {
  userId: string;
  currentDir: number;
  currentDIR?: number;
  baselineDir: number;
  startingDIR?: number;
  totalGain: number;
  fourQuarterGain?: number;
  cohortPercentile: number;
  percentileRank?: number;
  maturityTier: string;
  learningVelocity: number;
  learningVelocityClass: string;
  targetDir: number;
  targetDIR?: number;
  targetHorizon: string;
  targetProbability: number;
  projectedMonthsToTarget?: number;
  projectionConfidence?: number;
  milestones: EvolutionMilestone[];
  behaviorLedger: BehaviorLedgerItem[];
  capabilityLeaderboard: CapabilityRoiLeaderboardItem[];
  evolutionCoach: {
    biggestWin: string;
    biggestRisk: string;
    nextHabit: string;
    projectedMonthsToTarget: number;
    projectedConfidence: number;
  };
  attributionCoveragePct: number;
}


