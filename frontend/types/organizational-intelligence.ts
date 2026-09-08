/**
 * Phase 29: Organizational Intelligence — Data Contracts
 *
 * Formalizes the complete type system for the Organizational Decision Effectiveness
 * Index (ODEI), Institutional Knowledge Graph, Cross-Team Learning, Benchmarking,
 * Capability Impact Score (CIS), and Governance Invariants (INV-OI1–INV-OI10).
 *
 * ODEI = 0.35(DQ) + 0.30(OE) + 0.20(LE) + 0.15(OH)
 *
 * Phase 26 Quantitative Freeze Compliant: Pure frontend telemetry and presentation.
 */

// ---------------------------------------------------------------------------
// Core ODEI Types
// ---------------------------------------------------------------------------

export type ODEIClassification =
  | 'ELITE'
  | 'HIGH_PERFORMING'
  | 'EFFECTIVE'
  | 'DEVELOPING'
  | 'AT_RISK'
  | 'CRITICAL';

export type ODEITrend = 'UP' | 'DOWN' | 'FLAT';

export interface ODEIComponents {
  /** Decision Quality: evidence, risk, thesis, approval quality. Weight: 35% */
  decisionQuality: number;
  /** Outcome Effectiveness: success rate, prediction actionability, target completion. Weight: 30% */
  outcomeEffectiveness: number;
  /** Learning Effectiveness: outcome reviews, attribution, adoption, behavior change. Weight: 20% */
  learningEffectiveness: number;
  /** Organizational Health: consensus diversity, review participation, velocity. Weight: 15% */
  organizationalHealth: number;
}

export interface ODEIConfidenceModel {
  confidencePct: number;
  sampleSize: number;
  organizationsCompared: number;
  observationWindowDays: number;
}

export interface ODEIResult {
  score: number;
  priorScore: number;
  delta: number;
  trend: ODEITrend;
  classification: ODEIClassification;
  components: ODEIComponents;
  confidence: ODEIConfidenceModel;
  topTeam: string;
  topTeamScore: number;
  largestImprovement: string;
  largestImprovementDelta: number;
  largestRisk: string;
  largestRiskSeverity: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
}

export interface OrganizationalDecisionMetric {
  score: number;
  priorScore: number;
  trend: ODEITrend;
  confidence: number;
}

export interface OrganizationalReadinessIndex {
  score: number;
  decisionQuality: number;
  learningVelocity: number;
  governanceCompliance: number;
  adoptionRate: number;
  knowledgeReuse: number;
  trend: ODEITrend;
  confidence: number;
}

// ---------------------------------------------------------------------------
// Strategic KPIs (OM-01 to OM-05)
// ---------------------------------------------------------------------------

export interface StrategicOrgKPI {
  id: string;
  name: string;
  description: string;
  current: number;
  target: number;
  unit: string;
  status: 'PASS' | 'FAIL' | 'WATCH';
  trend: ODEITrend;
}

// ---------------------------------------------------------------------------
// Team Benchmarking
// ---------------------------------------------------------------------------

export type TeamCohort = 'EMERGING' | 'DEVELOPING' | 'HIGH_PERFORMING' | 'ELITE';

export interface TeamBenchmark {
  teamId: string;
  teamName: string;
  odei: number;
  cohort: TeamCohort;
  decisionQuality: number;
  learningVelocity: number;
  ruleAdherence: number;
  drift: number;
  percentile: number;
  trend: ODEITrend;
}

export interface RoleCohortBenchmark {
  role: 'ANALYST' | 'PORTFOLIO_MANAGER' | 'LEADERSHIP';
  avgDecisionQuality: number;
  avgLearningVelocity: number;
  avgRuleAdherence: number;
  sampleSize: number;
}

export interface OrganizationalCohortDistribution {
  emergingPct: number;   // ODEI < 70
  developingPct: number; // ODEI 70–79
  highPerformingPct: number; // ODEI 80–89
  elitePct: number;          // ODEI 90+
}

// ---------------------------------------------------------------------------
// Capability Impact Score (CIS)
// ---------------------------------------------------------------------------

export interface CapabilityImpactScore {
  capabilityId: string;
  capabilityName: string;
  cis: number;
  behaviorLiftPct: number;
  decisionQualityLift: number;
  capitalPreservedFormatted: string;
  capitalPreservedDollars: number;
  contributionPct: number;
  adoptionPct: number;
  confidence: number;
  sampleSize: number;
  investmentQuadrant: 'HIGH_IMPACT_HIGH_ADOPTION' | 'HIGH_IMPACT_LOW_ADOPTION' | 'LOW_IMPACT_HIGH_ADOPTION' | 'LOW_IMPACT_LOW_ADOPTION';
}

export interface CapabilityAttribution {
  capabilityId: string;
  capabilityName: string;
  qualityImpact: number;
  behaviorImpact: number;
  valueImpact: number;
  contributionPct: number;
}

export interface CapabilityAttributionSummary {
  totalAttributionPct: number; // Must equal 100.0
  capabilities: CapabilityImpactScore[];
  totalCapitalPreserved: string;
  excessReturnPct: number;
  topCapabilityId: string;
  isConservationSatisfied: boolean;
}

// ---------------------------------------------------------------------------
// Knowledge Graph
// ---------------------------------------------------------------------------

export type KnowledgeNodeType =
  | 'DECISION'
  | 'PREDICTION'
  | 'OUTCOME'
  | 'LEARNING'
  | 'PLAYBOOK'
  | 'GOVERNANCE';

export interface KnowledgeNode {
  nodeId: string;
  type: KnowledgeNodeType;
  title: string;
  teamId: string;
  confidence: number;
  createdAt: string;
  linkedNodes: string[];
}

export interface KnowledgeEdge {
  edgeId: string;
  sourceNodeId: string;
  targetNodeId: string;
  relationship: 'INFORMED_BY' | 'LED_TO' | 'UPDATED' | 'DERIVED_FROM' | 'GOVERNS';
  weight: number;
}

export interface KnowledgeGraph {
  decisionNodes: number;
  predictionNodes: number;
  outcomeNodes: number;
  learningNodes: number;
  playbookNodes: number;
  governanceNodes: number;
  totalNodes: number;
  totalEdges: number;
  relationshipCoverage: number; // Must equal 100
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
}

export type LearningPatternType = 'SUCCESS' | 'FAILURE' | 'EMERGING';

export interface LearningPattern {
  patternId: string;
  patternType: LearningPatternType;
  title: string;
  description: string;
  confidence: number;
  occurrences: number;
  affectedTeams: string[];
  economicImpact: string;
}

// ---------------------------------------------------------------------------
// Organizational Learning
// ---------------------------------------------------------------------------

export interface CrossTeamLearningItem {
  itemId: string;
  sourceTeam: string;
  learningTitle: string;
  learningType: 'MOMENTUM_PATTERN' | 'RISK_FILTER' | 'GOVERNANCE_RULE' | 'BEHAVIORAL_INSIGHT';
  relevanceScore: number;
  recommendedAction: 'ADOPT' | 'REVIEW' | 'MONITOR';
  potentialImpact: string;
  confidence: number;
}

export interface KnowledgePropagation {
  propagationId: string;
  sourceTeam: string;
  receivingTeam: string;
  ruleId: string;
  ruleName: string;
  adoptionRate: number;
  impactScore: number;
  status: 'ADOPTED' | 'PENDING' | 'REJECTED';
}

export interface LearningImpactRecord {
  teamId: string;
  teamName: string;
  adopted: number;
  improved: number;
  ignored: number;
  adoptionRate: number;
}

// ---------------------------------------------------------------------------
// Governance Invariants
// ---------------------------------------------------------------------------

export interface GovernanceInvariantCriterion {
  criterionId: string;
  description: string;
  passed: boolean;
  actual: string;
  target: string;
}

export interface GovernanceInvariantResult {
  invariantId: string;
  invariantName: string;
  passed: boolean;
  criteria: GovernanceInvariantCriterion[];
  details: string;
}

// ---------------------------------------------------------------------------
// Executive Organizational Intelligence
// ---------------------------------------------------------------------------

export interface OrganizationalNarrative {
  observation: string;
  learning: string;
  recommendedAction: string;
  whoAffected: string;
  expectedOutcome: string;
  confidence: number;
  evidenceId: string;
}

export interface ExecutiveOrganizationalBriefing {
  greeting: string;
  odei: number;
  classification: ODEIClassification;
  narrative: OrganizationalNarrative;
  topOpportunity: string;
  topOpportunityImpact: string;
  topRisk: string;
  topRiskSeverity: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  capitalPreserved: string;
  excessReturn: number;
}

// ---------------------------------------------------------------------------
// Phase 29 Certification
// ---------------------------------------------------------------------------

export interface Phase29CertificationGate {
  gateId: string;
  gateName: string;
  target: string;
  actual: string;
  status: 'PASS' | 'FAIL';
  details: string;
}

export interface Phase29CertificationResult {
  status: 'CERTIFIED' | 'RELEASE_CANDIDATE' | 'NOT_READY';
  overallScore: number;
  gates: Phase29CertificationGate[];
  certifiedAt: string;
  releaseTrain: string;
}

// ---------------------------------------------------------------------------
// Organizational Telemetry Event Taxonomy
// ---------------------------------------------------------------------------

export type OrganizationalEventName =
  | 'knowledge_node_created'
  | 'knowledge_node_linked'
  | 'knowledge_pattern_detected'
  | 'knowledge_reuse_detected'
  | 'best_practice_published'
  | 'best_practice_propagated'
  | 'best_practice_adopted'
  | 'learning_feed_viewed'
  | 'team_benchmark_viewed'
  | 'team_comparison_opened'
  | 'peer_cohort_opened'
  | 'top_performer_analyzed'
  | 'capability_impact_viewed'
  | 'capability_attribution_generated'
  | 'roi_report_opened'
  | 'investment_priority_viewed'
  | 'executive_home_viewed'
  | 'executive_briefing_opened'
  | 'organizational_health_viewed'
  | 'organizational_readiness_viewed'
  | 'strategic_opportunity_opened';

export interface OrganizationalTelemetryEvent {
  eventId: string;
  eventName: OrganizationalEventName;
  userId: string;
  role: string;
  teamId: string;
  sessionId: string;
  timestampUtc: string;
  releaseTrain: 'PHASE_29';
  featureArea: string;
  confidence?: number;
  evidenceId?: string;
}

// ---------------------------------------------------------------------------
// INV-OI11: Institutional Learning Non-Regression Invariant
// ---------------------------------------------------------------------------

export interface ProtectedPractice {
  practiceId: string;
  practiceName: string;
  confidence: number; // >= 95%
  sampleSize: number; // >= Nmin
  valueImpactDollars: number; // > 0
  governanceApproved: boolean; // strictly true
  baselineAdoption: number; // B
  currentAdoption: number; // A(t) >= B - 10%
  historicalEffectiveness: number; // Historical E
  currentEffectiveness: number; // E(t) >= Historical - 5%
  mappedTo: {
    type: 'PLAYBOOK' | 'GOVERNANCE' | 'CAPABILITY';
    targetId: string;
  };
  status: 'PROTECTED' | 'REGRESSED' | 'AT_RISK';
}

export interface LearningNonRegressionResult {
  satisfied: boolean;
  totalProtectedPractices: number;
  criticalRegressions: number;
  practices: Array<{
    practiceId: string;
    practiceName: string;
    adoptionVariance: number;
    effectivenessVariance: number;
    isAdoptionRegressed: boolean;
    isEffectivenessRegressed: boolean;
    status: 'PASS' | 'FAIL';
  }>;
  orphanLearningsCount: number;
  knowledgeReuseRate: number;
  details: string;
}

