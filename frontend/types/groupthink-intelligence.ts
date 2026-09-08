/**
 * Phase 31-M4: Groupthink & Risk Intelligence Data Contracts
 *
 * Implements data structures for:
 * - Epic M4-101: Groupthink Detection Engine (INV-OI19, INV-OI20, INV-OI21)
 * - Epic M4-102: Organizational Risk Registry (VR-R01 to VR-R06)
 * - Epic M4-103: Predictive Governance & Incident Forecasting (INV-OI22)
 */

export type RiskCategory =
  | 'GOVERNANCE'
  | 'LEARNING'
  | 'NETWORK'
  | 'REPLAY'
  | 'GROUPTHINK'
  | 'ATTRIBUTION';

export type RiskSeverity =
  | 'LOW'
  | 'MEDIUM'
  | 'HIGH'
  | 'CRITICAL';

export type RiskStatus =
  | 'OPEN'
  | 'MITIGATING'
  | 'RESOLVED'
  | 'CLOSED';

export interface GovernanceRisk {
  riskId: string; // e.g. RSK-001 (VR-R01)
  title: string;
  description: string;
  category: RiskCategory; // (VR-R05)
  severity: RiskSeverity;
  likelihoodPct: number; // 0 - 100 (VR-R02)
  impactScore: number; // 0 - 100 (VR-R03)
  exposureScore: number; // (likelihoodPct * impactScore) / 100 (VR-R04)
  status: RiskStatus;
  committeeId?: string;
  incidentIds: string[]; // Critical risks require at least one linked incident (VR-R06)
  createdAtUtc: string;
  updatedAtUtc?: string;
  mitigationPlan?: string;
  ownerId?: string;
}

export type GroupthinkSignalType =
  | 'EXCESSIVE_UNANIMITY'
  | 'DISSENT_EROSION'
  | 'DIVERSITY_DECLINE'
  | 'CONSENSUS_CONCENTRATION'
  | 'RECOMMENDATION_CONVERGENCE'
  | 'HIGH_PERFORMANCE_GROUPTHINK_RISK';

export interface GroupthinkSignal {
  signalId: string; // e.g. GT-001
  committeeId: string;
  signalType: GroupthinkSignalType;
  observedValue: number;
  thresholdValue: number;
  severity: RiskSeverity;
  detectedAtUtc: string;
  description: string;
  actionRequired: string;
}

export type GroupthinkRiskLevel =
  | 'LOW'
  | 'MEDIUM'
  | 'HIGH'
  | 'CRITICAL';

export interface GroupthinkAssessment {
  committeeId: string;
  assessedAtUtc: string;
  unanimousDecisionRatePct: number; // 0 - 100
  dissentRatePct: number; // 0 - 100
  dissentUtilizationRatePct: number; // 0 - 100
  influenceConcentrationPct: number; // 0 - 100
  diversityScore: number; // 0 - 100
  recommendationDiversityScore: number; // 0 - 100
  convergenceScore: number; // 0 - 100
  groupthinkScore: number; // 0 - 100 (INV-OI19)
  riskLevel: GroupthinkRiskLevel;
  invariantSatisfied: boolean; // groupthinkScore < 75.0 && diversityScore >= 60.0
  findings: string[];
  signals: GroupthinkSignal[];
  recommendations: string[];
}

export type ForecastPeriod =
  | '30D'
  | '90D'
  | '180D'
  | '365D';

export interface ForecastDriver {
  driverId: string;
  driverName: string;
  currentValue: number;
  projectedValue: number;
  contributionPct: number; // Must sum to 100% across drivers (INV-OI22)
  trend: 'UP' | 'DOWN' | 'STABLE';
}

export interface GovernanceForecast {
  forecastId: string;
  committeeId: string;
  forecastDateUtc: string;
  forecastPeriod: ForecastPeriod;
  projectedODEI: number;
  projectedDIRatio: number;
  projectedTransferRate: number;
  projectedCDQI: number;
  projectedCertificationProbability: number;
  projectedRiskScore: number;
  confidencePct: number;
  status: GroupthinkRiskLevel;
  drivers: ForecastDriver[];
}

export interface IncidentForecast {
  incidentId: string;
  currentSeverity: string;
  escalationProbability: number; // 0 - 100
  recurrenceProbability: number; // 0 - 100
  forecastDays: number;
  likelyRootCauses: string[];
  recommendedActions: string[];
}
