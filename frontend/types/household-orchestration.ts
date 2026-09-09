/**
 * Horizon 9 & 10: Household Strategy Orchestration & Life Portfolio Domain Models
 */

export type StrategyCategory =
  | 'CAREER'
  | 'FINANCIAL'
  | 'EDUCATION'
  | 'LOCATION'
  | 'BUSINESS'
  | 'HEALTH';

export interface HouseholdStrategy {
  strategyId: string;
  title: string;
  category: StrategyCategory;
  projectedHHI: number;
  projectedLHI: number;
  projectedResilience: number;
  projectedFinancialHealth: number;
  confidenceScore: number;
  personalImpact?: string;
  financialImpact?: string;
  relationalImpact?: string;
  householdImpact?: string;
  riskScore?: number;
}

export interface FutureState {
  horizonYears: number; // 1, 3, 5, 10
  projectedHHI: number;
  projectedLHI: number;
  projectedNetWorth: number;
  projectedRelationshipHealth: number;
  projectedResilience: number;
  contributingSignals?: string[];
  dependencyChain?: string[];
  forecastDrivers?: string[];
}

export interface StrategyTransition {
  fromStrategyId: string;
  toStrategyId: string;
  triggerReason: string;
  transitionRisk: number;
  rollbackAvailable: boolean;
}

export interface HouseholdPortfolioPlan {
  planId: string;
  primaryStrategy: HouseholdStrategy;
  fallbackStrategy: HouseholdStrategy;
  recoveryStrategy: HouseholdStrategy;
  futureStates: FutureState[];
  transitions: StrategyTransition[];
  overallResilienceScore: number;
}
