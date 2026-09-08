/**
 * Phase 29: Organizational Benchmark Engine
 *
 * Implements:
 * - INV-OI5 (Groupthink Detection): consensus without diversity → risk flag
 * - INV-OI6 (Benchmark Isolation): no team compared against itself
 * - INV-OI8 (Organizational Fairness): max influence ≤40%
 * - OI-301/302/303: Team benchmarks, role cohorts, top performer analysis
 */

import type {
  TeamBenchmark,
  RoleCohortBenchmark,
  OrganizationalCohortDistribution,
} from '@/types/organizational-intelligence';
import {
  CANONICAL_TEAM_BENCHMARKS,
  CANONICAL_ROLE_COHORTS,
  CANONICAL_ORGANIZATIONAL_COHORTS,
} from './odeiEngine';

export { CANONICAL_TEAM_BENCHMARKS, CANONICAL_ROLE_COHORTS, CANONICAL_ORGANIZATIONAL_COHORTS };

// ---------------------------------------------------------------------------
// INV-OI5: Groupthink Detection
// ---------------------------------------------------------------------------

export interface CommitteeConsensusInput {
  committeeId: string;
  approvalCount: number;
  dissentCount: number;
  evidenceVariance: number; // 0.0 = unanimous same evidence, 1.0 = diverse
  uniqueContributorRatio: number; // unique contributors / total approvers
}

export interface GroupthinkResult {
  committeeId: string;
  groupthinkRisk: boolean;
  riskLevel: 'NONE' | 'LOW' | 'MODERATE' | 'HIGH';
  approvalCount: number;
  dissentCount: number;
  evidenceVariance: number;
  diversityIndex: number;
  explanation: string;
}

export function detectGroupthinkRisk(input: CommitteeConsensusInput): GroupthinkResult {
  const hasZeroDissent = input.dissentCount === 0;
  const hasLowEvidenceVariance = input.evidenceVariance < 0.2;
  const hasLowDiversity = input.uniqueContributorRatio < 0.5;
  const diversityIndex = (input.evidenceVariance + input.uniqueContributorRatio) / 2;

  const isGroupthink = hasZeroDissent && hasLowEvidenceVariance;
  const riskLevel = isGroupthink
    ? (hasLowDiversity ? 'HIGH' : 'MODERATE')
    : hasZeroDissent
      ? 'LOW'
      : 'NONE';

  return {
    committeeId: input.committeeId,
    groupthinkRisk: isGroupthink,
    riskLevel,
    approvalCount: input.approvalCount,
    dissentCount: input.dissentCount,
    evidenceVariance: input.evidenceVariance,
    diversityIndex,
    explanation: isGroupthink
      ? `Committee shows ${input.approvalCount} approvals with 0 dissent and low evidence variance (${(input.evidenceVariance * 100).toFixed(0)}%). INV-OI5 GROUPTHINK RISK flagged.`
      : `Committee consensus shows healthy diversity. Risk level: ${riskLevel}.`,
  };
}

// ---------------------------------------------------------------------------
// INV-OI6: Benchmark Isolation
// ---------------------------------------------------------------------------

export interface BenchmarkIsolationResult {
  isIsolated: boolean;
  violations: string[];
  testedTeamCount: number;
  benchmarkPopulation: number;
}

export function verifyBenchmarkIsolation(
  targetTeamId: string,
  benchmarkTeamIds: string[]
): BenchmarkIsolationResult {
  const violations = benchmarkTeamIds.filter(id => id === targetTeamId);

  return {
    isIsolated: violations.length === 0,
    violations,
    testedTeamCount: benchmarkTeamIds.length,
    benchmarkPopulation: 42,
  };
}

// ---------------------------------------------------------------------------
// INV-OI8: Organizational Fairness — Influence Concentration
// ---------------------------------------------------------------------------

export interface InfluenceConcentrationInput {
  actorId: string;
  influencePct: number;
}

export interface FairnessResult {
  isConcentrated: boolean;
  maxInfluencePct: number;
  maxInfluenceActorId: string;
  threshold: number;
  explanation: string;
}

export function detectInfluenceConcentration(
  actors: InfluenceConcentrationInput[]
): FairnessResult {
  const threshold = 40.0;
  const maxActor = actors.reduce((prev, curr) => curr.influencePct > prev.influencePct ? curr : prev);

  return {
    isConcentrated: maxActor.influencePct > threshold,
    maxInfluencePct: maxActor.influencePct,
    maxInfluenceActorId: maxActor.actorId,
    threshold,
    explanation: maxActor.influencePct > threshold
      ? `INV-OI8 FAIRNESS VIOLATION: Actor "${maxActor.actorId}" contributes ${maxActor.influencePct}% — exceeds the 40% maximum threshold.`
      : `Influence distribution is fair. Max actor: "${maxActor.actorId}" at ${maxActor.influencePct}% (threshold: ${threshold}%).`,
  };
}

// ---------------------------------------------------------------------------
// Top Performer Pattern Analysis (OI-303)
// ---------------------------------------------------------------------------

export interface TopPerformerPattern {
  rank: number;
  teamName: string;
  odei: number;
  differentiatingBehavior: string;
  behaviorImpact: string;
  replicability: 'HIGH' | 'MEDIUM' | 'LOW';
}

export const CANONICAL_TOP_PERFORMER_PATTERNS: TopPerformerPattern[] = [
  {
    rank: 1,
    teamName: 'Committee Alpha',
    odei: 91,
    differentiatingBehavior: 'Daily institutional flow review before entry — 100% adherence',
    behaviorImpact: '+8.2 DQ points vs median team',
    replicability: 'HIGH',
  },
  {
    rank: 2,
    teamName: 'Growth Equity Team',
    odei: 86,
    differentiatingBehavior: 'Weekly outcome review with 100% resolution rate',
    behaviorImpact: '+5.1 DQ points, 94% mistake prevention on repeated patterns',
    replicability: 'HIGH',
  },
  {
    rank: 3,
    teamName: 'Macro Strategy',
    odei: 81,
    differentiatingBehavior: 'Pre-market macro signal integration — 18% faster decisions',
    behaviorImpact: '-22% decision cycle time',
    replicability: 'MEDIUM',
  },
];

// ---------------------------------------------------------------------------
// Canonical Groupthink Scenario (for verification)
// ---------------------------------------------------------------------------

export const CANONICAL_GROUPTHINK_SCENARIO: CommitteeConsensusInput = {
  committeeId: 'COMMITTEE-SYNTHETIC-001',
  approvalCount: 12,
  dissentCount: 0,
  evidenceVariance: 0.04,
  uniqueContributorRatio: 0.33,
};

// Result must show groupthinkRisk: true (INV-OI5 verification)
export const CANONICAL_GROUPTHINK_RESULT = detectGroupthinkRisk(CANONICAL_GROUPTHINK_SCENARIO);

// ---------------------------------------------------------------------------
// Canonical Benchmark Isolation Test
// ---------------------------------------------------------------------------

export const CANONICAL_ISOLATION_TEST_RESULT = verifyBenchmarkIsolation(
  'committee-alpha',
  ['growth-equity', 'macro-strategy', 'fixed-income', 'emerging-markets']
);

// ---------------------------------------------------------------------------
// Canonical Fairness Test — Concentration Violation Scenario
// ---------------------------------------------------------------------------

export const CANONICAL_CONCENTRATION_VIOLATION = detectInfluenceConcentration([
  { actorId: 'user-dominant', influencePct: 63 },
  { actorId: 'user-secondary', influencePct: 22 },
  { actorId: 'user-tertiary', influencePct: 15 },
]);
// → isConcentrated: true (63% > 40% threshold)

