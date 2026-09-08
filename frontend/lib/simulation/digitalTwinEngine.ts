/**
 * M12.2 Digital Twin Engine (Phase 31-M12)
 *
 * Implements:
 * - Committee Digital Twin: vote simulation, dissent friction, groupthink vulnerability
 * - Organization Digital Twin: aggregate institutional state projection
 * - Strict In-Memory Isolation (INV-OI66: zero mutation of live database/records)
 */

import type { CommitteeTwin, DigitalTwinState } from '../../types/simulation-intelligence';

export const CANONICAL_COMMITTEE_TWINS: CommitteeTwin[] = [
  {
    committeeId: 'COM-001',
    committeeName: 'Investment Committee',
    memberCount: 7,
    dissentFriction: 0.22,
    consensusThreshold: 0.70,
    projectedVoteDistribution: { approve: 5, reject: 1, abstain: 1 },
    groupthinkVulnerability: 0.18,
  },
  {
    committeeId: 'COM-002',
    committeeName: 'Risk Oversight Board',
    memberCount: 9,
    dissentFriction: 0.35,
    consensusThreshold: 0.75,
    projectedVoteDistribution: { approve: 7, reject: 2, abstain: 0 },
    groupthinkVulnerability: 0.12,
  },
  {
    committeeId: 'COM-003',
    committeeName: 'Audit & Compliance Committee',
    memberCount: 5,
    dissentFriction: 0.15,
    consensusThreshold: 0.80,
    projectedVoteDistribution: { approve: 4, reject: 1, abstain: 0 },
    groupthinkVulnerability: 0.08,
  },
  {
    committeeId: 'COM-004',
    committeeName: 'Strategic Capital Committee',
    memberCount: 6,
    dissentFriction: 0.28,
    consensusThreshold: 0.67,
    projectedVoteDistribution: { approve: 5, reject: 1, abstain: 0 },
    groupthinkVulnerability: 0.24,
  },
];

export function getCanonicalCommitteesTwin(): CommitteeTwin[] {
  return JSON.parse(JSON.stringify(CANONICAL_COMMITTEE_TWINS));
}

export function simulateCommitteeVote(
  twin: CommitteeTwin,
  shockIntensity: number // e.g. 0.0 to 1.0
): {
  approved: boolean;
  voteDistribution: { approve: number; reject: number; abstain: number };
  dissentRate: number;
  groupthinkAlert: boolean;
} {
  const safeShock = Math.max(0, Math.min(1, shockIntensity));
  const friction = twin.dissentFriction * (1 + safeShock);

  let reject = Math.round(twin.memberCount * friction);
  if (reject >= twin.memberCount) reject = twin.memberCount - 1;
  const abstain = safeShock > 0.5 ? 1 : 0;
  const approve = Math.max(0, twin.memberCount - reject - abstain);

  const approvalRate = approve / twin.memberCount;
  const approved = approvalRate >= twin.consensusThreshold;
  const dissentRate = Number((reject / twin.memberCount).toFixed(2));
  const groupthinkAlert = twin.groupthinkVulnerability > 0.20 && dissentRate < 0.10;

  return {
    approved,
    voteDistribution: { approve, reject, abstain },
    dissentRate,
    groupthinkAlert,
  };
}

export function createDigitalTwinState(
  twinId = 'TWIN-ORG-2026',
  committees = CANONICAL_COMMITTEE_TWINS
): DigitalTwinState {
  return {
    twinId,
    timestampUtc: new Date().toISOString(),
    committees: JSON.parse(JSON.stringify(committees)),
    simulatedOHI: 84.2,
    simulatedODEI: 86.4,
    isolated: true, // Strict INV-OI66
  };
}
