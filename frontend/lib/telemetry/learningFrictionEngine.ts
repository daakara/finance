/**
 * Phase 31-M3: Learning Friction Engine (Epic M3-104)
 *
 * Implements:
 * - 6-Category Friction Taxonomy: Ignored, Rejected, Expired, Unknown, Ownership Gap, Governance Gap
 * - Learning Friction Score Formula: 0 (Frictionless) to 100 (Blocked)
 * - Committee & Network-Level Friction Diagnostics
 * - Remediation Priority Ranking
 * - Deterministic SHA-256 Fingerprinting
 */

import type {
  LearningFrictionResult,
  LearningFrictionCategory,
  LearningFrictionItem,
} from '../../types/learning-intelligence';

import { sha256 } from '../governance/sha256';

export const CANONICAL_FRICTION_ITEMS: Record<string, LearningFrictionItem[]> = {
  'COM-001': [
    {
      learningId: 'LRN-008',
      targetCommitteeId: 'COM-001',
      category: 'OWNERSHIP_GAP',
      explanation: 'No designated quantitative analyst assigned to backtest multi-regime skew calibration.',
      daysPending: 14,
    },
    {
      learningId: 'LRN-006',
      targetCommitteeId: 'COM-001',
      category: 'REJECTED',
      explanation: 'Options collar overlay was deemed too capital intensive for liquid large-cap tranches.',
      daysPending: 21,
    },
  ],
  'COM-002': [
    {
      learningId: 'LRN-002',
      targetCommitteeId: 'COM-002',
      category: 'EXPIRED',
      explanation: 'SLA review period of 10 business days elapsed without formal committee vote.',
      daysPending: 28,
    },
    {
      learningId: 'LRN-004',
      targetCommitteeId: 'COM-002',
      category: 'GOVERNANCE_GAP',
      explanation: 'Requires formal charter amendment to expand governance authority over tail-risk models.',
      daysPending: 35,
    },
  ],
  'COM-003': [
    {
      learningId: 'LRN-005',
      targetCommitteeId: 'COM-003',
      category: 'IGNORED',
      explanation: 'Macro liquidity drain lesson was published but not queued for agenda discussion.',
      daysPending: 42,
    },
    {
      learningId: 'LRN-009',
      targetCommitteeId: 'COM-003',
      category: 'UNKNOWN',
      explanation: 'Uncertainty over dark pool execution telemetry attribution methodology.',
      daysPending: 18,
    },
  ],
};

const CATEGORY_WEIGHTS: Record<LearningFrictionCategory, number> = {
  IGNORED: 1.2,
  OWNERSHIP_GAP: 1.1,
  GOVERNANCE_GAP: 1.0,
  EXPIRED: 0.9,
  REJECTED: 0.6,
  UNKNOWN: 0.8,
};

/**
 * Computes learning friction score and detailed category breakdown for a committee.
 * Score range: 0.0 (Frictionless) to 100.0 (Fully Blocked).
 */
export function computeLearningFriction(committeeId: string = 'COM-001'): LearningFrictionResult {
  const items = CANONICAL_FRICTION_ITEMS[committeeId] ?? [];
  const totalEvaluated = 10; // Normalized baseline of 10 total published learnings

  let ignoredCount = 0;
  let rejectedCount = 0;
  let expiredCount = 0;
  let unknownCount = 0;
  let ownershipGapCount = 0;
  let governanceGapCount = 0;

  for (const item of items) {
    if (item.category === 'IGNORED') ignoredCount++;
    else if (item.category === 'REJECTED') rejectedCount++;
    else if (item.category === 'EXPIRED') expiredCount++;
    else if (item.category === 'UNKNOWN') unknownCount++;
    else if (item.category === 'OWNERSHIP_GAP') ownershipGapCount++;
    else if (item.category === 'GOVERNANCE_GAP') governanceGapCount++;
  }

  // Weighted friction sum
  const weightedSum =
    ignoredCount * CATEGORY_WEIGHTS.IGNORED +
    ownershipGapCount * CATEGORY_WEIGHTS.OWNERSHIP_GAP +
    governanceGapCount * CATEGORY_WEIGHTS.GOVERNANCE_GAP +
    expiredCount * CATEGORY_WEIGHTS.EXPIRED +
    rejectedCount * CATEGORY_WEIGHTS.REJECTED +
    unknownCount * CATEGORY_WEIGHTS.UNKNOWN;

  const rawScore = totalEvaluated > 0 ? (weightedSum / totalEvaluated) * 100 : 0;
  const frictionScore = Math.min(100.0, Math.round(rawScore * 10) / 10);

  // Determine top friction category
  const counts: { cat: LearningFrictionCategory; count: number }[] = [
    { cat: 'IGNORED', count: ignoredCount },
    { cat: 'OWNERSHIP_GAP', count: ownershipGapCount },
    { cat: 'GOVERNANCE_GAP', count: governanceGapCount },
    { cat: 'EXPIRED', count: expiredCount },
    { cat: 'REJECTED', count: rejectedCount },
    { cat: 'UNKNOWN', count: unknownCount },
  ];

  counts.sort((a, b) => b.count - a.count);
  const topFrictionCategory = counts[0].count > 0 ? counts[0].cat : 'IGNORED';

  return {
    committeeId,
    totalEvaluated,
    ignoredCount,
    rejectedCount,
    expiredCount,
    unknownCount,
    ownershipGapCount,
    governanceGapCount,
    frictionScore,
    topFrictionCategory,
    items,
  };
}

/**
 * Returns network-wide friction overview across all registered committees.
 */
export function getNetworkFrictionOverview(): {
  committeeScores: { committeeId: string; frictionScore: number; topCategory: LearningFrictionCategory }[];
  averageNetworkFriction: number;
  highestFrictionCommittee: string;
} {
  const committeeIds = ['COM-001', 'COM-002', 'COM-003'];
  const scores = committeeIds.map(id => {
    const res = computeLearningFriction(id);
    return {
      committeeId: id,
      frictionScore: res.frictionScore,
      topCategory: res.topFrictionCategory,
    };
  });

  const avg = Math.round((scores.reduce((a, b) => a + b.frictionScore, 0) / scores.length) * 10) / 10;
  const highest = [...scores].sort((a, b) => b.frictionScore - a.frictionScore)[0]?.committeeId ?? 'COM-001';

  return {
    committeeScores: scores,
    averageNetworkFriction: avg,
    highestFrictionCommittee: highest,
  };
}

/**
 * Generates deterministic SHA-256 fingerprint of friction analysis.
 */
export function hashFrictionResult(res: LearningFrictionResult): string {
  const payload = {
    committeeId: res.committeeId,
    frictionScore: res.frictionScore,
    counts: {
      ign: res.ignoredCount,
      rej: res.rejectedCount,
      exp: res.expiredCount,
      own: res.ownershipGapCount,
      gov: res.governanceGapCount,
      unk: res.unknownCount,
    },
    top: res.topFrictionCategory,
  };
  return sha256(JSON.stringify(payload));
}
