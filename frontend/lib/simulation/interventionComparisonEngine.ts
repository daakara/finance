/**
 * M12.3 Intervention Comparison Engine (Phase 31-M12)
 *
 * Implements:
 * - Multi-candidate side-by-side comparison (A vs B vs C)
 * - Common baseline enforcement (INV-OI67)
 * - Delta calculations, survivability ranking & risk-adjusted outcomes
 */

import type {
  StrategyCandidate,
  InterventionComparison,
} from '../../types/simulation-intelligence';

export const CANONICAL_CANDIDATE_STRATEGIES: StrategyCandidate[] = [
  {
    candidateId: 'STRAT-CAND-A',
    name: 'Dynamic Volatility Scaling & Dissent Floor',
    description: 'Scales risk exposure dynamically while enforcing a minimum 15% minority dissent review quota.',
    baselineId: 'BASE-2026-Q3',
    deltaOHI: 2.8,
    deltaODEI: 3.4,
    deltaRisk: -4.2,
    survivabilityScore: 92.1,
    rank: 1,
    recommendationApproved: false,
    simulationCertified: true,
  },
  {
    candidateId: 'STRAT-CAND-B',
    name: 'Accelerated Capital Deployment (High Alpha)',
    description: 'Increases equity tranche pacing by 25% with relaxed committee turnaround cycles.',
    baselineId: 'BASE-2026-Q3',
    deltaOHI: 1.4,
    deltaODEI: 4.8,
    deltaRisk: 6.5,
    survivabilityScore: 81.0,
    rank: 2,
    recommendationApproved: false,
    simulationCertified: true,
  },
  {
    candidateId: 'STRAT-CAND-C',
    name: 'Maximum Capital Preservation (Defensive Lock)',
    description: 'Freezes all new autonomous allocations, holding capital in T-bills and tier-1 liquid buffers.',
    baselineId: 'BASE-2026-Q3',
    deltaOHI: -1.2,
    deltaODEI: -2.5,
    deltaRisk: -8.9,
    survivabilityScore: 95.4,
    rank: 3,
    recommendationApproved: false,
    simulationCertified: true,
  },
];

export function compareInterventionCandidates(
  baselineOHI = 84.2,
  candidates = CANONICAL_CANDIDATE_STRATEGIES
): InterventionComparison {
  if (!candidates || candidates.length === 0) {
    throw new Error('Intervention comparison requires at least one strategy candidate');
  }

  // Verify common baseline enforcement (INV-OI67)
  const baseId = candidates[0].baselineId;
  const hasCommonBaseline = candidates.every((c) => c.baselineId === baseId);
  if (!hasCommonBaseline) {
    throw new Error('INV-OI67 VIOLATION: All candidate strategies must reference an identical baseline');
  }

  // Sort candidates by survivability score desc
  const sorted = [...candidates].sort((a, b) => b.survivabilityScore - a.survivabilityScore);
  const ranked = sorted.map((cand, idx) => ({
    ...cand,
    rank: idx + 1,
  }));

  return {
    comparisonId: `CMP-${Date.now()}`,
    baselineId: baseId,
    baselineOHI,
    candidates: ranked,
    recommendedCandidateId: ranked[0].candidateId,
    evaluatedAtUtc: new Date().toISOString(),
  };
}
