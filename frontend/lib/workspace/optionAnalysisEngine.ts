/**
 * Phase 31-M16: Option Analysis Engine
 *
 * Evaluates alternative options for decision packages (>= 3 options per package).
 * Computes multi-objective tradeoff scores, delta projections, and recommendation rankings.
 */

import { DecisionOption } from '../../types/executive-workspace-decision';

export function computeOptionTradeoffScore(opt: DecisionOption): number {
  // Tradeoff scoring formula:
  // (OHI Delta * 4.0) + (ODEI Delta * 3.0) - (Risk Delta * 2.5) + (ConfidencePct * 0.3) - (Cost / 25000)
  const ohiScore = opt.ohiDelta * 4.0;
  const odeiScore = opt.odeiDelta * 3.0;
  const riskScore = -opt.riskDelta * 2.5;
  const confidenceScore = opt.confidencePct * 0.3;
  const costPenalty = opt.implementationCostUSD / 25000;

  const raw = 50 + ohiScore + odeiScore + riskScore + confidenceScore - costPenalty;
  const clamped = Math.max(0, Math.min(100, raw));
  return Math.round(clamped * 10) / 10;
}

export function evaluateOptions(options: DecisionOption[]): DecisionOption[] {
  const scored = options.map(opt => ({
    ...opt,
    tradeoffScore: computeOptionTradeoffScore(opt),
  }));

  // Sort descending by tradeoff score to establish ranks
  scored.sort((a, b) => b.tradeoffScore - a.tradeoffScore);

  return scored.map((opt, idx) => ({
    ...opt,
    recommendationRank: idx + 1,
    isRecommended: idx === 0 && opt.governanceCompliance,
  }));
}

export function getRecommendedOption(options: DecisionOption[]): DecisionOption | undefined {
  const evaluated = evaluateOptions(options);
  return evaluated.find(o => o.isRecommended);
}

export function compareOptions(
  optA: DecisionOption,
  optB: DecisionOption
): {
  ohiDeltaDiff: number;
  riskDeltaDiff: number;
  costDiffUSD: number;
  tradeoffScoreDiff: number;
  superiorOptionId: string;
} {
  const ohiDeltaDiff = Math.round((optA.ohiDelta - optB.ohiDelta) * 10) / 10;
  const riskDeltaDiff = Math.round((optA.riskDelta - optB.riskDelta) * 10) / 10;
  const costDiffUSD = optA.implementationCostUSD - optB.implementationCostUSD;
  const tradeoffScoreDiff = Math.round((optA.tradeoffScore - optB.tradeoffScore) * 10) / 10;
  const superiorOptionId = optA.tradeoffScore >= optB.tradeoffScore ? optA.optionId : optB.optionId;

  return {
    ohiDeltaDiff,
    riskDeltaDiff,
    costDiffUSD,
    tradeoffScoreDiff,
    superiorOptionId,
  };
}
