/**
 * Phase 31-M5: Coaching Diversity Engine
 *
 * Implements:
 * - Alternative intervention path generation
 * - Coaching diversity score computation (entropy-based)
 * - Invariant INV-OI27 (Coaching Diversity >= 80.0)
 * - Recommendation stagnation detection (RECOMMENDATION_STAGNATION)
 */

import type { CoachingRecommendation } from '../../types/coaching-intelligence';
import { CANONICAL_COACHING_RECOMMENDATIONS } from './collectiveIntelligenceCoach';
import { sha256Hex } from './sha256';

export function computeCoachingDiversityScore(recommendations: CoachingRecommendation[]): number {
  if (recommendations.length === 0) return 100.0;

  // 1. Category spread entropy
  const categoryCounts: Record<string, number> = {};
  recommendations.forEach(r => {
    categoryCounts[r.category] = (categoryCounts[r.category] || 0) + 1;
  });

  const categories = Object.keys(categoryCounts);
  const total = recommendations.length;

  let entropy = 0;
  categories.forEach(cat => {
    const p = categoryCounts[cat] / total;
    if (p > 0) {
      entropy -= p * Math.log2(p);
    }
  });

  const maxEntropy = Math.log2(Math.max(categories.length, 6));
  const normalizedEntropy = maxEntropy > 0 ? (entropy / maxEntropy) : 1.0;

  // 2. Alternative options presence ratio
  const withAlternatives = recommendations.filter(r => r.alternatives && r.alternatives.length >= 2).length;
  const altRatio = withAlternatives / total;

  const score = (normalizedEntropy * 60.0) + (altRatio * 40.0);
  return Math.min(100.0, Math.round(score * 10) / 10);
}

// Invariant INV-OI27: Coaching Diversity
export function verifyINV_OI27(recommendations: CoachingRecommendation[]): { pass: boolean; diversityScore: number; violations: string[] } {
  const violations: string[] = [];
  const score = computeCoachingDiversityScore(recommendations);

  if (score < 80.0) {
    violations.push(`INV-OI27 Violation: Coaching diversity score ${score} is below threshold 80.0`);
  }

  return {
    pass: violations.length === 0,
    diversityScore: score,
    violations,
  };
}

// Detect Recommendation Stagnation (repeated emission of identical recommendation)
export function detectRecommendationStagnation(
  history: { recommendationId: string; issuedQuarter: string; outcomeImproved: boolean }[]
): { stagnationDetected: boolean; flaggedRecommendationId?: string; explanation?: string } {
  const countMap: Record<string, number> = {};
  const unimprovedMap: Record<string, number> = {};

  history.forEach(h => {
    countMap[h.recommendationId] = (countMap[h.recommendationId] || 0) + 1;
    if (!h.outcomeImproved) {
      unimprovedMap[h.recommendationId] = (unimprovedMap[h.recommendationId] || 0) + 1;
    }
  });

  for (const [recId, count] of Object.entries(countMap)) {
    if (count >= 3 && (unimprovedMap[recId] || 0) >= 3) {
      return {
        stagnationDetected: true,
        flaggedRecommendationId: recId,
        explanation: `Recommendation ${recId} issued ${count} times without positive outcome improvement. Stagnation alert triggered.`,
      };
    }
  }

  return { stagnationDetected: false };
}

export function hashDiversityState(recommendations: CoachingRecommendation[]): string {
  const score = computeCoachingDiversityScore(recommendations);
  return sha256Hex(`DIVERSITY_SCORE:${score.toFixed(2)}:COUNT:${recommendations.length}`);
}
