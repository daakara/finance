/**
 * Phase 31-M5: Intervention Planner Engine
 *
 * Implements:
 * - Structured intervention plans (PLAN-001 to PLAN-006)
 * - Sequential action orchestration, owner balancing, and progress tracking
 * - Validation rules for InterventionPlan schema
 */

import type { InterventionPlan, CoachingRecommendation } from '../../types/coaching-intelligence';
import { CANONICAL_COACHING_RECOMMENDATIONS } from './collectiveIntelligenceCoach';
import { sha256Hex } from './sha256';

export const CANONICAL_INTERVENTION_PLANS: InterventionPlan[] = [
  {
    planId: 'PLAN-001',
    committeeId: 'COM-001',
    title: "Q4 Equity Allocation De-biasing & Dissent Strengthening",
    generatedAtUtc: '2026-09-08T12:00:00Z',
    recommendations: [
      CANONICAL_COACHING_RECOMMENDATIONS[0], // REC-001
      CANONICAL_COACHING_RECOMMENDATIONS[6], // REC-007
    ],
    totalRiskReduction: 39.5,
    estimatedCompletionDays: 30,
    expectedOutcomeScore: 88.5,
    primaryOwnerId: 'USR-RSK-01',
  },
  {
    planId: 'PLAN-002',
    committeeId: 'COM-002',
    title: "Multi-Asset Volatility Stress & Liquidity Protection",
    generatedAtUtc: '2026-09-08T12:05:00Z',
    recommendations: [
      CANONICAL_COACHING_RECOMMENDATIONS[1], // REC-002
      CANONICAL_COACHING_RECOMMENDATIONS[7], // REC-008
    ],
    totalRiskReduction: 39.5,
    estimatedCompletionDays: 25,
    expectedOutcomeScore: 91.0,
    primaryOwnerId: 'USR-RSK-02',
  },
  {
    planId: 'PLAN-003',
    committeeId: 'COM-003',
    title: "Governance Determinism & Cryptographic Sign-off Reinforcement",
    generatedAtUtc: '2026-09-08T12:10:00Z',
    recommendations: [
      CANONICAL_COACHING_RECOMMENDATIONS[2], // REC-003
      CANONICAL_COACHING_RECOMMENDATIONS[8], // REC-009
    ],
    totalRiskReduction: 36.0,
    estimatedCompletionDays: 21,
    expectedOutcomeScore: 89.0,
    primaryOwnerId: 'USR-GOV-01',
  },
  {
    planId: 'PLAN-004',
    committeeId: 'COM-001',
    title: "Contrarian Thesis Onboarding for AI/Semiconductor Portfolio",
    generatedAtUtc: '2026-09-08T12:15:00Z',
    recommendations: [
      CANONICAL_COACHING_RECOMMENDATIONS[3], // REC-004
      CANONICAL_COACHING_RECOMMENDATIONS[9], // REC-010
    ],
    totalRiskReduction: 18.0,
    estimatedCompletionDays: 45,
    expectedOutcomeScore: 86.0,
    primaryOwnerId: 'USR-PM-02',
  },
  {
    planId: 'PLAN-005',
    committeeId: 'COM-002',
    title: "Cross-Committee Credit & Macro Transmission Playbook",
    generatedAtUtc: '2026-09-08T12:20:00Z',
    recommendations: [
      CANONICAL_COACHING_RECOMMENDATIONS[4], // REC-005
      CANONICAL_COACHING_RECOMMENDATIONS[10], // REC-011
    ],
    totalRiskReduction: 27.0,
    estimatedCompletionDays: 35,
    expectedOutcomeScore: 90.0,
    primaryOwnerId: 'USR-RSK-01',
  },
  {
    planId: 'PLAN-006',
    committeeId: 'COM-003',
    title: "Pre-Flight Evidence Automation & Friction Reduction",
    generatedAtUtc: '2026-09-08T12:25:00Z',
    recommendations: [
      CANONICAL_COACHING_RECOMMENDATIONS[5], // REC-006
      CANONICAL_COACHING_RECOMMENDATIONS[11], // REC-012
    ],
    totalRiskReduction: 17.0,
    estimatedCompletionDays: 40,
    expectedOutcomeScore: 85.0,
    primaryOwnerId: 'USR-GOV-02',
  },
];

export function validateInterventionPlan(plan: InterventionPlan): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (!plan.planId || !/^PLAN-[0-9]{3,}$/.test(plan.planId)) {
    errors.push(`Invalid planId "${plan.planId}" (must match ^PLAN-[0-9]{3,}$)`);
  }

  if (!plan.committeeId || !/^COM-[0-9]{3}$/.test(plan.committeeId)) {
    errors.push(`Invalid committeeId "${plan.committeeId}" (must match ^COM-[0-9]{3}$)`);
  }

  if (!plan.recommendations || !Array.isArray(plan.recommendations) || plan.recommendations.length === 0) {
    errors.push(`Intervention plan must contain at least 1 recommendation`);
  }

  if (typeof plan.totalRiskReduction !== 'number' || plan.totalRiskReduction < 0 || plan.totalRiskReduction > 100) {
    errors.push(`totalRiskReduction must be between 0 and 100`);
  }

  if (typeof plan.estimatedCompletionDays !== 'number' || plan.estimatedCompletionDays <= 0) {
    errors.push(`estimatedCompletionDays must be greater than 0`);
  }

  if (!plan.primaryOwnerId) {
    errors.push(`primaryOwnerId is required`);
  }

  return { valid: errors.length === 0, errors };
}

export function getInterventionPlans(committeeId?: string): InterventionPlan[] {
  if (!committeeId || committeeId === 'ALL') {
    return CANONICAL_INTERVENTION_PLANS;
  }
  return CANONICAL_INTERVENTION_PLANS.filter(p => p.committeeId === committeeId);
}

export function getInterventionPlanById(id: string): InterventionPlan | undefined {
  return CANONICAL_INTERVENTION_PLANS.find(p => p.planId === id);
}

export function hashInterventionPlans(plans: InterventionPlan[]): string {
  const sorted = [...plans].sort((a, b) => a.planId.localeCompare(b.planId));
  const payload = sorted.map(p => ({
    id: p.planId,
    cid: p.committeeId,
    recsCount: p.recommendations.length,
    riskReduction: p.totalRiskReduction,
    days: p.estimatedCompletionDays,
  }));
  return sha256Hex(JSON.stringify(payload));
}
