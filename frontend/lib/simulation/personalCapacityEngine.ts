/**
 * Personal Capacity & Invariant Verification Engine (Horizon 5)
 *
 * Implements:
 * - INV-OI75-P (Personal Capacity Feasibility Invariant)
 * - 168-Hour Weekly Time Budget Engine
 * - Multidimensional Resource Utilization (Time, Money, Energy, Attention)
 * - Automated Resource Rebalancing Suggestions
 */

import {
  PersonalCapacity,
  WeeklyTimeBudget,
  PersonalDemand,
  CapacityCheckResult,
  CapacityViolationType,
  AllocationDomain,
  PersonalAllocationPlan,
} from "../../types/personal-digital-twin";

export const DEFAULT_WEEKLY_TIME_BUDGET: WeeklyTimeBudget = {
  totalHours: 168,
  sleepHours: 56, // 8h/night (well above the 49h floor)
  workHours: 40,
  commuteHours: 5,
  familyHours: 20,
  exerciseHours: 5,
  adminHours: 17,
  discretionaryHours: 25, // 168 - 143 = 25 available hours
};

export const DEFAULT_PERSONAL_CAPACITY: PersonalCapacity = {
  weeklyHours: 25,
  monthlyBudget: 500, // €500 or $500 monthly growth budget
  energyCapacity: 75, // 75/100 energy battery
  attentionCapacity: 80, // 80/100 focus units
};

export const DEFAULT_ALLOCATION_DOMAINS: AllocationDomain[] = [
  {
    id: "career",
    name: "Career & Craft",
    hours: 8,
    impactScore: 18,
    color: "#3b82f6", // Blue
    description: "System design practice, architecture mastery, and portfolio acceleration",
  },
  {
    id: "learning",
    name: "AI & Deep Learning",
    hours: 5,
    impactScore: 14,
    color: "#10b981", // Emerald
    description: "Model fine-tuning, quantitative algorithms, and research papers",
  },
  {
    id: "health",
    name: "Aerobic & Strength Fitness",
    hours: 4,
    impactScore: 12,
    color: "#f59e0b", // Amber
    description: "Zone-2 running, progressive overload resistance training, recovery mobility",
  },
  {
    id: "relationships",
    name: "Core Relationships & Family",
    hours: 3,
    impactScore: 9,
    color: "#ec4899", // Pink
    description: "Uninterrupted presence, shared experiences, support network reciprocity",
  },
  {
    id: "finance",
    name: "Capital & Wealth Allocation",
    hours: 2,
    impactScore: 5,
    color: "#8b5cf6", // Purple
    description: "Portfolio rebalancing, tax optimization, runway modeling",
  },
];

/**
 * Calculates discretionary hours within the immutable 168-hour weekly budget.
 */
export function calculateWeeklyTimeBudget(
  partial: Partial<WeeklyTimeBudget>
): WeeklyTimeBudget {
  const sleep = partial.sleepHours ?? 56;
  const work = partial.workHours ?? 40;
  const commute = partial.commuteHours ?? 5;
  const family = partial.familyHours ?? 20;
  const exercise = partial.exerciseHours ?? 5;
  const admin = partial.adminHours ?? 17;

  const allocated = sleep + work + commute + family + exercise + admin;
  const discretionary = Math.max(0, 168 - allocated);

  return {
    totalHours: 168,
    sleepHours: sleep,
    workHours: work,
    commuteHours: commute,
    familyHours: family,
    exerciseHours: exercise,
    adminHours: admin,
    discretionaryHours: discretionary,
  };
}

/**
 * Validates INV-OI75-P (Personal Capacity Feasibility Invariant).
 *
 * A plan is mathematically feasible iff:
 * - demand.weeklyHours <= capacity.weeklyHours
 * - sleepHours >= 49 (7h/night physiological floor)
 * - demand.monthlyBudget <= capacity.monthlyBudget
 * - demand.energyDemand <= capacity.energyCapacity
 * - demand.attentionDemand <= capacity.attentionCapacity
 */
export function verifyPersonalCapacity(
  capacity: PersonalCapacity,
  demand: PersonalDemand,
  sleepHours: number = 56
): CapacityCheckResult {
  const violations: CapacityViolationType[] = [];

  // 1. Time capacity
  if (demand.weeklyHours > capacity.weeklyHours) {
    violations.push("TIME_CAPACITY_EXCEEDED");
  }

  // 2. Sleep floor (INV-OI75-P explicit physiological recovery requirement)
  if (sleepHours < 49) {
    violations.push("SLEEP_FLOOR_VIOLATION");
  }

  // 3. Financial capacity
  if (demand.monthlyBudget > capacity.monthlyBudget) {
    violations.push("MONEY_CAPACITY_EXCEEDED");
  }

  // 4. Energy capacity
  if (demand.energyDemand > capacity.energyCapacity) {
    violations.push("ENERGY_CAPACITY_EXCEEDED");
  }

  // 5. Attention / focus capacity
  if (demand.attentionDemand > capacity.attentionCapacity) {
    violations.push("ATTENTION_CAPACITY_EXCEEDED");
  }

  const isFeasible = violations.length === 0;

  const timePct = Math.round((demand.weeklyHours / Math.max(1, capacity.weeklyHours)) * 100);
  const moneyPct = Math.round((demand.monthlyBudget / Math.max(1, capacity.monthlyBudget)) * 100);
  const energyPct = Math.round((demand.energyDemand / Math.max(1, capacity.energyCapacity)) * 100);
  const attentionPct = Math.round((demand.attentionDemand / Math.max(1, capacity.attentionCapacity)) * 100);

  const rebalanceSuggestions: Array<{
    domain: string;
    suggestedReductionHours: number;
    reason: string;
  }> = [];

  if (demand.weeklyHours > capacity.weeklyHours) {
    const excess = demand.weeklyHours - capacity.weeklyHours;
    rebalanceSuggestions.push({
      domain: "Discretionary Allocations",
      suggestedReductionHours: excess,
      reason: `Weekly time exceeds available capacity by ${excess}h. Reduce low-leverage activities to restore balance.`,
    });
  }

  if (demand.attentionDemand > capacity.attentionCapacity) {
    rebalanceSuggestions.push({
      domain: "Deep Work / Learning",
      suggestedReductionHours: 2,
      reason: "High cognitive fragmentation. Batch meetings or defer one complex learning module.",
    });
  }

  return {
    isFeasible,
    violations,
    utilization: {
      timePct,
      moneyPct,
      energyPct,
      attentionPct,
    },
    rebalanceSuggestions,
  };
}

/**
 * Recalculates projected Life Health Index (LHI) based on interactive allocations.
 */
export function calculateProjectedOutcomes(
  allocations: Record<string, number>,
  baseLhi: number = 82.4
): PersonalAllocationPlan {
  const totalHours = Object.values(allocations).reduce((sum, h) => sum + (h || 0), 0);

  // Marginal gains based on domain efficiency
  const careerGain = (allocations["career"] || 0) * 0.45;
  const learningGain = (allocations["learning"] || 0) * 0.40;
  const healthGain = (allocations["health"] || 0) * 0.50;
  const relGain = (allocations["relationships"] || 0) * 0.35;
  const finGain = (allocations["finance"] || 0) * 0.30;

  // Diminishing returns & fatigue drag if totalHours > 22
  const overloadPenalty = totalHours > 22 ? (totalHours - 22) * 0.8 : 0;

  const net6mGain = Math.min(12, Math.max(0, (careerGain + learningGain + healthGain + relGain + finGain) * 0.6 - overloadPenalty));
  const net12mGain = Math.min(16, Math.max(0, (careerGain + learningGain + healthGain + relGain + finGain) * 1.1 - overloadPenalty * 1.5));

  const projectedLhiCurrent = baseLhi;
  const projectedLhi6m = Math.min(100, Number((baseLhi + net6mGain).toFixed(1)));
  const projectedLhi12m = Math.min(100, Number((baseLhi + net12mGain).toFixed(1)));

  // Confidence is high if allocations are moderate, lower if extreme
  const confidencePct = Math.max(65, Math.min(94, Math.round(90 - Math.abs(totalHours - 18) * 1.5)));

  return {
    domains: { ...allocations },
    projectedLhiCurrent,
    projectedLhi6m,
    projectedLhi12m,
    confidencePct,
  };
}
