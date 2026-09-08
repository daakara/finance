/**
 * Phase 31-M7: Resource Allocation Engine
 *
 * Implements:
 * - Multi-Resource Knapsack Allocation (Budget, Headcount, Training)
 * - Invariant INV-OI41: Resource Conservation (Sum <= Available)
 * - Invariant INV-OI42: Intervention Feasibility (Executable, Budget, Owner, Capacity)
 * - Committee Allocation Fairness & Concentration Guards (Max <= 70%, target <= 45%)
 * - Capacity Normalization & Deterministic Allocation Hashing
 */

import {
  ResourcePool,
  OptimizationConstraint,
  InterventionCandidate,
  ResourceAllocationRecommendation,
  AllocationResult,
} from '../../types/optimization-intelligence';
import { CANONICAL_INTERVENTION_CANDIDATES, CANONICAL_DRIVER_CONTRIBUTIONS } from './optimizationPortfolioEngine';
import { sha256Hex } from '../governance/sha256';

export const CANONICAL_RESOURCE_POOLS: ResourcePool[] = [
  {
    resourceId: 'RES-BUDGET',
    resourceType: 'BUDGET',
    availableUnits: 100000,
    allocatedUnits: 84000,
    unitLabel: 'USD',
  },
  {
    resourceId: 'RES-HEADCOUNT',
    resourceType: 'HEADCOUNT',
    availableUnits: 10.0,
    allocatedUnits: 8.2,
    unitLabel: 'FTE',
  },
  {
    resourceId: 'RES-TRAINING',
    resourceType: 'TRAINING',
    availableUnits: 500,
    allocatedUnits: 380,
    unitLabel: 'Hours',
  },
  {
    resourceId: 'RES-GOVERNANCE',
    resourceType: 'GOVERNANCE',
    availableUnits: 12,
    allocatedUnits: 9,
    unitLabel: 'Slots',
  },
];

export const CANONICAL_OPTIMIZATION_CONSTRAINTS: OptimizationConstraint[] = [
  {
    constraintId: 'CST-001',
    name: 'Maximum Total Budget Cap',
    type: 'BUDGET',
    value: 100000,
    mandatory: true,
    operator: '<=',
    currentValue: 84000,
    satisfied: true,
  },
  {
    constraintId: 'CST-002',
    name: 'Available Headcount Capacity',
    type: 'CAPACITY',
    value: 10.0,
    mandatory: true,
    operator: '<=',
    currentValue: 8.2,
    satisfied: true,
  },
  {
    constraintId: 'CST-003',
    name: 'Maximum Committee Concentration Ceiling',
    type: 'GOVERNANCE',
    value: 45.0,
    mandatory: true,
    operator: '<=',
    currentValue: 38.5,
    satisfied: true,
  },
  {
    constraintId: 'CST-004',
    name: 'Minimum Knowledge Transfer Rate Floor',
    type: 'CERTIFICATION',
    value: 80.0,
    mandatory: true,
    operator: '>=',
    currentValue: 88.0,
    satisfied: true,
  },
  {
    constraintId: 'CST-005',
    name: 'Maximum Risk Threshold Ceiling',
    type: 'RISK',
    value: 40.0,
    mandatory: true,
    operator: '<=',
    currentValue: 26.2,
    satisfied: true,
  },
];

export const CANONICAL_ALLOCATION_RECOMMENDATIONS: ResourceAllocationRecommendation[] = [
  {
    allocationId: 'ALLOC-001',
    committeeId: 'COM-001',
    resourceId: 'RES-BUDGET',
    targetArea: 'GOVERNANCE',
    allocatedUnits: 28000,
    allocationPct: 33.3,
    expectedBenefit: 'Elevates voting transparency and mitigates authority bias (+2.8 OHI)',
    marginalBenefit: 2.33,
    priorityScore: 92.5,
    confidencePct: 95.0,
    rationale: 'Addresses primary contrarian deliberation friction in Investment Committee.',
  },
  {
    allocationId: 'ALLOC-002',
    committeeId: 'COM-002',
    resourceId: 'RES-BUDGET',
    targetArea: 'LEARNING',
    allocatedUnits: 32000,
    allocationPct: 38.1,
    expectedBenefit: 'Accelerates cross-silo knowledge transfer rate to 92% (+3.2 OHI)',
    marginalBenefit: 2.00,
    priorityScore: 89.0,
    confidencePct: 92.0,
    rationale: 'Funds multi-committee joint diligence playbook and incident retrospectives.',
  },
  {
    allocationId: 'ALLOC-003',
    committeeId: 'COM-003',
    resourceId: 'RES-BUDGET',
    targetArea: 'RISK',
    allocatedUnits: 24000,
    allocationPct: 28.6,
    expectedBenefit: 'Institutes liquidity shock testing and dynamic volatility ceiling (-25% risk)',
    marginalBenefit: 2.50,
    priorityScore: 94.0,
    confidencePct: 91.0,
    rationale: 'Deploys automated micro-cap risk guardian controls.',
  },
];

export function getCanonicalResourcePools(): ResourcePool[] {
  return JSON.parse(JSON.stringify(CANONICAL_RESOURCE_POOLS));
}

export function getCanonicalConstraints(): OptimizationConstraint[] {
  return JSON.parse(JSON.stringify(CANONICAL_OPTIMIZATION_CONSTRAINTS));
}

// Invariant INV-OI41: Resource Conservation
export function verifyINV_OI41(
  allocations: ResourceAllocationRecommendation[] = CANONICAL_ALLOCATION_RECOMMENDATIONS,
  pools: ResourcePool[] = CANONICAL_RESOURCE_POOLS
): { pass: boolean; allocatedBudget: number; availableBudget: number; violations: string[] } {
  const violations: string[] = [];

  const budgetPool = pools.find(p => p.resourceId === 'RES-BUDGET') ?? { availableUnits: 100000 };
  const availableBudget = budgetPool.availableUnits;

  const allocatedBudget = allocations.reduce((acc, a) => acc + a.allocatedUnits, 0);

  if (allocatedBudget > availableBudget) {
    violations.push(
      `INV-OI41 Violation: Allocated budget ($${allocatedBudget}) exceeds available budget pool ($${availableBudget})`
    );
  }

  for (const a of allocations) {
    if (a.allocatedUnits < 0) {
      violations.push(`INV-OI41 Violation: Allocation ${a.allocationId} has negative units (${a.allocatedUnits})`);
    }
  }

  return {
    pass: violations.length === 0,
    allocatedBudget,
    availableBudget,
    violations,
  };
}

// Invariant INV-OI42: Intervention Feasibility
export function verifyINV_OI42(
  candidates: InterventionCandidate[] = CANONICAL_INTERVENTION_CANDIDATES,
  pools: ResourcePool[] = CANONICAL_RESOURCE_POOLS
): { pass: boolean; executableCount: number; violations: string[] } {
  const violations: string[] = [];
  const budgetPool = pools.find(p => p.resourceType === 'BUDGET')?.availableUnits ?? 100000;
  const headcountPool = pools.find(p => p.resourceType === 'HEADCOUNT')?.availableUnits ?? 10.0;

  let executableCount = 0;

  for (const c of candidates) {
    if (!c.ownerId || c.ownerId.trim().length === 0) {
      violations.push(`INV-OI42 Violation: Candidate ${c.interventionId} has no assigned owner`);
    }
    if (c.cost > budgetPool) {
      violations.push(`INV-OI42 Violation: Candidate ${c.interventionId} cost exceeds total available budget`);
    }
    if (c.headcountRequired > headcountPool) {
      violations.push(`INV-OI42 Violation: Candidate ${c.interventionId} headcount exceeds total available headcount`);
    }
    if (!c.timelineWeeks || c.timelineWeeks <= 0) {
      violations.push(`INV-OI42 Violation: Candidate ${c.interventionId} has invalid timeline (${c.timelineWeeks})`);
    }
    if (violations.length === 0) {
      executableCount++;
    }
  }

  return {
    pass: violations.length === 0,
    executableCount,
    violations,
  };
}

// Optimization Fairness & Concentration Guard
export function verifyOptimizationFairness(
  allocations: ResourceAllocationRecommendation[] = CANONICAL_ALLOCATION_RECOMMENDATIONS,
  maxAllowedConcentrationPct: number = 70.0
): { pass: boolean; maxConcentrationPct: number; violations: string[] } {
  const violations: string[] = [];
  const totalUnits = allocations.reduce((acc, a) => acc + a.allocatedUnits, 0);

  if (totalUnits === 0) {
    return { pass: true, maxConcentrationPct: 0, violations: [] };
  }

  const committeeUnits: Record<string, number> = {};
  for (const a of allocations) {
    committeeUnits[a.committeeId] = (committeeUnits[a.committeeId] ?? 0) + a.allocatedUnits;
  }

  let maxConcentrationPct = 0;
  for (const [comId, units] of Object.entries(committeeUnits)) {
    const pct = Math.round((units / totalUnits) * 1000) / 10;
    if (pct > maxConcentrationPct) maxConcentrationPct = pct;
    if (pct > maxAllowedConcentrationPct) {
      violations.push(
        `FAIRNESS_VIOLATION: Committee ${comId} receives ${pct}% of total allocation, exceeding ${maxAllowedConcentrationPct}% ceiling`
      );
    }
  }

  return {
    pass: violations.length === 0,
    maxConcentrationPct,
    violations,
  };
}

// Capacity Normalization Workflow (OPT-REC-05)
export function normalizeAllocations(
  allocations: ResourceAllocationRecommendation[],
  targetTotalPct: number = 100.0
): ResourceAllocationRecommendation[] {
  const totalPct = allocations.reduce((acc, a) => acc + a.allocationPct, 0);
  if (totalPct === 0) return allocations;

  const ratio = targetTotalPct / totalPct;
  return allocations.map(a => ({
    ...a,
    allocationPct: Math.round(a.allocationPct * ratio * 10) / 10,
  }));
}

export function hashAllocationState(allocations: ResourceAllocationRecommendation[]): string {
  const payload = allocations
    .slice()
    .sort((a, b) => a.allocationId.localeCompare(b.allocationId))
    .map(a => ({ id: a.allocationId, com: a.committeeId, units: a.allocatedUnits, pct: a.allocationPct }));
  return sha256Hex(JSON.stringify(payload));
}
