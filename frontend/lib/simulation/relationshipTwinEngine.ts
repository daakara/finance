/**
 * Horizon 8: Relational Twin & Household Intelligence Engine
 *
 * Implements:
 * - Multi-Twin Relational Model (Me + Partner + Child + Parent + Co-founder)
 * - Shared Resource Ledger (Time Windows, Household Capital, Caretaking Duties)
 * - INV-OI90-P (Shared Resource Integrity Invariant: Zero Double-Booking)
 * - INV-OI91-P (Relational Impact Visibility Invariant: 360-Degree Disclosure)
 * - Household Health Index (HHI) Multi-Twin Weighted Formula
 * - 4 Canonical Multi-Twin Scenarios (Relocation, Startup, Master's, Rhythm)
 */

import {
  RelationshipRole,
  RelationshipNode,
  SharedResourceType,
  CommitmentAllocation,
  SharedResourceAllocation,
  HouseholdHealthIndex,
  RelationalMemberImpact,
  RelationalScenarioResult,
} from "../../types/personal-digital-twin";

// ============================================================================
// 1. CANONICAL RELATIONAL TWIN NODES
// ============================================================================

export const CANONICAL_RELATIONSHIP_NODES: Record<string, RelationshipNode> = {
  REL_PARTNER: {
    relationshipId: "REL_PARTNER",
    name: "Elena",
    role: "PARTNER",
    relevanceWeight: 0.35,
    impactSensitivity: 1.4,
    sharedResources: [
      "SHARED_EVENING_BLOCKS",
      "WEEKEND_FAMILY_RESERVE",
      "HOUSEHOLD_DISCRETIONARY_BUDGET",
      "CARETAKING_PRIMARY_DUTY",
    ],
    baselineWellbeing: 82,
    currentWellbeing: 82,
    sentimentAlignment: 88,
    description: "Life partner & co-parent with independent engineering leadership career",
  },
  REL_CHILD: {
    relationshipId: "REL_CHILD",
    name: "Leo (7)",
    role: "CHILD",
    relevanceWeight: 0.15,
    impactSensitivity: 1.2,
    sharedResources: [
      "WEEKEND_FAMILY_RESERVE",
      "CARETAKING_PRIMARY_DUTY",
    ],
    baselineWellbeing: 88,
    currentWellbeing: 88,
    sentimentAlignment: 92,
    description: "Primary school child requiring structured evening & weekend developmental presence",
  },
  REL_PARENT: {
    relationshipId: "REL_PARENT",
    name: "David",
    role: "PARENT",
    relevanceWeight: 0.05,
    impactSensitivity: 0.9,
    sharedResources: [
      "HOUSEHOLD_DISCRETIONARY_BUDGET",
      "CARETAKING_PRIMARY_DUTY",
    ],
    baselineWellbeing: 74,
    currentWellbeing: 74,
    sentimentAlignment: 80,
    description: "Aging parent living within 20 minutes requiring intermittent healthcare coordination",
  },
  REL_COFOUNDER: {
    relationshipId: "REL_COFOUNDER",
    name: "Marcus",
    role: "COFOUNDER",
    relevanceWeight: 0.05,
    impactSensitivity: 1.1,
    sharedResources: [
      "SHARED_EVENING_BLOCKS",
    ],
    baselineWellbeing: 76,
    currentWellbeing: 76,
    sentimentAlignment: 85,
    description: "Technical venture partner sharing product architecture syncs",
  },
};

// ============================================================================
// 2. CANONICAL SHARED RESOURCES LEDGER
// ============================================================================

export const CANONICAL_SHARED_RESOURCES: SharedResourceAllocation[] = [
  {
    resourceId: "SHARED_EVENING_BLOCKS",
    name: "Shared Evening Connection (Mon-Fri 19:00-21:00)",
    type: "TIME_BLOCK",
    capacityUnits: 10,
    unit: "h/wk",
    allocatedCommitments: [
      {
        commitmentId: "COM_DINNER",
        allocatedTo: "FAMILY_DINNER_AND_BEDTIME",
        amount: 8,
        timeWindow: "WEEKDAYS_1900_2030",
        priority: 1,
      },
      {
        commitmentId: "COM_PARTNER_SYNC",
        allocatedTo: "PARTNER_UNWIND",
        amount: 2,
        timeWindow: "WEEKDAYS_2030_2100",
        priority: 2,
      },
    ],
    totalAllocated: 10,
    isDoubleBooked: false,
    conflictDetails: [],
  },
  {
    resourceId: "WEEKEND_FAMILY_RESERVE",
    name: "Protected Weekend Family Block (Sat-Sun)",
    type: "TIME_BLOCK",
    capacityUnits: 16,
    unit: "h/wk",
    allocatedCommitments: [
      {
        commitmentId: "COM_SAT_OUTING",
        allocatedTo: "OUTDOOR_ACTIVITIES",
        amount: 8,
        timeWindow: "SAT_1000_1800",
        priority: 1,
      },
      {
        commitmentId: "COM_SUN_MEAL",
        allocatedTo: "FAMILY_GATHERING",
        amount: 6,
        timeWindow: "SUN_1200_1800",
        priority: 2,
      },
    ],
    totalAllocated: 14,
    isDoubleBooked: false,
    conflictDetails: [],
  },
  {
    resourceId: "HOUSEHOLD_DISCRETIONARY_BUDGET",
    name: "Household Joint Capital Pool",
    type: "FINANCIAL_CAPITAL",
    capacityUnits: 1800,
    unit: "$/mo",
    allocatedCommitments: [
      {
        commitmentId: "COM_CHILD_EXTRAS",
        allocatedTo: "CHILD_SPORTS_AND_TUTORING",
        amount: 450,
        priority: 1,
      },
      {
        commitmentId: "COM_VACATION_FUND",
        allocatedTo: "ANNUAL_TRAVEL_RESERVE",
        amount: 600,
        priority: 2,
      },
      {
        commitmentId: "COM_PARENT_SUPPORT",
        allocatedTo: "ELDER_WELLNESS_SUBSIDY",
        amount: 300,
        priority: 2,
      },
    ],
    totalAllocated: 1350,
    isDoubleBooked: false,
    conflictDetails: [],
  },
  {
    resourceId: "CARETAKING_PRIMARY_DUTY",
    name: "Essential Caretaking & Logistics",
    type: "CARETAKING_DUTY",
    capacityUnits: 14,
    unit: "h/wk",
    allocatedCommitments: [
      {
        commitmentId: "COM_SCHOOL_RUN",
        allocatedTo: "MORNING_AND_AFTERNOON_TRANSIT",
        amount: 8,
        priority: 1,
      },
      {
        commitmentId: "COM_MEAL_PREP",
        allocatedTo: "JOINT_DINNER_PREPARATION",
        amount: 6,
        priority: 2,
      },
    ],
    totalAllocated: 14,
    isDoubleBooked: false,
    conflictDetails: [],
  },
];

// ============================================================================
// 3. INVARIANT AUDITOR: INV-OI90-P (SHARED RESOURCE INTEGRITY)
// ============================================================================

/**
 * INV-OI90-P: Shared Resource Integrity Invariant
 * The same physical or relational resource cannot be double-allocated.
 * Checks for:
 * 1. Overlapping calendar time windows within the same resource slot.
 * 2. Total allocation exceeding resource capacity.
 */
export function verifySharedResourceIntegrity(
  allocations: SharedResourceAllocation[]
): { valid: boolean; violations: string[]; verifiedAllocations: SharedResourceAllocation[] } {
  const violations: string[] = [];
  const verifiedAllocations: SharedResourceAllocation[] = [];

  allocations.forEach((res) => {
    let isDoubleBooked = false;
    const conflicts: string[] = [];

    // Check capacity over-allocation
    const totalAmount = res.allocatedCommitments.reduce((sum, c) => sum + c.amount, 0);
    if (totalAmount > res.capacityUnits) {
      isDoubleBooked = true;
      const msg = `INV-OI90-P CAPACITY VIOLATION: Resource "${res.name}" over-allocated (${totalAmount} ${res.unit} > ${res.capacityUnits} capacity).`;
      conflicts.push(msg);
      violations.push(msg);
    }

    // Check time-window collisions
    const windowMap = new Map<string, string>();
    res.allocatedCommitments.forEach((com) => {
      if (com.timeWindow) {
        if (windowMap.has(com.timeWindow)) {
          isDoubleBooked = true;
          const prior = windowMap.get(com.timeWindow);
          const msg = `INV-OI90-P CALENDAR CLASH: Resource "${res.name}" double-booked on slot [${com.timeWindow}] by "${prior}" and "${com.allocatedTo}".`;
          conflicts.push(msg);
          violations.push(msg);
        } else {
          windowMap.set(com.timeWindow, com.allocatedTo);
        }
      }
    });

    verifiedAllocations.push({
      ...res,
      totalAllocated: totalAmount,
      isDoubleBooked,
      conflictDetails: conflicts,
    });
  });

  return {
    valid: violations.length === 0,
    violations,
    verifiedAllocations,
  };
}

// ============================================================================
// 4. HOUSEHOLD HEALTH INDEX (HHI) ENGINE
// ============================================================================

export function calculateHouseholdHealthIndex(
  personalLhi: number,
  members: Record<string, RelationshipNode> = CANONICAL_RELATIONSHIP_NODES,
  allocations: SharedResourceAllocation[] = CANONICAL_SHARED_RESOURCES
): HouseholdHealthIndex {
  const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v));

  // Compute relational weighted score
  // Weights: Partner (0.35), Child (0.15), Parent (0.05), Co-founder (0.05) -> Sum = 0.60
  // Scaled alongside personal LHI (0.40)
  let relationalWeighted = 0;
  let totalRelWeight = 0;

  Object.values(members).forEach((m) => {
    relationalWeighted += m.relevanceWeight * m.currentWellbeing;
    totalRelWeight += m.relevanceWeight;
  });

  const normalizedRelational = totalRelWeight > 0 ? relationalWeighted / totalRelWeight : 80;

  // Friction penalty from shared resource conflicts
  const resourceAudit = verifySharedResourceIntegrity(allocations);
  let frictionPenalty = 0;
  if (!resourceAudit.valid) {
    frictionPenalty = resourceAudit.violations.length * 6.5;
  }

  // Base HHI: 45% Personal LHI + 55% Relational Family Wellbeing - Friction Penalty
  const baseHhi = 0.45 * personalLhi + 0.55 * normalizedRelational - frictionPenalty;
  const compositeHhi = Math.round(clamp(baseHhi, 0, 100) * 10) / 10;

  let trajectory: "STABLE" | "EXPANDING" | "STRAINED" | "CRITICAL" = "STABLE";
  if (frictionPenalty > 10 || compositeHhi < 65) {
    trajectory = "CRITICAL";
  } else if (frictionPenalty > 0 || compositeHhi < 75) {
    trajectory = "STRAINED";
  } else if (compositeHhi >= 85) {
    trajectory = "EXPANDING";
  }

  return {
    compositeHhi,
    personalLhi: Math.round(personalLhi * 10) / 10,
    relationalWeightedScore: Math.round(normalizedRelational * 10) / 10,
    householdFinancialRunway: 14,
    sharedTimeFrictionPenalty: Math.round(frictionPenalty * 10) / 10,
    delta: 0,
    trajectory,
  };
}

// ============================================================================
// 5. INVARIANT AUDITOR: INV-OI91-P (RELATIONAL IMPACT VISIBILITY)
// ============================================================================

/**
 * INV-OI91-P: Relational Impact Visibility Invariant
 * Every major recommendation or strategy must expose:
 * 1. Personal Impact
 * 2. Financial Impact
 * 3. Relational / Household Impact
 * 4. Member-by-member delta breakdown before ranking or recommendation.
 */
export function verifyRelationalImpactVisibility(
  scenario: Partial<RelationalScenarioResult>
): { valid: boolean; violations: string[] } {
  const violations: string[] = [];

  const card = scenario.relationalVisibilityCard;
  if (!card) {
    violations.push("INV-OI91-P VIOLATION: Missing relational visibility card in scenario result.");
    return { valid: false, violations };
  }

  if (!card.personalImpactSummary || card.personalImpactSummary.trim().length === 0) {
    violations.push("INV-OI91-P VIOLATION: Personal impact summary is missing or empty.");
  }
  if (!card.financialImpactSummary || card.financialImpactSummary.trim().length === 0) {
    violations.push("INV-OI91-P VIOLATION: Financial impact summary is missing or empty.");
  }
  if (!card.relationalImpactSummary || card.relationalImpactSummary.trim().length === 0) {
    violations.push("INV-OI91-P VIOLATION: Relational/household impact summary is missing or empty.");
  }

  // Check if unvarnished trade-offs disclose any member wellbeing drop >= 5 pts
  const memberImpacts = scenario.memberImpacts || [];
  const severeMemberDrop = memberImpacts.find((m) => m.delta <= -5.0);
  if (severeMemberDrop) {
    if (!card.unvarnishedTradeOffs || card.unvarnishedTradeOffs.length === 0) {
      violations.push(
        `INV-OI91-P VIOLATION: Severe relational collateral drop on ${severeMemberDrop.name} (${severeMemberDrop.delta} pts) hidden without unvarnished trade-off disclosure.`
      );
    }
  }

  return {
    valid: violations.length === 0,
    violations,
  };
}

// ============================================================================
// 6. 4 CANONICAL MULTI-TWIN SCENARIOS
// ============================================================================

export interface RelationalScenarioDefinition {
  id: string;
  title: string;
  description: string;
  personalLhiDelta: number;
  memberDeltas: Record<string, { delta: number; concerns: string[] }>;
  sharedResourceMutations?: Array<{
    resourceId: string;
    addCommitment?: CommitmentAllocation;
    capacityDelta?: number;
  }>;
  visibilityCard: {
    personalImpactSummary: string;
    financialImpactSummary: string;
    relationalImpactSummary: string;
    unvarnishedTradeOffs: string[];
  };
}

export const CANONICAL_RELATIONAL_SCENARIOS: RelationalScenarioDefinition[] = [
  {
    id: "move_city_relocation",
    title: "Relocate City for Tier-1 Tech Promotion",
    description:
      "Accept a high-prestige executive promotion in another metropolitan hub. Personal salary surges by +45%, but triggers major household dislocation.",
    personalLhiDelta: 4.2,
    memberDeltas: {
      REL_PARTNER: {
        delta: -14.0,
        concerns: [
          "Severing established local professional engineering network",
          "Forced career hiatus during relocation and settling phase",
        ],
      },
      REL_CHILD: {
        delta: -11.0,
        concerns: [
          "Disruption of elementary school continuity and sports peer group",
          "Anxiety associated with unfamiliar neighborhood environment",
        ],
      },
      REL_PARENT: {
        delta: -18.0,
        concerns: [
          "Elimination of 20-minute rapid medical assistance buffer",
          "Social isolation from grandchildren",
        ],
      },
      REL_COFOUNDER: {
        delta: -4.0,
        concerns: ["Shift from in-person whiteboarding to asynchronous syncs"],
      },
    },
    visibilityCard: {
      personalImpactSummary: "Salary increases to $268k/yr (+45%), strategic scope expands to VP-level systems.",
      financialImpactSummary: "Monthly savings expands from $2,400 to $4,600, but temporary relocation capital sink ($18k).",
      relationalImpactSummary: "Severe collective household drag (-14.2 pts average family dip). Household HHI declines despite personal career surge.",
      unvarnishedTradeOffs: [
        "Elena's career advancement put on hold for at least 12 months",
        "Leo loses established childhood community and school stability",
        "David requires hiring secondary local caregiver support",
      ],
    },
  },
  {
    id: "bootstrap_startup",
    title: "Bootstrap AI Venture from Home",
    description:
      "Found an autonomous AI startup. Zero commute, full location autonomy, but 40% initial salary drop and partner carries primary financial safety cushion.",
    personalLhiDelta: 2.1,
    memberDeltas: {
      REL_PARTNER: {
        delta: -6.0,
        concerns: [
          "Elevated financial anxiety as household single-earner anchor",
          "Absorbing fixed mortgage commitments without surplus margin",
        ],
      },
      REL_CHILD: {
        delta: 5.0,
        concerns: [
          "Positive: Parent present for afternoon pickup and dinners",
          "Watchout: Occasional work distraction at home office",
        ],
      },
      REL_PARENT: {
        delta: 2.0,
        concerns: ["Increased weekday flexibility for medical appointments"],
      },
      REL_COFOUNDER: {
        delta: 12.0,
        concerns: ["100% full-time co-founder bandwidth unlocked"],
      },
    },
    visibilityCard: {
      personalImpactSummary: "Autonomy and creative equity surge; career growth shifts from corporate ladder to equity value.",
      financialImpactSummary: "Monthly cash compensation drops by $75k/yr; household liquid runway contracts from 14 to 8 months.",
      relationalImpactSummary: "Mixed relational ripple: Partner bears financial anxiety while child and parent benefit from proximity.",
      unvarnishedTradeOffs: [
        "Elena assumes sole responsibility for baseline household liquidity",
        "Discretionary family vacation budget paused for 18 months",
      ],
    },
  },
  {
    id: "executive_masters",
    title: "Part-Time Executive Master's Degree",
    description:
      "Complete a weekend executive AI degree. $25k tuition outflow and 14h/wk weekend study drain, requiring partner to carry primary caretaking.",
    personalLhiDelta: 3.5,
    memberDeltas: {
      REL_PARTNER: {
        delta: -9.0,
        concerns: [
          "Shouldering 100% of Saturday and Sunday child caretaking solo",
          "Depleted couple shared time on weekend evenings",
        ],
      },
      REL_CHILD: {
        delta: -6.0,
        concerns: ["Parent absent for Saturday sports matches and weekend outings"],
      },
      REL_PARENT: {
        delta: -2.0,
        concerns: ["Weekend visits condensed into brief phone check-ins"],
      },
      REL_COFOUNDER: {
        delta: 1.0,
        concerns: ["Direct influx of state-of-the-art research models into project"],
      },
    },
    visibilityCard: {
      personalImpactSummary: "Deliberate technical practice surges; credential unlocks principal advisory opportunities.",
      financialImpactSummary: "Tuition drains $1,200/month from joint savings; investment pacing slowed.",
      relationalImpactSummary: "Domestic strain centered on weekend caretaking asymmetry.",
      unvarnishedTradeOffs: [
        "Elena absorbs 14 additional caretaking hours per week solo",
        "Weekend family block violated on Saturday mornings",
      ],
    },
  },
  {
    id: "balanced_household_rhythm",
    title: "Balanced Household Rhythm",
    description:
      "Lock evening family connection and achieve equitable caretaking parity. Modest personal career pacing with peak relational harmony.",
    personalLhiDelta: 1.8,
    memberDeltas: {
      REL_PARTNER: {
        delta: 8.0,
        concerns: [
          "Parity achieved in domestic and emotional labor",
          "Couple connection time reliably preserved 5 nights/week",
        ],
      },
      REL_CHILD: {
        delta: 6.0,
        concerns: ["Predictable bedtime routines and enriched weekend presence"],
      },
      REL_PARENT: {
        delta: 4.0,
        concerns: ["Consistent Sunday family dinner connection"],
      },
      REL_COFOUNDER: {
        delta: 0.0,
        concerns: ["Stable, predictable work cadence without crunch sprints"],
      },
    },
    visibilityCard: {
      personalImpactSummary: "Consistent energy, minimal burnout, high autonomic recovery.",
      financialImpactSummary: "Stable 32% savings rate and predictable investment compounding.",
      relationalImpactSummary: "All household members experience positive emotional and relational gains.",
      unvarnishedTradeOffs: [
        "Intentional deferral of aggressive corporate promotion sprint to next cycle",
      ],
    },
  },
];

// ============================================================================
// 7. MULTI-TWIN SCENARIO SIMULATOR
// ============================================================================

export function simulateRelationalScenario(
  scenarioDef: RelationalScenarioDefinition,
  personalLhiBaseline: number = 82.4,
  members: Record<string, RelationshipNode> = CANONICAL_RELATIONSHIP_NODES,
  resources: SharedResourceAllocation[] = CANONICAL_SHARED_RESOURCES
): RelationalScenarioResult {
  // 1. Calculate new personal LHI
  const newPersonalLhi = personalLhiBaseline + scenarioDef.personalLhiDelta;

  // 2. Compute member impacts
  const memberImpacts: RelationalMemberImpact[] = [];
  const updatedMembers: Record<string, RelationshipNode> = {};

  Object.entries(members).forEach(([id, member]) => {
    const deltaInfo = scenarioDef.memberDeltas[id] || { delta: 0, concerns: [] };
    const newWellbeing = Math.max(0, Math.min(100, member.baselineWellbeing + deltaInfo.delta));

    memberImpacts.push({
      relationshipId: id,
      name: member.name,
      role: member.role,
      wellbeingPrior: member.baselineWellbeing,
      wellbeingNew: Math.round(newWellbeing * 10) / 10,
      delta: Math.round(deltaInfo.delta * 10) / 10,
      keyConcerns: deltaInfo.concerns,
    });

    updatedMembers[id] = {
      ...member,
      currentWellbeing: newWellbeing,
    };
  });

  // 3. Mutate resources if scenario demands
  let currentResources = [...resources];
  if (scenarioDef.sharedResourceMutations) {
    currentResources = currentResources.map((res) => {
      const mut = scenarioDef.sharedResourceMutations?.find((m) => m.resourceId === res.resourceId);
      if (mut) {
        const newCommitments = mut.addCommitment
          ? [...res.allocatedCommitments, mut.addCommitment]
          : res.allocatedCommitments;
        const newCapacity = mut.capacityDelta ? res.capacityUnits + mut.capacityDelta : res.capacityUnits;
        return {
          ...res,
          capacityUnits: newCapacity,
          allocatedCommitments: newCommitments,
        };
      }
      return res;
    });
  }

  // 4. Invariant 1: INV-OI90-P Shared Resource Integrity
  const resourceAudit = verifySharedResourceIntegrity(currentResources);

  // 5. Compute Household Health Index
  const baselineHhi = calculateHouseholdHealthIndex(personalLhiBaseline, members, resources);
  const projectedHhi = calculateHouseholdHealthIndex(newPersonalLhi, updatedMembers, currentResources);
  const householdHhiDelta = Math.round((projectedHhi.compositeHhi - baselineHhi.compositeHhi) * 10) / 10;

  // 6. Invariant 2: INV-OI91-P Relational Impact Visibility
  const partialResult: Partial<RelationalScenarioResult> = {
    scenarioId: scenarioDef.id,
    scenarioTitle: scenarioDef.title,
    personalLhiDelta: scenarioDef.personalLhiDelta,
    householdHhiDelta,
    memberImpacts,
    isSharedResourceIntegral: resourceAudit.valid,
    sharedResourceViolations: resourceAudit.violations,
    isRelationalImpactVisible: true,
    visibilityViolations: [],
    relationalVisibilityCard: scenarioDef.visibilityCard,
  };

  const visibilityAudit = verifyRelationalImpactVisibility(partialResult);

  return {
    scenarioId: scenarioDef.id,
    scenarioTitle: scenarioDef.title,
    personalLhiDelta: scenarioDef.personalLhiDelta,
    householdHhiDelta,
    memberImpacts,
    isSharedResourceIntegral: resourceAudit.valid,
    sharedResourceViolations: resourceAudit.violations,
    isRelationalImpactVisible: visibilityAudit.valid,
    visibilityViolations: visibilityAudit.violations,
    relationalVisibilityCard: scenarioDef.visibilityCard,
  };
}
