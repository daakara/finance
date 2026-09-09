/**
 * Horizon 9: Multi-Year Household Trajectory Sequencing & Compounding Engine
 *
 * Implements:
 * - Multi-Year State Transitions: State(t+1) = State(t) + Impact(t) + Compounding(t)
 * - Trajectory Candidate Permutations & Multi-Criteria Ranking
 * - Pareto Frontier Identification (Non-Dominated Trajectories)
 * - Compounding Optionality & Latency Modeling
 * - Interactive Causal Ripple Map Builder
 * - Proactive Copilot Opportunity Detection Framework
 * - Invariant Verifiers:
 *   - INV-OI92-P: Recommendation Explainability Invariant (Zero Black-Box)
 *   - INV-OI93-P: Trajectory Stability Invariant (Monte Carlo Stability >= 4.0)
 *   - INV-OI94-P: Household Harm Visibility Invariant
 *   - INV-OI95-P: Optionality Preservation Invariant
 *   - INV-OI96-P: Compounding Traceability Invariant
 */

import {
  TrajectoryStep,
  HouseholdTrajectory,
  ExplainableRecommendation,
  RippleMapNode,
  RippleMapEdge,
  RippleMapData,
} from "../../types/personal-digital-twin";

// ============================================================================
// 1. CANONICAL MULTI-YEAR TRAJECTORIES
// ============================================================================

export const CANONICAL_TRAJECTORIES: HouseholdTrajectory[] = [
  {
    trajectoryId: "SEQ_ALPHA",
    name: "Sequence Alpha: Skill Moat First",
    description: "Y1 Executive Master's → Y2 Principal Tech Promotion → Y3 Strategic Metropolitan Relocation. Builds credential and comp leverage before moving.",
    steps: [
      {
        stepId: "STEP_A1",
        year: 1,
        interventionId: "EXECUTIVE_MASTERS",
        title: "Year 1: Deep Technical Specialization (Executive Master's)",
        personalLhiImpact: 2.5,
        householdHhiImpact: -1.2,
        annualCapitalDelta: -14400,
        discretionaryHoursDelta: -12,
        stateAfterStep: {
          SKILL_SCORE: 88,
          ANNUAL_COMP: 195000,
          RUNWAY_MONTHS: 12,
          OPTIONALITY: 75,
          PARTNER_WELLBEING: 78,
          CHILD_WELLBEING: 85,
        },
      },
      {
        stepId: "STEP_A2",
        year: 2,
        interventionId: "TECH_PROMOTION",
        title: "Year 2: Principal Systems Promotion in Current City",
        personalLhiImpact: 4.8,
        householdHhiImpact: 3.5,
        annualCapitalDelta: 45000,
        discretionaryHoursDelta: 4,
        stateAfterStep: {
          SKILL_SCORE: 92,
          ANNUAL_COMP: 240000,
          RUNWAY_MONTHS: 18,
          OPTIONALITY: 86,
          PARTNER_WELLBEING: 84,
          CHILD_WELLBEING: 88,
        },
      },
      {
        stepId: "STEP_A3",
        year: 3,
        interventionId: "STRATEGIC_RELOCATION",
        title: "Year 3: Strategic Relocation with VP Package & Partner Placement",
        personalLhiImpact: 3.2,
        householdHhiImpact: 2.1,
        annualCapitalDelta: 65000,
        discretionaryHoursDelta: 2,
        stateAfterStep: {
          SKILL_SCORE: 95,
          ANNUAL_COMP: 305000,
          RUNWAY_MONTHS: 24,
          OPTIONALITY: 94,
          PARTNER_WELLBEING: 86,
          CHILD_WELLBEING: 86,
        },
      },
    ],
    projectedLhi: 92.4,
    projectedHhi: 89.2,
    confidencePct: 86,
    totalCost: 14400,
    riskScore: 35,
    optionalityScore: 94,
    recoveryScore: 88,
    multiCriteriaScore: 87.6,
    stabilityScore: 5.4,
    isParetoOptimal: true,
    unintendedConsequences: [
      "Year 1 tuition and study drains weekend discretionary time",
      "Elena absorbs primary caretaking during Year 1 thesis completion",
    ],
  },
  {
    trajectoryId: "SEQ_BETA",
    name: "Sequence Beta: Premature Relocation",
    description: "Y1 Relocation → Y2 Executive Master's → Y3 Promotion. Dislocates household first before establishing technical leverage.",
    steps: [
      {
        stepId: "STEP_B1",
        year: 1,
        interventionId: "PREMATURE_RELOCATION",
        title: "Year 1: Relocate Without Senior Negotiation Leverage",
        personalLhiImpact: 3.8,
        householdHhiImpact: -6.8,
        annualCapitalDelta: 25000,
        discretionaryHoursDelta: -6,
        stateAfterStep: {
          SKILL_SCORE: 75,
          ANNUAL_COMP: 210000,
          RUNWAY_MONTHS: 10,
          OPTIONALITY: 62,
          PARTNER_WELLBEING: 68,
          CHILD_WELLBEING: 77,
        },
      },
      {
        stepId: "STEP_B2",
        year: 2,
        interventionId: "MASTERS_IN_TRANSIT",
        title: "Year 2: Master's Degree in New City While Ungrounded",
        personalLhiImpact: 0.8,
        householdHhiImpact: -4.5,
        annualCapitalDelta: -16000,
        discretionaryHoursDelta: -14,
        stateAfterStep: {
          SKILL_SCORE: 84,
          ANNUAL_COMP: 210000,
          RUNWAY_MONTHS: 8,
          OPTIONALITY: 58,
          PARTNER_WELLBEING: 64,
          CHILD_WELLBEING: 74,
        },
      },
      {
        stepId: "STEP_B3",
        year: 3,
        interventionId: "DELAYED_PROMOTION",
        title: "Year 3: Delayed Promotion Recovery",
        personalLhiImpact: 2.2,
        householdHhiImpact: 1.8,
        annualCapitalDelta: 35000,
        discretionaryHoursDelta: 2,
        stateAfterStep: {
          SKILL_SCORE: 89,
          ANNUAL_COMP: 245000,
          RUNWAY_MONTHS: 12,
          OPTIONALITY: 70,
          PARTNER_WELLBEING: 72,
          CHILD_WELLBEING: 80,
        },
      },
    ],
    projectedLhi: 86.5,
    projectedHhi: 74.2,
    confidencePct: 68,
    totalCost: 16000,
    riskScore: 65,
    optionalityScore: 70,
    recoveryScore: 62,
    multiCriteriaScore: 71.4,
    stabilityScore: 3.2,
    isParetoOptimal: false,
    unintendedConsequences: [
      "Severe relational friction in Y1 & Y2 due to unanchored moves",
      "Liquid runway dropped to dangerous 8-month floor in Year 2",
      "Leo experienced elementary school change without parental emotional availability",
    ],
  },
  {
    trajectoryId: "SEQ_GAMMA",
    name: "Sequence Gamma: Household Harmony First",
    description: "Y1 Protected Household Rhythm → Y2 Flexible Part-Time Master's → Y3 Principal Tech Promotion. Highest collective family well-being.",
    steps: [
      {
        stepId: "STEP_G1",
        year: 1,
        interventionId: "BALANCED_RHYTHM",
        title: "Year 1: Consolidate Household Parity & Core Savings",
        personalLhiImpact: 1.8,
        householdHhiImpact: 4.5,
        annualCapitalDelta: 18000,
        discretionaryHoursDelta: 6,
        stateAfterStep: {
          SKILL_SCORE: 78,
          ANNUAL_COMP: 195000,
          RUNWAY_MONTHS: 18,
          OPTIONALITY: 82,
          PARTNER_WELLBEING: 90,
          CHILD_WELLBEING: 92,
        },
      },
      {
        stepId: "STEP_G2",
        year: 2,
        interventionId: "FLEXIBLE_MASTERS",
        title: "Year 2: Asynchronous Master's with Protected Family Blocks",
        personalLhiImpact: 2.8,
        householdHhiImpact: 1.2,
        annualCapitalDelta: -12000,
        discretionaryHoursDelta: -8,
        stateAfterStep: {
          SKILL_SCORE: 89,
          ANNUAL_COMP: 205000,
          RUNWAY_MONTHS: 16,
          OPTIONALITY: 85,
          PARTNER_WELLBEING: 88,
          CHILD_WELLBEING: 90,
        },
      },
      {
        stepId: "STEP_G3",
        year: 3,
        interventionId: "PRINCIPAL_PROMOTION",
        title: "Year 3: Local Principal Promotion & Expanded Equity",
        personalLhiImpact: 3.8,
        householdHhiImpact: 3.2,
        annualCapitalDelta: 50000,
        discretionaryHoursDelta: 2,
        stateAfterStep: {
          SKILL_SCORE: 93,
          ANNUAL_COMP: 255000,
          RUNWAY_MONTHS: 22,
          OPTIONALITY: 90,
          PARTNER_WELLBEING: 92,
          CHILD_WELLBEING: 92,
        },
      },
    ],
    projectedLhi: 89.8,
    projectedHhi: 91.5,
    confidencePct: 92,
    totalCost: 12000,
    riskScore: 22,
    optionalityScore: 90,
    recoveryScore: 94,
    multiCriteriaScore: 89.8,
    stabilityScore: 6.2,
    isParetoOptimal: true, // Pareto optimal on HHI and Lowest Risk!
    unintendedConsequences: [
      "Modest deferral of top-bracket executive compensation by 12 months",
    ],
  },
  {
    trajectoryId: "SEQ_DELTA",
    name: "Sequence Delta: Entrepreneurial Venture Sprint",
    description: "Y1 Bootstrap AI Startup → Y2 Capital Recovery & Growth → Y3 Strategic Secondary Liquidity. Maximum optionality upside with high near-term volatility.",
    steps: [
      {
        stepId: "STEP_D1",
        year: 1,
        interventionId: "BOOTSTRAP_STARTUP",
        title: "Year 1: Launch Autonomous AI Venture from Home",
        personalLhiImpact: 2.1,
        householdHhiImpact: 0.8,
        annualCapitalDelta: -65000,
        discretionaryHoursDelta: 8,
        stateAfterStep: {
          SKILL_SCORE: 86,
          ANNUAL_COMP: 120000,
          RUNWAY_MONTHS: 9,
          OPTIONALITY: 88,
          PARTNER_WELLBEING: 76,
          CHILD_WELLBEING: 90,
        },
      },
      {
        stepId: "STEP_D2",
        year: 2,
        interventionId: "SEED_AND_GROWTH",
        title: "Year 2: Venture Capital Funding & Founder Salary Normalization",
        personalLhiImpact: 3.5,
        householdHhiImpact: 2.4,
        annualCapitalDelta: 60000,
        discretionaryHoursDelta: -4,
        stateAfterStep: {
          SKILL_SCORE: 91,
          ANNUAL_COMP: 180000,
          RUNWAY_MONTHS: 15,
          OPTIONALITY: 95,
          PARTNER_WELLBEING: 82,
          CHILD_WELLBEING: 88,
        },
      },
      {
        stepId: "STEP_D3",
        year: 3,
        interventionId: "SECONDARY_LIQUIDITY",
        title: "Year 3: Strategic Secondary Liquidity Event & Expansion",
        personalLhiImpact: 4.2,
        householdHhiImpact: 3.6,
        annualCapitalDelta: 180000,
        discretionaryHoursDelta: 0,
        stateAfterStep: {
          SKILL_SCORE: 96,
          ANNUAL_COMP: 360000,
          RUNWAY_MONTHS: 48,
          OPTIONALITY: 98,
          PARTNER_WELLBEING: 85,
          CHILD_WELLBEING: 88,
        },
      },
    ],
    projectedLhi: 88.2,
    projectedHhi: 76.5,
    confidencePct: 62,
    totalCost: 65000,
    riskScore: 68,
    optionalityScore: 98,
    recoveryScore: 72,
    multiCriteriaScore: 75.1,
    stabilityScore: 2.8,
    isParetoOptimal: true, // Pareto optimal on Maximum Future Optionality (98)
    unintendedConsequences: [
      "Year 1 cash drain tightens liquid runway to 9 months",
      "Partner assumes financial stability burden during pre-seed phase",
    ],
  },
];

// ============================================================================
// 2. MULTI-CRITERIA SCORING & PARETO RANKING
// ============================================================================

export function calculateMultiCriteriaScore(t: {
  projectedLhi: number;
  projectedHhi: number;
  confidencePct: number;
  optionalityScore: number;
  recoveryScore: number;
  riskScore: number;
}): number {
  const normLhi = Math.min(100, Math.max(0, t.projectedLhi));
  const normHhi = Math.min(100, Math.max(0, t.projectedHhi));
  const normConf = Math.min(100, Math.max(0, t.confidencePct));
  const normOpt = Math.min(100, Math.max(0, t.optionalityScore));
  const normRec = Math.min(100, Math.max(0, t.recoveryScore));
  const normRiskInv = Math.min(100, Math.max(0, 100 - t.riskScore));

  const score =
    0.30 * normLhi +
    0.25 * normHhi +
    0.15 * normConf +
    0.10 * normOpt +
    0.10 * normRec +
    0.10 * normRiskInv;

  return Math.round(score * 10) / 10;
}

export function identifyParetoFront(trajectories: HouseholdTrajectory[]): HouseholdTrajectory[] {
  return trajectories.map((candidate) => {
    // Check if any other trajectory strictly dominates candidate
    const isDominated = trajectories.some((other) => {
      if (other.trajectoryId === candidate.trajectoryId) return false;

      const otherBetterOrEqual =
        other.projectedLhi >= candidate.projectedLhi &&
        other.projectedHhi >= candidate.projectedHhi &&
        other.optionalityScore >= candidate.optionalityScore &&
        other.recoveryScore >= candidate.recoveryScore &&
        other.riskScore <= candidate.riskScore;

      const otherStrictlyBetter =
        other.projectedLhi > candidate.projectedLhi ||
        other.projectedHhi > candidate.projectedHhi ||
        other.optionalityScore > candidate.optionalityScore ||
        other.recoveryScore > candidate.recoveryScore ||
        other.riskScore < candidate.riskScore;

      return otherBetterOrEqual && otherStrictlyBetter;
    });

    return {
      ...candidate,
      isParetoOptimal: !isDominated,
    };
  });
}

// ============================================================================
// 3. VISUAL RIPPLE MAP BUILDER
// ============================================================================

export function buildRippleMap(interventionId: string): RippleMapData {
  if (interventionId === "STRATEGIC_RELOCATION" || interventionId === "PREMATURE_RELOCATION") {
    const nodes: RippleMapNode[] = [
      { id: "ROOT", label: "Executive Relocation", domain: "DECISION", magnitude: 24, delta: 0, unit: "event" },
      { id: "N_COMP", label: "Annual Compensation", domain: "CAREER", magnitude: 18, delta: 45000, unit: "$/yr" },
      { id: "N_SCOPE", label: "Executive Scope", domain: "CAREER", magnitude: 16, delta: 25, unit: "pts" },
      { id: "N_SAVINGS", label: "Monthly Savings", domain: "FINANCE", magnitude: 14, delta: 1800, unit: "$/mo" },
      { id: "N_PARTNER", label: "Elena (Partner Career)", domain: "RELATIONSHIPS", magnitude: 15, delta: -14, unit: "pts" },
      { id: "N_CHILD", label: "Leo (School Continuity)", domain: "RELATIONSHIPS", magnitude: 12, delta: -11, unit: "pts" },
      { id: "N_PARENT", label: "David (Medical Buffer)", domain: "RELATIONSHIPS", magnitude: 10, delta: -18, unit: "pts" },
      { id: "N_HHI", label: "Household Health Index", domain: "HOUSEHOLD", magnitude: 20, delta: -4.8, unit: "pts" },
    ];

    const edges: RippleMapEdge[] = [
      { id: "E1", from: "ROOT", to: "N_COMP", impact: 4, isPositive: true, latencyWeeks: 4 },
      { id: "E2", from: "ROOT", to: "N_SCOPE", impact: 3.5, isPositive: true, latencyWeeks: 2 },
      { id: "E3", from: "N_COMP", to: "N_SAVINGS", impact: 3, isPositive: true, latencyWeeks: 8 },
      { id: "E4", from: "ROOT", to: "N_PARTNER", impact: 4.5, isPositive: false, latencyWeeks: 2 },
      { id: "E5", from: "ROOT", to: "N_CHILD", impact: 3.5, isPositive: false, latencyWeeks: 4 },
      { id: "E6", from: "ROOT", to: "N_PARENT", impact: 3, isPositive: false, latencyWeeks: 2 },
      { id: "E7", from: "N_PARTNER", to: "N_HHI", impact: 4, isPositive: false, latencyWeeks: 6 },
      { id: "E8", from: "N_SAVINGS", to: "N_HHI", impact: 2, isPositive: true, latencyWeeks: 12 },
    ];

    return {
      rootIntervention: interventionId,
      nodes,
      edges,
      netHhiDelta: -4.8,
      netLhiDelta: 4.2,
    };
  }

  // Default: Master's degree ripple
  const nodes: RippleMapNode[] = [
    { id: "ROOT", label: "Executive Master's", domain: "DECISION", magnitude: 22, delta: 0, unit: "event" },
    { id: "N_SKILL", label: "AI & Systems Mastery", domain: "LEARNING", magnitude: 20, delta: 24, unit: "pts" },
    { id: "N_TUITION", label: "Tuition Outflow", domain: "FINANCE", magnitude: 14, delta: -1200, unit: "$/mo" },
    { id: "N_WEEKEND", label: "Weekend Family Time", domain: "TIME", magnitude: 16, delta: -14, unit: "h/wk" },
    { id: "N_PROMO", label: "Promotion Velocity", domain: "CAREER", magnitude: 18, delta: 35, unit: "pts" },
    { id: "N_OPTIONALITY", label: "Future Opportunity Set", domain: "STRATEGY", magnitude: 18, delta: 22, unit: "pts" },
    { id: "N_HHI", label: "Household Health Index", domain: "HOUSEHOLD", magnitude: 20, delta: 3.5, unit: "pts" },
  ];

  const edges: RippleMapEdge[] = [
    { id: "E1", from: "ROOT", to: "N_SKILL", impact: 4, isPositive: true, latencyWeeks: 6 },
    { id: "E2", from: "ROOT", to: "N_TUITION", impact: 3, isPositive: false, latencyWeeks: 1 },
    { id: "E3", from: "ROOT", to: "N_WEEKEND", impact: 4, isPositive: false, latencyWeeks: 1 },
    { id: "E4", from: "N_SKILL", to: "N_PROMO", impact: 4.5, isPositive: true, latencyWeeks: 24 },
    { id: "E5", from: "N_PROMO", to: "N_OPTIONALITY", impact: 3.8, isPositive: true, latencyWeeks: 52 },
    { id: "E6", from: "N_OPTIONALITY", to: "N_HHI", impact: 3.5, isPositive: true, latencyWeeks: 52 },
  ];

  return {
    rootIntervention: interventionId,
    nodes,
    edges,
    netHhiDelta: 3.5,
    netLhiDelta: 4.8,
  };
}

// ============================================================================
// 4. PROACTIVE COPILOT OPPORTUNITY DETECTOR
// ============================================================================

export function detectProactiveOpportunities(
  currentSignals: {
    energyLevel?: number;
    allocatedWeeklyHours?: number;
    marketDemandIndex?: number;
    financialRunwayMonths?: number;
  }
): ExplainableRecommendation[] {
  const recommendations: ExplainableRecommendation[] = [];

  const energy = currentSignals.energyLevel ?? 88;
  const hours = currentSignals.allocatedWeeklyHours ?? 28;
  const market = currentSignals.marketDemandIndex ?? 92;
  const runway = currentSignals.financialRunwayMonths ?? 14;

  // Trigger 1: Deep Work / AI Architect Leap
  if (energy >= 82 && hours <= 34 && market >= 80) {
    recommendations.push({
      recommendationId: "REC_AI_ARCHITECT_LEAP",
      title: "Submit Fast-Track Application for Principal AI Systems Architect",
      whyNow: `Autonomic energy is at ${energy}/100, weekly schedule has ${35 - hours}h discretionary margin, and Tier-1 market demand index is surging at ${market} pts.`,
      triggerSignals: [
        `HIGH_ENERGY_AUTONOMIC: ${energy}/100`,
        `CALENDAR_CAPACITY_SLACK: ${hours}h committed (healthy headroom)`,
        `MACRO_AI_MARKET_SURGE: ${market} percentile`,
      ],
      tracePath: [
        "SIGNAL_VITALITY_SURGE",
        "CALENDAR_HEADROOM_CONFIRMED",
        "AI_DISTRIBUTED_MARKET_WINDOW",
        "PROACTIVE_APPLICATION_SUBMIT",
        "CAREER_VELOCITY_COMPOUNDING",
      ],
      expectedLhiImpact: 3.4,
      expectedHhiImpact: 1.8,
      confidencePct: 89,
      p10: 1.2,
      p50: 3.4,
      p90: 5.2,
      stabilityScore: 5.1,
      timestampUtc: new Date().toISOString(),
    });
  }

  // Trigger 2: Caretaking Parity Rebalance
  if (hours > 38 && energy < 70) {
    recommendations.push({
      recommendationId: "REC_HOUSEHOLD_PARITY_RESET",
      title: "Initiate Mid-Sprint Household Caretaking Rebalance",
      whyNow: `Weekly commitments (${hours}h) exceed sustainable threshold with declining energy (${energy}/100). Rebalance 4h caretaking to protect weekend recovery.`,
      triggerSignals: [
        `HIGH_WORK_BURDEN: ${hours}h`,
        `DECLINING_RECOVERY: ${energy}/100`,
      ],
      tracePath: [
        "BURNOUT_THRESHOLD_WARNING",
        "SHARED_RESOURCE_DEFICIT",
        "CARETAKING_SCHEDULE_REBALANCE",
        "RECOVERY_RESTORED",
      ],
      expectedLhiImpact: 1.5,
      expectedHhiImpact: 3.2,
      confidencePct: 91,
      p10: 0.8,
      p50: 1.5,
      p90: 2.2,
      stabilityScore: 6.4,
      timestampUtc: new Date().toISOString(),
    });
  }

  return recommendations;
}

// ============================================================================
// 5. INVARIANT AUDITORS: INV-OI92-P THROUGH INV-OI96-P
// ============================================================================

/**
 * INV-OI92-P: Recommendation Explainability Invariant
 * Prohibits black-box suggestions. Every recommendation must contain whyNow,
 * tracePath, confidence, expected LHI/HHI, and p10/p50/p90 percentiles.
 */
export function verifyExplainableRecommendation(rec: ExplainableRecommendation): {
  valid: boolean;
  violations: string[];
} {
  const violations: string[] = [];

  if (!rec.whyNow || rec.whyNow.trim().length === 0) {
    violations.push("INV-OI92-P VIOLATION: Missing whyNow authoritative signal justification.");
  }
  if (!rec.tracePath || rec.tracePath.length < 2) {
    violations.push("INV-OI92-P VIOLATION: Incomplete tracePath causal lineage.");
  }
  if (typeof rec.expectedLhiImpact !== "number") {
    violations.push("INV-OI92-P VIOLATION: Missing numeric expectedLhiImpact.");
  }
  if (typeof rec.expectedHhiImpact !== "number") {
    violations.push("INV-OI92-P VIOLATION: Missing numeric expectedHhiImpact.");
  }
  if (rec.p10 === undefined || rec.p50 === undefined || rec.p90 === undefined) {
    violations.push("INV-OI92-P VIOLATION: Missing p10/p50/p90 uncertainty distribution.");
  }
  if (rec.p10 > rec.p50 || rec.p50 > rec.p90) {
    violations.push("INV-OI92-P VIOLATION: Invalid percentile ordering (p10 <= p50 <= p90 violated).");
  }

  return { valid: violations.length === 0, violations };
}

/**
 * INV-OI93-P: Trajectory Stability Invariant
 * The top-ranked multi-year trajectory must maintain stabilityScore >= 4.0
 * (expectedOutcome / variance).
 */
export function verifyTrajectoryStability(trajectory: HouseholdTrajectory): {
  valid: boolean;
  violations: string[];
} {
  const violations: string[] = [];
  if (trajectory.stabilityScore < 4.0) {
    violations.push(
      `INV-OI93-P STABILITY VIOLATION: Trajectory ${trajectory.trajectoryId} stability (${trajectory.stabilityScore.toFixed(1)}) is below minimum threshold (4.0).`
    );
  }
  return { valid: violations.length === 0, violations };
}

/**
 * INV-OI94-P: Household Harm Visibility Invariant
 * Prohibits hiding negative multi-year cross-twin effects.
 */
export function verifyHouseholdHarmVisibility(trajectory: HouseholdTrajectory): {
  valid: boolean;
  violations: string[];
} {
  const violations: string[] = [];
  // If HHI is significantly negative while LHI is positive, unintendedConsequences must disclose it
  if (trajectory.projectedHhi < 80 && trajectory.projectedLhi >= 85) {
    if (!trajectory.unintendedConsequences || trajectory.unintendedConsequences.length === 0) {
      violations.push(
        `INV-OI94-P VIOLATION: Trajectory ${trajectory.trajectoryId} creates significant household drag (HHI ${trajectory.projectedHhi}) without unvarnished disclosure.`
      );
    }
  }
  return { valid: violations.length === 0, violations };
}

/**
 * INV-OI95-P: Optionality Preservation Invariant
 * Trajectory scoring must explicitly penalize actions that permanently destroy future optionality.
 */
export function verifyOptionalityPreservation(trajectory: HouseholdTrajectory): {
  valid: boolean;
  violations: string[];
} {
  const violations: string[] = [];
  if (trajectory.optionalityScore < 50) {
    violations.push(
      `INV-OI95-P OPTIONALITY DEFICIT: Trajectory ${trajectory.trajectoryId} restricts future optionality (${trajectory.optionalityScore} < 50).`
    );
  }
  return { valid: violations.length === 0, violations };
}

/**
 * INV-OI96-P: Compounding Traceability Invariant
 * Every multi-year compounding gain must be traceable through intermediate causal links.
 */
export function verifyCompoundingTraceability(trajectory: HouseholdTrajectory): {
  valid: boolean;
  violations: string[];
} {
  const violations: string[] = [];
  if (trajectory.steps.length < 2) {
    violations.push("INV-OI96-P VIOLATION: Multi-year compounding requires at least 2 sequential steps.");
  }
  trajectory.steps.forEach((step, idx) => {
    if (step.year !== idx + 1) {
      violations.push(`INV-OI96-P VIOLATION: Step ${step.stepId} year index mismatch (${step.year} !== ${idx + 1}).`);
    }
  });
  return { valid: violations.length === 0, violations };
}
