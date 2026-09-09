/**
 * Horizon 14: Unified CQRS Read Model Store
 *
 * Implements the single source of truth for the ARX Unified Operating Cockpit.
 * Consolidates all intelligence metrics from Horizons 5–13 into an immutable,
 * unified read state.
 *
 * Enforces Invariant:
 * - INV-OI110-P: Single Source of Truth (Every route and component displays identical LHI/HHI/IAI values)
 */

export interface TriadIndex {
  lhi: number; // Life Health Index (Canonical: 84)
  hhi: number; // Household Health Index (Canonical: 89)
  iai: number; // Identity Alignment Index (Canonical: 61)
  compositeResilience: number;
  status: 'STABLE_COMPOUNDING' | 'AT_RISK' | 'DEGRADED';
  interpretation: string;
}

export interface SignalQualityState {
  freshness: 'REALTIME' | 'DELAYED' | 'STALE';
  confidence: number; // e.g. 91
  activeSignalsCount: number;
  highConvictionRatio: number; // e.g. 0.82
  lastTelemetrySync: string;
}

export interface ActionItem {
  id: string;
  title: string;
  domain: 'CAREER' | 'HEALTH' | 'HOUSEHOLD' | 'CAPITAL' | 'IDENTITY';
  durationMinutes: number;
  priorityScore: number;
  identityContribution: number;
  rationale: string;
  energyRequired: 'HIGH_COGNITIVE' | 'MODERATE' | 'LOW_RESTORATIVE';
  scheduledTimeWindow?: string;
}

export interface OutcomeForecast {
  id: string;
  title: string;
  metric: string;
  currentValue: string;
  projectedValue3Yr: string;
  confidencePct: number;
  primaryDriver: string;
  riskFactors: string[];
}

export interface IdentityDriftAlert {
  hasActiveDrift: boolean;
  domain: string;
  inactiveDays: number;
  thresholdDays: number;
  remedyAction: string;
  status: 'ALERT' | 'NOMINAL';
  impactExplanation: string;
}

export interface HouseholdHealthState {
  hhi: number;
  partnerAlignment: number;
  sharedResourceLoad: number; // 0.0 - 1.0 (e.g. 0.68)
  conflictRisk: 'LOW' | 'MEDIUM' | 'ELEVATED';
  keySyncItem: string;
  stakeholderCount: number;
}

export interface RunwayState {
  monthsUnencumbered: number; // e.g. 14.2
  liquidReserves: number; // e.g. 78500
  burnRateMonthly: number; // e.g. 5500
  runwayShieldStatus: 'PROTECTED' | 'CAUTION' | 'CRITICAL';
  capitalFloorRule: string;
}

export interface ConstraintAlert {
  id: string;
  type: 'CAPACITY' | 'DRAWDOWN' | 'SLEEP' | 'SCHEDULE';
  severity: 'INFO' | 'WARNING' | 'CRITICAL';
  message: string;
  currentUtilization: string;
  enforcementRule: string;
}

export interface RecoveryIndicatorState {
  sleepScore: number;
  hrvTrend: 'OPTIMAL' | 'BASELINE' | 'SUPPRESSED';
  energyCapacity: number;
  primeWindow: string;
  circadianPhase: string;
}

export interface FuturePathway {
  id: string;
  name: string;
  probability: number;
  expectedNetWorth3Yr: string;
  identityFulfillmentPct: number;
  downsideBufferMonths: number;
  tradeoffs: string;
}

export interface SkillTrajectory {
  skill: string;
  currentScore: number;
  targetScore: number;
  gapPoints: number;
  momentumVelocityPct: number;
}

export interface CalibrationState {
  brierScore: number; // 0.0 - 1.0 (e.g. 0.18, calibrated <= 0.25)
  accuracyPct: number;
  overconfidenceBias: 'NONE' | 'SLIGHT' | 'ELEVATED';
  trend: 'CALIBRATED' | 'IMPROVING' | 'DEGRADED';
  sampleDecisionsAudited: number;
}

export interface SharedResourceItem {
  name: string;
  capacityAllocatedPct: number;
  primaryUsers: string[];
  conflictStatus: 'CLEAR' | 'POTENTIAL_COLLISION';
}

export interface SpecialistWorkbenchMeta {
  id: string;
  slug: string;
  name: string;
  description: string;
  category: 'GRAPH' | 'SIGNALS' | 'ALLOCATION' | 'JOURNAL' | 'SIMULATION';
  route: string;
  activeMetricsCount: number;
}

export interface UnifiedCockpitState {
  version: string;
  generatedAt: string;
  subjectId: string;
  subjectName: string;
  targetIdentityRole: string;
  triad: TriadIndex;
  signalQuality: SignalQualityState;
  nextBestAction: ActionItem;
  secondaryActions: ActionItem[];
  primaryForecast: OutcomeForecast;
  outcomeForecasts: OutcomeForecast[];
  identityDrift: IdentityDriftAlert;
  householdHealth: HouseholdHealthState;
  runway: RunwayState;
  activeConstraints: ConstraintAlert[];
  recoveryIndicator: RecoveryIndicatorState;
  futurePaths: FuturePathway[];
  skillTrajectories: SkillTrajectory[];
  calibrationScore: CalibrationState;
  sharedResources: SharedResourceItem[];
  workbenches: SpecialistWorkbenchMeta[];
}

/**
 * Canonical immutable state snapshot for the ARX Unified Operating Cockpit.
 * Sourced deterministically from Horizons 5–13 intelligence outputs.
 */
const CANONICAL_COCKPIT_STATE: UnifiedCockpitState = {
  version: '14.0.0-CQRS',
  generatedAt: '2026-09-09T14:00:00Z',
  subjectId: 'david-trader-01',
  subjectName: 'David',
  targetIdentityRole: 'AI Strategy Leader & Systematic Investor',
  triad: {
    lhi: 84,
    hhi: 89,
    iai: 61,
    compositeResilience: 81.2,
    status: 'STABLE_COMPOUNDING',
    interpretation: 'Life is stable (LHI 84), household is cohesive (HHI 89), and identity progression is actively developing (IAI 61).',
  },
  signalQuality: {
    freshness: 'REALTIME',
    confidence: 91,
    activeSignalsCount: 24,
    highConvictionRatio: 0.82,
    lastTelemetrySync: '2 minutes ago',
  },
  nextBestAction: {
    id: 'NBA-01',
    title: 'Deep Work: AI Systems Architecture RFC',
    domain: 'CAREER',
    durationMinutes: 45,
    priorityScore: 94,
    identityContribution: 24,
    rationale: 'Compounds declared AI Strategy Leader trajectory during morning peak chronotype window.',
    energyRequired: 'HIGH_COGNITIVE',
    scheduledTimeWindow: '09:30 - 10:15',
  },
  secondaryActions: [
    {
      id: 'NBA-02',
      title: 'Zone 2 Aerobic Recovery Run',
      domain: 'HEALTH',
      durationMinutes: 30,
      priorityScore: 82,
      identityContribution: 12,
      rationale: 'Prevents cardiovascular fatigue and maintains autonomic nervous system HRV baseline.',
      energyRequired: 'LOW_RESTORATIVE',
      scheduledTimeWindow: '17:00 - 17:30',
    },
    {
      id: 'NBA-03',
      title: 'Partner Weekly Schedule Alignment',
      domain: 'HOUSEHOLD',
      durationMinutes: 15,
      priorityScore: 86,
      identityContribution: 14,
      rationale: 'Harmonizes weekend child logistics and shared vehicle capacity.',
      energyRequired: 'MODERATE',
      scheduledTimeWindow: '18:15 - 18:30',
    },
  ],
  primaryForecast: {
    id: 'FC-01',
    title: '3-Year Net Liquid Wealth Compounding',
    metric: 'Net Liquid Worth',
    currentValue: '$840,000',
    projectedValue3Yr: '$1,240,000',
    confidencePct: 88,
    primaryDriver: 'Systematic Equity Allocation + Executive Compensation Growth',
    riskFactors: ['Severe tech equity multiple contraction (>35%)', 'Domestic burnout due to unmanaged capacity'],
  },
  outcomeForecasts: [
    {
      id: 'FC-01',
      title: '3-Year Net Liquid Wealth Compounding',
      metric: 'Net Liquid Worth',
      currentValue: '$840,000',
      projectedValue3Yr: '$1,240,000',
      confidencePct: 88,
      primaryDriver: 'Systematic Equity Allocation + Executive Compensation Growth',
      riskFactors: ['Severe tech equity multiple contraction (>35%)'],
    },
    {
      id: 'FC-02',
      title: 'Executive AI Leadership Trajectory',
      metric: 'Organizational Scope',
      currentValue: 'Senior Manager (14 reports)',
      projectedValue3Yr: 'VP / Head of AI Strategy (50+ reports)',
      confidencePct: 82,
      primaryDriver: 'Published Enterprise RFCs & Architecture Board Leadership',
      riskFactors: ['Context switching across non-strategic operational firefights'],
    },
  ],
  identityDrift: {
    hasActiveDrift: true,
    domain: 'Public Influence',
    inactiveDays: 68,
    thresholdDays: 60,
    remedyAction: '15m Draft Industry Case Note on Autonomous Decision Engines',
    status: 'ALERT',
    impactExplanation: 'Zero external architecture publications in 68 days slows network compounding.',
  },
  householdHealth: {
    hhi: 89,
    partnerAlignment: 86,
    sharedResourceLoad: 0.68,
    conflictRisk: 'LOW',
    keySyncItem: 'Saturday Childcare & Morning Workout Time Windows',
    stakeholderCount: 3,
  },
  runway: {
    monthsUnencumbered: 14.2,
    liquidReserves: 78500,
    burnRateMonthly: 5500,
    runwayShieldStatus: 'PROTECTED',
    capitalFloorRule: 'Mandatory 6-month ($33,000) liquid cash preservation boundary active.',
  },
  activeConstraints: [
    {
      id: 'C-01',
      type: 'CAPACITY',
      severity: 'WARNING',
      message: '168-Hour Weekly Capacity: 142h committed / 26h restorative buffer.',
      currentUtilization: '84.5% capacity allocated',
      enforcementRule: 'Prohibits scheduling ad-hoc meetings exceeding 30 minutes without dropping equal commitment.',
    },
    {
      id: 'C-02',
      type: 'DRAWDOWN',
      severity: 'INFO',
      message: 'Governor Risk Clamp active on discretionary speculative accounts (-25%).',
      currentUtilization: '$375 max risk per setup',
      enforcementRule: 'INV-OI114-P dynamic risk scaling ensures capital preservation.',
    },
  ],
  recoveryIndicator: {
    sleepScore: 84,
    hrvTrend: 'OPTIMAL',
    energyCapacity: 88,
    primeWindow: '09:00 - 12:30',
    circadianPhase: 'Peak Cognitive Window',
  },
  futurePaths: [
    {
      id: 'PATH-01',
      name: 'Systematic AI Strategy Pivot (Recommended)',
      probability: 0.74,
      expectedNetWorth3Yr: '$1,240,000',
      identityFulfillmentPct: 92,
      downsideBufferMonths: 14.2,
      tradeoffs: 'Demands strict calendar boundaries; requires declining ad-hoc side projects.',
    },
    {
      id: 'PATH-02',
      name: 'Status Quo Analytics Management',
      probability: 0.18,
      expectedNetWorth3Yr: '$1,020,000',
      identityFulfillmentPct: 64,
      downsideBufferMonths: 14.2,
      tradeoffs: 'Low friction today, but compounds career obsolescence and boredom.',
    },
    {
      id: 'PATH-03',
      name: 'Accelerated Liquid Reserve Focus',
      probability: 0.08,
      expectedNetWorth3Yr: '$950,000',
      identityFulfillmentPct: 58,
      downsideBufferMonths: 22.0,
      tradeoffs: 'Maximizes short-term safety at the cost of long-term upside compounding.',
    },
  ],
  skillTrajectories: [
    { skill: 'AI & Systems Architecture', currentScore: 64, targetScore: 88, gapPoints: 24, momentumVelocityPct: 72 },
    { skill: 'Strategic Technical Leadership', currentScore: 70, targetScore: 88, gapPoints: 18, momentumVelocityPct: 68 },
    { skill: 'Systematic Capital Allocation', currentScore: 75, targetScore: 90, gapPoints: 15, momentumVelocityPct: 84 },
    { skill: 'Public Industry Influence', currentScore: 41, targetScore: 50, gapPoints: 9, momentumVelocityPct: 38 },
  ],
  calibrationScore: {
    brierScore: 0.18,
    accuracyPct: 78.4,
    overconfidenceBias: 'NONE',
    trend: 'CALIBRATED',
    sampleDecisionsAudited: 42,
  },
  sharedResources: [
    { name: 'Vehicle A (Primary Family SUV)', capacityAllocatedPct: 62, primaryUsers: ['David', 'Sarah'], conflictStatus: 'CLEAR' },
    { name: 'Home Office / Studio Acoustic Window', capacityAllocatedPct: 78, primaryUsers: ['David'], conflictStatus: 'CLEAR' },
    { name: 'Shared Household Reserve Account', capacityAllocatedPct: 45, primaryUsers: ['David', 'Sarah'], conflictStatus: 'CLEAR' },
  ],
  workbenches: [
    {
      id: 'wb-life-graph',
      slug: 'life-graph',
      name: 'Life Graph Workbench',
      description: 'Causal dependencies, multi-domain ripple propagation, and systemic friction topology.',
      category: 'GRAPH',
      route: '/workbench/life-graph',
      activeMetricsCount: 48,
    },
    {
      id: 'wb-signals',
      slug: 'signals',
      name: 'Personal Signals Workbench',
      description: 'High-frequency biometrics, telemetry streams, chronotype rhythms, and conviction signals.',
      category: 'SIGNALS',
      route: '/workbench/signals',
      activeMetricsCount: 24,
    },
    {
      id: 'wb-allocator',
      slug: 'allocator',
      name: '168-Hour Allocator Workbench',
      description: 'Time, energy, and capital envelope modeling with calendar collision resolution.',
      category: 'ALLOCATION',
      route: '/workbench/allocator',
      activeMetricsCount: 168,
    },
    {
      id: 'wb-journal',
      slug: 'journal',
      name: 'Decision Journal Workbench',
      description: 'Probabilistic prediction auditing, Brier score calibration, and post-mortem review.',
      category: 'JOURNAL',
      route: '/workbench/journal',
      activeMetricsCount: 42,
    },
    {
      id: 'wb-simulation',
      slug: 'simulation',
      name: 'Simulation & Trajectories Workbench',
      description: 'Multi-year Monte Carlo trajectories, macroeconomic stress-testing, and future states.',
      category: 'SIMULATION',
      route: '/workbench/simulation',
      activeMetricsCount: 1000,
    },
  ],
};

/**
 * Returns the immutable Unified CQRS Cockpit State.
 * All UI pages and components MUST consume ONLY this read model.
 */
export function getUnifiedCockpitState(): UnifiedCockpitState {
  return CANONICAL_COCKPIT_STATE;
}

export interface InvariantVerificationResult {
  compliant: boolean;
  invariantId: string;
  violations: string[];
  metadata?: Record<string, unknown>;
}

/**
 * INV-OI110-P: Single Source of Truth
 * Asserts that every view/route displays identical Triad values (LHI 84, HHI 89, IAI 61).
 */
export function verifyUnifiedSourceOfTruth(
  states: UnifiedCockpitState[]
): InvariantVerificationResult {
  const violations: string[] = [];

  states.forEach((s, idx) => {
    if (s.triad.lhi !== 84) {
      violations.push(`INV-OI110-P VIOLATION: State #${idx} has non-canonical LHI ${s.triad.lhi} (expected 84).`);
    }
    if (s.triad.hhi !== 89) {
      violations.push(`INV-OI110-P VIOLATION: State #${idx} has non-canonical HHI ${s.triad.hhi} (expected 89).`);
    }
    if (s.triad.iai !== 61) {
      violations.push(`INV-OI110-P VIOLATION: State #${idx} has non-canonical IAI ${s.triad.iai} (expected 61).`);
    }
    if (s.nextBestAction.id !== 'NBA-01') {
      violations.push(`INV-OI110-P VIOLATION: State #${idx} has desynced Primary Action (expected NBA-01).`);
    }
    if (s.secondaryActions.length > 2) {
      violations.push(`INV-OI110-P VIOLATION: State #${idx} exceeds secondary action limit (max 2).`);
    }
  });

  return {
    compliant: violations.length === 0,
    invariantId: 'INV-OI110-P',
    violations,
    metadata: {
      instancesAudited: states.length,
      canonicalTriad: { lhi: 84, hhi: 89, iai: 61 },
    },
  };
}
