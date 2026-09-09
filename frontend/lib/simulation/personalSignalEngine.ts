/**
 * Personal Signal Engine & Invariant Verification (Horizon 6)
 *
 * Implements:
 * - Layer 1: Connectors (Calendar, Health, Finance, Professional)
 * - Layer 2: Canonical Signal Normalizer
 * - Layer 3: INV-OI84-P (Personal Signal Freshness Invariant)
 * - Layer 4: INV-OI86-P (Conflict Resolution Integrity & Audit Ledger)
 * - Layer 5: INV-OI85-P (Decision Outcome Capture & Calibration)
 * - Layer 6: INV-OI87-P (Composite Signal Quality Coverage)
 * - Dynamic Digital Twin State Hydrator
 */

import {
  PersonalSignal,
  SignalCategory,
  SignalFreshnessResult,
  SignalReliability,
  SignalConflict,
  DecisionOutcome,
  SignalQualityComposite,
  ConnectedSource,
  PersonalCapacity,
  EnergyState,
  AttentionState,
} from "../../types/personal-digital-twin";

export const CATEGORY_FRESHNESS_CEILINGS: Record<SignalCategory, number> = {
  HEALTH: 6,      // 6 hours max (Sleep/HRV metrics age quickly)
  TIME: 24,       // 24 hours max (Calendar updates daily)
  FINANCE: 24,    // 24 hours max (Transactions settle daily)
  CAREER: 48,     // 48 hours max (GitHub commits / work deliverables)
  LEARNING: 72,   // 72 hours max (Course progress / reading modules)
};

export const SOURCE_RELIABILITY_REGISTRY: Record<string, SignalReliability> = {
  APPLE_HEALTH: { source: "APPLE_HEALTH", confidencePct: 98, priority: 1 },
  WHOOP: { source: "WHOOP", confidencePct: 96, priority: 1 },
  OURA: { source: "OURA", confidencePct: 96, priority: 1 },
  GOOGLE_CALENDAR: { source: "GOOGLE_CALENDAR", confidencePct: 95, priority: 2 },
  MICROSOFT_OUTLOOK: { source: "MICROSOFT_OUTLOOK", confidencePct: 94, priority: 2 },
  PLAID: { source: "PLAID", confidencePct: 96, priority: 2 },
  GITHUB: { source: "GITHUB", confidencePct: 92, priority: 2 },
  LINKEDIN: { source: "LINKEDIN", confidencePct: 75, priority: 3 },
  MANUAL_ENTRY: { source: "MANUAL_ENTRY", confidencePct: 70, priority: 3 },
  MODEL_PREDICTION: { source: "MODEL_PREDICTION", confidencePct: 60, priority: 4 },
};

export const CANONICAL_CONNECTED_SOURCES: ConnectedSource[] = [
  {
    id: "SRC-CAL-01",
    name: "Google Calendar Sync",
    category: "TIME",
    provider: "GOOGLE_CALENDAR",
    status: "SYNCED",
    lastSyncUtc: "2026-09-09T09:30:00Z",
    freshnessScore: 99,
    confidencePct: 95,
    totalSignalsTracked: 14,
  },
  {
    id: "SRC-HLT-01",
    name: "Apple Health & Oura Ring",
    category: "HEALTH",
    provider: "APPLE_HEALTH",
    status: "SYNCED",
    lastSyncUtc: "2026-09-09T10:15:00Z",
    freshnessScore: 94,
    confidencePct: 98,
    totalSignalsTracked: 22,
  },
  {
    id: "SRC-FIN-01",
    name: "Plaid Financial Accounts",
    category: "FINANCE",
    provider: "PLAID",
    status: "SYNCED",
    lastSyncUtc: "2026-09-08T22:00:00Z",
    freshnessScore: 82,
    confidencePct: 96,
    totalSignalsTracked: 8,
  },
  {
    id: "SRC-PRO-01",
    name: "GitHub & Learning Hub",
    category: "LEARNING",
    provider: "GITHUB",
    status: "SYNCED",
    lastSyncUtc: "2026-09-08T14:00:00Z",
    freshnessScore: 88,
    confidencePct: 92,
    totalSignalsTracked: 11,
  },
];

export const CANONICAL_SIGNALS: PersonalSignal[] = [
  {
    signalId: "SIG-TIME-001",
    category: "TIME",
    metricId: "WEEKLY_FOCUS_HOURS",
    value: 18.5,
    unit: "hours",
    metadata: {
      source: "GOOGLE_CALENDAR",
      observedAtUtc: "2026-09-09T09:30:00Z",
      confidencePct: 95,
      freshnessHours: 1.8,
      maxAllowedAgeHours: 24,
    },
  },
  {
    signalId: "SIG-TIME-002",
    category: "TIME",
    metricId: "COMMUTE_HOURS",
    value: 4.5,
    unit: "hours",
    metadata: {
      source: "GOOGLE_CALENDAR",
      observedAtUtc: "2026-09-09T09:30:00Z",
      confidencePct: 95,
      freshnessHours: 1.8,
      maxAllowedAgeHours: 24,
    },
  },
  {
    signalId: "SIG-HLT-001",
    category: "HEALTH",
    metricId: "SLEEP_DURATION",
    value: 7.8,
    unit: "hours",
    metadata: {
      source: "APPLE_HEALTH",
      observedAtUtc: "2026-09-09T07:15:00Z",
      confidencePct: 98,
      freshnessHours: 4.1,
      maxAllowedAgeHours: 6,
    },
  },
  {
    signalId: "SIG-HLT-002",
    category: "HEALTH",
    metricId: "HEART_RATE_VARIABILITY",
    value: 68.0,
    unit: "ms",
    metadata: {
      source: "APPLE_HEALTH",
      observedAtUtc: "2026-09-09T07:15:00Z",
      confidencePct: 98,
      freshnessHours: 4.1,
      maxAllowedAgeHours: 6,
    },
  },
  {
    signalId: "SIG-FIN-001",
    category: "FINANCE",
    metricId: "LIQUID_SAVINGS",
    value: 11800,
    unit: "€",
    metadata: {
      source: "PLAID",
      observedAtUtc: "2026-09-08T22:00:00Z",
      confidencePct: 96,
      freshnessHours: 13.5,
      maxAllowedAgeHours: 24,
    },
  },
  {
    signalId: "SIG-LRN-001",
    category: "LEARNING",
    metricId: "CURRICULUM_MODULES_SHIPPED",
    value: 8.0,
    unit: "modules",
    metadata: {
      source: "GITHUB",
      observedAtUtc: "2026-09-08T14:00:00Z",
      confidencePct: 92,
      freshnessHours: 21.5,
      maxAllowedAgeHours: 72,
    },
  },
];

export const CANONICAL_CONFLICTS: SignalConflict[] = [
  {
    conflictId: "CNF-001",
    metricId: "WEEKLY_FOCUS_HOURS",
    sourceA: "GOOGLE_CALENDAR",
    valueA: 18.5,
    confidenceA: 95,
    sourceB: "MANUAL_ENTRY",
    valueB: 12.0,
    confidenceB: 70,
    resolutionMethod: "PRIORITY",
    resolvedValue: 18.5,
    auditReason: "Google Calendar automated telemetry (Priority 2) takes precedence over manual estimate (Priority 3).",
    resolvedAtUtc: "2026-09-09T09:35:00Z",
  },
  {
    conflictId: "CNF-002",
    metricId: "SLEEP_DURATION",
    sourceA: "APPLE_HEALTH",
    valueA: 7.8,
    confidenceA: 98,
    sourceB: "OURA",
    valueB: 7.4,
    confidenceB: 96,
    resolutionMethod: "WEIGHTED",
    resolvedValue: 7.6,
    auditReason: "Weighted merge between two tier-1 wearable sources: (7.8*98 + 7.4*96) / (98 + 96) = 7.60h.",
    resolvedAtUtc: "2026-09-09T07:20:00Z",
  },
];

export const CANONICAL_DECISION_OUTCOMES: DecisionOutcome[] = [
  {
    decisionId: "DEC-2026-W30",
    recommendationTitle: "Enroll in Distributed Systems Architecture Sprint",
    category: "CAREER",
    expectedMetricGain: 6.0,
    actualMetricGain: 7.2,
    calibrationDeltaPct: 20.0,
    decidedAtUtc: "2026-07-28T18:00:00Z",
    outcomeObservedAtUtc: "2026-09-01T12:00:00Z",
    brierScoreContribution: 0.014,
  },
  {
    decisionId: "DEC-2026-W32",
    recommendationTitle: "Audit Recurring Cloud & Media SaaS Subscriptions",
    category: "FINANCE",
    expectedMetricGain: 120.0,
    actualMetricGain: 110.0,
    calibrationDeltaPct: -8.3,
    decidedAtUtc: "2026-08-11T09:00:00Z",
    outcomeObservedAtUtc: "2026-09-05T00:00:00Z",
    brierScoreContribution: 0.007,
  },
  {
    decisionId: "DEC-2026-W34",
    recommendationTitle: "Shift High-Intensity Sprint to 35m Zone-2 Aerobic Run",
    category: "HEALTH",
    expectedMetricGain: 8.0,
    actualMetricGain: 8.5,
    calibrationDeltaPct: 6.25,
    decidedAtUtc: "2026-08-25T07:00:00Z",
    outcomeObservedAtUtc: "2026-09-08T08:00:00Z",
    brierScoreContribution: 0.003,
  },
];

/**
 * Validates INV-OI84-P (Personal Signal Freshness Invariant).
 *
 * Verifies:
 * 1. Source is known and registered
 * 2. Timestamp is valid and present
 * 3. Age does not exceed maxAllowedAgeHours
 * 4. Computes natural decay: freshnessScore = 100 * max(0, 1 - ageHours / maxAgeHours)
 */
export function verifySignalFreshness(signal: PersonalSignal): SignalFreshnessResult {
  if (!signal.metadata.source || !SOURCE_RELIABILITY_REGISTRY[signal.metadata.source]) {
    return {
      isFresh: false,
      freshnessScore: 0,
      ageHours: signal.metadata.freshnessHours || 0,
      maxAgeHours: signal.metadata.maxAllowedAgeHours || 24,
      status: "STALE",
      errorCode: "SIGNAL_SOURCE_UNKNOWN",
    };
  }

  if (!signal.metadata.observedAtUtc) {
    return {
      isFresh: false,
      freshnessScore: 0,
      ageHours: signal.metadata.freshnessHours || 0,
      maxAgeHours: signal.metadata.maxAllowedAgeHours || 24,
      status: "STALE",
      errorCode: "SIGNAL_TIMESTAMP_MISSING",
    };
  }

  const age = Math.max(0, signal.metadata.freshnessHours);
  const maxAge = Math.max(1, signal.metadata.maxAllowedAgeHours || CATEGORY_FRESHNESS_CEILINGS[signal.category] || 24);

  if (age > maxAge) {
    return {
      isFresh: false,
      freshnessScore: 0,
      ageHours: age,
      maxAgeHours: maxAge,
      status: "STALE",
      errorCode: "SIGNAL_STALE",
    };
  }

  const freshnessScore = Math.round(100 * Math.max(0, 1 - age / maxAge));
  const status = freshnessScore >= 70 ? "FRESH" : freshnessScore >= 30 ? "AGING" : "STALE";
  const isFresh = freshnessScore >= 30;

  return {
    isFresh,
    freshnessScore,
    ageHours: age,
    maxAgeHours: maxAge,
    status,
  };
}

/**
 * Validates INV-OI86-P (Conflict Resolution Integrity Invariant).
 *
 * Employs deterministic resolution hierarchy:
 * Priority 1 (Wearable/Device) > Priority 2 (Connected API) > Priority 3 (Manual) > Priority 4 (Model)
 * If priorities are equal, applies weighted merge: sum(v_i * c_i) / sum(c_i).
 * Produces an immutable conflict ledger entry.
 */
export function resolveSignalConflicts(signals: PersonalSignal[]): {
  canonicalSignals: PersonalSignal[];
  conflicts: SignalConflict[];
} {
  const byMetric = new Map<string, PersonalSignal[]>();

  for (const s of signals) {
    const list = byMetric.get(s.metricId) || [];
    list.push(s);
    byMetric.set(s.metricId, list);
  }

  const canonicalSignals: PersonalSignal[] = [];
  const conflicts: SignalConflict[] = [];

  for (const [metricId, list] of byMetric.entries()) {
    if (list.length === 1) {
      canonicalSignals.push(list[0]);
      continue;
    }

    // Conflict exists: evaluate values
    const primary = list[0];
    const secondary = list[1];
    const diffPct = Math.abs((primary.value - secondary.value) / Math.max(1, secondary.value)) * 100;

    if (diffPct <= 2.0) {
      // De minimis variance -> take primary
      canonicalSignals.push(primary);
      continue;
    }

    const relA = SOURCE_RELIABILITY_REGISTRY[primary.metadata.source] || { priority: 99, confidencePct: 50 };
    const relB = SOURCE_RELIABILITY_REGISTRY[secondary.metadata.source] || { priority: 99, confidencePct: 50 };

    let resolutionMethod: "PRIORITY" | "WEIGHTED" | "USER_OVERRIDE" = "PRIORITY";
    let resolvedValue = primary.value;
    let auditReason = "";

    if (relA.priority < relB.priority) {
      resolvedValue = primary.value;
      resolutionMethod = "PRIORITY";
      auditReason = `${primary.metadata.source} (Priority ${relA.priority}) overrides ${secondary.metadata.source} (Priority ${relB.priority}).`;
    } else if (relB.priority < relA.priority) {
      resolvedValue = secondary.value;
      resolutionMethod = "PRIORITY";
      auditReason = `${secondary.metadata.source} (Priority ${relB.priority}) overrides ${primary.metadata.source} (Priority ${relA.priority}).`;
    } else {
      // Equal priority -> weighted merge
      const totalConf = relA.confidencePct + relB.confidencePct;
      resolvedValue = Number(((primary.value * relA.confidencePct + secondary.value * relB.confidencePct) / totalConf).toFixed(2));
      resolutionMethod = "WEIGHTED";
      auditReason = `Weighted merge between ${primary.metadata.source} (${relA.confidencePct}%) and ${secondary.metadata.source} (${relB.confidencePct}%).`;
    }

    const conflict: SignalConflict = {
      conflictId: `CNF-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
      metricId,
      sourceA: primary.metadata.source,
      valueA: primary.value,
      confidenceA: relA.confidencePct,
      sourceB: secondary.metadata.source,
      valueB: secondary.value,
      confidenceB: relB.confidencePct,
      resolutionMethod,
      resolvedValue,
      auditReason,
      resolvedAtUtc: new Date().toISOString(),
    };

    conflicts.push(conflict);

    canonicalSignals.push({
      ...primary,
      value: resolvedValue,
    });
  }

  return { canonicalSignals, conflicts };
}

/**
 * Validates INV-OI87-P (Signal Quality Coverage Invariant).
 *
 * Computes:
 * SignalQuality = (Freshness + Coverage + Confidence) / 3
 */
export function calculateSignalQuality(signals: PersonalSignal[]): SignalQualityComposite {
  if (signals.length === 0) {
    return {
      freshnessScore: 0,
      coverageScore: 0,
      confidenceScore: 0,
      overallQualityScore: 0,
      status: "CRITICAL",
    };
  }

  let totalFreshness = 0;
  let totalConfidence = 0;
  const categoriesPresent = new Set<string>();

  for (const s of signals) {
    const freshnessRes = verifySignalFreshness(s);
    totalFreshness += freshnessRes.freshnessScore;
    totalConfidence += s.metadata.confidencePct || 80;
    if (freshnessRes.isFresh) {
      categoriesPresent.add(s.category);
    }
  }

  const freshnessScore = Math.round(totalFreshness / signals.length);
  const confidenceScore = Math.round(totalConfidence / signals.length);
  const coverageScore = Math.round((categoriesPresent.size / 5) * 100); // 5 core categories

  const overallQualityScore = Math.round((freshnessScore + coverageScore + confidenceScore) / 3);

  let status: "EXCELLENT" | "GOOD" | "DEGRADED" | "CRITICAL" = "CRITICAL";
  if (overallQualityScore >= 90) {
    status = "EXCELLENT";
  } else if (overallQualityScore >= 75) {
    status = "GOOD";
  } else if (overallQualityScore >= 50) {
    status = "DEGRADED";
  }

  return {
    freshnessScore,
    coverageScore,
    confidenceScore,
    overallQualityScore,
    status,
  };
}

/**
 * Validates INV-OI85-P (Decision Outcome Capture Invariant).
 *
 * Computes calibration delta: (actual - expected) / expected * 100
 */
export function recordDecisionOutcome(
  decisionId: string,
  recommendationTitle: string,
  category: SignalCategory,
  expectedMetricGain: number,
  actualMetricGain: number,
  decidedAtUtc: string
): DecisionOutcome {
  const calibrationDeltaPct = expectedMetricGain === 0
    ? 0
    : Number((((actualMetricGain - expectedMetricGain) / expectedMetricGain) * 100).toFixed(2));

  // Normalized Brier quadratic penalty
  const rawDiff = actualMetricGain - expectedMetricGain;
  const brierScoreContribution = Number((Math.min(1.0, (rawDiff * rawDiff) / 1000)).toFixed(4));

  return {
    decisionId,
    recommendationTitle,
    category,
    expectedMetricGain,
    actualMetricGain,
    calibrationDeltaPct,
    decidedAtUtc,
    outcomeObservedAtUtc: new Date().toISOString(),
    brierScoreContribution,
  };
}

/**
 * Hydrates live Personal Digital Twin state from canonical verified signals.
 */
export function hydrateTwinFromSignals(
  canonicalSignals: PersonalSignal[],
  baseCapacity: PersonalCapacity
): {
  capacity: PersonalCapacity;
  energy: EnergyState;
  attention: AttentionState;
} {
  const signalMap = new Map<string, number>();
  for (const s of canonicalSignals) {
    signalMap.set(s.metricId, s.value);
  }

  // 1. Time / Focus hours
  const weeklyFocus = signalMap.get("WEEKLY_FOCUS_HOURS") || 18.5;
  const sleepDuration = signalMap.get("SLEEP_DURATION") || 7.8;
  const hrv = signalMap.get("HEART_RATE_VARIABILITY") || 68.0;

  // 2. Energy state hydration
  const recoveryScore = Math.min(100, Math.round((hrv / 80) * 100));
  const energyScore = Math.min(100, Math.round((sleepDuration / 8.0) * 50 + (hrv / 70) * 50));

  // 3. Attention capacity
  const focusCapacity = Math.min(100, Math.round((weeklyFocus / 20) * 100));

  return {
    capacity: {
      ...baseCapacity,
      energyCapacity: energyScore,
      attentionCapacity: focusCapacity,
    },
    energy: {
      energyScore,
      sleepQuality: Math.round(sleepDuration * 11),
      recoveryScore,
      stressLoad: Math.max(10, 100 - recoveryScore),
    },
    attention: {
      focusCapacity,
      contextSwitches: 6,
      cognitiveLoad: Math.max(15, 100 - focusCapacity),
    },
  };
}
