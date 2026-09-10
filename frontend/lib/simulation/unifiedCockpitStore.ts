"use client";

/**
 * Horizon 14: Unified CQRS Read Model Store
 *
 * Implements the single source of truth for the ARX Unified Operating Cockpit.
 * Consolidates all intelligence metrics from Horizons 5–13 into an authentic,
 * API-backed read state.
 *
 * Zero-Login Architecture:
 * Profile identifiers are local record selectors, not proofs of identity.
 * Default selector is "default". Never requires authentication.
 *
 * Epistemic Invariant:
 * - INV-OI110-P: Single Source of Truth (All components and routes consume the shared state store).
 * - Fictional snapshot (fake David, fake 84/89/61) is completely eliminated from production.
 */

import { useState, useEffect } from "react";
import { getAnonymousUserId } from "../portfolio";

export interface TriadIndex {
  lhi: number;
  hhi: number;
  iai: number;
  compositeResilience: number;
  status: 'STABLE_COMPOUNDING' | 'AT_RISK' | 'DEGRADED';
  interpretation: string;
}

export interface SignalQualityState {
  freshness: 'REALTIME' | 'DELAYED' | 'STALE' | 'UNAVAILABLE';
  confidence: number | null;
  activeSignalsCount: number;
  highConvictionRatio: number | null;
  lastTelemetrySync: string | null;
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
  projectedValue3Yr: string | null;
  confidencePct: number | null;
  runwayStatus?: string;
  explanation?: string;
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
  sharedResourceLoad: number;
  conflictRisk: 'LOW' | 'MEDIUM' | 'ELEVATED';
  keySyncItem: string;
  stakeholderCount: number;
}

export interface RunwayState {
  monthsUnencumbered: number | null;
  liquidReserves: number | null;
  burnRateMonthly: number | null;
  runwayStatus?: 'CALCULATED' | 'ZERO_EXPENDITURE' | 'EXPENDITURE_UNRECORDED' | 'RESERVES_UNRECORDED' | 'UNAVAILABLE';
  runwayShieldStatus: 'PROTECTED' | 'CAUTION' | 'CRITICAL' | 'UNCONFIGURED';
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
  brierScore: number;
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

export const DEFAULT_WORKBENCHES: SpecialistWorkbenchMeta[] = [
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
];

export interface CockpitPortfolioSummary {
  holdingsCount: number;
  totalMarketValue: number | null;
  totalCostBasis: number;
  unrealizedPnL: number | null;
  isComplete: boolean;
  status: string;
}

export type UnifiedCockpitStatus = 'IDLE' | 'LOADING' | 'AVAILABLE' | 'UNAVAILABLE' | 'ERROR' | 'PERSISTED_STORE';

export interface UnifiedCockpitState {
  version: string;
  generatedAt?: string;
  status: UnifiedCockpitStatus;
  available: boolean;
  errorMessage: string | null;
  subjectId: string;
  subjectName?: string;
  targetIdentityRole?: string;
  triad: TriadIndex | null;
  portfolio: CockpitPortfolioSummary | null;
  signalQuality: SignalQualityState | null;
  nextBestAction: ActionItem | null;
  secondaryActions: ActionItem[];
  primaryForecast: OutcomeForecast | null;
  outcomeForecasts: OutcomeForecast[];
  identityDrift: IdentityDriftAlert | null;
  householdHealth: HouseholdHealthState | null;
  runway: RunwayState | null;
  activeConstraints: ConstraintAlert[];
  recoveryIndicator: RecoveryIndicatorState | null;
  futurePaths: FuturePathway[];
  skillTrajectories: SkillTrajectory[];
  calibrationScore: CalibrationState | null;
  sharedResources: SharedResourceItem[];
  workbenches: SpecialistWorkbenchMeta[];
}

export const EMPTY_COCKPIT_STATE: UnifiedCockpitState = {
  version: '14.1.0-CQRS',
  status: 'UNAVAILABLE',
  available: false,
  errorMessage: null,
  subjectId: 'default',
  triad: null,
  portfolio: null,
  signalQuality: null,
  nextBestAction: null,
  secondaryActions: [],
  primaryForecast: null,
  outcomeForecasts: [],
  identityDrift: null,
  householdHealth: null,
  runway: null,
  activeConstraints: [],
  recoveryIndicator: null,
  futurePaths: [],
  skillTrajectories: [],
  calibrationScore: null,
  sharedResources: [],
  workbenches: DEFAULT_WORKBENCHES,
};

// Singleton reactive store state
let globalCockpitState: UnifiedCockpitState = EMPTY_COCKPIT_STATE;
const storeListeners = new Set<() => void>();
let isFetching = false;
let hasAttemptedInitialFetch = false;
let activeRequestId = 0;
let activeRecordSelector: string = "default";

export function getActiveRecordSelector(): string {
  return activeRecordSelector;
}

export function setActiveRecordSelector(selector: string): void {
  activeRecordSelector = selector;
}

function notifyListeners(): void {
  storeListeners.forEach((fn) => {
    try {
      fn();
    } catch (err) {
      console.error("Store listener error:", err);
    }
  });
}

/**
 * Fetches authoritative CQRS Cockpit Read Model from backend API.
 * Sequence-tracked to prevent older asynchronous responses from overwriting newer context.
 * Strictly re-checks request currency before committing data after await res.json().
 */
export async function fetchUnifiedCockpitState(profileId?: string, force = false): Promise<UnifiedCockpitState> {
  if (typeof window === "undefined") return EMPTY_COCKPIT_STATE;

  const resolvedSelector = (profileId && profileId.trim())
    ? profileId.trim()
    : activeRecordSelector || getAnonymousUserId() || "default";

  const selectorChanged = resolvedSelector !== activeRecordSelector;
  activeRecordSelector = resolvedSelector;

  if (isFetching && !force && !selectorChanged) return globalCockpitState;

  isFetching = true;
  const requestId = ++activeRequestId;

  // Prevent data from previous selector being presented as belonging to a newly selected record
  if (selectorChanged || globalCockpitState.subjectId !== resolvedSelector) {
    globalCockpitState = {
      ...EMPTY_COCKPIT_STATE,
      status: 'LOADING',
      subjectId: resolvedSelector,
      errorMessage: null,
    };
    notifyListeners();
  } else if (globalCockpitState.status !== 'AVAILABLE') {
    globalCockpitState = {
      ...globalCockpitState,
      status: 'LOADING',
      errorMessage: null,
    };
    notifyListeners();
  }

  try {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "https://web-production-e370b.up.railway.app/api/v1";
    const res = await fetch(`${baseUrl}/cockpit/state`, {
      headers: {
        "Content-Type": "application/json",
        "X-Profile-Id": resolvedSelector,
        "X-User-Id": resolvedSelector,
        "Cache-Control": "no-cache",
      },
      signal: AbortSignal.timeout(5000),
    });

    // Prevent stale responses from overwriting newer context before reading body
    if (requestId !== activeRequestId) {
      return globalCockpitState;
    }

    if (res.ok) {
      const data = await res.json();

      // Crucial: Re-verify request currency after awaiting res.json() before committing state
      if (requestId !== activeRequestId) {
        return globalCockpitState;
      }

      hasAttemptedInitialFetch = true;
      globalCockpitState = {
        version: data.version || '14.1.0-CQRS',
        generatedAt: data.generatedAt,
        status: data.status || (data.available ? 'AVAILABLE' : 'UNAVAILABLE'),
        available: Boolean(data.available),
        errorMessage: null,
        subjectId: data.subjectId || resolvedSelector,
        subjectName: data.subjectName,
        targetIdentityRole: data.targetIdentityRole,
        triad: data.triad || null,
        portfolio: data.portfolio || null,
        signalQuality: data.signalQuality || null,
        nextBestAction: data.nextBestAction || null,
        secondaryActions: Array.isArray(data.secondaryActions) ? data.secondaryActions : [],
        primaryForecast: data.primaryForecast || null,
        outcomeForecasts: Array.isArray(data.outcomeForecasts) ? data.outcomeForecasts : [],
        identityDrift: data.identityDrift || null,
        householdHealth: data.householdHealth || null,
        runway: data.runway || null,
        activeConstraints: Array.isArray(data.activeConstraints) ? data.activeConstraints : [],
        recoveryIndicator: data.recoveryIndicator || null,
        futurePaths: Array.isArray(data.futurePaths) ? data.futurePaths : [],
        skillTrajectories: Array.isArray(data.skillTrajectories) ? data.skillTrajectories : [],
        calibrationScore: data.calibrationScore || null,
        sharedResources: Array.isArray(data.sharedResources) ? data.sharedResources : [],
        workbenches: DEFAULT_WORKBENCHES,
      };
      notifyListeners();
      return globalCockpitState;
    } else {
      // Re-verify request currency before committing error state
      if (requestId !== activeRequestId) {
        return globalCockpitState;
      }

      hasAttemptedInitialFetch = true;
      globalCockpitState = {
        ...globalCockpitState,
        status: 'ERROR',
        available: false,
        errorMessage: `API request failed with HTTP status ${res.status}`,
      };
      notifyListeners();
      return globalCockpitState;
    }
  } catch (err: any) {
    if (requestId === activeRequestId) {
      hasAttemptedInitialFetch = true;
      globalCockpitState = {
        ...globalCockpitState,
        status: 'ERROR',
        available: false,
        errorMessage: err?.message || 'Network error fetching cockpit state',
      };
      notifyListeners();
    }
    console.warn("Could not fetch unified cockpit state from API:", err);
  } finally {
    if (requestId === activeRequestId) {
      isFetching = false;
    }
  }
  return globalCockpitState;
}

/**
 * Triggers a forced reload/retry of the Unified Cockpit State.
 * Retains the active record selector when called without arguments.
 */
export async function refreshUnifiedCockpit(profileId?: string): Promise<UnifiedCockpitState> {
  const target = profileId || activeRecordSelector;
  return fetchUnifiedCockpitState(target, true);
}

/**
 * Invalidates current store state and requests a fresh background update.
 * Retains the active record selector when called without arguments.
 */
export function invalidateCockpitState(profileId?: string): void {
  hasAttemptedInitialFetch = false;
  const target = profileId || activeRecordSelector;
  fetchUnifiedCockpitState(target, true).catch(() => {});
}

// Auto-wire confirmed mutations (portfolio additions/edits/deletions) to refresh cockpit read state
if (typeof window !== "undefined") {
  window.addEventListener("finance:portfolio-updated", () => {
    refreshUnifiedCockpit().catch(() => {});
  });
}

/**
 * Synchronously returns the current Unified CQRS Cockpit State.
 * Automatically triggers background fetch if running in browser and uninitialized.
 */
export function getUnifiedCockpitState(): UnifiedCockpitState {
  if (typeof window !== "undefined" && !hasAttemptedInitialFetch && !isFetching) {
    fetchUnifiedCockpitState().catch(() => {});
  }
  return globalCockpitState;
}

/**
 * React hook returning the reactive Unified CQRS Cockpit State.
 * Automatically triggers background fetch and subscribes to updates.
 */
export function useUnifiedCockpit(): UnifiedCockpitState {
  const [state, setState] = useState<UnifiedCockpitState>(globalCockpitState);

  useEffect(() => {
    const handleUpdate = () => {
      setState(globalCockpitState);
    };

    storeListeners.add(handleUpdate);
    if (!hasAttemptedInitialFetch && !isFetching) {
      fetchUnifiedCockpitState().catch(() => {});
    }

    return () => {
      storeListeners.delete(handleUpdate);
    };
  }, []);

  return state;
}

export interface InvariantVerificationResult {
  compliant: boolean;
  invariantId: string;
  violations: string[];
  metadata?: Record<string, unknown>;
}

/**
 * INV-OI110-P: Single Source of Truth
 * Asserts cross-component consistency: all views must consume and display identical metrics.
 */
export function verifyUnifiedSourceOfTruth(
  states: UnifiedCockpitState[]
): InvariantVerificationResult {
  const violations: string[] = [];

  if (states.length === 0) {
    return {
      compliant: true,
      invariantId: 'INV-OI110-P',
      violations: [],
    };
  }

  const first = states[0];
  states.forEach((s, idx) => {
    if (s.triad?.lhi !== first.triad?.lhi) {
      violations.push(`INV-OI110-P VIOLATION: State #${idx} has desynced LHI (${s.triad?.lhi} vs ${first.triad?.lhi}).`);
    }
    if (s.triad?.hhi !== first.triad?.hhi) {
      violations.push(`INV-OI110-P VIOLATION: State #${idx} has desynced HHI (${s.triad?.hhi} vs ${first.triad?.hhi}).`);
    }
    if (s.triad?.iai !== first.triad?.iai) {
      violations.push(`INV-OI110-P VIOLATION: State #${idx} has desynced IAI (${s.triad?.iai} vs ${first.triad?.iai}).`);
    }
    if (s.nextBestAction?.id !== first.nextBestAction?.id) {
      violations.push(`INV-OI110-P VIOLATION: State #${idx} has desynced Primary Action.`);
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
      status: first.status,
      available: first.available,
    },
  };
}
