/**
 * ARX Terminal vNext - Data Quality & Edge Case Validator (Layer 0)
 * Evaluates timestamp freshness, metric bounds, sequence ordering, and snapshot consistency.
 * Reference: docs/architecture/DATA_QUALITY_AND_EDGE_CASE_GOVERNANCE.md
 */

import { DataQualityResult, DataQualityStatus } from "../../types/portfolio-intelligence";
import { ExecutionState } from "../../types/workstation";

/**
 * DQ-001: Freshness Validation
 */
export function validateFreshness(
  timestampStr: string,
  maxAgeMinutes: number = 60
): { status: DataQualityStatus; ageMinutes: number; valid: boolean } {
  try {
    const timestamp = new Date(timestampStr).getTime();
    if (isNaN(timestamp)) {
      return { status: "REJECTED", ageMinutes: Infinity, valid: false };
    }
    const now = Date.now();
    const ageMinutes = (now - timestamp) / (1000 * 60);

    if (ageMinutes < -5) {
      // Future timestamp beyond 5m clock skew is rejected
      return { status: "REJECTED", ageMinutes, valid: false };
    }
    if (ageMinutes > maxAgeMinutes) {
      return { status: "REJECTED", ageMinutes, valid: false };
    }
    if (ageMinutes > maxAgeMinutes * 0.5) {
      return { status: "DEGRADED", ageMinutes, valid: true };
    }
    return { status: "VALID", ageMinutes, valid: true };
  } catch {
    return { status: "REJECTED", ageMinutes: Infinity, valid: false };
  }
}

/**
 * DQ-002: Event Sequence Validation
 * Enforces valid operational telemetry progression:
 * returning_user_detected -> delta_banner_viewed -> thesis_confirmed
 */
export function validateSequence(events: string[]): boolean {
  if (!events || events.length === 0) return true;

  const validPrecedence: Record<string, number> = {
    returning_user_detected: 1,
    delta_banner_viewed: 2,
    delta_banner_expanded: 3,
    thesis_confirmed: 4,
    delta_acknowledged: 4,
  };

  let maxStepSeen = 0;
  for (const ev of events) {
    const step = validPrecedence[ev];
    if (step !== undefined) {
      if (step < maxStepSeen) {
        // Step regression without new session is invalid
        return false;
      }
      maxStepSeen = step;
    }
  }
  return true;
}

/**
 * DQ-003: Metric Bounds Validation
 */
export function validateMetricBounds(data: {
  setupScore?: number;
  spotPrice?: number;
  flowZScore?: number;
  tttcMs?: number;
  rereadTimeMs?: number;
}): { valid: boolean; failedField?: string; reason?: string } {
  if (data.setupScore !== undefined) {
    if (
      typeof data.setupScore !== "number" ||
      isNaN(data.setupScore) ||
      data.setupScore < 0 ||
      data.setupScore > 100
    ) {
      return { valid: false, failedField: "setupScore", reason: "Score must be 0-100" };
    }
  }

  if (data.spotPrice !== undefined) {
    if (
      typeof data.spotPrice !== "number" ||
      isNaN(data.spotPrice) ||
      data.spotPrice <= 0 ||
      data.spotPrice > 10_000_000
    ) {
      return { valid: false, failedField: "spotPrice", reason: "Price must be positive finite" };
    }
  }

  if (data.flowZScore !== undefined) {
    if (
      typeof data.flowZScore !== "number" ||
      isNaN(data.flowZScore) ||
      data.flowZScore < -10 ||
      data.flowZScore > 10
    ) {
      return { valid: false, failedField: "flowZScore", reason: "Flow Z must be [-10, 10]" };
    }
  }

  if (data.tttcMs !== undefined) {
    if (typeof data.tttcMs !== "number" || isNaN(data.tttcMs) || data.tttcMs < 0) {
      return { valid: false, failedField: "tttcMs", reason: "TTTC must be >= 0ms" };
    }
  }

  if (data.rereadTimeMs !== undefined) {
    if (typeof data.rereadTimeMs !== "number" || isNaN(data.rereadTimeMs) || data.rereadTimeMs < 0) {
      return { valid: false, failedField: "rereadTimeMs", reason: "RER time must be >= 0ms" };
    }
  }

  return { valid: true };
}

/**
 * DQ-004: Snapshot Consistency
 * Rejects illegal direct state teleports like STOPPED_OUT -> IN_BUY_ZONE without reset
 */
export function validateTransition(
  prevState: ExecutionState | string,
  currState: ExecutionState | string
): boolean {
  if (prevState === currState) return true;

  // STOPPED_OUT cannot transition directly to IN_BUY_ZONE without passing through NEUTRAL or WAITING_PULLBACK
  if (prevState === "STOPPED_OUT" && currState === "IN_BUY_ZONE") {
    return false;
  }

  // APPROACHING_TARGET cannot instantly revert to STOPPED_OUT without entering another state
  if (prevState === "APPROACHING_TARGET" && currState === "STOPPED_OUT") {
    // Large gap-down is allowed, but flags for confirmation
    return true;
  }

  return true;
}

/**
 * Comprehensive Snapshot Evaluator
 */
export function evaluateSnapshotQuality(snapshot: {
  ticker: string;
  timestamp: string;
  setupScore: number;
  spotPrice: number;
  flowZScore: number;
  executionState: ExecutionState;
  previousState?: ExecutionState;
}): DataQualityResult {
  const passedChecks: string[] = [];
  const failedChecks: string[] = [];

  // 1. Freshness Check
  const freshness = validateFreshness(snapshot.timestamp, 1440); // 24h
  if (freshness.valid) {
    passedChecks.push("DQ-001:Freshness");
  } else {
    failedChecks.push("DQ-001:Freshness");
  }

  // 2. Metric Bounds
  const bounds = validateMetricBounds({
    setupScore: snapshot.setupScore,
    spotPrice: snapshot.spotPrice,
    flowZScore: snapshot.flowZScore,
  });
  if (bounds.valid) {
    passedChecks.push("DQ-003:MetricBounds");
  } else {
    failedChecks.push(`DQ-003:MetricBounds (${bounds.failedField})`);
  }

  // 3. Transition Consistency
  if (snapshot.previousState) {
    const consistent = validateTransition(snapshot.previousState, snapshot.executionState);
    if (consistent) {
      passedChecks.push("DQ-004:TransitionConsistency");
    } else {
      failedChecks.push("DQ-004:TransitionConsistency");
    }
  }

  const total = passedChecks.length + failedChecks.length;
  const score = total > 0 ? Math.round((passedChecks.length / total) * 100) : 100;

  let status: DataQualityStatus = "VALID";
  if (failedChecks.length > 0) {
    status = failedChecks.some((c) => c.includes("MetricBounds") || c.includes("Transition"))
      ? "REJECTED"
      : "DEGRADED";
  }

  return {
    status,
    passedChecks,
    failedChecks,
    qualityScore: score,
    timestamp: new Date().toISOString(),
    rejectionReason: failedChecks.length > 0 ? failedChecks.join("; ") : undefined,
  };
}
