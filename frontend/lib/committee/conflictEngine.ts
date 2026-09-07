/**
 * ARX Terminal vNext - Deterministic Baseline Conflict Resolution Engine
 * Reference: docs/architecture/COMMITTEE_INTELLIGENCE_ARCHITECTURE.md
 */

import { BaselineConflict, CommitteeBaseline } from "../../types/committee-intelligence";

export function resolveConflict(
  baselineA: CommitteeBaseline,
  baselineB: CommitteeBaseline
): CommitteeBaseline {
  // If either is not approved or pending, the approved one wins
  if (baselineA.status === "ACTIVE" && baselineB.status !== "ACTIVE") return baselineA;
  if (baselineB.status === "ACTIVE" && baselineA.status !== "ACTIVE") return baselineB;

  // If both approved, latest timestamp wins
  const timeA = new Date(baselineA.acknowledgedAt).getTime();
  const timeB = new Date(baselineB.acknowledgedAt).getTime();

  return timeB >= timeA ? baselineB : baselineA;
}

export function evaluateConflict(
  baselineA: CommitteeBaseline,
  baselineB: CommitteeBaseline
): BaselineConflict {
  const conflictCreated = baselineA.snapshotHash !== baselineB.snapshotHash;
  const winner = resolveConflict(baselineA, baselineB);

  return {
    conflictId: `conf-${Date.now()}`,
    ticker: baselineA.ticker,
    committeeId: baselineA.committeeId,
    baselineA: baselineA.baselineId,
    baselineB: baselineB.baselineId,
    resolvedBy: winner.approvedBy || "system",
    resolution: "LATEST_ACCEPTED",
    conflictCreated,
    timestamp: new Date().toISOString(),
  };
}

export function canActivate(baseline: CommitteeBaseline): boolean {
  return baseline.status === "ACTIVE" || Boolean(baseline.approvedBy);
}
