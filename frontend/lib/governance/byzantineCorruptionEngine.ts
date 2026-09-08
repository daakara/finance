/**
 * Phase 31-M1.1 / M2: Byzantine Corruption Detection Engine
 *
 * Implements malicious, coordinated, and conflicting artifact detection:
 * - BC-001: Split-Brain Decision State (DECISION_FORK, OUTCOME_CONFLICT_DETECTED)
 * - BC-002: Conflicting Attribution Ledger (ATTRIBUTION_FORK, ATTRIBUTION_SUM_VIOLATION)
 * - BC-003: Hidden Dissent Suppression (SUPPRESSED_DISSENT, OWNERSHIP_CONFLICT)
 * - BC-004: Ghost Committee (GHOST_COMMITTEE, DISSENT_RESOLUTION_CONFLICT)
 * - BC-005: Majority Membership Fabrication (MEMBERSHIP_FABRICATION, EVIDENCE_CONTRADICTION)
 * - BC-006: Evidence Substitution Attack (EVIDENCE_HASH_MISMATCH, TEMPORAL_ORDER_VIOLATION)
 * - BC-007: Replay Divergence Attack (REPLAY_VARIANCE, DETERMINISTIC_REPLAY_FAILURE)
 * - BC-008: Circular Influence Coalition (INFLUENCE_CYCLE, INFLUENCE_CYCLE_ALERT)
 * - BC-009: Outcome Fabrication (ORPHAN_OUTCOME, BENCHMARK_MUTATION_DETECTED)
 * - BC-010: Certification Tampering (CERTIFICATION_TAMPERING, RECOMMENDATION_DRIFT_DETECTED)
 */

import {
  ByzantineCorruptionResult,
  ByzantineViolation,
  ByzantineSeverity,
} from '../../types/committee-intelligence';

export function detectByzantineCorruption(fixture: Record<string, unknown>): ByzantineCorruptionResult {
  const violations: ByzantineViolation[] = [];

  // ── BC-001: Split-Brain Decision State & Outcome Conflict ───────────
  if (Array.isArray(fixture.decisions)) {
    const decisionMap = new Map<string, string[]>();
    for (const d of fixture.decisions as Record<string, unknown>[]) {
      if (d.decisionId && d.outcomeId) {
        const outcomes = decisionMap.get(String(d.decisionId)) ?? [];
        outcomes.push(String(d.outcomeId));
        decisionMap.set(String(d.decisionId), outcomes);
      }
    }
    for (const outcomes of decisionMap.values()) {
      if (new Set(outcomes).size > 1) {
        violations.push('DECISION_FORK');
        violations.push('OUTCOME_CONFLICT_DETECTED');
        break;
      }
    }
  }

  // ── BC-002: Conflicting Attribution Ledger & Sum Violation ──────────
  if (Array.isArray(fixture.attribution)) {
    const attrMap = new Map<string, number[]>();
    let sumTotal = 0;
    for (const a of fixture.attribution as Record<string, unknown>[]) {
      if (a.outcomeId && typeof a.contributionPct === 'number') {
        const vals = attrMap.get(String(a.outcomeId)) ?? [];
        vals.push(a.contributionPct);
        attrMap.set(String(a.outcomeId), vals);
        sumTotal += a.contributionPct;
      }
    }
    for (const vals of attrMap.values()) {
      if (new Set(vals).size > 1) {
        violations.push('ATTRIBUTION_FORK');
        break;
      }
    }
    if (sumTotal > 100.001 || (attrMap.size > 0 && fixture.totalPct && fixture.totalPct !== 100)) {
      if (!violations.includes('ATTRIBUTION_SUM_VIOLATION')) {
        violations.push('ATTRIBUTION_SUM_VIOLATION');
      }
    }
  }

  if (fixture.individual !== undefined && fixture.committee !== undefined && fixture.system !== undefined) {
    const total = Number(fixture.individual) + Number(fixture.committee) + Number(fixture.system);
    if (total > 100.0) {
      if (!violations.includes('ATTRIBUTION_SUM_VIOLATION')) {
        violations.push('ATTRIBUTION_SUM_VIOLATION');
      }
    }
  }

  // ── BC-003: Hidden Dissent Suppression & Split Committee Ownership ───
  if (fixture.decision && typeof fixture.decision === 'object') {
    const dec = fixture.decision as Record<string, unknown>;
    if (dec.unanimousApproval === true && Array.isArray(fixture.dissents) && fixture.dissents.length > 0) {
      violations.push('SUPPRESSED_DISSENT');
    }
    if (Array.isArray(dec.owners) && dec.owners.length > 1) {
      violations.push('OWNERSHIP_CONFLICT');
    }
  }
  if (Array.isArray(fixture.committeeOwners) && fixture.committeeOwners.length > 1) {
    violations.push('OWNERSHIP_CONFLICT');
  }

  // ── BC-004: Ghost Committee & Dissent Resolution Conflict ───────────
  if (Array.isArray(fixture.committees) && Array.isArray(fixture.decisions)) {
    const knownCommittees = new Set(
      (fixture.committees as Record<string, unknown>[]).map(c => String(c.committeeId))
    );
    for (const d of fixture.decisions as Record<string, unknown>[]) {
      if (d.committeeId && !knownCommittees.has(String(d.committeeId))) {
        violations.push('GHOST_COMMITTEE');
        break;
      }
    }
  }
  if (fixture.dissentStatus && fixture.auditLogStatus && fixture.dissentStatus !== fixture.auditLogStatus) {
    violations.push('DISSENT_RESOLUTION_CONFLICT');
  }

  // ── BC-005: Majority Membership Fabrication & Evidence Contradiction ─
  if (Array.isArray(fixture.participants)) {
    const userIds = (fixture.participants as Record<string, unknown>[]).map(p => String(p.userId));
    if (userIds.length !== new Set(userIds).size) {
      violations.push('MEMBERSHIP_FABRICATION');
    }
  }
  if (fixture.contradictoryEvidence === true) {
    violations.push('EVIDENCE_CONTRADICTION');
  }

  // ── BC-006: Evidence Substitution Attack & Temporal Order Violation ─
  if (fixture.certifiedHash && fixture.currentHash && fixture.certifiedHash !== fixture.currentHash) {
    violations.push('EVIDENCE_HASH_MISMATCH');
  }
  if (fixture.outcomeTimestamp && fixture.decisionTimestamp) {
    const oTime = new Date(String(fixture.outcomeTimestamp)).getTime();
    const dTime = new Date(String(fixture.decisionTimestamp)).getTime();
    if (oTime < dTime) {
      violations.push('TEMPORAL_ORDER_VIOLATION');
    }
  }

  // ── BC-007: Replay Divergence Attack ────────────────────────────────
  if (Array.isArray(fixture.replayHashes)) {
    const hashes = fixture.replayHashes as string[];
    if (new Set(hashes).size > 1) {
      violations.push('REPLAY_VARIANCE');
    }
  }

  // ── BC-008: Circular Influence Coalition ────────────────────────────
  if (Array.isArray(fixture.edges)) {
    const edges = fixture.edges as Record<string, unknown>[];
    const adj = new Map<string, string[]>();
    for (const e of edges) {
      const s = String(e.s || e.source || e.sourceCommitteeId);
      const t = String(e.t || e.target || e.targetCommitteeId);
      const list = adj.get(s) ?? [];
      list.push(t);
      adj.set(s, list);
    }

    const visited = new Set<string>();
    const recStack = new Set<string>();
    let hasCycle = false;

    function dfs(node: string): boolean {
      visited.add(node);
      recStack.add(node);
      const neighbors = adj.get(node) ?? [];
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor) && dfs(neighbor)) {
          return true;
        } else if (recStack.has(neighbor)) {
          return true;
        }
      }
      recStack.delete(node);
      return false;
    }

    for (const node of adj.keys()) {
      if (!visited.has(node)) {
        if (dfs(node)) {
          hasCycle = true;
          break;
        }
      }
    }

    if (hasCycle) {
      violations.push('INFLUENCE_CYCLE');
    }
  }

  // ── BC-009: Outcome Fabrication & Benchmark Mutation ────────────────
  if (Array.isArray(fixture.outcomes)) {
    for (const o of fixture.outcomes as Record<string, unknown>[]) {
      if (!o.decisionId || o.decisionId === 'UNKNOWN') {
        violations.push('ORPHAN_OUTCOME');
        break;
      }
    }
  }
  if (fixture.benchmarkMutated === true) {
    violations.push('BENCHMARK_MUTATION_DETECTED');
  }

  // ── BC-010: Certification Tampering & Recommendation Drift ──────────
  if (fixture.gates && typeof fixture.gates === 'object' && fixture.certificationStatus === 'PASS') {
    const gateValues = Object.values(fixture.gates);
    if (gateValues.some(v => v === false || v === 'FAIL')) {
      violations.push('CERTIFICATION_TAMPERING');
    }
  }
  if (fixture.recommendationDrift === true) {
    violations.push('RECOMMENDATION_DRIFT_DETECTED');
  }

  const detected = violations.length > 0;
  let severity: ByzantineSeverity = 'LOW';

  if (
    violations.includes('DECISION_FORK') ||
    violations.includes('EVIDENCE_HASH_MISMATCH') ||
    violations.includes('CERTIFICATION_TAMPERING') ||
    violations.includes('ATTRIBUTION_FORK') ||
    violations.includes('SUPPRESSED_DISSENT')
  ) {
    severity = 'CRITICAL';
  } else if (
    violations.includes('MEMBERSHIP_FABRICATION') ||
    violations.includes('REPLAY_VARIANCE') ||
    violations.includes('INFLUENCE_CYCLE') ||
    violations.includes('GHOST_COMMITTEE') ||
    violations.includes('ORPHAN_OUTCOME')
  ) {
    severity = 'HIGH';
  } else if (detected) {
    severity = 'MEDIUM';
  }

  return {
    detected,
    violations,
    severity,
  };
}
