/**
 * Phase 31-M1.1: Governance Fixture Validation Engine
 *
 * Implements strict schema verification for:
 * - Replay Fixtures (FIX-R01 to FIX-R05)
 * - Stress Fixtures (FIX-S01 to FIX-S04)
 * - Corruption Fixtures
 * - Certified Audit Snapshots
 */

import {
  ReplayFixture,
  StressProfile,
  CorruptionFixture,
  CommitteeAuditSnapshot,
} from '../../types/committee-intelligence';

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

export function validateReplayFixture(fixture: Partial<ReplayFixture>): ValidationResult {
  const errors: string[] = [];
  if (!fixture.fixtureId) errors.push('MISSING_FIXTURE_ID');
  if (!fixture.fixtureVersion) errors.push('MISSING_VERSION');
  if (!fixture.committees || fixture.committees.length === 0) errors.push('MISSING_COMMITTEE_STATE');
  if (!fixture.decisions) errors.push('MISSING_DECISIONS');
  if (!fixture.expectedResults) {
    errors.push('MISSING_EXPECTED_RESULTS');
  } else {
    if (
      fixture.expectedResults.committeeODEI === undefined ||
      fixture.expectedResults.committeeODEI < 0 ||
      fixture.expectedResults.committeeODEI > 100 ||
      Number.isNaN(fixture.expectedResults.committeeODEI)
    ) {
      errors.push('INVALID_ODEI');
    }
    if (
      fixture.expectedResults.transparencyCoveragePct === undefined ||
      fixture.expectedResults.transparencyCoveragePct < 0 ||
      fixture.expectedResults.transparencyCoveragePct > 100
    ) {
      errors.push('INVALID_TRANSPARENCY_COVERAGE');
    }
  }
  return { valid: errors.length === 0, errors };
}

export function validateStressFixture(fixture: Partial<StressProfile>): ValidationResult {
  const errors: string[] = [];
  if (fixture.committeeCount === undefined || fixture.committeeCount <= 0) {
    errors.push('INVALID_COMMITTEE_COUNT');
  }
  if (fixture.decisionCount === undefined || fixture.decisionCount < 0) {
    errors.push('INVALID_DECISION_COUNT');
  }
  if (fixture.participantCount === undefined || fixture.participantCount <= 0) {
    errors.push('INVALID_MEMBER_COUNT');
  }
  if (fixture.targetDurationMs === undefined || fixture.targetDurationMs <= 0) {
    errors.push('MISSING_EXPECTATION');
  }
  return { valid: errors.length === 0, errors };
}

export function validateCorruptionFixture(fixture: Partial<CorruptionFixture>): ValidationResult {
  const errors: string[] = [];
  if (!fixture.fixtureId) errors.push('MISSING_FIXTURE_ID');
  if (!fixture.corruptionType) errors.push('MISSING_CORRUPTION_TYPE');
  if (!fixture.expectedInvariantViolation) errors.push('MISSING_EXPECTED_VIOLATION');
  if (!fixture.payload) errors.push('MISSING_PAYLOAD');
  return { valid: errors.length === 0, errors };
}

export function validateCertifiedSnapshot(snapshot: Partial<CommitteeAuditSnapshot>): ValidationResult {
  const errors: string[] = [];
  if (!snapshot.snapshotId) errors.push('MISSING_SNAPSHOT_ID');
  if (!snapshot.hash) errors.push('MISSING_HASH');
  if (!snapshot.capturedAtUtc) errors.push('MISSING_CERTIFICATION_DATE');
  if (!snapshot.proposalHash) errors.push('MISSING_PROPOSAL_HASH');
  return { valid: errors.length === 0, errors };
}
