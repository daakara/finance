import { ReleaseReadinessResponse } from '../../../types/release-dashboard';
import { APPROVED_RELEASE_FIXTURE } from './approved';

export const BLOCKED_RELEASE_FIXTURE: ReleaseReadinessResponse = {
  ...APPROVED_RELEASE_FIXTURE,
  releaseId: 'REL-2026.09-QUARANTINED',
  overallReadinessPct: 68,
  releaseDecision: 'BLOCKED',
  security: {
    criticalVulnerabilities: 1,
    replayDriftIncidents: 1,
    consistencyViolations: 0,
    governanceViolations: 0,
  },
  kpis: {
    qualityScore: 65.0,
    governanceScore: 70.0,
    accessibilityScore: 100.0,
    resilienceScore: 62.0,
    performanceScore: 90.0,
    executiveReadinessScore: 60.0,
  },
  gates: APPROVED_RELEASE_FIXTURE.gates.map((g, idx) => {
    if (idx === 7) { // M8
      return {
        ...g,
        status: 'FAIL',
        passedAssertions: g.assertionCount - 15,
        failureReason: 'FAIL-CLOSED: Replay drift detected in failover witness node (hash discrepancy).',
      };
    }
    return g;
  }),
  replayHash: 'REL-HASH-0xBLOCKEDdeadbeef',
};
