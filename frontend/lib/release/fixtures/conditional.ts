import { ReleaseReadinessResponse } from '../../../types/release-dashboard';
import { APPROVED_RELEASE_FIXTURE } from './approved';

export const CONDITIONAL_RELEASE_FIXTURE: ReleaseReadinessResponse = {
  ...APPROVED_RELEASE_FIXTURE,
  releaseId: 'REL-2026.09-STAGING',
  overallReadinessPct: 84,
  releaseDecision: 'CONDITIONAL',
  kpis: {
    ...APPROVED_RELEASE_FIXTURE.kpis,
    performanceScore: 82.0,
    qualityScore: 88.5,
  },
  gates: APPROVED_RELEASE_FIXTURE.gates.map((g, idx) => {
    if (idx === 6) { // M7
      return {
        ...g,
        status: 'WARNING',
        passedAssertions: g.assertionCount - 3,
        failureReason: 'Non-blocking Pareto frontier convergence latency warning (exceeded 500ms SLA).',
      };
    }
    return g;
  }),
  replayHash: 'REL-HASH-0xWARN8f1a2b3c4d5e',
};
