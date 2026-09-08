/**
 * Phase 31: Release API Mock Service
 *
 * Provides static-export compatible mock data adapter for:
 * - Approved Production Baseline (REL-2026.09-PROD)
 * - Conditional Warning Baseline (REL-2026.09-WARN)
 * - Blocked Governance Baseline (REL-2026.09-BLOCK)
 */

import { ReleaseReadinessResponse } from '../../types/release-dashboard';
import { APPROVED_RELEASE_FIXTURE } from './fixtures/approved';
import { CONDITIONAL_RELEASE_FIXTURE } from './fixtures/conditional';
import { BLOCKED_RELEASE_FIXTURE } from './fixtures/blocked';

export type MockReleaseScenario = 'APPROVED' | 'CONDITIONAL' | 'BLOCKED';

export interface ReleaseApiOptions {
  scenario?: MockReleaseScenario;
  simulatedDelayMs?: number;
}

export async function fetchReleaseReadiness(
  options: ReleaseApiOptions = {}
): Promise<ReleaseReadinessResponse> {
  const scenario = options.scenario || 'APPROVED';
  const delay = options.simulatedDelayMs ?? 0;

  if (delay > 0) {
    await new Promise(resolve => setTimeout(resolve, delay));
  }

  switch (scenario) {
    case 'CONDITIONAL':
      return JSON.parse(JSON.stringify(CONDITIONAL_RELEASE_FIXTURE));
    case 'BLOCKED':
      return JSON.parse(JSON.stringify(BLOCKED_RELEASE_FIXTURE));
    case 'APPROVED':
    default:
      return JSON.parse(JSON.stringify(APPROVED_RELEASE_FIXTURE));
  }
}
