/**
 * Phase 31-M4: Organizational Risk Registry Engine (Epic M4-102)
 *
 * Implements:
 * - Canonical Organizational Risk Ledger (RSK-001 through RSK-010)
 * - Strict Validation Rules (VR-R01 to VR-R06)
 * - Exposure Calculation: exposureScore = (likelihoodPct * impactScore) / 100
 * - Portfolio-wide Risk Aggregation
 * - Deterministic SHA-256 State Hash Lock
 */

import type {
  GovernanceRisk,
  RiskCategory,
  RiskSeverity,
  RiskStatus,
} from '../../types/groupthink-intelligence';

import { sha256 } from './sha256';

export const CANONICAL_RISKS: GovernanceRisk[] = [
  {
    riskId: 'RSK-001',
    title: 'Suppressed Dissent in High-Beta Allocations',
    description: 'Investment Committee exhibiting declining dissent submission rates during rapid volatility escalation.',
    category: 'GROUPTHINK',
    severity: 'HIGH',
    likelihoodPct: 85,
    impactScore: 78,
    exposureScore: 66.3,
    status: 'OPEN',
    committeeId: 'COM-001',
    incidentIds: ['INC-201'],
    createdAtUtc: '2026-09-08T10:00:00Z',
    mitigationPlan: "Institute mandatory Devil's Advocate counter-thesis review for all allocations > 15% NAV.",
    ownerId: 'USR-RSK-01',
  },
  {
    riskId: 'RSK-002',
    title: 'Decision Unanimity Drift in Risk Committee',
    description: 'Risk Committee approval unanimity exceeding 95% over trailing 90 days, indicating consensus fatigue.',
    category: 'GROUPTHINK',
    severity: 'CRITICAL',
    likelihoodPct: 90,
    impactScore: 92,
    exposureScore: 82.8,
    status: 'OPEN',
    committeeId: 'COM-002',
    incidentIds: ['INC-202'],
    createdAtUtc: '2026-09-08T10:30:00Z',
    mitigationPlan: 'Rotate external voting members and mandate anonymous initial voting ballots.',
    ownerId: 'USR-GOV-01',
  },
  {
    riskId: 'RSK-003',
    title: 'Cross-Committee Knowledge Transfer Decay',
    description: 'Trailing adoption rate between COM-001 and COM-003 nearing 80.0% floor threshold.',
    category: 'LEARNING',
    severity: 'HIGH',
    likelihoodPct: 75,
    impactScore: 80,
    exposureScore: 60.0,
    status: 'OPEN',
    committeeId: 'COM-001',
    incidentIds: ['INC-201'],
    createdAtUtc: '2026-09-08T11:00:00Z',
    mitigationPlan: 'Weekly inter-committee learning sync and automated adoption reminder triggers.',
    ownerId: 'USR-PM-01',
  },
  {
    riskId: 'RSK-004',
    title: 'Circular Influence Dependency Exposure',
    description: 'Bidirectional influence between Investment and Governance committees creating deadlock potential.',
    category: 'NETWORK',
    severity: 'CRITICAL',
    likelihoodPct: 88,
    impactScore: 90,
    exposureScore: 79.2,
    status: 'OPEN',
    committeeId: 'COM-003',
    incidentIds: ['INC-202'],
    createdAtUtc: '2026-09-08T11:30:00Z',
    mitigationPlan: 'Introduce asynchronous tie-breaking quorum rules with independent external auditor.',
    ownerId: 'USR-GOV-02',
  },
  {
    riskId: 'RSK-005',
    title: 'Replay Variance on Micro-Cap Liquidity Shocks',
    description: 'Non-deterministic float rounding detected on multi-asset Cornish-Fisher VaR replay runs.',
    category: 'REPLAY',
    severity: 'MEDIUM',
    likelihoodPct: 45,
    impactScore: 70,
    exposureScore: 31.5,
    status: 'OPEN',
    committeeId: 'COM-001',
    incidentIds: [],
    createdAtUtc: '2026-09-08T12:00:00Z',
    mitigationPlan: 'Implement arbitrary precision decimal math on volatility surface calculations.',
    ownerId: 'USR-RSK-02',
  },
  {
    riskId: 'RSK-006',
    title: 'Cosmetic Dissent Proliferation Without Adoption',
    description: 'Minor dissenting opinions filed but 0% incorporated into ultimate target execution parameters.',
    category: 'GOVERNANCE',
    severity: 'HIGH',
    likelihoodPct: 80,
    impactScore: 75,
    exposureScore: 60.0,
    status: 'OPEN',
    committeeId: 'COM-002',
    incidentIds: ['INC-201'],
    createdAtUtc: '2026-09-08T12:30:00Z',
    mitigationPlan: 'Mandate explicit outcome attribution reconciliation on all rejected dissents.',
    ownerId: 'USR-CHAIR-02',
  },
  {
    riskId: 'RSK-007',
    title: 'Attribution Lineage Breakage in Secondary Strategies',
    description: 'Secondary strategy adoptions missing source decision IDs in historical audit logs.',
    category: 'ATTRIBUTION',
    severity: 'MEDIUM',
    likelihoodPct: 50,
    impactScore: 65,
    exposureScore: 32.5,
    status: 'OPEN',
    committeeId: 'COM-001',
    incidentIds: [],
    createdAtUtc: '2026-09-08T13:00:00Z',
    mitigationPlan: 'Enforce pre-commit schema validation on all adoption publishing pipelines.',
    ownerId: 'USR-PM-02',
  },
  {
    riskId: 'RSK-008',
    title: 'Artificial Post-Dissent Voting Conformity',
    description: 'Initial committee deliberations exhibit disagreement but final votes converge to 100% uniformity.',
    category: 'GROUPTHINK',
    severity: 'HIGH',
    likelihoodPct: 78,
    impactScore: 82,
    exposureScore: 64.0,
    status: 'OPEN',
    committeeId: 'COM-003',
    incidentIds: ['INC-202'],
    createdAtUtc: '2026-09-08T13:30:00Z',
    mitigationPlan: 'Record initial straw-poll tallies directly into immutable audit ledger.',
    ownerId: 'USR-GOV-01',
  },
  {
    riskId: 'RSK-009',
    title: 'Single Influencer Consensus Dominance',
    description: 'Lead portfolio manager driving over 80% of unanimous committee approvals.',
    category: 'GROUPTHINK',
    severity: 'CRITICAL',
    likelihoodPct: 92,
    impactScore: 88,
    exposureScore: 81.0,
    status: 'OPEN',
    committeeId: 'COM-001',
    incidentIds: ['INC-201'],
    createdAtUtc: '2026-09-08T14:00:00Z',
    mitigationPlan: 'Enforce rotating sponsorship where junior analysts author primary thesis briefs.',
    ownerId: 'USR-RSK-01',
  },
  {
    riskId: 'RSK-010',
    title: 'Unmonitored High-Performance Groupthink Traps',
    description: 'Record high ODEI (92.0) obscuring systemic elimination of dissenting theses.',
    category: 'GROUPTHINK',
    severity: 'HIGH',
    likelihoodPct: 82,
    impactScore: 85,
    exposureScore: 69.7,
    status: 'OPEN',
    committeeId: 'COM-002',
    incidentIds: ['INC-202'],
    createdAtUtc: '2026-09-08T14:30:00Z',
    mitigationPlan: 'Decouple committee evaluation bonus formulas from unanimous consensus rates.',
    ownerId: 'USR-CHAIR-02',
  },
];

// Active in-memory risk registry
const riskRegistry: GovernanceRisk[] = CANONICAL_RISKS.map(r => ({ ...r }));

/**
 * Validates a risk record against institutional rules VR-R01 to VR-R06.
 */
export function validateRiskRecord(risk: GovernanceRisk): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // VR-R01: riskId required matching RSK-xxx
  if (!risk.riskId || !/^RSK-\d{3,}$/.test(risk.riskId)) {
    errors.push('VR-R01: riskId must match RSK-xxx format.');
  }

  // VR-R02: likelihoodPct between 0 and 100
  if (risk.likelihoodPct < 0 || risk.likelihoodPct > 100 || !Number.isFinite(risk.likelihoodPct)) {
    errors.push('VR-R02: likelihoodPct must be a number between 0 and 100.');
  }

  // VR-R03: impactScore between 0 and 100
  if (risk.impactScore < 0 || risk.impactScore > 100 || !Number.isFinite(risk.impactScore)) {
    errors.push('VR-R03: impactScore must be a number between 0 and 100.');
  }

  // VR-R04: exposureScore = (likelihoodPct * impactScore) / 100
  const expectedExposure = Math.round(((risk.likelihoodPct * risk.impactScore) / 100) * 10) / 10;
  if (Math.abs(risk.exposureScore - expectedExposure) > 0.15) {
    errors.push(`VR-R04: exposureScore (${risk.exposureScore}) must equal (likelihood * impact) / 100 (${expectedExposure}).`);
  }

  // VR-R05: valid category
  const validCategories: RiskCategory[] = ['GOVERNANCE', 'LEARNING', 'NETWORK', 'REPLAY', 'GROUPTHINK', 'ATTRIBUTION'];
  if (!validCategories.includes(risk.category)) {
    errors.push(`VR-R05: category must be one of: ${validCategories.join(', ')}.`);
  }

  // VR-R06: Critical risks require incident linkage
  if (risk.severity === 'CRITICAL' && (!risk.incidentIds || risk.incidentIds.length === 0)) {
    errors.push('VR-R06: CRITICAL risks require at least one linked correlated incident ID.');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

export function getRiskRegistry(): GovernanceRisk[] {
  return riskRegistry.map(r => ({ ...r, incidentIds: [...r.incidentIds] }));
}

export function getRiskById(riskId: string): GovernanceRisk | undefined {
  return riskRegistry.find(r => r.riskId === riskId);
}

export function registerRisk(risk: Omit<GovernanceRisk, 'exposureScore'>): GovernanceRisk {
  const exposureScore = Math.round(((risk.likelihoodPct * risk.impactScore) / 100) * 10) / 10;
  const newRisk: GovernanceRisk = {
    ...risk,
    exposureScore,
  };

  const validation = validateRiskRecord(newRisk);
  if (!validation.valid) {
    throw new Error(`Risk registration failed: ${validation.errors.join('; ')}`);
  }

  riskRegistry.push(newRisk);
  return newRisk;
}

export function updateRisk(riskId: string, updates: Partial<GovernanceRisk>): GovernanceRisk {
  const risk = riskRegistry.find(r => r.riskId === riskId);
  if (!risk) {
    throw new Error(`Risk ${riskId} not found in registry.`);
  }

  Object.assign(risk, updates);
  if (updates.likelihoodPct !== undefined || updates.impactScore !== undefined) {
    risk.exposureScore = Math.round(((risk.likelihoodPct * risk.impactScore) / 100) * 10) / 10;
  }
  risk.updatedAtUtc = new Date().toISOString();

  const validation = validateRiskRecord(risk);
  if (!validation.valid) {
    throw new Error(`Risk update failed: ${validation.errors.join('; ')}`);
  }

  return risk;
}

export function closeRisk(riskId: string, mitigationPlan?: string): GovernanceRisk {
  return updateRisk(riskId, {
    status: 'CLOSED',
    mitigationPlan: mitigationPlan ?? 'Risk closed and mitigation verified.',
  });
}

/**
 * Calculates aggregated exposure across open risks.
 */
export function calculatePortfolioExposure(committeeId?: string): {
  totalRisks: number;
  openRisks: number;
  criticalRisks: number;
  averageExposure: number;
  maxExposure: number;
  topRiskCategory: RiskCategory;
} {
  const filtered = committeeId
    ? riskRegistry.filter(r => !r.committeeId || r.committeeId === committeeId)
    : riskRegistry;

  const open = filtered.filter(r => r.status === 'OPEN' || r.status === 'MITIGATING');
  const critical = open.filter(r => r.severity === 'CRITICAL');
  const exposures = open.map(r => r.exposureScore);

  const avg = exposures.length > 0
    ? Math.round((exposures.reduce((a, b) => a + b, 0) / exposures.length) * 10) / 10
    : 0.0;
  const max = exposures.length > 0 ? Math.max(...exposures) : 0.0;

  // Category counts
  const catCounts = new Map<RiskCategory, number>();
  for (const r of open) {
    catCounts.set(r.category, (catCounts.get(r.category) ?? 0) + 1);
  }
  let topCat: RiskCategory = 'GROUPTHINK';
  let topCount = 0;
  for (const [cat, cnt] of catCounts.entries()) {
    if (cnt > topCount) {
      topCount = cnt;
      topCat = cat;
    }
  }

  return {
    totalRisks: filtered.length,
    openRisks: open.length,
    criticalRisks: critical.length,
    averageExposure: avg,
    maxExposure: max,
    topRiskCategory: topCat,
  };
}

/**
 * Deterministic SHA-256 state hash of risk registry.
 */
export function hashRiskRegistry(): string {
  const payload = riskRegistry.map(r => ({
    id: r.riskId,
    sev: r.severity,
    lik: r.likelihoodPct,
    imp: r.impactScore,
    exp: r.exposureScore,
    st: r.status,
    com: r.committeeId,
    incs: [...r.incidentIds].sort(),
  }));
  return sha256(JSON.stringify(payload));
}
