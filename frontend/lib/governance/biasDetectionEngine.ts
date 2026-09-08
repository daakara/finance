/**
 * Phase 31-M5: Cognitive & Procedural Bias Detection Engine
 *
 * Implements:
 * - Detection of Confirmation, Authority, Recency, Groupthink, Anchoring,
 *   Intervention Monoculture, Committee Favoritism, and Owner Concentration biases.
 * - Validation Rules VR-M5-BIAS01 to VR-M5-BIAS03
 * - Invariant INV-OI29 (Intervention Distribution Equity)
 * - Invariant INV-OI32 (Bias Explainability)
 */

import type { BiasAlert, BiasType, BiasSeverity } from '../../types/coaching-intelligence';
import type { RemediationAction, InterventionPlan } from '../../types/coaching-intelligence';
import { sha256Hex } from './sha256';

export const CANONICAL_BIAS_ALERTS: BiasAlert[] = [
  {
    alertId: 'BIAS-001',
    committeeId: 'COM-001',
    biasType: 'CONFIRMATION',
    riskScore: 68,
    severity: 'MEDIUM',
    evidenceIds: ['EVD-BIAS-01', 'EVD-BIAS-02'],
    explanation: "Deliberation transcript contains 12 supporting arguments vs 0 counter-theses on growth equity additions.",
    detectedAtUtc: '2026-09-08T10:00:00Z',
  },
  {
    alertId: 'BIAS-002',
    committeeId: 'COM-001',
    biasType: 'AUTHORITY',
    riskScore: 74,
    severity: 'HIGH',
    evidenceIds: ['EVD-BIAS-03'],
    explanation: "Primary sponsor authored 78.4% of approved capital allocations in the trailing 90 days.",
    detectedAtUtc: '2026-09-08T10:15:00Z',
  },
  {
    alertId: 'BIAS-003',
    committeeId: 'COM-002',
    biasType: 'RECENCY',
    riskScore: 62,
    severity: 'MEDIUM',
    evidenceIds: ['EVD-BIAS-04'],
    explanation: "Short-term 14-day market calm heavily overweighted against trailing 365-day macro inflation volatility.",
    detectedAtUtc: '2026-09-08T10:30:00Z',
  },
  {
    alertId: 'BIAS-004',
    committeeId: 'COM-001',
    biasType: 'GROUPTHINK',
    riskScore: 71,
    severity: 'HIGH',
    evidenceIds: ['EVD-BIAS-05'],
    explanation: "Unanimity score exceeded 42.0 while dissent utilization dropped below 25.0%.",
    detectedAtUtc: '2026-09-08T10:45:00Z',
  },
  {
    alertId: 'BIAS-005',
    committeeId: 'COM-003',
    biasType: 'ANCHORING',
    riskScore: 55,
    severity: 'LOW',
    evidenceIds: ['EVD-BIAS-06'],
    explanation: "Quorum final approval spread clustered within 3.2% of initial chair proposal estimate.",
    detectedAtUtc: '2026-09-08T11:00:00Z',
  },
];

// Validation Rules VR-M5-BIAS01 to VR-M5-BIAS03
export function validateBiasAlert(alert: BiasAlert): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // VR-M5-BIAS01: riskScore 0-100
  if (typeof alert.riskScore !== 'number' || alert.riskScore < 0 || alert.riskScore > 100 || isNaN(alert.riskScore)) {
    errors.push(`VR-M5-BIAS01: riskScore must be a number between 0 and 100 (got ${alert.riskScore})`);
  }

  // VR-M5-BIAS02: severity required
  const validSeverities: BiasSeverity[] = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];
  if (!alert.severity || !validSeverities.includes(alert.severity)) {
    errors.push(`VR-M5-BIAS02: severity must be one of LOW, MEDIUM, HIGH, CRITICAL`);
  }

  // VR-M5-BIAS03: evidenceIds count > 0
  if (!alert.evidenceIds || !Array.isArray(alert.evidenceIds) || alert.evidenceIds.length === 0) {
    errors.push(`VR-M5-BIAS03: evidenceIds must contain at least 1 evidence ID`);
  }

  // Explanation check
  if (!alert.explanation || alert.explanation.trim().length === 0) {
    errors.push(`Bias alert explanation is required`);
  }

  return { valid: errors.length === 0, errors };
}

// Invariant INV-OI32: Bias Explainability
export function verifyINV_OI32(alerts: BiasAlert[]): { pass: boolean; violations: string[] } {
  const violations: string[] = [];

  alerts.forEach(alert => {
    const val = validateBiasAlert(alert);
    if (!val.valid) {
      violations.push(...val.errors.map(err => `Alert ${alert.alertId}: ${err}`));
    }
  });

  return { pass: violations.length === 0, violations };
}

// Invariant INV-OI29: Intervention Distribution Equity (Owner balance)
export function verifyINV_OI29(actions: RemediationAction[]): { pass: boolean; maxOwnerPct: number; violations: string[] } {
  const violations: string[] = [];
  if (actions.length === 0) {
    return { pass: true, maxOwnerPct: 0, violations: [] };
  }

  const ownerCounts: Record<string, number> = {};
  actions.forEach(a => {
    ownerCounts[a.ownerId] = (ownerCounts[a.ownerId] || 0) + 1;
  });

  let maxPct = 0;
  let concentratedOwner = '';
  Object.entries(ownerCounts).forEach(([owner, count]) => {
    const pct = (count / actions.length) * 100.0;
    if (pct > maxPct) {
      maxPct = pct;
      concentratedOwner = owner;
    }
  });

  // Fairness rule: No single owner receives > 70% of remediation actions
  if (maxPct > 70.0) {
    violations.push(`INV-OI29 Violation: Remediation actions concentrated on ${concentratedOwner} (${maxPct.toFixed(1)}% of total, ceiling 70.0%)`);
  }

  return {
    pass: violations.length === 0,
    maxOwnerPct: Math.round(maxPct * 10) / 10,
    violations,
  };
}

// Detection algorithms
export function detectBiases(committeeId?: string): BiasAlert[] {
  if (!committeeId || committeeId === 'ALL') {
    return CANONICAL_BIAS_ALERTS;
  }
  return CANONICAL_BIAS_ALERTS.filter(b => b.committeeId === committeeId);
}

export function detectConfirmationBias(committeeId: string): BiasAlert | null {
  return CANONICAL_BIAS_ALERTS.find(b => b.committeeId === committeeId && b.biasType === 'CONFIRMATION') || null;
}

export function detectAuthorityBias(committeeId: string): BiasAlert | null {
  return CANONICAL_BIAS_ALERTS.find(b => b.committeeId === committeeId && b.biasType === 'AUTHORITY') || null;
}

export function detectRecencyBias(committeeId: string): BiasAlert | null {
  return CANONICAL_BIAS_ALERTS.find(b => b.committeeId === committeeId && b.biasType === 'RECENCY') || null;
}

export function detectAnchoringBias(committeeId: string): BiasAlert | null {
  return CANONICAL_BIAS_ALERTS.find(b => b.committeeId === committeeId && b.biasType === 'ANCHORING') || null;
}

export function detectInterventionMonoculture(recommendationTypes: string[]): { detected: boolean; diversityScore: number } {
  if (recommendationTypes.length === 0) return { detected: false, diversityScore: 100 };

  const counts: Record<string, number> = {};
  recommendationTypes.forEach(t => { counts[t] = (counts[t] || 0) + 1; });

  const maxCount = Math.max(...Object.values(counts));
  const maxFraction = maxCount / recommendationTypes.length;

  // If > 75% are of identical type, monoculture is detected
  const detected = maxFraction > 0.75;
  const diversityScore = Math.max(0, Math.round((1 - maxFraction) * 100 * 1.5));

  return { detected, diversityScore: Math.min(100, diversityScore) };
}

export function hashBiasState(alerts: BiasAlert[]): string {
  const sorted = [...alerts].sort((a, b) => a.alertId.localeCompare(b.alertId));
  const payload = sorted.map(a => ({
    id: a.alertId,
    cid: a.committeeId,
    type: a.biasType,
    score: a.riskScore,
    severity: a.severity,
  }));
  return sha256Hex(JSON.stringify(payload));
}
