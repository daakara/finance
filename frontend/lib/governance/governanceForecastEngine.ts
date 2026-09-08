/**
 * Phase 31-M4: Governance Forecast Engine (Epic M4-103 / INV-OI22)
 *
 * Implements:
 * - Multi-Horizon Predictive Projections (30D, 90D, 180D, 365D):
 *   - Projected ODEI
 *   - Projected DIRatio
 *   - Projected Knowledge Transfer Rate
 *   - Projected CDQI
 *   - Certification Probability
 *   - Projected Institutional Risk Score
 * - Invariant INV-OI22: Predictive Governance Explainability (Drivers sum to 100.0%)
 * - Incident Escalation & Repeat Forecasting
 * - Deterministic SHA-256 Hash Lock
 */

import type {
  GovernanceForecast,
  ForecastDriver,
  ForecastPeriod,
  IncidentForecast,
} from '../../types/groupthink-intelligence';

import { sha256 } from './sha256';

export const COMMITTEE_BASELINES: Record<string, { odei: number; diratio: number; transferRate: number; cdqi: number }> = {
  'COM-001': { odei: 85.0, diratio: 24.2, transferRate: 84.0, cdqi: 86.5 },
  'COM-002': { odei: 83.0, diratio: 22.0, transferRate: 80.0, cdqi: 84.0 },
  'COM-003': { odei: 87.0, diratio: 26.5, transferRate: 88.0, cdqi: 89.0 },
};

/**
 * Computes multi-horizon predictive governance forecast.
 */
export function computeGovernanceForecast(
  committeeId: string = 'COM-001',
  period: ForecastPeriod = '90D'
): GovernanceForecast {
  const base = COMMITTEE_BASELINES[committeeId] ?? COMMITTEE_BASELINES['COM-001'];

  const horizonMultiplier = period === '30D' ? 0.33 : period === '90D' ? 1.0 : period === '180D' ? 2.0 : 4.0;

  // Project metrics with dampening
  const projectedODEI = Math.round((base.odei + (2.4 * horizonMultiplier * 0.75)) * 10) / 10;
  const projectedDIRatio = Math.round((base.diratio + (0.8 * horizonMultiplier * 0.6)) * 10) / 10;
  const projectedTransferRate = Math.round(Math.min(98.0, base.transferRate + (1.2 * horizonMultiplier * 0.5)) * 10) / 10;
  const projectedCDQI = Math.round((base.cdqi + (1.5 * horizonMultiplier * 0.7)) * 10) / 10;

  // Projected Risk Score (lower is better)
  const projectedRiskScore = Math.max(12.0, Math.round((34.0 - (2.1 * horizonMultiplier)) * 10) / 10);
  const projectedCertificationProbability = Math.min(99.0, Math.round((94.5 + (0.8 * horizonMultiplier)) * 10) / 10);
  const confidencePct = Math.round((92.0 - (horizonMultiplier * 2.5)) * 10) / 10;

  // Drivers must sum strictly to 100.0% (INV-OI22)
  const drivers: ForecastDriver[] = [
    {
      driverId: `DRV-${committeeId}-01`,
      driverName: 'Positive Learning Velocity Momentum',
      currentValue: base.odei,
      projectedValue: projectedODEI,
      contributionPct: 40.0,
      trend: 'UP',
    },
    {
      driverId: `DRV-${committeeId}-02`,
      driverName: 'Cross-Committee Knowledge Propagation',
      currentValue: base.transferRate,
      projectedValue: projectedTransferRate,
      contributionPct: 25.0,
      trend: 'UP',
    },
    {
      driverId: `DRV-${committeeId}-03`,
      driverName: 'Dissent Shield & Counter-Thesis Utilization',
      currentValue: base.diratio,
      projectedValue: projectedDIRatio,
      contributionPct: 20.0,
      trend: 'UP',
    },
    {
      driverId: `DRV-${committeeId}-04`,
      driverName: 'Byzantine Replay Stability & Verification',
      currentValue: 98.5,
      projectedValue: 99.2,
      contributionPct: 15.0,
      trend: 'STABLE',
    },
  ];

  return {
    forecastId: `FCST-${committeeId}-${period}`,
    committeeId,
    forecastDateUtc: '2026-09-08T15:00:00Z',
    forecastPeriod: period,
    projectedODEI,
    projectedDIRatio,
    projectedTransferRate,
    projectedCDQI,
    projectedCertificationProbability,
    projectedRiskScore,
    confidencePct,
    status: projectedRiskScore < 25.0 ? 'LOW' : projectedRiskScore < 50.0 ? 'MEDIUM' : 'HIGH',
    drivers,
  };
}

/**
 * Validates Invariant INV-OI22 (Predictive Governance Explainability).
 * Asserts that the sum of forecast driver contribution percentages is strictly 100.0%.
 */
export function verifyINV_OI22(forecast: GovernanceForecast): {
  valid: boolean;
  totalContributionPct: number;
  driverCount: number;
  message: string;
} {
  const total = Math.round(forecast.drivers.reduce((acc, d) => acc + d.contributionPct, 0) * 10) / 10;
  const valid = Math.abs(total - 100.0) <= 0.1;

  return {
    valid,
    totalContributionPct: total,
    driverCount: forecast.drivers.length,
    message: valid
      ? `INV-OI22 PASSED: 100.0% driver attribution confirmed across ${forecast.drivers.length} leading indicators.`
      : `INV-OI22 VIOLATION: Driver contributions sum to ${total}% (100.0% required).`,
  };
}

/**
 * Predicts escalation and recurrence probabilities for correlated incidents.
 */
export function predictIncidentEscalation(incidentId: string = 'INC-201'): IncidentForecast {
  // Mock deterministic escalation predictions based on incident ID
  if (incidentId.includes('201') || incidentId.includes('CORR-01')) {
    return {
      incidentId,
      currentSeverity: 'HIGH',
      escalationProbability: 68.5,
      recurrenceProbability: 42.0,
      forecastDays: 14,
      likelyRootCauses: [
        'Trailing committee knowledge adoption stagnation',
        'Unaddressed ownership gap in allocation reviews',
      ],
      recommendedActions: [
        'Mandate inter-committee alignment review within 48 hours',
        'Assign designated ownership to pending friction items',
      ],
    };
  }

  return {
    incidentId,
    currentSeverity: 'HIGH',
    escalationProbability: 35.0,
    recurrenceProbability: 25.0,
    forecastDays: 30,
    likelyRootCauses: [
      'Periodic consensus feedback loops',
      'Informal voting pre-commitments',
    ],
    recommendedActions: [
      'Audit voting deliberation transcripts',
      'Enable blind voting ballots on controversial items',
    ],
  };
}

/**
 * Cryptographic SHA-256 hash lock of forecast.
 */
export function hashGovernanceForecast(forecast: GovernanceForecast): string {
  const payload = {
    id: forecast.forecastId,
    com: forecast.committeeId,
    per: forecast.forecastPeriod,
    odei: forecast.projectedODEI,
    risk: forecast.projectedRiskScore,
    conf: forecast.confidencePct,
    drivers: forecast.drivers.map(d => ({ id: d.driverId, val: d.projectedValue, pct: d.contributionPct })),
  };
  return sha256(JSON.stringify(payload));
}
