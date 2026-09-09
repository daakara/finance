/**
 * Personal Drift Engine & Visualization Models (Horizon 5)
 *
 * Implements:
 * - Personal Metric Drift Tracking (Career, Finance, Health, Learning)
 * - Progress Rail, Drift Gauge, and Drift Waterfall Generators
 * - Non-Moralizing Adaptive Drift Recalibration (INV-OI80-P)
 */

import { PersonalDriftCard } from "../../types/personal-digital-twin";

export const CANONICAL_PERSONAL_DRIFT_CARDS: PersonalDriftCard[] = [
  {
    domain: "CAREER",
    metric: "System Architecture & Skill Score",
    expected: 78,
    actual: 72,
    driftPct: -7.7,
    severity: "HIGH",
    unit: "pts",
    recommendation: "Reallocate 2h/week from admin to focused system design mock reviews.",
    waterfallCauses: [
      { cause: "Meeting Fragmentation & Context Switching", impact: -3.2 },
      { cause: "Deferred Deep Work Blocks", impact: -2.8 },
      { cause: "Evening Cognitive Fatigue", impact: -1.7 },
    ],
  },
  {
    domain: "FINANCE",
    metric: "Cumulative Annual Savings",
    expected: 14000,
    actual: 11800,
    driftPct: -15.7,
    severity: "HIGH",
    unit: "€",
    recommendation: "Audit automated discretionary subscriptions and adjust travel budget by €180/mo.",
    waterfallCauses: [
      { cause: "Macro Cost-of-Living & Food Inflation", impact: -6.0 },
      { cause: "Unplanned Dental / Medical Out-of-Pocket", impact: -4.2 },
      { cause: "Summer Travel Flight Premiums", impact: -3.1 },
      { cause: "Recurring SaaS Subscriptions", impact: -2.4 },
    ],
  },
  {
    domain: "HEALTH",
    metric: "Cardiovascular & Aerobic Fitness Score",
    expected: 82,
    actual: 79,
    driftPct: -3.7,
    severity: "MEDIUM",
    unit: "pts",
    recommendation: "Substitute one high-intensity workout with a 35m restorative zone-2 walk.",
    waterfallCauses: [
      { cause: "Consecutive Late Work Deliverables", impact: -2.1 },
      { cause: "Reduced REM / Deep Sleep Quality", impact: -1.6 },
    ],
  },
  {
    domain: "LEARNING",
    metric: "Technical Mastery Curriculum Modules",
    expected: 12,
    actual: 8,
    driftPct: -33.3,
    severity: "HIGH",
    unit: "modules",
    recommendation: "Convert commute and travel into audio paper reviews or enroll in a 3-week sprint.",
    waterfallCauses: [
      { cause: "Late Afternoon Meeting Spillover", impact: -15.0 },
      { cause: "Commute Exhaustion", impact: -10.0 },
      { cause: "Lack of Dedicated Sunday Planning", impact: -8.3 },
    ],
  },
];

/**
 * Calculates drift percentage and assigns severity.
 */
export function calculateDrift(
  expected: number,
  actual: number
): { driftPct: number; severity: "LOW" | "MEDIUM" | "HIGH" } {
  if (expected === 0) return { driftPct: 0, severity: "LOW" };
  const driftPct = Number((((actual - expected) / expected) * 100).toFixed(1));
  const absDrift = Math.abs(driftPct);

  let severity: "LOW" | "MEDIUM" | "HIGH" = "LOW";
  if (absDrift > 7.0) {
    severity = "HIGH";
  } else if (absDrift >= 3.0) {
    severity = "MEDIUM";
  }

  return { driftPct, severity };
}

/**
 * Maps drift percentage to gauge visualization properties.
 */
export function getDriftGaugeProperties(driftPct: number): {
  color: string;
  badgeClass: string;
  statusLabel: string;
} {
  const abs = Math.abs(driftPct);
  if (abs < 3.0) {
    return {
      color: "#10b981", // Emerald Green
      badgeClass: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
      statusLabel: "NOMINAL",
    };
  }
  if (abs <= 7.0) {
    return {
      color: "#f59e0b", // Amber
      badgeClass: "bg-amber-500/10 text-amber-400 border-amber-500/20",
      statusLabel: "MODERATE DRIFT",
    };
  }
  return {
    color: "#ef4444", // Rose/Red
    badgeClass: "bg-rose-500/10 text-rose-400 border-rose-500/20",
    statusLabel: "ELEVATED DRIFT",
  };
}

/**
 * Calculates normalized Progress Rail offsets (0% to 100%).
 */
export function calculateProgressRail(
  expected: number,
  actual: number,
  maxScale: number = 100
): { expectedOffsetPct: number; actualOffsetPct: number } {
  const normExpected = Math.max(0, Math.min(100, (expected / maxScale) * 100));
  const normActual = Math.max(0, Math.min(100, (actual / maxScale) * 100));
  return {
    expectedOffsetPct: Number(normExpected.toFixed(1)),
    actualOffsetPct: Number(normActual.toFixed(1)),
  };
}
