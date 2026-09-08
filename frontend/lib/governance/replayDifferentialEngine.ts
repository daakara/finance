/**
 * Phase 31-M1.1 / M2: Replay Differential Testing Engine (REPLAY-DIFF)
 *
 * Implements differential sensitivity verification:
 * - REPLAY-DIFF-01: ODEI Sensitivity (Input A != Input B -> Output A != Output B)
 * - REPLAY-DIFF-02: Dissent Coverage Impact (100% -> 60% causes score and status degradation)
 * - REPLAY-DIFF-03: Attribution Integrity Impact (100% -> 105% triggers PASS -> FAIL)
 */

import { ReplayDifferentialResult } from '../../types/committee-intelligence';

export interface ODEIComponents {
  dq: number;
  oe: number;
  le: number;
  oh: number;
}

export function computeDifferentialODEI(comp: ODEIComponents): number {
  const score = 0.35 * comp.dq + 0.30 * comp.oe + 0.20 * comp.le + 0.15 * comp.oh;
  return Math.round(score * 10) / 10;
}

export function evaluateODEISensitivity(
  baseline: ODEIComponents,
  modified: ODEIComponents
): ReplayDifferentialResult {
  const baselineScore = computeDifferentialODEI(baseline);
  const modifiedScore = computeDifferentialODEI(modified);
  const delta = Math.round((modifiedScore - baselineScore) * 10) / 10;

  return {
    sensitive: Math.abs(delta) > 0,
    baselineResult: baselineScore,
    modifiedResult: modifiedScore,
    delta,
    description: `ODEI shifted from ${baselineScore} to ${modifiedScore} (delta: ${delta})`,
  };
}

export function evaluateDissentCoverageImpact(
  baselinePct: number,
  modifiedPct: number
): ReplayDifferentialResult {
  const baselinePass = baselinePct === 100.0;
  const modifiedPass = modifiedPct === 100.0;
  const statusChanged = baselinePass !== modifiedPass;

  return {
    sensitive: statusChanged && baselinePct !== modifiedPct,
    baselineResult: { coverage: baselinePct, pass: baselinePass },
    modifiedResult: { coverage: modifiedPct, pass: modifiedPass },
    delta: modifiedPct - baselinePct,
    description: `Dissent coverage shifted from ${baselinePct}% (${baselinePass ? 'PASS' : 'FAIL'}) to ${modifiedPct}% (${modifiedPass ? 'PASS' : 'FAIL'})`,
  };
}

export function evaluateAttributionIntegrityImpact(
  baselineSum: number,
  modifiedSum: number
): ReplayDifferentialResult {
  const baselinePass = baselineSum === 100.0;
  const modifiedPass = modifiedSum === 100.0;

  return {
    sensitive: baselinePass !== modifiedPass,
    baselineResult: { sum: baselineSum, status: baselinePass ? 'PASS' : 'FAIL' },
    modifiedResult: { sum: modifiedSum, status: modifiedPass ? 'PASS' : 'FAIL' },
    delta: modifiedSum - baselineSum,
    description: `Attribution sum changed from ${baselineSum}% to ${modifiedSum}%, triggering status transition`,
  };
}
