/**
 * Phase 31-M1.1 / M2: Fixture Diversity Score (FDS) Engine
 *
 * Prevents test overfitting and shallow happy-path validation:
 * FDS = 0.30(CD) + 0.25(DD) + 0.20(ND) + 0.15(DV) + 0.10(OD)
 *
 * Target: FDS >= 80 (STRONG or EXCELLENT)
 * Mutation Resilience: FDS >= 75
 */

import {
  FixtureDiversityResult,
  FixtureDiversityComponents,
  FixtureDiversityClassification,
} from '../../types/committee-intelligence';

export interface FixtureDiversityInput {
  committees?: Record<string, unknown>[];
  decisions?: Record<string, unknown>[];
  dissents?: Record<string, unknown>[];
  nodes?: Record<string, unknown>[];
  edges?: Record<string, unknown>[];
  outcomes?: Record<string, unknown>[];
}

export function computeFixtureDiversityScore(dataset: FixtureDiversityInput): FixtureDiversityResult {
  const committees = dataset.committees ?? [];
  const decisions = dataset.decisions ?? [];
  const dissents = dataset.dissents ?? [];
  const nodes = dataset.nodes ?? [];
  const edges = dataset.edges ?? [];
  const outcomes = dataset.outcomes ?? [];

  // ── Component A: Committee Diversity (CD, weight 0.30) ──────────────
  let cd = 50.0;
  if (committees.length >= 3) cd += 20.0;
  if (committees.length >= 5) cd += 15.0;
  const uniqueNames = new Set(committees.map(c => c.name || c.committeeName)).size;
  if (uniqueNames >= 3) cd += 15.0;
  const committeeDiversity = Math.min(100.0, Math.max(0.0, cd));

  // ── Component B: Dissent Diversity (DD, weight 0.25) ────────────────
  let dd = 40.0;
  const severities = new Set(dissents.map(d => d.severity));
  if (severities.has('MATERIAL')) dd += 20.0;
  if (severities.has('HIGH')) dd += 15.0;
  if (severities.has('MEDIUM') || severities.has('LOW')) dd += 10.0;
  const authors = new Set(dissents.map(d => d.authorId)).size;
  if (authors >= 2) dd += 15.0;
  const dissentDiversity = Math.min(100.0, Math.max(0.0, dd));

  // ── Component C: Network Diversity (ND, weight 0.20) ────────────────
  let nd = 45.0;
  if (nodes.length >= 3) nd += 20.0;
  if (edges.length >= 3) nd += 20.0;
  const influenceScores = edges.map(e => Number(e.influenceScore) || 0);
  const minInf = Math.min(...influenceScores, 50);
  const maxInf = Math.max(...influenceScores, 50);
  if (maxInf - minInf >= 15.0) nd += 15.0;
  const networkDiversity = Math.min(100.0, Math.max(0.0, nd));

  // ── Component D: Decision Diversity (DV, weight 0.15) ───────────────
  let dv = 50.0;
  const finalDecisions = new Set(decisions.map(d => d.finalDecision || d.status)).size;
  if (finalDecisions >= 3) dv += 30.0;
  const qualityScores = decisions.map(d => Number(d.decisionQuality) || 80);
  const minQ = Math.min(...qualityScores, 80);
  const maxQ = Math.max(...qualityScores, 80);
  if (maxQ - minQ >= 5.0) dv += 20.0;
  const decisionDiversity = Math.min(100.0, Math.max(0.0, dv));

  // ── Component E: Outcome Diversity (OD, weight 0.10) ────────────────
  let od = 50.0;
  if (outcomes.length >= 3) od += 25.0;
  const dollarValues = outcomes.map(o => Number(o.realizedValueDollars) || 0);
  if (new Set(dollarValues).size >= 3) od += 25.0;
  const outcomeDiversity = Math.min(100.0, Math.max(0.0, od));

  // ── Weighted FDS ───────────────────────────────────────────────────
  const rawFds =
    0.30 * committeeDiversity +
    0.25 * dissentDiversity +
    0.20 * networkDiversity +
    0.15 * decisionDiversity +
    0.10 * outcomeDiversity;

  const fds = Math.round(rawFds * 10) / 10;

  let classification: FixtureDiversityClassification = 'OVERFIT_RISK';
  if (fds >= 90.0) {
    classification = 'EXCELLENT';
  } else if (fds >= 80.0) {
    classification = 'STRONG';
  } else if (fds >= 70.0) {
    classification = 'ADEQUATE';
  } else if (fds >= 60.0) {
    classification = 'WEAK';
  }

  return {
    fds,
    classification,
    components: {
      committeeDiversity,
      dissentDiversity,
      networkDiversity,
      decisionDiversity,
      outcomeDiversity,
    },
  };
}
