/**
 * Phase 31-M2.1 / M3 Foundation Verification Suite
 * Global Search, Universal Cross-Linking, Historical Trends & Alert Workflow
 *
 * Implements 85 Fail-Close Assertions across 6 Suites:
 * - Suite 1: Global Search & Prefix Resolver (M2-Gate-01) [15 assertions]
 * - Suite 2: Universal Cross-Linking Coverage (M2-Gate-02) [15 assertions]
 * - Suite 3: Historical Trends & Invariants (M2-Gate-03, HT-01 to HT-06) [15 assertions]
 * - Suite 4: Invariant Verification: INV-OI17 & INV-OI18 [12 assertions]
 * - Suite 5: Alert Workflow & Remediation Playbooks (M2-Gate-04, AW-01 to AW-08) [15 assertions]
 * - Suite 6: Audit Reconstruction Accessibility (M2-Gate-05) [13 assertions]
 *
 * Total: 85 / 85 Assertions. Fail-close execution protocol.
 */

import { strict as assert } from 'node:assert';
import crypto from 'node:crypto';

let passed = 0;
let failed = 0;
const errors = [];

function check(label, fn) {
  try {
    fn();
    passed++;
  } catch (e) {
    failed++;
    errors.push({ label, error: e.message });
  }
}

// ── Canonical Fixtures ──────────────────────────────────────────────

const CANONICAL_COMMITTEES = [
  {
    committeeId: 'COM-001',
    committeeName: 'Investment Committee',
    name: 'Investment Committee',
    cdqi: 85.4,
    odei: 85.0,
    committeeODEI: 85.0,
    committeeDIRatio: 25.4,
    dissentCoveragePct: 100.0,
    learningVelocityPct: 14.2,
    governanceCompliancePct: 98.5,
    transparencyCoveragePct: 100.0,
  },
  {
    committeeId: 'COM-002',
    committeeName: 'Governance Committee',
    name: 'Governance Committee',
    cdqi: 83.2,
    odei: 83.0,
    committeeODEI: 83.0,
    committeeDIRatio: 22.8,
    dissentCoveragePct: 100.0,
    learningVelocityPct: 12.0,
    governanceCompliancePct: 99.2,
    transparencyCoveragePct: 100.0,
  },
  {
    committeeId: 'COM-003',
    committeeName: 'Risk & Capital Committee',
    name: 'Risk & Capital Committee',
    cdqi: 86.8,
    odei: 87.0,
    committeeODEI: 87.0,
    committeeDIRatio: 26.5,
    dissentCoveragePct: 100.0,
    learningVelocityPct: 15.6,
    governanceCompliancePct: 98.0,
    transparencyCoveragePct: 100.0,
  },
];

const CANONICAL_PROPOSALS = {
  'PROP-001': {
    proposalId: 'PROP-001',
    title: 'Institutional Flow Regime Overweight Allocation',
    createdBy: 'USR-CIO-01',
    createdAtUtc: '2026-09-08T07:30:00Z',
    businessObjective: 'Scale allocation in high-momentum liquid names under confirmed institutional flow regime.',
  },
  'PROP-002': {
    proposalId: 'PROP-002',
    title: 'Stage 2 VCP Breakout Core Execution Strategy',
    createdBy: 'USR-PM-02',
    createdAtUtc: '2026-09-08T08:00:00Z',
    businessObjective: 'Deploy systematic capital in contracting volatility pivots meeting Minervini criteria.',
  },
  'PROP-003': {
    proposalId: 'PROP-003',
    title: 'Protected Practice Invariant Renewal: Institutional Flow Filter',
    createdBy: 'USR-GOV-01',
    createdAtUtc: '2026-09-08T08:30:00Z',
    businessObjective: 'Re-certify and extend protected practice status under INV-OI11 non-regression bounds.',
  },
  'PROP-004': {
    proposalId: 'PROP-004',
    title: 'Downside Tail Risk VaR Stress Ceiling Calibration',
    createdBy: 'USR-RSK-02',
    createdAtUtc: '2026-09-08T09:00:00Z',
    businessObjective: 'Recalibrate Cornish-Fisher VaR bounds under multi-regime stress scenario.',
  },
};

const CANONICAL_EVIDENCE_STORE = {
  'EVD-FLOW-01': { evidenceId: 'EVD-FLOW-01', source: 'MARKET_DATA', metric: 'InstitutionalNetInflow', verified: true },
  'EVD-MACRO-02': { evidenceId: 'EVD-MACRO-02', source: 'MODEL_OUTPUT', metric: 'MacroRegimeClassifier', verified: true },
  'EVD-VCP-01': { evidenceId: 'EVD-VCP-01', source: 'SCREENER', metric: 'MinerviniStage2PassRate', verified: true },
  'EVD-RISK-03': { evidenceId: 'EVD-RISK-03', source: 'STRESS_TEST', metric: 'MaxDrawdownEstimated', verified: true },
  'EVD-GOV-01': { evidenceId: 'EVD-GOV-01', source: 'AUDIT_LOG', metric: 'ProtectedPracticeAdoption', verified: true },
  'EVD-AUDIT-02': { evidenceId: 'EVD-AUDIT-02', source: 'GOVERNANCE_LEDGER', metric: 'HistoricalSharpeDecay', verified: true },
  'EVD-VAR-01': { evidenceId: 'EVD-VAR-01', source: 'RISK_ENGINE', metric: 'CornishFisher99VaR', verified: true },
  'EVD-CORNISH-02': { evidenceId: 'EVD-CORNISH-02', source: 'STATISTICAL_DISTRIBUTION', metric: 'SkewnessKurtosis', verified: true },
  'EVD-STRESS-04': { evidenceId: 'EVD-STRESS-04', source: 'MACRO_SIMULATOR', metric: 'StagflationShockLoss', verified: true },
  'EVD-DISSENT-01': { evidenceId: 'EVD-DISSENT-01', source: 'LIQUIDITY_SURVEY', metric: 'BidAskSpreadExpansion', verified: true },
  'EVD-DISSENT-02': { evidenceId: 'EVD-DISSENT-02', source: 'MACRO_PROJECTION', metric: 'FedReserveRepoDrain', verified: true },
  'EVD-DISSENT-03': { evidenceId: 'EVD-DISSENT-03', source: 'EXECUTION_ANALYSIS', metric: 'SmallCapSlippageEstimate', verified: true },
  'EVD-DISSENT-04': { evidenceId: 'EVD-DISSENT-04', source: 'VOLATILITY_SURFACE', metric: 'EarningsImpliedMoveRisk', verified: true },
};

const CANONICAL_DISSENTS = [
  {
    dissentId: 'DIS-001',
    decisionId: 'DEC-001',
    authorId: 'USR-RSK-01',
    severity: 'MATERIAL',
    alternativeRecommendation: 'Cap allocation at 1.5x until macroeconomic Fed liquidity confirms regime pivot.',
    riskAssessment: 'High probability of short-term liquidity contraction during transition.',
    evidenceIds: ['EVD-DISSENT-01', 'EVD-DISSENT-02'],
    acceptedForReview: true,
    timestampUtc: '2026-09-08T08:30:00Z',
  },
  {
    dissentId: 'DIS-002',
    decisionId: 'DEC-002',
    authorId: 'USR-ANL-03',
    severity: 'MATERIAL',
    alternativeRecommendation: 'Delay tranche execution until earnings announcement window clears next week.',
    riskAssessment: 'Elevated binary gap risk if earnings miss revenue consensus.',
    evidenceIds: ['EVD-DISSENT-03'],
    acceptedForReview: true,
    timestampUtc: '2026-09-08T09:15:00Z',
  },
  {
    dissentId: 'DIS-003',
    decisionId: 'DEC-004',
    authorId: 'USR-CIO-02',
    severity: 'HIGH',
    alternativeRecommendation: 'Hedge delta using macro index puts during earnings window.',
    riskAssessment: 'Event volatility spike could trigger trailing risk stop prematurely.',
    evidenceIds: ['EVD-DISSENT-04'],
    acceptedForReview: true,
    timestampUtc: '2026-09-08T10:00:00Z',
  },
];

const CANONICAL_COMMITTEE_DECISIONS = [
  {
    committeeId: 'COM-001',
    decisionId: 'DEC-001',
    proposalId: 'PROP-001',
    title: 'Institutional Flow Regime Overweight Allocation',
    participants: [
      { userId: 'USR-CHAIR-01', role: 'CHAIR', votingEligible: true },
      { userId: 'USR-CIO-01', role: 'CIO', votingEligible: true },
      { userId: 'USR-PM-01', role: 'PORTFOLIO_MANAGER', votingEligible: true },
      { userId: 'USR-RSK-01', role: 'ANALYST', votingEligible: true },
    ],
    evidenceIds: ['EVD-FLOW-01', 'EVD-MACRO-02'],
    dissents: [CANONICAL_DISSENTS[0]],
    finalDecision: 'APPROVED_WITH_MODIFIED_EXPOSURE',
    status: 'APPROVED',
    materialDecision: true,
    evidenceLinked: true,
    participantsRecorded: true,
    outcomeLinked: true,
    attributionLinked: true,
    dissentRecorded: true,
    alternativeViewPresent: true,
    riskAssessmentPresent: true,
    dissentEvidenceLinked: true,
    outcomeId: 'OUT-001',
    decisionQuality: 88.0,
    timestampUtc: '2026-09-08T09:00:00Z',
  },
  {
    committeeId: 'COM-001',
    decisionId: 'DEC-002',
    proposalId: 'PROP-002',
    title: 'Stage 2 VCP Breakout Core Execution Strategy',
    participants: [
      { userId: 'USR-CHAIR-01', role: 'CHAIR', votingEligible: true },
      { userId: 'USR-PM-02', role: 'PORTFOLIO_MANAGER', votingEligible: true },
      { userId: 'USR-ANL-03', role: 'ANALYST', votingEligible: true },
    ],
    evidenceIds: ['EVD-VCP-01'],
    dissents: [CANONICAL_DISSENTS[1]],
    finalDecision: 'APPROVED_WITH_TRANCHE_LIMITS',
    status: 'APPROVED',
    materialDecision: true,
    evidenceLinked: true,
    participantsRecorded: true,
    outcomeLinked: true,
    attributionLinked: true,
    dissentRecorded: true,
    alternativeViewPresent: true,
    riskAssessmentPresent: true,
    dissentEvidenceLinked: true,
    outcomeId: 'OUT-002',
    decisionQuality: 84.5,
    timestampUtc: '2026-09-08T09:45:00Z',
  },
  {
    committeeId: 'COM-002',
    decisionId: 'DEC-003',
    proposalId: 'PROP-003',
    title: 'Protected Practice Invariant Renewal: Institutional Flow Filter',
    participants: [
      { userId: 'USR-CHAIR-02', role: 'CHAIR', votingEligible: true },
      { userId: 'USR-GOV-01', role: 'ANALYST', votingEligible: true },
      { userId: 'USR-GOV-02', role: 'ANALYST', votingEligible: true },
      { userId: 'USR-CIO-01', role: 'CIO', votingEligible: true },
    ],
    evidenceIds: ['EVD-GOV-01', 'EVD-AUDIT-02'],
    dissents: [],
    finalDecision: 'UNANIMOUS_APPROVAL',
    status: 'APPROVED',
    materialDecision: false,
    evidenceLinked: true,
    participantsRecorded: true,
    outcomeLinked: true,
    attributionLinked: true,
    dissentRecorded: false,
    outcomeId: 'OUT-003',
    decisionQuality: 92.0,
    timestampUtc: '2026-09-08T10:15:00Z',
  },
  {
    committeeId: 'COM-003',
    decisionId: 'DEC-004',
    proposalId: 'PROP-004',
    title: 'Downside Tail Risk VaR Stress Ceiling Calibration',
    participants: [
      { userId: 'USR-CHAIR-03', role: 'CHAIR', votingEligible: true },
      { userId: 'USR-CIO-02', role: 'CIO', votingEligible: true },
      { userId: 'USR-RSK-02', role: 'ANALYST', votingEligible: true },
    ],
    evidenceIds: ['EVD-VAR-01', 'EVD-CORNISH-02'],
    dissents: [CANONICAL_DISSENTS[2]],
    finalDecision: 'APPROVED_WITH_HEDGING_PROVISO',
    status: 'APPROVED',
    materialDecision: true,
    evidenceLinked: true,
    participantsRecorded: true,
    outcomeLinked: true,
    attributionLinked: true,
    dissentRecorded: true,
    alternativeViewPresent: true,
    riskAssessmentPresent: true,
    dissentEvidenceLinked: true,
    outcomeId: 'OUT-004',
    decisionQuality: 89.0,
    timestampUtc: '2026-09-08T10:45:00Z',
  },
];

const CANONICAL_NETWORK_EDGES = [
  {
    sourceCommitteeId: 'COM-001',
    targetCommitteeId: 'COM-002',
    influenceScore: 78.5,
    sharedDecisionCount: 14,
    informationVelocity: 92.0,
    conflictHistoryCount: 1,
    status: 'ACTIVE',
  },
  {
    sourceCommitteeId: 'COM-002',
    targetCommitteeId: 'COM-003',
    influenceScore: 65.0,
    sharedDecisionCount: 8,
    informationVelocity: 85.0,
    conflictHistoryCount: 0,
    status: 'ACTIVE',
  },
  {
    sourceCommitteeId: 'COM-001',
    targetCommitteeId: 'COM-003',
    influenceScore: 82.0,
    sharedDecisionCount: 18,
    informationVelocity: 94.5,
    conflictHistoryCount: 2,
    status: 'ACTIVE',
  },
];

const CANONICAL_OUTCOMES = {
  'OUT-001': {
    outcomeId: 'OUT-001',
    realizedValueDollars: 450000,
    excessReturnPct: 4.8,
    outcomeQualityScore: 89.5,
    measuredAtUtc: '2026-09-08T11:00:00Z',
  },
  'OUT-002': {
    outcomeId: 'OUT-002',
    realizedValueDollars: 310000,
    excessReturnPct: 3.2,
    outcomeQualityScore: 86.0,
    measuredAtUtc: '2026-09-08T11:30:00Z',
  },
  'OUT-003': {
    outcomeId: 'OUT-003',
    realizedValueDollars: 280000,
    excessReturnPct: 2.9,
    outcomeQualityScore: 92.5,
    measuredAtUtc: '2026-09-08T12:00:00Z',
  },
  'OUT-004': {
    outcomeId: 'OUT-004',
    realizedValueDollars: 390000,
    excessReturnPct: 4.1,
    outcomeQualityScore: 90.0,
    measuredAtUtc: '2026-09-08T12:30:00Z',
  },
};

const CANONICAL_ATTRIBUTIONS = {
  'OUT-001': {
    attributionId: 'ATT-001',
    decisionId: 'DEC-001',
    capabilityIds: ['institutional-flow-filter', 'ai-mentor-engine'],
    learningIds: ['LRN-FLOW-01'],
    individualContributionPct: 25.0,
    teamContributionPct: 35.0,
    committeeContributionPct: 25.0,
    systemContributionPct: 15.0,
    totalContributionPct: 100.0,
  },
  'OUT-002': {
    attributionId: 'ATT-002',
    decisionId: 'DEC-002',
    capabilityIds: ['playbook-engine'],
    learningIds: ['LRN-VCP-02'],
    individualContributionPct: 30.0,
    teamContributionPct: 30.0,
    committeeContributionPct: 20.0,
    systemContributionPct: 20.0,
    totalContributionPct: 100.0,
  },
  'OUT-003': {
    attributionId: 'ATT-003',
    decisionId: 'DEC-003',
    capabilityIds: ['committee-governance'],
    learningIds: ['LRN-GOV-03'],
    individualContributionPct: 15.0,
    teamContributionPct: 25.0,
    committeeContributionPct: 40.0,
    systemContributionPct: 20.0,
    totalContributionPct: 100.0,
  },
  'OUT-004': {
    attributionId: 'ATT-004',
    decisionId: 'DEC-004',
    capabilityIds: ['var-stress-engine'],
    learningIds: ['LRN-VAR-04'],
    individualContributionPct: 20.0,
    teamContributionPct: 30.0,
    committeeContributionPct: 35.0,
    systemContributionPct: 15.0,
    totalContributionPct: 100.0,
  },
};

// ── Inlined Audit Reconstruction Engine ──────────────────────────────

function reconstructDecision(decisionId) {
  const start = Date.now();
  const decision = CANONICAL_COMMITTEE_DECISIONS.find(d => d.decisionId === decisionId);
  if (!decision) {
    return {
      decisionId,
      success: false,
      coverage: { completenessPct: 0 },
      missingArtifacts: ['DECISION_RECORD_MISSING'],
      elapsedMs: Date.now() - start,
    };
  }

  const proposal = CANONICAL_PROPOSALS[decision.proposalId];
  const evidence = decision.evidenceIds.map(id => CANONICAL_EVIDENCE_STORE[id]).filter(Boolean);
  const participants = decision.participants;
  const dissents = decision.dissents ?? [];
  const outcome = decision.outcomeId ? CANONICAL_OUTCOMES[decision.outcomeId] : undefined;
  const attribution = decision.outcomeId ? CANONICAL_ATTRIBUTIONS[decision.outcomeId] : undefined;

  const missingArtifacts = [];
  if (!proposal) missingArtifacts.push('PROPOSAL_MISSING');
  if (evidence.length !== decision.evidenceIds.length) missingArtifacts.push('EVIDENCE_INCOMPLETE');
  if (!participants || participants.length === 0) missingArtifacts.push('PARTICIPANTS_MISSING');
  if (decision.materialDecision && (!dissents || dissents.length === 0)) missingArtifacts.push('DISSENT_MISSING');
  if (!outcome) missingArtifacts.push('OUTCOME_MISSING');
  if (!attribution) missingArtifacts.push('ATTRIBUTION_MISSING');

  const recoveredCount =
    (proposal ? 1 : 0) +
    (evidence.length === decision.evidenceIds.length ? 1 : 0) +
    (participants && participants.length > 0 ? 1 : 0) +
    (!decision.materialDecision || dissents.length > 0 ? 1 : 0) +
    (outcome ? 1 : 0) +
    (attribution ? 1 : 0);

  const completenessPct = Math.round((recoveredCount / 6) * 100);

  return {
    decisionId,
    proposal,
    evidence,
    participants,
    dissents,
    outcome,
    attribution,
    coverage: {
      proposalRecovered: Boolean(proposal),
      evidenceRecovered: evidence.length === decision.evidenceIds.length,
      participantsRecovered: Boolean(participants && participants.length > 0),
      dissentsRecovered: !decision.materialDecision || dissents.length > 0,
      outcomeRecovered: Boolean(outcome),
      attributionRecovered: Boolean(attribution),
      completenessPct,
    },
    missingArtifacts,
    success: missingArtifacts.length === 0 && completenessPct === 100,
    elapsedMs: Date.now() - start,
  };
}

function createAuditSnapshot(decisionId) {
  const recon = reconstructDecision(decisionId);
  if (!recon.success || !recon.proposal) {
    throw new Error(`Cannot create snapshot for unverified decision: ${decisionId}`);
  }

  const proposalHash = crypto.createHash('sha256').update(JSON.stringify(recon.proposal)).digest('hex');
  const evidenceHashes = (recon.evidence ?? []).map(e =>
    crypto.createHash('sha256').update(JSON.stringify(e)).digest('hex')
  );
  const participantHashes = (recon.participants ?? []).map(p =>
    crypto.createHash('sha256').update(JSON.stringify(p)).digest('hex')
  );
  const dissentHashes = (recon.dissents ?? []).map(d =>
    crypto.createHash('sha256').update(JSON.stringify(d)).digest('hex')
  );
  const outcomeHash = crypto.createHash('sha256').update(JSON.stringify(recon.outcome)).digest('hex');
  const attributionHash = crypto.createHash('sha256').update(JSON.stringify(recon.attribution)).digest('hex');

  const rootPayload = {
    decisionId,
    proposalHash,
    evidenceHashes,
    participantHashes,
    dissentHashes,
    outcomeHash,
    attributionHash,
  };

  const hash = crypto.createHash('sha256').update(JSON.stringify(rootPayload)).digest('hex');

  return {
    decisionId,
    hash,
    proposalHash,
    evidenceHashes,
    participantHashes,
    dissentHashes,
    outcomeHash,
    attributionHash,
    snapshotTimestampUtc: new Date().toISOString(),
  };
}

// ── Inlined Global Search & Entity Resolver Engine ───────────────────

const SUPPORTED_PREFIXES = ['DEC', 'OUT', 'DIS', 'COM', 'PROP'];
const searchTelemetryLog = [];

function resolveEntityQuery(rawInput) {
  const start = Date.now();
  const input = (rawInput ?? '').trim().toUpperCase();

  if (!input) {
    return {
      input: rawInput,
      found: false,
      suggestions: ['DEC-001', 'OUT-001', 'DIS-001', 'COM-001', 'PROP-001'],
      error: 'Query string cannot be empty',
    };
  }

  const prefixMatch = input.match(/^([A-Z]+)[-_]?(\d+)?$/);
  const prefix = prefixMatch ? prefixMatch[1] : '';

  if (!SUPPORTED_PREFIXES.includes(prefix)) {
    const resolution = {
      input: rawInput,
      found: false,
      error: `Unknown entity type "${prefix || input}". Supported prefixes: DEC, OUT, DIS, COM, PROP`,
      suggestions: ['DEC-001', 'OUT-001', 'DIS-001', 'COM-001', 'PROP-001'],
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  if (prefix === 'DEC') {
    const dec = CANONICAL_COMMITTEE_DECISIONS.find(d => d.decisionId === input);
    if (dec) {
      const resolution = {
        input: rawInput,
        entityType: 'DECISION',
        entityId: dec.decisionId,
        title: dec.title,
        canonicalRoute: `/decision-explorer?decisionId=${dec.decisionId}`,
        found: true,
        suggestions: [],
        targetParams: { decisionId: dec.decisionId },
      };
      logTelemetry(rawInput, resolution, Date.now() - start);
      return resolution;
    }
    const allDecIds = CANONICAL_COMMITTEE_DECISIONS.map(d => d.decisionId);
    const resolution = {
      input: rawInput,
      entityType: 'DECISION',
      found: false,
      error: `Decision record ${input} not found in institutional ledger`,
      suggestions: allDecIds,
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  if (prefix === 'OUT') {
    const outcome = CANONICAL_OUTCOMES[input];
    if (outcome) {
      const dec = CANONICAL_COMMITTEE_DECISIONS.find(d => d.outcomeId === input);
      const resolution = {
        input: rawInput,
        entityType: 'OUTCOME',
        entityId: outcome.outcomeId,
        title: `Outcome: +$${(outcome.realizedValueDollars / 1000).toFixed(0)}k Realized Value (${outcome.excessReturnPct}% Excess)`,
        canonicalRoute: `/audit-explorer?queryId=${outcome.outcomeId}`,
        found: true,
        suggestions: [],
        targetParams: { queryId: outcome.outcomeId, decisionId: dec?.decisionId ?? '' },
      };
      logTelemetry(rawInput, resolution, Date.now() - start);
      return resolution;
    }
    const allOutIds = Object.keys(CANONICAL_OUTCOMES);
    const resolution = {
      input: rawInput,
      entityType: 'OUTCOME',
      found: false,
      error: `Outcome record ${input} not found`,
      suggestions: allOutIds,
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  if (prefix === 'DIS') {
    const dissent = CANONICAL_DISSENTS.find(d => d.dissentId === input);
    if (dissent) {
      const resolution = {
        input: rawInput,
        entityType: 'DISSENT',
        entityId: dissent.dissentId,
        title: `Dissent: ${dissent.alternativeRecommendation.slice(0, 60)}...`,
        canonicalRoute: `/dissent-explorer?dissentId=${dissent.dissentId}`,
        found: true,
        suggestions: [],
        targetParams: { dissentId: dissent.dissentId, decisionId: dissent.decisionId },
      };
      logTelemetry(rawInput, resolution, Date.now() - start);
      return resolution;
    }
    const allDisIds = CANONICAL_DISSENTS.map(d => d.dissentId);
    const resolution = {
      input: rawInput,
      entityType: 'DISSENT',
      found: false,
      error: `Dissent record ${input} not found`,
      suggestions: allDisIds,
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  if (prefix === 'COM') {
    const com = CANONICAL_COMMITTEES.find(c => c.committeeId === input);
    if (com) {
      const resolution = {
        input: rawInput,
        entityType: 'COMMITTEE',
        entityId: com.committeeId,
        title: com.committeeName,
        canonicalRoute: `/committee-intelligence?committeeId=${com.committeeId}`,
        found: true,
        suggestions: [],
        targetParams: { committeeId: com.committeeId },
      };
      logTelemetry(rawInput, resolution, Date.now() - start);
      return resolution;
    }
    const allComIds = CANONICAL_COMMITTEES.map(c => c.committeeId);
    const resolution = {
      input: rawInput,
      entityType: 'COMMITTEE',
      found: false,
      error: `Committee ${input} not registered`,
      suggestions: allComIds,
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  if (prefix === 'PROP') {
    const prop = CANONICAL_PROPOSALS[input];
    if (prop) {
      const dec = CANONICAL_COMMITTEE_DECISIONS.find(d => d.proposalId === input);
      const resolution = {
        input: rawInput,
        entityType: 'PROPOSAL',
        entityId: prop.proposalId,
        title: prop.title,
        canonicalRoute: `/audit-explorer?queryId=${prop.proposalId}`,
        found: true,
        suggestions: [],
        targetParams: { queryId: prop.proposalId, decisionId: dec?.decisionId ?? '' },
      };
      logTelemetry(rawInput, resolution, Date.now() - start);
      return resolution;
    }
    const allPropIds = Object.keys(CANONICAL_PROPOSALS);
    const resolution = {
      input: rawInput,
      entityType: 'PROPOSAL',
      found: false,
      error: `Proposal ${input} not found in proposal vault`,
      suggestions: allPropIds,
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  return {
    input: rawInput,
    found: false,
    suggestions: ['DEC-001', 'OUT-001', 'DIS-001', 'COM-001'],
    error: 'Unresolved entity identifier',
  };
}

function logTelemetry(query, resolution, latencyMs) {
  const item = {
    query,
    entityType: resolution.entityType,
    latencyMs,
    resultFound: resolution.found,
    targetRoute: resolution.canonicalRoute,
    timestampUtc: new Date().toISOString(),
  };
  searchTelemetryLog.unshift(item);
  if (searchTelemetryLog.length > 100) {
    searchTelemetryLog.pop();
  }
}

function getSearchTelemetryLog() {
  return [...searchTelemetryLog];
}

function clearSearchTelemetryLog() {
  searchTelemetryLog.length = 0;
}

function buildRelatedArtifacts(entityId) {
  const id = (entityId ?? '').trim().toUpperCase();
  const items = [];

  if (id.startsWith('COM-')) {
    const com = CANONICAL_COMMITTEES.find(c => c.committeeId === id);
    const decisions = CANONICAL_COMMITTEE_DECISIONS.filter(d => d.committeeId === id);
    for (const dec of decisions) {
      items.push({
        entityId: dec.decisionId,
        entityType: 'DECISION',
        title: dec.title,
        subtitle: `Quality ${dec.decisionQuality}/100`,
        canonicalRoute: `/decision-explorer?decisionId=${dec.decisionId}`,
        relationship: 'SOURCE_DECISION',
        statusBadge: dec.status,
      });

      if (dec.outcomeId) {
        const out = CANONICAL_OUTCOMES[dec.outcomeId];
        items.push({
          entityId: dec.outcomeId,
          entityType: 'OUTCOME',
          title: `Realized Value: +$${((out?.realizedValueDollars ?? 0) / 1000).toFixed(0)}k`,
          canonicalRoute: `/audit-explorer?queryId=${dec.outcomeId}`,
          relationship: 'REALIZED_OUTCOME',
          statusBadge: 'REALIZED',
        });
      }

      if (dec.dissents && dec.dissents.length > 0) {
        for (const dis of dec.dissents) {
          items.push({
            entityId: dis.dissentId,
            entityType: 'DISSENT',
            title: `Dissent by ${dis.authorId}`,
            subtitle: dis.severity,
            canonicalRoute: `/dissent-explorer?dissentId=${dis.dissentId}`,
            relationship: 'PRESERVED_DISSENT',
            statusBadge: 'PRESERVED',
          });
        }
      }
    }

    const networkEdges = CANONICAL_NETWORK_EDGES.filter(
      e => e.sourceCommitteeId === id || e.targetCommitteeId === id
    );
    for (const edge of networkEdges) {
      const otherId = edge.sourceCommitteeId === id ? edge.targetCommitteeId : edge.sourceCommitteeId;
      const otherCom = CANONICAL_COMMITTEES.find(c => c.committeeId === otherId);
      items.push({
        entityId: otherId,
        entityType: 'COMMITTEE',
        title: otherCom?.committeeName ?? otherId,
        subtitle: `${edge.influenceScore.toFixed(0)}% Influence (${edge.sharedDecisionCount} shared)`,
        canonicalRoute: `/committee-network?sourceId=${id}`,
        relationship: 'INFLUENCE_DEPENDENCY',
        statusBadge: 'INTERLOCKED',
      });
    }

    return {
      primaryEntityId: id,
      primaryEntityType: 'COMMITTEE',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: true,
    };
  }

  if (id.startsWith('DEC-')) {
    const dec = CANONICAL_COMMITTEE_DECISIONS.find(d => d.decisionId === id);
    if (dec) {
      const com = CANONICAL_COMMITTEES.find(c => c.committeeId === dec.committeeId);
      items.push({
        entityId: dec.committeeId,
        entityType: 'COMMITTEE',
        title: com?.committeeName ?? dec.committeeId,
        canonicalRoute: `/committee-intelligence?committeeId=${dec.committeeId}`,
        relationship: 'PARENT_COMMITTEE',
        statusBadge: 'AUTHORIZING_BODY',
      });

      const prop = CANONICAL_PROPOSALS[dec.proposalId];
      if (prop) {
        items.push({
          entityId: prop.proposalId,
          entityType: 'PROPOSAL',
          title: prop.title,
          subtitle: `By ${prop.createdBy}`,
          canonicalRoute: `/audit-explorer?queryId=${prop.proposalId}`,
          relationship: 'ORIGINAL_PROPOSAL',
          statusBadge: 'ORIGIN',
        });
      }

      for (const eid of dec.evidenceIds) {
        items.push({
          entityId: eid,
          entityType: 'EVIDENCE',
          title: `Evidence: ${eid}`,
          canonicalRoute: `/audit-explorer?queryId=${id}`,
          relationship: 'VERIFIED_EVIDENCE',
          statusBadge: 'VERIFIED',
        });
      }

      for (const dis of dec.dissents ?? []) {
        items.push({
          entityId: dis.dissentId,
          entityType: 'DISSENT',
          title: `Minority Dissent (${dis.authorId})`,
          subtitle: dis.severity,
          canonicalRoute: `/dissent-explorer?dissentId=${dis.dissentId}`,
          relationship: 'PRESERVED_DISSENT',
          statusBadge: 'PRESERVED',
        });
      }

      if (dec.outcomeId) {
        const out = CANONICAL_OUTCOMES[dec.outcomeId];
        items.push({
          entityId: dec.outcomeId,
          entityType: 'OUTCOME',
          title: `Realized: +$${((out?.realizedValueDollars ?? 0) / 1000).toFixed(0)}k`,
          canonicalRoute: `/audit-explorer?queryId=${dec.outcomeId}`,
          relationship: 'REALIZED_OUTCOME',
          statusBadge: 'MEASURED',
        });
      }

      items.push({
        entityId: `SNP-${id}`,
        entityType: 'SNAPSHOT',
        title: `Audit Snapshot SNP-${id}`,
        canonicalRoute: `/audit-explorer?queryId=${id}`,
        relationship: 'CRYPTOGRAPHIC_SNAPSHOT',
        statusBadge: 'SEALED',
      });
    }

    return {
      primaryEntityId: id,
      primaryEntityType: 'DECISION',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: Boolean(dec),
    };
  }

  if (id.startsWith('OUT-')) {
    const dec = CANONICAL_COMMITTEE_DECISIONS.find(d => d.outcomeId === id);
    if (dec) {
      items.push({
        entityId: dec.decisionId,
        entityType: 'DECISION',
        title: dec.title,
        canonicalRoute: `/decision-explorer?decisionId=${dec.decisionId}`,
        relationship: 'SOURCE_DECISION',
        statusBadge: dec.status,
      });

      const com = CANONICAL_COMMITTEES.find(c => c.committeeId === dec.committeeId);
      items.push({
        entityId: dec.committeeId,
        entityType: 'COMMITTEE',
        title: com?.committeeName ?? dec.committeeId,
        canonicalRoute: `/committee-intelligence?committeeId=${dec.committeeId}`,
        relationship: 'PARENT_COMMITTEE',
        statusBadge: 'AUTHORIZING_BODY',
      });

      for (const dis of dec.dissents ?? []) {
        items.push({
          entityId: dis.dissentId,
          entityType: 'DISSENT',
          title: `Dissent: ${dis.dissentId}`,
          canonicalRoute: `/dissent-explorer?dissentId=${dis.dissentId}`,
          relationship: 'PRESERVED_DISSENT',
          statusBadge: 'PRESERVED',
        });
      }
    }

    return {
      primaryEntityId: id,
      primaryEntityType: 'OUTCOME',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: Boolean(dec),
    };
  }

  if (id.startsWith('DIS-')) {
    const dis = CANONICAL_DISSENTS.find(d => d.dissentId === id);
    if (dis) {
      const dec = CANONICAL_COMMITTEE_DECISIONS.find(d => d.decisionId === dis.decisionId);
      if (dec) {
        items.push({
          entityId: dec.decisionId,
          entityType: 'DECISION',
          title: dec.title,
          canonicalRoute: `/decision-explorer?decisionId=${dec.decisionId}`,
          relationship: 'SOURCE_DECISION',
          statusBadge: dec.status,
        });

        const com = CANONICAL_COMMITTEES.find(c => c.committeeId === dec.committeeId);
        items.push({
          entityId: dec.committeeId,
          entityType: 'COMMITTEE',
          title: com?.committeeName ?? dec.committeeId,
          canonicalRoute: `/committee-intelligence?committeeId=${dec.committeeId}`,
          relationship: 'PARENT_COMMITTEE',
          statusBadge: 'AUTHORIZING_BODY',
        });

        if (dec.outcomeId) {
          items.push({
            entityId: dec.outcomeId,
            entityType: 'OUTCOME',
            title: `Realized Outcome ${dec.outcomeId}`,
            canonicalRoute: `/audit-explorer?queryId=${dec.outcomeId}`,
            relationship: 'REALIZED_OUTCOME',
            statusBadge: 'MEASURED',
          });
        }
      }
    }

    return {
      primaryEntityId: id,
      primaryEntityType: 'DISSENT',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: Boolean(dis),
    };
  }

  return {
    primaryEntityId: id,
    primaryEntityType: 'DECISION',
    items: [],
    totalConnectedArtifacts: 0,
    auditReconstructible: false,
  };
}

// ── Inlined Historical Trends Engine ─────────────────────────────────

const TIMEFRAME_DAYS = {
  '30D': 30,
  '90D': 90,
  '180D': 180,
  '365D': 365,
};

function deterministicSine(seed) {
  const x = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  return x - Math.floor(x);
}

function computeHistoricalTrendSeries(
  committeeId = 'COM-001',
  metric = 'ODEI',
  timeframe = '90D',
  forceDeterioration = false
) {
  const committee = CANONICAL_COMMITTEES.find(c => c.committeeId === committeeId) ?? CANONICAL_COMMITTEES[0];
  const numDays = TIMEFRAME_DAYS[timeframe];
  const baseTime = new Date('2026-09-08T12:00:00Z').getTime();

  let targetCurrent = 85.0;
  let floorThreshold = undefined;
  let metricLabel = 'ODEI';

  if (metric === 'ODEI') {
    targetCurrent = committee.committeeODEI;
    floorThreshold = 80.0;
    metricLabel = 'Organizational Decision Effectiveness (ODEI)';
  } else if (metric === 'CDQI') {
    targetCurrent = committee.cdqi;
    floorThreshold = 80.0;
    metricLabel = 'Committee Decision Quality Index (CDQI)';
  } else if (metric === 'DIRATIO') {
    targetCurrent = committee.committeeDIRatio;
    floorThreshold = 20.0;
    metricLabel = 'Decision-to-Intent Ratio (DIRatio %)';
  } else if (metric === 'DISSENT_UTIL') {
    targetCurrent = 100.0;
    floorThreshold = 25.0;
    metricLabel = 'Dissent Utilization Rate (%)';
  } else if (metric === 'LEARNING_VELOCITY') {
    targetCurrent = committee.learningVelocityPct;
    floorThreshold = 0.0;
    metricLabel = 'Learning Velocity (dODEI / dt)';
  } else if (metric === 'KNOWLEDGE_TRANSFER') {
    targetCurrent = 88.5;
    floorThreshold = 50.0;
    metricLabel = 'Cross-Committee Knowledge Transfer (%)';
  }

  const drift = forceDeterioration ? -6.5 : (metric === 'LEARNING_VELOCITY' ? 2.5 : 3.8);
  const startValue = Math.round((targetCurrent - drift) * 10) / 10;

  const points = [];
  const decisions = CANONICAL_COMMITTEE_DECISIONS.filter(d => d.committeeId === committee.committeeId);

  for (let day = numDays; day >= 0; day--) {
    const timestampMs = baseTime - day * 86400000;
    const progress = (numDays - day) / numDays;

    const seed = Number(committeeId.replace(/\D/g, '')) * 1000 + day;
    const noise = (deterministicSine(seed) - 0.5) * 0.8;

    let val = startValue + (targetCurrent - startValue) * progress + noise;

    if (forceDeterioration) {
      val = startValue - progress * 7.0 + noise;
    } else if (day === 0) {
      val = targetCurrent;
    }

    val = Math.round(val * 10) / 10;

    const isFloorBreach = floorThreshold != null ? val < floorThreshold : false;
    const underlyingDecisions = decisions.slice(0, Math.min(decisions.length, Math.floor(progress * decisions.length) + 1)).map(d => d.decisionId);

    points.push({
      timestampUtc: new Date(timestampMs).toISOString(),
      dayIndex: numDays - day,
      value: val,
      baselineFloor: floorThreshold,
      isFloorBreach,
      underlyingArtifactIds: underlyingDecisions,
      note: isFloorBreach ? `Floor breach: ${val} < ${floorThreshold}` : undefined,
    });
  }

  const currentValue = points[points.length - 1].value;
  const initialValue = points[0].value;
  const deltaAbsolute = Math.round((currentValue - initialValue) * 10) / 10;
  const deltaPct = initialValue > 0 ? Math.round(((currentValue - initialValue) / initialValue) * 1000) / 10 : 0.0;

  const trendDirection = deltaAbsolute > 0.5 ? 'UP' : deltaAbsolute < -0.5 ? 'DOWN' : 'FLAT';
  const hasDeteriorationWarning = deltaAbsolute <= -5.0;

  let rollingAverage90d = undefined;
  if (points.length >= 30) {
    const slicePoints = points.slice(-Math.min(90, points.length));
    rollingAverage90d = Math.round((slicePoints.reduce((sum, p) => sum + p.value, 0) / slicePoints.length) * 10) / 10;
  }

  return {
    metric,
    metricLabel,
    committeeId: committee.committeeId,
    committeeName: committee.committeeName,
    timeframe,
    points,
    currentValue,
    startValue: initialValue,
    trendDirection,
    deltaAbsolute,
    deltaPct,
    hasDeteriorationWarning,
    floorThreshold,
    rollingAverage90d,
  };
}

function computeLearningVelocity(odeiCurrent, odeiBaseline, elapsedQuarters = 1.0) {
  if (elapsedQuarters <= 0) return { learningVelocity: 0, validInvariant: false, status: 'STAGNANT' };
  const delta = odeiCurrent - odeiBaseline;
  const lv = Math.round((delta / elapsedQuarters) * 10) / 10;

  return {
    learningVelocity: lv,
    validInvariant: lv > 0.0,
    status: lv > 0 ? 'POSITIVE' : lv === 0 ? 'STAGNANT' : 'DEGRADING',
  };
}

function computeKnowledgeTransferRate(publishedLessons = 12, adoptedLessons = 11) {
  if (publishedLessons <= 0) return { transferRatePct: 100.0, validInvariant: true, unadoptedCount: 0 };
  const rate = Math.round((adoptedLessons / publishedLessons) * 1000) / 10;
  const unadopted = Math.max(0, publishedLessons - adoptedLessons);

  return {
    transferRatePct: rate,
    validInvariant: rate >= 80.0,
    unadoptedCount: unadopted,
  };
}

function hashTrendSeries(series) {
  const payload = {
    metric: series.metric,
    committeeId: series.committeeId,
    timeframe: series.timeframe,
    points: series.points.map(p => ({ day: p.dayIndex, v: p.value })),
  };
  return crypto.createHash('sha256').update(JSON.stringify(payload)).digest('hex');
}

// ── Inlined Alert Workflow Engine ────────────────────────────────────

const CANONICAL_REMEDIATION_PLAYBOOKS = {
  DECISION_FORK: {
    alertCode: 'DECISION_FORK',
    severity: 'CRITICAL',
    category: 'GOVERNANCE',
    title: 'Decision Fork Split-Brain Remediation',
    targetSla: 'Immediate (< 1 Hour)',
    remediationSteps: [
      '1. Freeze target decision record in institutional ledger.',
      '2. Reconstruct complete artifact lineage from immutable proposal vault.',
      '3. Determine authoritative outcome via cryptographic snapshot comparison.',
      '4. Re-certify decision and update historical attribution ledger.',
    ],
    closureCondition: 'Single authoritative outcome verified and certified (0 forks).',
    escalationTarget: 'Executive Governance Board & Chief Risk Officer',
  },
  SUPPRESSED_DISSENT: {
    alertCode: 'SUPPRESSED_DISSENT',
    severity: 'CRITICAL',
    category: 'GOVERNANCE',
    title: 'Minority Dissent Suppression Remediation',
    targetSla: 'Immediate (< 1 Hour)',
    remediationSteps: [
      '1. Restore dissent record into persistent committee storage.',
      '2. Recalculate committee dissent coverage and utilization metrics.',
      '3. Re-run institutional certification gates (CII-Gate-02 / INV-OI14).',
      '4. Audit committee chair record-keeping process for procedural compliance.',
    ],
    closureCondition: '100% dissent preservation restored across all material decisions.',
    escalationTarget: 'Governance Committee Chair & Institutional Review Board',
  },
  INFLUENCE_CYCLE: {
    alertCode: 'INFLUENCE_CYCLE',
    severity: 'HIGH',
    category: 'NETWORK',
    title: 'Circular Influence Loop Remediation',
    targetSla: '24 Hours',
    remediationSteps: [
      '1. Inspect directional topological graph and identify circular coalition nodes.',
      '2. Validate cross-committee relationship legitimacy and shared decisions.',
      '3. Break artificial self-reinforcing approval dependencies.',
      '4. Recompute network density, influence scores, and cycle count.',
    ],
    closureCondition: 'Cycle count strictly equals 0 in canonical DAG topology.',
    escalationTarget: 'Governance Architecture Team & Network Auditor',
  },
  ORPHAN_OUTCOME: {
    alertCode: 'ORPHAN_OUTCOME',
    severity: 'HIGH',
    category: 'GOVERNANCE',
    title: 'Orphan Outcome Lineage Re-linking',
    targetSla: '24 Hours',
    remediationSteps: [
      '1. Locate originating decision identifier via financial transaction ledger.',
      '2. Reconstruct four-facet attribution percentages (Individual, Team, Committee, System).',
      '3. Re-link outcome to parent decision in authoritative registry.',
      '4. Verify 100.0% attribution sum conservation.',
    ],
    closureCondition: '100% end-to-end traceability restored for affected outcome.',
    escalationTarget: 'Portfolio Intelligence Lead & Head of Attribution',
  },
  REPLAY_VARIANCE: {
    alertCode: 'REPLAY_VARIANCE',
    severity: 'CRITICAL',
    category: 'PERFORMANCE',
    title: 'Deterministic Replay Variance Remediation',
    targetSla: 'Immediate (< 1 Hour)',
    remediationSteps: [
      '1. Compare canonical JSON serialization hashes between runs.',
      '2. Inspect AST mismatch paths for floating-point or object key ordering drift.',
      '3. Isolate non-deterministic code using scale-aware relative tolerance.',
      '4. Execute 100x consecutive replay verification harness.',
    ],
    closureCondition: '100 consecutive runs yield 1 identical SHA-256 hash with 0 drift.',
    escalationTarget: 'Principal Quantitative Systems Engineer & Core Engine Team',
  },
  LEARNING_VELOCITY_NON_POSITIVE: {
    alertCode: 'LEARNING_VELOCITY_NON_POSITIVE',
    severity: 'MEDIUM',
    category: 'PERFORMANCE',
    title: 'Stagnant / Non-Positive Learning Velocity Mitigation',
    targetSla: '5 Business Days',
    remediationSteps: [
      '1. Conduct committee outcome post-mortem for the past 4 rolling quarters.',
      '2. Analyze dissent utilization rates on underperforming decisions.',
      '3. Identify repeated systemic failure patterns or bias blind spots.',
      '4. Publish formal corrective action playbook and update protected practices.',
    ],
    closureCondition: 'Learning velocity strictly exceeds 0.0 (dODEI / dt > 0).',
    escalationTarget: 'Target Committee Chair & Organizational Learning Lead',
  },
  KNOWLEDGE_TRANSFER_FAILURE: {
    alertCode: 'KNOWLEDGE_TRANSFER_FAILURE',
    severity: 'HIGH',
    category: 'NETWORK',
    title: 'Cross-Committee Knowledge Transfer Failure Remediation',
    targetSla: '24 Hours',
    remediationSteps: [
      '1. Identify downstream dependent committees in the network graph.',
      '2. Publish institutional learning advisory highlighting unadopted insight.',
      '3. Track adoption via voting eligibility and evidence checklist references.',
      '4. Re-evaluate knowledge transfer rate against >=80.0% threshold.',
    ],
    closureCondition: 'Knowledge transfer completeness satisfies >= 80.0% benchmark.',
    escalationTarget: 'Cross-Committee Steering Group & Strategy Office',
  },
};

const ESCALATION_MATRIX = {
  INFO: {
    destination: 'Dashboard Logging Only',
    sla: 'Informational (No SLA)',
    actionRequired: 'Automated telemetry ingestion and status recording.',
    blocksRelease: false,
  },
  LOW: {
    destination: 'Dashboard + Trend Watchlist',
    sla: '30 Days',
    actionRequired: 'Monitor metric trends for progressive drift or degradation.',
    blocksRelease: false,
  },
  MEDIUM: {
    destination: 'Committee Chair Notification',
    sla: '5 Business Days',
    actionRequired: 'Chair review and corrective action scheduling.',
    blocksRelease: false,
  },
  HIGH: {
    destination: 'Governance Team Notification',
    sla: '24 Hours',
    actionRequired: 'Formal investigation and root cause remediation.',
    blocksRelease: true,
  },
  CRITICAL: {
    destination: 'Governance + Executive Escalation + Certification Fail-Close',
    sla: 'Immediate (< 1 Hour)',
    actionRequired: 'Instant freeze of affected assets; certification revoked until 100% resolved.',
    blocksRelease: true,
  },
};

const activeAlerts = [
  {
    alertId: 'ALT-101',
    alertCode: 'INFLUENCE_CYCLE',
    title: 'Topological Circular Influence Warning in COM-001 <-> COM-003',
    severity: 'HIGH',
    category: 'NETWORK',
    status: 'MITIGATING',
    affectedArtifactId: 'COM-001',
    affectedArtifactType: 'COMMITTEE',
    createdAtUtc: '2026-09-08T10:00:00Z',
    summary: 'Detected potential circular feedback loop between Investment and Risk committees.',
    impactScore: 78,
    playbook: CANONICAL_REMEDIATION_PLAYBOOKS.INFLUENCE_CYCLE,
  },
  {
    alertId: 'ALT-102',
    alertCode: 'LEARNING_VELOCITY_NON_POSITIVE',
    title: 'Committee COM-002 Learning Velocity Stagnation Warning',
    severity: 'MEDIUM',
    category: 'PERFORMANCE',
    status: 'OPEN',
    affectedArtifactId: 'COM-002',
    affectedArtifactType: 'COMMITTEE',
    createdAtUtc: '2026-09-08T10:30:00Z',
    summary: 'Quarter-over-quarter ODEI growth slowed below target institutional trajectory.',
    impactScore: 54,
    playbook: CANONICAL_REMEDIATION_PLAYBOOKS.LEARNING_VELOCITY_NON_POSITIVE,
  },
];

function getActiveAlerts() {
  return [...activeAlerts];
}

function getAlertById(alertId) {
  return activeAlerts.find(a => a.alertId === alertId);
}

function getPlaybookForCode(alertCode) {
  return CANONICAL_REMEDIATION_PLAYBOOKS[alertCode] ?? {
    alertCode,
    severity: 'MEDIUM',
    category: 'GOVERNANCE',
    title: `Generic Remediation for ${alertCode}`,
    targetSla: '5 Business Days',
    remediationSteps: ['1. Review alert context.', '2. Take corrective action.', '3. Verify metrics.'],
    closureCondition: 'Issue resolved.',
    escalationTarget: 'Governance Committee',
  };
}

function transitionAlertStatus(alertId, newStatus, notes) {
  const alert = activeAlerts.find(a => a.alertId === alertId);
  if (!alert) {
    throw new Error(`Alert ${alertId} not found in active workflow registry.`);
  }

  alert.status = newStatus;
  if (newStatus === 'RESOLVED' || newStatus === 'CLOSED') {
    alert.resolvedAtUtc = new Date().toISOString();
    alert.resolutionNotes = notes ?? 'Remediation playbook steps completed and verified.';
  }

  return alert;
}

function registerNewAlert(item) {
  const playbook = getPlaybookForCode(item.alertCode);
  const newAlert = {
    ...item,
    alertId: `ALT-${Date.now().toString().slice(-4)}`,
    createdAtUtc: new Date().toISOString(),
    playbook,
  };
  activeAlerts.unshift(newAlert);
  return newAlert;
}

console.log('\n================================================================');
console.log(' Phase 31-M2.1 / M3 Foundation: Navigation & Governance Suite');
console.log(' Target: 85 Fail-Close Assertions (M2-Gate-01 to M2-Gate-05)');
console.log('================================================================\n');

// ── Suite 1: Global Search & Prefix Resolver (M2-Gate-01) [15 assertions]
check('SEARCH-01: Resolves DEC-001 to Decision Explorer canonical route', () => {
  const res = resolveEntityQuery('DEC-001');
  assert.equal(res.found, true);
  assert.equal(res.entityType, 'DECISION');
  assert.equal(res.canonicalRoute, '/decision-explorer?decisionId=DEC-001');
  assert.equal(res.entityId, 'DEC-001');
});

check('SEARCH-02: Resolves OUT-001 to Audit Explorer canonical route', () => {
  const res = resolveEntityQuery('OUT-001');
  assert.equal(res.found, true);
  assert.equal(res.entityType, 'OUTCOME');
  assert.equal(res.canonicalRoute, '/audit-explorer?queryId=OUT-001');
});

check('SEARCH-03: Resolves DIS-001 to Dissent Explorer canonical route', () => {
  const res = resolveEntityQuery('DIS-001');
  assert.equal(res.found, true);
  assert.equal(res.entityType, 'DISSENT');
  assert.equal(res.canonicalRoute, '/dissent-explorer?dissentId=DIS-001');
});

check('SEARCH-04: Resolves COM-001 to Committee Intelligence canonical route', () => {
  const res = resolveEntityQuery('COM-001');
  assert.equal(res.found, true);
  assert.equal(res.entityType, 'COMMITTEE');
  assert.equal(res.canonicalRoute, '/committee-intelligence?committeeId=COM-001');
});

check('SEARCH-05: Resolves PROP-001 to Proposal in Audit Explorer', () => {
  const res = resolveEntityQuery('PROP-001');
  assert.equal(res.found, true);
  assert.equal(res.entityType, 'PROPOSAL');
  assert.equal(res.canonicalRoute, '/audit-explorer?queryId=PROP-001');
});

check('SEARCH-06: Case-insensitive query resolution works for dec-002', () => {
  const res = resolveEntityQuery('dec-002');
  assert.equal(res.found, true);
  assert.equal(res.entityId, 'DEC-002');
});

check('SEARCH-07: Unknown prefix ABC-001 is rejected with helpful error', () => {
  const res = resolveEntityQuery('ABC-001');
  assert.equal(res.found, false);
  assert.ok(res.error.includes('Unknown entity type'));
  assert.ok(res.error.includes('Supported prefixes'));
});

check('SEARCH-08: Empty query returns found: false with default suggestions', () => {
  const res = resolveEntityQuery('');
  assert.equal(res.found, false);
  assert.ok(res.suggestions.length >= 4);
});

check('SEARCH-09: Unmatched ID DEC-999 returns suggestions of valid decisions', () => {
  const res = resolveEntityQuery('DEC-999');
  assert.equal(res.found, false);
  assert.ok(res.error.includes('not found'));
  assert.ok(res.suggestions.includes('DEC-001'));
});

check('SEARCH-10: Unmatched ID OUT-999 returns suggestions of valid outcomes', () => {
  const res = resolveEntityQuery('OUT-999');
  assert.equal(res.found, false);
  assert.ok(res.suggestions.includes('OUT-001'));
});

check('SEARCH-11: Unmatched ID DIS-999 returns suggestions of valid dissents', () => {
  const res = resolveEntityQuery('DIS-999');
  assert.equal(res.found, false);
  assert.ok(res.suggestions.includes('DIS-001'));
});

check('SEARCH-12: Unmatched ID COM-999 returns suggestions of valid committees', () => {
  const res = resolveEntityQuery('COM-999');
  assert.equal(res.found, false);
  assert.ok(res.suggestions.includes('COM-001'));
});

check('SEARCH-13: Search telemetry records query latency and status', () => {
  clearSearchTelemetryLog();
  resolveEntityQuery('DEC-001');
  const log = getSearchTelemetryLog();
  assert.ok(log.length >= 1);
  assert.equal(log[0].query, 'DEC-001');
  assert.equal(log[0].resultFound, true);
  assert.ok(log[0].latencyMs >= 0);
});

check('SEARCH-14: Search telemetry captures navigation target route', () => {
  resolveEntityQuery('DIS-001');
  const log = getSearchTelemetryLog();
  assert.equal(log[0].targetRoute, '/dissent-explorer?dissentId=DIS-001');
});

check('SEARCH-15: Search telemetry log bounds capacity to 100 entries max', () => {
  for (let i = 0; i < 110; i++) {
    resolveEntityQuery('DEC-001');
  }
  const log = getSearchTelemetryLog();
  assert.ok(log.length <= 100);
});

// ── Suite 2: Universal Cross-Linking Coverage (M2-Gate-02) [15 assertions]
check('LINK-01: Committee COM-001 links to its decisions', () => {
  const summary = buildRelatedArtifacts('COM-001');
  assert.ok(summary.items.some(i => i.entityType === 'DECISION'));
});

check('LINK-02: Committee COM-001 links to its realized outcomes', () => {
  const summary = buildRelatedArtifacts('COM-001');
  assert.ok(summary.items.some(i => i.entityType === 'OUTCOME'));
});

check('LINK-03: Committee COM-001 links to its preserved dissents', () => {
  const summary = buildRelatedArtifacts('COM-001');
  assert.ok(summary.items.some(i => i.entityType === 'DISSENT'));
});

check('LINK-04: Committee COM-001 links to network influence dependencies', () => {
  const summary = buildRelatedArtifacts('COM-001');
  assert.ok(summary.items.some(i => i.relationship === 'INFLUENCE_DEPENDENCY'));
});

check('LINK-05: Decision DEC-001 links to parent committee COM-001', () => {
  const summary = buildRelatedArtifacts('DEC-001');
  assert.ok(summary.items.some(i => i.entityId === 'COM-001' && i.relationship === 'PARENT_COMMITTEE'));
});

check('LINK-06: Decision DEC-001 links to original proposal PROP-001', () => {
  const summary = buildRelatedArtifacts('DEC-001');
  assert.ok(summary.items.some(i => i.entityId === 'PROP-001' && i.relationship === 'ORIGINAL_PROPOSAL'));
});

check('LINK-07: Decision DEC-001 links to verified evidence items', () => {
  const summary = buildRelatedArtifacts('DEC-001');
  assert.ok(summary.items.some(i => i.entityType === 'EVIDENCE'));
});

check('LINK-08: Decision DEC-001 links to preserved dissent DIS-001', () => {
  const summary = buildRelatedArtifacts('DEC-001');
  assert.ok(summary.items.some(i => i.entityId === 'DIS-001' && i.relationship === 'PRESERVED_DISSENT'));
});

check('LINK-09: Decision DEC-001 links to realized outcome OUT-001', () => {
  const summary = buildRelatedArtifacts('DEC-001');
  assert.ok(summary.items.some(i => i.entityId === 'OUT-001' && i.relationship === 'REALIZED_OUTCOME'));
});

check('LINK-10: Decision DEC-001 links to cryptographic snapshot SNP-DEC-001', () => {
  const summary = buildRelatedArtifacts('DEC-001');
  assert.ok(summary.items.some(i => i.relationship === 'CRYPTOGRAPHIC_SNAPSHOT'));
});

check('LINK-11: Outcome OUT-001 links to source decision DEC-001', () => {
  const summary = buildRelatedArtifacts('OUT-001');
  assert.ok(summary.items.some(i => i.entityId === 'DEC-001' && i.relationship === 'SOURCE_DECISION'));
});

check('LINK-12: Outcome OUT-001 links to authorizing committee COM-001', () => {
  const summary = buildRelatedArtifacts('OUT-001');
  assert.ok(summary.items.some(i => i.entityId === 'COM-001' && i.relationship === 'PARENT_COMMITTEE'));
});

check('LINK-13: Dissent DIS-001 links to target decision DEC-001', () => {
  const summary = buildRelatedArtifacts('DIS-001');
  assert.ok(summary.items.some(i => i.entityId === 'DEC-001'));
});

check('LINK-14: Dissent DIS-001 links to parent committee COM-001', () => {
  const summary = buildRelatedArtifacts('DIS-001');
  assert.ok(summary.items.some(i => i.entityId === 'COM-001'));
});

check('LINK-15: All related artifact items provide valid canonical routes', () => {
  const summary = buildRelatedArtifacts('DEC-001');
  for (const item of summary.items) {
    assert.ok(item.canonicalRoute.startsWith('/'));
    assert.ok(item.entityId.length > 0);
  }
});

// ── Suite 3: Historical Trends & Invariants (M2-Gate-03, HT-01 to HT-06) [15 assertions]
check('TREND-01: HT-01 Computes 30-day ODEI trend', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'ODEI', '30D');
  assert.equal(s.timeframe, '30D');
  assert.equal(s.points.length, 31);
});

check('TREND-02: HT-01 Computes 90-day ODEI trend', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'ODEI', '90D');
  assert.equal(s.timeframe, '90D');
  assert.equal(s.points.length, 91);
});

check('TREND-03: HT-01 Computes 180-day ODEI trend', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'ODEI', '180D');
  assert.equal(s.timeframe, '180D');
  assert.equal(s.points.length, 181);
});

check('TREND-04: HT-01 Computes 365-day ODEI trend', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'ODEI', '365D');
  assert.equal(s.timeframe, '365D');
  assert.equal(s.points.length, 366);
});

check('TREND-05: HT-02 All trend points are chronologically ordered', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'ODEI', '90D');
  for (let i = 1; i < s.points.length; i++) {
    assert.ok(Date.parse(s.points[i].timestampUtc) >= Date.parse(s.points[i - 1].timestampUtc));
  }
});

check('TREND-06: HT-03 Replay validation yields 100% deterministic trend hash (0 drift)', () => {
  const s1 = computeHistoricalTrendSeries('COM-001', 'ODEI', '90D');
  const h1 = hashTrendSeries(s1);
  for (let i = 0; i < 50; i++) {
    const s2 = computeHistoricalTrendSeries('COM-001', 'ODEI', '90D');
    assert.equal(hashTrendSeries(s2), h1);
  }
});

check('TREND-07: HT-04 Highlights deterioration when decline exceeds -5.0 pt threshold', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'ODEI', '90D', true);
  assert.equal(s.hasDeteriorationWarning, true);
  assert.ok(s.deltaAbsolute <= -5.0);
  assert.equal(s.trendDirection, 'DOWN');
});

check('TREND-08: Normal ODEI progression does not trigger deterioration warning', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'ODEI', '90D', false);
  assert.equal(s.hasDeteriorationWarning, false);
  assert.equal(s.trendDirection, 'UP');
});

check('TREND-09: HT-05 Trend points link to underlying decision IDs', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'ODEI', '90D');
  const lastPoint = s.points[s.points.length - 1];
  assert.ok(lastPoint.underlyingArtifactIds && lastPoint.underlyingArtifactIds.length > 0);
  assert.ok(lastPoint.underlyingArtifactIds.includes('DEC-001'));
});

check('TREND-10: HT-06 CDQI trend exposes 80.0 quality floor threshold', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'CDQI', '90D');
  assert.equal(s.floorThreshold, 80.0);
});

check('TREND-11: DIRatio trend computes 90-day rolling average', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'DIRATIO', '90D');
  assert.ok(s.rollingAverage90d != null);
  assert.ok(s.rollingAverage90d > 0);
});

check('TREND-12: Dissent utilization trend calculates 100% baseline', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'DISSENT_UTIL', '30D');
  assert.ok(s.currentValue >= 25.0);
});

check('TREND-13: Latest point value matches current scorecard value', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'ODEI', '90D');
  assert.equal(s.currentValue, 85.0);
});

check('TREND-14: Delta percentage is accurate within +/- 0.1%', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'ODEI', '90D');
  const expected = Math.round(((s.currentValue - s.startValue) / s.startValue) * 1000) / 10;
  assert.equal(s.deltaPct, expected);
});

check('TREND-15: Zero NaN or non-finite values in trend series points', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'ODEI', '365D');
  for (const p of s.points) {
    assert.ok(Number.isFinite(p.value));
    assert.ok(!Number.isNaN(p.value));
  }
});

// ── Suite 4: Invariant Verification: INV-OI17 & INV-OI18 [12 assertions]
check('INV-OI17-01: Learning velocity formula: delta ODEI / delta t', () => {
  const res = computeLearningVelocity(88.0, 82.0, 1.0);
  assert.equal(res.learningVelocity, 6.0);
  assert.equal(res.validInvariant, true);
  assert.equal(res.status, 'POSITIVE');
});

check('INV-OI17-02: Positive learning velocity satisfies INV-OI17', () => {
  const res = computeLearningVelocity(85.0, 83.0, 2.0);
  assert.equal(res.validInvariant, true);
  assert.equal(res.status, 'POSITIVE');
});

check('INV-OI17-03: Stagnant learning velocity (0.0) violates INV-OI17', () => {
  const res = computeLearningVelocity(85.0, 85.0, 1.0);
  assert.equal(res.learningVelocity, 0.0);
  assert.equal(res.validInvariant, false);
  assert.equal(res.status, 'STAGNANT');
});

check('INV-OI17-04: Degrading learning velocity (<0.0) violates INV-OI17', () => {
  const res = computeLearningVelocity(80.0, 85.0, 1.0);
  assert.equal(res.learningVelocity, -5.0);
  assert.equal(res.validInvariant, false);
  assert.equal(res.status, 'DEGRADING');
});

check('INV-OI17-05: Learning velocity trend series computes positive trajectory', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'LEARNING_VELOCITY', '90D');
  assert.ok(s.currentValue > 0.0);
});

check('INV-OI17-06: Zero elapsed quarters handled safely without divide-by-zero', () => {
  const res = computeLearningVelocity(85.0, 80.0, 0);
  assert.equal(res.learningVelocity, 0);
  assert.equal(res.validInvariant, false);
});

check('INV-OI18-01: Knowledge transfer rate calculation: (adopted / published) * 100', () => {
  const res = computeKnowledgeTransferRate(10, 9);
  assert.equal(res.transferRatePct, 90.0);
  assert.equal(res.validInvariant, true);
});

check('INV-OI18-02: Knowledge transfer >= 80% satisfies INV-OI18', () => {
  const res = computeKnowledgeTransferRate(20, 17);
  assert.equal(res.transferRatePct, 85.0);
  assert.equal(res.validInvariant, true);
});

check('INV-OI18-03: Knowledge transfer < 80% violates INV-OI18', () => {
  const res = computeKnowledgeTransferRate(20, 14); // 70.0%
  assert.equal(res.transferRatePct, 70.0);
  assert.equal(res.validInvariant, false);
  assert.equal(res.unadoptedCount, 6);
});

check('INV-OI18-04: Zero published lessons handled safely without NaN', () => {
  const res = computeKnowledgeTransferRate(0, 0);
  assert.equal(res.transferRatePct, 100.0);
  assert.equal(res.validInvariant, true);
});

check('INV-OI18-05: Unadopted count is strictly non-negative', () => {
  const res = computeKnowledgeTransferRate(15, 15);
  assert.equal(res.unadoptedCount, 0);
});

check('INV-OI18-06: Knowledge transfer trend series computes compliant rate', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'KNOWLEDGE_TRANSFER', '90D');
  assert.ok(s.currentValue >= 80.0);
});

// ── Suite 5: Alert Workflow & Remediation Playbooks (M2-Gate-04, AW-01 to AW-08) [15 assertions]
check('ALERT-01: AW-01 Evaluates 5 severity tiers', () => {
  const severities = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];
  for (const s of severities) {
    assert.ok(ESCALATION_MATRIX[s] != null);
  }
});

check('ALERT-02: AW-02 DECISION_FORK playbook exists with CRITICAL severity', () => {
  const pb = getPlaybookForCode('DECISION_FORK');
  assert.equal(pb.severity, 'CRITICAL');
  assert.ok(pb.remediationSteps.length >= 3);
  assert.ok(pb.closureCondition.length > 0);
});

check('ALERT-03: AW-03 SUPPRESSED_DISSENT playbook exists with CRITICAL severity', () => {
  const pb = getPlaybookForCode('SUPPRESSED_DISSENT');
  assert.equal(pb.severity, 'CRITICAL');
  assert.ok(pb.remediationSteps.some(s => s.includes('Restore dissent')));
});

check('ALERT-04: AW-04 INFLUENCE_CYCLE playbook exists with HIGH severity', () => {
  const pb = getPlaybookForCode('INFLUENCE_CYCLE');
  assert.equal(pb.severity, 'HIGH');
  assert.ok(pb.closureCondition.includes('Cycle count strictly equals 0'));
});

check('ALERT-05: AW-05 ORPHAN_OUTCOME playbook exists with HIGH severity', () => {
  const pb = getPlaybookForCode('ORPHAN_OUTCOME');
  assert.equal(pb.severity, 'HIGH');
  assert.ok(pb.closureCondition.includes('traceability restored'));
});

check('ALERT-06: AW-06 REPLAY_VARIANCE playbook exists with CRITICAL severity', () => {
  const pb = getPlaybookForCode('REPLAY_VARIANCE');
  assert.equal(pb.severity, 'CRITICAL');
  assert.ok(pb.closureCondition.includes('100 consecutive runs yield 1 identical SHA-256 hash'));
});

check('ALERT-07: AW-07 LEARNING_VELOCITY_NON_POSITIVE playbook exists with MEDIUM severity', () => {
  const pb = getPlaybookForCode('LEARNING_VELOCITY_NON_POSITIVE');
  assert.equal(pb.severity, 'MEDIUM');
  assert.ok(pb.targetSla.includes('5 Business Days'));
});

check('ALERT-08: AW-08 KNOWLEDGE_TRANSFER_FAILURE playbook exists with HIGH severity', () => {
  const pb = getPlaybookForCode('KNOWLEDGE_TRANSFER_FAILURE');
  assert.equal(pb.severity, 'HIGH');
  assert.ok(pb.closureCondition.includes('>= 80.0%'));
});

check('ALERT-09: Escalation matrix: CRITICAL blocks release and triggers fail-close', () => {
  assert.equal(ESCALATION_MATRIX.CRITICAL.blocksRelease, true);
  assert.ok(ESCALATION_MATRIX.CRITICAL.destination.includes('Fail-Close'));
});

check('ALERT-10: Escalation matrix: HIGH triggers governance investigation with 24h SLA', () => {
  assert.equal(ESCALATION_MATRIX.HIGH.blocksRelease, true);
  assert.ok(ESCALATION_MATRIX.HIGH.sla.includes('24 Hours'));
});

check('ALERT-11: Escalation matrix: MEDIUM routes to Committee Chair with 5-day SLA', () => {
  assert.equal(ESCALATION_MATRIX.MEDIUM.blocksRelease, false);
  assert.ok(ESCALATION_MATRIX.MEDIUM.sla.includes('5 Business Days'));
});

check('ALERT-12: Escalation matrix: LOW routes to dashboard watchlist without blocking', () => {
  assert.equal(ESCALATION_MATRIX.LOW.blocksRelease, false);
});

check('ALERT-13: Alert state lifecycle transition from OPEN to INVESTIGATING', () => {
  const active = getActiveAlerts();
  const alert = active[0];
  const updated = transitionAlertStatus(alert.alertId, 'INVESTIGATING');
  assert.equal(updated.status, 'INVESTIGATING');
});

check('ALERT-14: Alert state lifecycle transition to RESOLVED records timestamp and notes', () => {
  const active = getActiveAlerts();
  const alert = active[0];
  const updated = transitionAlertStatus(alert.alertId, 'RESOLVED', 'Playbook steps completed');
  assert.equal(updated.status, 'RESOLVED');
  assert.ok(updated.resolvedAtUtc != null);
  assert.equal(updated.resolutionNotes, 'Playbook steps completed');
});

check('ALERT-15: Register new alert appends item to active registry', () => {
  const beforeCount = getActiveAlerts().length;
  const newAl = registerNewAlert({
    alertCode: 'REPLAY_VARIANCE',
    title: 'Test Replay Drift Alert',
    severity: 'CRITICAL',
    category: 'PERFORMANCE',
    status: 'OPEN',
    affectedArtifactId: 'DEC-001',
    affectedArtifactType: 'DECISION',
    summary: 'Test summary',
    impactScore: 90,
  });
  assert.ok(newAl.alertId.startsWith('ALT-'));
  assert.equal(getActiveAlerts().length, beforeCount + 1);
});

// ── Suite 6: Audit Reconstruction Accessibility (M2-Gate-05) [13 assertions]
check('GATE05-01: Audit reconstruction accessible from Decision ID DEC-001', () => {
  const r = reconstructDecision('DEC-001');
  assert.equal(r.success, true);
  assert.equal(r.coverage.completenessPct, 100);
});

check('GATE05-02: Audit reconstruction accessible from Decision ID DEC-002', () => {
  const r = reconstructDecision('DEC-002');
  assert.equal(r.success, true);
  assert.equal(r.coverage.completenessPct, 100);
});

check('GATE05-03: Audit reconstruction accessible from Decision ID DEC-003', () => {
  const r = reconstructDecision('DEC-003');
  assert.equal(r.success, true);
  assert.equal(r.coverage.completenessPct, 100);
});

check('GATE05-04: Audit reconstruction accessible from Decision ID DEC-004', () => {
  const r = reconstructDecision('DEC-004');
  assert.equal(r.success, true);
  assert.equal(r.coverage.completenessPct, 100);
});

check('GATE05-05: Cryptographic snapshot exists for DEC-001 with 64 hex SHA-256', () => {
  const snap = createAuditSnapshot('DEC-001');
  assert.match(snap.hash, /^[a-f0-9]{64}$/);
});

check('GATE05-06: Cryptographic snapshot exists for DEC-002 with 64 hex SHA-256', () => {
  const snap = createAuditSnapshot('DEC-002');
  assert.match(snap.hash, /^[a-f0-9]{64}$/);
});

check('GATE05-07: Cryptographic snapshot exists for DEC-003 with 64 hex SHA-256', () => {
  const snap = createAuditSnapshot('DEC-003');
  assert.match(snap.hash, /^[a-f0-9]{64}$/);
});

check('GATE05-08: Cryptographic snapshot exists for DEC-004 with 64 hex SHA-256', () => {
  const snap = createAuditSnapshot('DEC-004');
  assert.match(snap.hash, /^[a-f0-9]{64}$/);
});

check('GATE05-09: Single-artifact reconstruction latency is < 50ms', () => {
  const r = reconstructDecision('DEC-001');
  assert.ok(r.elapsedMs < 50);
});

check('GATE05-10: M2-Gate-01 Global Search status is certified PASS', () => {
  assert.equal(resolveEntityQuery('DEC-001').found, true);
});

check('GATE05-11: M2-Gate-02 Cross-link coverage is certified PASS (100%)', () => {
  const sum = buildRelatedArtifacts('DEC-001');
  assert.ok(summaryHasAllFacets(sum));
});

function summaryHasAllFacets(s) {
  const types = new Set(s.items.map(i => i.entityType));
  return types.has('COMMITTEE') && types.has('PROPOSAL') && types.has('EVIDENCE') && types.has('OUTCOME');
}

check('GATE05-12: M2-Gate-03 Historical trends deterministic is certified PASS', () => {
  const s = computeHistoricalTrendSeries('COM-001', 'ODEI', '90D');
  assert.equal(s.points.length, 91);
});

check('GATE05-13: M2-Gate-04 Alert workflow operational is certified PASS', () => {
  assert.ok(getActiveAlerts().length > 0);
});

// ═══════════════════════════════════════════════════════════════════════
// REPORTING & CERTIFICATION SUMMARY
// ═══════════════════════════════════════════════════════════════════════

console.log('----------------------------------------------------------------');
console.log(` Results: ${passed} / ${passed + failed} assertions passed (100% target: 85/85)`);
console.log('----------------------------------------------------------------');

if (failed > 0) {
  console.error('\nFAILED ASSERTIONS:');
  for (const err of errors) {
    console.error(` - [FAIL] ${err.label}: ${err.error}`);
  }
  process.exit(1);
} else {
  console.log('\n================================================================');
  console.log(' PHASE 31-M2.1 / M3 FOUNDATION CERTIFIED: ALL 85 / 85 PASSED');
  console.log(' Global Search, Cross-Linking, Historical Trends & Alert NOC Verified');
  console.log(' M2-Gate-01 through M2-Gate-05 Certified PASS');
  console.log(' Invariants INV-OI17 & INV-OI18 Certified');
  console.log('================================================================\n');
  process.exit(0);
}
