/**
 * Phase 31-M1.1: Audit Reconstruction Engine (Epic AI-001 / INV-OI13-A)
 *
 * Implements complete institutional chain reconstruction:
 * Proposal -> Evidence -> Participants -> Dissents -> Decision -> Outcome -> Attribution
 *
 * Guaranteed 100% coverage from any single artifact ID (Decision ID or Outcome ID).
 */

import { sha256 } from '../governance/sha256';
import {
  CommitteeProposal,
  CommitteeEvidence,
  CommitteeOutcome,
  CommitteeAttribution,
  CommitteeAuditSnapshot,
  AuditReconstructionRequest,
  AuditReconstructionResult,
  ReconstructionCoverage,
  CommitteeDecision,
  CommitteeDissent,
} from '../../types/committee-intelligence';
import {
  CANONICAL_COMMITTEE_DECISIONS,
  CANONICAL_DISSENTS,
} from './committeeIntelligenceEngine';

export const CANONICAL_PROPOSALS: Record<string, CommitteeProposal> = {
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

export const CANONICAL_EVIDENCE_STORE: Record<string, CommitteeEvidence> = {
  'EVD-FLOW-01': {
    evidenceId: 'EVD-FLOW-01',
    sourceType: 'ORDER_FLOW_TELEMETRY',
    sourceReference: 'darkpool_volume_zscore_v2',
    confidencePct: 96.5,
  },
  'EVD-MACRO-02': {
    evidenceId: 'EVD-MACRO-02',
    sourceType: 'MACRO_REGIME_INDICATOR',
    sourceReference: 'fed_liquidity_index_180d',
    confidencePct: 94.0,
  },
  'EVD-VCP-01': {
    evidenceId: 'EVD-VCP-01',
    sourceType: 'TECHNICAL_VCP_SCANNER',
    sourceReference: 'volatility_contraction_matrix',
    confidencePct: 92.8,
  },
  'EVD-GOV-01': {
    evidenceId: 'EVD-GOV-01',
    sourceType: 'GOVERNANCE_LEDGER',
    sourceReference: 'inv_oi11_protected_practice_registry',
    confidencePct: 99.0,
  },
  'EVD-AUDIT-02': {
    evidenceId: 'EVD-AUDIT-02',
    sourceType: 'COMPLIANCE_AUDIT_LOG',
    sourceReference: 'sec_17a4_compliance_hash',
    confidencePct: 100.0,
  },
  'EVD-VAR-01': {
    evidenceId: 'EVD-VAR-01',
    sourceType: 'RISK_ENGINE_VAR',
    sourceReference: 'cornish_fisher_expansion_99',
    confidencePct: 95.2,
  },
  'EVD-CORNISH-02': {
    evidenceId: 'EVD-CORNISH-02',
    sourceType: 'STATISTICAL_DISTRIBUTION',
    sourceReference: 'skewness_kurtosis_calibration',
    confidencePct: 93.5,
  },
  'EVD-DISSENT-01': {
    evidenceId: 'EVD-DISSENT-01',
    sourceType: 'DISSENT_COUNTER_EVIDENCE',
    sourceReference: 'short_term_repo_rate_spike',
    confidencePct: 91.0,
  },
  'EVD-DISSENT-02': {
    evidenceId: 'EVD-DISSENT-02',
    sourceType: 'DISSENT_COUNTER_EVIDENCE',
    sourceReference: 'market_breadth_divergence',
    confidencePct: 89.5,
  },
  'EVD-DISSENT-03': {
    evidenceId: 'EVD-DISSENT-03',
    sourceType: 'DISSENT_COUNTER_EVIDENCE',
    sourceReference: 'smallcap_slippage_backtest',
    confidencePct: 88.0,
  },
  'EVD-DISSENT-04': {
    evidenceId: 'EVD-DISSENT-04',
    sourceType: 'DISSENT_COUNTER_EVIDENCE',
    sourceReference: 'implied_volatility_skew_curve',
    confidencePct: 90.0,
  },
};

export const CANONICAL_OUTCOMES: Record<string, CommitteeOutcome> = {
  'OUT-001': {
    outcomeId: 'OUT-001',
    realizedValueDollars: 450000,
    outcomeQualityScore: 89.5,
    measuredAtUtc: '2026-09-08T11:00:00Z',
  },
  'OUT-002': {
    outcomeId: 'OUT-002',
    realizedValueDollars: 310000,
    outcomeQualityScore: 86.0,
    measuredAtUtc: '2026-09-08T11:30:00Z',
  },
  'OUT-003': {
    outcomeId: 'OUT-003',
    realizedValueDollars: 280000,
    outcomeQualityScore: 92.5,
    measuredAtUtc: '2026-09-08T12:00:00Z',
  },
  'OUT-004': {
    outcomeId: 'OUT-004',
    realizedValueDollars: 390000,
    outcomeQualityScore: 90.0,
    measuredAtUtc: '2026-09-08T12:30:00Z',
  },
};

export const CANONICAL_ATTRIBUTIONS: Record<string, CommitteeAttribution> = {
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
    capabilityIds: ['decision-simulator'],
    learningIds: ['LRN-VAR-04'],
    individualContributionPct: 20.0,
    teamContributionPct: 30.0,
    committeeContributionPct: 35.0,
    systemContributionPct: 15.0,
    totalContributionPct: 100.0,
  },
};

export function reconstructDecision(decisionId: string): AuditReconstructionResult {
  const start = Date.now();
  const decision = CANONICAL_COMMITTEE_DECISIONS.find(d => d.decisionId === decisionId);

  if (!decision) {
    return {
      decisionId,
      success: false,
      coverage: {
        proposalRecovered: false,
        evidenceRecovered: false,
        participantsRecovered: false,
        dissentsRecovered: false,
        outcomeRecovered: false,
        attributionRecovered: false,
        completenessPct: 0,
      },
      missingArtifacts: ['DECISION_RECORD_MISSING'],
      elapsedMs: Date.now() - start,
    };
  }

  const proposal = CANONICAL_PROPOSALS[decision.proposalId];
  const evidence = decision.evidenceIds
    .map(id => CANONICAL_EVIDENCE_STORE[id])
    .filter(Boolean);
  const participants = decision.participants;
  const dissents = decision.dissents ?? [];
  const outcome = decision.outcomeId ? CANONICAL_OUTCOMES[decision.outcomeId] : undefined;
  const attribution = decision.outcomeId ? CANONICAL_ATTRIBUTIONS[decision.outcomeId] : undefined;

  const missingArtifacts: string[] = [];
  if (!proposal) missingArtifacts.push('PROPOSAL_MISSING');
  if (evidence.length !== decision.evidenceIds.length) missingArtifacts.push('EVIDENCE_ITEMS_INCOMPLETE');
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

  const totalRequired = 6;
  const completenessPct = Math.round((recoveredCount / totalRequired) * 100);

  const coverage: ReconstructionCoverage = {
    proposalRecovered: Boolean(proposal),
    evidenceRecovered: evidence.length === decision.evidenceIds.length,
    participantsRecovered: Boolean(participants && participants.length > 0),
    dissentsRecovered: !decision.materialDecision || dissents.length > 0,
    outcomeRecovered: Boolean(outcome),
    attributionRecovered: Boolean(attribution),
    completenessPct,
  };

  return {
    decisionId,
    success: missingArtifacts.length === 0 && completenessPct === 100,
    coverage,
    proposal,
    evidence,
    participants,
    dissents,
    outcome,
    attribution,
    missingArtifacts,
    elapsedMs: Date.now() - start,
  };
}

export function reconstructOutcome(outcomeId: string): AuditReconstructionResult {
  const decision = CANONICAL_COMMITTEE_DECISIONS.find(d => d.outcomeId === outcomeId);
  if (!decision) {
    return {
      decisionId: 'UNKNOWN',
      success: false,
      coverage: {
        proposalRecovered: false,
        evidenceRecovered: false,
        participantsRecovered: false,
        dissentsRecovered: false,
        outcomeRecovered: false,
        attributionRecovered: false,
        completenessPct: 0,
      },
      missingArtifacts: [`OUTCOME_UNLINKED:${outcomeId}`],
      elapsedMs: 0,
    };
  }
  return reconstructDecision(decision.decisionId);
}

export function reconstructDissent(dissentId: string): {
  dissent: CommitteeDissent | undefined;
  decision: CommitteeDecision | undefined;
  evidence: CommitteeEvidence[];
  success: boolean;
} {
  const dissent = CANONICAL_DISSENTS.find(d => d.dissentId === dissentId);
  if (!dissent) {
    return { dissent: undefined, decision: undefined, evidence: [], success: false };
  }
  const decision = CANONICAL_COMMITTEE_DECISIONS.find(d => d.decisionId === dissent.decisionId);
  const evidence = dissent.evidenceIds.map(id => CANONICAL_EVIDENCE_STORE[id]).filter(Boolean);

  return {
    dissent,
    decision,
    evidence,
    success: Boolean(decision && evidence.length === dissent.evidenceIds.length),
  };
}

export function reconstructAttribution(outcomeId: string): {
  attribution: CommitteeAttribution | undefined;
  valid100PctSum: boolean;
  totalPct: number;
} {
  const attribution = CANONICAL_ATTRIBUTIONS[outcomeId];
  if (!attribution) {
    return { attribution: undefined, valid100PctSum: false, totalPct: 0 };
  }

  const total =
    attribution.individualContributionPct +
    attribution.teamContributionPct +
    attribution.committeeContributionPct +
    attribution.systemContributionPct;

  return {
    attribution,
    valid100PctSum: total === 100.0,
    totalPct: total,
  };
}

export function createAuditSnapshot(decisionId: string): CommitteeAuditSnapshot {
  const recon = reconstructDecision(decisionId);
  if (!recon.success || !recon.proposal) {
    throw new Error(`Cannot create snapshot for unverified decision: ${decisionId}`);
  }

  const proposalHash = sha256(JSON.stringify(recon.proposal));
  const evidenceHashes = (recon.evidence ?? []).map(e =>
    sha256(JSON.stringify(e))
  );
  const participantHashes = (recon.participants ?? []).map(p =>
    sha256(JSON.stringify(p))
  );
  const dissentHashes = (recon.dissents ?? []).map(d =>
    sha256(JSON.stringify(d))
  );
  const outcomeHash = recon.outcome
    ? sha256(JSON.stringify(recon.outcome))
    : undefined;
  const attributionHash = recon.attribution
    ? sha256(JSON.stringify(recon.attribution))
    : undefined;

  const masterPayload = {
    decisionId,
    proposalHash,
    evidenceHashes,
    participantHashes,
    dissentHashes,
    outcomeHash,
    attributionHash,
  };

  const hash = sha256(JSON.stringify(masterPayload));

  return {
    snapshotId: `SNP-${decisionId}`,
    committeeId: recon.decisionId,
    decisionId,
    capturedAtUtc: new Date().toISOString(),
    hash,
    proposalHash,
    evidenceHashes,
    participantHashes,
    dissentHashes,
    outcomeHash,
    attributionHash,
    reconstructionVersion: '1.0.0',
  };
}
