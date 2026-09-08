/**
 * Phase 31-M2: Decision Network Intelligence Engine (Epic AI-002)
 *
 * Implements:
 * - INV-OI15: Cross-Committee Influence Integrity Invariant
 * - INV-OI16: Network Completeness and Explainability Invariant
 * - Network Topology Metrics (density, cycle detection, hub concentration)
 * - Pairwise Influence Heatmap Matrix
 * - Decision Audit Chronology and Timeline Builder
 * - Audit Trail Export Generator
 */

import {
  CommitteeNetworkNode,
  CommitteeNetworkEdge,
  NetworkMetrics,
  InfluenceMatrixEntry,
  InfluenceHeatmapMatrix,
  DecisionTimelineStep,
  AuditExplorerExport,
} from '../../types/committee-intelligence';

import {
  CANONICAL_COMMITTEES,
  CANONICAL_COMMITTEE_DECISIONS,
  CANONICAL_DISSENTS,
  CANONICAL_NETWORK_NODES,
  CANONICAL_NETWORK_EDGES,
} from './committeeIntelligenceEngine';

import {
  CANONICAL_PROPOSALS,
  CANONICAL_EVIDENCE_STORE,
  CANONICAL_OUTCOMES,
  CANONICAL_ATTRIBUTIONS,
  reconstructDecision,
  createAuditSnapshot,
} from './auditReconstructionEngine';

export const CANONICAL_INFLUENCE_ENTRIES: InfluenceMatrixEntry[] = [
  {
    sourceCommitteeId: 'COM-001',
    sourceCommitteeName: 'Investment Committee',
    targetCommitteeId: 'COM-003',
    targetCommitteeName: 'Risk & Capital Committee',
    influenceScore: 78.5,
    sharedDecisionCount: 22,
    alignmentPct: 91.2,
    rationale: 'Strategic alignment on high-momentum equity allocation vs tail-risk VaR hedging parameters.',
  },
  {
    sourceCommitteeId: 'COM-001',
    sourceCommitteeName: 'Investment Committee',
    targetCommitteeId: 'COM-002',
    targetCommitteeName: 'Governance Committee',
    influenceScore: 64.0,
    sharedDecisionCount: 16,
    alignmentPct: 87.5,
    rationale: 'Adherence to protected practice invariants and execution rulesets.',
  },
  {
    sourceCommitteeId: 'COM-003',
    sourceCommitteeName: 'Risk & Capital Committee',
    targetCommitteeId: 'COM-002',
    targetCommitteeName: 'Governance Committee',
    influenceScore: 58.2,
    sharedDecisionCount: 14,
    alignmentPct: 84.0,
    rationale: 'Capital allocation limits compliance and model audit sign-offs.',
  },
];

export function computeNetworkMetrics(
  nodes: CommitteeNetworkNode[] = CANONICAL_NETWORK_NODES,
  edges: CommitteeNetworkEdge[] = CANONICAL_NETWORK_EDGES
): NetworkMetrics {
  const totalNodes = nodes.length;
  const totalEdges = edges.length;

  const maxPossibleEdges = totalNodes > 1 ? totalNodes * (totalNodes - 1) : 1;
  const density = Math.round((totalEdges / maxPossibleEdges) * 1000) / 1000;

  const avgInfluence =
    edges.length > 0
      ? Math.round((edges.reduce((sum, e) => sum + e.influenceScore, 0) / edges.length) * 10) / 10
      : 0.0;

  const cycleResult = detectNetworkCycles(edges);

  const connectedNodeIds = new Set<string>();
  for (const edge of edges) {
    connectedNodeIds.add(edge.sourceCommitteeId);
    connectedNodeIds.add(edge.targetCommitteeId);
  }

  const disconnectedCount = nodes.filter(n => !connectedNodeIds.has(n.committeeId)).length;

  return {
    totalNodes,
    totalEdges,
    density,
    averageInfluenceScore: avgInfluence,
    cycleCount: cycleResult.cycles.length,
    disconnectedCount,
  };
}

export function detectNetworkCycles(edges: CommitteeNetworkEdge[] = CANONICAL_NETWORK_EDGES): {
  hasCycle: boolean;
  cycles: string[][];
  alerts: string[];
} {
  const adj = new Map<string, string[]>();
  for (const edge of edges) {
    const list = adj.get(edge.sourceCommitteeId) ?? [];
    list.push(edge.targetCommitteeId);
    adj.set(edge.sourceCommitteeId, list);
  }

  const visited = new Set<string>();
  const recStack = new Set<string>();
  const cycles: string[][] = [];
  const currentPath: string[] = [];

  function dfs(node: string) {
    visited.add(node);
    recStack.add(node);
    currentPath.push(node);

    const neighbors = adj.get(node) ?? [];
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        dfs(neighbor);
      } else if (recStack.has(neighbor)) {
        const cycleStartIndex = currentPath.indexOf(neighbor);
        if (cycleStartIndex !== -1) {
          cycles.push(currentPath.slice(cycleStartIndex).concat(neighbor));
        }
      }
    }

    currentPath.pop();
    recStack.delete(node);
  }

  for (const node of adj.keys()) {
    if (!visited.has(node)) {
      dfs(node);
    }
  }

  const alerts = cycles.map(
    cycle => `INFLUENCE_CYCLE_ALERT: Circular influence loop detected along ${cycle.join(' -> ')}`
  );

  return {
    hasCycle: cycles.length > 0,
    cycles,
    alerts,
  };
}

export function verifyInfluenceIntegrity(edges: CommitteeNetworkEdge[] = CANONICAL_NETWORK_EDGES): {
  valid: boolean;
  violations: string[];
} {
  const violations: string[] = [];

  for (const edge of edges) {
    if (edge.influenceScore < 0 || edge.influenceScore > 100 || Number.isNaN(edge.influenceScore)) {
      violations.push(`INVALID_INFLUENCE_SCORE: ${edge.sourceCommitteeId} -> ${edge.targetCommitteeId} score ${edge.influenceScore}`);
    }
    if (edge.sharedDecisionCount <= 0) {
      violations.push(`SPURIOUS_INFLUENCE_EDGE: ${edge.sourceCommitteeId} -> ${edge.targetCommitteeId} has zero shared decisions`);
    }
    if (edge.sourceCommitteeId === edge.targetCommitteeId) {
      violations.push(`SELF_REFERENTIAL_INFLUENCE: ${edge.sourceCommitteeId} references itself`);
    }
  }

  return {
    valid: violations.length === 0,
    violations,
  };
}

export function verifyNetworkCompleteness(
  nodes: CommitteeNetworkNode[] = CANONICAL_NETWORK_NODES,
  edges: CommitteeNetworkEdge[] = CANONICAL_NETWORK_EDGES
): {
  complete: boolean;
  completenessPct: number;
  violations: string[];
} {
  const violations: string[] = [];
  const metrics = computeNetworkMetrics(nodes, edges);

  if (metrics.disconnectedCount > 0) {
    violations.push(`ISOLATED_COMMITTEE_NODES: ${metrics.disconnectedCount} committee nodes have zero network edges`);
  }

  const nodeIds = new Set(nodes.map(n => n.committeeId));
  for (const edge of edges) {
    if (!nodeIds.has(edge.sourceCommitteeId)) {
      violations.push(`UNKNOWN_SOURCE_NODE: ${edge.sourceCommitteeId}`);
    }
    if (!nodeIds.has(edge.targetCommitteeId)) {
      violations.push(`UNKNOWN_TARGET_NODE: ${edge.targetCommitteeId}`);
    }
  }

  const completenessPct = nodes.length > 0
    ? Math.round(((nodes.length - metrics.disconnectedCount) / nodes.length) * 100)
    : 100;

  return {
    complete: violations.length === 0 && completenessPct === 100,
    completenessPct,
    violations,
  };
}

export function computeInfluenceHeatmap(
  nodes: CommitteeNetworkNode[] = CANONICAL_NETWORK_NODES,
  edges: CommitteeNetworkEdge[] = CANONICAL_NETWORK_EDGES
): InfluenceHeatmapMatrix {
  const committeeIds = nodes.map(n => n.committeeId);
  const committeeNames: Record<string, string> = {};
  for (const n of nodes) {
    committeeNames[n.committeeId] = n.committeeName;
  }

  const entries: InfluenceMatrixEntry[] = [];
  for (const edge of edges) {
    const matching = CANONICAL_INFLUENCE_ENTRIES.find(
      e => e.sourceCommitteeId === edge.sourceCommitteeId && e.targetCommitteeId === edge.targetCommitteeId
    );

    entries.push({
      sourceCommitteeId: edge.sourceCommitteeId,
      sourceCommitteeName: committeeNames[edge.sourceCommitteeId] ?? edge.sourceCommitteeId,
      targetCommitteeId: edge.targetCommitteeId,
      targetCommitteeName: committeeNames[edge.targetCommitteeId] ?? edge.targetCommitteeId,
      influenceScore: edge.influenceScore,
      sharedDecisionCount: edge.sharedDecisionCount,
      alignmentPct: matching?.alignmentPct ?? 85.0,
      rationale: matching?.rationale ?? `Cross-committee coordination with ${edge.sharedDecisionCount} shared approvals.`,
    });
  }

  const scores = entries.map(e => e.influenceScore);
  const maxInfluenceScore = scores.length > 0 ? Math.max(...scores) : 0;
  const minInfluenceScore = scores.length > 0 ? Math.min(...scores) : 0;

  return {
    committeeIds,
    committeeNames,
    entries,
    maxInfluenceScore,
    minInfluenceScore,
  };
}

export function buildDecisionTimeline(decisionId: string): DecisionTimelineStep[] {
  const recon = reconstructDecision(decisionId);
  const decision = CANONICAL_COMMITTEE_DECISIONS.find(d => d.decisionId === decisionId);

  if (!decision) {
    return [
      {
        stepId: 'STEP-MISSING',
        stepName: 'DECISION',
        title: 'Decision Record Missing',
        timestampUtc: new Date().toISOString(),
        status: 'FAILED',
        details: `No decision record identified for ID ${decisionId}`,
      },
    ];
  }

  const steps: DecisionTimelineStep[] = [];

  // Step 1: Proposal
  const proposal = recon.proposal ?? CANONICAL_PROPOSALS[decision.proposalId];
  steps.push({
    stepId: 'STEP-01',
    stepName: 'PROPOSAL',
    title: proposal ? `Proposal: ${proposal.title}` : 'Proposal Missing',
    timestampUtc: proposal?.createdAtUtc ?? decision.timestampUtc,
    status: proposal ? 'COMPLETED' : 'FAILED',
    actorId: proposal?.createdBy ?? 'SYSTEM',
    details: proposal?.businessObjective ?? 'No business objective recorded.',
    artifactId: decision.proposalId,
  });

  // Step 2: Evidence Gathering
  const evidenceCount = decision.evidenceIds.length;
  steps.push({
    stepId: 'STEP-02',
    stepName: 'EVIDENCE',
    title: `Evidence Vault: ${evidenceCount} Items Verified`,
    timestampUtc: decision.timestampUtc,
    status: evidenceCount > 0 ? 'COMPLETED' : 'WARNING',
    details: `Evidence identifiers: ${decision.evidenceIds.join(', ')}`,
    artifactId: decision.evidenceIds[0],
  });

  // Step 3: Quorum Review
  const chair = decision.participants.find(p => p.role === 'CHAIR');
  steps.push({
    stepId: 'STEP-03',
    stepName: 'QUORUM',
    title: `Committee Quorum (${decision.participants.length} Members)`,
    timestampUtc: decision.timestampUtc,
    status: chair && decision.participants.length >= 3 ? 'COMPLETED' : 'WARNING',
    actorId: chair?.userId,
    actorRole: 'CHAIR',
    details: `Review led by Chair ${chair?.userId ?? 'UNASSIGNED'} with ${decision.participants.filter(p => p.votingEligible).length} voting eligible members.`,
  });

  // Step 4: Dissent Logging
  const dissents = decision.dissents ?? [];
  if (decision.materialDecision || dissents.length > 0) {
    steps.push({
      stepId: 'STEP-04',
      stepName: 'DISSENT',
      title: dissents.length > 0 ? `Dissent Preserved: ${dissents[0].dissentId}` : 'No Material Dissent Filed',
      timestampUtc: dissents[0]?.timestampUtc ?? decision.timestampUtc,
      status: dissents.length > 0 ? 'COMPLETED' : 'WARNING',
      actorId: dissents[0]?.authorId,
      details: dissents[0]
        ? `Alternative View: "${dissents[0].alternativeRecommendation}" | Risk: "${dissents[0].riskAssessment}"`
        : 'Decision certified with zero material objections.',
      artifactId: dissents[0]?.dissentId,
    });
  }

  // Step 5: Final Decision
  steps.push({
    stepId: 'STEP-05',
    stepName: 'DECISION',
    title: `Formal Approval: ${decision.finalDecision}`,
    timestampUtc: decision.timestampUtc,
    status: decision.status === 'APPROVED' ? 'COMPLETED' : 'FAILED',
    details: `Decision Quality score rated at ${decision.decisionQuality} / 100.`,
    artifactId: decision.decisionId,
  });

  // Step 6: Measured Outcome
  const outcome = recon.outcome ?? (decision.outcomeId ? CANONICAL_OUTCOMES[decision.outcomeId] : undefined);
  if (outcome) {
    steps.push({
      stepId: 'STEP-06',
      stepName: 'OUTCOME',
      title: `Realized Outcome: +$${(outcome.realizedValueDollars / 1000).toFixed(0)}k Value`,
      timestampUtc: outcome.measuredAtUtc,
      status: 'COMPLETED',
      details: `Outcome Quality score measured at ${outcome.outcomeQualityScore} / 100.`,
      artifactId: outcome.outcomeId,
    });
  }

  // Step 7: Attribution
  const attribution = recon.attribution ?? (decision.outcomeId ? CANONICAL_ATTRIBUTIONS[decision.outcomeId] : undefined);
  if (attribution) {
    steps.push({
      stepId: 'STEP-07',
      stepName: 'ATTRIBUTION',
      title: `Institutional Attribution (${attribution.totalContributionPct}% Total)`,
      timestampUtc: outcome?.measuredAtUtc ?? decision.timestampUtc,
      status: attribution.totalContributionPct === 100.0 ? 'COMPLETED' : 'FAILED',
      details: `Individual: ${attribution.individualContributionPct}%, Team: ${attribution.teamContributionPct}%, Committee: ${attribution.committeeContributionPct}%, System: ${attribution.systemContributionPct}%`,
      artifactId: attribution.attributionId,
    });
  }

  return steps;
}

export function generateAuditExport(queryId: string): AuditExplorerExport {
  let decisionId = queryId;
  if (queryId.startsWith('OUT-')) {
    const dec = CANONICAL_COMMITTEE_DECISIONS.find(d => d.outcomeId === queryId);
    if (dec) decisionId = dec.decisionId;
  } else if (queryId.startsWith('PROP-')) {
    const dec = CANONICAL_COMMITTEE_DECISIONS.find(d => d.proposalId === queryId);
    if (dec) decisionId = dec.decisionId;
  } else if (queryId.startsWith('DIS-')) {
    const dis = CANONICAL_DISSENTS.find(d => d.dissentId === queryId);
    if (dis) decisionId = dis.decisionId;
  }

  const recon = reconstructDecision(decisionId);
  const auditSnapshot = createAuditSnapshot(decisionId);
  const timeline = buildDecisionTimeline(decisionId);

  return {
    exportedAtUtc: new Date().toISOString(),
    queryId,
    reconstructedDecisionId: decisionId,
    snapshotHash: auditSnapshot.hash,
    completenessPct: recon.coverage.completenessPct,
    missingArtifacts: recon.missingArtifacts,
    auditSnapshot,
    timeline,
  };
}
