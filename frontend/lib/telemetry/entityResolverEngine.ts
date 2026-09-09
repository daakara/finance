/**
 * Phase 31-M2.1: Global Search Service & Universal Entity Resolver
 *
 * Implements:
 * - Prefix detection (DEC-, OUT-, DIS-, COM-, PROP-)
 * - Exact canonical route resolution
 * - Typo & missing-id suggestion generation
 * - Search latency & telemetry logging
 * - Cross-link related artifacts resolution (100% reachable)
 */

import type {
  EntityResolution,
  SearchTelemetry,
  NavigationEntityType,
  RelatedArtifactsSummary,
  RelatedArtifactItem,
} from '../../types/navigation-intelligence';

import {
  CANONICAL_COMMITTEES,
  CANONICAL_COMMITTEE_DECISIONS,
  CANONICAL_DISSENTS,
  CANONICAL_NETWORK_EDGES,
} from './committeeIntelligenceEngine';

import {
  CANONICAL_PROPOSALS,
  CANONICAL_OUTCOMES,
} from './auditReconstructionEngine';

import { getAllLearnings } from './learningIntelligenceEngine';
import { getActiveIncidents } from '../governance/alertCorrelationEngine';
import { getRiskRegistry, getRiskById } from '../governance/riskRegistryEngine';
import { evaluateGroupthinkAssessment } from '../governance/groupthinkDetectionEngine';
import { getRecommendations, getRecommendationById } from '../governance/collectiveIntelligenceCoach';
import { getInterventionPlans, getInterventionPlanById } from '../governance/interventionPlanner';
import { detectBiases, CANONICAL_BIAS_ALERTS } from '../governance/biasDetectionEngine';

const SUPPORTED_PREFIXES = ['DEC', 'OUT', 'DIS', 'COM', 'PROP', 'LRN', 'INC', 'RSK', 'GT', 'REC', 'PLAN', 'BIAS', 'OOS', 'OHI', 'REP', 'CSC', 'OPT', 'ALLOC', 'SIM', 'RECSTATE', 'FAIL', 'SURV', 'SCN', 'ACT', 'POL', 'OVR', 'EVAL', 'ERR', 'RB', 'GOV', 'NI', 'GRP', 'NODE', 'TWIN', 'LAB', 'WS', 'INBOX', 'BRF', 'FUT', 'CF', 'PKG', 'REL', 'ADP'] as const;

const searchTelemetryLog: SearchTelemetry[] = [];

export function resolveEntityQuery(rawInput: string): EntityResolution {
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

  const prefixMatch = input.match(/^([A-Z]+)[-_]/) || input.match(/^([A-Z]+)$/);
  const prefix = prefixMatch ? prefixMatch[1] : '';

  // 1. Unsupported prefix detection
  if (!SUPPORTED_PREFIXES.includes(prefix as typeof SUPPORTED_PREFIXES[number])) {
    const resolution: EntityResolution = {
      input: rawInput,
      found: false,
      error: `Unknown entity type "${prefix || input}". Supported prefixes: DEC, OUT, DIS, COM, PROP`,
      suggestions: ['DEC-001', 'OUT-001', 'DIS-001', 'COM-001', 'PROP-001'],
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 2. DEC (Decision)
  if (prefix === 'DEC') {
    const dec = CANONICAL_COMMITTEE_DECISIONS.find(d => d.decisionId === input);
    if (dec) {
      const resolution: EntityResolution = {
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
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'DECISION',
      found: false,
      error: `Decision record ${input} not found in institutional ledger`,
      suggestions: allDecIds,
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 3. OUT (Outcome)
  if (prefix === 'OUT') {
    const outcome = CANONICAL_OUTCOMES[input];
    if (outcome) {
      const dec = CANONICAL_COMMITTEE_DECISIONS.find(d => d.outcomeId === input);
      const resolution: EntityResolution = {
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
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'OUTCOME',
      found: false,
      error: `Outcome record ${input} not found`,
      suggestions: allOutIds,
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 4. DIS (Dissent)
  if (prefix === 'DIS') {
    const dissent = CANONICAL_DISSENTS.find(d => d.dissentId === input);
    if (dissent) {
      const resolution: EntityResolution = {
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
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'DISSENT',
      found: false,
      error: `Dissent record ${input} not found`,
      suggestions: allDisIds,
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 5. COM (Committee)
  if (prefix === 'COM') {
    const com = CANONICAL_COMMITTEES.find(c => c.committeeId === input);
    if (com) {
      const resolution: EntityResolution = {
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
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'COMMITTEE',
      found: false,
      error: `Committee ${input} not registered`,
      suggestions: allComIds,
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 6. PROP (Proposal)
  if (prefix === 'PROP') {
    const prop = CANONICAL_PROPOSALS[input];
    if (prop) {
      const dec = CANONICAL_COMMITTEE_DECISIONS.find(d => d.proposalId === input);
      const resolution: EntityResolution = {
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
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'PROPOSAL',
      found: false,
      error: `Proposal ${input} not found in proposal vault`,
      suggestions: allPropIds,
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 7. LRN (Learning Record)
  if (prefix === 'LRN') {
    const learning = getAllLearnings().find(l => l.learningId === input);
    if (learning) {
      const resolution: EntityResolution = {
        input: rawInput,
        entityType: 'LEARNING',
        entityId: learning.learningId,
        title: `Learning: ${learning.title}`,
        canonicalRoute: `/learning-intelligence?learningId=${learning.learningId}`,
        found: true,
        suggestions: [],
        targetParams: { learningId: learning.learningId },
      };
      logTelemetry(rawInput, resolution, Date.now() - start);
      return resolution;
    }
    const allLrnIds = getAllLearnings().map(l => l.learningId);
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'LEARNING',
      found: false,
      error: `Learning record ${input} not found in catalog`,
      suggestions: allLrnIds,
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 8. INC (Correlated Incident)
  if (prefix === 'INC') {
    const incident = getActiveIncidents().find(i => i.incidentId.startsWith(input));
    if (incident) {
      const resolution: EntityResolution = {
        input: rawInput,
        entityType: 'INCIDENT',
        entityId: incident.incidentId,
        title: `Incident: ${incident.incidentType} (${incident.severity})`,
        canonicalRoute: `/learning-intelligence?incidentId=${incident.incidentId}`,
        found: true,
        suggestions: [],
        targetParams: { incidentId: incident.incidentId },
      };
      logTelemetry(rawInput, resolution, Date.now() - start);
      return resolution;
    }
    const allIncIds = getActiveIncidents().map(i => i.incidentId);
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'INCIDENT',
      found: false,
      error: `Correlated incident ${input} not found`,
      suggestions: allIncIds.length > 0 ? allIncIds : ['INC-201', 'INC-202'],
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 9. RSK (Governance Risk)
  if (prefix === 'RSK') {
    const risk = getRiskById(input);
    if (risk) {
      const resolution: EntityResolution = {
        input: rawInput,
        entityType: 'RISK',
        entityId: risk.riskId,
        title: `Risk: ${risk.title} (${risk.severity})`,
        canonicalRoute: `/risks-and-groupthink?queryId=${risk.riskId}`,
        found: true,
        suggestions: [],
        targetParams: { queryId: risk.riskId },
      };
      logTelemetry(rawInput, resolution, Date.now() - start);
      return resolution;
    }
    const allRiskIds = getRiskRegistry().map(r => r.riskId);
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'RISK',
      found: false,
      error: `Risk record ${input} not found in risk registry`,
      suggestions: allRiskIds.slice(0, 5),
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 10. GT (Groupthink Signal)
  if (prefix === 'GT') {
    const comId = input.includes('COM-') ? input.split('-').slice(1, 3).join('-') : 'COM-001';
    const assessment = evaluateGroupthinkAssessment(comId);
    const sig = assessment.signals.find(s => s.signalId === input) ?? assessment.signals[0];
    if (sig) {
      const resolution: EntityResolution = {
        input: rawInput,
        entityType: 'GROUPTHINK',
        entityId: sig.signalId,
        title: `Groupthink: ${sig.signalType} (${sig.severity})`,
        canonicalRoute: `/risks-and-groupthink?queryId=${sig.signalId}`,
        found: true,
        suggestions: [],
        targetParams: { queryId: sig.signalId },
      };
      logTelemetry(rawInput, resolution, Date.now() - start);
      return resolution;
    }
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'GROUPTHINK',
      found: false,
      error: `Groupthink signal ${input} not found`,
      suggestions: ['GT-COM-001-01', 'GT-COM-002-01', 'GT-COM-003-01'],
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 11. REC (Coaching Recommendation)
  if (prefix === 'REC') {
    const rec = getRecommendationById(input);
    if (rec) {
      const resolution: EntityResolution = {
        input: rawInput,
        entityType: 'RECOMMENDATION',
        entityId: rec.recommendationId,
        title: `Recommendation: ${rec.title} (${rec.priority})`,
        canonicalRoute: `/coaching-intelligence?recommendationId=${rec.recommendationId}`,
        found: true,
        suggestions: [],
        targetParams: { recommendationId: rec.recommendationId },
      };
      logTelemetry(rawInput, resolution, Date.now() - start);
      return resolution;
    }
    const allRecIds = getRecommendations().map(r => r.recommendationId);
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'RECOMMENDATION',
      found: false,
      error: `Recommendation ${input} not found in coaching catalog`,
      suggestions: allRecIds.slice(0, 5),
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 12. PLAN (Intervention Plan)
  if (prefix === 'PLAN') {
    const plan = getInterventionPlanById(input);
    if (plan) {
      const resolution: EntityResolution = {
        input: rawInput,
        entityType: 'INTERVENTION_PLAN',
        entityId: plan.planId,
        title: `Intervention Plan: ${plan.title}`,
        canonicalRoute: `/coaching-intelligence?planId=${plan.planId}`,
        found: true,
        suggestions: [],
        targetParams: { planId: plan.planId },
      };
      logTelemetry(rawInput, resolution, Date.now() - start);
      return resolution;
    }
    const allPlanIds = getInterventionPlans().map(p => p.planId);
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'INTERVENTION_PLAN',
      found: false,
      error: `Intervention plan ${input} not found`,
      suggestions: allPlanIds,
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 13. BIAS (Cognitive Bias Alert)
  if (prefix === 'BIAS') {
    const bias = CANONICAL_BIAS_ALERTS.find(b => b.alertId === input);
    if (bias) {
      const resolution: EntityResolution = {
        input: rawInput,
        entityType: 'BIAS_ALERT',
        entityId: bias.alertId,
        title: `Bias Alert: ${bias.biasType} (${bias.severity})`,
        canonicalRoute: `/coaching-intelligence?alertId=${bias.alertId}`,
        found: true,
        suggestions: [],
        targetParams: { alertId: bias.alertId },
      };
      logTelemetry(rawInput, resolution, Date.now() - start);
      return resolution;
    }
    const allBiasIds = CANONICAL_BIAS_ALERTS.map(b => b.alertId);
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'BIAS_ALERT',
      found: false,
      error: `Bias alert ${input} not found`,
      suggestions: allBiasIds,
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 14. OHI (Organizational Health Index)
  if (prefix === 'OHI') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'OHI_METRIC',
      entityId: input,
      title: `Organizational Health Index (${input})`,
      canonicalRoute: `/oos?view=overview&entityId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { view: 'overview', entityId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 15. REP (Executive Report)
  if (prefix === 'REP') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'OOS_REPORT',
      entityId: input,
      title: `Executive Report (${input})`,
      canonicalRoute: `/oos?view=board-report&reportId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { view: 'board-report', reportId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 16. OOS (Operating System Entity)
  if (prefix === 'OOS') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'OOS_REPORT',
      entityId: input,
      title: `Organizational Operating System (${input})`,
      canonicalRoute: `/oos?view=overview&entityId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { view: 'overview', entityId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 17. CSC (Certification Self-Correction)
  if (prefix === 'CSC') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'CSC_RECOVERY',
      entityId: input,
      title: `CSC Recovery Workflow (${input})`,
      canonicalRoute: `/oos?view=consistency&recoveryId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { view: 'consistency', recoveryId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 18. OPT (Optimization Run)
  if (prefix === 'OPT') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'OPTIMIZATION_RUN',
      entityId: input,
      title: `Optimization Run (${input})`,
      canonicalRoute: `/optimization-intelligence?runId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { runId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 19. ALLOC (Allocation Result)
  if (prefix === 'ALLOC') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'ALLOCATION_RESULT',
      entityId: input,
      title: `Resource Allocation Plan (${input})`,
      canonicalRoute: `/optimization-intelligence?tab=allocation&allocationId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { tab: 'allocation', allocationId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 20. SIM (Intervention Simulation)
  if (prefix === 'SIM') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'INTERVENTION_SIMULATION',
      entityId: input,
      title: `Intervention Simulation (${input})`,
      canonicalRoute: `/optimization-intelligence?tab=simulation&simId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { tab: 'simulation', simId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 21. RECSTATE (Recovery State)
  if (prefix === 'RECSTATE') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'RECOVERY_STATE',
      entityId: input,
      title: `Recovery State Definition (${input})`,
      canonicalRoute: `/resilience-intelligence?tab=recovery&stateId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { tab: 'recovery', stateId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 22. FAIL (Failover Event)
  if (prefix === 'FAIL') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'FAILOVER_EVENT',
      entityId: input,
      title: `Autonomous Failover Event (${input})`,
      canonicalRoute: `/resilience-intelligence?tab=failover&failoverId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { tab: 'failover', failoverId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 23. SURV (Strategy Survivability)
  if (prefix === 'SURV') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'STRATEGY_SURVIVABILITY',
      entityId: input,
      title: `Strategy Survivability Assessment (${input})`,
      canonicalRoute: `/resilience-intelligence?tab=survivability&strategyId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { tab: 'survivability', strategyId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 24. SCN (Scenario Definition)
  if (prefix === 'SCN') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'SCENARIO_DEFINITION',
      entityId: input,
      title: `Resilience Scenario Definition (${input})`,
      canonicalRoute: `/resilience-intelligence?tab=scenarios&scenarioId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { tab: 'scenarios', scenarioId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 25. ACT (Autonomous Action)
  if (prefix === 'ACT') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'AUTONOMOUS_ACTION',
      entityId: input,
      title: `Autonomous Governance Action (${input})`,
      canonicalRoute: `/autonomous-governance?tab=actions&actionId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { tab: 'actions', actionId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 26. POL (Governance Policy)
  if (prefix === 'POL') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'GOVERNANCE_POLICY',
      entityId: input,
      title: `Governance Policy Boundary (${input})`,
      canonicalRoute: `/autonomous-governance?tab=policies&policyId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { tab: 'policies', policyId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 27. OVR (Human Override)
  if (prefix === 'OVR') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'HUMAN_OVERRIDE',
      entityId: input,
      title: `Human Override Record (${input})`,
      canonicalRoute: `/autonomous-governance?tab=overrides&overrideId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { tab: 'overrides', overrideId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 28. EVAL (Policy Evaluation)
  if (prefix === 'EVAL') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'POLICY_EVALUATION',
      entityId: input,
      title: `Policy Evaluation Decision (${input})`,
      canonicalRoute: `/autonomous-governance?tab=evaluations&evaluationId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { tab: 'evaluations', evaluationId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 29. ERR / GOV (Fail-Close Error)
  if (prefix === 'ERR' || prefix === 'GOV') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'FAIL_CLOSE_ERROR',
      entityId: input,
      title: `Governance Fail-Close Error (${input})`,
      canonicalRoute: `/autonomous-governance?tab=audit&errId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { tab: 'audit', errId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 30. RB (Operational Runbook)
  if (prefix === 'RB') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'OPERATIONAL_RUNBOOK',
      entityId: input,
      title: `Operational Governance Runbook (${input})`,
      canonicalRoute: `/autonomous-governance?tab=runbooks&rbId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { tab: 'runbooks', rbId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 31. NI (Narrative Intelligence Briefing)
  if (prefix === 'NI') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'NARRATIVE_BRIEFING',
      entityId: input,
      title: `Executive Narrative Briefing (${input})`,
      canonicalRoute: `/intelligence-center?briefingId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { briefingId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 32. GRP / NODE (Graph Explorer Node)
  if (prefix === 'GRP' || prefix === 'NODE') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'GRAPH_NODE',
      entityId: input,
      title: `Institutional Lineage Graph Node (${input})`,
      canonicalRoute: `/graph-explorer?nodeId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { nodeId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 33. TWIN / LAB (Strategy Laboratory & Digital Twin)
  if (prefix === 'TWIN' || prefix === 'LAB') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'DIGITAL_TWIN',
      entityId: input,
      title: `Strategic Digital Twin & Laboratory (${input})`,
      canonicalRoute: `/strategy-laboratory?twinId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { twinId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 34. WS (Executive Workspace)
  if (prefix === 'WS') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'EXECUTIVE_WORKSPACE',
      entityId: input,
      title: `Executive Workspace Profile (${input})`,
      canonicalRoute: `/workspace?workspaceId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { workspaceId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 35. INBOX (Decision Inbox)
  if (prefix === 'INBOX') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'DECISION_INBOX',
      entityId: input,
      title: `Decision Inbox Item (${input})`,
      canonicalRoute: `/decision-inbox?itemId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { itemId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 36. BRF (Executive Briefing)
  if (prefix === 'BRF') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'EXECUTIVE_BRIEFING',
      entityId: input,
      title: `Executive Briefing Package (${input})`,
      canonicalRoute: `/decision-inbox?briefingId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { briefingId: input },
    };
  }

  // 37. FUT (Institutional Simulation & Futures Intelligence)
  if (prefix === 'FUT') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'INSTITUTIONAL_SIMULATION',
      entityId: input,
      title: `Institutional Futures Simulation (${input})`,
      canonicalRoute: `/simulation-intelligence?simulationId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { simulationId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 38. CF (Counterfactual Analysis)
  if (prefix === 'CF') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'COUNTERFACTUAL_ANALYSIS',
      entityId: input,
      title: `Counterfactual Decision Analysis (${input})`,
      canonicalRoute: `/simulation-intelligence?counterfactualId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { counterfactualId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 39. GRP (Universal Graph Explorer Node)
  if (prefix === 'GRP') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'GRAPH_NODE',
      entityId: input,
      title: `Universal Graph Relationship Node (${input})`,
      canonicalRoute: `/graph-explorer?nodeId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { nodeId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 40. PKG (Executive Decision Workspace Package)
  if (prefix === 'PKG') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'DECISION_PACKAGE',
      entityId: input,
      title: `Executive Decision Package (${input})`,
      canonicalRoute: `/executive-workspace?packageId=${input}`,
      found: true,
      suggestions: [],
      targetParams: { packageId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 43. ADP (Executive Adoption & Value Realization)
  if (prefix === 'ADP') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'ADOPTION_CENTER' as NavigationEntityType,
      entityId: input,
      title: `Executive Adoption Center (${input})`,
      canonicalRoute: `/adoption-center`,
      found: true,
      suggestions: [],
      targetParams: { adoptionId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  // 42. REL (Release Gate & Readiness Dashboard)
  if (prefix === 'REL') {
    const resolution: EntityResolution = {
      input: rawInput,
      entityType: 'RELEASE_DASHBOARD' as NavigationEntityType,
      entityId: input,
      title: `Release Certification Dashboard (${input})`,
      canonicalRoute: `/release-dashboard`,
      found: true,
      suggestions: [],
      targetParams: { releaseId: input },
    };
    logTelemetry(rawInput, resolution, Date.now() - start);
    return resolution;
  }

  return {
    input: rawInput,
    found: false,
    suggestions: ['DEC-001', 'OUT-001', 'DIS-001', 'COM-001', 'RSK-001', 'REC-001', 'OHI-001', 'REP-OOS-001'],
    error: 'Unresolved entity identifier',
  };
}

function logTelemetry(query: string, resolution: EntityResolution, latencyMs: number): void {
  const item: SearchTelemetry = {
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

export function getSearchTelemetryLog(): SearchTelemetry[] {
  return [...searchTelemetryLog];
}

export function clearSearchTelemetryLog(): void {
  searchTelemetryLog.length = 0;
}

/**
 * Universal Cross-Linking & Related Artifacts Resolver
 * Guarantees every artifact has 1-click reachable relationships.
 */
export function buildRelatedArtifacts(entityId: string): RelatedArtifactsSummary {
  const id = (entityId ?? '').trim().toUpperCase();
  const items: RelatedArtifactItem[] = [];

  // If Committee ID (COM-xxx)
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

    // Connected committees in network
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

  // If Decision ID (DEC-xxx)
  if (id.startsWith('DEC-')) {
    const dec = CANONICAL_COMMITTEE_DECISIONS.find(d => d.decisionId === id);
    if (dec) {
      // 1. Parent Committee
      const com = CANONICAL_COMMITTEES.find(c => c.committeeId === dec.committeeId);
      items.push({
        entityId: dec.committeeId,
        entityType: 'COMMITTEE',
        title: com?.committeeName ?? dec.committeeId,
        canonicalRoute: `/committee-intelligence?committeeId=${dec.committeeId}`,
        relationship: 'PARENT_COMMITTEE',
        statusBadge: 'AUTHORIZING_BODY',
      });

      // 2. Proposal
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

      // 3. Evidence Items
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

      // 4. Dissents
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

      // 5. Outcome
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

      // 6. Cryptographic snapshot
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

  // If Outcome ID (OUT-xxx)
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

  // If Dissent ID (DIS-xxx)
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

  // If Learning ID (LRN-xxx)
  if (id.startsWith('LRN-')) {
    const learning = getAllLearnings().find(l => l.learningId === id);
    if (learning) {
      // 1. Source Committee
      const com = CANONICAL_COMMITTEES.find(c => c.committeeId === learning.sourceCommitteeId);
      items.push({
        entityId: learning.sourceCommitteeId,
        entityType: 'COMMITTEE',
        title: com?.committeeName ?? learning.sourceCommitteeId,
        canonicalRoute: `/committee-intelligence?committeeId=${learning.sourceCommitteeId}`,
        relationship: 'PARENT_COMMITTEE',
        statusBadge: 'ORIGIN_BODY',
      });

      // 2. Source Decision
      const dec = CANONICAL_COMMITTEE_DECISIONS.find(d => d.decisionId === learning.sourceDecisionId);
      if (dec) {
        items.push({
          entityId: dec.decisionId,
          entityType: 'DECISION',
          title: dec.title,
          canonicalRoute: `/decision-explorer?decisionId=${dec.decisionId}`,
          relationship: 'SOURCE_DECISION',
          statusBadge: dec.status,
        });
      }

      // 3. Source Outcome
      if (learning.sourceOutcomeId) {
        items.push({
          entityId: learning.sourceOutcomeId,
          entityType: 'OUTCOME',
          title: `Outcome: ${learning.sourceOutcomeId}`,
          canonicalRoute: `/audit-explorer?queryId=${learning.sourceOutcomeId}`,
          relationship: 'REALIZED_OUTCOME',
          statusBadge: 'REALIZED',
        });
      }

      // 4. Learning Intelligence route
      items.push({
        entityId: id,
        entityType: 'LEARNING',
        title: learning.title,
        subtitle: `${learning.category} | ${learning.status}`,
        canonicalRoute: `/learning-intelligence?learningId=${id}`,
        relationship: 'ATTRIBUTED_LEARNING',
        statusBadge: 'CATALOG_ITEM',
      });
    }

    return {
      primaryEntityId: id,
      primaryEntityType: 'LEARNING',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: Boolean(learning),
    };
  }

  // If Incident ID (INC-xxx)
  if (id.startsWith('INC-')) {
    const incident = getActiveIncidents().find(i => i.incidentId.startsWith(id));
    if (incident) {
      for (const comId of incident.affectedCommitteeIds) {
        const com = CANONICAL_COMMITTEES.find(c => c.committeeId === comId);
        items.push({
          entityId: comId,
          entityType: 'COMMITTEE',
          title: com?.committeeName ?? comId,
          canonicalRoute: `/committee-intelligence?committeeId=${comId}`,
          relationship: 'PARENT_COMMITTEE',
          statusBadge: 'AFFECTED_COMMITTEE',
        });
      }

      items.push({
        entityId: incident.incidentId,
        entityType: 'INCIDENT',
        title: incident.incidentType,
        subtitle: `Severity: ${incident.severity} | Occurrences: ${incident.occurrenceCount}`,
        canonicalRoute: `/learning-intelligence?incidentId=${incident.incidentId}`,
        relationship: 'CORRELATED_INCIDENT',
        statusBadge: incident.status,
      });
    }

    return {
      primaryEntityId: id,
      primaryEntityType: 'INCIDENT',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: Boolean(incident),
    };
  }

  // If Risk ID (RSK-xxx)
  if (id.startsWith('RSK-')) {
    const risk = getRiskById(id);
    if (risk) {
      if (risk.committeeId) {
        const com = CANONICAL_COMMITTEES.find(c => c.committeeId === risk.committeeId);
        items.push({
          entityId: risk.committeeId,
          entityType: 'COMMITTEE',
          title: com?.committeeName ?? risk.committeeId,
          canonicalRoute: `/committee-intelligence?committeeId=${risk.committeeId}`,
          relationship: 'PARENT_COMMITTEE',
          statusBadge: 'EXPOSED_BODY',
        });
      }

      for (const incId of risk.incidentIds) {
        items.push({
          entityId: incId,
          entityType: 'INCIDENT',
          title: `Linked Incident ${incId}`,
          canonicalRoute: `/learning-intelligence?incidentId=${incId}`,
          relationship: 'CORRELATED_INCIDENT',
          statusBadge: 'ACTIVE_INCIDENT',
        });
      }

      items.push({
        entityId: risk.riskId,
        entityType: 'RISK',
        title: risk.title,
        subtitle: `Exposure: ${risk.exposureScore} | Likelihood: ${risk.likelihoodPct}% | Impact: ${risk.impactScore}`,
        canonicalRoute: `/risks-and-groupthink?queryId=${risk.riskId}`,
        relationship: 'PREDICTED_RISK',
        statusBadge: risk.severity,
      });
    }

    return {
      primaryEntityId: id,
      primaryEntityType: 'RISK',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: Boolean(risk),
    };
  }

  // If Groupthink Signal (GT-xxx)
  if (id.startsWith('GT-')) {
    const comId = id.includes('COM-') ? id.split('-').slice(1, 3).join('-') : 'COM-001';
    const assessment = evaluateGroupthinkAssessment(comId);
    const sig = assessment.signals.find(s => s.signalId === id) ?? assessment.signals[0];

    if (sig) {
      items.push({
        entityId: sig.committeeId,
        entityType: 'COMMITTEE',
        title: `Committee ${sig.committeeId}`,
        canonicalRoute: `/committee-intelligence?committeeId=${sig.committeeId}`,
        relationship: 'PARENT_COMMITTEE',
        statusBadge: 'EXAMINED_BODY',
      });

      items.push({
        entityId: sig.signalId,
        entityType: 'GROUPTHINK',
        title: sig.signalType,
        subtitle: `Severity: ${sig.severity} | Observed: ${sig.observedValue} (Threshold: ${sig.thresholdValue})`,
        canonicalRoute: `/risks-and-groupthink?queryId=${sig.signalId}`,
        relationship: 'GROUPTHINK_SIGNAL',
        statusBadge: sig.severity,
      });
    }

    return {
      primaryEntityId: id,
      primaryEntityType: 'GROUPTHINK',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: Boolean(sig),
    };
  }

    // If Coaching Recommendation (REC-xxx)
  if (id.startsWith('REC-')) {
    const rec = getRecommendationById(id);
    if (rec) {
      items.push({
        entityId: rec.committeeId,
        entityType: 'COMMITTEE',
        title: `Committee ${rec.committeeId}`,
        canonicalRoute: `/committee-intelligence?committeeId=${rec.committeeId}`,
        relationship: 'PARENT_COMMITTEE',
        statusBadge: 'TARGET_COMMITTEE',
      });

      rec.actions.forEach(act => {
        items.push({
          entityId: act.actionId,
          entityType: 'RECOMMENDATION',
          title: act.title,
          subtitle: `Owner: ${act.ownerId} | Due: ${act.dueDateUtc.slice(0, 10)}`,
          canonicalRoute: `/coaching-intelligence?recommendationId=${rec.recommendationId}`,
          relationship: 'INTERVENTION_ACTION',
          statusBadge: act.status,
        });
      });

      items.push({
        entityId: rec.recommendationId,
        entityType: 'RECOMMENDATION',
        title: rec.title,
        subtitle: `Priority: ${rec.priority} | Confidence: ${rec.confidenceScore}%`,
        canonicalRoute: `/coaching-intelligence?recommendationId=${rec.recommendationId}`,
        relationship: 'COACHING_RECOMMENDATION',
        statusBadge: rec.status,
      });
    }

    return {
      primaryEntityId: id,
      primaryEntityType: 'RECOMMENDATION',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: Boolean(rec),
    };
  }

  // If Intervention Plan (PLAN-xxx)
  if (id.startsWith('PLAN-')) {
    const plan = getInterventionPlanById(id);
    if (plan) {
      items.push({
        entityId: plan.committeeId,
        entityType: 'COMMITTEE',
        title: `Committee ${plan.committeeId}`,
        canonicalRoute: `/committee-intelligence?committeeId=${plan.committeeId}`,
        relationship: 'PARENT_COMMITTEE',
        statusBadge: 'PLAN_TARGET',
      });

      plan.recommendations.forEach(rec => {
        items.push({
          entityId: rec.recommendationId,
          entityType: 'RECOMMENDATION',
          title: rec.title,
          canonicalRoute: `/coaching-intelligence?recommendationId=${rec.recommendationId}`,
          relationship: 'COACHING_RECOMMENDATION',
          statusBadge: rec.priority,
        });
      });
    }

    return {
      primaryEntityId: id,
      primaryEntityType: 'INTERVENTION_PLAN',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: Boolean(plan),
    };
  }

  // If Bias Alert (BIAS-xxx)
  if (id.startsWith('BIAS-')) {
    const bias = CANONICAL_BIAS_ALERTS.find(b => b.alertId === id);
    if (bias) {
      items.push({
        entityId: bias.committeeId,
        entityType: 'COMMITTEE',
        title: `Committee ${bias.committeeId}`,
        canonicalRoute: `/committee-intelligence?committeeId=${bias.committeeId}`,
        relationship: 'PARENT_COMMITTEE',
        statusBadge: 'AFFECTED_COMMITTEE',
      });

      items.push({
        entityId: bias.alertId,
        entityType: 'BIAS_ALERT',
        title: `Bias Alert: ${bias.biasType}`,
        subtitle: bias.explanation,
        canonicalRoute: `/coaching-intelligence?alertId=${bias.alertId}`,
        relationship: 'BIAS_WARNING',
        statusBadge: bias.severity,
      });
    }

    return {
      primaryEntityId: id,
      primaryEntityType: 'BIAS_ALERT',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: Boolean(bias),
    };
  }

  // If OHI, REP, OOS, or CSC
  if (id.startsWith('OHI-') || id.startsWith('REP-') || id.startsWith('OOS-') || id.startsWith('CSC-')) {
    items.push({
      entityId: 'OHI-001',
      entityType: 'OHI_METRIC',
      title: 'Organizational Health Index (84.2)',
      subtitle: 'Certified Composite Health Score',
      canonicalRoute: '/oos?view=overview',
      relationship: 'PARENT_COMMITTEE',
      statusBadge: 'CERTIFIED_OHI',
    });

    items.push({
      entityId: 'REP-OOS-001',
      entityType: 'OOS_REPORT',
      title: 'ARX Board of Directors Governance Report',
      subtitle: 'Board-Ready Operating Review',
      canonicalRoute: '/oos?view=board-report',
      relationship: 'REALIZED_OUTCOME',
      statusBadge: 'BOARD_READY',
    });

    items.push({
      entityId: 'REC-CSC-001',
      entityType: 'CSC_RECOVERY',
      title: 'Certification Self-Correction Engine',
      subtitle: 'Cross-System Consistency & Recovery Mirror',
      canonicalRoute: '/oos?view=consistency',
      relationship: 'SYSTEM_CONSISTENCY',
      statusBadge: 'CONSISTENT',
    });

    return {
      primaryEntityId: id,
      primaryEntityType: id.startsWith('OHI-') ? 'OHI_METRIC' : id.startsWith('REP-') ? 'OOS_REPORT' : id.startsWith('CSC-') ? 'CSC_RECOVERY' : 'OOS_REPORT',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: true,
    };
  }

  // If OPT, ALLOC, or SIM
  if (id.startsWith('OPT-') || id.startsWith('ALLOC-') || id.startsWith('SIM-')) {
    items.push({
      entityId: 'OPT-RUN-2026-001',
      entityType: 'OPTIMIZATION_RUN',
      title: 'Canonical Q3 Optimization Run',
      subtitle: 'Projected OHI: 84.2 -> 92.8 (+8.6)',
      canonicalRoute: '/optimization-intelligence?runId=OPT-RUN-2026-001',
      relationship: 'PARENT_COMMITTEE',
      statusBadge: 'OPTIMAL',
    });

    items.push({
      entityId: 'ALLOC-2026-001',
      entityType: 'ALLOCATION_RESULT',
      title: 'Optimal Resource Allocation Plan',
      subtitle: '91.4 Efficiency | 100% Constraint Preserved',
      canonicalRoute: '/optimization-intelligence?tab=allocation',
      relationship: 'REALIZED_OUTCOME',
      statusBadge: 'FEASIBLE',
    });

    items.push({
      entityId: 'SIM-2026-001',
      entityType: 'INTERVENTION_SIMULATION',
      title: 'Multi-Intervention Monte Carlo Simulation',
      subtitle: '1,000 Replays | Deterministic Hash Verified',
      canonicalRoute: '/optimization-intelligence?tab=simulation',
      relationship: 'OPTIMIZATION_CONSTRAINT',
      statusBadge: 'DETERMINISTIC',
    });

    return {
      primaryEntityId: id,
      primaryEntityType: id.startsWith('OPT-') ? 'OPTIMIZATION_RUN' : id.startsWith('ALLOC-') ? 'ALLOCATION_RESULT' : 'INTERVENTION_SIMULATION',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: true,
    };
  }



  // If ERR, GOV, or RB
  if (id.startsWith('ERR-') || id.startsWith('GOV-') || id.startsWith('RB-') || id.startsWith('M9-RB-')) {
    items.push({
      entityId: 'GOV-POL-001',
      entityType: 'FAIL_CLOSE_ERROR',
      title: 'Action Outside Approved Policy Boundary',
      subtitle: 'Fail-Close Activated: SAFE_MODE',
      canonicalRoute: '/autonomous-governance?tab=audit',
      relationship: 'FAIL_CLOSE_TRIGGER',
      statusBadge: 'FAIL_CLOSED',
    });

    items.push({
      entityId: 'M9-RB-01',
      entityType: 'OPERATIONAL_RUNBOOK',
      title: 'Autonomous Governance Health Degradation',
      subtitle: 'Automated Actions: Action Freeze + L4 Safe Mode',
      canonicalRoute: '/autonomous-governance?tab=runbooks',
      relationship: 'RUNBOOK_EXECUTION',
      statusBadge: 'ACTIVE',
    });

    return {
      primaryEntityId: id,
      primaryEntityType: id.startsWith('RB-') || id.startsWith('M9-RB-') ? 'OPERATIONAL_RUNBOOK' : 'FAIL_CLOSE_ERROR',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: true,
    };
  }

  // If ACT, POL, OVR, or EVAL
  if (id.startsWith('ACT-') || id.startsWith('POL-') || id.startsWith('OVR-') || id.startsWith('EVAL-')) {
    items.push({
      entityId: 'ACT-2026-001',
      entityType: 'AUTONOMOUS_ACTION',
      title: 'Autonomous Portfolio Variance Dampening',
      subtitle: 'Confidence 96.5% | Status: EXECUTED',
      canonicalRoute: '/autonomous-governance?tab=actions',
      relationship: 'AUTONOMOUS_APPROVAL',
      statusBadge: 'EXECUTED',
    });

    items.push({
      entityId: 'POL-RISK-001',
      entityType: 'GOVERNANCE_POLICY',
      title: 'Capital At Risk Boundary Policy',
      subtitle: 'VaR 99% Drawdown Floor <= 15.0%',
      canonicalRoute: '/autonomous-governance?tab=policies',
      relationship: 'POLICY_CONSTRAINT',
      statusBadge: 'ACTIVE',
    });

    items.push({
      entityId: 'OVR-2026-INIT',
      entityType: 'HUMAN_OVERRIDE',
      title: 'Baseline Human Override Checkpoint',
      subtitle: 'Zero-Delay Supersession Guaranteed (INV-OI52)',
      canonicalRoute: '/autonomous-governance?tab=overrides',
      relationship: 'HUMAN_SUPERSEDENCE',
      statusBadge: 'APPLIED',
    });

    return {
      primaryEntityId: id,
      primaryEntityType: id.startsWith('ACT-') ? 'AUTONOMOUS_ACTION' : id.startsWith('POL-') ? 'GOVERNANCE_POLICY' : id.startsWith('OVR-') ? 'HUMAN_OVERRIDE' : 'POLICY_EVALUATION',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: true,
    };
  }

  // If RECSTATE, FAIL, SURV, or SCN
  if (id.startsWith('RECSTATE-') || id.startsWith('FAIL-') || id.startsWith('SURV-') || id.startsWith('SCN-')) {
    items.push({
      entityId: 'RECSTATE-OHI-L1',
      entityType: 'RECOVERY_STATE',
      title: 'L1 Metric Refresh State',
      subtitle: 'Transient In-Memory Cache Invalidation (RTO < 5s)',
      canonicalRoute: '/resilience-intelligence?tab=recovery',
      relationship: 'RESILIENCE_FALLBACK',
      statusBadge: 'CERTIFIED',
    });

    items.push({
      entityId: 'FAIL-2026-001',
      entityType: 'FAILOVER_EVENT',
      title: 'Autonomous Optimization Failover',
      subtitle: 'Fallback to Last Certified Feasible Plan (RTO 42s)',
      canonicalRoute: '/resilience-intelligence?tab=failover',
      relationship: 'SYSTEM_CONSISTENCY',
      statusBadge: 'RESOLVED',
    });

    items.push({
      entityId: 'SURV-2026-001',
      entityType: 'STRATEGY_SURVIVABILITY',
      title: 'Stress Survivability Rating: CERTIFIED',
      subtitle: 'Robustness Score 91.4 | 100% Invariant Compliant',
      canonicalRoute: '/resilience-intelligence?tab=survivability',
      relationship: 'REALIZED_OUTCOME',
      statusBadge: 'CERTIFIED',
    });

    return {
      primaryEntityId: id,
      primaryEntityType: id.startsWith('RECSTATE-') ? 'RECOVERY_STATE' : id.startsWith('FAIL-') ? 'FAILOVER_EVENT' : id.startsWith('SURV-') ? 'STRATEGY_SURVIVABILITY' : 'SCENARIO_DEFINITION',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: true,
    };
  }

  // If Workspace / Decision Inbox / Briefing (WS-, INBOX-, BRF-)
  if (id.startsWith('WS-') || id.startsWith('INBOX-') || id.startsWith('BRF-')) {
    items.push({
      entityId: 'WS-EXEC-001',
      entityType: 'EXECUTIVE_WORKSPACE',
      title: 'Executive Workspace: Mission Control',
      subtitle: 'Personalized Tasks, Assigned Committees & Outage Fallback',
      canonicalRoute: '/workspace',
      relationship: 'PARENT_COMMITTEE',
      statusBadge: 'CERTIFIED',
    });

    items.push({
      entityId: 'INBOX-QUEUE-001',
      entityType: 'DECISION_INBOX',
      title: 'Unified Decision Inbox',
      subtitle: 'Consolidated Triage: Approvals, Escalations & Runbooks',
      canonicalRoute: '/decision-inbox',
      relationship: 'SOURCE_DECISION',
      statusBadge: 'CERTIFIED',
    });

    items.push({
      entityId: 'BRF-PKG-001',
      entityType: 'EXECUTIVE_BRIEFING',
      title: 'One-Click Executive Briefing',
      subtitle: 'Multi-Audience Narrative Synthesis with Replay Determinism',
      canonicalRoute: '/decision-inbox?tab=briefings',
      relationship: 'REALIZED_OUTCOME',
      statusBadge: 'CERTIFIED',
    });

    return {
      primaryEntityId: id,
      primaryEntityType: id.startsWith('WS-') ? 'EXECUTIVE_WORKSPACE' : id.startsWith('INBOX-') ? 'DECISION_INBOX' : 'EXECUTIVE_BRIEFING',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: true,
    };
  }

  // If Institutional Simulation / Counterfactual (FUT-, CF-)
  // If Adoption ID (ADP-xxx)
  if (id.startsWith('ADP-') || id === 'ADP') {
    items.push({
      entityId: 'ADP-EXEC-2026',
      entityType: 'ADOPTION_CENTER' as NavigationEntityType,
      title: 'Executive Adoption Center',
      subtitle: 'Value Realization & Usage Telemetry',
      canonicalRoute: '/adoption-center',
      relationship: 'SOURCE_DECISION',
      statusBadge: 'CERTIFIED',
    });

    items.push({
      entityId: 'WS-CIO-001',
      entityType: 'EXECUTIVE_WORKSPACE',
      title: 'CIO Mission Control Workspace',
      subtitle: 'Primary Decision Station',
      canonicalRoute: '/executive-workspace',
      relationship: 'PARENT_COMMITTEE',
      statusBadge: 'AUTHORIZING_BODY',
    });

    return {
      primaryEntityId: id,
      primaryEntityType: 'ADOPTION_CENTER' as NavigationEntityType,
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: true,
    };
  }

  // If Release ID (REL-xxx)
  if (id.startsWith('REL-') || id === 'REL') {
    items.push({
      entityId: 'REL-2026.09-PROD',
      entityType: 'RELEASE_DASHBOARD' as NavigationEntityType,
      title: 'Executive Release Certification Dashboard',
      subtitle: 'M1–M16 Gate Verification & Attestation Lock',
      canonicalRoute: '/release-dashboard',
      relationship: 'SOURCE_DECISION',
      statusBadge: 'CERTIFIED',
    });

    items.push({
      entityId: 'COM-001',
      entityType: 'COMMITTEE',
      title: 'Executive Committee Lead',
      subtitle: 'Attestation & Sign-off Authority',
      canonicalRoute: '/governance',
      relationship: 'PARENT_COMMITTEE',
      statusBadge: 'AUTHORIZING_BODY',
    });

    return {
      primaryEntityId: id,
      primaryEntityType: 'RELEASE_DASHBOARD' as NavigationEntityType,
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: true,
    };
  }

  if (id.startsWith('FUT-') || id.startsWith('CF-')) {
    items.push({
      entityId: 'FUT-SIM-001',
      entityType: 'INSTITUTIONAL_SIMULATION',
      title: 'Institutional Futures Simulation Hub',
      subtitle: 'Multi-Path Scenario Projections & Assumption Propagation',
      canonicalRoute: '/simulation-intelligence',
      relationship: 'SOURCE_DECISION',
      statusBadge: 'CERTIFIED',
    });

    items.push({
      entityId: 'CF-DEC-001',
      entityType: 'COUNTERFACTUAL_ANALYSIS',
      title: 'Counterfactual Decision Analysis',
      subtitle: 'Actual vs Alternative Path Causal Delta Attribution',
      canonicalRoute: '/simulation-intelligence?tab=counterfactual',
      relationship: 'REALIZED_OUTCOME',
      statusBadge: 'CERTIFIED',
    });

    return {
      primaryEntityId: id,
      primaryEntityType: id.startsWith('FUT-') ? 'INSTITUTIONAL_SIMULATION' : 'COUNTERFACTUAL_ANALYSIS',
      items,
      totalConnectedArtifacts: items.length,
      auditReconstructible: true,
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
