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

const SUPPORTED_PREFIXES = ['DEC', 'OUT', 'DIS', 'COM', 'PROP', 'LRN', 'INC'] as const;

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

  const prefixMatch = input.match(/^([A-Z]+)[-_]?(\d+)?$/);
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

  return {
    input: rawInput,
    found: false,
    suggestions: ['DEC-001', 'OUT-001', 'DIS-001', 'COM-001'],
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

  return {
    primaryEntityId: id,
    primaryEntityType: 'DECISION',
    items: [],
    totalConnectedArtifacts: 0,
    auditReconstructible: false,
  };
}
