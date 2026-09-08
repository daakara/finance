/**
 * Phase 29: Organizational Telemetry Service
 *
 * Implements the full 21-event taxonomy for Phase 29 Organizational Intelligence.
 * Every event extends the OrganizationalTelemetryEvent envelope.
 *
 * Categories:
 *   1. Knowledge Network (4 events)
 *   2. Organizational Learning (4 events)
 *   3. Benchmarking (4 events)
 *   4. Capability Attribution (4 events)
 *   5. Executive Intelligence (5 events)
 */

import type {
  OrganizationalTelemetryEvent,
  OrganizationalEventName,
} from '@/types/organizational-intelligence';

export const PHASE_29_RELEASE_TRAIN = 'PHASE_29' as const;

export const PHASE_29_FEATURE_AREAS = {
  KNOWLEDGE_NETWORK: 'knowledge_network',
  ORGANIZATIONAL_LEARNING: 'organizational_learning',
  BENCHMARKING: 'benchmarking',
  CAPABILITY_ATTRIBUTION: 'capability_attribution',
  EXECUTIVE_INTELLIGENCE: 'executive_intelligence',
} as const;

let _eventCounter = 0;

function generateEventId(): string {
  _eventCounter += 1;
  return `P29-EVT-${Date.now()}-${String(_eventCounter).padStart(4, '0')}`;
}

class OrganizationalTelemetryServiceImpl {
  private _events: OrganizationalTelemetryEvent[] = [];

  emit(
    eventName: OrganizationalEventName,
    userId: string,
    role: string,
    teamId: string,
    sessionId: string,
    featureArea: string,
    options: { confidence?: number; evidenceId?: string } = {}
  ): OrganizationalTelemetryEvent {
    const event: OrganizationalTelemetryEvent = {
      eventId: generateEventId(),
      eventName,
      userId,
      role,
      teamId,
      sessionId,
      timestampUtc: new Date().toISOString(),
      releaseTrain: PHASE_29_RELEASE_TRAIN,
      featureArea,
      ...options,
    };
    this._events.push(event);
    return event;
  }

  getEvents(): OrganizationalTelemetryEvent[] {
    return [...this._events];
  }

  getEventCount(): number {
    return this._events.length;
  }

  getEventsByName(name: OrganizationalEventName): OrganizationalTelemetryEvent[] {
    return this._events.filter(e => e.eventName === name);
  }

  flush(): OrganizationalTelemetryEvent[] {
    const events = [...this._events];
    this._events = [];
    return events;
  }
}

export const OrganizationalTelemetryService = new OrganizationalTelemetryServiceImpl();

// ---------------------------------------------------------------------------
// Typed emit helpers for each of the 21 events
// ---------------------------------------------------------------------------

export const orgTelemetry = {
  knowledgeNodeCreated: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('knowledge_node_created', userId, 'analyst', teamId, sessionId, PHASE_29_FEATURE_AREAS.KNOWLEDGE_NETWORK),

  knowledgeNodeLinked: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('knowledge_node_linked', userId, 'analyst', teamId, sessionId, PHASE_29_FEATURE_AREAS.KNOWLEDGE_NETWORK),

  knowledgePatternDetected: (userId: string, teamId: string, sessionId: string, confidence: number) =>
    OrganizationalTelemetryService.emit('knowledge_pattern_detected', userId, 'analyst', teamId, sessionId, PHASE_29_FEATURE_AREAS.KNOWLEDGE_NETWORK, { confidence }),

  knowledgeReuseDetected: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('knowledge_reuse_detected', userId, 'analyst', teamId, sessionId, PHASE_29_FEATURE_AREAS.KNOWLEDGE_NETWORK),

  bestPracticePublished: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('best_practice_published', userId, 'portfolio_manager', teamId, sessionId, PHASE_29_FEATURE_AREAS.ORGANIZATIONAL_LEARNING),

  bestPracticePropagated: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('best_practice_propagated', userId, 'system', teamId, sessionId, PHASE_29_FEATURE_AREAS.ORGANIZATIONAL_LEARNING),

  bestPracticeAdopted: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('best_practice_adopted', userId, 'analyst', teamId, sessionId, PHASE_29_FEATURE_AREAS.ORGANIZATIONAL_LEARNING),

  learningFeedViewed: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('learning_feed_viewed', userId, 'analyst', teamId, sessionId, PHASE_29_FEATURE_AREAS.ORGANIZATIONAL_LEARNING),

  teamBenchmarkViewed: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('team_benchmark_viewed', userId, 'leadership', teamId, sessionId, PHASE_29_FEATURE_AREAS.BENCHMARKING),

  teamComparisonOpened: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('team_comparison_opened', userId, 'leadership', teamId, sessionId, PHASE_29_FEATURE_AREAS.BENCHMARKING),

  peerCohortOpened: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('peer_cohort_opened', userId, 'portfolio_manager', teamId, sessionId, PHASE_29_FEATURE_AREAS.BENCHMARKING),

  topPerformerAnalyzed: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('top_performer_analyzed', userId, 'leadership', teamId, sessionId, PHASE_29_FEATURE_AREAS.BENCHMARKING),

  capabilityImpactViewed: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('capability_impact_viewed', userId, 'leadership', teamId, sessionId, PHASE_29_FEATURE_AREAS.CAPABILITY_ATTRIBUTION),

  capabilityAttributionGenerated: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('capability_attribution_generated', userId, 'system', teamId, sessionId, PHASE_29_FEATURE_AREAS.CAPABILITY_ATTRIBUTION),

  roiReportOpened: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('roi_report_opened', userId, 'leadership', teamId, sessionId, PHASE_29_FEATURE_AREAS.CAPABILITY_ATTRIBUTION),

  investmentPriorityViewed: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('investment_priority_viewed', userId, 'leadership', teamId, sessionId, PHASE_29_FEATURE_AREAS.CAPABILITY_ATTRIBUTION),

  executiveHomeViewed: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('executive_home_viewed', userId, 'leadership', teamId, sessionId, PHASE_29_FEATURE_AREAS.EXECUTIVE_INTELLIGENCE),

  executiveBriefingOpened: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('executive_briefing_opened', userId, 'leadership', teamId, sessionId, PHASE_29_FEATURE_AREAS.EXECUTIVE_INTELLIGENCE),

  organizationalHealthViewed: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('organizational_health_viewed', userId, 'leadership', teamId, sessionId, PHASE_29_FEATURE_AREAS.EXECUTIVE_INTELLIGENCE),

  organizationalReadinessViewed: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('organizational_readiness_viewed', userId, 'leadership', teamId, sessionId, PHASE_29_FEATURE_AREAS.EXECUTIVE_INTELLIGENCE),

  strategicOpportunityOpened: (userId: string, teamId: string, sessionId: string) =>
    OrganizationalTelemetryService.emit('strategic_opportunity_opened', userId, 'leadership', teamId, sessionId, PHASE_29_FEATURE_AREAS.EXECUTIVE_INTELLIGENCE),
};

