/**
 * ARX Terminal — Unified P1 Denominator, Eligibility, Exclusion, and Natural-Traffic Decision Engine.
 *
 * Implements Section 9 of the ARX Terminal Governance Framework:
 * - 9.1: Canonical observation unit (ETF_INTENT_ATTEMPT)
 * - 9.2: Canonical accounting states (INVALID > QUARANTINED > EXCLUDED > VALID)
 * - 9.3: 15-step canonical decision algorithm
 * - 9.4: Structural validation
 * - 9.5: Required provenance and unresolved facts handling
 * - 9.6: Release identity decision rules
 * - 9.7: Synthetic markers & traffic classification
 * - 9.8: Replay & duplicate detection
 * - 9.9: Positive natural-production classification (N1–N8)
 * - 9.10: Canonical natural-traffic decision procedure
 * - 9.11: Epoch and boundary validation
 * - 9.12: Denominator eligibility conditions (E1–E10)
 * - 9.13: Canonical denominator accounting
 * - 9.14: Canonical exclusion taxonomy
 * - 9.15: Canonical classification table
 * - 9.16: Contradictory signals handling
 * - 9.17: Failure-path non-exclusion rule
 * - 9.18: Accounting conservation
 * - 9.19: Required 16 audit fields
 */

export type PrimaryAccountingState = "INVALID" | "QUARANTINED" | "EXCLUDED" | "VALID";

export type TrafficClass =
  | "NATURAL_PRODUCTION"
  | "SYNTHETIC_E2E"
  | "MANUAL_QA"
  | "DEVELOPER"
  | "CI"
  | "UPTIME_PROBE"
  | "MONITORING"
  | "BOT"
  | "CRAWLER"
  | "REPLAY"
  | "DUPLICATE"
  | "NON_PRODUCTION"
  | "NON_AUTHORIZED_RELEASE"
  | "UNKNOWN"
  | "NOT_APPLICABLE";

export type ExclusionCode =
  | "SYNTHETIC_E2E"
  | "MANUAL_QA"
  | "DEVELOPER"
  | "CI"
  | "UPTIME_PROBE"
  | "MONITORING"
  | "BOT"
  | "CRAWLER"
  | "REPLAY"
  | "DUPLICATE"
  | "LOCAL"
  | "PREVIEW_DEPLOYMENT"
  | "NON_PRODUCTION"
  | "WRONG_RELEASE"
  | "OUTSIDE_EPOCH"
  | "NON_ETF_INTENT"
  | "UNSUPPORTED_INPUT"
  | "NOT_APPLICABLE";

export type QuarantineReason =
  | "RELEASE_UNATTRIBUTABLE"
  | "ENVIRONMENT_OR_DEPLOYMENT_UNRESOLVED"
  | "UNRESOLVED_TRAFFIC_CLASS"
  | "AMBIGUOUS_REPLAY_OR_DUPLICATE"
  | "MISSING_REQUIRED_PROVENANCE"
  | "NOT_APPLICABLE";

export interface RawCandidateRecord {
  event_id?: string;
  observation_unit_id?: string;
  session_id?: string;
  timestamp?: string; // ISO 8601
  symbol?: string;
  normalized_symbol?: string;
  environment?: string;
  deployment_identity?: string;
  release_sha?: string;
  intent_boundary?: string; // e.g. "ETF_COCKPIT_INTENT", "STOCK_INTENT", etc.
  source_component?: string;
  synthetic_marker?: boolean;
  ci_marker?: boolean;
  qa_marker?: boolean;
  developer_marker?: boolean;
  monitoring_marker?: boolean;
  uptime_probe_marker?: boolean;
  bot_signal?: boolean;
  crawler_signal?: boolean;
  replay_identity?: string;
  prior_observation_ref?: string;
  is_truncated?: boolean;
  malformed_structure?: boolean;
  // Downstream outcome fields (strictly non-gating for denominator)
  downstream_routing_success?: boolean;
  downstream_render_success?: boolean;
  cost_of_ownership_available?: boolean;
  downstream_error?: string;
}

export interface ClassifiedObservationAuditRecord {
  event_id: string;
  observation_unit_id: string;
  session_id: string;
  timestamp: string;
  normalized_symbol: string;
  environment: string;
  deployment_identity: string;
  release_sha: string;
  traffic_class: TrafficClass;
  classification_state: PrimaryAccountingState;
  classification_reason: string;
  exclusion_code: ExclusionCode;
  quarantine_reason: QuarantineReason;
  deduplication_key: string;
  replay_identity: string;
  source_component: string;
  is_denominator_member: boolean;
  downstream_outcome?: {
    routing_success?: boolean;
    render_success?: boolean;
    cost_data_available?: boolean;
    error?: string;
  };
}

export interface EpochContract {
  epoch_id: string;
  epoch_start_utc: string; // ISO 8601
  epoch_end_utc: string;   // ISO 8601
  authorized_releases: string[]; // List of authorized commit SHAs
  permitted_deployments: string[]; // List of permitted deployment domains/ids
}

export interface AccountingLedger {
  raw_events_count: number;
  valid_events_count: number;
  quarantined_events_count: number;
  excluded_events_count: number;
  invalid_events_count: number;

  raw_observation_candidates: number;
  valid_observations: number;
  quarantined_observations: number;
  excluded_observations: number;
  invalid_observations: number;

  prospective_denominator: number;
  accepted_observation_ids: Set<string>;
  deduplication_registry: Map<string, string>; // deduplication_key -> observation_unit_id
  audit_records: ClassifiedObservationAuditRecord[];
}

export function createAccountingLedger(): AccountingLedger {
  return {
    raw_events_count: 0,
    valid_events_count: 0,
    quarantined_events_count: 0,
    excluded_events_count: 0,
    invalid_events_count: 0,
    raw_observation_candidates: 0,
    valid_observations: 0,
    quarantined_observations: 0,
    excluded_observations: 0,
    invalid_observations: 0,
    prospective_denominator: 0,
    accepted_observation_ids: new Set(),
    deduplication_registry: new Map(),
    audit_records: [],
  };
}

/**
 * 15-Step Canonical Decision Engine.
 * Evaluates candidate records in strict Section 9 normative order and absolute precedence:
 * INVALID > QUARANTINED > EXCLUDED > VALID
 */
export function classifyCandidateRecord(
  raw: RawCandidateRecord,
  epoch: EpochContract,
  ledger?: AccountingLedger
): ClassifiedObservationAuditRecord {
  // Step 1 & 2 & 3: Structural Validation (Section 9.4)
  if (
    raw.malformed_structure ||
    raw.is_truncated ||
    !raw ||
    typeof raw !== "object" ||
    (!raw.event_id && !raw.observation_unit_id)
  ) {
    return {
      event_id: raw?.event_id || "UNAVAILABLE_DUE_TO_INVALID_STRUCTURE",
      observation_unit_id: raw?.observation_unit_id || "UNAVAILABLE_DUE_TO_INVALID_STRUCTURE",
      session_id: raw?.session_id || "UNAVAILABLE_DUE_TO_INVALID_STRUCTURE",
      timestamp: raw?.timestamp || "UNAVAILABLE_DUE_TO_INVALID_STRUCTURE",
      normalized_symbol: raw?.normalized_symbol || raw?.symbol || "UNAVAILABLE_DUE_TO_INVALID_STRUCTURE",
      environment: raw?.environment || "UNAVAILABLE_DUE_TO_INVALID_STRUCTURE",
      deployment_identity: raw?.deployment_identity || "UNAVAILABLE_DUE_TO_INVALID_STRUCTURE",
      release_sha: raw?.release_sha || "UNAVAILABLE_DUE_TO_INVALID_STRUCTURE",
      traffic_class: "NOT_APPLICABLE",
      classification_state: "INVALID",
      classification_reason: "Candidate record is malformed, truncated, or structurally impossible",
      exclusion_code: "NOT_APPLICABLE",
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: "UNAVAILABLE_DUE_TO_INVALID_STRUCTURE",
      replay_identity: raw?.replay_identity || "NOT_APPLICABLE",
      source_component: raw?.source_component || "UNAVAILABLE_DUE_TO_INVALID_STRUCTURE",
      is_denominator_member: false,
    };
  }

  const eventId = raw.event_id || "MISSING";
  const obsUnitId = raw.observation_unit_id || `OBS_${eventId}`;
  const sessionId = raw.session_id || "MISSING";
  const timestampStr = raw.timestamp || "MISSING";
  const normSymbol = (raw.normalized_symbol || raw.symbol || "").trim().toUpperCase();
  const environment = (raw.environment || "").trim().toLowerCase();
  const deploymentId = (raw.deployment_identity || "").trim();
  const releaseSha = (raw.release_sha || "").trim().toLowerCase();
  const dedupKey = `${sessionId}_${normSymbol}_${eventId}`;
  const sourceComp = raw.source_component || "NOT_APPLICABLE";
  const replayId = raw.replay_identity || "NOT_APPLICABLE";

  // Step 4 & 5: Required Provenance and Unresolved Facts (Section 9.5 & 9.6)
  if (
    !environment ||
    environment === "unresolved" ||
    !deploymentId ||
    deploymentId === "unresolved"
  ) {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment || "UNRESOLVED",
      deployment_identity: deploymentId || "UNRESOLVED",
      release_sha: releaseSha || "NOT_APPLICABLE",
      traffic_class: "UNKNOWN",
      classification_state: "QUARANTINED",
      classification_reason: "Environment or deployment identity unresolved",
      exclusion_code: "NOT_APPLICABLE",
      quarantine_reason: "ENVIRONMENT_OR_DEPLOYMENT_UNRESOLVED",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  if (
    !releaseSha ||
    releaseSha === "missing" ||
    releaseSha.length !== 40 ||
    /^[0]+$/.test(releaseSha)
  ) {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha || "MISSING",
      traffic_class: "UNKNOWN",
      classification_state: "QUARANTINED",
      classification_reason: "Release identity missing, malformed, or contradictory",
      exclusion_code: "NOT_APPLICABLE",
      quarantine_reason: "RELEASE_UNATTRIBUTABLE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  // Step 6 & 7: Deterministic Exclusion Evidence (Section 9.7 & 9.10)
  if (raw.ci_marker) {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "CI",
      classification_state: "EXCLUDED",
      classification_reason: "Deterministic CI marker present",
      exclusion_code: "CI",
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  if (raw.synthetic_marker) {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "SYNTHETIC_E2E",
      classification_state: "EXCLUDED",
      classification_reason: "Deterministic synthetic/E2E test marker present",
      exclusion_code: "SYNTHETIC_E2E",
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  if (raw.qa_marker) {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "MANUAL_QA",
      classification_state: "EXCLUDED",
      classification_reason: "Deterministic manual QA provenance present",
      exclusion_code: "MANUAL_QA",
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  if (raw.developer_marker) {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "DEVELOPER",
      classification_state: "EXCLUDED",
      classification_reason: "Internal developer marker present",
      exclusion_code: "DEVELOPER",
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  if (raw.uptime_probe_marker) {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "UPTIME_PROBE",
      classification_state: "EXCLUDED",
      classification_reason: "Uptime probe provenance present",
      exclusion_code: "UPTIME_PROBE",
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  if (raw.monitoring_marker) {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "MONITORING",
      classification_state: "EXCLUDED",
      classification_reason: "Monitoring beacon provenance present",
      exclusion_code: "MONITORING",
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  if (raw.bot_signal) {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "BOT",
      classification_state: "EXCLUDED",
      classification_reason: "Deterministic bot user-agent signal present",
      exclusion_code: "BOT",
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  if (raw.crawler_signal) {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "CRAWLER",
      classification_state: "EXCLUDED",
      classification_reason: "Deterministic crawler user-agent signal present",
      exclusion_code: "CRAWLER",
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  // Non-production environment check
  if (environment !== "production") {
    const code: ExclusionCode =
      environment === "local"
        ? "LOCAL"
        : environment === "preview"
        ? "PREVIEW_DEPLOYMENT"
        : "NON_PRODUCTION";
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "NON_PRODUCTION",
      classification_state: "EXCLUDED",
      classification_reason: `Non-production environment: ${environment}`,
      exclusion_code: code,
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  // Release authorization check (Section 9.6 & 9.10 Step 5)
  if (!epoch.authorized_releases.includes(releaseSha)) {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "NON_AUTHORIZED_RELEASE",
      classification_state: "EXCLUDED",
      classification_reason: `Release SHA ${releaseSha} is outside the authorized epoch release set`,
      exclusion_code: "WRONG_RELEASE",
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  // Timestamp parsing and epoch boundary check (Section 9.11)
  const eventTime = new Date(timestampStr).getTime();
  const epochStart = new Date(epoch.epoch_start_utc).getTime();
  const epochEnd = new Date(epoch.epoch_end_utc).getTime();

  if (isNaN(eventTime)) {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "UNKNOWN",
      classification_state: "QUARANTINED",
      classification_reason: "Unparseable timestamp",
      exclusion_code: "NOT_APPLICABLE",
      quarantine_reason: "MISSING_REQUIRED_PROVENANCE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  if (eventTime < epochStart || eventTime > epochEnd) {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "NATURAL_PRODUCTION",
      classification_state: "EXCLUDED",
      classification_reason: `Event timestamp ${timestampStr} is outside the authorized observation epoch`,
      exclusion_code: "OUTSIDE_EPOCH",
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  // Replay & Duplicate Detection (Section 9.8)
  if (raw.replay_identity && raw.replay_identity !== "NOT_APPLICABLE") {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "REPLAY",
      classification_state: "EXCLUDED",
      classification_reason: `Deterministic replay of prior observation ${raw.replay_identity}`,
      exclusion_code: "REPLAY",
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: dedupKey,
      replay_identity: raw.replay_identity,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  if (ledger && ledger.deduplication_registry.has(dedupKey)) {
    const priorRef = ledger.deduplication_registry.get(dedupKey)!;
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "DUPLICATE",
      classification_state: "EXCLUDED",
      classification_reason: `Deterministic duplicate of accepted observation unit ${priorRef}`,
      exclusion_code: "DUPLICATE",
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  // Step 8 & 9: Positive Natural Production Evaluation (Section 9.9)
  // N1: environment = production (passed)
  // N2: deployment is permitted
  if (!epoch.permitted_deployments.includes(deploymentId)) {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "UNKNOWN",
      classification_state: "QUARANTINED",
      classification_reason: `Deployment identity ${deploymentId} is not in the permitted production deployment list`,
      exclusion_code: "NOT_APPLICABLE",
      quarantine_reason: "ENVIRONMENT_OR_DEPLOYMENT_UNRESOLVED",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  // Step 10: Denominator Eligibility Checks (E1–E10, Section 9.12)
  // E10 & E1: ETF P1 intent boundary check
  const intentBoundary = (raw.intent_boundary || "").trim().toUpperCase();
  if (intentBoundary !== "ETF_COCKPIT_INTENT" && intentBoundary !== "ETF_INTENT") {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "NOT_APPLICABLE",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "NATURAL_PRODUCTION",
      classification_state: "EXCLUDED",
      classification_reason: `Non-ETF intent: ${intentBoundary || "UNSPECIFIED"}`,
      exclusion_code: "NON_ETF_INTENT",
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  // E2: Valid normalized symbol
  if (!normSymbol || normSymbol.length < 1 || normSymbol.length > 10) {
    return {
      event_id: eventId,
      observation_unit_id: obsUnitId,
      session_id: sessionId,
      timestamp: timestampStr,
      normalized_symbol: normSymbol || "EMPTY",
      environment: environment,
      deployment_identity: deploymentId,
      release_sha: releaseSha,
      traffic_class: "NATURAL_PRODUCTION",
      classification_state: "EXCLUDED",
      classification_reason: `Unsupported input symbol: ${normSymbol}`,
      exclusion_code: "UNSUPPORTED_INPUT",
      quarantine_reason: "NOT_APPLICABLE",
      deduplication_key: dedupKey,
      replay_identity: replayId,
      source_component: sourceComp,
      is_denominator_member: false,
    };
  }

  // Step 13: All eligibility conditions passed!
  // State = VALID, Traffic Class = NATURAL_PRODUCTION
  // Downstream outcome is stored separately and NEVER alters primary accounting state (Section 9.17)
  const classified: ClassifiedObservationAuditRecord = {
    event_id: eventId,
    observation_unit_id: obsUnitId,
    session_id: sessionId,
    timestamp: timestampStr,
    normalized_symbol: normSymbol,
    environment: environment,
    deployment_identity: deploymentId,
    release_sha: releaseSha,
    traffic_class: "NATURAL_PRODUCTION",
    classification_state: "VALID",
    classification_reason: "Positively classified NATURAL_PRODUCTION satisfying every eligibility condition",
    exclusion_code: "NOT_APPLICABLE",
    quarantine_reason: "NOT_APPLICABLE",
    deduplication_key: dedupKey,
    replay_identity: "NOT_APPLICABLE",
    source_component: sourceComp,
    is_denominator_member: true,
    downstream_outcome: {
      routing_success: raw.downstream_routing_success,
      render_success: raw.downstream_render_success,
      cost_data_available: raw.cost_of_ownership_available,
      error: raw.downstream_error,
    },
  };

  // Record in ledger if present
  if (ledger) {
    ledger.accepted_observation_ids.add(obsUnitId);
    ledger.deduplication_registry.set(dedupKey, obsUnitId);
  }

  return classified;
}

/**
 * Ingestion pipeline processor that updates conservation ledger.
 */
export function ingestCandidate(
  raw: RawCandidateRecord,
  epoch: EpochContract,
  ledger: AccountingLedger
): ClassifiedObservationAuditRecord {
  ledger.raw_events_count += 1;
  ledger.raw_observation_candidates += 1;

  const record = classifyCandidateRecord(raw, epoch, ledger);

  switch (record.classification_state) {
    case "INVALID":
      ledger.invalid_events_count += 1;
      ledger.invalid_observations += 1;
      break;
    case "QUARANTINED":
      ledger.quarantined_events_count += 1;
      ledger.quarantined_observations += 1;
      break;
    case "EXCLUDED":
      ledger.excluded_events_count += 1;
      ledger.excluded_observations += 1;
      break;
    case "VALID":
      ledger.valid_events_count += 1;
      ledger.valid_observations += 1;
      ledger.prospective_denominator += 1;
      break;
  }

  ledger.audit_records.push(record);
  return record;
}
