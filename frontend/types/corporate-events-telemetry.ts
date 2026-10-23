/**
 * ARX Terminal — Corporate Events Telemetry Contract Types (v1.1.0).
 *
 * Increment 1.1 Hardening:
 * - Versioned Acknowledgment Contract (contract_version: "1.1.0").
 * - Multi-state AcknowledgmentStatus (no single-boolean conflation).
 * - Explicit PersistenceState (CONFIRMED | NOT_PERSISTED | UNKNOWN).
 * - Distinguishes Confirmed Duplicate, Conflicting Identity, and Ambiguous Conflict.
 * - Adds Canonical Payload Digest for tamper resistance and deduplication auditing.
 */

export type CorporateEventCategory =
  | "EARNINGS_ANNOUNCEMENT"
  | "DIVIDEND"
  | "SPLIT"
  | "ACQUISITION"
  | "NONE"
  | "UNAVAILABLE";

export type MarketSession = "BMO" | "AMC" | "DURING" | "UNKNOWN";

export type EventSourceStatus = "CONFIRMED" | "UNVERIFIED" | "DATA_UNAVAILABLE";

export type FreshnessStatus = "FRESH" | "STALE" | "UNVERIFIED";

export type RiskBucket = "ACUTE" | "CAUTION" | "NOMINAL" | "UNAVAILABLE";

export type EvaluationOutcome = "COMPLETED" | "UNAVAILABLE" | "ERROR";

export type RulesetVersion = "RC5" | "RC4";

export type ProvenanceClass =
  | "NATURAL_PRODUCTION_ELIGIBLE"
  | "SYNTHETIC_DIAGNOSTIC"
  | "AUTOMATED_SMOKE_TEST"
  | "MANUAL_QA"
  | "REPLAY"
  | "HISTORICAL_IMPORT"
  | "DUPLICATE"
  | "MALFORMED"
  | "UNVERIFIABLE_PROVENANCE"
  | "OUTSIDE_OBSERVATION_EPOCH";

/**
 * The 17-field canonical evaluation payload for Corporate Events RC5 (Increment 1.1).
 */
export interface CorporateEventCanonicalPayload {
  schema_version: "1.0.0" | "1.1.0";
  event_id: string;
  occurred_at_utc: string;
  release_sha: string;
  ruleset_version: RulesetVersion;
  evaluation_context_id: string;
  symbol: string;
  event_category: CorporateEventCategory;
  event_time_utc: string | null;
  market_session: MarketSession;
  event_source_status: EventSourceStatus;
  freshness_status: FreshnessStatus;
  days_to_earnings: number | null;
  risk_bucket: RiskBucket;
  applied_penalty: number;
  evaluation_outcome: EvaluationOutcome;
  provenance_class?: ProvenanceClass;
  // Transport & deduplication metadata
  evaluation_id?: string;
  delivery_attempt_id?: string;
  deduplication_key?: string;
  canonical_digest?: string;
  synthetic_marker?: boolean;
  ci_marker?: boolean;
  manual_qa_marker?: boolean;
}

/**
 * Transport envelope wrapping the canonical evaluation payload.
 * Separates transmission attempt metadata from the frozen evaluation calculation.
 */
export interface CorporateEventTransportEnvelope {
  transport_version: "1.0.0" | "1.1.0";
  evaluation_id: string;
  delivery_attempt_id: string;
  client_timestamp: string;
  client_epoch_ms: number;
  synthetic_marker?: boolean;
  ci_marker?: boolean;
  manual_qa_marker?: boolean;
  canonical_digest?: string;
  payload: CorporateEventCanonicalPayload;
}

/**
 * Explicit machine-readable acknowledgment states (Gate B & C).
 * Prevents collapsing transport, validation, forwarding, persistence, and deduplication.
 */
export type AcknowledgmentStatus =
  | "RECEIVED"
  | "VALIDATED"
  | "FORWARD_PENDING"
  | "PERSISTED"
  | "CONFIRMED_DUPLICATE"
  | "CONFLICTING_IDENTITY"
  | "AMBIGUOUS_CONFLICT"
  | "DISABLED"
  | "REJECTED"
  | "TEMPORARILY_UNAVAILABLE"
  | "OUTCOME_UNKNOWN";

export type PersistenceState = "CONFIRMED" | "NOT_PERSISTED" | "UNKNOWN";

export type GatewayIngestionStatus =
  | "ACCEPTED"
  | "DUPLICATE_ACKNOWLEDGED"
  | "DISABLED"
  | "REJECTED_MALFORMED"
  | "REJECTED_PRIVACY_VIOLATION"
  | "UPSTREAM_UNAVAILABLE"
  | "GATEWAY_TIMEOUT"
  | AcknowledgmentStatus;

/**
 * Formal Acknowledgment Envelope (Increment 1.1).
 */
export interface GatewayResponse {
  contract_version: "1.1.0";
  status: AcknowledgmentStatus;
  accepted: boolean;
  forwarded: boolean;
  persisted: PersistenceState;
  retryable: boolean;
  evaluation_id?: string;
  delivery_attempt_id?: string;
  deduplication_key?: string;
  canonical_digest?: string;
  provenance_class?: ProvenanceClass;
  error_code?: string;
  message?: string;
  rejection_reason?: string;
  prohibited_field?: string;
  prohibited_rule?: string;
  details?: unknown;
}
