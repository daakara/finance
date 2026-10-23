/**
 * ARX Terminal — Corporate Events Telemetry Contract Types (v1.0.0).
 *
 * Defines the canonical 17-field evaluation payload and the transport envelope
 * ensuring separation between Evaluation Identity (evaluation_id) and
 * Delivery Attempt Identity (delivery_attempt_id).
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
 * The 17-field canonical evaluation payload for Corporate Events RC5.
 */
export interface CorporateEventCanonicalPayload {
  schema_version: "1.0.0";
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
  // Optional transport & deduplication hints
  evaluation_id?: string;
  delivery_attempt_id?: string;
  deduplication_key?: string;
  synthetic_marker?: boolean;
  ci_marker?: boolean;
  manual_qa_marker?: boolean;
}

/**
 * Transport envelope wrapping the canonical evaluation payload.
 * Separates transmission attempt metadata from the frozen evaluation calculation.
 */
export interface CorporateEventTransportEnvelope {
  transport_version: "1.0.0";
  evaluation_id: string;
  delivery_attempt_id: string;
  client_timestamp: string;
  client_epoch_ms: number;
  synthetic_marker?: boolean;
  ci_marker?: boolean;
  manual_qa_marker?: boolean;
  payload: CorporateEventCanonicalPayload;
}

export type GatewayIngestionStatus =
  | "ACCEPTED"
  | "DUPLICATE_ACKNOWLEDGED"
  | "DISABLED"
  | "REJECTED_MALFORMED"
  | "REJECTED_PRIVACY_VIOLATION"
  | "UPSTREAM_UNAVAILABLE"
  | "GATEWAY_TIMEOUT";

export interface GatewayResponse {
  status: GatewayIngestionStatus;
  accepted: boolean;
  forwarded: boolean;
  evaluation_id?: string;
  delivery_attempt_id?: string;
  deduplication_key?: string;
  provenance_class?: ProvenanceClass;
  message?: string;
  rejection_reason?: string;
}
