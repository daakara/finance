/**
 * ARX Terminal — Corporate Events Telemetry Client Helper (v1.1.0 Hardened).
 *
 * Implements frozen evaluation identity across transmission retries,
 * separate delivery attempt tracking, and fail-open transport dispatch.
 */

import {
  CorporateEventCanonicalPayload,
  CorporateEventTransportEnvelope,
  GatewayResponse,
  AcknowledgmentStatus,
  PersistenceState,
} from "../../types/corporate-events-telemetry";
import { deriveAuthoritativeEvaluationIdSync } from "./corporateEventsContract";

export const CORPORATE_EVENTS_TELEMETRY_ENDPOINT = "/api/telemetry/events";

/**
 * Generates an RFC 4122 compliant UUIDv4 string using crypto.randomUUID or fallback.
 */
export function generateUuidV4(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

let sessionContextId = "";

/**
 * Returns an ephemeral anonymous context ID for the current session.
 * Destroyed when browser tab is closed.
 */
export function getEvaluationContextId(): string {
  if (sessionContextId) return sessionContextId;
  if (typeof window !== "undefined" && window.sessionStorage) {
    try {
      const stored = window.sessionStorage.getItem("arx-telemetry-context-id");
      if (stored) {
        sessionContextId = stored;
        return sessionContextId;
      }
      sessionContextId = `ctx-${generateUuidV4()}`;
      window.sessionStorage.setItem("arx-telemetry-context-id", sessionContextId);
      return sessionContextId;
    } catch {
      // Fallback if sessionStorage blocked
    }
  }
  sessionContextId = `ctx-${generateUuidV4()}`;
  return sessionContextId;
}

export function resetEvaluationContext(): void {
  sessionContextId = "";
  if (typeof window !== "undefined" && window.sessionStorage) {
    try {
      window.sessionStorage.removeItem("arx-telemetry-context-id");
    } catch {
      // Ignore
    }
  }
}

export interface CreateEvaluationParams {
  symbol: string;
  release_sha: string;
  days_to_earnings: number | null;
  risk_bucket: "ACUTE" | "CAUTION" | "NOMINAL" | "UNAVAILABLE";
  applied_penalty: number;
  event_category?: "EARNINGS_ANNOUNCEMENT" | "DIVIDEND" | "SPLIT" | "ACQUISITION" | "NONE" | "UNAVAILABLE";
  event_time_utc?: string | null;
  market_session?: "BMO" | "AMC" | "DURING" | "UNKNOWN";
  event_source_status?: "CONFIRMED" | "UNVERIFIED" | "DATA_UNAVAILABLE";
  freshness_status?: "FRESH" | "STALE" | "UNVERIFIED";
  evaluation_outcome?: "COMPLETED" | "UNAVAILABLE" | "ERROR";
  ruleset_version?: "RC5" | "RC4";
  occurred_at_utc?: string;
  evaluation_id?: string;
  canonical_digest?: string;
}

/**
 * Assembles a frozen evaluation payload with a permanent evaluation_id.
 */
export function assembleCorporateEventPayload(
  params: CreateEvaluationParams
): CorporateEventCanonicalPayload {
  const occurredAt = params.occurred_at_utc || new Date().toISOString();
  const evalId =
    params.evaluation_id ||
    deriveAuthoritativeEvaluationIdSync({
      release_sha: params.release_sha,
      symbol: params.symbol,
      ruleset_version: params.ruleset_version || "RC5",
      event_time_utc: params.event_time_utc ?? null,
      market_session: params.market_session || "UNKNOWN",
      evaluation_timestamp: occurredAt,
    });

  return {
    schema_version: "1.1.0",
    event_id: generateUuidV4(),
    occurred_at_utc: occurredAt,
    release_sha: params.release_sha.toLowerCase(),
    ruleset_version: params.ruleset_version || "RC5",
    evaluation_context_id: getEvaluationContextId(),
    symbol: params.symbol.toUpperCase(),
    event_category: params.event_category || "EARNINGS_ANNOUNCEMENT",
    event_time_utc: params.event_time_utc ?? null,
    market_session: params.market_session || "UNKNOWN",
    event_source_status: params.event_source_status || "CONFIRMED",
    freshness_status: params.freshness_status || "FRESH",
    days_to_earnings: params.days_to_earnings,
    risk_bucket: params.risk_bucket,
    applied_penalty: params.applied_penalty,
    evaluation_outcome: params.evaluation_outcome || "COMPLETED",
    evaluation_id: evalId,
    canonical_digest: params.canonical_digest,
  };
}

/**
 * Constructs a transport envelope for dispatch, retaining the frozen evaluation_id
 * while generating a distinct delivery_attempt_id.
 */
export function buildTransportEnvelope(
  payload: CorporateEventCanonicalPayload,
  _isRetry = false
): CorporateEventTransportEnvelope {
  const now = new Date();
  const isSynthetic =
    typeof window !== "undefined" && Boolean((window as any).__ARX_SYNTHETIC__);

  return {
    transport_version: "1.1.0",
    evaluation_id: payload.evaluation_id || payload.event_id,
    delivery_attempt_id: generateUuidV4(),
    client_timestamp: now.toISOString(),
    client_epoch_ms: now.getTime(),
    synthetic_marker: isSynthetic,
    canonical_digest: payload.canonical_digest,
    payload,
  };
}

/**
 * Dispatches the transport envelope to the edge gateway.
 * Fail-open: catches all transport errors and returns safe failure result without throwing.
 */
export async function dispatchCorporateEventTelemetry(
  envelope: CorporateEventTransportEnvelope,
  endpoint = CORPORATE_EVENTS_TELEMETRY_ENDPOINT
): Promise<GatewayResponse> {
  const serialized = JSON.stringify(envelope);

  // If sendBeacon is available and preferred
  if (typeof navigator !== "undefined" && navigator.sendBeacon) {
    try {
      const beaconSent = navigator.sendBeacon(endpoint, serialized);
      if (beaconSent) {
        return {
          contract_version: "1.1.0",
          status: "RECEIVED",
          accepted: true,
          forwarded: false,
          persisted: "UNKNOWN",
          retryable: false,
          evaluation_id: envelope.evaluation_id,
          delivery_attempt_id: envelope.delivery_attempt_id,
          canonical_digest: envelope.canonical_digest,
          message: "Dispatched via sendBeacon; downstream persistence unknown",
        };
      }
    } catch {
      // Fall back to fetch
    }
  }

  // Fetch fallback
  if (typeof fetch !== "undefined") {
    try {
      const resp = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: serialized,
        keepalive: true,
      });

      const bodyText = await resp.text();
      let parsedJson: Record<string, unknown> = {};
      try {
        parsedJson = JSON.parse(bodyText);
      } catch {
        // Raw text response
      }

      if (resp.ok) {
        const status = (parsedJson.status as AcknowledgmentStatus) || "RECEIVED";
        const accepted = typeof parsedJson.accepted === "boolean" ? parsedJson.accepted : true;
        const forwarded = Boolean(parsedJson.forwarded);
        const persisted = (parsedJson.persisted as PersistenceState) || "UNKNOWN";

        return {
          contract_version: "1.1.0",
          status,
          accepted,
          forwarded,
          persisted,
          retryable: Boolean(parsedJson.retryable),
          evaluation_id: envelope.evaluation_id,
          delivery_attempt_id: envelope.delivery_attempt_id,
          deduplication_key: parsedJson.deduplication_key as string | undefined,
          canonical_digest: envelope.canonical_digest,
          message: String(parsedJson.message || "Delivered successfully"),
        };
      }

      return {
        contract_version: "1.1.0",
        status: (parsedJson.status as AcknowledgmentStatus) || (resp.status === 400 ? "REJECTED" : "TEMPORARILY_UNAVAILABLE"),
        accepted: false,
        forwarded: false,
        persisted: "NOT_PERSISTED",
        retryable: resp.status >= 500 || resp.status === 429,
        evaluation_id: envelope.evaluation_id,
        delivery_attempt_id: envelope.delivery_attempt_id,
        error_code: String(parsedJson.error_code || `HTTP_${resp.status}`),
        rejection_reason: String(parsedJson.error || parsedJson.message || `HTTP ${resp.status}`),
      };
    } catch (err) {
      return {
        contract_version: "1.1.0",
        status: "OUTCOME_UNKNOWN",
        accepted: false,
        forwarded: false,
        persisted: "UNKNOWN",
        retryable: true,
        evaluation_id: envelope.evaluation_id,
        delivery_attempt_id: envelope.delivery_attempt_id,
        error_code: "CLIENT_DISPATCH_EXCEPTION",
        rejection_reason: `Network dispatch failed: ${String(err)}`,
      };
    }
  }

  return {
    contract_version: "1.1.0",
    status: "DISABLED",
    accepted: false,
    forwarded: false,
    persisted: "NOT_PERSISTED",
    retryable: false,
    evaluation_id: envelope.evaluation_id,
    delivery_attempt_id: envelope.delivery_attempt_id,
    message: "No HTTP transport available",
  };
}
