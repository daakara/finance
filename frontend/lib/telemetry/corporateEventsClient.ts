/**
 * ARX Terminal — Corporate Events Telemetry Client Helper.
 *
 * Implements frozen evaluation identity across transmission retries,
 * separate delivery attempt tracking, and fail-open transport dispatch.
 */

import {
  CorporateEventCanonicalPayload,
  CorporateEventTransportEnvelope,
  GatewayResponse,
} from "../../types/corporate-events-telemetry";

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
}

/**
 * Assembles a frozen evaluation payload with a permanent evaluation_id.
 */
export function assembleCorporateEventPayload(
  params: CreateEvaluationParams
): CorporateEventCanonicalPayload {
  const evalId = params.evaluation_id || generateUuidV4();
  const occurredAt = params.occurred_at_utc || new Date().toISOString();

  return {
    schema_version: "1.0.0",
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
  };
}

/**
 * Constructs a transport envelope for dispatch, retaining the evaluation_id
 * while generating a distinct delivery_attempt_id.
 */
export function buildTransportEnvelope(
  payload: CorporateEventCanonicalPayload,
  isRetry = false
): CorporateEventTransportEnvelope {
  const now = new Date();
  const isSynthetic =
    typeof window !== "undefined" && Boolean((window as any).__ARX_SYNTHETIC__);

  return {
    transport_version: "1.0.0",
    evaluation_id: payload.evaluation_id || payload.event_id,
    delivery_attempt_id: generateUuidV4(),
    client_timestamp: now.toISOString(),
    client_epoch_ms: now.getTime(),
    synthetic_marker: isSynthetic,
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
          status: "ACCEPTED",
          accepted: true,
          forwarded: false,
          evaluation_id: envelope.evaluation_id,
          delivery_attempt_id: envelope.delivery_attempt_id,
          message: "Dispatched via sendBeacon",
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
        return {
          status: (parsedJson.status as any) || "ACCEPTED",
          accepted: true,
          forwarded: Boolean(parsedJson.forwarded),
          evaluation_id: envelope.evaluation_id,
          delivery_attempt_id: envelope.delivery_attempt_id,
          message: String(parsedJson.message || "Delivered successfully"),
        };
      }

      return {
        status: resp.status === 400 ? "REJECTED_MALFORMED" : "UPSTREAM_UNAVAILABLE",
        accepted: false,
        forwarded: false,
        evaluation_id: envelope.evaluation_id,
        delivery_attempt_id: envelope.delivery_attempt_id,
        rejection_reason: String(parsedJson.error || `HTTP ${resp.status}`),
      };
    } catch (err) {
      return {
        status: "UPSTREAM_UNAVAILABLE",
        accepted: false,
        forwarded: false,
        evaluation_id: envelope.evaluation_id,
        delivery_attempt_id: envelope.delivery_attempt_id,
        rejection_reason: `Network dispatch failed: ${String(err)}`,
      };
    }
  }

  return {
    status: "DISABLED",
    accepted: false,
    forwarded: false,
    evaluation_id: envelope.evaluation_id,
    delivery_attempt_id: envelope.delivery_attempt_id,
    message: "No HTTP transport available",
  };
}
