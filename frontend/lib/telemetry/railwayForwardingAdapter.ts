/**
 * ARX Terminal — Railway Telemetry Backend Forwarding Adapter.
 *
 * Implements resilient edge-to-backend forwarding with:
 * - Strict timeout enforcement via AbortController.
 * - HTTP status code classification (retryable vs non-retryable).
 * - Upstream signature header (X-Internal-Secret).
 * - Safe response sanitization (no internal infrastructure exposure).
 */

import {
  CorporateEventCanonicalPayload,
  GatewayResponse,
} from "../../types/corporate-events-telemetry";

export interface ForwardingOptions {
  upstreamUrl?: string;
  internalSecret?: string;
  timeoutMs?: number;
  fetchFn?: typeof fetch;
}

export interface UpstreamResult {
  ok: boolean;
  httpStatus: number;
  gatewayResponse: GatewayResponse;
  retryable: boolean;
}

export const DEFAULT_RAILWAY_ENDPOINT =
  "https://web-production-e370b.up.railway.app/api/v1/telemetry/corporate-events/persist";
export const DEFAULT_TIMEOUT_MS = 5000;

/**
 * Forwards a validated Corporate Events evaluation payload to the Railway persistence backend.
 */
export async function forwardToRailwayBackend(
  payload: CorporateEventCanonicalPayload,
  envelopeMeta: {
    delivery_attempt_id?: string;
    client_timestamp?: string;
    bot_score?: number | null;
    host?: string;
    user_agent?: string;
    provenance_class?: string;
  },
  options: ForwardingOptions = {}
): Promise<UpstreamResult> {
  const upstreamUrl = options.upstreamUrl || DEFAULT_RAILWAY_ENDPOINT;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const fetchClient = options.fetchFn || (typeof fetch !== "undefined" ? fetch : null);

  if (!fetchClient) {
    return {
      ok: false,
      httpStatus: 500,
      retryable: true,
      gatewayResponse: {
        status: "UPSTREAM_UNAVAILABLE",
        accepted: false,
        forwarded: false,
        evaluation_id: payload.evaluation_id,
        delivery_attempt_id: envelopeMeta.delivery_attempt_id,
        rejection_reason: "Fetch runtime environment unavailable",
      },
    };
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const requestBody = JSON.stringify({
    raw_event: {
      schema_version: payload.schema_version,
      event_id: payload.event_id,
      evaluation_id: payload.evaluation_id,
      delivery_attempt_id: envelopeMeta.delivery_attempt_id,
      occurred_at_utc: payload.occurred_at_utc,
      client_timestamp: envelopeMeta.client_timestamp,
      release_sha: payload.release_sha,
      ruleset_version: payload.ruleset_version,
      evaluation_context_id: payload.evaluation_context_id,
      symbol: payload.symbol,
      event_category: payload.event_category,
      event_time_utc: payload.event_time_utc,
      market_session: payload.market_session,
      event_source_status: payload.event_source_status,
      freshness_status: payload.freshness_status,
      days_to_earnings: payload.days_to_earnings,
      risk_bucket: payload.risk_bucket,
      applied_penalty: payload.applied_penalty,
      evaluation_outcome: payload.evaluation_outcome,
      provenance_class: envelopeMeta.provenance_class || payload.provenance_class,
      deduplication_key: payload.deduplication_key,
      bot_score: envelopeMeta.bot_score,
      host: envelopeMeta.host,
      user_agent: envelopeMeta.user_agent,
    },
  });

  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "User-Agent": "ARX-Edge-Gateway/1.0",
    };
    if (options.internalSecret) {
      headers["X-Internal-Secret"] = options.internalSecret;
    }

    const response = await fetchClient(upstreamUrl, {
      method: "POST",
      headers,
      body: requestBody,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    const bodyText = await response.text();
    let parsed: Record<string, unknown> = {};
    try {
      parsed = JSON.parse(bodyText);
    } catch {
      // Non-JSON upstream body
    }

    // 1. Success (200 / 201)
    if (response.status === 200 || response.status === 201) {
      return {
        ok: true,
        httpStatus: 200,
        retryable: false,
        gatewayResponse: {
          status: "ACCEPTED",
          accepted: true,
          forwarded: true,
          evaluation_id: payload.evaluation_id,
          delivery_attempt_id: envelopeMeta.delivery_attempt_id,
          deduplication_key: payload.deduplication_key,
          message: "Persisted upstream successfully",
        },
      };
    }

    // 2. Duplicate Acknowledged (409 Conflict)
    if (response.status === 409) {
      return {
        ok: true,
        httpStatus: 200,
        retryable: false,
        gatewayResponse: {
          status: "DUPLICATE_ACKNOWLEDGED",
          accepted: true,
          forwarded: true,
          evaluation_id: payload.evaluation_id,
          delivery_attempt_id: envelopeMeta.delivery_attempt_id,
          deduplication_key: payload.deduplication_key,
          message: "Duplicate evaluation acknowledged; ignored for denominator",
        },
      };
    }

    // 3. Upstream 400 Bad Request
    if (response.status === 400) {
      return {
        ok: false,
        httpStatus: 400,
        retryable: false,
        gatewayResponse: {
          status: "REJECTED_MALFORMED",
          accepted: false,
          forwarded: false,
          evaluation_id: payload.evaluation_id,
          delivery_attempt_id: envelopeMeta.delivery_attempt_id,
          rejection_reason: String(parsed.error || "Upstream rejected payload structure"),
        },
      };
    }

    // 4. Upstream 401 / 403 Authentication Error
    if (response.status === 401 || response.status === 403) {
      return {
        ok: false,
        httpStatus: 502,
        retryable: false,
        gatewayResponse: {
          status: "UPSTREAM_UNAVAILABLE",
          accepted: false,
          forwarded: false,
          evaluation_id: payload.evaluation_id,
          delivery_attempt_id: envelopeMeta.delivery_attempt_id,
          rejection_reason: "Upstream authentication configuration error",
        },
      };
    }

    // 5. Upstream 429 Rate Limited
    if (response.status === 429) {
      return {
        ok: false,
        httpStatus: 429,
        retryable: true,
        gatewayResponse: {
          status: "UPSTREAM_UNAVAILABLE",
          accepted: false,
          forwarded: false,
          evaluation_id: payload.evaluation_id,
          delivery_attempt_id: envelopeMeta.delivery_attempt_id,
          rejection_reason: "Upstream rate limit exceeded",
        },
      };
    }

    // 6. Upstream 5xx Server Error
    return {
      ok: false,
      httpStatus: 502,
      retryable: true,
      gatewayResponse: {
        status: "UPSTREAM_UNAVAILABLE",
        accepted: false,
        forwarded: false,
        evaluation_id: payload.evaluation_id,
        delivery_attempt_id: envelopeMeta.delivery_attempt_id,
        rejection_reason: `Upstream service error (HTTP ${response.status})`,
      },
    };
  } catch (err: any) {
    clearTimeout(timeoutId);

    if (err?.name === "AbortError" || String(err).includes("aborted")) {
      return {
        ok: false,
        httpStatus: 504,
        retryable: true,
        gatewayResponse: {
          status: "GATEWAY_TIMEOUT",
          accepted: false,
          forwarded: false,
          evaluation_id: payload.evaluation_id,
          delivery_attempt_id: envelopeMeta.delivery_attempt_id,
          rejection_reason: `Upstream gateway timeout after ${timeoutMs}ms`,
        },
      };
    }

    return {
      ok: false,
      httpStatus: 502,
      retryable: true,
      gatewayResponse: {
        status: "UPSTREAM_UNAVAILABLE",
        accepted: false,
        forwarded: false,
        evaluation_id: payload.evaluation_id,
        delivery_attempt_id: envelopeMeta.delivery_attempt_id,
        rejection_reason: `Upstream connection failed: ${String(err)}`,
      },
    };
  }
}
