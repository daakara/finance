/**
 * ARX Terminal — Railway Telemetry Backend Forwarding Adapter (v1.1.0 Hardened).
 *
 * Increment 1.1 Hardening:
 * - Gate B: Explicit Acknowledgment State Machine (persisted: CONFIRMED | NOT_PERSISTED | UNKNOWN).
 * - Gate C: Strict duplicate classification:
 *     - Confirmed Duplicate (authoritative upstream proof with matching digest) -> 200 CONFIRMED_DUPLICATE
 *     - Conflicting Identity (same ID, mismatched digest or fields) -> 409 CONFLICTING_IDENTITY
 *     - Ambiguous Conflict (unverified 409) -> 502 AMBIGUOUS_CONFLICT
 * - Gate D & E: Timeout enforcement (5000ms), safe sanitization, zero secret leakage.
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
        contract_version: "1.1.0",
        status: "TEMPORARILY_UNAVAILABLE",
        accepted: false,
        forwarded: false,
        persisted: "NOT_PERSISTED",
        retryable: true,
        evaluation_id: payload.evaluation_id,
        delivery_attempt_id: envelopeMeta.delivery_attempt_id,
        error_code: "RUNTIME_UNAVAILABLE",
        message: "Fetch runtime environment unavailable",
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
      canonical_digest: payload.canonical_digest,
      bot_score: envelopeMeta.bot_score,
      host: envelopeMeta.host,
      user_agent: envelopeMeta.user_agent,
    },
  });

  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "User-Agent": "ARX-Edge-Gateway/1.1",
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

    // 1. Success (200 / 201) -> Durably Persisted
    if (response.status === 200 || response.status === 201) {
      return {
        ok: true,
        httpStatus: 200,
        retryable: false,
        gatewayResponse: {
          contract_version: "1.1.0",
          status: "PERSISTED",
          accepted: true,
          forwarded: true,
          persisted: "CONFIRMED",
          retryable: false,
          evaluation_id: payload.evaluation_id,
          delivery_attempt_id: envelopeMeta.delivery_attempt_id,
          deduplication_key: payload.deduplication_key,
          canonical_digest: payload.canonical_digest,
          message: "Persisted upstream successfully",
        },
      };
    }

    // 2. Gate C: 409 Conflict Classification
    if (response.status === 409) {
      // Case A: Confirmed Duplicate with matching identity and digest
      if (
        parsed.status === "CONFIRMED_DUPLICATE" ||
        (parsed.duplicate === true && parsed.evaluation_id === payload.evaluation_id)
      ) {
        // If upstream echoes canonical_digest, verify match
        if (
          parsed.canonical_digest &&
          payload.canonical_digest &&
          parsed.canonical_digest !== payload.canonical_digest
        ) {
          // Digest mismatch indicates conflicting identity!
          return {
            ok: false,
            httpStatus: 409,
            retryable: false,
            gatewayResponse: {
              contract_version: "1.1.0",
              status: "CONFLICTING_IDENTITY",
              accepted: false,
              forwarded: true,
              persisted: "NOT_PERSISTED",
              retryable: false,
              evaluation_id: payload.evaluation_id,
              delivery_attempt_id: envelopeMeta.delivery_attempt_id,
              error_code: "CONFLICTING_IDENTITY_DIGEST_MISMATCH",
              message: "Upstream conflict: evaluation_id reused with conflicting payload digest",
            },
          };
        }

        return {
          ok: true,
          httpStatus: 200,
          retryable: false,
          gatewayResponse: {
            contract_version: "1.1.0",
            status: "CONFIRMED_DUPLICATE",
            accepted: true,
            forwarded: true,
            persisted: "CONFIRMED",
            retryable: false,
            evaluation_id: payload.evaluation_id,
            delivery_attempt_id: envelopeMeta.delivery_attempt_id,
            deduplication_key: payload.deduplication_key,
            canonical_digest: payload.canonical_digest,
            message: "Authoritative duplicate acknowledged; persistence verified from prior submission",
          },
        };
      }

      // Case B: Explicit Conflicting Identity reported by upstream
      if (
        parsed.status === "CONFLICTING_IDENTITY" ||
        parsed.error_code === "CONFLICTING_IDENTITY"
      ) {
        return {
          ok: false,
          httpStatus: 409,
          retryable: false,
          gatewayResponse: {
            contract_version: "1.1.0",
            status: "CONFLICTING_IDENTITY",
            accepted: false,
            forwarded: true,
            persisted: "NOT_PERSISTED",
            retryable: false,
            evaluation_id: payload.evaluation_id,
            delivery_attempt_id: envelopeMeta.delivery_attempt_id,
            error_code: "CONFLICTING_IDENTITY",
            message: String(parsed.message || "Evaluation ID reused with conflicting immutable attributes"),
          },
        };
      }

      // Case C: Ambiguous Conflict (generic 409 without contract proof) -> FAIL CLOSED
      return {
        ok: false,
        httpStatus: 502,
        retryable: false,
        gatewayResponse: {
          contract_version: "1.1.0",
          status: "AMBIGUOUS_CONFLICT",
          accepted: false,
          forwarded: true,
          persisted: "UNKNOWN",
          retryable: false,
          evaluation_id: payload.evaluation_id,
          delivery_attempt_id: envelopeMeta.delivery_attempt_id,
          error_code: "AMBIGUOUS_CONFLICT",
          message: "Upstream returned generic 409 Conflict without authoritative duplicate proof",
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
          contract_version: "1.1.0",
          status: "REJECTED",
          accepted: false,
          forwarded: false,
          persisted: "NOT_PERSISTED",
          retryable: false,
          evaluation_id: payload.evaluation_id,
          delivery_attempt_id: envelopeMeta.delivery_attempt_id,
          error_code: "UPSTREAM_PAYLOAD_REJECTED",
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
          contract_version: "1.1.0",
          status: "TEMPORARILY_UNAVAILABLE",
          accepted: false,
          forwarded: false,
          persisted: "NOT_PERSISTED",
          retryable: false,
          evaluation_id: payload.evaluation_id,
          delivery_attempt_id: envelopeMeta.delivery_attempt_id,
          error_code: "UPSTREAM_AUTH_ERROR",
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
          contract_version: "1.1.0",
          status: "TEMPORARILY_UNAVAILABLE",
          accepted: false,
          forwarded: false,
          persisted: "NOT_PERSISTED",
          retryable: true,
          evaluation_id: payload.evaluation_id,
          delivery_attempt_id: envelopeMeta.delivery_attempt_id,
          error_code: "RATE_LIMITED",
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
        contract_version: "1.1.0",
        status: "TEMPORARILY_UNAVAILABLE",
        accepted: false,
        forwarded: false,
        persisted: "NOT_PERSISTED",
        retryable: true,
        evaluation_id: payload.evaluation_id,
        delivery_attempt_id: envelopeMeta.delivery_attempt_id,
        error_code: "UPSTREAM_SERVER_ERROR",
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
          contract_version: "1.1.0",
          status: "OUTCOME_UNKNOWN",
          accepted: false,
          forwarded: true,
          persisted: "UNKNOWN",
          retryable: true,
          evaluation_id: payload.evaluation_id,
          delivery_attempt_id: envelopeMeta.delivery_attempt_id,
          error_code: "GATEWAY_TIMEOUT",
          rejection_reason: `Upstream gateway timeout after ${timeoutMs}ms`,
        },
      };
    }

    return {
      ok: false,
      httpStatus: 502,
      retryable: true,
      gatewayResponse: {
        contract_version: "1.1.0",
        status: "TEMPORARILY_UNAVAILABLE",
        accepted: false,
        forwarded: false,
        persisted: "NOT_PERSISTED",
        retryable: true,
        evaluation_id: payload.evaluation_id,
        delivery_attempt_id: envelopeMeta.delivery_attempt_id,
        error_code: "CONNECTION_FAILED",
        rejection_reason: `Upstream connection failed: ${String(err)}`,
      },
    };
  }
}
