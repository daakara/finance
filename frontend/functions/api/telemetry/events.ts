// Cloudflare Pages Function: /api/telemetry/events
// Edge Ingestion & Hardened Gateway for Corporate Events RC5 Telemetry (v1.1.0)

import {
  scanForProhibitedFields,
  validateCorporateEventPayload,
  computeServerDeduplicationKey,
  computeCanonicalDigest,
} from "../../../lib/telemetry/corporateEventsContract";
import {
  forwardToRailwayBackend,
  ForwardingOptions,
} from "../../../lib/telemetry/railwayForwardingAdapter";
import {
  CorporateEventCanonicalPayload,
  CorporateEventTransportEnvelope,
  GatewayResponse,
} from "../../../types/corporate-events-telemetry";

const ALLOWED_ORIGINS = new Set([
  "https://www.arxterminal.com",
  "https://arxterminal.com",
  "https://finance-xp8.pages.dev",
  "http://localhost:3000",
  "http://localhost:3005",
  "http://127.0.0.1:3000",
  "http://127.0.0.1:3005",
]);

const MAX_PAYLOAD_BYTES = 16 * 1024; // 16 KB
const SYNTHETIC_HMAC_SECRET_FALLBACK = "arx_synthetic_test_hmac_secret_v1";

function resolveAllowedOrigin(originHeader: string | null): string | null {
  if (!originHeader) return null;
  if (ALLOWED_ORIGINS.has(originHeader)) return originHeader;
  if (/^https:\/\/(www\.)?arxterminal\.com$/.test(originHeader)) return originHeader;
  if (/^https:\/\/[a-z0-9-]+\.finance-xp8\.pages\.dev$/.test(originHeader)) return originHeader;
  if (/^http:\/\/localhost:\d+$/.test(originHeader)) return originHeader;
  if (/^http:\/\/127\.0\.0\.1:\d+$/.test(originHeader)) return originHeader;
  return null;
}

export const onRequest = async (context: any): Promise<Response> => {
  const req = context.request;
  const origin = req.headers.get("Origin");
  const allowedOrigin = resolveAllowedOrigin(origin);

  // Standard response headers
  const getCorsHeaders = (): Record<string, string> => {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "Cache-Control": "no-store, no-cache, must-revalidate",
      "Vary": "Origin",
    };
    if (allowedOrigin) {
      headers["Access-Control-Allow-Origin"] = allowedOrigin;
      headers["Access-Control-Allow-Credentials"] = "true";
    }
    return headers;
  };

  // 1. CORS Pre-Flight Handling
  if (req.method === "OPTIONS") {
    const preflightHeaders: Record<string, string> = {
      ...getCorsHeaders(),
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers":
        "Content-Type, X-Attempt-Id, X-Deduplication-Key, X-Arx-Synthetic, X-Arx-CI, X-Arx-QA-Session, Authorization",
      "Access-Control-Max-Age": "86400",
    };
    return new Response(null, { status: 204, headers: preflightHeaders });
  }

  // 2. Method Enforcement (POST only)
  if (req.method !== "POST") {
    const resp: GatewayResponse = {
      contract_version: "1.1.0",
      status: "REJECTED",
      accepted: false,
      forwarded: false,
      persisted: "NOT_PERSISTED",
      retryable: false,
      error_code: "METHOD_NOT_ALLOWED",
      message: "Method Not Allowed. Only POST is supported.",
    };
    return new Response(JSON.stringify(resp), {
      status: 405,
      headers: { ...getCorsHeaders(), Allow: "POST, OPTIONS" },
    });
  }

  // 3. Payload Size Limitation (16 KB)
  let rawBodyText = "";
  try {
    rawBodyText = await req.text();
    if (rawBodyText.length > MAX_PAYLOAD_BYTES) {
      const resp: GatewayResponse = {
        contract_version: "1.1.0",
        status: "REJECTED",
        accepted: false,
        forwarded: false,
        persisted: "NOT_PERSISTED",
        retryable: false,
        error_code: "PAYLOAD_TOO_LARGE",
        message: `Payload Too Large: received ${rawBodyText.length} bytes, maximum permitted is ${MAX_PAYLOAD_BYTES} bytes`,
      };
      return new Response(JSON.stringify(resp), { status: 413, headers: getCorsHeaders() });
    }
  } catch {
    const resp: GatewayResponse = {
      contract_version: "1.1.0",
      status: "REJECTED",
      accepted: false,
      forwarded: false,
      persisted: "NOT_PERSISTED",
      retryable: false,
      error_code: "BODY_READ_ERROR",
      message: "Failed to read request body stream",
    };
    return new Response(JSON.stringify(resp), { status: 400, headers: getCorsHeaders() });
  }

  // 4. JSON Deserialization
  let parsedBody: any;
  try {
    parsedBody = JSON.parse(rawBodyText);
  } catch {
    const resp: GatewayResponse = {
      contract_version: "1.1.0",
      status: "REJECTED",
      accepted: false,
      forwarded: false,
      persisted: "NOT_PERSISTED",
      retryable: false,
      error_code: "MALFORMED_JSON",
      message: "Malformed JSON payload. Parsing failed.",
    };
    return new Response(JSON.stringify(resp), { status: 400, headers: getCorsHeaders() });
  }

  // 5. Gate F: Strict Privacy Pre-Filter (Deny-list field and regex value scan)
  const privacyCheck = scanForProhibitedFields(parsedBody);
  if (privacyCheck.detected) {
    const resp: GatewayResponse = {
      contract_version: "1.1.0",
      status: "REJECTED",
      accepted: false,
      forwarded: false,
      persisted: "NOT_PERSISTED",
      retryable: false,
      error_code: "PRIVACY_BOUNDARY_VIOLATION",
      prohibited_field: privacyCheck.prohibitedField,
      prohibited_rule: privacyCheck.prohibitedRule,
      message: `Privacy Boundary Violation: Prohibited field or sensitive value pattern detected in '${privacyCheck.prohibitedField}'`,
    };
    return new Response(JSON.stringify(resp), { status: 400, headers: getCorsHeaders() });
  }

  // 6. Unpack Transport Envelope or Direct Canonical Payload
  let canonicalPayloadRaw: any;
  let deliveryAttemptId: string | undefined;
  let clientTimestamp: string | undefined;
  let syntheticFromEnvelope: boolean | undefined;

  if (
    parsedBody &&
    typeof parsedBody === "object" &&
    (parsedBody.transport_version === "1.0.0" || parsedBody.transport_version === "1.1.0") &&
    parsedBody.payload
  ) {
    const envelope = parsedBody as CorporateEventTransportEnvelope;
    canonicalPayloadRaw = envelope.payload;
    deliveryAttemptId = envelope.delivery_attempt_id;
    clientTimestamp = envelope.client_timestamp;
    syntheticFromEnvelope = envelope.synthetic_marker;
  } else {
    canonicalPayloadRaw = parsedBody;
    deliveryAttemptId = parsedBody?.delivery_attempt_id;
    clientTimestamp = parsedBody?.client_timestamp;
    syntheticFromEnvelope = parsedBody?.synthetic_marker;
  }

  // 7. Canonical Schema Validation & Replay Window (Gate B & D)
  const validation = validateCorporateEventPayload(canonicalPayloadRaw);
  if (!validation.valid || !validation.payload) {
    const resp: GatewayResponse = {
      contract_version: "1.1.0",
      status: "REJECTED",
      accepted: false,
      forwarded: false,
      persisted: "NOT_PERSISTED",
      retryable: false,
      error_code: validation.errorCode || "SCHEMA_VALIDATION_ERROR",
      message: validation.errors.join("; "),
      details: validation.errors,
    };
    return new Response(JSON.stringify(resp), { status: 400, headers: getCorsHeaders() });
  }

  const payload = validation.payload;

  // 8. Derive Server-Ratified Metadata & Deduplication Key
  const host = (req.headers.get("Host") || "").split(":")[0];
  const userAgent = req.headers.get("User-Agent") || "";
  const botScore = req.cf?.botManagement?.score ?? null;

  const isSynthetic =
    Boolean(syntheticFromEnvelope) ||
    Boolean(payload.synthetic_marker) ||
    req.headers.get("X-Arx-Synthetic") === "true" ||
    req.url.includes("synthetic=true");

  const isCi =
    Boolean(payload.ci_marker) ||
    req.headers.get("X-Arx-CI") === "true";

  const isQa =
    Boolean(payload.manual_qa_marker) ||
    req.headers.get("X-Arx-QA-Session") === "true";

  payload.synthetic_marker = isSynthetic;
  payload.ci_marker = isCi;
  payload.manual_qa_marker = isQa;

  const evaluationId = payload.evaluation_id || payload.event_id;
  payload.evaluation_id = evaluationId;

  // Derive Canonical Digest (Gate C & D)
  try {
    payload.canonical_digest = await computeCanonicalDigest(payload);
  } catch {
    const resp: GatewayResponse = {
      contract_version: "1.1.0",
      status: "REJECTED",
      accepted: false,
      forwarded: false,
      persisted: "NOT_PERSISTED",
      retryable: false,
      error_code: "DIGEST_COMPUTATION_FAILED",
      message: "Cryptographic digest computation failed",
    };
    return new Response(JSON.stringify(resp), { status: 500, headers: getCorsHeaders() });
  }

  // Derive Deduplication Key with Server Secret (Gate E)
  const hmacSecret =
    context.env?.TELEMETRY_HMAC_SECRET ||
    context.env?.TELEMETRY_HMAC_SALT ||
    (isSynthetic ? SYNTHETIC_HMAC_SECRET_FALLBACK : "");

  if (!hmacSecret) {
    // Fail-closed in non-synthetic environment if secret is unconfigured
    const resp: GatewayResponse = {
      contract_version: "1.1.0",
      status: "TEMPORARILY_UNAVAILABLE",
      accepted: false,
      forwarded: false,
      persisted: "NOT_PERSISTED",
      retryable: true,
      error_code: "UNCONFIGURED_HMAC_SECRET",
      message: "Server deduplication secret unconfigured; failing closed",
    };
    return new Response(JSON.stringify(resp), { status: 500, headers: getCorsHeaders() });
  }

  const keyId = context.env?.TELEMETRY_KEY_ID || "k1";

  try {
    payload.deduplication_key = await computeServerDeduplicationKey(
      payload.evaluation_context_id,
      payload.symbol,
      payload.event_time_utc,
      evaluationId,
      hmacSecret,
      keyId
    );
  } catch {
    const resp: GatewayResponse = {
      contract_version: "1.1.0",
      status: "TEMPORARILY_UNAVAILABLE",
      accepted: false,
      forwarded: false,
      persisted: "NOT_PERSISTED",
      retryable: true,
      error_code: "CRYPTOGRAPHIC_DERIVATION_FAILED",
      message: "Cryptographic deduplication derivation failed",
    };
    return new Response(JSON.stringify(resp), { status: 500, headers: getCorsHeaders() });
  }

  // 9. Feature Flag Check (Disabled by default)
  const isEnabled = context.env?.ENABLE_CORPORATE_EVENTS_TELEMETRY === "true";

  if (!isEnabled) {
    const disabledResponse: GatewayResponse = {
      contract_version: "1.1.0",
      status: "DISABLED",
      accepted: false,
      forwarded: false,
      persisted: "NOT_PERSISTED",
      retryable: false,
      evaluation_id: evaluationId,
      delivery_attempt_id: deliveryAttemptId,
      deduplication_key: payload.deduplication_key,
      canonical_digest: payload.canonical_digest,
      message: "Corporate Events telemetry gateway is disabled by policy",
    };
    return new Response(JSON.stringify(disabledResponse), {
      status: 200,
      headers: getCorsHeaders(),
    });
  }

  // 10. Resilient Upstream Forwarding
  const forwardingOpts: ForwardingOptions = {
    upstreamUrl: context.env?.TELEMETRY_UPSTREAM_URL,
    internalSecret: context.env?.TELEMETRY_INTERNAL_SECRET,
    timeoutMs: Number(context.env?.TELEMETRY_TIMEOUT_MS) || 5000,
  };

  const forwardResult = await forwardToRailwayBackend(
    payload,
    {
      delivery_attempt_id: deliveryAttemptId,
      client_timestamp: clientTimestamp,
      bot_score: botScore,
      host,
      user_agent: userAgent,
    },
    forwardingOpts
  );

  return new Response(JSON.stringify(forwardResult.gatewayResponse), {
    status: forwardResult.httpStatus,
    headers: getCorsHeaders(),
  });
};
