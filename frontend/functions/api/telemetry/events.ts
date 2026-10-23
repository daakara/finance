// Cloudflare Pages Function: /api/telemetry/events
// Edge Ingestion & Validation Gateway for Corporate Events RC5 Telemetry

import {
  scanForProhibitedFields,
  validateCorporateEventPayload,
  computeServerDeduplicationKey,
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
const DEFAULT_HMAC_SALT = "arx_telemetry_corporate_events_salt_v1";

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

  // Common response headers
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
    return new Response(
      JSON.stringify({ error: "Method Not Allowed. Only POST is supported." }),
      { status: 405, headers: getCorsHeaders() }
    );
  }

  // 3. Payload Size Limitation (16 KB)
  let rawBodyText = "";
  try {
    rawBodyText = await req.text();
    if (rawBodyText.length > MAX_PAYLOAD_BYTES) {
      return new Response(
        JSON.stringify({
          error: "Payload Too Large",
          max_bytes: MAX_PAYLOAD_BYTES,
          received_bytes: rawBodyText.length,
        }),
        { status: 413, headers: getCorsHeaders() }
      );
    }
  } catch (err) {
    return new Response(
      JSON.stringify({ error: "Failed to read request body stream" }),
      { status: 400, headers: getCorsHeaders() }
    );
  }

  // 4. JSON Deserialization
  let parsedBody: any;
  try {
    parsedBody = JSON.parse(rawBodyText);
  } catch (err) {
    return new Response(
      JSON.stringify({ error: "Malformed JSON payload. Parsing failed." }),
      { status: 400, headers: getCorsHeaders() }
    );
  }

  // 5. Strict Privacy Pre-Filter (Deny-list scan)
  const privacyCheck = scanForProhibitedFields(parsedBody);
  if (privacyCheck.detected) {
    return new Response(
      JSON.stringify({
        error: "Privacy Boundary Violation: Prohibited field detected",
        prohibited_field: privacyCheck.prohibitedField,
      }),
      { status: 400, headers: getCorsHeaders() }
    );
  }

  // 6. Unpack Transport Envelope or Direct Canonical Payload
  let canonicalPayloadRaw: any;
  let deliveryAttemptId: string | undefined;
  let clientTimestamp: string | undefined;
  let syntheticFromEnvelope: boolean | undefined;

  if (
    parsedBody &&
    typeof parsedBody === "object" &&
    parsedBody.transport_version === "1.0.0" &&
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

  // 7. Canonical Schema Validation
  const validation = validateCorporateEventPayload(canonicalPayloadRaw);
  if (!validation.valid || !validation.payload) {
    return new Response(
      JSON.stringify({
        error: "Schema Validation Error",
        details: validation.errors,
      }),
      { status: 400, headers: getCorsHeaders() }
    );
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

  // Derive Deduplication Key (Server authority)
  const salt = context.env?.TELEMETRY_HMAC_SALT || DEFAULT_HMAC_SALT;
  const evaluationId = payload.evaluation_id || payload.event_id;
  payload.evaluation_id = evaluationId;

  try {
    payload.deduplication_key = await computeServerDeduplicationKey(
      payload.evaluation_context_id,
      payload.symbol,
      payload.event_time_utc,
      evaluationId,
      salt
    );
  } catch {
    // Cryptographic failure fails closed
    return new Response(
      JSON.stringify({ error: "Cryptographic deduplication derivation failed" }),
      { status: 500, headers: getCorsHeaders() }
    );
  }

  // 9. Feature Flag Check (Disabled by default)
  const isEnabled = context.env?.ENABLE_CORPORATE_EVENTS_TELEMETRY === "true";

  if (!isEnabled) {
    const disabledResponse: GatewayResponse = {
      status: "DISABLED",
      accepted: false,
      forwarded: false,
      evaluation_id: evaluationId,
      delivery_attempt_id: deliveryAttemptId,
      deduplication_key: payload.deduplication_key,
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
