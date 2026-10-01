// Cloudflare Pages Function: /api/telemetry/etf-intent
// Edge Ingestion & Canonical Telemetry Classification Endpoint

import {
  classifyCandidateRecord,
  RawCandidateRecord,
  ClassifiedObservationAuditRecord,
  EpochContract,
} from "../../../lib/telemetry/etfDenominatorEngine";

const ALLOWED_ORIGINS = new Set([
  "https://www.arxterminal.com",
  "https://arxterminal.com",
  "https://finance-xp8.pages.dev",
  "http://localhost:3000",
  "http://localhost:3005",
  "http://127.0.0.1:3000",
  "http://127.0.0.1:3005",
]);

const PROD_RELEASE_SHA = "9d1cce52070ba242136c3346ce7fd6c83b6ad3ed";
const MAX_PAYLOAD_BYTES = 16 * 1024; // 16 KB

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
  const origin = context.request.headers.get("Origin");
  const allowedOrigin = resolveAllowedOrigin(origin);

  // 1. CORS Pre-Flight
  if (context.request.method === "OPTIONS") {
    const headers: Record<string, string> = {
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, X-Attempt-Id, X-Deduplication-Key, X-Arx-Synthetic, Authorization",
      "Access-Control-Max-Age": "86400",
      "Vary": "Origin",
    };
    if (allowedOrigin) {
      headers["Access-Control-Allow-Origin"] = allowedOrigin;
      headers["Access-Control-Allow-Credentials"] = "true";
    }
    return new Response(null, { status: 204, headers });
  }

  if (context.request.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method Not Allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json" },
    });
  }

  // 2. Read and Validate Payload Size
  let rawBodyText = "";
  try {
    rawBodyText = await context.request.text();
    if (rawBodyText.length > MAX_PAYLOAD_BYTES) {
      return new Response(
        JSON.stringify({ error: "Payload Too Large", max_bytes: MAX_PAYLOAD_BYTES }),
        { status: 413, headers: { "Content-Type": "application/json" } }
      );
    }
  } catch (err) {
    return new Response(JSON.stringify({ error: "Failed to read request body" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  let parsed: any;
  try {
    parsed = JSON.parse(rawBodyText);
  } catch (err) {
    // Malformed JSON is structurally impossible
    return new Response(
      JSON.stringify({
        status: "INVALID",
        classification_state: "INVALID",
        reason: "MALFORMED_JSON_PAYLOAD",
      }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  // 3. Derive Server-Trusted Identity
  const host = context.request.headers.get("Host") || "";
  const isProdHost = host === "arxterminal.com" || host === "www.arxterminal.com";
  const branch = context.env?.CF_PAGES_BRANCH || "main";
  const environment = isProdHost && branch === "main" ? "production" : "preview";
  const deploymentIdentity = isProdHost
    ? "production-cloudflare-pages"
    : `preview-${branch}-cloudflare-pages`;

  const serverReleaseSha =
    context.env?.CF_PAGES_COMMIT_SHA ||
    context.env?.NEXT_PUBLIC_RELEASE_SHA ||
    PROD_RELEASE_SHA;

  // Header-based synthetic / bot signals
  const syntheticHeader = context.request.headers.get("X-Arx-Synthetic");
  const isSynthetic =
    Boolean(parsed.synthetic_marker) ||
    syntheticHeader === "true" ||
    context.request.url.includes("synthetic=true");

  const botScore = context.request.cf?.botManagement?.score;
  const isBot = typeof botScore === "number" && botScore < 30;
  const userAgent = context.request.headers.get("User-Agent") || parsed.user_agent_raw || "";

  // 4. Construct Server-Ratified Candidate Record
  const rawCandidate: RawCandidateRecord = {
    event_id: String(parsed.event_id || ""),
    session_id: String(parsed.session_id || ""),
    observation_unit_id: String(parsed.observation_unit_id || ""),
    timestamp: String(parsed.timestamp || ""),
    symbol: String(parsed.normalized_symbol || "").toUpperCase(),
    normalized_symbol: String(parsed.normalized_symbol || "").toUpperCase(),
    intent_boundary: "ETF_COCKPIT_INTENT",
    source_component: String(parsed.source_component || "handleSelectSymbol"),
    environment,
    deployment_identity: deploymentIdentity,
    release_sha: serverReleaseSha,
    synthetic_marker: isSynthetic,
    ci_marker: Boolean(parsed.ci_run_marker),
    qa_marker: Boolean(parsed.manual_qa_marker),
    bot_signal: isBot,
  };

  // 5. Execute Canonical Classifier
  // Note: Before explicit prospective epoch activation, active epoch is NONE.
  // Any event timestamp pre-epoch deterministically evaluates to OUTSIDE_EPOCH -> EXCLUDED.
  const activeEpoch: EpochContract = {
    epoch_id: "NONE",
    epoch_start_utc: "9999-12-31T23:59:59Z",
    epoch_end_utc: "9999-12-31T23:59:59Z",
    authorized_releases: [],
    permitted_deployments: [
      "https://www.arxterminal.com",
      "https://finance-xp8.pages.dev",
      "production-cloudflare-pages",
      "preview-main-cloudflare-pages",
    ],
  };
  const classified: ClassifiedObservationAuditRecord = classifyCandidateRecord(
    rawCandidate,
    activeEpoch
  );

  // 6. Forward to Canonical Persistence Endpoint (Railway API Backend)
  const backendUrl = "https://web-production-e370b.up.railway.app/api/v1/telemetry/persist";
  try {
    await fetch(backendUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Internal-Secret": context.env?.TELEMETRY_INTERNAL_SECRET || "",
      },
      body: JSON.stringify({
        raw_event: {
          schema_version: parsed.schema_version || "1.0.0",
          event_id: rawCandidate.event_id,
          session_id: rawCandidate.session_id,
          observation_unit_id: rawCandidate.observation_unit_id,
          deduplication_key: classified.deduplication_key,
          timestamp: rawCandidate.timestamp,
          normalized_symbol: rawCandidate.normalized_symbol,
          intent_type: "ETF_SYMBOL_SELECT",
          route: String(parsed.route || "/"),
          source_component: rawCandidate.source_component,
          environment: rawCandidate.environment,
          deployment_identity: rawCandidate.deployment_identity,
          release_sha: rawCandidate.release_sha,
          synthetic_marker: rawCandidate.synthetic_marker,
          ci_run_marker: rawCandidate.ci_marker,
          manual_qa_marker: rawCandidate.qa_marker,
          user_agent_raw: userAgent,
        },
        audit_record: classified,
      }),
    }).catch(() => {
      // Background persistence attempt; edge returns classification response fail-closed
    });
  } catch {
    // Edge continues
  }

  // 7. Response with Classified Metadata
  const responseHeaders: Record<string, string> = {
    "Content-Type": "application/json",
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Vary": "Origin",
  };
  if (allowedOrigin) {
    responseHeaders["Access-Control-Allow-Origin"] = allowedOrigin;
    responseHeaders["Access-Control-Allow-Credentials"] = "true";
  }

  return new Response(
    JSON.stringify({
      status: "ACCEPTED",
      observation_unit_id: classified.observation_unit_id,
      deduplication_key: classified.deduplication_key,
      classification_state: classified.classification_state,
      traffic_class: classified.traffic_class,
      classification_reason: classified.classification_reason,
      exclusion_code: classified.exclusion_code,
      quarantine_reason: classified.quarantine_reason,
    }),
    { status: 200, headers: responseHeaders }
  );
};
