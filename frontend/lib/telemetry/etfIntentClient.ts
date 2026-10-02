/**
 * ARX Terminal — ETF Cockpit P1 Telemetry Client Module
 *
 * Emits canonical ETF_INTENT_ATTEMPT candidate records at the authoritative P1 intent boundary.
 * Enforces:
 * 1. ONE_EXPLICIT_USER_ETF_INTENT = ONE_ETF_INTENT_ATTEMPT.
 * 2. Non-ETF assets (stocks, crypto, unknown) strictly suppressed (zero network emission).
 * 3. Render/re-render/mount/useEffect triggers strictly prohibited from emitting attempts.
 * 4. Anonymous session identity via browser sessionStorage (arx_p1_session_id).
 * 5. Deterministic composite deduplication key: SHA-256(session_id:normalized_symbol:attempt_id).
 * 6. Transport via navigator.sendBeacon with fetch keepalive fallback.
 * 7. Client failure does not fabricate canonical server success.
 */

import { isETF } from "../assetTypeUtils";
import { sha256 } from "../governance/sha256";
import { RawCandidateRecord } from "./etfDenominatorEngine";

export const SESSION_STORAGE_KEY = "arx_p1_session_id";
export const TELEMETRY_INGESTION_ENDPOINT = "/api/telemetry/etf-intent";

const UUID_V4_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

/**
 * Generate an RFC 4122 compliant UUIDv4.
 */
export function generateUuidV4(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // Fallback RFC 4122 compliant UUIDv4 using crypto.getRandomValues or Math.random
  const bytes = new Uint8Array(16);
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < 16; i++) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 10
  const hex = Array.from(bytes).map((b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20, 32)}`;
}

/**
 * Retrieve or initialize anonymous session identity from browser sessionStorage.
 * Never stores or transmits personal identifiers.
 */
export function getSessionId(): string {
  if (typeof window === "undefined" || !window.sessionStorage) {
    return "00000000-0000-4000-8000-000000000000";
  }

  try {
    const existing = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (existing && UUID_V4_REGEX.test(existing)) {
      return existing;
    }
    const fresh = generateUuidV4();
    window.sessionStorage.setItem(SESSION_STORAGE_KEY, fresh);
    return fresh;
  } catch {
    // If sessionStorage is disabled (e.g. strict privacy mode), fallback to ephemeral in-memory UUID
    return generateUuidV4();
  }
}

/**
 * Reset active session (for test isolation and session termination).
 */
export function resetSession(): void {
  if (typeof window !== "undefined" && window.sessionStorage) {
    try {
      window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
    } catch {
      // Ignore storage errors in restricted contexts
    }
  }
}

/**
 * Derive deterministic composite deduplication key:
 * deduplication_key = SHA-256(session_id + ":" + normalized_symbol + ":" + attempt_id)
 */
export function computeDeduplicationKey(
  sessionId: string,
  normalizedSymbol: string,
  attemptId: string
): string {
  const seed = `${sessionId}:${normalizedSymbol.toUpperCase().trim()}:${attemptId}`;
  return sha256(seed).toLowerCase();
}

export interface ClientIntentPayload extends RawCandidateRecord {
  schema_version: string;
  deduplication_key: string;
  intent_type?: string;
  route?: string;
  ci_run_marker?: boolean;
  manual_qa_marker?: boolean;
  user_agent_raw?: string;
}

export interface EmitResult {
  emitted: boolean;
  attemptId: string;
  deduplicationKey: string;
  payload: ClientIntentPayload;
}

/**
 * Build the canonical raw candidate intent record.
 * Returns null if symbol is not an ETF or if context is invalid.
 */
export function buildRawIntentPayload(params: {
  symbol: string;
  sourceComponent?: string;
  attemptId?: string;
  overrideTimestamp?: string;
}): ClientIntentPayload | null {
  const rawSymbol = params.symbol || "";
  const cleanSymbol = rawSymbol.trim().toUpperCase().replace(/.*:/, "");

  // Strict Invariant: Non-ETF selections NEVER create ETF intent attempts
  if (!cleanSymbol || !isETF(cleanSymbol)) {
    return null;
  }

  const sessionId = getSessionId();
  const attemptId = params.attemptId && UUID_V4_REGEX.test(params.attemptId)
    ? params.attemptId
    : generateUuidV4();

  const dedupKey = computeDeduplicationKey(sessionId, cleanSymbol, attemptId);

  // Extract client interaction provenance
  const isBrowser = typeof window !== "undefined";
  const searchParams = isBrowser && window.location ? window.location.search : "";
  const path = isBrowser && window.location ? window.location.pathname : "/";
  const userAgent = typeof navigator !== "undefined" ? navigator.userAgent : "";

  // Controlled test/synthetic markers
  const syntheticMarker = Boolean(
    isBrowser &&
      ((window as any).__ARX_SYNTHETIC__ === true ||
        searchParams.includes("synthetic=true") ||
        searchParams.includes("arx_test=true"))
  );
  const ciRunMarker = Boolean(isBrowser && searchParams.includes("ci_run=true"));
  const manualQaMarker = Boolean(isBrowser && searchParams.includes("qa_run=true"));

  return {
    schema_version: "1.0.0",
    event_id: generateUuidV4(),
    session_id: sessionId,
    observation_unit_id: attemptId,
    deduplication_key: dedupKey,
    timestamp: params.overrideTimestamp || new Date().toISOString(),
    symbol: cleanSymbol,
    normalized_symbol: cleanSymbol,
    intent_boundary: "ETF_COCKPIT_INTENT",
    intent_type: "ETF_SYMBOL_SELECT",
    route: path,
    source_component: params.sourceComponent || "handleSelectSymbol",
    synthetic_marker: syntheticMarker,
    ci_marker: ciRunMarker,
    qa_marker: manualQaMarker,
    ci_run_marker: ciRunMarker,
    manual_qa_marker: manualQaMarker,
    user_agent_raw: userAgent,
  };
}

/**
 * Emit an explicit user ETF intent attempt to the canonical ingestion endpoint.
 *
 * MUST ONLY be called from deliberate user interaction handlers (e.g. handleSelectSymbol).
 * MUST NEVER be called from useEffect, render blocks, mount hooks, or polling intervals.
 */
export async function emitEtfIntentAttempt(
  symbol: string,
  sourceComponent: string = "handleSelectSymbol",
  existingAttemptId?: string
): Promise<EmitResult | null> {
  const payload = buildRawIntentPayload({
    symbol,
    sourceComponent,
    attemptId: existingAttemptId,
  });

  if (!payload) {
    return null;
  }

  const attemptId = payload.observation_unit_id || generateUuidV4();

  if (typeof window === "undefined") {
    return {
      emitted: false,
      attemptId,
      deduplicationKey: payload.deduplication_key,
      payload,
    };
  }

  const payloadString = JSON.stringify(payload);
  let transmitted = false;

  // Preference 1: navigator.sendBeacon (non-blocking, survives tab close)
  if (typeof navigator !== "undefined" && typeof navigator.sendBeacon === "function") {
    try {
      const blob = new Blob([payloadString], { type: "application/json" });
      transmitted = navigator.sendBeacon(TELEMETRY_INGESTION_ENDPOINT, blob);
    } catch {
      transmitted = false;
    }
  }

  // Preference 2: fetch with keepalive fallback if sendBeacon returns false or fails
  if (!transmitted && typeof fetch === "function") {
    try {
      fetch(TELEMETRY_INGESTION_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Attempt-Id": attemptId,
          "X-Deduplication-Key": payload.deduplication_key,
        },
        body: payloadString,
        keepalive: true,
      }).catch(() => {
        // Transport error caught; does not fabricate success
      });
      transmitted = true;
    } catch {
      transmitted = false;
    }
  }

  return {
    emitted: transmitted,
    attemptId,
    deduplicationKey: payload.deduplication_key,
    payload,
  };
}

/**
 * Emit an authorized Phase P2 ETF interaction event (e.g. view, vernacular toggle, sector expand).
 * Preserves P1 denominator invariance (downstream failure / interaction invariance).
 */
export async function emitEtfInteractionEvent(
  eventType: "ETF_RISK_PROFILE_VIEW" | "ETF_VERNACULAR_TOGGLE" | "ETF_SECTOR_ALLOCATION_EXPAND",
  symbol: string,
  extraMetadata?: Record<string, any>
): Promise<EmitResult | null> {
  const cleanSym = (symbol || "").trim().toUpperCase().replace(/.*:/, "");
  if (!cleanSym || !isETF(cleanSym)) {
    return null;
  }
  const sessionId = getSessionId();
  const attemptId = generateUuidV4();
  const dedupKey = computeDeduplicationKey(sessionId, cleanSym, `${eventType}:${attemptId}`);

  const isBrowser = typeof window !== "undefined";
  const searchParams = isBrowser && window.location ? window.location.search : "";
  const path = isBrowser && window.location ? window.location.pathname : "/";
  const userAgent = typeof navigator !== "undefined" ? navigator.userAgent : "";

  const syntheticMarker = Boolean(
    isBrowser &&
      ((window as any).__ARX_SYNTHETIC__ === true ||
        searchParams.includes("synthetic=true") ||
        searchParams.includes("arx_test=true"))
  );
  const ciRunMarker = Boolean(isBrowser && searchParams.includes("ci_run=true"));
  const manualQaMarker = Boolean(isBrowser && searchParams.includes("qa_run=true"));

  const payload: ClientIntentPayload = {
    schema_version: "1.0.0",
    event_id: generateUuidV4(),
    session_id: sessionId,
    observation_unit_id: attemptId,
    deduplication_key: dedupKey,
    timestamp: new Date().toISOString(),
    symbol: cleanSym,
    normalized_symbol: cleanSym,
    intent_boundary: "ETF_COCKPIT_INTERACTION",
    intent_type: eventType,
    route: path,
    source_component: "EtfRiskProfileCard",
    synthetic_marker: syntheticMarker,
    ci_marker: ciRunMarker,
    qa_marker: manualQaMarker,
    ci_run_marker: ciRunMarker,
    manual_qa_marker: manualQaMarker,
    user_agent_raw: userAgent,
    ...(extraMetadata || {}),
  };

  if (!isBrowser) {
    return { emitted: false, attemptId, deduplicationKey: dedupKey, payload };
  }

  const payloadString = JSON.stringify(payload);
  let transmitted = false;

  if (typeof navigator !== "undefined" && typeof navigator.sendBeacon === "function") {
    try {
      const blob = new Blob([payloadString], { type: "application/json" });
      transmitted = navigator.sendBeacon(TELEMETRY_INGESTION_ENDPOINT, blob);
    } catch {
      transmitted = false;
    }
  }

  if (!transmitted && typeof fetch === "function") {
    try {
      fetch(TELEMETRY_INGESTION_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Attempt-Id": attemptId,
          "X-Deduplication-Key": dedupKey,
        },
        body: payloadString,
        keepalive: true,
      }).catch(() => {});
      transmitted = true;
    } catch {
      transmitted = false;
    }
  }

  return {
    emitted: transmitted,
    attemptId,
    deduplicationKey: dedupKey,
    payload,
  };
}
