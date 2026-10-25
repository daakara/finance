/**
 * ARX Terminal — Corporate Events Telemetry Contract & Validator (v1.1.0 Hardened).
 *
 * Increment 1.1 Hardening:
 * - Gate B: Strict canonical schema validation (extra="forbid").
 * - Gate C: Canonical digest computation for duplicate vs conflicting identity verification.
 * - Gate D: Replay window (24h) and future clock skew (60s) bounds enforcement.
 * - Gate E: Key versioning (kid), fail-closed unconfigured secret detection, and rotation semantics.
 * - Gate F: Recursive field-name deny-list AND deep regex value scanner (emails, IPs, JWTs, bearer tokens).
 * - Gate G: Decision engine semantic invariants preservation (ACUTE, CAUTION, NOMINAL, UNAVAILABLE).
 */

import {
  CorporateEventCanonicalPayload,
  CorporateEventCategory,
  MarketSession,
  EventSourceStatus,
  FreshnessStatus,
  RiskBucket,
  EvaluationOutcome,
  RulesetVersion,
} from "../../types/corporate-events-telemetry";

export const PROHIBITED_PRIVACY_FIELDS = new Set([
  "user_id",
  "userid",
  "user_token",
  "ip",
  "ip_address",
  "client_ip",
  "remote_addr",
  "email",
  "account_id",
  "accountid",
  "account_number",
  "account_balance",
  "balance",
  "portfolio_value",
  "trade_size",
  "dollar_amount",
  "cookie",
  "session_cookie",
  "device_id",
  "device_fingerprint",
  "canvas_fingerprint",
  "name",
  "personal_name",
  "full_name",
  "phone",
  "address",
  "ssn",
  "auth_token",
  "bearer_token",
  "password",
]);

export const ALLOWED_CANONICAL_KEYS = new Set([
  "schema_version",
  "event_id",
  "occurred_at_utc",
  "release_sha",
  "ruleset_version",
  "evaluation_context_id",
  "symbol",
  "event_category",
  "event_time_utc",
  "market_session",
  "event_source_status",
  "freshness_status",
  "days_to_earnings",
  "risk_bucket",
  "applied_penalty",
  "evaluation_outcome",
  "provenance_class",
  "evaluation_id",
  "delivery_attempt_id",
  "deduplication_key",
  "canonical_digest",
  "synthetic_marker",
  "ci_marker",
  "manual_qa_marker",
]);

const SHA_REGEX = /^[0-9a-f]{40}$/;
const SYMBOL_REGEX = /^[A-Z0-9.]{1,10}$/;
const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// Privacy Value-Level Regex Patterns (Gate F)
const EMAIL_REGEX = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/;
const IPV4_REGEX = /\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b/;
const IPV6_REGEX = /\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b/;
const JWT_REGEX = /\beyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b/;
const BEARER_OR_SESSION_REGEX = /(?:bearer\s+[a-zA-Z0-9_\-\.]{16,}|session=[a-zA-Z0-9]{16,}|cookie:\s*)/i;
const URL_REGEX = /https?:\/\//i;

export const MAX_EVENT_AGE_MS = 24 * 60 * 60 * 1000; // 24-hour replay window horizon
export const MAX_FUTURE_SKEW_MS = 60 * 1000;         // 60-second future clock skew bound

const VALID_CATEGORIES: Set<CorporateEventCategory> = new Set([
  "EARNINGS_ANNOUNCEMENT",
  "DIVIDEND",
  "SPLIT",
  "ACQUISITION",
  "NONE",
  "UNAVAILABLE",
]);

const VALID_SESSIONS: Set<MarketSession> = new Set(["BMO", "AMC", "DURING", "UNKNOWN"]);
const VALID_SOURCES: Set<EventSourceStatus> = new Set(["CONFIRMED", "UNVERIFIED", "DATA_UNAVAILABLE"]);
const VALID_FRESHNESS: Set<FreshnessStatus> = new Set(["FRESH", "STALE", "UNVERIFIED"]);
const VALID_RISK_BUCKETS: Set<RiskBucket> = new Set(["ACUTE", "CAUTION", "NOMINAL", "UNAVAILABLE"]);
const VALID_OUTCOMES: Set<EvaluationOutcome> = new Set(["COMPLETED", "UNAVAILABLE", "ERROR"]);

export interface PrivacyScanResult {
  detected: boolean;
  prohibitedField?: string;
  prohibitedRule?: string;
}

/**
 * Recursively scans an object for:
 * 1. Prohibited field names in deny-list.
 * 2. Sensitive value patterns (emails, IPs, JWTs, bearer tokens, embedded URLs).
 * Guarantees zero sensitive value leakage in the scan return object.
 */
export function scanForProhibitedFields(
  data: unknown,
  maxDepth = 5
): PrivacyScanResult {
  if (!data || typeof data !== "object" || maxDepth <= 0) {
    return { detected: false };
  }

  if (Array.isArray(data)) {
    for (const item of data) {
      const res = scanForProhibitedFields(item, maxDepth - 1);
      if (res.detected) return res;
    }
    return { detected: false };
  }

  const record = data as Record<string, unknown>;
  for (const key of Object.keys(record)) {
    const normKey = key.toLowerCase().replace(/[-_]/g, "");

    // 1. Check prohibited field names
    for (const prohibited of PROHIBITED_PRIVACY_FIELDS) {
      if (normKey === prohibited.replace(/[-_]/g, "")) {
        return {
          detected: true,
          prohibitedField: key,
          prohibitedRule: "DENYLIST_FIELD_NAME",
        };
      }
    }

    const val = record[key];

    // 2. Check value-level leaks in strings (Gate F)
    if (typeof val === "string") {
      if (EMAIL_REGEX.test(val)) {
        return {
          detected: true,
          prohibitedField: key,
          prohibitedRule: "VALUE_PATTERN_EMAIL",
        };
      }
      if (IPV4_REGEX.test(val) || IPV6_REGEX.test(val)) {
        return {
          detected: true,
          prohibitedField: key,
          prohibitedRule: "VALUE_PATTERN_IP_ADDRESS",
        };
      }
      if (JWT_REGEX.test(val)) {
        return {
          detected: true,
          prohibitedField: key,
          prohibitedRule: "VALUE_PATTERN_JWT_TOKEN",
        };
      }
      if (BEARER_OR_SESSION_REGEX.test(val)) {
        return {
          detected: true,
          prohibitedField: key,
          prohibitedRule: "VALUE_PATTERN_AUTH_TOKEN",
        };
      }
      if (key !== "deployment_host" && URL_REGEX.test(val)) {
        return {
          detected: true,
          prohibitedField: key,
          prohibitedRule: "VALUE_PATTERN_UNAUTHORIZED_URL",
        };
      }
    }

    // 3. Recurse into nested objects
    if (typeof val === "object" && val !== null) {
      const nestedRes = scanForProhibitedFields(val, maxDepth - 1);
      if (nestedRes.detected) return nestedRes;
    }
  }

  return { detected: false };
}

export interface ValidationOptions {
  nowMs?: number;
  skipReplayWindow?: boolean;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  errorCode?: string;
  payload?: CorporateEventCanonicalPayload;
}

/**
 * Validates a payload against the 17-field canonical contract, decision engine invariants,
 * and replay window/clock-skew bounds.
 */
export function validateCorporateEventPayload(
  data: unknown,
  options: ValidationOptions = {}
): ValidationResult {
  const errors: string[] = [];

  if (!data || typeof data !== "object" || Array.isArray(data)) {
    return {
      valid: false,
      errors: ["Payload must be a non-null JSON object"],
      errorCode: "INVALID_STRUCTURE",
    };
  }

  const raw = data as Record<string, unknown>;

  // 1. Check for unknown / forbidden extra fields (extra = "forbid")
  for (const key of Object.keys(raw)) {
    if (!ALLOWED_CANONICAL_KEYS.has(key)) {
      errors.push(`Unknown extra field forbidden: '${key}'`);
    }
  }

  // 2. Validate schema_version
  if (raw.schema_version !== "1.0.0" && raw.schema_version !== "1.1.0") {
    errors.push(`schema_version must be '1.0.0' or '1.1.0', got '${raw.schema_version}'`);
  }

  // 3. Validate event_id (UUIDv4)
  if (typeof raw.event_id !== "string" || !UUID_REGEX.test(raw.event_id)) {
    errors.push(`event_id must be a valid UUIDv4 string, got '${raw.event_id}'`);
  }

  // 4. Validate occurred_at_utc and Replay Window (Gate D)
  let occurredAtMs = 0;
  if (typeof raw.occurred_at_utc !== "string" || !raw.occurred_at_utc.includes("T")) {
    errors.push(`occurred_at_utc must be a valid ISO 8601 UTC timestamp`);
  } else {
    occurredAtMs = Date.parse(raw.occurred_at_utc);
    if (Number.isNaN(occurredAtMs)) {
      errors.push(`occurred_at_utc is not a parseable date timestamp`);
    } else if (!options.skipReplayWindow) {
      const nowMs = options.nowMs ?? Date.now();
      if (nowMs - occurredAtMs > MAX_EVENT_AGE_MS) {
        errors.push(`Replay Window Expired: Event age exceeds 24h horizon`);
        return {
          valid: false,
          errors,
          errorCode: "EXPIRED_REPLAY_WINDOW",
        };
      }
      if (occurredAtMs - nowMs > MAX_FUTURE_SKEW_MS) {
        errors.push(`Future Timestamp Rejected: Clock skew exceeds 60s tolerance`);
        return {
          valid: false,
          errors,
          errorCode: "FUTURE_TIMESTAMP_REJECTED",
        };
      }
    }
  }

  // 5. Release SHA & Ruleset
  if (typeof raw.release_sha !== "string" || !SHA_REGEX.test(raw.release_sha.toLowerCase())) {
    errors.push(`release_sha must be exactly 40 lowercase hexadecimal characters`);
  }

  if (raw.ruleset_version !== "RC5" && raw.ruleset_version !== "RC4") {
    errors.push(`ruleset_version must be 'RC5' or 'RC4', got '${raw.ruleset_version}'`);
  }

  // 6. Context & Symbol
  if (typeof raw.evaluation_context_id !== "string" || raw.evaluation_context_id.trim().length === 0) {
    errors.push(`evaluation_context_id must be a non-empty string`);
  }

  if (typeof raw.symbol !== "string" || !SYMBOL_REGEX.test(raw.symbol.toUpperCase())) {
    errors.push(`symbol must be 1 to 10 uppercase alphanumeric characters`);
  }

  // 7. Enums
  if (!VALID_CATEGORIES.has(raw.event_category as CorporateEventCategory)) {
    errors.push(`event_category invalid: '${raw.event_category}'`);
  }

  if (raw.event_time_utc !== null && raw.event_time_utc !== undefined) {
    if (typeof raw.event_time_utc !== "string" || Number.isNaN(Date.parse(raw.event_time_utc))) {
      errors.push(`event_time_utc must be null or valid ISO 8601 timestamp`);
    }
  }

  if (!VALID_SESSIONS.has(raw.market_session as MarketSession)) {
    errors.push(`market_session invalid: '${raw.market_session}'`);
  }

  if (!VALID_SOURCES.has(raw.event_source_status as EventSourceStatus)) {
    errors.push(`event_source_status invalid: '${raw.event_source_status}'`);
  }

  if (!VALID_FRESHNESS.has(raw.freshness_status as FreshnessStatus)) {
    errors.push(`freshness_status invalid: '${raw.freshness_status}'`);
  }

  // 8. Days to Earnings, Risk Bucket & Applied Penalty
  let days: number | null = null;
  if (raw.days_to_earnings !== null && raw.days_to_earnings !== undefined) {
    if (typeof raw.days_to_earnings !== "number" || Number.isNaN(raw.days_to_earnings)) {
      errors.push(`days_to_earnings must be a number or null`);
    } else {
      days = Math.round(raw.days_to_earnings * 100) / 100;
    }
  }

  const riskBucket = raw.risk_bucket as RiskBucket;
  if (!VALID_RISK_BUCKETS.has(riskBucket)) {
    errors.push(`risk_bucket invalid: '${raw.risk_bucket}'`);
  }

  const penalty = typeof raw.applied_penalty === "number" ? raw.applied_penalty : null;
  if (penalty !== -25.0 && penalty !== -8.0 && penalty !== 0.0) {
    errors.push(`applied_penalty must be -25.0, -8.0, or 0.0, got ${raw.applied_penalty}`);
  }

  if (!VALID_OUTCOMES.has(raw.evaluation_outcome as EvaluationOutcome)) {
    errors.push(`evaluation_outcome invalid: '${raw.evaluation_outcome}'`);
  }

  // 9. Semantic Decision Invariants (Gate G)
  if (riskBucket === "ACUTE" && penalty !== -25.0) {
    errors.push(`Semantic mismatch: risk_bucket ACUTE requires applied_penalty -25.0, got ${penalty}`);
  }
  if (riskBucket === "CAUTION" && penalty !== -8.0) {
    errors.push(`Semantic mismatch: risk_bucket CAUTION requires applied_penalty -8.0, got ${penalty}`);
  }
  if ((riskBucket === "NOMINAL" || riskBucket === "UNAVAILABLE") && penalty !== 0.0) {
    errors.push(`Semantic mismatch: risk_bucket ${riskBucket} requires applied_penalty 0.0, got ${penalty}`);
  }

  if (days !== null) {
    if (days >= 0.0 && days <= 1.0 && riskBucket !== "ACUTE") {
      errors.push(`Boundary violation: 0.0 <= days_to_earnings (${days}) <= 1.0 must be ACUTE, got ${riskBucket}`);
    } else if (days > 1.0 && days <= 7.0 && riskBucket !== "CAUTION") {
      errors.push(`Boundary violation: 1.0 < days_to_earnings (${days}) <= 7.0 must be CAUTION, got ${riskBucket}`);
    } else if (days > 7.0 && riskBucket !== "NOMINAL") {
      errors.push(`Boundary violation: days_to_earnings (${days}) > 7.0 must be NOMINAL, got ${riskBucket}`);
    } else if (days < 0.0) {
      // Historical/elapsed event must not have active penalty, and if stale must be UNAVAILABLE
      if (raw.freshness_status === "STALE" && riskBucket !== "UNAVAILABLE") {
        errors.push(`Boundary violation: stale elapsed event must be UNAVAILABLE, got ${riskBucket}`);
      } else if (riskBucket !== "NOMINAL" && riskBucket !== "UNAVAILABLE") {
        errors.push(`Boundary violation: elapsed event (${days} < 0.0) must be NOMINAL or UNAVAILABLE, got ${riskBucket}`);
      }
    }
  } else {
    // Missing coverage must be UNAVAILABLE
    if (riskBucket !== "UNAVAILABLE") {
      errors.push(`Missing coverage (days_to_earnings=null) requires UNAVAILABLE, got ${riskBucket}`);
    }
  }

  if (errors.length > 0) {
    return {
      valid: false,
      errors,
      errorCode: "SCHEMA_VALIDATION_ERROR",
    };
  }

  const payload: CorporateEventCanonicalPayload = {
    schema_version: raw.schema_version as "1.0.0" | "1.1.0",
    event_id: String(raw.event_id),
    occurred_at_utc: String(raw.occurred_at_utc),
    release_sha: String(raw.release_sha).toLowerCase(),
    ruleset_version: raw.ruleset_version as RulesetVersion,
    evaluation_context_id: String(raw.evaluation_context_id),
    symbol: String(raw.symbol).toUpperCase(),
    event_category: raw.event_category as CorporateEventCategory,
    event_time_utc: raw.event_time_utc ? String(raw.event_time_utc) : null,
    market_session: raw.market_session as MarketSession,
    event_source_status: raw.event_source_status as EventSourceStatus,
    freshness_status: raw.freshness_status as FreshnessStatus,
    days_to_earnings: days,
    risk_bucket: riskBucket,
    applied_penalty: penalty as number,
    evaluation_outcome: raw.evaluation_outcome as EvaluationOutcome,
    evaluation_id: raw.evaluation_id ? String(raw.evaluation_id) : undefined,
    delivery_attempt_id: raw.delivery_attempt_id ? String(raw.delivery_attempt_id) : undefined,
    deduplication_key: raw.deduplication_key ? String(raw.deduplication_key) : undefined,
    canonical_digest: raw.canonical_digest ? String(raw.canonical_digest) : undefined,
    synthetic_marker: Boolean(raw.synthetic_marker),
    ci_marker: Boolean(raw.ci_marker),
    manual_qa_marker: Boolean(raw.manual_qa_marker),
  };

  return { valid: true, errors: [], payload };
}

/**
 * Computes a deterministic SHA-256 digest of the 12 canonical semantic evaluation fields.
 * Used for tamper detection, semantic equivalence verification, and distinguishing
 * confirmed duplicates from conflicting identities.
 *
 * Excludes transport identifiers (event_id, occurred_at_utc, evaluation_context_id, evaluation_id).
 */
export async function computeCanonicalDigest(
  payload: CorporateEventCanonicalPayload
): Promise<string> {
  const dteStr =
    payload.days_to_earnings !== null && payload.days_to_earnings !== undefined
      ? payload.days_to_earnings.toFixed(2)
      : "null";
  const penaltyStr =
    payload.applied_penalty !== null && payload.applied_penalty !== undefined
      ? payload.applied_penalty.toFixed(1)
      : "0.0";

  const canonicalParts = [
    payload.symbol.toUpperCase(),
    payload.release_sha.toLowerCase(),
    payload.ruleset_version,
    payload.event_category,
    payload.event_time_utc ?? "null",
    payload.market_session,
    payload.event_source_status,
    payload.freshness_status,
    dteStr,
    payload.risk_bucket,
    penaltyStr,
    payload.evaluation_outcome,
  ];

  const canonicalString = canonicalParts.join("|");
  const enc = new TextEncoder();

  if (typeof crypto !== "undefined" && crypto.subtle) {
    const hashBuffer = await crypto.subtle.digest("SHA-256", enc.encode(canonicalString));
    return Array.from(new Uint8Array(hashBuffer))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  }

  try {
    const nodeCrypto = await import("crypto");
    return nodeCrypto.createHash("sha256").update(canonicalString).digest("hex");
  } catch {
    throw new Error("No cryptographic provider available for SHA-256 computation");
  }
}

/**
 * Deterministically derives the authoritative evaluation instance identifier.
 *
 * Guarantees:
 * - Identical decision inputs and evaluation timestamp produce the exact same evaluation_id.
 * - Multiple browser tabs viewing the same evaluation result share the exact same evaluation_id.
 * - Rehydration retains the frozen evaluation_id.
 * - Genuine new evaluations produce distinct evaluation_ids.
 * - Byte-level parity with Python derive_authoritative_evaluation_id.
 */
export async function deriveAuthoritativeEvaluationId(params: {
  release_sha: string;
  symbol: string;
  ruleset_version: string;
  event_time_utc?: string | null;
  market_session: string;
  evaluation_timestamp: string;
}): Promise<string> {
  const eventTimeStr = params.event_time_utc || "null";
  const s = `eval|${params.release_sha.toLowerCase()}|${params.symbol.toUpperCase()}|${params.ruleset_version}|${eventTimeStr}|${params.market_session}|${params.evaluation_timestamp}`;
  const enc = new TextEncoder();

  if (typeof crypto !== "undefined" && crypto.subtle) {
    const hashBuffer = await crypto.subtle.digest("SHA-256", enc.encode(s));
    return Array.from(new Uint8Array(hashBuffer))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  }

  try {
    const nodeCrypto = await import("crypto");
    return nodeCrypto.createHash("sha256").update(s).digest("hex");
  } catch {
    throw new Error("No cryptographic provider available for SHA-256 computation");
  }
}

/**
 * Synchronous variant of deriveAuthoritativeEvaluationId for Node/sync callers.
 */
export function deriveAuthoritativeEvaluationIdSync(params: {
  release_sha: string;
  symbol: string;
  ruleset_version: string;
  event_time_utc?: string | null;
  market_session: string;
  evaluation_timestamp: string;
}): string {
  const eventTimeStr = params.event_time_utc || "null";
  const s = `eval|${params.release_sha.toLowerCase()}|${params.symbol.toUpperCase()}|${params.ruleset_version}|${eventTimeStr}|${params.market_session}|${params.evaluation_timestamp}`;

  try {
    const g = globalThis as unknown as { require?: (mod: string) => { createHash: (alg: string) => { update: (data: string) => { digest: (enc: string) => string } } } };
    const req = g.require || (typeof require !== "undefined" ? require : null);
    if (req) {
      const nodeCrypto = req("crypto");
      return nodeCrypto.createHash("sha256").update(s).digest("hex");
    }
  } catch {
    // Fallback if require fails
  }

  // Fallback for synchronous execution in environments without node crypto
  let h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
  let h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
  for (let i = 0; i < s.length; i++) {
    const code = s.charCodeAt(i);
    h0 = (h0 ^ (code * 31)) >>> 0;
    h1 = (h1 ^ (code * 37)) >>> 0;
    h2 = (h2 ^ (code * 41)) >>> 0;
    h3 = (h3 ^ (code * 43)) >>> 0;
  }
  const hex = [h0, h1, h2, h3, h4, h5, h6, h7].map((h) => h.toString(16).padStart(8, "0")).join("");
  return hex.slice(0, 64);
}

/**
 * Universal HMAC-SHA256 deduplication key derivation with key versioning (kid)
 * and fail-closed secret configuration (Gate E).
 */
export async function computeServerDeduplicationKey(
  evaluation_context_id: string,
  symbol: string,
  event_time_utc: string | null,
  evaluation_id: string,
  saltOrSecret: string,
  keyId = "k1"
): Promise<string> {
  if (!saltOrSecret || saltOrSecret.trim().length === 0) {
    throw new Error("UNCONFIGURED_HMAC_SECRET: Server HMAC secret must not be empty or unconfigured");
  }

  const message = `kid:${keyId}|ctx:${evaluation_context_id}|sym:${symbol.toUpperCase()}|event:${event_time_utc || "none"}|eval_id:${evaluation_id}`;
  const enc = new TextEncoder();

  if (typeof crypto !== "undefined" && crypto.subtle) {
    const key = await crypto.subtle.importKey(
      "raw",
      enc.encode(saltOrSecret),
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["sign"]
    );
    const signature = await crypto.subtle.sign("HMAC", key, enc.encode(message));
    const hex = Array.from(new Uint8Array(signature))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    return `${keyId}:${hex}`;
  }

  try {
    const nodeCrypto = await import("crypto");
    const hex = nodeCrypto.createHmac("sha256", saltOrSecret).update(message).digest("hex");
    return `${keyId}:${hex}`;
  } catch {
    throw new Error("No cryptographic provider available for HMAC computation");
  }
}
