/**
 * ARX Terminal — Corporate Events Telemetry Contract & Validator (v1.0.0).
 *
 * Implements strict server-side validation, privacy deny-list scanning,
 * semantic invariant enforcement, and HMAC-SHA256 deduplication key computation.
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
  "synthetic_marker",
  "ci_marker",
  "manual_qa_marker",
]);

const SHA_REGEX = /^[0-9a-f]{40}$/;
const SYMBOL_REGEX = /^[A-Z0-9]{1,5}$/;
const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const VALID_CATEGORIES: Set<CorporateEventCategory> = new Set([
  "EARNINGS_ANNOUNCEMENT",
  "DIVIDEND",
  "SPLIT",
  "ACQUISITION",
  "NONE",
  "UNAVAILABLE",
]);

const VALID_SESSIONS: Set<MarketSession> = new Set(["BMO", "AMC", "DURING", "UNKNOWN"]);

const VALID_SOURCES: Set<EventSourceStatus> = new Set([
  "CONFIRMED",
  "UNVERIFIED",
  "DATA_UNAVAILABLE",
]);

const VALID_FRESHNESS: Set<FreshnessStatus> = new Set(["FRESH", "STALE", "UNVERIFIED"]);

const VALID_RISK_BUCKETS: Set<RiskBucket> = new Set([
  "ACUTE",
  "CAUTION",
  "NOMINAL",
  "UNAVAILABLE",
]);

const VALID_OUTCOMES: Set<EvaluationOutcome> = new Set([
  "COMPLETED",
  "UNAVAILABLE",
  "ERROR",
]);

/**
 * Recursively scans an object for prohibited privacy keys.
 */
export function scanForProhibitedFields(
  data: unknown,
  maxDepth = 5
): { detected: boolean; prohibitedField?: string } {
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
    for (const prohibited of PROHIBITED_PRIVACY_FIELDS) {
      if (normKey === prohibited.replace(/[-_]/g, "")) {
        return { detected: true, prohibitedField: key };
      }
    }
    const val = record[key];
    if (typeof val === "object" && val !== null) {
      const nestedRes = scanForProhibitedFields(val, maxDepth - 1);
      if (nestedRes.detected) return nestedRes;
    }
  }

  return { detected: false };
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  payload?: CorporateEventCanonicalPayload;
}

/**
 * Validates a payload against the 17-field canonical contract and decision engine invariants.
 */
export function validateCorporateEventPayload(data: unknown): ValidationResult {
  const errors: string[] = [];

  if (!data || typeof data !== "object" || Array.isArray(data)) {
    return { valid: false, errors: ["Payload must be a non-null JSON object"] };
  }

  const raw = data as Record<string, unknown>;

  // 1. Check for unexpected / forbidden extra fields
  for (const key of Object.keys(raw)) {
    if (!ALLOWED_CANONICAL_KEYS.has(key)) {
      errors.push(`Unknown extra field forbidden: '${key}'`);
    }
  }

  // 2. Validate required string fields
  if (raw.schema_version !== "1.0.0") {
    errors.push(`schema_version must be '1.0.0', got '${raw.schema_version}'`);
  }

  if (typeof raw.event_id !== "string" || !UUID_REGEX.test(raw.event_id)) {
    errors.push(`event_id must be a valid UUIDv4 string, got '${raw.event_id}'`);
  }

  if (typeof raw.occurred_at_utc !== "string" || !raw.occurred_at_utc.includes("T")) {
    errors.push(`occurred_at_utc must be a valid ISO 8601 UTC timestamp`);
  } else {
    const parsedDate = Date.parse(raw.occurred_at_utc);
    if (Number.isNaN(parsedDate)) {
      errors.push(`occurred_at_utc is not a parseable date timestamp`);
    }
  }

  if (typeof raw.release_sha !== "string" || !SHA_REGEX.test(raw.release_sha.toLowerCase())) {
    errors.push(`release_sha must be exactly 40 lowercase hexadecimal characters`);
  }

  if (raw.ruleset_version !== "RC5" && raw.ruleset_version !== "RC4") {
    errors.push(`ruleset_version must be 'RC5' or 'RC4', got '${raw.ruleset_version}'`);
  }

  if (typeof raw.evaluation_context_id !== "string" || raw.evaluation_context_id.trim().length === 0) {
    errors.push(`evaluation_context_id must be a non-empty string`);
  }

  if (typeof raw.symbol !== "string" || !SYMBOL_REGEX.test(raw.symbol.toUpperCase())) {
    errors.push(`symbol must be 1 to 5 uppercase alphanumeric characters`);
  }

  // 3. Enums
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

  // 4. Numeric & Semantic Rules
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

  // 5. Invariant Checks
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
    } else if (days < 0.0 && riskBucket !== "NOMINAL") {
      errors.push(`Boundary violation: elapsed event (${days} < 0.0) must be NOMINAL, got ${riskBucket}`);
    }
  } else {
    if (riskBucket !== "UNAVAILABLE" && riskBucket !== "NOMINAL") {
      errors.push(`Missing days_to_earnings requires UNAVAILABLE or NOMINAL, got ${riskBucket}`);
    }
  }

  if (errors.length > 0) {
    return { valid: false, errors };
  }

  const payload: CorporateEventCanonicalPayload = {
    schema_version: "1.0.0",
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
    synthetic_marker: Boolean(raw.synthetic_marker),
    ci_marker: Boolean(raw.ci_marker),
    manual_qa_marker: Boolean(raw.manual_qa_marker),
  };

  return { valid: true, errors: [], payload };
}

/**
 * Universal HMAC-SHA256 deduplication key derivation.
 * Works seamlessly in Browser (WebCrypto), Cloudflare Workers, and Node.js.
 */
export async function computeServerDeduplicationKey(
  evaluation_context_id: string,
  symbol: string,
  event_time_utc: string | null,
  evaluation_id: string,
  salt: string
): Promise<string> {
  const message = `ctx:${evaluation_context_id}|sym:${symbol.toUpperCase()}|event:${event_time_utc || "none"}|eval_id:${evaluation_id}`;
  const enc = new TextEncoder();

  if (typeof crypto !== "undefined" && crypto.subtle) {
    const key = await crypto.subtle.importKey(
      "raw",
      enc.encode(salt),
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["sign"]
    );
    const signature = await crypto.subtle.sign("HMAC", key, enc.encode(message));
    return Array.from(new Uint8Array(signature))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  }

  // Fallback for Node without crypto.subtle (dynamic require)
  try {
    const nodeCrypto = await import("crypto");
    return nodeCrypto.createHmac("sha256", salt).update(message).digest("hex");
  } catch (err) {
    throw new Error("No cryptographic provider available for HMAC computation");
  }
}
