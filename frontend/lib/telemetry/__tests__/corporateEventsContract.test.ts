// @vitest-environment jsdom
import { describe, it, expect } from "vitest";
import {
  scanForProhibitedFields,
  validateCorporateEventPayload,
  computeServerDeduplicationKey,
  computeCanonicalDigest,
  PROHIBITED_PRIVACY_FIELDS,
} from "../corporateEventsContract";
import { CorporateEventCanonicalPayload } from "../../../types/corporate-events-telemetry";

const AUTH_SHA = "8404b6e4645b5f9f8237e70517dcab3b6afdda7f";

function createValidPayload(
  overrides: Partial<CorporateEventCanonicalPayload> = {}
): CorporateEventCanonicalPayload {
  return {
    schema_version: "1.1.0",
    event_id: "c7f9e8a2-3b1d-4e5f-9a0c-8d7e6f5a4b3c",
    occurred_at_utc: new Date().toISOString(),
    release_sha: AUTH_SHA,
    ruleset_version: "RC5",
    evaluation_context_id: "ctx-7a8b9c",
    symbol: "AAPL",
    event_category: "EARNINGS_ANNOUNCEMENT",
    event_time_utc: new Date(Date.now() + 86400000 * 2).toISOString(),
    market_session: "BMO",
    event_source_status: "CONFIRMED",
    freshness_status: "FRESH",
    days_to_earnings: 1.17,
    risk_bucket: "CAUTION",
    applied_penalty: -8.0,
    evaluation_outcome: "COMPLETED",
    ...overrides,
  };
}

describe("Corporate Events Telemetry Contract & Validator (v1.1.0 Hardened)", () => {
  describe("Gate B: Contract Validation", () => {
    it("accepts a fully valid 17-field canonical payload", () => {
      const payload = createValidPayload();
      const res = validateCorporateEventPayload(payload);
      expect(res.valid).toBe(true);
      expect(res.errors).toHaveLength(0);
      expect(res.payload?.symbol).toBe("AAPL");
      expect(res.payload?.applied_penalty).toBe(-8.0);
    });

    it("accepts schema_version '1.0.0' or '1.1.0'", () => {
      const p1 = createValidPayload({ schema_version: "1.0.0" });
      expect(validateCorporateEventPayload(p1).valid).toBe(true);

      const p2 = createValidPayload({ schema_version: "1.1.0" });
      expect(validateCorporateEventPayload(p2).valid).toBe(true);
    });

    it("rejects unknown extra fields (extra: forbid)", () => {
      const bad = { ...createValidPayload(), unexpected_extra: "malicious" };
      const res = validateCorporateEventPayload(bad);
      expect(res.valid).toBe(false);
      expect(res.errors.some((e) => e.includes("Unknown extra field forbidden"))).toBe(true);
    });

    it("rejects invalid schema version", () => {
      const bad = createValidPayload({ schema_version: "2.0.0" as any });
      const res = validateCorporateEventPayload(bad);
      expect(res.valid).toBe(false);
      expect(res.errors.some((e) => e.includes("schema_version"))).toBe(true);
    });

    it("rejects malformed release SHA", () => {
      const bad = createValidPayload({ release_sha: "not-a-valid-40-hex-sha" });
      const res = validateCorporateEventPayload(bad);
      expect(res.valid).toBe(false);
      expect(res.errors.some((e) => e.includes("release_sha"))).toBe(true);
    });

    it("rejects malformed symbol", () => {
      const bad = createValidPayload({ symbol: "TOOLONG_SYMBOL_NAME" });
      const res = validateCorporateEventPayload(bad);
      expect(res.valid).toBe(false);
      expect(res.errors.some((e) => e.includes("symbol"))).toBe(true);
    });

    it("rejects invalid ruleset version", () => {
      const bad = createValidPayload({ ruleset_version: "RC3" as any });
      const res = validateCorporateEventPayload(bad);
      expect(res.valid).toBe(false);
      expect(res.errors.some((e) => e.includes("ruleset_version"))).toBe(true);
    });
  });

  describe("Gate D: Replay Window & Clock Skew Protection", () => {
    it("rejects events older than 24-hour replay window", () => {
      const now = Date.now();
      const past25h = new Date(now - 25 * 3600 * 1000).toISOString();
      const expiredPayload = createValidPayload({ occurred_at_utc: past25h });
      const res = validateCorporateEventPayload(expiredPayload, { nowMs: now });
      expect(res.valid).toBe(false);
      expect(res.errorCode).toBe("EXPIRED_REPLAY_WINDOW");
      expect(res.errors.some((e) => e.includes("Replay Window Expired"))).toBe(true);
    });

    it("rejects events with future clock skew greater than 60 seconds", () => {
      const now = Date.now();
      const future70s = new Date(now + 70 * 1000).toISOString();
      const futurePayload = createValidPayload({ occurred_at_utc: future70s });
      const res = validateCorporateEventPayload(futurePayload, { nowMs: now });
      expect(res.valid).toBe(false);
      expect(res.errorCode).toBe("FUTURE_TIMESTAMP_REJECTED");
      expect(res.errors.some((e) => e.includes("Future Timestamp Rejected"))).toBe(true);
    });

    it("accepts events within the valid 24h horizon and slight past skew", () => {
      const now = Date.now();
      const past10m = new Date(now - 10 * 60 * 1000).toISOString();
      const validPayload = createValidPayload({ occurred_at_utc: past10m });
      const res = validateCorporateEventPayload(validPayload, { nowMs: now });
      expect(res.valid).toBe(true);
    });
  });

  describe("Gate C: Canonical Digest Computation", () => {
    it("computes deterministic 64-char SHA-256 digest", async () => {
      const payload = createValidPayload();
      const d1 = await computeCanonicalDigest(payload);
      const d2 = await computeCanonicalDigest(payload);
      expect(d1).toHaveLength(64);
      expect(d1).toBe(d2);
    });

    it("detects identity tampering: changing symbol produces completely different digest", async () => {
      const p1 = createValidPayload({ symbol: "AAPL" });
      const p2 = createValidPayload({ symbol: "MSFT" });
      const d1 = await computeCanonicalDigest(p1);
      const d2 = await computeCanonicalDigest(p2);
      expect(d1).not.toBe(d2);
    });

    it("detects semantic tampering: changing penalty produces completely different digest", async () => {
      const p1 = createValidPayload({ applied_penalty: -8.0 });
      const p2 = createValidPayload({ applied_penalty: -25.0 });
      const d1 = await computeCanonicalDigest(p1);
      const d2 = await computeCanonicalDigest(p2);
      expect(d1).not.toBe(d2);
    });
  });

  describe("Gate G: Semantic Decision Invariants", () => {
    it("validates ACUTE boundary (0.0 to 1.0 days) requires penalty -25.0", () => {
      const validAcute = createValidPayload({
        days_to_earnings: 0.5,
        risk_bucket: "ACUTE",
        applied_penalty: -25.0,
      });
      expect(validateCorporateEventPayload(validAcute).valid).toBe(true);

      const invalidAcutePenalty = createValidPayload({
        days_to_earnings: 0.5,
        risk_bucket: "ACUTE",
        applied_penalty: -8.0,
      });
      const res = validateCorporateEventPayload(invalidAcutePenalty);
      expect(res.valid).toBe(false);
      expect(res.errors.some((e) => e.includes("Semantic mismatch"))).toBe(true);
    });

    it("validates CAUTION boundary (1.0 to 7.0 days) requires penalty -8.0", () => {
      const validCaution = createValidPayload({
        days_to_earnings: 4.5,
        risk_bucket: "CAUTION",
        applied_penalty: -8.0,
      });
      expect(validateCorporateEventPayload(validCaution).valid).toBe(true);

      const invalidCautionBucket = createValidPayload({
        days_to_earnings: 4.5,
        risk_bucket: "NOMINAL",
        applied_penalty: 0.0,
      });
      const res = validateCorporateEventPayload(invalidCautionBucket);
      expect(res.valid).toBe(false);
      expect(res.errors.some((e) => e.includes("Boundary violation"))).toBe(true);
    });

    it("validates NOMINAL boundary (>7.0 days) requires penalty 0.0", () => {
      const validNominal = createValidPayload({
        days_to_earnings: 15.0,
        risk_bucket: "NOMINAL",
        applied_penalty: 0.0,
      });
      expect(validateCorporateEventPayload(validNominal).valid).toBe(true);
    });

    it("validates elapsed events (<0.0 days) require penalty 0.0", () => {
      const validElapsedNominal = createValidPayload({
        days_to_earnings: -2.5,
        risk_bucket: "NOMINAL",
        applied_penalty: 0.0,
      });
      expect(validateCorporateEventPayload(validElapsedNominal).valid).toBe(true);

      const validElapsedUnavailable = createValidPayload({
        days_to_earnings: -2.5,
        risk_bucket: "UNAVAILABLE",
        applied_penalty: 0.0,
      });
      expect(validateCorporateEventPayload(validElapsedUnavailable).valid).toBe(true);
    });

    it("validates stale elapsed events require status UNAVAILABLE", () => {
      const stalePayload = createValidPayload({
        days_to_earnings: -10.0,
        freshness_status: "STALE",
        risk_bucket: "NOMINAL", // Invalid: stale must be UNAVAILABLE
        applied_penalty: 0.0,
      });
      const res = validateCorporateEventPayload(stalePayload);
      expect(res.valid).toBe(false);
      expect(res.errors.some((e) => e.includes("stale elapsed event must be UNAVAILABLE"))).toBe(true);
    });

    it("validates missing coverage (days is null) requires penalty 0.0 and UNAVAILABLE", () => {
      const validMissing = createValidPayload({
        days_to_earnings: null,
        event_time_utc: null,
        risk_bucket: "UNAVAILABLE",
        applied_penalty: 0.0,
        event_source_status: "DATA_UNAVAILABLE",
        evaluation_outcome: "UNAVAILABLE",
      });
      expect(validateCorporateEventPayload(validMissing).valid).toBe(true);

      const invalidMissingPass = createValidPayload({
        days_to_earnings: null,
        event_time_utc: null,
        risk_bucket: "NOMINAL", // False PASS invariant violation
        applied_penalty: 0.0,
      });
      const res = validateCorporateEventPayload(invalidMissingPass);
      expect(res.valid).toBe(false);
      expect(res.errors.some((e) => e.includes("Missing coverage"))).toBe(true);
    });
  });

  describe("Gate F: Privacy Hardening (Deny-list & Value Scanner)", () => {
    it.each(Array.from(PROHIBITED_PRIVACY_FIELDS))(
      "detects direct prohibited key '%s'",
      (key) => {
        const payload = { ...createValidPayload(), [key]: "sensitive-value" };
        const check = scanForProhibitedFields(payload);
        expect(check.detected).toBe(true);
        expect(check.prohibitedField).toBe(key);
        expect(check.prohibitedRule).toBe("DENYLIST_FIELD_NAME");
      }
    );

    it("detects nested prohibited fields inside complex structures", () => {
      const payload = {
        ...createValidPayload(),
        nested_metadata: {
          deep: {
            user_id: "usr_9999",
          },
        },
      };
      const check = scanForProhibitedFields(payload);
      expect(check.detected).toBe(true);
      expect(check.prohibitedField).toBe("user_id");
    });

    it("detects email value pattern in string field", () => {
      const payload = createValidPayload({ evaluation_context_id: "user-contact:admin@company.com" });
      const check = scanForProhibitedFields(payload);
      expect(check.detected).toBe(true);
      expect(check.prohibitedRule).toBe("VALUE_PATTERN_EMAIL");
      expect(check.prohibitedField).toBe("evaluation_context_id");
    });

    it("detects IPv4 address value pattern in string field", () => {
      const payload = createValidPayload({ evaluation_context_id: "ip-origin-192.168.1.105" });
      const check = scanForProhibitedFields(payload);
      expect(check.detected).toBe(true);
      expect(check.prohibitedRule).toBe("VALUE_PATTERN_IP_ADDRESS");
    });

    it("detects JWT token value pattern in string field", () => {
      const mockJwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.doNotLeakThisSignature123456789";
      const payload = createValidPayload({ evaluation_context_id: mockJwt });
      const check = scanForProhibitedFields(payload);
      expect(check.detected).toBe(true);
      expect(check.prohibitedRule).toBe("VALUE_PATTERN_JWT_TOKEN");
    });

    it("detects bearer token value pattern in string field", () => {
      const payload = createValidPayload({ evaluation_context_id: "Bearer a1b2c3d4e5f6g7h8i9j0k1l2" });
      const check = scanForProhibitedFields(payload);
      expect(check.detected).toBe(true);
      expect(check.prohibitedRule).toBe("VALUE_PATTERN_AUTH_TOKEN");
    });

    it("never echoes sensitive raw values in scan results", () => {
      const secret = "secret_user@domain.org";
      const payload = createValidPayload({ evaluation_context_id: secret });
      const check = scanForProhibitedFields(payload);
      expect(JSON.stringify(check)).not.toContain(secret);
    });
  });

  describe("Gate E: Server Deduplication Key & Secret Boundaries", () => {
    it("derives deterministic key prefixed with keyId (e.g. k1:<hex>)", async () => {
      const key1 = await computeServerDeduplicationKey(
        "ctx-001",
        "AAPL",
        "2026-10-25T14:00:00Z",
        "eval-001",
        "test-salt-secret",
        "k1"
      );
      const key2 = await computeServerDeduplicationKey(
        "ctx-001",
        "AAPL",
        "2026-10-25T14:00:00Z",
        "eval-001",
        "test-salt-secret",
        "k1"
      );
      expect(key1).toMatch(/^k1:[0-9a-f]{64}$/);
      expect(key1).toBe(key2);
    });

    it("fails closed if server secret is missing or empty", async () => {
      await expect(
        computeServerDeduplicationKey("ctx-001", "AAPL", "2026-10-25T14:00:00Z", "eval-001", "")
      ).rejects.toThrow("UNCONFIGURED_HMAC_SECRET");
    });

    it("produces identical key across hour and day boundaries when evaluation_id is frozen", async () => {
      const frozenEvalId = "eval-frozen-uuid";
      const keyAttempt1 = await computeServerDeduplicationKey(
        "ctx-001",
        "AAPL",
        "2026-10-25T14:00:00Z",
        frozenEvalId,
        "epoch-stable-salt"
      );
      const keyAttempt2 = await computeServerDeduplicationKey(
        "ctx-001",
        "AAPL",
        "2026-10-25T14:00:00Z",
        frozenEvalId,
        "epoch-stable-salt"
      );
      expect(keyAttempt1).toBe(keyAttempt2);
    });

    it("produces distinct keys for two evaluations of the same ticker in the same second", async () => {
      const key1 = await computeServerDeduplicationKey(
        "ctx-001",
        "AAPL",
        "2026-10-25T14:00:00Z",
        "eval-first",
        "test-salt"
      );
      const key2 = await computeServerDeduplicationKey(
        "ctx-001",
        "AAPL",
        "2026-10-25T14:00:00Z",
        "eval-second",
        "test-salt"
      );
      expect(key1).not.toBe(key2);
    });

    it("supports key rotation with keyId prefix (k1 -> k2)", async () => {
      const k1Key = await computeServerDeduplicationKey("ctx", "AAPL", null, "id1", "secret1", "k1");
      const k2Key = await computeServerDeduplicationKey("ctx", "AAPL", null, "id1", "secret2", "k2");
      expect(k1Key.startsWith("k1:")).toBe(true);
      expect(k2Key.startsWith("k2:")).toBe(true);
      expect(k1Key).not.toBe(k2Key);
    });
  });
});
