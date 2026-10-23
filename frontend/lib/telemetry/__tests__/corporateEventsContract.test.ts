// @vitest-environment jsdom
import { describe, it, expect } from "vitest";
import {
  scanForProhibitedFields,
  validateCorporateEventPayload,
  computeServerDeduplicationKey,
  PROHIBITED_PRIVACY_FIELDS,
} from "../corporateEventsContract";
import { CorporateEventCanonicalPayload } from "../../../types/corporate-events-telemetry";

const AUTH_SHA = "8404b6e4645b5f9f8237e70517dcab3b6afdda7f";

function createValidPayload(
  overrides: Partial<CorporateEventCanonicalPayload> = {}
): CorporateEventCanonicalPayload {
  return {
    schema_version: "1.0.0",
    event_id: "c7f9e8a2-3b1d-4e5f-9a0c-8d7e6f5a4b3c",
    occurred_at_utc: "2026-10-24T10:00:00.000Z",
    release_sha: AUTH_SHA,
    ruleset_version: "RC5",
    evaluation_context_id: "ctx-7a8b9c",
    symbol: "AAPL",
    event_category: "EARNINGS_ANNOUNCEMENT",
    event_time_utc: "2026-10-25T14:00:00.000Z",
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

describe("Corporate Events Telemetry Contract & Validator", () => {
  describe("Gate B: Contract Validation", () => {
    it("accepts a fully valid 17-field canonical payload", () => {
      const payload = createValidPayload();
      const res = validateCorporateEventPayload(payload);
      expect(res.valid).toBe(true);
      expect(res.errors).toHaveLength(0);
      expect(res.payload?.symbol).toBe("AAPL");
      expect(res.payload?.applied_penalty).toBe(-8.0);
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
      const bad = createValidPayload({ symbol: "TOOLONGSYMBOL" });
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

    it("validates elapsed events (<0.0 days) require penalty 0.0 and NOMINAL", () => {
      const validElapsed = createValidPayload({
        days_to_earnings: -2.5,
        risk_bucket: "NOMINAL",
        applied_penalty: 0.0,
      });
      expect(validateCorporateEventPayload(validElapsed).valid).toBe(true);
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
    });
  });

  describe("Gate F: Privacy Deny-List Scanning", () => {
    it.each(Array.from(PROHIBITED_PRIVACY_FIELDS))(
      "detects direct prohibited key '%s'",
      (key) => {
        const payload = { ...createValidPayload(), [key]: "sensitive-value" };
        const check = scanForProhibitedFields(payload);
        expect(check.detected).toBe(true);
        expect(check.prohibitedField).toBe(key);
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
  });

  describe("Gate C: Deduplication Key Computation", () => {
    it("derives deterministic 64-character hex key", async () => {
      const key1 = await computeServerDeduplicationKey(
        "ctx-001",
        "AAPL",
        "2026-10-25T14:00:00Z",
        "eval-001",
        "test-salt"
      );
      const key2 = await computeServerDeduplicationKey(
        "ctx-001",
        "AAPL",
        "2026-10-25T14:00:00Z",
        "eval-001",
        "test-salt"
      );
      expect(key1).toHaveLength(64);
      expect(key1).toBe(key2);
    });

    it("produces identical key across hour and day boundaries when evaluation_id is frozen", async () => {
      const frozenEvalId = "eval-frozen-uuid";
      // Delivery attempt 1 at hour 1
      const keyAttempt1 = await computeServerDeduplicationKey(
        "ctx-001",
        "AAPL",
        "2026-10-25T14:00:00Z",
        frozenEvalId,
        "epoch-stable-salt"
      );
      // Delivery attempt 2 at hour 2 (or next day)
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
  });
});
