// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import {
  generateUuidV4,
  getSessionId,
  resetSession,
  computeDeduplicationKey,
  buildRawIntentPayload,
  emitEtfIntentAttempt,
  SESSION_STORAGE_KEY,
  TELEMETRY_INGESTION_ENDPOINT,
} from "../etfIntentClient";

describe("ARX ETF Cockpit P1 — Telemetry Client Module (etfIntentClient)", () => {
  const UUID_V4_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

  beforeEach(() => {
    resetSession();
    vi.restoreAllMocks();
    delete (window as any).__ARX_SYNTHETIC__;
    if (typeof navigator !== "undefined") {
      (navigator as any).sendBeacon = vi.fn();
    }
  });

  afterEach(() => {
    resetSession();
    vi.restoreAllMocks();
  });

  describe("UUIDv4 and Session Identity Management", () => {
    it("C01: Generates RFC 4122 compliant UUIDv4 strings", () => {
      const id1 = generateUuidV4();
      const id2 = generateUuidV4();
      expect(id1).toMatch(UUID_V4_REGEX);
      expect(id2).toMatch(UUID_V4_REGEX);
      expect(id1).not.toBe(id2);
    });

    it("C02: Retrieves or initializes anonymous session ID in sessionStorage", () => {
      expect(window.sessionStorage.getItem(SESSION_STORAGE_KEY)).toBeNull();
      const session1 = getSessionId();
      expect(session1).toMatch(UUID_V4_REGEX);
      expect(window.sessionStorage.getItem(SESSION_STORAGE_KEY)).toBe(session1);

      // Subsequent call reuses the existing session ID
      const session2 = getSessionId();
      expect(session2).toBe(session1);
    });

    it("C03: Resets active session ID on resetSession()", () => {
      const session1 = getSessionId();
      expect(window.sessionStorage.getItem(SESSION_STORAGE_KEY)).toBe(session1);

      resetSession();
      expect(window.sessionStorage.getItem(SESSION_STORAGE_KEY)).toBeNull();

      const session2 = getSessionId();
      expect(session2).toMatch(UUID_V4_REGEX);
      expect(session2).not.toBe(session1);
    });

    it("C04: Gracefully handles corrupt or invalid stored session IDs", () => {
      window.sessionStorage.setItem(SESSION_STORAGE_KEY, "invalid-non-uuid-string");
      const session = getSessionId();
      expect(session).toMatch(UUID_V4_REGEX);
      expect(session).not.toBe("invalid-non-uuid-string");
    });
  });

  describe("Deduplication Key Computation", () => {
    it("C05: Derives deterministic composite SHA-256 deduplication key", () => {
      const sessionId = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d";
      const symbol = "SPY";
      const attemptId = "f1e2d3c4-b5a6-4978-90ab-cdef01234567";

      const key1 = computeDeduplicationKey(sessionId, symbol, attemptId);
      const key2 = computeDeduplicationKey(sessionId, symbol, attemptId);

      expect(key1).toBe(key2);
      expect(key1).toHaveLength(64);
      expect(key1).toMatch(/^[0-9a-f]{64}$/);
    });

    it("C06: Deduplication key is symbol case-insensitive and trims whitespace", () => {
      const sessionId = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d";
      const attemptId = "f1e2d3c4-b5a6-4978-90ab-cdef01234567";

      const keyUpper = computeDeduplicationKey(sessionId, "SPY", attemptId);
      const keyLower = computeDeduplicationKey(sessionId, "spy", attemptId);
      const keySpaced = computeDeduplicationKey(sessionId, "  SPY  ", attemptId);

      expect(keyUpper).toBe(keyLower);
      expect(keyUpper).toBe(keySpaced);
    });

    it("C07: Distinct attempt IDs yield distinct deduplication keys", () => {
      const sessionId = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d";
      const key1 = computeDeduplicationKey(sessionId, "SPY", generateUuidV4());
      const key2 = computeDeduplicationKey(sessionId, "SPY", generateUuidV4());
      expect(key1).not.toBe(key2);
    });
  });

  describe("Payload Construction (buildRawIntentPayload)", () => {
    it("C08: Returns null for non-ETF symbols (strict suppression invariant)", () => {
      expect(buildRawIntentPayload({ symbol: "AAPL" })).toBeNull();
      expect(buildRawIntentPayload({ symbol: "MSFT" })).toBeNull();
      expect(buildRawIntentPayload({ symbol: "NVDA" })).toBeNull();
      expect(buildRawIntentPayload({ symbol: "BTC" })).toBeNull();
      expect(buildRawIntentPayload({ symbol: "UNKNOWN_TICKER_99" })).toBeNull();
      expect(buildRawIntentPayload({ symbol: "" })).toBeNull();
      expect(buildRawIntentPayload({ symbol: "   " })).toBeNull();
    });

    it("C09: Constructs valid RawCandidateRecord for recognized ETFs", () => {
      const payload = buildRawIntentPayload({
        symbol: "SPY",
        sourceComponent: "handleSelectSymbol",
      });

      expect(payload).not.toBeNull();
      if (!payload) return;

      expect(payload.schema_version).toBe("1.0.0");
      expect(payload.event_id).toMatch(UUID_V4_REGEX);
      expect(payload.session_id).toMatch(UUID_V4_REGEX);
      expect(payload.observation_unit_id).toMatch(UUID_V4_REGEX);
      expect(payload.deduplication_key).toMatch(/^[0-9a-f]{64}$/);
      expect(payload.normalized_symbol).toBe("SPY");
      expect(payload.intent_type).toBe("ETF_SYMBOL_SELECT");
      expect(payload.source_component).toBe("handleSelectSymbol");
      expect(payload.synthetic_marker).toBe(false);
      expect(payload.ci_run_marker).toBe(false);
      expect(payload.manual_qa_marker).toBe(false);
      expect(new Date(payload.timestamp).getTime()).not.toBeNaN();
    });

    it("C10: Normalizes ticker prefix (e.g. BATS:QQQ -> QQQ)", () => {
      const payload = buildRawIntentPayload({ symbol: "BATS:QQQ" });
      expect(payload).not.toBeNull();
      expect(payload?.normalized_symbol).toBe("QQQ");
    });

    it("C11: Preserves existing attemptId across retries", () => {
      const existingId = generateUuidV4();
      const payload = buildRawIntentPayload({
        symbol: "XLK",
        attemptId: existingId,
      });

      expect(payload).not.toBeNull();
      expect(payload?.observation_unit_id).toBe(existingId);
    });

    it("C12: Detects window.__ARX_SYNTHETIC__ and flags synthetic_marker", () => {
      (window as any).__ARX_SYNTHETIC__ = true;
      const payload = buildRawIntentPayload({ symbol: "IWM" });
      expect(payload).not.toBeNull();
      expect(payload?.synthetic_marker).toBe(true);
    });
  });

  describe("Attempt Emission (emitEtfIntentAttempt)", () => {
    it("C13: Suppresses emission entirely for non-ETF selections (returns null)", async () => {
      const sendBeaconSpy = vi.spyOn(navigator, "sendBeacon");
      const fetchSpy = vi.spyOn(globalThis, "fetch");

      const res = await emitEtfIntentAttempt("AAPL");
      expect(res).toBeNull();
      expect(sendBeaconSpy).not.toHaveBeenCalled();
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("C14: Emits via navigator.sendBeacon when available and returns true", async () => {
      const sendBeaconSpy = vi.spyOn(navigator, "sendBeacon").mockReturnValue(true);
      const fetchSpy = vi.spyOn(globalThis, "fetch");

      const res = await emitEtfIntentAttempt("SPY", "handleSelectSymbol");
      expect(res).not.toBeNull();
      expect(res?.emitted).toBe(true);
      expect(sendBeaconSpy).toHaveBeenCalledTimes(1);
      expect(sendBeaconSpy).toHaveBeenCalledWith(
        TELEMETRY_INGESTION_ENDPOINT,
        expect.any(Blob)
      );
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("C15: Falls back to fetch keepalive when sendBeacon returns false", async () => {
      vi.spyOn(navigator, "sendBeacon").mockReturnValue(false);
      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response());

      const res = await emitEtfIntentAttempt("QQQ", "OmniSearch");
      expect(res).not.toBeNull();
      expect(res?.emitted).toBe(true);
      expect(fetchSpy).toHaveBeenCalledTimes(1);
      expect(fetchSpy).toHaveBeenCalledWith(
        TELEMETRY_INGESTION_ENDPOINT,
        expect.objectContaining({
          method: "POST",
          keepalive: true,
          headers: expect.objectContaining({
            "Content-Type": "application/json",
            "X-Attempt-Id": res?.attemptId,
            "X-Deduplication-Key": res?.deduplicationKey,
          }),
        })
      );
    });

    it("C16: Handles transport failure without fabricating success or throwing", async () => {
      vi.spyOn(navigator, "sendBeacon").mockImplementation(() => {
        throw new Error("Network quota exceeded");
      });
      vi.spyOn(globalThis, "fetch").mockImplementation(() => {
        throw new Error("Offline");
      });

      const res = await emitEtfIntentAttempt("XLK");
      expect(res).not.toBeNull();
      expect(res?.emitted).toBe(false);
      expect(res?.payload.normalized_symbol).toBe("XLK");
    });
  });
});
