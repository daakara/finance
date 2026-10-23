// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import {
  generateUuidV4,
  getEvaluationContextId,
  resetEvaluationContext,
  assembleCorporateEventPayload,
  buildTransportEnvelope,
  dispatchCorporateEventTelemetry,
} from "../corporateEventsClient";

describe("Corporate Events Telemetry Client Helper", () => {
  const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  const AUTH_SHA = "8404b6e4645b5f9f8237e70517dcab3b6afdda7f";

  beforeEach(() => {
    resetEvaluationContext();
    vi.restoreAllMocks();
    delete (window as any).__ARX_SYNTHETIC__;
    if (typeof navigator !== "undefined") {
      (navigator as any).sendBeacon = vi.fn();
    }
  });

  afterEach(() => {
    resetEvaluationContext();
    vi.restoreAllMocks();
  });

  describe("Gate C: Identity & Session Context", () => {
    it("generates RFC 4122 compliant UUIDv4 strings", () => {
      const id1 = generateUuidV4();
      const id2 = generateUuidV4();
      expect(id1).toMatch(UUID_REGEX);
      expect(id2).toMatch(UUID_REGEX);
      expect(id1).not.toBe(id2);
    });

    it("maintains stable evaluation_context_id within session", () => {
      const ctx1 = getEvaluationContextId();
      const ctx2 = getEvaluationContextId();
      expect(ctx1).toBe(ctx2);
      expect(ctx1).toContain("ctx-");
    });

    it("resets context when resetEvaluationContext is called", () => {
      const ctx1 = getEvaluationContextId();
      resetEvaluationContext();
      const ctx2 = getEvaluationContextId();
      expect(ctx1).not.toBe(ctx2);
    });
  });

  describe("Gate C: Frozen Evaluation Identity vs. Distinct Delivery Identity", () => {
    it("assembles payload with frozen evaluation_id", () => {
      const payload = assembleCorporateEventPayload({
        symbol: "NVDA",
        release_sha: AUTH_SHA,
        days_to_earnings: 2.5,
        risk_bucket: "CAUTION",
        applied_penalty: -8.0,
      });

      expect(payload.evaluation_id).toBeDefined();
      expect(payload.evaluation_id).toMatch(UUID_REGEX);
      expect(payload.occurred_at_utc).toBeDefined();
    });

    it("preserves evaluation_id across multiple retries while assigning distinct delivery_attempt_id", () => {
      const payload = assembleCorporateEventPayload({
        symbol: "TSLA",
        release_sha: AUTH_SHA,
        days_to_earnings: 0.8,
        risk_bucket: "ACUTE",
        applied_penalty: -25.0,
      });

      const attempt1 = buildTransportEnvelope(payload, false);
      const attempt2 = buildTransportEnvelope(payload, true);
      const attempt3 = buildTransportEnvelope(payload, true);

      // Logical evaluation ID remains strictly identical
      expect(attempt1.evaluation_id).toBe(payload.evaluation_id);
      expect(attempt2.evaluation_id).toBe(payload.evaluation_id);
      expect(attempt3.evaluation_id).toBe(payload.evaluation_id);

      // Delivery attempt IDs must be distinct per transmission attempt
      expect(attempt1.delivery_attempt_id).toMatch(UUID_REGEX);
      expect(attempt2.delivery_attempt_id).toMatch(UUID_REGEX);
      expect(attempt3.delivery_attempt_id).toMatch(UUID_REGEX);
      expect(attempt1.delivery_attempt_id).not.toBe(attempt2.delivery_attempt_id);
      expect(attempt2.delivery_attempt_id).not.toBe(attempt3.delivery_attempt_id);
    });
  });

  describe("Gate E: Transport Dispatch & Fail-Open Behavior", () => {
    it("dispatches via sendBeacon when available and returns accepted", async () => {
      (navigator.sendBeacon as any).mockReturnValue(true);

      const payload = assembleCorporateEventPayload({
        symbol: "MSFT",
        release_sha: AUTH_SHA,
        days_to_earnings: 10.0,
        risk_bucket: "NOMINAL",
        applied_penalty: 0.0,
      });
      const envelope = buildTransportEnvelope(payload);

      const res = await dispatchCorporateEventTelemetry(envelope);
      expect(res.status).toBe("ACCEPTED");
      expect(res.accepted).toBe(true);
      expect(navigator.sendBeacon).toHaveBeenCalledWith(
        "/api/telemetry/events",
        expect.any(String)
      );
    });

    it("falls back to fetch when sendBeacon returns false", async () => {
      (navigator.sendBeacon as any).mockReturnValue(false);
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        text: () => Promise.resolve(JSON.stringify({ status: "ACCEPTED", forwarded: true })),
      });

      const payload = assembleCorporateEventPayload({
        symbol: "AMZN",
        release_sha: AUTH_SHA,
        days_to_earnings: null,
        risk_bucket: "UNAVAILABLE",
        applied_penalty: 0.0,
      });
      const envelope = buildTransportEnvelope(payload);

      const res = await dispatchCorporateEventTelemetry(envelope);
      expect(res.status).toBe("ACCEPTED");
      expect(res.accepted).toBe(true);
      expect(global.fetch).toHaveBeenCalled();
    });

    it("fails open on network exception without throwing", async () => {
      (navigator.sendBeacon as any).mockReturnValue(false);
      global.fetch = vi.fn().mockRejectedValue(new Error("Failed to fetch (offline)"));

      const payload = assembleCorporateEventPayload({
        symbol: "GOOG",
        release_sha: AUTH_SHA,
        days_to_earnings: 12.0,
        risk_bucket: "NOMINAL",
        applied_penalty: 0.0,
      });
      const envelope = buildTransportEnvelope(payload);

      const res = await dispatchCorporateEventTelemetry(envelope);
      expect(res.accepted).toBe(false);
      expect(res.status).toBe("UPSTREAM_UNAVAILABLE");
      expect(res.rejection_reason).toContain("offline");
    });
  });
});
