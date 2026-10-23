// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from "vitest";
import { onRequest } from "../../../functions/api/telemetry/events";
import { CorporateEventCanonicalPayload } from "../../../types/corporate-events-telemetry";

const AUTH_SHA = "8404b6e4645b5f9f8237e70517dcab3b6afdda7f";

function createValidPayload(): CorporateEventCanonicalPayload {
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
    evaluation_id: "eval-uuid-0001",
    delivery_attempt_id: "attempt-uuid-0001",
  };
}

function mockRequestContext(options: {
  method?: string;
  body?: string;
  headers?: Record<string, string>;
  env?: Record<string, string>;
  cf?: Record<string, any>;
}) {
  const method = options.method || "POST";
  const body = options.body !== undefined ? options.body : JSON.stringify(createValidPayload());
  const headerMap = new Map(Object.entries(options.headers || {}));
  if (!headerMap.has("Content-Type")) {
    headerMap.set("Content-Type", "application/json");
  }
  if (!headerMap.has("Origin")) {
    headerMap.set("Origin", "https://arxterminal.com");
  }

  return {
    request: {
      method,
      text: () => Promise.resolve(body),
      headers: {
        get: (name: string) => headerMap.get(name) || null,
      },
      cf: options.cf || { botManagement: { score: 80 } },
      url: "https://arxterminal.com/api/telemetry/events",
    },
    env: options.env || {},
  };
}

describe("Cloudflare Pages Gateway: /api/telemetry/events", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  describe("Gate D: HTTP & Ingress Controls", () => {
    it("handles OPTIONS CORS pre-flight with 204", async () => {
      const context = mockRequestContext({ method: "OPTIONS" });
      const resp = await onRequest(context);
      expect(resp.status).toBe(204);
      expect(resp.headers.get("Access-Control-Allow-Methods")).toContain("POST");
    });

    it("rejects non-POST HTTP methods with 405 Method Not Allowed", async () => {
      const contextGet = mockRequestContext({ method: "GET" });
      const respGet = await onRequest(contextGet);
      expect(respGet.status).toBe(405);

      const contextPut = mockRequestContext({ method: "PUT" });
      const respPut = await onRequest(contextPut);
      expect(respPut.status).toBe(405);
    });

    it("rejects oversized request body (>16 KB) with 413 Payload Too Large", async () => {
      const largeStr = "a".repeat(17 * 1024);
      const context = mockRequestContext({ body: largeStr });
      const resp = await onRequest(context);
      expect(resp.status).toBe(413);
      const data = await resp.json();
      expect(data.error).toContain("Payload Too Large");
    });

    it("rejects malformed JSON syntax with 400 Bad Request", async () => {
      const context = mockRequestContext({ body: "{ invalid json syntax: true " });
      const resp = await onRequest(context);
      expect(resp.status).toBe(400);
      const data = await resp.json();
      expect(data.error).toContain("Malformed JSON payload");
    });
  });

  describe("Gate F: Privacy Deny-List Enforcement", () => {
    it("strictly rejects payload containing user_id with 400", async () => {
      const bad = { ...createValidPayload(), user_id: "user-12345" };
      const context = mockRequestContext({ body: JSON.stringify(bad) });
      const resp = await onRequest(context);
      expect(resp.status).toBe(400);
      const data = await resp.json();
      expect(data.error).toContain("Privacy Boundary Violation");
      expect(data.prohibited_field).toBe("user_id");
    });

    it("strictly rejects payload containing client_ip with 400", async () => {
      const bad = { ...createValidPayload(), client_ip: "192.168.1.1" };
      const context = mockRequestContext({ body: JSON.stringify(bad) });
      const resp = await onRequest(context);
      expect(resp.status).toBe(400);
      const data = await resp.json();
      expect(data.error).toContain("Privacy Boundary Violation");
    });
  });

  describe("Gate B & G: Schema Validation & Invariants", () => {
    it("rejects unknown extra field with 400", async () => {
      const bad = { ...createValidPayload(), unexpected_extra: "field" };
      const context = mockRequestContext({ body: JSON.stringify(bad) });
      const resp = await onRequest(context);
      expect(resp.status).toBe(400);
      const data = await resp.json();
      expect(data.error).toContain("Schema Validation Error");
    });

    it("rejects semantic penalty mismatch with 400", async () => {
      const bad = { ...createValidPayload(), risk_bucket: "ACUTE", applied_penalty: -8.0 };
      const context = mockRequestContext({ body: JSON.stringify(bad) });
      const resp = await onRequest(context);
      expect(resp.status).toBe(400);
    });
  });

  describe("Gate E: Disabled-by-Default Feature Flag", () => {
    it("returns HTTP 200 with status DISABLED when feature flag is absent", async () => {
      const context = mockRequestContext({ env: {} });
      const resp = await onRequest(context);
      expect(resp.status).toBe(200);
      const data = await resp.json();
      expect(data.status).toBe("DISABLED");
      expect(data.accepted).toBe(false);
      expect(data.forwarded).toBe(false);
      expect(data.deduplication_key).toBeDefined();
    });

    it("returns HTTP 200 with status DISABLED when feature flag is explicitly 'false'", async () => {
      const context = mockRequestContext({
        env: { ENABLE_CORPORATE_EVENTS_TELEMETRY: "false" },
      });
      const resp = await onRequest(context);
      expect(resp.status).toBe(200);
      const data = await resp.json();
      expect(data.status).toBe("DISABLED");
      expect(data.accepted).toBe(false);
    });
  });

  describe("Gate E: Upstream Forwarding Resiliency (Enabled State)", () => {
    it("forwards to upstream and returns ACCEPTED on upstream 200", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 200,
        text: () => Promise.resolve(JSON.stringify({ status: "ACCEPTED", record_id: "rec-1" })),
      });

      const context = mockRequestContext({
        env: {
          ENABLE_CORPORATE_EVENTS_TELEMETRY: "true",
          TELEMETRY_UPSTREAM_URL: "http://127.0.0.1:8080/persist",
        },
      });

      const resp = await onRequest(context);
      expect(resp.status).toBe(200);
      const data = await resp.json();
      expect(data.status).toBe("ACCEPTED");
      expect(data.forwarded).toBe(true);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8080/persist",
        expect.objectContaining({ method: "POST" })
      );
    });

    it("handles upstream 409 duplicate acknowledgement gracefully", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 409,
        text: () => Promise.resolve(JSON.stringify({ error: "Duplicate deduplication_key" })),
      });

      const context = mockRequestContext({
        env: {
          ENABLE_CORPORATE_EVENTS_TELEMETRY: "true",
        },
      });

      const resp = await onRequest(context);
      expect(resp.status).toBe(200);
      const data = await resp.json();
      expect(data.status).toBe("DUPLICATE_ACKNOWLEDGED");
    });

    it("returns 502 UPSTREAM_UNAVAILABLE on upstream 500 error", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 500,
        text: () => Promise.resolve("Internal Server Error"),
      });

      const context = mockRequestContext({
        env: {
          ENABLE_CORPORATE_EVENTS_TELEMETRY: "true",
        },
      });

      const resp = await onRequest(context);
      expect(resp.status).toBe(502);
      const data = await resp.json();
      expect(data.status).toBe("UPSTREAM_UNAVAILABLE");
    });

    it("returns 504 GATEWAY_TIMEOUT on upstream timeout / abort", async () => {
      const abortError = new Error("The operation was aborted");
      abortError.name = "AbortError";
      global.fetch = vi.fn().mockRejectedValue(abortError);

      const context = mockRequestContext({
        env: {
          ENABLE_CORPORATE_EVENTS_TELEMETRY: "true",
        },
      });

      const resp = await onRequest(context);
      expect(resp.status).toBe(504);
      const data = await resp.json();
      expect(data.status).toBe("GATEWAY_TIMEOUT");
    });
  });
});
