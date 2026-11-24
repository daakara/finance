import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  fetchAssetAnalytics,
  generateFallbackAnalytics,
  AssetNotFoundError,
  InsufficientHistoryError,
} from "../api";
import { generateQuantitativeInsight } from "../insightGenerator";

describe("analytics 404 and unavailable asset handling", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("throws AssetNotFoundError when backend returns 404 and asset is unverified in security master", async () => {
    const mockFetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes("/analytics/")) {
        return Promise.resolve(new Response(JSON.stringify({ detail: "No valid price action data found for symbol ANTHROPIC" }), { status: 404 }));
      }
      if (url.includes("/market/instruments/")) {
        return Promise.resolve(new Response(JSON.stringify({
          symbol: "ANTHROPIC",
          listing_status: "UNVERIFIED",
          execution_eligibility: "FAIL_CLOSED"
        }), { status: 200 }));
      }
      return Promise.resolve(new Response(JSON.stringify({ chart: { result: null, error: { code: "Not Found" } } }), { status: 404 }));
    });
    vi.stubGlobal("fetch", mockFetch);

    await expect(fetchAssetAnalytics("ANTHROPIC", "1y", "1d")).rejects.toThrow(AssetNotFoundError);
  });

  it("throws InsufficientHistoryError when backend returns 404 but asset is ACTIVE in security master", async () => {
    const mockFetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes("/analytics/")) {
        return Promise.resolve(new Response(JSON.stringify({ detail: "No valid price action data found for symbol ANT" }), { status: 404 }));
      }
      if (url.includes("/market/instruments/")) {
        return Promise.resolve(new Response(JSON.stringify({
          symbol: "ANT",
          primary_exchange: "NASDAQ",
          security_type: "COMMON_STOCK",
          listing_status: "ACTIVE",
          classification_status: "VERIFIED",
          execution_eligibility: "STOCK_EXECUTION"
        }), { status: 200 }));
      }
      return Promise.resolve(new Response(JSON.stringify({ chart: { result: null, error: { code: "Not Found" } } }), { status: 404 }));
    });
    vi.stubGlobal("fetch", mockFetch);

    try {
      await fetchAssetAnalytics("ANT", "1y", "1d");
      expect.fail("Should have thrown InsufficientHistoryError");
    } catch (err: any) {
      expect(err).toBeInstanceOf(InsufficientHistoryError);
      expect(err.message).toContain("insufficient");
      expect(err.message).toContain("ANT");
      expect(err.message).toContain("NASDAQ");
    }
  });

  it("throws InsufficientHistoryError when backend returns 422 with session counts", async () => {
    const mockFetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes("/analytics/")) {
        return Promise.resolve(new Response(JSON.stringify({
          detail: "Asset ANT has only 3 daily sessions on record (observed 2026-11-19 to 2026-11-23). Minimum 15 daily sessions required."
        }), { status: 422 }));
      }
      return Promise.resolve(new Response(null, { status: 500 }));
    });
    vi.stubGlobal("fetch", mockFetch);

    try {
      await fetchAssetAnalytics("ANT", "1y", "1d");
      expect.fail("Should have thrown InsufficientHistoryError");
    } catch (err: any) {
      expect(err).toBeInstanceOf(InsufficientHistoryError);
      expect(err.message).toContain("only 3 daily sessions");
    }
  });

  it("does not fabricate stalenessDays: 999 on fallback analytics", () => {
    const fallback = generateFallbackAnalytics("ANTHROPIC", "1y", "1d");
    expect(fallback.freshness?.stalenessDays).toBeUndefined();
    expect(fallback.freshness?.status).toBe("UNAVAILABLE");
  });

  it("does not claim Stale Historical Tape (> 4 Days) or fabricate 0/100 score when status is UNAVAILABLE", () => {
    const fallback = generateFallbackAnalytics("UNKNOWNASSET", "1y", "1d");
    const insight = generateQuantitativeInsight(
      "UNKNOWNASSET",
      "Unknown Asset",
      0,
      0,
      undefined,
      undefined,
      "SWING",
      "NOT_OWNED",
      "USER_DECLARED",
      [],
      "unavailable",
      undefined,
      fallback.decisionTrace,
      fallback.optimalExecution,
      fallback.freshness
    );

    // Invariant: UNAVAILABLE is NOT a stale historical tape!
    expect(insight.verdictLabel).not.toBe("Stale Historical Tape (> 4 Days)");
    expect(insight.decisionState).not.toBe("STALE_DATA");
    expect(insight.verdictLabel).toBe("Unverified Asset — Live Tape Required");
  });
});
