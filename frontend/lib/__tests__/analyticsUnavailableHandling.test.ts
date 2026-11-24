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

  it("throws AssetNotFoundError when backend returns 404 and direct market data has no candles", async () => {
    const mockFetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes("/analytics/")) {
        return Promise.resolve(new Response(JSON.stringify({ detail: "Asset ANTHROPIC is not recognized on exchange tape" }), { status: 404 }));
      }
      return Promise.resolve(new Response(JSON.stringify({ chart: { result: null, error: { code: "Not Found" } } }), { status: 404 }));
    });
    vi.stubGlobal("fetch", mockFetch);

    await expect(fetchAssetAnalytics("ANTHROPIC", "1y", "1d")).rejects.toThrow(AssetNotFoundError);
  });

  it("throws InsufficientHistoryError when backend returns 422", async () => {
    const mockFetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes("/analytics/")) {
        return Promise.resolve(new Response(JSON.stringify({ detail: "Asset NEWIPO has only 5 sessions on record. Minimum 15 daily sessions required." }), { status: 422 }));
      }
      return Promise.resolve(new Response(null, { status: 500 }));
    });
    vi.stubGlobal("fetch", mockFetch);

    await expect(fetchAssetAnalytics("NEWIPO", "1y", "1d")).rejects.toThrow(InsufficientHistoryError);
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
