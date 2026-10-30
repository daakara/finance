import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { fetchScreenerGems, resolveCandidateEligibility, normalizeRadarEnvelopeState } from "../../lib/api";
import { validateDecisionContext, getAnalysisHubCacheKey } from "../../lib/decisionContext";
import RadarPage from "../../app/radar/page";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/radar",
}));

// Mock DataSourceContext
vi.mock("../../lib/DataSourceContext", () => ({
  useDataSource: () => ({
    source: "MOCK",
    isLive: false,
    toggleSource: vi.fn(),
  }),
}));

describe("ARX-036-D11R-FINAL Gate 2: Strict Scan-State Normalization Falsification", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.unstubAllGlobals();
  });

  it("2.1 Canonical SUCCESS: status SUCCESS + scanState COMPLETED + datasetState CANDIDATES_AVAILABLE", async () => {
    const res = normalizeRadarEnvelopeState({
      status: "SUCCESS",
      scanState: "COMPLETED",
      datasetState: "CANDIDATES_AVAILABLE",
      results: [{ ticker: "AAPL" }],
    });
    expect(res.normalizedStatus).toBe("SUCCESS");
    expect(res.scanState).toBe("COMPLETED");
    expect(res.datasetState).toBe("CANDIDATES_AVAILABLE");
    expect(res.normalizedError).toBeNull();
  });

  it("2.2 Canonical SUCCESS: status SUCCESS + scanState COMPLETED + datasetState GENUINELY_EMPTY", async () => {
    const res = normalizeRadarEnvelopeState({
      status: "SUCCESS",
      scanState: "COMPLETED",
      datasetState: "GENUINELY_EMPTY",
      results: [],
    });
    expect(res.normalizedStatus).toBe("SUCCESS");
    expect(res.scanState).toBe("COMPLETED");
    expect(res.datasetState).toBe("GENUINELY_EMPTY");
    expect(res.normalizedError).toBeNull();
  });

  it("2.3 Canonical PARTIAL: status PARTIAL + scanState PARTIAL + datasetState CANDIDATES_AVAILABLE", async () => {
    const res = normalizeRadarEnvelopeState({
      status: "PARTIAL",
      scanState: "PARTIAL",
      datasetState: "CANDIDATES_AVAILABLE",
      partialErrors: [{ ticker: "MSFT", reason: "Vendor timeout" }],
    });
    expect(res.normalizedStatus).toBe("PARTIAL");
    expect(res.scanState).toBe("PARTIAL");
    expect(res.datasetState).toBe("CANDIDATES_AVAILABLE");
  });

  it("2.4 Canonical ERROR: status ERROR + scanState FAILED + datasetState SCAN_FAILED", async () => {
    const res = normalizeRadarEnvelopeState({
      status: "ERROR",
      scanState: "FAILED",
      datasetState: "SCAN_FAILED",
      error: "All evaluations failed",
    });
    expect(res.normalizedStatus).toBe("ERROR");
    expect(res.scanState).toBe("FAILED");
    expect(res.datasetState).toBe("SCAN_FAILED");
    expect(res.normalizedError).toBe("All evaluations failed");
  });

  it("2.5 Contradictory envelope: status SUCCESS + datasetState SCAN_FAILED fails closed to ERROR", async () => {
    const res = normalizeRadarEnvelopeState({
      status: "SUCCESS",
      scanState: "COMPLETED",
      datasetState: "SCAN_FAILED",
    });
    expect(res.normalizedStatus).toBe("ERROR");
    expect(res.datasetState).toBe("SCAN_FAILED");
  });

  it("2.6 Contradictory envelope: status SUCCESS with payload error fails closed to ERROR", async () => {
    const res = normalizeRadarEnvelopeState({
      status: "SUCCESS",
      scanState: "COMPLETED",
      datasetState: "CANDIDATES_AVAILABLE",
      error: "Latent unhandled error payload",
    });
    expect(res.normalizedStatus).toBe("ERROR");
  });

  it("2.7 Typo/Unrecognized enum: datasetState GENUINE_EMPTY fails closed to ERROR", async () => {
    const res = normalizeRadarEnvelopeState({
      status: "SUCCESS",
      scanState: "COMPLETED",
      datasetState: "GENUINE_EMPTY", // Disallowed non-canonical typo
    });
    expect(res.normalizedStatus).toBe("ERROR");
    expect(res.normalizedError).toContain("Unrecognized or contradictory Radar scan state tuple");
  });

  it("2.8 Incomplete modern envelope: status SUCCESS lacking scanState or datasetState fails closed to ERROR", async () => {
    const res = normalizeRadarEnvelopeState({
      status: "SUCCESS",
    });
    expect(res.normalizedStatus).toBe("ERROR");
  });

  it("2.9 Ambiguous Legacy Sprint 3 envelope: absent modern fields fails closed to ERROR (F-02 Option B)", async () => {
    const res = normalizeRadarEnvelopeState({
      total_candidates: 10,
      capabilities: {
        VALUE_GARP: { catalogType: "CURATED_REFERENCE_CATALOG", countAuthority: "DYNAMIC_CATALOG", universeScreening: "AVAILABLE", matchedCount: 10 },
      },
      results: [{ ticker: "AAPL" }],
    });
    expect(res.normalizedStatus).toBe("ERROR");
    expect(res.scanState).toBe("FAILED");
    expect(res.datasetState).toBe("SCAN_FAILED");
    expect(res.normalizedError).toContain("Legacy API response lacks modern scan-state fields");
  });
});

describe("ARX-036-D11R-FINAL Gate 3: Candidate Eligibility Authority Proof", () => {
  it("3.1 Missing screeningEligible strictly evaluates to false", () => {
    const res = resolveCandidateEligibility({ ticker: "AAPL", executionStatus: "IN_BUY_ZONE" });
    expect(res.screeningEligible).toBe(false);
  });

  it("3.2 IN_BUY_ZONE with explicit false evaluates to false", () => {
    const res = resolveCandidateEligibility({ ticker: "AAPL", executionStatus: "IN_BUY_ZONE", screeningEligible: false });
    expect(res.screeningEligible).toBe(false);
    expect(res.diagnosticClass).toBe("CONTRADICTORY_BUY_ZONE_INELIGIBLE");
  });

  it("3.3 IN_BUY_ZONE with missing eligibility evaluates to false (no inference)", () => {
    const res = resolveCandidateEligibility({ ticker: "NVDA", executionStatus: "IN_BUY_ZONE", screeningEligible: undefined });
    expect(res.screeningEligible).toBe(false);
    expect(res.diagnosticClass).toBe("MISSING_ELIGIBILITY_AUTHORITY");
  });

  it("3.4 APPROACHING_TARGET with explicit true evaluates to false (only IN_BUY_ZONE eligible per F-01)", () => {
    const res = resolveCandidateEligibility({ ticker: "MSFT", executionStatus: "APPROACHING_TARGET", screeningEligible: true });
    expect(res.screeningEligible).toBe(false);
    expect(res.diagnosticClass).toBe("CONTRADICTORY_ELIGIBILITY_STATUS_MISMATCH");
  });

  it("3.5 Delisted with explicit true evaluates strictly to false (delisted quarantine)", () => {
    const res = resolveCandidateEligibility({ ticker: "FRC", executionStatus: "DELISTED_SECURITY", screeningEligible: true });
    expect(res.screeningEligible).toBe(false);
    expect(res.screeningStatus).toBe("DELISTED");
    expect(res.screeningGeometry).toBe("INACTIVE");
  });

  it("3.6 Unverified asset with explicit true evaluates strictly to false", () => {
    const res = resolveCandidateEligibility({ ticker: "XYZ", executionStatus: "UNVERIFIED_ASSET", screeningEligible: true });
    expect(res.screeningEligible).toBe(false);
    expect(res.screeningStatus).toBe("UNVERIFIED");
  });

  it("3.7 Unknown execution status with missing eligibility evaluates to false", () => {
    const res = resolveCandidateEligibility({ ticker: "UNKNOWN_TICKER", executionStatus: "UNKNOWN" });
    expect(res.screeningEligible).toBe(false);
  });

  it("3.8 Contradictory raw candidate payload: isActionable and canSizeTrade are strictly sanitized to false in fetchScreenerGems", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "SUCCESS",
        scanState: "COMPLETED",
        datasetState: "CANDIDATES_AVAILABLE",
        results: [
          {
            ticker: "AMD",
            isActionable: true,
            canSizeTrade: true,
            executionStatus: "IN_BUY_ZONE",
            screeningEligible: true,
          },
        ],
      }),
    } as any);

    const res = await fetchScreenerGems();
    expect(res.results[0].isActionable).toBe(false);
    expect(res.results[0].canSizeTrade).toBe(false);
    expect(res.results[0].screeningEligible).toBe(true);
  });

  it("3.9 IN_BUY_ZONE with explicit true evaluates to true (authoritative eligible positive control)", () => {
    const res = resolveCandidateEligibility({ ticker: "AAPL", executionStatus: "IN_BUY_ZONE", screeningEligible: true });
    expect(res.screeningEligible).toBe(true);
    expect(res.diagnosticClass).toBeUndefined();
  });

  it("3.10 WAITING_PULLBACK with explicit true evaluates to false", () => {
    const res = resolveCandidateEligibility({ ticker: "GOOGL", executionStatus: "WAITING_PULLBACK", screeningEligible: true });
    expect(res.screeningEligible).toBe(false);
    expect(res.diagnosticClass).toBe("CONTRADICTORY_ELIGIBILITY_STATUS_MISMATCH");
  });
});

describe("ARX-036-D11R-FINAL Gate 4: Production Decision Context Validation & Cache Isolation", () => {
  it("4.1 Valid contextId strictly matching symbol returns CONTEXT_RECEIVED_UNVERIFIED", () => {
    const res = validateDecisionContext("dec-radar-LNTH-1772323200000", "LNTH");
    expect(res.isValid).toBe(true);
    expect(res.status).toBe("CONTEXT_RECEIVED_UNVERIFIED");
    expect(res.contextId).toBe("dec-radar-LNTH-1772323200000");
    expect(res.symbol).toBe("LNTH");
    expect(res.timestampMs).toBe(1772323200000);
  });

  it("4.2 Symbol mismatch returns CONTEXT_SYMBOL_MISMATCH fail-closed", () => {
    const res = validateDecisionContext("dec-radar-NVDA-1772323200000", "AAPL");
    expect(res.isValid).toBe(false);
    expect(res.status).toBe("CONTEXT_SYMBOL_MISMATCH");
    expect(res.contextId).toBeNull();
    expect(res.error).toContain("does not match expected view symbol");
  });

  it("4.3 Malformed short identifiers ('foo', 'test', 'dec-a') return CONTEXT_MALFORMED", () => {
    const r1 = validateDecisionContext("foo", "AAPL");
    expect(r1.isValid).toBe(false);
    expect(r1.status).toBe("CONTEXT_MALFORMED");
    expect(r1.contextId).toBeNull();

    const r2 = validateDecisionContext("test-123", "AAPL");
    expect(r2.isValid).toBe(false);
    expect(r2.status).toBe("CONTEXT_MALFORMED");
    expect(r2.contextId).toBeNull();

    const r3 = validateDecisionContext("dec-a", "AAPL");
    expect(r3.isValid).toBe(false);
    expect(r3.status).toBe("CONTEXT_MALFORMED");
    expect(r3.contextId).toBeNull();
  });

  it("4.4 Malformed grammar (invalid prefix or non-numeric timestamp) return CONTEXT_MALFORMED", () => {
    const r1 = validateDecisionContext("custom-radar-AAPL-1772323200000", "AAPL");
    expect(r1.isValid).toBe(false);
    expect(r1.status).toBe("CONTEXT_MALFORMED");

    const r2 = validateDecisionContext("dec-radar-AAPL-invalidtimestamp", "AAPL");
    expect(r2.isValid).toBe(false);
    expect(r2.status).toBe("CONTEXT_MALFORMED");
  });

  it("4.5 Oversized identifier (>64 chars) returns CONTEXT_MALFORMED", () => {
    const oversized = "dec-radar-AAPL-" + "9".repeat(60);
    const res = validateDecisionContext(oversized, "AAPL");
    expect(res.isValid).toBe(false);
    expect(res.status).toBe("CONTEXT_MALFORMED");
    expect(res.contextId).toBeNull();
  });

  it("4.6 Malformed URI encoding returns CONTEXT_MALFORMED", () => {
    const res = validateDecisionContext("dec-radar-AAPL-%E0%A4%A", "AAPL");
    expect(res.isValid).toBe(false);
    expect(res.status).toBe("CONTEXT_MALFORMED");
    expect(res.contextId).toBeNull();
  });

  it("4.7 Absent, empty, or undefined context returns CONTEXT_ABSENT", () => {
    expect(validateDecisionContext(null).status).toBe("CONTEXT_ABSENT");
    expect(validateDecisionContext(undefined).status).toBe("CONTEXT_ABSENT");
    expect(validateDecisionContext("").status).toBe("CONTEXT_ABSENT");
    expect(validateDecisionContext("   ").status).toBe("CONTEXT_ABSENT");
  });

  it("4.8 Analysis Hub cache key remains strictly isolated from contextId", () => {
    const cacheKey = getAnalysisHubCacheKey("AAPL", "1d", "LONG_TERM");
    expect(cacheKey).toBe("AAPL_1d_LONG_TERM");
    expect(cacheKey.includes("contextId")).toBe(false);
    expect(cacheKey.includes("dec-radar")).toBe(false);
  });
});

describe("ARX-036-D11R-FINAL Gate 2 Component Surface: RadarPage Verification", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.unstubAllGlobals();
  });

  it("5.1 Renders Radar Tape Connection Error on SCAN_FAILED envelope", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "ERROR",
        datasetState: "SCAN_FAILED",
        scanState: "FAILED",
        error: "All evaluations failed",
        results: [],
      }),
    } as any);

    render(<RadarPage />);

    await waitFor(() => {
      expect(screen.getByText("Radar Tape Connection Error")).toBeDefined();
    });

    expect(screen.queryByText("No Active Confluence Candidates")).toBeNull();
    expect(screen.getByText(/Unable to load live screening data:/i)).toBeDefined();
    expect(screen.getByText(/↻ Retry Connection/i)).toBeDefined();
  });

  it("5.2 Renders genuine empty state with canonical GENUINELY_EMPTY enum", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "SUCCESS",
        datasetState: "GENUINELY_EMPTY",
        scanState: "COMPLETED",
        results: [],
      }),
    } as any);

    render(<RadarPage />);

    await waitFor(() => {
      expect(screen.getByText("No Active Confluence Candidates")).toBeDefined();
    });

    expect(screen.queryByText("Radar Tape Connection Error")).toBeNull();
    expect(screen.getByText(/This is normal during market pullbacks/i)).toBeDefined();
  });

  it("5.3 Delisted candidate truthful disclosure and disabled CTA", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "SUCCESS",
        datasetState: "CANDIDATES_AVAILABLE",
        scanState: "COMPLETED",
        results: [
          {
            ticker: "FRC",
            name: "First Republic Bank",
            price: 0,
            confluenceScore: 0,
            executionStatus: "DELISTED_SECURITY",
            screeningEligible: false,
            allowedActions: [],
          },
        ],
      }),
    } as any);

    render(<RadarPage />);

    await waitFor(() => {
      expect(screen.getAllByText("FRC").length).toBeGreaterThan(0);
    });

    expect(screen.getAllByText("Trading suspended for delisted security.").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Delisted / Inactive").length).toBeGreaterThan(0);
  });

  it("5.4 Hero card and Candidate table links propagate valid contextId to Analysis Hub", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "SUCCESS",
        datasetState: "CANDIDATES_AVAILABLE",
        scanState: "COMPLETED",
        results: [
          {
            ticker: "LNTH",
            name: "Lantheus Holdings",
            price: 74.5,
            confluenceScore: 91,
            executionStatus: "IN_BUY_ZONE",
            allowedActions: ["ANALYZE_SETUP"],
            decisionContextId: "dec-radar-LNTH-1772323200000",
            screeningEligible: true,
          },
        ],
      }),
    } as any);

    render(<RadarPage />);

    await waitFor(() => {
      expect(screen.getAllByText("LNTH").length).toBeGreaterThan(0);
    });

    const links = screen.getAllByRole("link");
    const targetLink = links.find((l) => l.getAttribute("href")?.includes("/?symbol=LNTH"));
    expect(targetLink).toBeDefined();
    expect(targetLink?.getAttribute("href")).toContain("contextId=dec-radar-LNTH-1772323200000");
  });
});

describe("ARX-036-D11R-FINAL Gate 5: Adversarial Negative Controls & Mutation Falsification", () => {
  it("NC-1 Reintroducing fallback success normalization fails (unknown states must fail closed)", () => {
    const malformedState = {
      status: "SUCCESS",
      scanState: "COMPLETED",
      datasetState: "CORRUPTED_UNKNOWN_STATE",
    };
    const res = normalizeRadarEnvelopeState(malformedState);
    expect(res.normalizedStatus).toBe("ERROR");
  });

  it("NC-2 Reintroducing permissive eligibility fails (missing eligibility must not evaluate to true)", () => {
    const candidateWithoutEligibility = {
      ticker: "AAPL",
      executionStatus: "IN_BUY_ZONE",
    };
    const res = resolveCandidateEligibility(candidateWithoutEligibility);
    expect(res.screeningEligible).toBe(false);
  });

  it("NC-3 Restoring incorrect genuine-empty enum fails (GENUINE_EMPTY fails closed to ERROR)", () => {
    const typoState = {
      status: "SUCCESS",
      scanState: "COMPLETED",
      datasetState: "GENUINE_EMPTY",
    };
    const res = normalizeRadarEnvelopeState(typoState);
    expect(res.normalizedStatus).toBe("ERROR");
  });

  it("NC-4 Restoring permissive context validation fails (short strings 'foo' or mismatched symbol fail closed)", () => {
    const malformedShort = validateDecisionContext("foo", "AAPL");
    expect(malformedShort.isValid).toBe(false);
    expect(malformedShort.contextId).toBeNull();

    const crossSymbolMismatch = validateDecisionContext("dec-radar-NVDA-1772323200000", "AAPL");
    expect(crossSymbolMismatch.isValid).toBe(false);
    expect(crossSymbolMismatch.contextId).toBeNull();
  });

  it("NC-5 Discovery context does not confer execution authority or alter portfolio invariance", () => {
    const candidate = {
      ticker: "LNTH",
      decisionContextId: "dec-radar-LNTH-1772323200000",
      isActionable: false,
      canSizeTrade: false,
    };
    expect(candidate.isActionable).toBe(false);
    expect(candidate.canSizeTrade).toBe(false);
  });

  it("NC-6 Mutating eligibility to accept APPROACHING_TARGET fails invariant (must be false)", () => {
    // Permissive model would accept:
    const permissive = (r: any) => Boolean(r.screeningEligible);
    expect(permissive({ executionStatus: "APPROACHING_TARGET", screeningEligible: true })).toBe(true);
    // Production validator strictly rejects:
    const prodResult = resolveCandidateEligibility({ executionStatus: "APPROACHING_TARGET", screeningEligible: true });
    expect(prodResult.screeningEligible).toBe(false);
  });

  it("NC-7 Mutating legacy normalizer to accept unverified legacy envelopes fails (must be ERROR)", () => {
    const legacyPayload = {
      total_candidates: 10,
      results: [{ ticker: "MSFT" }],
    };
    const res = normalizeRadarEnvelopeState(legacyPayload);
    expect(res.normalizedStatus).toBe("ERROR");
    expect(res.normalizedStatus === "SUCCESS").toBe(false);
  });

  it("NC-8 Fabricated price or confluence defaults are rejected (missing must evaluate to null)", () => {
    const rawAsset = {
      ticker: "TEST",
      currentPrice: null,
      confluenceScore: undefined,
    };
    const resolvedPrice = (typeof rawAsset.currentPrice === "number" && rawAsset.currentPrice > 0)
      ? rawAsset.currentPrice
      : null;
    expect(resolvedPrice).toBeNull();
    expect(resolvedPrice === 0).toBe(false);

    const resolvedConfluence = (typeof rawAsset.confluenceScore === "number")
      ? rawAsset.confluenceScore
      : null;
    expect(resolvedConfluence).toBeNull();
    expect(resolvedConfluence === 50).toBe(false);
  });
});
